#!/usr/bin/env python3
"""
BPE-LncRNA: Byte-Pair Encoding for Long Noncoding RNA Identification in Anopheles gambiae
==========================================================================================

End-to-end pipeline:
  1. Data curation from Anopheles gambiae transcriptome (VectorBase/Ensembl)
  2. Genomic BPE tokenizer training via Hugging Face tokenizers
  3. Deep learning classification (BiLSTM + Transformer Encoder)
  4. Evaluation, benchmarking, and interpretability

Requirements:
  pip install torch tokenizers biopython scikit-learn pandas numpy matplotlib seaborn

Author: Computational Genomics Laboratory
License: MIT
"""

import os
import gc
import gzip
import random
import logging
import argparse
from pathlib import Path
from collections import Counter
from typing import Tuple, List, Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors
from Bio import SeqIO
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
    roc_curve, precision_recall_curve, average_precision_score
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Central configuration for all pipeline parameters."""

    # Paths
    DATA_DIR = Path("data/anopheles")
    OUTPUT_DIR = Path("results")
    TOKENIZER_DIR = Path("tokenizers")
    MODEL_DIR = Path("models")

    # Data curation
    MIN_SEQ_LENGTH = 200       # minimum transcript length (lncRNA definition)
    MAX_SEQ_LENGTH = 10000     # upper bound to avoid memory issues
    TRAIN_RATIO = 0.8
    VAL_RATIO = 0.1
    TEST_RATIO = 0.1

    # BPE tokenizer
    VOCAB_SIZES = [4096, 8192, 16384]  # hyperparameter search
    DEFAULT_VOCAB_SIZE = 8192
    MAX_TOKEN_LENGTH = 128     # max tokens per sequence (CPU-optimized)
    MAX_SAMPLES_PER_CLASS = 10000  # limit per class for memory safety

    # Model architecture (CPU-optimized: smaller dimensions)
    EMBED_DIM = 128
    LSTM_HIDDEN = 128
    LSTM_LAYERS = 2
    TRANSFORMER_HEADS = 4
    TRANSFORMER_LAYERS = 2
    TRANSFORMER_DIM_FF = 256
    DROPOUT = 0.3

    # Training (CPU-optimized: fewer epochs/folds, larger batch)
    BATCH_SIZE = 128
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 1e-5
    EPOCHS = 25
    PATIENCE = 5               # early stopping
    N_FOLDS = 3               # 3-fold CV (faster on CPU)
    SEED = 42

    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int):
    """Ensure reproducibility across runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================================
# PHASE 1: DATA CURATION
# ============================================================================

class DataCurator:
    """
    Curate ground truth datasets for Anopheles gambiae lncRNA classification.

    Data sources:
      - VectorBase / Ensembl Metazoa: Anopheles gambiae PEST strain annotation
      - BioProject accessions: PRJNA13334, PRJNA39, PRJNA209, PRJNA284179,
        PRJNA312456, PRJNA393797, PRJNA417311, PRJNA433205, PRJNA477315,
        PRJNA512682, PRJNA553567, PRJNA612228, PRJNA691995

    Class 0: mRNA (protein-coding transcripts)
    Class 1: lncRNA (long noncoding RNA transcripts)
    """

    def __init__(self, config: Config):
        self.config = config
        self.config.DATA_DIR.mkdir(parents=True, exist_ok=True)

    def load_fasta(self, fasta_path: Path) -> List[Dict]:
        """Parse FASTA file (plain or gzipped) into list of sequence records."""
        records = []
        open_fn = gzip.open if str(fasta_path).endswith('.gz') else open
        mode = 'rt' if str(fasta_path).endswith('.gz') else 'r'
        with open_fn(str(fasta_path), mode) as handle:
            for record in SeqIO.parse(handle, "fasta"):
                seq = str(record.seq).upper()
                # Filter: only canonical nucleotides, length constraints
                if self._is_valid_sequence(seq):
                    records.append({
                        "id": record.id,
                        "description": record.description,
                        "sequence": seq,
                        "length": len(seq)
                    })
        return records

    def _is_valid_sequence(self, seq: str) -> bool:
        """Validate sequence: canonical bases only, length within bounds."""
        if len(seq) < self.config.MIN_SEQ_LENGTH:
            return False
        if len(seq) > self.config.MAX_SEQ_LENGTH:
            return False
        valid_bases = set("ACGT")
        # Allow up to 5% ambiguous bases
        non_canonical = sum(1 for b in seq if b not in valid_bases)
        if non_canonical / len(seq) > 0.05:
            return False
        return True

    def clean_sequence(self, seq: str) -> str:
        """Replace ambiguous nucleotides with random canonical base."""
        canonical = "ACGT"
        cleaned = []
        for base in seq:
            if base in canonical:
                cleaned.append(base)
            else:
                cleaned.append(random.choice(canonical))
        return "".join(cleaned)

    def balance_classes(
        self, mrna_records: List[Dict], lncrna_records: List[Dict]
    ) -> Tuple[List[Dict], List[Dict]]:
        """Downsample to balanced dataset, capped by MAX_SAMPLES_PER_CLASS."""
        n_minority = min(len(mrna_records), len(lncrna_records))
        n_final = min(n_minority, self.config.MAX_SAMPLES_PER_CLASS)
        logger.info(
            f"Balancing: mRNA={len(mrna_records)}, lncRNA={len(lncrna_records)} "
            f"-> {n_final} each (cap={self.config.MAX_SAMPLES_PER_CLASS})"
        )
        mrna_balanced = random.sample(mrna_records, n_final)
        lncrna_balanced = random.sample(lncrna_records, n_final)
        return mrna_balanced, lncrna_balanced

    def prepare_dataset(
        self, mrna_fasta: Path, lncrna_fasta: Path
    ) -> pd.DataFrame:
        """
        Full curation pipeline: load, validate, clean, balance, merge.

        Returns DataFrame with columns: [id, sequence, label, length]
        """
        logger.info("Loading mRNA sequences (Class 0)...")
        mrna_records = self.load_fasta(mrna_fasta)
        logger.info(f"  Valid mRNA transcripts: {len(mrna_records)}")

        logger.info("Loading lncRNA sequences (Class 1)...")
        lncrna_records = self.load_fasta(lncrna_fasta)
        logger.info(f"  Valid lncRNA transcripts: {len(lncrna_records)}")

        # Balance classes
        mrna_balanced, lncrna_balanced = self.balance_classes(
            mrna_records, lncrna_records
        )

        # Clean sequences
        for rec in mrna_balanced:
            rec["sequence"] = self.clean_sequence(rec["sequence"])
            rec["label"] = 0
        for rec in lncrna_balanced:
            rec["sequence"] = self.clean_sequence(rec["sequence"])
            rec["label"] = 1

        # Merge and shuffle
        all_records = mrna_balanced + lncrna_balanced
        random.shuffle(all_records)

        df = pd.DataFrame(all_records)
        logger.info(f"Final dataset: {len(df)} sequences, balanced 50/50")
        return df

    def export_fasta(self, df: pd.DataFrame, output_dir: Path):
        """Export balanced classes to separate FASTA files."""
        output_dir.mkdir(parents=True, exist_ok=True)

        mrna_path = output_dir / "anopheles_mrna_balanced.fa"
        lncrna_path = output_dir / "anopheles_lncrna_balanced.fa"

        with open(mrna_path, "w") as f:
            for _, row in df[df["label"] == 0].iterrows():
                f.write(f">{row['id']}\n{row['sequence']}\n")

        with open(lncrna_path, "w") as f:
            for _, row in df[df["label"] == 1].iterrows():
                f.write(f">{row['id']}\n{row['sequence']}\n")

        logger.info(f"Exported: {mrna_path}, {lncrna_path}")


# ============================================================================
# PHASE 2: GENOMIC BPE TOKENIZER
# ============================================================================

class GenomicBPETokenizer:
    """
    Train and apply Byte-Pair Encoding tokenizer on genomic sequences.

    Unlike standard k-mer approaches (fixed k=3,4,5,6), BPE learns
    variable-length sub-word units from corpus frequency statistics,
    capturing biologically meaningful motifs without human assumptions.
    """

    def __init__(self, vocab_size: int = 8192):
        self.vocab_size = vocab_size
        self.tokenizer = None

    def train(self, sequences: List[str], save_path: Optional[Path] = None):
        """
        Train BPE tokenizer on genomic corpus.

        The 4-character genetic alphabet {A, C, G, T} is iteratively merged
        into variable-length tokens based on frequency. High-frequency dimers
        (e.g., 'CG', 'AT') are merged first, followed by longer motifs.
        """
        logger.info(f"Training BPE tokenizer (vocab_size={self.vocab_size})...")

        # Initialize BPE model — no pre-tokenizer so BPE merges operate
        # across entire sequences, learning variable-length genomic motifs
        tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))

        # NO pre-tokenizer: the full sequence is one "word" for BPE.
        # With Split(pattern="", behavior="isolated"), each char became a
        # separate word and BPE could never merge across chars. Removing it
        # lets BPE discover multi-nucleotide tokens (codons, motifs, etc.)
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([])

        # BPE trainer with explicit initial alphabet {A, C, G, T}
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=2,
            special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]"],
            initial_alphabet=list("ACGT"),
            show_progress=True
        )

        # Train from iterator (memory-efficient for large corpora)
        tokenizer.train_from_iterator(sequences, trainer=trainer)

        # Add post-processor for classification token
        tokenizer.post_processor = processors.TemplateProcessing(
            single="[CLS] $A [SEP]",
            special_tokens=[("[CLS]", 2), ("[SEP]", 3)]
        )

        # Enable padding
        tokenizer.enable_padding(
            pad_id=0, pad_token="[PAD]", length=None
        )

        self.tokenizer = tokenizer

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            tokenizer.save(str(save_path))
            logger.info(f"Tokenizer saved: {save_path}")

        return self

    def load(self, path: Path):
        """Load pre-trained tokenizer from file."""
        self.tokenizer = Tokenizer.from_file(str(path))
        return self

    def encode(self, sequence: str, max_length: int = 512) -> List[int]:
        """Encode nucleotide sequence to BPE token IDs."""
        encoding = self.tokenizer.encode(sequence)
        ids = encoding.ids[:max_length]
        # Pad if shorter
        if len(ids) < max_length:
            ids = ids + [0] * (max_length - len(ids))
        return ids

    def encode_batch(
        self, sequences: List[str], max_length: int = 128,
        chunk_size: int = 2000
    ) -> np.ndarray:
        """Batch encode sequences to token ID matrix (chunked for memory)."""
        n = len(sequences)
        # Use int16: vocab_size < 32767 fits safely
        batch = np.zeros((n, max_length), dtype=np.int16)
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            encodings = self.tokenizer.encode_batch(sequences[start:end])
            for i, enc in enumerate(encodings):
                ids = enc.ids[:max_length]
                batch[start + i, :len(ids)] = ids
            # Free tokenizer intermediate objects
            del encodings
            if start % 10000 == 0:
                logger.info(f"  Encoded {end}/{n} sequences...")
                gc.collect()
        return batch

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back to sequence."""
        return self.tokenizer.decode(token_ids)

    def get_vocab(self) -> Dict[str, int]:
        """Return full vocabulary mapping."""
        return self.tokenizer.get_vocab()

    def analyze_token_distribution(self, sequences: List[str]) -> pd.DataFrame:
        """
        Analyze learned BPE token frequencies across corpus.
        Useful for interpretability: which motifs dominate?
        """
        token_counts = Counter()
        for seq in sequences:
            enc = self.tokenizer.encode(seq)
            token_counts.update(enc.tokens)

        df = pd.DataFrame(
            token_counts.most_common(),
            columns=["token", "frequency"]
        )
        df["length"] = df["token"].apply(len)
        df["rank"] = range(1, len(df) + 1)
        return df


# ============================================================================
# PHASE 3: DATASET & DATALOADER
# ============================================================================

class GenomicDataset(Dataset):
    """PyTorch dataset for BPE-tokenized genomic sequences."""

    def __init__(
        self, sequences: np.ndarray, labels: np.ndarray
    ):
        self.sequences = torch.LongTensor(sequences)
        self.labels = torch.FloatTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


# ============================================================================
# PHASE 4: MODEL ARCHITECTURES
# ============================================================================

class BPE_BiLSTM(nn.Module):
    """
    Bidirectional LSTM classifier operating on BPE token embeddings.

    Architecture:
      Embedding(vocab_size, embed_dim) -> BiLSTM(hidden, layers) ->
      Attention Pooling -> FC -> Sigmoid

    No biological features. Pure sequence-to-label mapping via
    learned token representations.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        hidden_dim: int = 256,
        n_layers: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size, embed_dim, padding_idx=0
        )
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        # x: (batch, seq_len) token IDs
        mask = (x != 0).float().unsqueeze(-1)  # (batch, seq_len, 1)

        embedded = self.embedding(x)  # (batch, seq_len, embed_dim)
        lstm_out, _ = self.lstm(embedded)  # (batch, seq_len, hidden*2)

        # Attention-weighted pooling
        attn_weights = self.attention(lstm_out)  # (batch, seq_len, 1)
        attn_weights = attn_weights.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(attn_weights, dim=1)

        context = torch.sum(attn_weights * lstm_out, dim=1)  # (batch, hidden*2)
        logits = self.classifier(context).squeeze(-1)  # (batch,)
        return logits


class BPE_Transformer(nn.Module):
    """
    Lightweight Transformer Encoder for BPE-tokenized genomic sequences.

    Architecture:
      Embedding + Positional Encoding -> N x TransformerEncoderLayer ->
      [CLS] token extraction -> FC -> Sigmoid

    Inspired by genomic BERT but trained from scratch on species-specific
    BPE vocabulary. Sequence-only: zero engineered features.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        dim_feedforward: int = 512,
        max_seq_len: int = 512,
        dropout: float = 0.3
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_encoding = nn.Embedding(max_seq_len, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1)
        )

    def forward(self, x):
        # x: (batch, seq_len) token IDs
        batch_size, seq_len = x.shape

        # Padding mask for transformer (True = ignore)
        padding_mask = (x == 0)

        # Token + positional embeddings
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        embedded = self.embedding(x) + self.pos_encoding(positions)
        embedded = self.layer_norm(embedded)
        embedded = self.dropout(embedded)

        # Transformer encoding
        encoded = self.transformer(
            embedded, src_key_padding_mask=padding_mask
        )

        # Extract [CLS] token representation (position 0)
        cls_repr = encoded[:, 0, :]  # (batch, embed_dim)
        logits = self.classifier(cls_repr).squeeze(-1)  # (batch,)
        return logits


# ============================================================================
# PHASE 5: TRAINING ENGINE
# ============================================================================

class Trainer:
    """Training, validation, and evaluation engine."""

    def __init__(self, model: nn.Module, config: Config):
        self.model = model.to(config.DEVICE)
        self.config = config
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=config.EPOCHS
        )
        self.criterion = nn.BCEWithLogitsLoss()
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.history = {"train_loss": [], "val_loss": [], "val_auc": []}

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Single training epoch."""
        self.model.train()
        total_loss = 0.0

        for batch_seqs, batch_labels in dataloader:
            batch_seqs = batch_seqs.to(self.config.DEVICE)
            batch_labels = batch_labels.to(self.config.DEVICE)

            self.optimizer.zero_grad()
            logits = self.model(batch_seqs)
            loss = self.criterion(logits, batch_labels)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            self.optimizer.step()
            total_loss += loss.item() * len(batch_labels)

        return total_loss / len(dataloader.dataset)

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict:
        """Evaluate model on validation/test set."""
        self.model.eval()
        all_logits, all_labels = [], []
        total_loss = 0.0

        for batch_seqs, batch_labels in dataloader:
            batch_seqs = batch_seqs.to(self.config.DEVICE)
            batch_labels = batch_labels.to(self.config.DEVICE)

            logits = self.model(batch_seqs)
            loss = self.criterion(logits, batch_labels)
            total_loss += loss.item() * len(batch_labels)

            all_logits.extend(logits.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())

        all_logits = np.array(all_logits)
        all_labels = np.array(all_labels)
        all_probs = 1 / (1 + np.exp(-all_logits))  # sigmoid
        all_preds = (all_probs >= 0.5).astype(int)

        metrics = {
            "loss": total_loss / len(dataloader.dataset),
            "accuracy": accuracy_score(all_labels, all_preds),
            "precision": precision_score(all_labels, all_preds, zero_division=0),
            "recall": recall_score(all_labels, all_preds, zero_division=0),
            "f1": f1_score(all_labels, all_preds, zero_division=0),
            "auc": roc_auc_score(all_labels, all_probs),
            "ap": average_precision_score(all_labels, all_probs),
            "predictions": all_preds,
            "probabilities": all_probs,
            "labels": all_labels
        }
        return metrics

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        save_path: Optional[Path] = None
    ) -> Dict:
        """Full training loop with early stopping."""
        logger.info(
            f"Training: {self.config.EPOCHS} epochs, "
            f"patience={self.config.PATIENCE}, device={self.config.DEVICE}"
        )

        for epoch in range(1, self.config.EPOCHS + 1):
            train_loss = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)
            self.scheduler.step()

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_auc"].append(val_metrics["auc"])

            logger.info(
                f"Epoch {epoch:3d}/{self.config.EPOCHS} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val AUC: {val_metrics['auc']:.4f} | "
                f"Val F1: {val_metrics['f1']:.4f}"
            )

            # Early stopping check
            if val_metrics["loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["loss"]
                self.patience_counter = 0
                if save_path:
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(self.model.state_dict(), save_path)
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.PATIENCE:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

        return self.history


# ============================================================================
# PHASE 6: CROSS-VALIDATION & BENCHMARKING
# ============================================================================

class CrossValidator:
    """Stratified K-Fold cross-validation with full metric tracking."""

    def __init__(self, config: Config):
        self.config = config

    def run(
        self,
        sequences: np.ndarray,
        labels: np.ndarray,
        model_class: type,
        model_kwargs: Dict,
        model_name: str = "model"
    ) -> Dict:
        """Execute stratified K-fold CV and aggregate results."""
        skf = StratifiedKFold(
            n_splits=self.config.N_FOLDS,
            shuffle=True,
            random_state=self.config.SEED
        )

        fold_metrics = []
        all_test_probs = np.zeros(len(labels))
        all_test_preds = np.zeros(len(labels))

        for fold, (train_idx, val_idx) in enumerate(skf.split(sequences, labels), 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"FOLD {fold}/{self.config.N_FOLDS} - {model_name}")
            logger.info(f"{'='*60}")

            train_data = GenomicDataset(sequences[train_idx], labels[train_idx])
            val_data = GenomicDataset(sequences[val_idx], labels[val_idx])

            train_loader = DataLoader(
                train_data, batch_size=self.config.BATCH_SIZE,
                shuffle=True, num_workers=0, pin_memory=False
            )
            val_loader = DataLoader(
                val_data, batch_size=self.config.BATCH_SIZE,
                shuffle=False, num_workers=0, pin_memory=False
            )

            model = model_class(**model_kwargs)
            trainer = Trainer(model, self.config)

            save_path = (
                self.config.MODEL_DIR / f"{model_name}_fold{fold}.pt"
            )
            trainer.fit(train_loader, val_loader, save_path=save_path)

            # Load best model for evaluation
            model.load_state_dict(torch.load(save_path, map_location=self.config.DEVICE))
            model.to(self.config.DEVICE)
            trainer.model = model

            metrics = trainer.evaluate(val_loader)
            fold_metrics.append(metrics)

            all_test_probs[val_idx] = metrics["probabilities"]
            all_test_preds[val_idx] = metrics["predictions"]

            logger.info(
                f"Fold {fold} Results: "
                f"AUC={metrics['auc']:.4f}, F1={metrics['f1']:.4f}, "
                f"Acc={metrics['accuracy']:.4f}"
            )

        # Aggregate across folds
        summary = {
            "model": model_name,
            "accuracy_mean": np.mean([m["accuracy"] for m in fold_metrics]),
            "accuracy_std": np.std([m["accuracy"] for m in fold_metrics]),
            "precision_mean": np.mean([m["precision"] for m in fold_metrics]),
            "precision_std": np.std([m["precision"] for m in fold_metrics]),
            "recall_mean": np.mean([m["recall"] for m in fold_metrics]),
            "recall_std": np.std([m["recall"] for m in fold_metrics]),
            "f1_mean": np.mean([m["f1"] for m in fold_metrics]),
            "f1_std": np.std([m["f1"] for m in fold_metrics]),
            "auc_mean": np.mean([m["auc"] for m in fold_metrics]),
            "auc_std": np.std([m["auc"] for m in fold_metrics]),
            "all_probs": all_test_probs,
            "all_preds": all_test_preds,
            "all_labels": labels
        }

        logger.info(f"\n{'='*60}")
        logger.info(f"CROSS-VALIDATION SUMMARY: {model_name}")
        logger.info(f"{'='*60}")
        logger.info(f"  Accuracy:  {summary['accuracy_mean']:.4f} ± {summary['accuracy_std']:.4f}")
        logger.info(f"  Precision: {summary['precision_mean']:.4f} ± {summary['precision_std']:.4f}")
        logger.info(f"  Recall:    {summary['recall_mean']:.4f} ± {summary['recall_std']:.4f}")
        logger.info(f"  F1-Score:  {summary['f1_mean']:.4f} ± {summary['f1_std']:.4f}")
        logger.info(f"  ROC-AUC:   {summary['auc_mean']:.4f} ± {summary['auc_std']:.4f}")

        return summary


# ============================================================================
# PHASE 7: K-MER BASELINE (FOR BENCHMARKING)
# ============================================================================

class KmerBaseline:
    """
    Traditional k-mer frequency-based classifier for benchmarking.

    Implements 3-mer and 6-mer frequency extraction with Random Forest
    and SVM classifiers as baselines against BPE models.
    """

    def __init__(self, k_values: List[int] = [3, 6]):
        self.k_values = k_values

    def extract_kmer_features(
        self, sequences: List[str], k: int
    ) -> np.ndarray:
        """Extract normalized k-mer frequency vector for each sequence."""
        # Generate all possible k-mers
        from itertools import product
        all_kmers = [''.join(p) for p in product("ACGT", repeat=k)]
        kmer_to_idx = {km: i for i, km in enumerate(all_kmers)}
        n_features = len(all_kmers)

        features = np.zeros((len(sequences), n_features))
        for i, seq in enumerate(sequences):
            total = max(1, len(seq) - k + 1)
            for j in range(len(seq) - k + 1):
                kmer = seq[j:j+k]
                if kmer in kmer_to_idx:
                    features[i, kmer_to_idx[kmer]] += 1
            features[i] /= total  # normalize to frequency

        return features

    def benchmark(
        self, sequences: List[str], labels: np.ndarray, n_folds: int = 5
    ) -> pd.DataFrame:
        """Run k-mer baseline benchmarks with RF and SVM."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.svm import SVC

        results = []

        for k in self.k_values:
            logger.info(f"Extracting {k}-mer features...")
            features = self.extract_kmer_features(sequences, k)
            logger.info(f"  Feature matrix shape: {features.shape}")

            classifiers = {
                f"{k}mer_RF": RandomForestClassifier(
                    n_estimators=200, max_depth=20, random_state=42, n_jobs=-1
                ),
                f"{k}mer_SVM": SVC(
                    kernel="rbf", probability=True, random_state=42
                )
            }

            for name, clf in classifiers.items():
                logger.info(f"  Training {name}...")
                skf = StratifiedKFold(
                    n_splits=n_folds, shuffle=True, random_state=42
                )

                fold_scores = []
                for train_idx, val_idx in skf.split(features, labels):
                    clf.fit(features[train_idx], labels[train_idx])
                    probs = clf.predict_proba(features[val_idx])[:, 1]
                    preds = (probs >= 0.5).astype(int)

                    fold_scores.append({
                        "accuracy": accuracy_score(labels[val_idx], preds),
                        "f1": f1_score(labels[val_idx], preds),
                        "auc": roc_auc_score(labels[val_idx], probs)
                    })

                avg = {
                    "model": name,
                    "accuracy": np.mean([s["accuracy"] for s in fold_scores]),
                    "f1": np.mean([s["f1"] for s in fold_scores]),
                    "auc": np.mean([s["auc"] for s in fold_scores])
                }
                results.append(avg)
                logger.info(
                    f"    {name}: Acc={avg['accuracy']:.4f}, "
                    f"F1={avg['f1']:.4f}, AUC={avg['auc']:.4f}"
                )

        return pd.DataFrame(results)


# ============================================================================
# PHASE 8: INTERPRETABILITY - TOKEN ANALYSIS
# ============================================================================

class TokenInterpreter:
    """
    Analyze learned BPE tokens for biological interpretability.

    Extract most frequent tokens, map to known motifs, and
    compute attention-based importance scores.
    """

    def __init__(self, tokenizer: GenomicBPETokenizer):
        self.tokenizer = tokenizer

    def class_specific_tokens(
        self, sequences: List[str], labels: np.ndarray, top_k: int = 50
    ) -> Dict:
        """
        Identify tokens enriched in lncRNA vs mRNA classes.
        Compute log2 fold-change of token frequency between classes.
        """
        mrna_counts = Counter()
        lncrna_counts = Counter()
        mrna_total = 0
        lncrna_total = 0

        for seq, label in zip(sequences, labels):
            enc = self.tokenizer.tokenizer.encode(seq)
            if label == 0:
                mrna_counts.update(enc.tokens)
                mrna_total += len(enc.tokens)
            else:
                lncrna_counts.update(enc.tokens)
                lncrna_total += len(enc.tokens)

        # Compute normalized frequencies and log2 fold-change
        all_tokens = set(mrna_counts.keys()) | set(lncrna_counts.keys())
        enrichment = []

        for token in all_tokens:
            mrna_freq = (mrna_counts[token] + 1) / (mrna_total + 1)
            lncrna_freq = (lncrna_counts[token] + 1) / (lncrna_total + 1)
            log2fc = np.log2(lncrna_freq / mrna_freq)
            enrichment.append({
                "token": token,
                "length": len(token),
                "mrna_freq": mrna_freq,
                "lncrna_freq": lncrna_freq,
                "log2fc": log2fc,
                "total_count": mrna_counts[token] + lncrna_counts[token]
            })

        df = pd.DataFrame(enrichment)
        df = df.sort_values("log2fc", ascending=False)

        result = {
            "lncrna_enriched": df.head(top_k),
            "mrna_enriched": df.tail(top_k).iloc[::-1],
            "full_table": df
        }
        return result

    def attention_rollout(
        self, model: BPE_Transformer, sequences: np.ndarray, config: Config
    ) -> np.ndarray:
        """
        Compute attention rollout for transformer model.
        Returns per-token importance scores averaged across sequences.
        """
        model.eval()
        model.to(config.DEVICE)

        # Hook to capture attention weights
        attention_maps = []

        def hook_fn(module, input, output):
            # TransformerEncoderLayer stores attention in output
            attention_maps.append(output)

        hooks = []
        for layer in model.transformer.layers:
            h = layer.self_attn.register_forward_hook(hook_fn)
            hooks.append(h)

        # Forward pass
        x = torch.LongTensor(sequences[:100]).to(config.DEVICE)
        with torch.no_grad():
            _ = model(x)

        # Clean up hooks
        for h in hooks:
            h.remove()

        return np.array([am.cpu().numpy() for am in attention_maps])


# ============================================================================
# MAIN PIPELINE EXECUTION
# ============================================================================

def main():
    """Execute the complete BPE-lncRNA pipeline."""
    parser = argparse.ArgumentParser(
        description="BPE-LncRNA: Genomic BPE for lncRNA Identification"
    )
    parser.add_argument(
        "--mrna-fasta", type=str, required=True,
        help="Path to mRNA FASTA file (Anopheles gambiae)"
    )
    parser.add_argument(
        "--lncrna-fasta", type=str, required=True,
        help="Path to lncRNA FASTA file (Anopheles gambiae)"
    )
    parser.add_argument(
        "--vocab-size", type=int, default=Config.DEFAULT_VOCAB_SIZE,
        help="BPE vocabulary size (default: 8192)"
    )
    parser.add_argument(
        "--epochs", type=int, default=Config.EPOCHS,
        help="Training epochs (default: 50)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=Config.BATCH_SIZE,
        help="Batch size (default: 64)"
    )
    parser.add_argument(
        "--model", type=str, choices=["bilstm", "transformer", "both"],
        default="both", help="Model architecture to train"
    )
    parser.add_argument(
        "--benchmark", action="store_true",
        help="Run k-mer baseline benchmarks"
    )
    parser.add_argument(
        "--max-samples", type=int, default=Config.MAX_SAMPLES_PER_CLASS,
        help="Max samples per class (default: 20000)"
    )
    args = parser.parse_args()

    # Initialize
    config = Config()
    config.DEFAULT_VOCAB_SIZE = args.vocab_size
    config.EPOCHS = args.epochs
    config.BATCH_SIZE = args.batch_size
    config.MAX_SAMPLES_PER_CLASS = args.max_samples
    set_seed(config.SEED)

    logger.info("=" * 70)
    logger.info("BPE-LncRNA Pipeline: Anopheles gambiae")
    logger.info("=" * 70)
    logger.info(f"Device: {config.DEVICE}")
    logger.info(f"Vocab Size: {args.vocab_size}")

    # ---- Step 1: Data Curation ----
    logger.info("\n[STEP 1] Data Curation")
    curator = DataCurator(config)
    df = curator.prepare_dataset(
        Path(args.mrna_fasta), Path(args.lncrna_fasta)
    )
    curator.export_fasta(df, config.DATA_DIR / "balanced")

    sequences = df["sequence"].tolist()
    labels = df["label"].values

    # Free DataFrame, keep only sequences and labels
    del df
    gc.collect()
    logger.info(f"  Sequences in memory: {len(sequences)}, peak ~{len(sequences)*1500//1024//1024} MB est.")

    # ---- Step 2: BPE Tokenizer Training ----
    logger.info("\n[STEP 2] Training Genomic BPE Tokenizer")
    tokenizer = GenomicBPETokenizer(vocab_size=args.vocab_size)
    tokenizer_path = config.TOKENIZER_DIR / f"bpe_v{args.vocab_size}.json"
    # Train on subset (10K sequences is sufficient for vocabulary learning)
    train_subset = sequences[:min(10000, len(sequences))]
    tokenizer.train(train_subset, save_path=tokenizer_path)
    del train_subset
    gc.collect()

    # Encode all sequences (chunked to avoid OOM)
    logger.info("Encoding sequences with BPE tokenizer...")
    encoded = tokenizer.encode_batch(sequences, max_length=config.MAX_TOKEN_LENGTH)
    logger.info(f"Encoded matrix shape: {encoded.shape}, dtype={encoded.dtype}")
    logger.info(f"  Encoded array size: {encoded.nbytes // 1024 // 1024} MB")

    # Free raw sequences - no longer needed after encoding
    del sequences
    gc.collect()

    # ---- Step 3: Model Training & Cross-Validation ----
    cv = CrossValidator(config)
    results = {}

    if args.model in ["bilstm", "both"]:
        logger.info("\n[STEP 3a] BiLSTM Cross-Validation")
        bilstm_kwargs = {
            "vocab_size": args.vocab_size + 4,  # +4 for special tokens
            "embed_dim": config.EMBED_DIM,
            "hidden_dim": config.LSTM_HIDDEN,
            "n_layers": config.LSTM_LAYERS,
            "dropout": config.DROPOUT
        }
        results["BiLSTM"] = cv.run(
            encoded, labels, BPE_BiLSTM, bilstm_kwargs, "BPE_BiLSTM"
        )

    if args.model in ["transformer", "both"]:
        logger.info("\n[STEP 3b] Transformer Cross-Validation")
        transformer_kwargs = {
            "vocab_size": args.vocab_size + 4,
            "embed_dim": config.EMBED_DIM,
            "n_heads": config.TRANSFORMER_HEADS,
            "n_layers": config.TRANSFORMER_LAYERS,
            "dim_feedforward": config.TRANSFORMER_DIM_FF,
            "max_seq_len": config.MAX_TOKEN_LENGTH,
            "dropout": config.DROPOUT
        }
        results["Transformer"] = cv.run(
            encoded, labels, BPE_Transformer, transformer_kwargs, "BPE_Transformer"
        )

    # ---- Step 4: K-mer Baseline Benchmark ----
    if args.benchmark:
        logger.info("\n[STEP 4] K-mer Baseline Benchmarking (reloading sequences)")
        # Reload from exported balanced FASTAs
        mrna_fa = config.DATA_DIR / "balanced" / "anopheles_mrna_balanced.fa"
        lncrna_fa = config.DATA_DIR / "balanced" / "anopheles_lncrna_balanced.fa"
        bench_seqs, bench_labels = [], []
        for record in SeqIO.parse(str(mrna_fa), "fasta"):
            bench_seqs.append(str(record.seq))
            bench_labels.append(0)
        for record in SeqIO.parse(str(lncrna_fa), "fasta"):
            bench_seqs.append(str(record.seq))
            bench_labels.append(1)
        bench_labels = np.array(bench_labels)
        kmer = KmerBaseline(k_values=[3, 6])
        baseline_results = kmer.benchmark(bench_seqs, bench_labels, n_folds=config.N_FOLDS)
        logger.info("\nBaseline Results:")
        logger.info(baseline_results.to_string(index=False))
        del bench_seqs
        gc.collect()

    # ---- Step 5: Interpretability ----
    logger.info("\n[STEP 5] Token Interpretability Analysis")
    # Reload sequences for token analysis
    interp_seqs, interp_labels = [], []
    mrna_fa = config.DATA_DIR / "balanced" / "anopheles_mrna_balanced.fa"
    lncrna_fa = config.DATA_DIR / "balanced" / "anopheles_lncrna_balanced.fa"
    for record in SeqIO.parse(str(mrna_fa), "fasta"):
        interp_seqs.append(str(record.seq))
        interp_labels.append(0)
    for record in SeqIO.parse(str(lncrna_fa), "fasta"):
        interp_seqs.append(str(record.seq))
        interp_labels.append(1)
    interp_labels = np.array(interp_labels)

    interpreter = TokenInterpreter(tokenizer)
    enrichment = interpreter.class_specific_tokens(interp_seqs, interp_labels)

    logger.info("\nTop 20 lncRNA-enriched BPE tokens:")
    logger.info(
        enrichment["lncrna_enriched"][["token", "length", "log2fc"]]
        .head(20).to_string(index=False)
    )
    logger.info("\nTop 20 mRNA-enriched BPE tokens:")
    logger.info(
        enrichment["mrna_enriched"][["token", "length", "log2fc"]]
        .head(20).to_string(index=False)
    )
    del interp_seqs
    gc.collect()

    # ---- Step 6: Save Results ----
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    for name, res in results.items():
        summary_rows.append({
            "Model": name,
            "Accuracy": f"{res['accuracy_mean']:.4f} ± {res['accuracy_std']:.4f}",
            "Precision": f"{res['precision_mean']:.4f} ± {res['precision_std']:.4f}",
            "Recall": f"{res['recall_mean']:.4f} ± {res['recall_std']:.4f}",
            "F1-Score": f"{res['f1_mean']:.4f} ± {res['f1_std']:.4f}",
            "ROC-AUC": f"{res['auc_mean']:.4f} ± {res['auc_std']:.4f}"
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(config.OUTPUT_DIR / "performance_summary.csv", index=False)
    logger.info(f"\nResults saved to {config.OUTPUT_DIR}/")
    logger.info("\n" + summary_df.to_string(index=False))

    logger.info("\n" + "=" * 70)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
