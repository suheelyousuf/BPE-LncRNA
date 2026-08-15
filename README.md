# BPE-LncRNA
### Byte-Pair Encoding for Long Noncoding RNA Identification in *Anopheles gambiae*

> A feature-free deep learning framework for lncRNA classification using genomic Byte-Pair Encoding (BPE), Bidirectional LSTM (BiLSTM), and Transformer architectures.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](httpse/PyTorch-2.x-red
![Licensemg.shields.io/badge/License-MIT-green

---

## Overview

**BPE-LncRNA** is an end-to-end pipeline for identifying **long noncoding RNAs (lncRNAs)** from transcript sequences in *Anopheles gambiae* using **Byte-Pair Encoding (BPE)** and deep learning.

Unlike traditional lncRNA prediction tools that rely on handcrafted biological features such as:

- ORF length
- Hexamer frequencies
- Codon usage bias
- GC content
- Fickett scores

BPE-LncRNA treats genomic sequences as a language and learns biologically meaningful motifs directly from sequence data using **sub-word tokenization**.

The framework includes:

- Data curation from transcript FASTA files
- Genomic BPE tokenizer training
- BiLSTM classification
- Transformer classification
- Cross-validation benchmarking
- Traditional k-mer baseline comparison
- Token interpretability analysis

---

## Pipeline Architecture

```text
FASTA Files
     │
     ▼
Data Curation
     │
     ▼
Balanced Dataset
(mRNA vs lncRNA)
     │
     ▼
BPE Tokenizer Training
     │
     ▼
Sequence Encoding
     │
     ▼
┌──────────────────┬─────────────────┐
│                  │                 │
▼                  ▼                 ▼
BiLSTM       Transformer      k-mer Baselines
│                  │
└──────────┬───────┘
           ▼
Cross Validation
           ▼
Performance Metrics
           ▼
Token Enrichment Analysis
```

---

## Key Features

- ✅ Completely sequence-based classification
- ✅ No handcrafted biological features
- ✅ Genomic BPE vocabulary learning
- ✅ Attention-based BiLSTM architecture
- ✅ Lightweight Transformer encoder
- ✅ Stratified K-Fold cross-validation
- ✅ CPU-friendly implementation
- ✅ Built-in k-mer benchmarking
- ✅ Biological interpretability through token enrichment
- ✅ Reproducible training and evaluation

---

## Installation

### Clone Repository

```bash
git clone https://github.com/suheelyousuf/BPE-LncRNA.git
cd BPE-LncRNA
```

### Create Environment

```bash
python -m venv venv
source venv/bin/activate
```

### Install Dependencies

```bash
pip install \
  torch \
  tokenizers \
  biopython \
  scikit-learn \
  pandas \
  numpy \
  matplotlib \
  seaborn
```

---

## Repository Structure

```text
BPE-LncRNA/
│
├── bpe_lncrna.py
│
├── data/
│   └── anopheles/
│
├── tokenizers/
│
├── models/
│
├── results/
│
└── README.md
```

After execution:

```text
data/anopheles/
└── balanced/
    ├── anopheles_mrna_balanced.fa
    └── anopheles_lncrna_balanced.fa

tokenizers/
└── bpe_v8192.json

models/
├── BPE_BiLSTM_fold1.pt
├── BPE_BiLSTM_fold2.pt
├── BPE_BiLSTM_fold3.pt
├── BPE_Transformer_fold1.pt
├── BPE_Transformer_fold2.pt
└── BPE_Transformer_fold3.pt

results/
└── performance_summary.csv
```

---

## Input Data

The pipeline requires two FASTA files:

### mRNA FASTA

```fasta
>Transcript_1
ATGCTAGCTAGCTAGCTAGCTA...
```

### lncRNA FASTA

```fasta
>lncRNA_1
TTTTGACCTGGATCCAGTCA...
```

Supported formats:

- `.fa`
- `.fasta`
- `.fa.gz`
- `.fasta.gz`

---

## Data Curation

### Quality Filters

| Parameter | Value |
|------------|----------|
| Minimum sequence length | 200 nt |
| Maximum sequence length | 10,000 nt |
| Maximum ambiguous bases | 5% |
| Class balancing | Yes |
| Maximum samples per class | 10,000 |

### Classes

| Label | Class |
|---------|---------|
| 0 | Protein-coding mRNA |
| 1 | Long non-coding RNA |

---

## Genomic BPE Tokenizer

The tokenizer begins with the nucleotide alphabet:

```text
A
C
G
T
```

and learns variable-length genomic motifs using Byte-Pair Encoding.

### Special Tokens

```text
[PAD]
[UNK]
[CLS]
[SEP]
```

### Supported Vocabulary Sizes

```python
VOCAB_SIZES = [4096, 8192, 16384]
```

Default:

```python
8192
```

### Why BPE?

| Traditional k-mers | BPE |
|--------------------|-----|
| Fixed-length | Variable-length |
| Human-defined | Data-driven |
| Sparse representation | Compact representation |
| Feature engineering required | No feature engineering |
| Hard to interpret globally | Learns biological motifs |

---

## Deep Learning Architectures

### BPE-BiLSTM

Architecture:

```text
Embedding
   ↓
BiLSTM
   ↓
Attention Pooling
   ↓
Dense Layers
   ↓
Binary Classification
```

Configuration:

| Parameter | Value |
|------------|----------|
| Embedding size | 128 |
| Hidden size | 128 |
| LSTM layers | 2 |
| Dropout | 0.3 |

---

### BPE-Transformer

Architecture:

```text
Token Embeddings
        +
Positional Embeddings
        ↓
Transformer Encoder
        ↓
[CLS] Representation
        ↓
Classifier
```

Configuration:

| Parameter | Value |
|------------|----------|
| Embedding size | 128 |
| Attention heads | 4 |
| Encoder layers | 2 |
| Feed-forward dimension | 256 |
| Dropout | 0.3 |

---

## Training Configuration

| Parameter | Value |
|------------|----------|
| Learning Rate | 3e-4 |
| Optimizer | AdamW |
| Weight Decay | 1e-5 |
| Scheduler | Cosine Annealing |
| Epochs | 25 |
| Batch Size | 128 |
| Early Stopping Patience | 5 |
| Cross Validation | 3-Fold |
| Random Seed | 42 |

---

## Running the Pipeline

### Train Both Models

```bash
python bpe_lncrna.py \
    --mrna-fasta data/mrna.fa.gz \
    --lncrna-fasta data/lncrna.fa.gz
```

---

### Train Only BiLSTM

```bash
python bpe_lncrna.py \
    --mrna-fasta data/mrna.fa.gz \
    --lncrna-fasta data/lncrna.fa.gz \
    --model bilstm
```

---

### Train Only Transformer

```bash
python bpe_lncrna.py \
    --mrna-fasta data/mrna.fa.gz \
    --lncrna-fasta data/lncrna.fa.gz \
    --model transformer
```

---

### Run k-mer Baselines

```bash
python bpe_lncrna.py \
    --mrna-fasta data/mrna.fa.gz \
    --lncrna-fasta data/lncrna.fa.gz \
    --benchmark
```

---

### Custom Vocabulary Size

```bash
python bpe_lncrna.py \
    --mrna-fasta data/mrna.fa.gz \
    --lncrna-fasta data/lncrna.fa.gz \
    --vocab-size 16384
```

---

### Custom Training Parameters

```bash
python bpe_lncrna.py \
    --mrna-fasta data/mrna.fa.gz \
    --lncrna-fasta data/lncrna.fa.gz \
    --epochs 50 \
    --batch-size 64 \
    --max-samples 20000
```

---

## Command-Line Options

| Argument | Description |
|-----------|-------------|
| `--mrna-fasta` | mRNA FASTA file |
| `--lncrna-fasta` | lncRNA FASTA file |
| `--vocab-size` | BPE vocabulary size |
| `--epochs` | Training epochs |
| `--batch-size` | Batch size |
| `--model` | `bilstm`, `transformer`, or `both` |
| `--benchmark` | Run k-mer baseline evaluation |
| `--max-samples` | Maximum transcripts per class |

---

## Evaluation Metrics

The pipeline reports:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Average Precision (AP)

Metrics are aggregated across cross-validation folds as:

```text
Mean ± Standard Deviation
```

---

## Example Output

```text
============================================================
CROSS-VALIDATION SUMMARY: BPE_BiLSTM
============================================================

Accuracy : 0.8421 ± 0.0082
Precision: 0.8390 ± 0.0094
Recall   : 0.8467 ± 0.0105
F1-Score : 0.8428 ± 0.0074
ROC-AUC  : 0.9115 ± 0.0048
```

---

## Interpretability Analysis

The framework enables biological interpretation of learned BPE motifs.

Outputs include:

- Most frequent genomic tokens
- lncRNA-enriched motifs
- mRNA-enriched motifs
- Log₂ fold change tables
- Transformer attention analysis

Example:

### lncRNA-Enriched Tokens

```text
ACTCCCACCC
TGGGTTCTTGG
ATTCCAGAAC
ATTA
```

### mRNA-Enriched Tokens

```text
GCCGCCGCCGCC
GGCGGCGGCGGC
TGGTGGCCC
```

---

## Reproducibility

The pipeline enforces deterministic execution:

```python
SEED = 42
```

Applied to:

- Python
- NumPy
- PyTorch
- CUDA (if available)

---

## System Requirements

| Resource | Recommendation |
|-----------|----------------|
| Python | ≥ 3.9 |
| RAM | 8–32 GB |
| CPU | Multi-core |
| GPU | Optional |

The implementation is optimized for CPU execution while automatically utilizing GPUs when available.

---

## Citation

If you use BPE-LncRNA in your research, please cite:

```bibtex
@article{BPELncRNA,
  title={BPE-LncRNA: Byte-Pair Encoding for Long Noncoding RNA Identification in Anopheles gambiae},
  author={Suheel Yousuf Wani, Abdul Wahid}
}
```

---

## Future Directions

- Cross-species lncRNA prediction
- Large-scale pretraining on genomic corpora
- DNABERT-style transfer learning
- Foundation models for insect genomics
- Comparative tokenizer analysis
- Functional motif discovery

---

## License

This project is distributed under the **MIT License**.

---

## Authors

**Suheel Yousuf Wani**
**Abdul Wahid**
---

## Acknowledgments

This work explores the intersection of **Natural Language Processing (NLP)** and **Computational Genomics**, leveraging modern sub-word tokenization approaches for biological sequence understanding and lncRNA discovery.
