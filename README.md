# Semantic Plagiarism Engine

An advanced quantitative plagiarism detection engine built in Python. This system analyzes text corpora to catch duplicates and rewrites using three parallel detection approaches:
1. **Jaccard Similarity** — Exact lexical overlap detection
2. **MinHash** — Fast approximate Jaccard similarity
3. **SimHash** — TF-IDF weighted semantic fingerprinting for paraphrase detection

## Team Members
* **Arian Jafari**
* **Mehrnia Amouei**

---

## How It Works

### Three Detection Methods

#### 1. Jaccard Similarity (Exact Baseline)
- Measures exact word n-gram overlap between documents
- Fast for small-scale comparisons
- Recommended threshold: 0.10–0.30

#### 2. MinHash (Fast Approximation)
- Creates fixed-length signatures (128 hash values) for each document
- Approximates Jaccard similarity without full set comparison
- Can be accelerated with **Locality-Sensitive Hashing (LSH)** for large corpora
- Recommended threshold: 0.10–0.30

#### 3. SimHash (Semantic Fingerprinting)
- Creates TF-IDF weighted 64-bit fingerprints
- Captures token importance beyond exact word matches
- Better at detecting paraphrased plagiarism
- Recommended threshold: 0.75–0.95

---

## Development Setup

### 1. Create Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt -e .
```

### 3. Verify Installation
```bash
python -m plagiarism_engine.cli --help
```

---

## CLI Commands

The engine provides 4 commands via CLI:

### Command 1: `compare` — Compare Two Files

Compare two individual text files using all three methods simultaneously.

**Basic Usage:**
```bash
python -m plagiarism_engine.cli compare \
    --file-a data/sample_corpus/doc_01.txt \
    --file-b data/sample_corpus/doc_02.txt
```

**With All Parameters:**
```bash
python -m plagiarism_engine.cli compare \
    --file-a data/sample_corpus/doc_01.txt \
    --file-b data/sample_corpus/doc_02.txt \
    --shingle-size 3 \
    --num-hashes 128 \
    --simhash-bits 64 \
    --output outputs/two_file_compare.json
```

**Parameters:**
- `--file-a` (required): Path to first text file
- `--file-b` (required): Path to second text file
- `--shingle-size` (default: 3): Size of word n-grams for Jaccard/MinHash
- `--num-hashes` (default: 128): Number of hash functions for MinHash
- `--simhash-bits` (default: 64): Bit size for SimHash fingerprint
- `--output` (optional): JSON file to save results (prints to stdout if omitted)

**Real Output Example** *(doc_01.txt vs doc_02.txt — near-duplicate pair)*:
```json
{
  "jaccard": 0.5625,
  "minhash": 0.6015625,
  "simhash_similarity": 0.84375,
  "simhash_hamming_distance": 10,
  "shingles_a": 124,
  "shingles_b": 126,
  "tokens_a": 126,
  "tokens_b": 128,
  "file_a": "data/sample_corpus/doc_01.txt",
  "file_b": "data/sample_corpus/doc_02.txt"
}
```

---

### Command 2: `corpus` — Scan Document Folder

Find similar document pairs across an entire folder using MinHash signatures.

**Brute Force (small datasets, guaranteed results):**
```bash
python -m plagiarism_engine.cli corpus \
    --data data/sample_corpus \
    --threshold 0.15 \
    --no-use-lsh \
    --output outputs/candidates.csv
```

**With LSH (recommended for large datasets):**
```bash
python -m plagiarism_engine.cli corpus \
    --data data/sample_corpus \
    --threshold 0.25 \
    --shingle-size 3 \
    --num-hashes 128 \
    --bands 32 \
    --use-lsh \
    --output outputs/candidates.csv
```

**Parameters:**
- `--data` (required): Path to folder containing text files (`.txt`, `.md`)
- `--threshold` (default: 0.25): Minimum Jaccard similarity to report a match
- `--shingle-size` (default: 3): Size of word n-grams
- `--num-hashes` (default: 128): Number of MinHash hash functions
- `--bands` (default: 32): Number of LSH bands (`num-hashes` must be divisible by `bands`)
- `--use-lsh` / `--no-use-lsh` (default: `--use-lsh`): Toggle LSH acceleration
- `--output` (optional): CSV file to save matched pairs

**LSH vs Brute Force:**
- **Brute Force** (`--no-use-lsh`): Compares all n(n−1)/2 pairs — guarantees no misses, slow at scale
- **LSH** (`--use-lsh`): Only scores candidate pairs from band collisions — much faster, may miss pairs at very low similarity

**Real CSV Output** *(5 sample documents, threshold=0.15, brute force)*:
```csv
doc_a,doc_b,jaccard,minhash
doc_01.txt,doc_02.txt,0.562500,0.601562
doc_03.txt,doc_04.txt,0.519481,0.562500
```

**Console Summary:**
```json
{
  "documents": 5,
  "total_pairs": 10,
  "candidate_pairs": 10,
  "comparison_reduction": 0.0,
  "matches": 2,
  "seconds": 0.010703
}
```

---

### Command 3: `pairs` — Evaluate Labeled Dataset

Evaluate plagiarism detection accuracy on a CSV with labeled pairs.

#### Jaccard Method (Threshold: 0.10)
```bash
python -m plagiarism_engine.cli pairs \
    --pairs data/raw/questions.csv \
    --text-col-a question1 \
    --text-col-b question2 \
    --label-col is_duplicate \
    --method jaccard \
    --threshold 0.10 \
    --limit 5000 \
    --output outputs/metrics_jaccard.csv \
    --predictions-output outputs/predictions_jaccard.csv
```

#### MinHash Method (Threshold: 0.10)
```bash
python -m plagiarism_engine.cli pairs \
    --pairs data/raw/questions.csv \
    --text-col-a question1 \
    --text-col-b question2 \
    --label-col is_duplicate \
    --method minhash \
    --threshold 0.10 \
    --num-hashes 128 \
    --limit 5000 \
    --output outputs/metrics_minhash.csv \
    --predictions-output outputs/predictions_minhash.csv
```

#### SimHash Method (Threshold: 0.75)
```bash
python -m plagiarism_engine.cli pairs \
    --pairs data/raw/questions.csv \
    --text-col-a question1 \
    --text-col-b question2 \
    --label-col is_duplicate \
    --method simhash \
    --threshold 0.75 \
    --simhash-bits 64 \
    --limit 5000 \
    --output outputs/metrics_simhash.csv \
    --predictions-output outputs/predictions_simhash.csv
```

**Parameters:**
- `--pairs` (required): Path to CSV file with text pairs
- `--text-col-a` (required): Name of first text column
- `--text-col-b` (required): Name of second text column
- `--label-col` (optional): Label column for evaluation (0 = different, 1 = duplicate)
- `--method` (default: `simhash`): `jaccard`, `minhash`, or `simhash`
- `--threshold` (default: 0.75): Similarity threshold for binary prediction
- `--shingle-size` (default: 3): Word n-gram size
- `--num-hashes` (default: 128): Number of hash functions (MinHash only)
- `--simhash-bits` (default: 64): Bit size for SimHash fingerprint
- `--limit` (optional): Maximum number of pairs to process
- `--output` (optional): CSV file for summary metrics
- `--predictions-output` (optional): CSV file for per-pair scores and predictions

**Real Metrics Output** *(5,000 Quora pairs)*:
```csv
method,threshold,pairs,seconds,precision,recall,f1,accuracy,true_positive,false_positive,true_negative,false_negative
jaccard,0.1,5000,0.082104,0.554348,0.426778,0.482270,0.649600,816,656,2432,1096
minhash,0.1,5000,1.276489,0.552650,0.408996,0.470093,0.647400,782,633,2455,1130
simhash,0.75,5000,0.540156,0.570937,0.433577,0.492866,0.658800,829,623,2465,1083
```

**Predictions Output Example:**
```csv
pair_id,method,score,prediction,label
0,simhash,0.687500,0,0
1,simhash,0.562500,0,0
2,simhash,0.718750,0,0
```

---

### Command 4: `preprocess` — Pre-compute Cleaned Dataset

Pre-process a raw dataset once and save cleaned columns for faster repeated experiments.

**Basic Usage:**
```bash
python -m plagiarism_engine.cli preprocess \
    --input data/raw/questions.csv \
    --output data/processed/questions_preprocessed.csv
```

**With All Parameters:**
```bash
python -m plagiarism_engine.cli preprocess \
    --input data/raw/questions.csv \
    --output data/processed/questions_preprocessed.csv \
    --text-col-a question1 \
    --text-col-b question2 \
    --label-col is_duplicate \
    --limit 5000 \
    --keep-stopwords
```

**Parameters:**
- `--input` (required): Path to raw CSV file
- `--output` (required): Path to save the preprocessed CSV
- `--text-col-a` (default: `question1`): Name of first text column
- `--text-col-b` (default: `question2`): Name of second text column
- `--label-col` (default: `is_duplicate`): Name of label column
- `--limit` (optional): Maximum rows to process
- `--keep-stopwords`: If set, stopwords are kept in the `_tokens` columns

**Output CSV Columns:**
- `question1`, `question2`: Original raw text (preserved)
- `question1_clean`, `question2_clean`: Normalized text — lowercase, no punctuation, **stopwords kept**
- `question1_tokens`, `question2_tokens`: Fully preprocessed tokens — stopwords removed, ready for shingling/hashing
- Other input columns (e.g. `id`, `qid1`, `qid2`, `is_duplicate`) are preserved as-is

**Output Example:**
```csv
id,qid1,qid2,question1,question2,question1_clean,question2_clean,question1_tokens,question2_tokens,is_duplicate
0,1,2,"What is the step by step guide to invest in share market in india?","What is the step by step guide to invest in share market?","what is the step by step guide to invest in share market in india","what is the step by step guide to invest in share market","what step step guide invest share market india","what step step guide invest share market",0
```

---

## Run Full Test Suite

Run all six evaluations at once (compare, corpus, and all three methods on the dataset):

```bash
bash run_tests.sh
```

This will:
1. Compare `doc_01.txt` vs `doc_02.txt` (near-duplicate pair)
2. Compare `doc_01.txt` vs `doc_05.txt` (unrelated pair — sanity check)
3. Scan the sample corpus for similar documents (brute force, threshold=0.15)
4. Evaluate 5,000 Quora pairs with Jaccard (threshold=0.10)
5. Evaluate 5,000 Quora pairs with MinHash (threshold=0.10)
6. Evaluate 5,000 Quora pairs with SimHash (threshold=0.75)

All results are saved to the `outputs/` directory.

---

## Project Structure

```
DMP3/
├── src/plagiarism_engine/
│   ├── __init__.py          # Public API
│   ├── preprocessing.py     # Normalize → tokenize → stopwords → shingles
│   ├── jaccard.py           # Exact Jaccard similarity
│   ├── minhash.py           # MinHash signatures + similarity estimate
│   ├── simhash.py           # TF-IDF weighted SimHash + Hamming distance
│   ├── lsh.py               # Band-based LSH candidate search
│   ├── dataset.py           # CSV and text corpus loaders
│   ├── evaluation.py        # Precision / Recall / F1 / Accuracy
│   └── cli.py               # Argparse CLI (4 subcommands)
├── data/
│   ├── raw/                 # Raw dataset (questions.csv — Quora pairs)
│   ├── processed/           # Preprocessed dataset (questions_preprocessed.csv)
│   └── sample_corpus/       # 5 sample text documents for demo runs
├── outputs/                 # All generated results (CSVs, JSONs, PNGs)
├── notebooks/
│   └── results.ipynb        # Pre-executed results dashboard (styled DataFrames + charts)
├── tests/
│   └── test_engine.py       # 7 unit and integration tests
├── run_tests.sh             # Full evaluation script
├── generate_report.py       # Generates outputs/report.docx (Persian technical report)
├── generate_results_notebook.py  # Rebuilds notebooks/results.ipynb
├── pyproject.toml
└── requirements.txt
```

---

## Running Tests

```bash
python -m pytest tests/ -v
```

Expected output: **7 passed**.

---

## Algorithm Details

### Jaccard Similarity
- **Formula**: `J(A, B) = |A ∩ B| / |A ∪ B|`
- **Complexity**: O(n) per pair where n = shingle set size
- **Limitation**: O(n²) pairs must be evaluated for a full corpus scan — use LSH to avoid this

### MinHash
- **Formula**: Estimates Jaccard via `h(x) = (ax + b) mod p` hash functions
- **Complexity**: O(k) per document where k = number of hash functions
- **LSH Acceleration**: Splits the signature into bands; only pairs sharing a band bucket are compared

### SimHash
- **Formula**: TF-IDF weighted 64-bit fingerprint; similarity = `1 − hamming_distance / 64`
- **Complexity**: O(m) per document where m = number of unique tokens
- **Advantage**: Token importance weighting helps detect paraphrased duplicates

### LSH (Locality-Sensitive Hashing)
- Splits MinHash signatures into `b` bands of `r` rows each (`b × r = num_hashes`)
- Collision probability: `P ≈ 1 − (1 − sim^r)^b`
- **Rule**: `num-hashes` must be divisible by `bands`
- **When to use**: Any corpus with more than a few hundred documents

---

## Performance Guide

| Dataset Size | Recommended Approach | Approximate Time |
|---|---|---|
| 2–10 documents | `compare` command | < 1 second |
| 10–100 documents | `corpus --no-use-lsh` | 1–5 seconds |
| 100–1,000 documents | `corpus --use-lsh` | 5–30 seconds |
| 1,000+ documents | `corpus --use-lsh` with tuned bands | seconds to minutes |

---

## Troubleshooting

### `"Signature length must be divisible by number of bands"`
Ensure `num-hashes % bands == 0`. With `--num-hashes 128` use bands: 2, 4, 8, 16, **32**, 64, or 128.

### `"Missing required CSV columns"`
Ensure your CSV contains the columns named in `--text-col-a` and `--text-col-b`.

### LSH not finding expected pairs
- The collision probability at low Jaccard (< 0.3) with 32 bands is very low
- Use `--no-use-lsh` to guarantee all pairs are compared
- Or increase `--bands` (e.g. 64) to raise sensitivity at lower similarity thresholds
