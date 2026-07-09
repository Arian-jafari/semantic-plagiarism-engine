# Semantic Plagiarism Engine

An advanced quantitative plagiarism detection engine built in Python. This system analyzes text corpora to catch duplicates and rewrites using three parallel detection approaches:
1. **Jaccard Similarity** - Exact lexical overlap detection
2. **MinHash** - Fast approximate Jaccard similarity
3. **SimHash** - TF-IDF weighted semantic similarity for paraphrasing detection

## Team Members
* **Arian Jafari**
* **Mehrnia Amouei**

## How It Works

### Three Detection Methods

#### 1. Jaccard Similarity (Exact Baseline)
- Measures exact word n-gram overlap between documents
- Fast for small-scale comparisons
- Threshold: 0.1-0.3 (lower = more permissive)

#### 2. MinHash (Fast Approximation)
- Creates fixed-length signatures (128-256 hash values) for each document
- Approximates Jaccard similarity without full comparison
- Can be optimized with **Locality-Sensitive Hashing (LSH)** for large corpora
- Threshold: 0.1-0.3 (same as Jaccard)

#### 3. SimHash (Semantic Detection)
- Creates TF-IDF weighted fingerprints (64-bit hash)
- Captures semantic meaning beyond exact word matches
- Better at detecting paraphrased plagiarism
- Threshold: 0.75-0.95 (higher = stricter)

## Development Setup

### 1. Create Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -e .
```

### 3. Verify Installation
```bash
python -m plagiarism_engine.cli --help
```

## CLI Commands

The engine provides 4 main commands via CLI:

### Command 1: `compare` - Compare Two Files

Compare two individual text files using all three methods.

**Basic Usage:**
```bash
python -m plagiarism_engine.cli compare \
    --file-a document1.txt \
    --file-b document2.txt
```

**With Custom Parameters:**
```bash
python -m plagiarism_engine.cli compare \
    --file-a document1.txt \
    --file-b document2.txt \
    --shingle-size 3 \
    --num-hashes 128 \
    --simhash-bits 64 \
    --output results/comparison.json
```

**Parameters:**
- `--file-a` (required): Path to first text file
- `--file-b` (required): Path to second text file
- `--shingle-size` (default: 3): Size of word n-grams for Jaccard/MinHash
- `--num-hashes` (default: 128): Number of hash functions for MinHash
- `--simhash-bits` (default: 64): Bit size for SimHash fingerprint
- `--output` (optional): JSON file to save results (prints to stdout if not provided)

**Output Example:**
```json
{
  "jaccard": 0.523,
  "minhash": 0.516,
  "simhash_similarity": 0.821,
  "simhash_hamming_distance": 11,
  "shingles_a": 45,
  "shingles_b": 52,
  "tokens_a": 89,
  "tokens_b": 101,
  "file_a": "document1.txt",
  "file_b": "document2.txt"
}
```

---

### Command 2: `corpus` - Scan Document Folder

Find similar document pairs in an entire folder using MinHash signatures.

**Basic Usage (Brute Force):**
```bash
python -m plagiarism_engine.cli corpus \
    --data ./documents \
    --no-use-lsh
```

**With LSH Optimization (Recommended for large datasets):**
```bash
python -m plagiarism_engine.cli corpus \
    --data ./documents \
    --use-lsh \
    --bands 32 \
    --threshold 0.25 \
    --output results/duplicates.csv
```

**With Custom Parameters:**
```bash
python -m plagiarism_engine.cli corpus \
    --data ./documents \
    --threshold 0.15 \
    --shingle-size 3 \
    --num-hashes 128 \
    --bands 32 \
    --use-lsh \
    --output results/candidates.csv
```

**Parameters:**
- `--data` (required): Path to folder containing text files
- `--threshold` (default: 0.25): Minimum Jaccard similarity to report
- `--shingle-size` (default: 3): Size of word n-grams
- `--num-hashes` (default: 128): Number of hash functions
- `--bands` (default: 32): Number of LSH bands (signature_length / bands must be integer)
- `--use-lsh` (default: true): Use LSH for speed, or `--no-use-lsh` for brute force
- `--output` (optional): CSV file to save results

**LSH vs Brute Force:**
- **Brute Force** (`--no-use-lsh`): Compares all pairs, guarantees all duplicates found, slow for large datasets
- **LSH** (`--use-lsh`): Only compares likely candidates, much faster, may miss some edge cases

**Output Example:**
```csv
doc_a,doc_b,jaccard,minhash
doc1.txt,doc2.txt,0.523,0.516
doc3.txt,doc5.txt,0.301,0.298
```

**Summary Output:**
```json
{
  "documents": 100,
  "total_pairs": 4950,
  "candidate_pairs": 245,
  "comparison_reduction": 0.950,
  "matches": 12,
  "seconds": 2.345
}
```

---

### Command 3: `pairs` - Evaluate Labeled Dataset

Evaluate plagiarism detection on a CSV dataset with labeled pairs.

#### A. Using Raw Data (On-the-fly Preprocessing)

**Jaccard Method (Threshold: 0.10):**
```bash
python -m plagiarism_engine.cli pairs \
    --pairs data/questions.csv \
    --text-col-a question1 \
    --text-col-b question2 \
    --label-col is_duplicate \
    --method jaccard \
    --threshold 0.10 \
    --limit 5000 \
    --output results/metrics_jaccard.csv \
    --predictions-output results/predictions_jaccard.csv
```

**MinHash Method (Threshold: 0.10):**
```bash
python -m plagiarism_engine.cli pairs \
    --pairs data/questions.csv \
    --text-col-a question1 \
    --text-col-b question2 \
    --label-col is_duplicate \
    --method minhash \
    --threshold 0.10 \
    --num-hashes 128 \
    --limit 5000 \
    --output results/metrics_minhash.csv \
    --predictions-output results/predictions_minhash.csv
```

**SimHash Method (Threshold: 0.75):**
```bash
python -m plagiarism_engine.cli pairs \
    --pairs data/questions.csv \
    --text-col-a question1 \
    --text-col-b question2 \
    --label-col is_duplicate \
    --method simhash \
    --threshold 0.75 \
    --simhash-bits 64 \
    --limit 5000 \
    --output results/metrics_simhash.csv \
    --predictions-output results/predictions_simhash.csv
```

#### B. Using Pre-processed Data (Faster for Repeated Experiments)

**Step 1: Pre-process Dataset Once**
```bash
python -m plagiarism_engine.cli preprocess \
    --input data/raw/questions.csv \
    --output data/processed/questions_processed.csv \
    --text-col-a question1 \
    --text-col-b question2 \
    --label-col is_duplicate \
    --limit 5000
```

**Step 2: Run Pairs Command on Processed Data**
```bash
python -m plagiarism_engine.cli pairs \
    --pairs data/processed/questions_processed.csv \
    --text-col-a question1 \
    --text-col-b question2 \
    --label-col is_duplicate \
    --method simhash \
    --threshold 0.75 \
    --limit 5000 \
    --output results/metrics_simhash.csv \
    --predictions-output results/predictions_simhash.csv
```

**Parameters for `pairs` Command:**
- `--pairs` (required): Path to CSV file with text pairs
- `--text-col-a` (required): Name of first text column in CSV
- `--text-col-b` (required): Name of second text column in CSV
- `--label-col` (optional): Name of label column (0=different, 1=duplicate)
- `--method` (default: simhash): Detection method - `jaccard`, `minhash`, or `simhash`
- `--threshold` (default: 0.75): Similarity threshold for prediction
- `--shingle-size` (default: 3): Word n-gram size
- `--num-hashes` (default: 128): Number of hash functions (for MinHash)
- `--simhash-bits` (default: 64): Bit size for SimHash
- `--limit` (optional): Maximum pairs to process
- `--output` (optional): CSV file for metrics (precision, recall, F1, accuracy)
- `--predictions-output` (optional): CSV file for detailed predictions

**Metrics Output Example:**
```csv
method,threshold,pairs,seconds,precision,recall,f1,accuracy,true_positive,false_positive,true_negative,false_negative
simhash,0.75,5000,12.345,0.876,0.834,0.854,0.855,417,61,4082,440
```

**Predictions Output Example:**
```csv
pair_id,method,score,prediction,label
0,simhash,0.821354,1,1
1,simhash,0.234567,0,0
2,simhash,0.654321,1,0
```

---

### Command 4: `preprocess` - Pre-compute Cleaned Dataset

Pre-process a raw dataset to create cleaned and tokenized versions for reusable experiments.

**Basic Usage:**
```bash
python -m plagiarism_engine.cli preprocess \
    --input data/raw/questions.csv \
    --output data/processed/questions_processed.csv
```

**With Custom Parameters:**
```bash
python -m plagiarism_engine.cli preprocess \
    --input data/raw/questions.csv \
    --output data/processed/questions_processed.csv \
    --text-col-a question1 \
    --text-col-b question2 \
    --label-col is_duplicate \
    --limit 5000 \
    --keep-stopwords
```

**Parameters:**
- `--input` (required): Path to raw CSV file
- `--output` (required): Path to save processed CSV
- `--text-col-a` (default: question1): Name of first text column
- `--text-col-b` (default: question2): Name of second text column
- `--label-col` (default: is_duplicate): Name of label column
- `--limit` (optional): Maximum rows to process
- `--keep-stopwords` (default: false): If set, keeps stopwords in token columns

**Output CSV Structure:**
```csv
id,qid1,qid2,question1,question2,question1_clean,question2_clean,question1_tokens,question2_tokens,is_duplicate
1,1,2,"What is Python?","What is Python programming?","what is python","what is python programming","python programming","python programming",1
```

**Output Columns:**
- `question1`, `question2`: Original raw text
- `question1_clean`, `question2_clean`: Normalized (lowercase, no punctuation, extra spaces removed)
- `question1_tokens`, `question2_tokens`: Tokenized (space-separated cleaned tokens)
- Other columns: Preserved from input

---

## Usage Workflows

### Workflow 1: Quick Comparison (Two Files)
```bash
python -m plagiarism_engine.cli compare \
    --file-a doc1.txt \
    --file-b doc2.txt \
    --output results/comparison.json
```

### Workflow 2: Find Duplicates in Folder (Small Dataset)
```bash
python -m plagiarism_engine.cli corpus \
    --data ./my_documents \
    --threshold 0.15 \
    --no-use-lsh \
    --output results/duplicates.csv
```

### Workflow 3: Find Duplicates in Folder (Large Dataset with LSH)
```bash
python -m plagiarism_engine.cli corpus \
    --data ./my_documents \
    --threshold 0.15 \
    --bands 32 \
    --use-lsh \
    --output results/duplicates_optimized.csv
```

### Workflow 4: Evaluate on Labeled Data (Single Method)
```bash
python -m plagiarism_engine.cli pairs \
    --pairs data/labeled_pairs.csv \
    --text-col-a text1 \
    --text-col-b text2 \
    --label-col label \
    --method simhash \
    --threshold 0.75 \
    --output results/metrics.csv \
    --predictions-output results/predictions.csv
```

### Workflow 5: Compare All Three Methods
```bash
# Jaccard
python -m plagiarism_engine.cli pairs \
    --pairs data/labeled.csv \
    --text-col-a text1 \
    --text-col-b text2 \
    --label-col label \
    --method jaccard \
    --threshold 0.10 \
    --output results/metrics_jaccard.csv

# MinHash
python -m plagiarism_engine.cli pairs \
    --pairs data/labeled.csv \
    --text-col-a text1 \
    --text-col-b text2 \
    --label-col label \
    --method minhash \
    --threshold 0.10 \
    --output results/metrics_minhash.csv

# SimHash
python -m plagiarism_engine.cli pairs \
    --pairs data/labeled.csv \
    --text-col-a text1 \
    --text-col-b text2 \
    --label-col label \
    --method simhash \
    --threshold 0.75 \
    --output results/metrics_simhash.csv
```

### Workflow 6: Pre-process Once, Experiment Multiple Times
```bash
# Step 1: Pre-process dataset (one time)
python -m plagiarism_engine.cli preprocess \
    --input data/raw/questions.csv \
    --output data/processed/questions.csv \
    --limit 10000

# Step 2: Run experiments on processed data with different thresholds
python -m plagiarism_engine.cli pairs \
    --pairs data/processed/questions.csv \
    --text-col-a question1 \
    --text-col-b question2 \
    --label-col is_duplicate \
    --method simhash \
    --threshold 0.70 \
    --output results/metrics_t70.csv

python -m plagiarism_engine.cli pairs \
    --pairs data/processed/questions.csv \
    --text-col-a question1 \
    --text-col-b question2 \
    --label-col is_duplicate \
    --method simhash \
    --threshold 0.75 \
    --output results/metrics_t75.csv

python -m plagiarism_engine.cli pairs \
    --pairs data/processed/questions.csv \
    --text-col-a question1 \
    --text-col-b question2 \
    --label-col is_duplicate \
    --method simhash \
    --threshold 0.80 \
    --output results/metrics_t80.csv
```

---

## Run Full Test Suite

To run all tests and comparisons:

```bash
bash run_tests.sh
```

This will:
1. Compare two similar documents
2. Compare two unrelated documents
3. Scan sample corpus for duplicates
4. Evaluate on 5,000 Quora question pairs using all three methods
5. Generate detailed metrics and predictions

Results are saved in `outputs/` directory.

---

## Algorithm Details

### Jaccard Similarity
- **Formula**: |A ∩ B| / |A ∪ B|
- **Time**: O(n) where n = number of shingles
- **Use Case**: Exact duplicate detection, fast baseline
- **Recommended Threshold**: 0.10-0.30

### MinHash
- **Formula**: Estimates Jaccard via signature matching
- **Time**: O(k) where k = number of hash functions (typically 128)
- **Use Case**: Fast similarity estimation, can be optimized with LSH
- **Recommended Threshold**: 0.10-0.30
- **LSH Optimization**: Groups documents into bands, only compares within bands

### SimHash
- **Formula**: TF-IDF weighted 64-bit fingerprint, similarity via Hamming distance
- **Time**: O(m) where m = number of unique tokens
- **Use Case**: Semantic similarity, catches paraphrases
- **Recommended Threshold**: 0.75-0.95

### LSH (Locality-Sensitive Hashing)
- **Bands**: Signature split into bands, each band hashed
- **Probability**: `P(collision in ≥1 band) ≈ 1 - (1 - sim^rows)^bands`
- **Trade-off**: More bands → faster but fewer matches; fewer bands → slower but more matches
- **When to use**: Datasets with >1000 documents

---

## CSV Format Requirements

### Input CSV for `pairs` Command
```csv
question1,question2,is_duplicate
"What is Python?","What is Python programming?",1
"How to learn?","Best books for learning?",0
```

**Required Columns:** At least two text columns (column names specified via `--text-col-a` and `--text-col-b`)

**Optional Columns:** Label column for evaluation (specified via `--label-col`)

### Input CSV for `preprocess` Command
Same format as `pairs` input.

---

## Performance Guide

| Dataset Size | Recommended Approach | Time Estimate |
|--------------|---------------------|---------------|
| 2-10 documents | `compare` command | <1 second |
| 10-100 documents | `corpus` with `--no-use-lsh` | 1-5 seconds |
| 100-1000 documents | `corpus` with `--use-lsh` | 5-30 seconds |
| 1000+ documents | `corpus` with `--use-lsh` + high bands | 30 seconds - minutes |

---

## Troubleshooting

### Error: "Signature length must be divisible by number of bands"
**Solution**: Ensure `num_hashes % bands == 0`. With `--num-hashes 128`, use bands like 2, 4, 8, 16, 32, 64, 128.

### Error: "Missing required CSV columns"
**Solution**: Ensure your CSV has the columns specified in `--text-col-a` and `--text-col-b`.

### LSH not finding expected pairs
**Possible Causes:**
- Pairs below LSH threshold
- Threshold set too high
- Signatures too different for band collisions
- **Solution**: Use `--no-use-lsh` to guarantee all pairs are compared

---



<!-- # Semantic Plagiarism Engine

An advanced quantitative plagiarism detection engine built in Python. This system analyzes text corpora to catch duplicates and rewrites using two parallel engineering approaches:
1. **Shingling & MinHash (Jaccard Similarity)**
2. **Semantic / Vector-Based Alignment**

## Team Members
* **Arian Jafari**
* **Mehrnia Amouei**

## Development Setup
To configure the isolated local environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e . -->