# Semantic Plagiarism Engine

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
pip install -e .