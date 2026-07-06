#!/usr/bin/env bash
# =============================================================================
# run_tests.sh  --  Full evaluation script for the Semantic Plagiarism Engine
# =============================================================================
# Usage:
#   bash run_tests.sh
#
# Requires the virtual environment to already be installed:
#   python3 -m venv .venv
#   pip install -r requirements.txt -e .
# =============================================================================
#
# Threshold notes (Quora duplicate-question dataset, first 5 000 pairs):
#   - jaccard / minhash : shingling works on lexical overlap; many semantic
#     duplicates share few exact n-grams, so a low threshold (0.1) is needed
#     to achieve reasonable recall.
#   - simhash           : TF-IDF weighted fingerprints capture broader token
#     similarity; a higher threshold (0.75) is needed to reduce false positives.
#
# Corpus note:
#   LSH band-collision probability at Jaccard ~0.3 with 32 bands / 4 rows is
#   only ~2 %, so --no-lsh (brute force) is used for the small sample corpus
#   to guarantee all pairs are evaluated.
# =============================================================================

set -euo pipefail

PYTHON=".venv/bin/python"
CLI="$PYTHON -m plagiarism_engine.cli"
OUTPUTS="outputs"

echo "============================================================"
echo "  Semantic Plagiarism Engine -- Test Run"
echo "============================================================"


# ------------------------------------------------------------
# 1. COMPARE  --  near-duplicate pair
# ------------------------------------------------------------
echo ""
echo "[1/5] Comparing doc_01.txt vs doc_02.txt (near-duplicate pair) ..."
$CLI compare \
    --file-a       data/sample_corpus/doc_01.txt \
    --file-b       data/sample_corpus/doc_02.txt \
    --shingle-size 3 \
    --num-hashes   128 \
    --simhash-bits 64 \
    --output       "$OUTPUTS/two_file_compare.json"
echo "      -> $OUTPUTS/two_file_compare.json"


# ------------------------------------------------------------
# 2. COMPARE  --  unrelated pair (sanity check)
# ------------------------------------------------------------
echo ""
echo "[2/5] Comparing doc_01.txt vs doc_05.txt (unrelated pair) ..."
$CLI compare \
    --file-a       data/sample_corpus/doc_01.txt \
    --file-b       data/sample_corpus/doc_05.txt \
    --shingle-size 3 \
    --num-hashes   128 \
    --simhash-bits 64 \
    --output       "$OUTPUTS/two_file_compare_unrelated.json"
echo "      -> $OUTPUTS/two_file_compare_unrelated.json"


# ------------------------------------------------------------
# 3. CORPUS  --  scan all 5 sample documents (brute force)
#    --no-lsh : guarantees all 10 pairs are scored regardless
#               of band-collision luck on a tiny corpus.
#    --threshold 0.15 : low enough to surface near-duplicates
#                       that share ~30 % exact shingles.
# ------------------------------------------------------------
echo ""
echo "[3/5] Scanning sample corpus (brute-force, threshold=0.15) ..."
$CLI corpus \
    --data         data/sample_corpus \
    --threshold    0.15 \
    --shingle-size 3 \
    --num-hashes   128 \
    --no-use-lsh \
    --output       "$OUTPUTS/candidates.csv"
echo "      -> $OUTPUTS/candidates.csv"


# ------------------------------------------------------------
# 4. PAIRS  --  Quora dataset, 5 000 pairs, all three methods
# ------------------------------------------------------------
PAIRS_CSV="data/raw/questions.csv"
LIMIT=5000

echo ""
echo "[4/5] Evaluating $LIMIT pairs -- jaccard (threshold=0.10) ..."
$CLI pairs \
    --pairs              "$PAIRS_CSV" \
    --text-col-a         question1 \
    --text-col-b         question2 \
    --label-col          is_duplicate \
    --limit              $LIMIT \
    --method             jaccard \
    --threshold          0.10 \
    --shingle-size       3 \
    --num-hashes         128 \
    --simhash-bits       64 \
    --output             "$OUTPUTS/metrics_jaccard.csv" \
    --predictions-output "$OUTPUTS/predictions_jaccard.csv"
echo "      -> $OUTPUTS/metrics_jaccard.csv"
echo "      -> $OUTPUTS/predictions_jaccard.csv"

echo ""
echo "[5/6] Evaluating $LIMIT pairs -- minhash (threshold=0.10) ..."
$CLI pairs \
    --pairs              "$PAIRS_CSV" \
    --text-col-a         question1 \
    --text-col-b         question2 \
    --label-col          is_duplicate \
    --limit              $LIMIT \
    --method             minhash \
    --threshold          0.10 \
    --shingle-size       3 \
    --num-hashes         128 \
    --simhash-bits       64 \
    --output             "$OUTPUTS/metrics_minhash.csv" \
    --predictions-output "$OUTPUTS/predictions_minhash.csv"
echo "      -> $OUTPUTS/metrics_minhash.csv"
echo "      -> $OUTPUTS/predictions_minhash.csv"

echo ""
echo "[6/6] Evaluating $LIMIT pairs -- simhash (threshold=0.75) ..."
$CLI pairs \
    --pairs              "$PAIRS_CSV" \
    --text-col-a         question1 \
    --text-col-b         question2 \
    --label-col          is_duplicate \
    --limit              $LIMIT \
    --method             simhash \
    --threshold          0.75 \
    --shingle-size       3 \
    --num-hashes         128 \
    --simhash-bits       64 \
    --output             "$OUTPUTS/metrics_simhash.csv" \
    --predictions-output "$OUTPUTS/predictions_simhash.csv"
echo "      -> $OUTPUTS/metrics_simhash.csv"
echo "      -> $OUTPUTS/predictions_simhash.csv"


# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------
echo ""
echo "============================================================"
echo "  All outputs written to: $OUTPUTS/"
echo "============================================================"
ls -lh "$OUTPUTS/"
echo ""
echo "Done."
