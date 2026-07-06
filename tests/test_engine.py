from pathlib import Path

from plagiarism_engine.cli import main
from plagiarism_engine.dataset import load_pair_csv, load_text_corpus
from plagiarism_engine.evaluation import binary_metrics
from plagiarism_engine.jaccard import jaccard_similarity
from plagiarism_engine.lsh import lsh_candidates
from plagiarism_engine.minhash import MinHasher, minhash_similarity
from plagiarism_engine.preprocessing import preprocess_document
from plagiarism_engine.simhash import (
    hamming_distance,
    inverse_document_frequencies,
    simhash,
    simhash_similarity,
)


def test_preprocessing_and_jaccard_similarity():
    doc_a = preprocess_document("The quick brown fox jumps over the dog", k=2)
    doc_b = preprocess_document("A quick brown fox jumps over a dog", k=2)

    assert doc_a
    assert doc_b
    assert jaccard_similarity(doc_a, doc_a) == 1.0
    assert jaccard_similarity(doc_a, doc_b) > 0.5


def test_minhash_similarity_tracks_jaccard():
    shingles_a = preprocess_document("alpha beta gamma delta epsilon", k=2)
    shingles_b = preprocess_document("alpha beta gamma delta zeta", k=2)
    shingles_c = preprocess_document("network database query index table", k=2)

    minhasher = MinHasher(num_hashes=64, seed=7)
    sig_a = minhasher.signature(shingles_a)
    sig_b = minhasher.signature(shingles_b)
    sig_c = minhasher.signature(shingles_c)

    assert len(sig_a) == 64
    assert minhash_similarity(sig_a, sig_a) == 1.0
    assert minhash_similarity(sig_a, sig_b) > minhash_similarity(sig_a, sig_c)


def test_lsh_returns_colliding_candidate_pair():
    minhasher = MinHasher(num_hashes=64, seed=11)
    signatures = {
        "a.txt": minhasher.signature(preprocess_document("alpha beta gamma delta", k=2)),
        "b.txt": minhasher.signature(preprocess_document("alpha beta gamma delta", k=2)),
        "c.txt": minhasher.signature(preprocess_document("one two three four", k=2)),
    }

    candidates = lsh_candidates(signatures, bands=16)

    assert ("a.txt", "b.txt") in candidates


def test_simhash_identical_text_has_zero_hamming_distance():
    tokens = ["alpha", "beta", "gamma", "alpha"]
    idf = inverse_document_frequencies([tokens])
    hash_a = simhash(tokens, idf=idf)
    hash_b = simhash(tokens, idf=idf)

    assert hamming_distance(hash_a, hash_b) == 0
    assert simhash_similarity(hash_a, hash_b) == 1.0


def test_dataset_helpers_load_text_and_pairs(tmp_path: Path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.txt").write_text("alpha beta", encoding="utf-8")
    (corpus / "ignored.csv").write_text("not text", encoding="utf-8")
    pairs = tmp_path / "pairs.csv"
    pairs.write_text(
        "question1,question2,is_duplicate\n"
        "alpha beta,alpha beta,1\n"
        "alpha beta,table index,0\n",
        encoding="utf-8",
    )

    documents = load_text_corpus(corpus)
    records = load_pair_csv(pairs, "question1", "question2", "is_duplicate")

    assert list(documents) == ["a.txt"]
    assert len(records) == 2
    assert records[0].label == 1


def test_binary_metrics():
    metrics = binary_metrics([1, 1, 0, 0], [1, 0, 1, 0])

    assert metrics.true_positive == 1
    assert metrics.false_positive == 1
    assert metrics.true_negative == 1
    assert metrics.false_negative == 1
    assert metrics.precision == 0.5
    assert metrics.recall == 0.5
    assert metrics.f1 == 0.5


def test_cli_compare_and_pairs_write_outputs(tmp_path: Path):
    file_a = tmp_path / "a.txt"
    file_b = tmp_path / "b.txt"
    compare_output = tmp_path / "compare.json"
    pairs_csv = tmp_path / "pairs.csv"
    metrics_output = tmp_path / "metrics.csv"

    file_a.write_text("alpha beta gamma delta", encoding="utf-8")
    file_b.write_text("alpha beta gamma zeta", encoding="utf-8")
    pairs_csv.write_text(
        "question1,question2,is_duplicate\n"
        "alpha beta gamma,alpha beta gamma,1\n"
        "alpha beta gamma,table query index,0\n",
        encoding="utf-8",
    )

    main(
        [
            "compare",
            "--file-a",
            str(file_a),
            "--file-b",
            str(file_b),
            "--output",
            str(compare_output),
            "--num-hashes",
            "64",
        ]
    )
    main(
        [
            "pairs",
            "--pairs",
            str(pairs_csv),
            "--text-col-a",
            "question1",
            "--text-col-b",
            "question2",
            "--label-col",
            "is_duplicate",
            "--method",
            "jaccard",
            "--threshold",
            "0.5",
            "--output",
            str(metrics_output),
        ]
    )

    assert compare_output.exists()
    assert metrics_output.exists()
    assert "precision" in metrics_output.read_text(encoding="utf-8")
