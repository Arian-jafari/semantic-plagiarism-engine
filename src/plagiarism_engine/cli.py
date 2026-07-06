"""
Command line interface for the plagiarism engine.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

from .dataset import (
    load_pair_csv,
    load_text_corpus,
    preprocessed_text,
    read_text_file,
    token_pipeline,
)
from .evaluation import binary_metrics
from .jaccard import jaccard_similarity
from .lsh import all_pairs, lsh_candidates
from .minhash import MinHasher, minhash_similarity
from .preprocessing import preprocess_document
from .simhash import (
    hamming_distance,
    inverse_document_frequencies,
    simhash,
    simhash_similarity,
)


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _write_csv(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _print_or_write_json(payload: dict[str, Any], output: str | None) -> None:
    if output:
        _write_json(output, payload)
    else:
        print(json.dumps(payload, indent=2, ensure_ascii=False))


def compare_texts(
    text_a: str,
    text_b: str,
    shingle_size: int = 3,
    num_hashes: int = 128,
    simhash_bits: int = 64,
) -> dict[str, Any]:
    """
    Compare two raw texts with exact Jaccard, MinHash, and SimHash.
    """

    shingles_a = preprocess_document(text_a, k=shingle_size)
    shingles_b = preprocess_document(text_b, k=shingle_size)

    minhasher = MinHasher(num_hashes=num_hashes)
    signature_a = minhasher.signature(shingles_a)
    signature_b = minhasher.signature(shingles_b)

    tokens_a = token_pipeline(text_a)
    tokens_b = token_pipeline(text_b)
    idf = inverse_document_frequencies([tokens_a, tokens_b])
    simhash_a = simhash(tokens_a, idf=idf, bits=simhash_bits)
    simhash_b = simhash(tokens_b, idf=idf, bits=simhash_bits)
    sim_similarity = simhash_similarity(simhash_a, simhash_b, bits=simhash_bits)

    return {
        "jaccard": jaccard_similarity(shingles_a, shingles_b),
        "minhash": minhash_similarity(signature_a, signature_b),
        "simhash_similarity": sim_similarity,
        "simhash_hamming_distance": hamming_distance(simhash_a, simhash_b),
        "shingles_a": len(shingles_a),
        "shingles_b": len(shingles_b),
        "tokens_a": len(tokens_a),
        "tokens_b": len(tokens_b),
    }


def handle_compare(args: argparse.Namespace) -> None:
    text_a = read_text_file(args.file_a)
    text_b = read_text_file(args.file_b)
    payload = compare_texts(
        text_a,
        text_b,
        shingle_size=args.shingle_size,
        num_hashes=args.num_hashes,
        simhash_bits=args.simhash_bits,
    )
    payload.update({"file_a": args.file_a, "file_b": args.file_b})
    _print_or_write_json(payload, args.output)


def handle_corpus(args: argparse.Namespace) -> None:
    documents = load_text_corpus(args.data)
    start_time = time.perf_counter()

    minhasher = MinHasher(num_hashes=args.num_hashes)
    shingles = {
        doc_id: preprocess_document(text, k=args.shingle_size)
        for doc_id, text in documents.items()
    }
    signatures = {
        doc_id: minhasher.signature(doc_shingles)
        for doc_id, doc_shingles in shingles.items()
    }

    if args.use_lsh:
        candidates = lsh_candidates(signatures, bands=args.bands)
    else:
        candidates = all_pairs(documents)

    rows: list[dict[str, Any]] = []
    for doc_a, doc_b in sorted(candidates):
        exact = jaccard_similarity(shingles[doc_a], shingles[doc_b])
        estimate = minhash_similarity(signatures[doc_a], signatures[doc_b])
        if exact >= args.threshold:
            rows.append(
                {
                    "doc_a": doc_a,
                    "doc_b": doc_b,
                    "jaccard": f"{exact:.6f}",
                    "minhash": f"{estimate:.6f}",
                }
            )

    elapsed = time.perf_counter() - start_time
    total_pairs = len(all_pairs(documents))
    compared_pairs = len(candidates)
    reduction = 1.0 - (compared_pairs / total_pairs) if total_pairs else 0.0

    fieldnames = ["doc_a", "doc_b", "jaccard", "minhash"]
    if args.output:
        _write_csv(args.output, rows, fieldnames)
    else:
        for row in rows:
            print(row)

    summary = {
        "documents": len(documents),
        "total_pairs": total_pairs,
        "candidate_pairs": compared_pairs,
        "comparison_reduction": round(reduction, 6),
        "matches": len(rows),
        "seconds": round(elapsed, 6),
    }
    print(json.dumps(summary, indent=2))


def _pair_scores(
    text_a: str,
    text_b: str,
    method: str,
    shingle_size: int,
    simhash_bits: int,
    minhasher: MinHasher | None = None,
    idf: dict[str, float] | None = None,
) -> float:
    if method == "jaccard":
        return jaccard_similarity(
            preprocess_document(text_a, k=shingle_size),
            preprocess_document(text_b, k=shingle_size),
        )

    if method == "minhash":
        if minhasher is None:
            raise ValueError("minhasher is required for minhash scoring.")
        signature_a = minhasher.signature(preprocess_document(text_a, k=shingle_size))
        signature_b = minhasher.signature(preprocess_document(text_b, k=shingle_size))
        return minhash_similarity(signature_a, signature_b)

    if method == "simhash":
        tokens_a = token_pipeline(text_a)
        tokens_b = token_pipeline(text_b)
        hash_a = simhash(tokens_a, idf=idf, bits=simhash_bits)
        hash_b = simhash(tokens_b, idf=idf, bits=simhash_bits)
        return simhash_similarity(hash_a, hash_b, bits=simhash_bits)

    raise ValueError(f"Unsupported method: {method}")


def handle_pairs(args: argparse.Namespace) -> None:
    records = load_pair_csv(
        args.pairs,
        text_col_a=args.text_col_a,
        text_col_b=args.text_col_b,
        label_col=args.label_col,
        limit=args.limit,
    )

    start_time = time.perf_counter()
    rows: list[dict[str, Any]] = []
    labels: list[int] = []
    predictions: list[int] = []
    minhasher = MinHasher(num_hashes=args.num_hashes) if args.method == "minhash" else None
    idf = None
    if args.method == "simhash":
        token_documents = []
        for record in records:
            token_documents.append(token_pipeline(record.text_a))
            token_documents.append(token_pipeline(record.text_b))
        idf = inverse_document_frequencies(token_documents)

    for index, record in enumerate(records):
        score = _pair_scores(
            record.text_a,
            record.text_b,
            method=args.method,
            shingle_size=args.shingle_size,
            simhash_bits=args.simhash_bits,
            minhasher=minhasher,
            idf=idf,
        )
        prediction = 1 if score >= args.threshold else 0
        row: dict[str, Any] = {
            "pair_id": record.pair_id or str(index),
            "method": args.method,
            "score": f"{score:.6f}",
            "prediction": prediction,
        }

        if record.label is not None:
            labels.append(record.label)
            predictions.append(prediction)
            row["label"] = record.label

        rows.append(row)

    elapsed = time.perf_counter() - start_time
    metrics = binary_metrics(labels, predictions) if labels else None

    metric_row = {
        "method": args.method,
        "threshold": args.threshold,
        "pairs": len(records),
        "seconds": f"{elapsed:.6f}",
        "precision": f"{metrics.precision:.6f}" if metrics else "",
        "recall": f"{metrics.recall:.6f}" if metrics else "",
        "f1": f"{metrics.f1:.6f}" if metrics else "",
        "accuracy": f"{metrics.accuracy:.6f}" if metrics else "",
        "true_positive": metrics.true_positive if metrics else "",
        "false_positive": metrics.false_positive if metrics else "",
        "true_negative": metrics.true_negative if metrics else "",
        "false_negative": metrics.false_negative if metrics else "",
    }

    fieldnames = list(metric_row.keys())
    if args.output:
        _write_csv(args.output, [metric_row], fieldnames)
    else:
        print(json.dumps(metric_row, indent=2))

    if args.predictions_output:
        prediction_fields = ["pair_id", "method", "score", "prediction"]
        if labels:
            prediction_fields.append("label")
        _write_csv(args.predictions_output, rows, prediction_fields)


def handle_preprocess(args: argparse.Namespace) -> None:
    """
    Preprocess a pair CSV and save a cleaned CSV for repeatable experiments.
    """

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    start_time = time.perf_counter()
    rows_written = 0

    with input_path.open("r", encoding="utf-8", errors="replace", newline="") as source:
        reader = csv.DictReader(source)
        missing = {args.text_col_a, args.text_col_b} - set(reader.fieldnames or [])
        if args.label_col:
            missing = missing | ({args.label_col} - set(reader.fieldnames or []))
        if missing:
            missing_columns = ", ".join(sorted(missing))
            raise ValueError(f"Missing required CSV columns: {missing_columns}")

        base_fields = [field for field in ["id", "qid1", "qid2"] if field in (reader.fieldnames or [])]
        fieldnames = [
            *base_fields,
            args.text_col_a,
            args.text_col_b,
            f"{args.text_col_a}_clean",
            f"{args.text_col_b}_clean",
            f"{args.text_col_a}_tokens",
            f"{args.text_col_b}_tokens",
        ]
        if args.label_col:
            fieldnames.append(args.label_col)

        with output_path.open("w", encoding="utf-8", newline="") as target:
            writer = csv.DictWriter(target, fieldnames=fieldnames)
            writer.writeheader()

            for row in reader:
                clean_a = preprocessed_text(
                    row.get(args.text_col_a, ""),
                    remove_stops=not args.keep_stopwords,
                )
                clean_b = preprocessed_text(
                    row.get(args.text_col_b, ""),
                    remove_stops=not args.keep_stopwords,
                )
                output_row: dict[str, Any] = {
                    field: row.get(field, "")
                    for field in base_fields
                }
                output_row.update(
                    {
                        args.text_col_a: row.get(args.text_col_a, ""),
                        args.text_col_b: row.get(args.text_col_b, ""),
                        f"{args.text_col_a}_clean": clean_a,
                        f"{args.text_col_b}_clean": clean_b,
                        f"{args.text_col_a}_tokens": clean_a,
                        f"{args.text_col_b}_tokens": clean_b,
                    }
                )
                if args.label_col:
                    output_row[args.label_col] = row.get(args.label_col, "")

                writer.writerow(output_row)
                rows_written += 1

                if args.limit is not None and rows_written >= args.limit:
                    break

    elapsed = time.perf_counter() - start_time
    summary = {
        "input": str(input_path),
        "output": str(output_path),
        "rows": rows_written,
        "seconds": round(elapsed, 6),
    }
    print(json.dumps(summary, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="plagiarism_engine",
        description="Semantic duplicate and near-plagiarism detection CLI.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    compare = subparsers.add_parser("compare", help="Compare two text files.")
    compare.add_argument("--file-a", required=True)
    compare.add_argument("--file-b", required=True)
    compare.add_argument("--output")
    compare.add_argument("--shingle-size", type=int, default=3)
    compare.add_argument("--num-hashes", type=int, default=128)
    compare.add_argument("--simhash-bits", type=int, default=64)
    compare.set_defaults(func=handle_compare)

    corpus = subparsers.add_parser("corpus", help="Search a folder for similar documents.")
    corpus.add_argument("--data", required=True)
    corpus.add_argument("--threshold", type=float, default=0.25)
    corpus.add_argument("--shingle-size", type=int, default=3)
    corpus.add_argument("--num-hashes", type=int, default=128)
    corpus.add_argument("--bands", type=int, default=32)
    corpus.add_argument("--output")
    corpus.add_argument("--use-lsh", action=argparse.BooleanOptionalAction, default=True)
    corpus.set_defaults(func=handle_corpus)

    pairs = subparsers.add_parser("pairs", help="Evaluate on a labeled pair CSV.")
    pairs.add_argument("--pairs", required=True)
    pairs.add_argument("--text-col-a", required=True)
    pairs.add_argument("--text-col-b", required=True)
    pairs.add_argument("--label-col")
    pairs.add_argument("--limit", type=int)
    pairs.add_argument("--method", choices=["jaccard", "minhash", "simhash"], default="simhash")
    pairs.add_argument("--threshold", type=float, default=0.75)
    pairs.add_argument("--shingle-size", type=int, default=3)
    pairs.add_argument("--num-hashes", type=int, default=128)
    pairs.add_argument("--simhash-bits", type=int, default=64)
    pairs.add_argument("--output")
    pairs.add_argument("--predictions-output")
    pairs.set_defaults(func=handle_pairs)

    preprocess = subparsers.add_parser(
        "preprocess",
        help="Preprocess a pair CSV and save cleaned text columns.",
    )
    preprocess.add_argument("--input", required=True)
    preprocess.add_argument("--output", required=True)
    preprocess.add_argument("--text-col-a", default="question1")
    preprocess.add_argument("--text-col-b", default="question2")
    preprocess.add_argument("--label-col", default="is_duplicate")
    preprocess.add_argument("--limit", type=int)
    preprocess.add_argument("--keep-stopwords", action="store_true")
    preprocess.set_defaults(func=handle_preprocess)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
