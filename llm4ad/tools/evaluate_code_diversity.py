from __future__ import annotations

import argparse
import ast
import io
import json
import math
import tempfile
import tokenize
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import squareform

from llm4ad.tools.report_metrics import load_final_records

DEFAULT_EMBEDDING_MODEL = "Salesforce/codet5p-110m-embedding"


class _DocstringRemover(ast.NodeTransformer):
    def _strip_docstring(self, node):
        self.generic_visit(node)
        if (
            getattr(node, "body", None)
            and isinstance(node.body[0], ast.Expr)
            and isinstance(getattr(node.body[0], "value", None), ast.Constant)
            and isinstance(node.body[0].value.value, str)
        ):
            node.body = node.body[1:]
        return node

    def visit_Module(self, node):
        return self._strip_docstring(node)

    def visit_FunctionDef(self, node):
        return self._strip_docstring(node)

    def visit_AsyncFunctionDef(self, node):
        return self._strip_docstring(node)

    def visit_ClassDef(self, node):
        return self._strip_docstring(node)


def canonicalize_code(code: str) -> str:
    code = _remove_comments_and_docstrings(code)
    code = _format_python_code(code)
    try:
        tree = ast.parse(code)
        tree = _DocstringRemover().visit(tree)
        ast.fix_missing_locations(tree)
        return ast.unparse(tree)
    except SyntaxError:
        return _strip_comments_fallback(code)


def _strip_comments_fallback(code: str) -> str:
    kept = []
    try:
        stream = io.StringIO(code).readline
        for token in tokenize.generate_tokens(stream):
            if token.type in {tokenize.COMMENT, tokenize.ENCODING, tokenize.NL, tokenize.NEWLINE}:
                continue
            kept.append(token.string)
    except tokenize.TokenError:
        return code
    return " ".join(kept)


def _remove_comments_and_docstrings(code: str) -> str:
    try:
        import tree_sitter_python as tspython
        from tree_sitter import Language, Parser

        language = Language(tspython.language())
        try:
            parser = Parser(language)
        except TypeError:
            parser = Parser()
            parser.set_language(language)
        tree = parser.parse(code.encode("utf-8"))
        lines = code.splitlines()
        removals = []

        def visit(node):
            if node.type == "comment":
                removals.append((node.start_point, node.end_point))
            elif node.type == "string" and node.parent and node.parent.type in {"expression_statement", "module"}:
                removals.append((node.start_point, node.end_point))
            for child in node.children:
                visit(child)

        visit(tree.root_node)
        for start, end in reversed(removals):
            start_row, start_col = start
            end_row, end_col = end
            if start_row >= len(lines):
                continue
            if start_row == end_row:
                lines[start_row] = lines[start_row][:start_col] + lines[start_row][end_col:]
            else:
                lines[start_row] = lines[start_row][:start_col]
                for row in range(start_row + 1, min(end_row, len(lines))):
                    lines[row] = ""
                if end_row < len(lines):
                    lines[end_row] = lines[end_row][end_col:]
        return "\n".join(line for line in lines if line.strip())
    except Exception:
        try:
            tree = ast.parse(code)
            tree = _DocstringRemover().visit(tree)
            ast.fix_missing_locations(tree)
            return ast.unparse(tree)
        except SyntaxError:
            return _strip_comments_fallback(code)


def _format_python_code(code: str) -> str:
    try:
        import autopep8

        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".py", encoding="utf-8") as temp_file:
            temp_file.write(code)
            temp_filename = temp_file.name
        try:
            args = ["--in-place", "--aggressive", "--aggressive", temp_filename]
            autopep8.fix_file(temp_filename, options=autopep8.parse_args(args))
            with open(temp_filename, "r", encoding="utf-8") as file:
                return file.read()
        finally:
            Path(temp_filename).unlink(missing_ok=True)
    except Exception:
        return code


def _code_tokens(code: str) -> list[str]:
    tokens = []
    try:
        stream = io.StringIO(code).readline
        for token in tokenize.generate_tokens(stream):
            if token.type in {
                tokenize.COMMENT,
                tokenize.ENCODING,
                tokenize.NL,
                tokenize.NEWLINE,
                tokenize.INDENT,
                tokenize.DEDENT,
                tokenize.ENDMARKER,
            }:
                continue
            tokens.append(token.string)
    except tokenize.TokenError:
        tokens = code.split()
    return tokens


def _token_features(code: str) -> Counter:
    tokens = _code_tokens(canonicalize_code(code))
    features = Counter(tokens)
    features.update(f"{left} {right}" for left, right in zip(tokens, tokens[1:]))
    return features


def _tfidf_vectors(codes: list[str]) -> np.ndarray:
    counters = [_token_features(code) for code in codes]
    vocab = sorted({feature for counter in counters for feature in counter})
    if not vocab:
        return np.zeros((len(codes), 0), dtype=float)
    index = {feature: idx for idx, feature in enumerate(vocab)}
    df = np.zeros(len(vocab), dtype=float)
    for counter in counters:
        for feature in counter:
            df[index[feature]] += 1.0
    idf = np.log((1.0 + len(counters)) / (1.0 + df)) + 1.0

    vectors = np.zeros((len(codes), len(vocab)), dtype=float)
    for row, counter in enumerate(counters):
        total = float(sum(counter.values())) or 1.0
        for feature, count in counter.items():
            vectors[row, index[feature]] = (count / total) * idf[index[feature]]
    norms = np.linalg.norm(vectors, axis=1)
    nonzero = norms > 1e-12
    vectors[nonzero] = vectors[nonzero] / norms[nonzero, None]
    return vectors


def _embedding_vectors(
    codes: list[str],
    *,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    device: str = "auto",
) -> np.ndarray:
    if not codes:
        return np.zeros((0, 0), dtype=float)
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Embedding mode requires torch and transformers in the selected Python environment."
        ) from exc

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
    model.eval()

    vectors = []
    with torch.no_grad():
        for code in codes:
            processed = canonicalize_code(code)
            inputs = tokenizer(
                processed,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}
            input_ids = inputs.get("input_ids")
            try:
                output = model(input_ids)[0]
            except Exception:
                output = model(**inputs)[0]
            embedding = output.detach().cpu().numpy()
            if embedding.ndim == 3:
                embedding = embedding.mean(axis=1)
            vectors.append(embedding.reshape(-1))

    max_dim = max(len(vector) for vector in vectors)
    arr = np.zeros((len(vectors), max_dim), dtype=float)
    for idx, vector in enumerate(vectors):
        arr[idx, : len(vector)] = vector
    return arr


def _cosine_similarity_matrix(vectors: np.ndarray) -> np.ndarray:
    if vectors.size == 0:
        return np.zeros((len(vectors), len(vectors)), dtype=float)
    norms = np.linalg.norm(vectors, axis=1)
    normalized = np.zeros_like(vectors, dtype=float)
    nonzero = norms > 1e-12
    normalized[nonzero] = vectors[nonzero] / norms[nonzero, None]
    similarity = normalized @ normalized.T
    np.fill_diagonal(similarity, 1.0)
    return np.clip(similarity, -1.0, 1.0)


def _cluster_vectors(vectors: np.ndarray, alpha: float) -> list[list[int]]:
    if len(vectors) == 0:
        return []
    if len(vectors) == 1:
        return [[0]]
    similarity = _cosine_similarity_matrix(vectors)
    distance = np.clip(1.0 - similarity, 0.0, None)
    np.fill_diagonal(distance, 0.0)
    condensed = squareform(distance, checks=False)
    tree = linkage(condensed, method="complete")
    labels = fcluster(tree, t=1.0 - alpha, criterion="distance")
    clusters_by_label: dict[int, list[int]] = {}
    for idx, label in enumerate(labels):
        clusters_by_label.setdefault(int(label), []).append(idx)
    return list(clusters_by_label.values())


def _shannon_entropy(probabilities: np.ndarray) -> float:
    probabilities = probabilities[probabilities > 0.0]
    if probabilities.size == 0:
        return 0.0
    return float(-np.sum(probabilities * np.log(probabilities)))


def _swdi(clusters: list[list[int]], total: int) -> float:
    if total <= 0:
        return 0.0
    probabilities = np.asarray([len(cluster) / total for cluster in clusters], dtype=float)
    return _shannon_entropy(probabilities)


def _mst_edge_distances(vectors: np.ndarray) -> list[float]:
    n = len(vectors)
    if n <= 1:
        return []
    distance_matrix = np.zeros((n, n), dtype=float)
    for i in range(n):
        distance_matrix[i] = np.linalg.norm(vectors - vectors[i], axis=1)
    mst = minimum_spanning_tree(distance_matrix).toarray()
    return [float(value) for value in mst[mst != 0.0]]


def _cdi(vectors: np.ndarray) -> float:
    edges = np.asarray(_mst_edge_distances(vectors), dtype=float)
    total = float(np.sum(edges))
    if total <= 1e-12:
        return 0.0
    return _shannon_entropy(edges / total)


def _mean_pairwise_cosine(vectors: np.ndarray) -> float | None:
    n = len(vectors)
    if n <= 1:
        return None
    sims = _cosine_similarity_matrix(vectors)
    upper = sims[np.triu_indices(n, k=1)]
    return float(np.mean(upper)) if upper.size else None


def evaluate_code_diversity(
    records: list[dict[str, Any]],
    *,
    alpha: float = 0.95,
    method: str | None = None,
    problem: str | None = None,
    log_dir: str | Path | None = None,
    encoding: str = "codet5p",
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    device: str = "auto",
) -> dict[str, Any]:
    codes = [
        record.get("function")
        for record in records
        if isinstance(record.get("function"), str) and record.get("function").strip()
    ]
    if encoding == "tfidf":
        vectors = _tfidf_vectors(codes)
        encoding_name = "ast_autopep8_token_tfidf_unigram_bigram"
        embedding_model = None
    elif encoding == "codet5p":
        vectors = _embedding_vectors(codes, model_name=model_name, device=device)
        encoding_name = "tree_sitter_autopep8_codet5p_embedding"
        embedding_model = model_name
    else:
        raise ValueError(f"Unsupported encoding={encoding!r}")
    clusters = _cluster_vectors(vectors, alpha) if len(codes) else []
    report = {
        "method": method,
        "problem": problem,
        "metric": "code_population_diversity",
        "encoding": encoding_name,
        "embedding_model": embedding_model,
        "alpha": float(alpha),
        "log_dir": str(log_dir) if log_dir is not None else None,
        "num_records": len(records),
        "num_encoded": len(codes),
        "num_clusters": len(clusters),
        "cluster_sizes": [len(cluster) for cluster in clusters],
        "swdi": _swdi(clusters, len(codes)),
        "cdi": _cdi(vectors),
        "mean_pairwise_cosine": _mean_pairwise_cosine(vectors),
    }
    return report


def write_code_diversity_report(report: dict[str, Any]) -> None:
    log_dir = report.get("log_dir")
    if not log_dir:
        return
    log_path = Path(log_dir)
    json_path = log_path / "code_diversity_report.json"
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    md_path = log_path / "code_diversity_report.md"
    with md_path.open("w", encoding="utf-8") as file:
        file.write("# Code Diversity Report\n\n")
        for key, value in report.items():
            file.write(f"- `{key}`: {value}\n")


def format_report(report: dict[str, Any]) -> str:
    return (
        "| method | problem | encoded | clusters | SWDI | CDI | mean cosine |\n"
        "| --- | --- | --- | --- | --- | --- | --- |\n"
        "| {method} | {problem} | {encoded} | {clusters} | {swdi:.6g} | {cdi:.6g} | {cosine} |"
    ).format(
        method=report.get("method") or "",
        problem=report.get("problem") or "",
        encoded=report.get("num_encoded"),
        clusters=report.get("num_clusters"),
        swdi=float(report.get("swdi") or 0.0),
        cdi=float(report.get("cdi") or 0.0),
        cosine="" if report.get("mean_pairwise_cosine") is None else f"{report['mean_pairwise_cosine']:.6g}",
    )


def main():
    parser = argparse.ArgumentParser(description="Measure SWDI and CDI code diversity for a saved heuristic log.")
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--method", default=None)
    parser.add_argument("--problem", default=None)
    parser.add_argument("--alpha", type=float, default=0.95)
    parser.add_argument("--encoding", choices=["codet5p", "tfidf"], default="codet5p")
    parser.add_argument("--model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    records = load_final_records(args.log_dir)
    report = evaluate_code_diversity(
        records,
        alpha=args.alpha,
        method=args.method,
        problem=args.problem,
        log_dir=args.log_dir,
        encoding=args.encoding,
        model_name=args.model,
        device=args.device,
    )
    if not args.no_write:
        write_code_diversity_report(report)
    print(format_report(report))


if __name__ == "__main__":
    main()
