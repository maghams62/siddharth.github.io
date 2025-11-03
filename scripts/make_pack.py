#!/usr/bin/env python3
"""
Generate a lightweight TF-IDF pack from a resume PDF for the browser-side RAG chatbot.

Usage:
    python scripts/make_pack.py assets/Siddharth_Suresh_Resume.pdf assets/rag/resume_rag_pack.json
"""
import argparse
import collections
import datetime as dt
import json
import math
import pathlib
import re
import sys
from typing import Dict, Iterable, List, Sequence, Tuple


STOPWORDS: Sequence[str] = tuple(
    """
    a about above after again against all am an and any are aren't as at be because been
    before being below between both but by can can't could couldn't did didn't do does
    doesn't doing don't down during each few for from further had hadn't has hasn't have
    haven't having he he'd he'll he's her here here's hers herself him himself his how
    how's i i'd i'll i'm i've if in into is isn't it it's its itself let's me more most
    mustn't my myself no nor not of off on once only or other ought our ours ourselves
    out over own same shan't she she'd she'll she's should shouldn't so some such than
    that that's the their theirs them themselves then there there's these they they'd
    they'll they're they've this those through to too under until up very was wasn't we
    we'd we'll we're we've were weren't what what's when when's where where's which while
    who who's whom why why's with won't would wouldn't you you'd you'll you're you've your
    yours yourself yourselves
    """.split()
)

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")
UNICODE_DASHES = "‒–—―−"
BULLET_CHARS = "•●▪▪︎◦◉"


def load_pdf_text(path: pathlib.Path) -> str:
    """Extract text from the PDF using PyPDF2, falling back to pdfminer if needed."""
    try:
        from PyPDF2 import PdfReader  # type: ignore

        reader = PdfReader(str(path))
        pages = []
        for page in reader.pages:
            try:
                pages.append(page.extract_text() or "")
            except Exception:
                continue
        text = "\n".join(pages)
        if text.strip():
            return text
    except ModuleNotFoundError:
        pass
    except Exception as err:
        sys.stderr.write(f"[warn] PyPDF2 failed: {err}\n")

    try:
        from pdfminer.high_level import extract_text  # type: ignore

        return extract_text(str(path))
    except ModuleNotFoundError:
        raise SystemExit(
            "Neither PyPDF2 nor pdfminer.six is available. Install one of them first."
        )


def normalize_text(raw: str) -> str:
    """Clean bullet characters and normalize whitespace."""
    cleaned = raw.replace("\r\n", "\n")
    for ch in BULLET_CHARS:
        cleaned = cleaned.replace(ch, "-")
    for ch in UNICODE_DASHES:
        cleaned = cleaned.replace(ch, "-")
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"\n-\s*", "\n- ", cleaned)
    cleaned = cleaned.encode("ascii", "ignore").decode("ascii")
    cleaned = cleaned.replace("ap-arker-alt", "Location: ")
    cleaned = cleaned.replace("/envelpe", " Email: ")
    cleaned = cleaned.replace("phone-alt", " Phone: ")
    return cleaned.strip()


def split_chunks(text: str) -> List[str]:
    """Break text into resume sections and bullet-level chunks."""
    lines = text.splitlines()
    chunks: List[str] = []
    current: List[str] = []

    def flush() -> None:
        if not current:
            return
        chunk = " ".join(current).strip()
        chunk = re.sub(r"\s{2,}", " ", chunk)
        if len(chunk.split()) >= 5:
            chunks.append(chunk)
        current.clear()

    for line in lines:
        stripped = line.strip()
        if not stripped:
            flush()
            continue
        if stripped.startswith("-") and current:
            flush()
        current.append(stripped)
    flush()
    return chunks


def tokenize(text: str) -> List[str]:
    tokens: List[str] = []
    for match in TOKEN_PATTERN.finditer(text.lower()):
        token = match.group(0)
        if token.isdigit():
            continue
        if token in STOPWORDS:
            continue
        tokens.append(token)
    return tokens


def build_pack(chunks: Sequence[str]) -> Dict[str, object]:
    tokenized_chunks: List[List[str]] = [tokenize(chunk) for chunk in chunks]
    filtered_chunks: List[Tuple[str, List[str]]] = []
    for original, tokens in zip(chunks, tokenized_chunks):
        if len(tokens) < 5:
            continue
        filtered_chunks.append((original, tokens))

    if not filtered_chunks:
        raise SystemExit("No chunks with sufficient tokens were extracted from the PDF.")

    vocab_counter: collections.Counter[str] = collections.Counter()
    for _, tokens in filtered_chunks:
        vocab_counter.update(tokens)

    vocab = sorted(vocab_counter.keys())
    token_to_index = {token: idx for idx, token in enumerate(vocab)}

    doc_freq: Dict[int, int] = collections.Counter()
    chunk_tf_vectors: List[List[Tuple[int, int]]] = []
    for _, tokens in filtered_chunks:
        tf_counter = collections.Counter(tokens)
        tf_vector: List[Tuple[int, int]] = []
        seen_indices = set()
        for token, count in tf_counter.items():
            idx = token_to_index[token]
            tf_vector.append((idx, count))
            seen_indices.add(idx)
        for idx in seen_indices:
            doc_freq[idx] += 1
        chunk_tf_vectors.append(sorted(tf_vector))

    num_docs = len(filtered_chunks)
    idf_values: Dict[int, float] = {}
    for idx, df in doc_freq.items():
        idf = math.log((1 + num_docs) / (1 + df)) + 1.0
        idf_values[idx] = idf

    idf_pairs = sorted((idx, idf_values[idx]) for idx in idf_values)

    chunk_entries = []
    for (text, _), tf_vector in zip(filtered_chunks, chunk_tf_vectors):
        norm_sum = 0.0
        weighted = []
        for idx, raw_tf in tf_vector:
            weight = raw_tf * idf_values[idx]
            norm_sum += weight * weight
            weighted.append((idx, raw_tf))
        norm = math.sqrt(norm_sum) if norm_sum else 1.0
        chunk_entries.append(
            {
                "text": text,
                "tf": [[idx, raw_tf] for idx, raw_tf in weighted],
                "norm": norm,
            }
        )

    pack = {
        "schema": "tfidf-rag-pack@v1",
        "source": "Siddharth_Suresh_Resume.pdf",
        "created": dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "vocab": vocab,
        "idf_pairs": idf_pairs,
        "chunks": chunk_entries,
    }
    return pack


def write_pack(pack: Dict[str, object], output_path: pathlib.Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(pack, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate TF-IDF resume pack for browser RAG.")
    parser.add_argument("pdf_path", type=pathlib.Path, help="Path to resume PDF")
    parser.add_argument("output_path", type=pathlib.Path, help="Path to write JSON pack")
    return parser.parse_args(argv)


def main(argv: Sequence[str]) -> None:
    args = parse_args(argv)
    if not args.pdf_path.exists():
        raise SystemExit(f"PDF not found: {args.pdf_path}")

    raw_text = load_pdf_text(args.pdf_path)
    normalized = normalize_text(raw_text)
    chunks = split_chunks(normalized)
    pack = build_pack(chunks)
    write_pack(pack, args.output_path)
    print(f"Saved pack with {len(pack['chunks'])} chunks, vocab size {len(pack['vocab'])}.")


if __name__ == "__main__":
    main(sys.argv[1:])
