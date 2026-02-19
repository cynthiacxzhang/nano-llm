# data/fetch.py — fetch arXiv ML/NLP abstracts
#
# Uses the arXiv API (Atom XML) — no dependencies beyond stdlib.
# Targets cs.LG (machine learning) and cs.CL (computation and language).
#
# Usage:
#   python data/fetch.py              # fetch 5000 abstracts (default)
#   python data/fetch.py --n 10000    # fetch 10000 abstracts
#
# Output: data/corpus.txt, one abstract per line.

import urllib.request
import xml.etree.ElementTree as ET
import time
import argparse
import re
import os

ARXIV_API = "http://export.arxiv.org/api/query"
CATEGORIES = "cat:cs.LG+OR+cat:cs.CL"
BATCH_SIZE = 100          # arXiv API max per request
DELAY_SECONDS = 3.5       # arXiv asks for >3s between requests

ATOM_NS = "http://www.w3.org/2005/Atom"


def fetch_batch(start: int, max_results: int) -> list[str]:
    """Fetch one batch of abstracts from the arXiv API."""
    url = (
        f"{ARXIV_API}"
        f"?search_query={CATEGORIES}"
        f"&start={start}"
        f"&max_results={max_results}"
        f"&sortBy=submittedDate"
        f"&sortOrder=descending"
    )
    with urllib.request.urlopen(url, timeout=30) as resp:
        xml = resp.read()

    root = ET.fromstring(xml)
    abstracts = []
    for entry in root.findall(f"{{{ATOM_NS}}}entry"):
        summary = entry.find(f"{{{ATOM_NS}}}summary")
        if summary is not None and summary.text:
            abstracts.append(clean(summary.text))
    return abstracts


def clean(text: str) -> str:
    """Normalize whitespace and remove arXiv formatting artifacts."""
    # collapse internal newlines/tabs to a single space
    text = re.sub(r"\s+", " ", text)
    # strip leading/trailing whitespace
    text = text.strip()
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5000, help="number of abstracts to fetch")
    args = parser.parse_args()

    out_path = os.path.join(os.path.dirname(__file__), "corpus.txt")
    total = 0
    start = 0

    print(f"fetching {args.n} abstracts from arXiv (cs.LG + cs.CL)...")

    with open(out_path, "w", encoding="utf-8") as f:
        while total < args.n:
            batch_size = min(BATCH_SIZE, args.n - total)
            try:
                abstracts = fetch_batch(start, batch_size)
            except Exception as e:
                print(f"  error at offset {start}: {e} — retrying in 10s")
                time.sleep(10)
                continue

            if not abstracts:
                print("  no more results, stopping early.")
                break

            for abstract in abstracts:
                f.write(abstract + "\n")

            total += len(abstracts)
            start += batch_size
            print(f"  fetched {total}/{args.n}")

            if total < args.n:
                time.sleep(DELAY_SECONDS)

    print(f"done. wrote {total} abstracts to {out_path}")
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"corpus size: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
