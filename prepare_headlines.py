#!/usr/bin/env python3
"""
prepare_headlines.py
====================

Convert the "A Million News Headlines" CSV (column: headline_text) into a
one-document-per-line .txt file that the benchmark's `text` data source loads.

Usage (run from the vecbench folder):
    python prepare_headlines.py --csv abcnews-date-text.csv --out data/headlines.txt --n 100000

Then point configs/text_sbert.yaml at it:
    data:
      source: text
      path: data/headlines.txt
"""

from __future__ import annotations

import argparse
import csv
import os
import random


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare the headlines dataset.")
    ap.add_argument("--csv", required=True, help="path to abcnews-date-text.csv")
    ap.add_argument("--out", default="data/headlines.txt", help="output text file")
    ap.add_argument("--n", type=int, default=100000,
                    help="how many unique headlines to keep (sampled)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # Read the headline column.
    rows = []
    with open(args.csv, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if "headline_text" not in (reader.fieldnames or []):
            raise SystemExit(
                f"Expected a 'headline_text' column, found: {reader.fieldnames}"
            )
        for r in reader:
            t = (r.get("headline_text") or "").strip()
            if t:
                rows.append(t)

    # De-duplicate (preserve first occurrence) so we benchmark distinct content.
    seen, uniq = set(), []
    for t in rows:
        if t not in seen:
            seen.add(t)
            uniq.append(t)

    # Sample down to the requested size for a manageable run.
    random.seed(args.seed)
    if len(uniq) > args.n:
        uniq = random.sample(uniq, args.n)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(uniq) + "\n")

    print(f"Read {len(rows):,} rows -> {len(seen):,} unique -> wrote {len(uniq):,} "
          f"headlines to {args.out}")


if __name__ == "__main__":
    main()
