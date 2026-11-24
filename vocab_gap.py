"""Vocabulary-gap analysis: how much of a plain-language query's vocabulary appears in the section that answers it?

    python3 vocab_gap.py
Descriptive statistics over the labelled query sets (no model, no tuning). For each query:
  overlap      = share of its content stems (vague words removed) found in the union of its gold sections' text
  overlap+rw   = the same after appending the legal vocabulary the rewriting rules add
Reported per split, plus the share of queries with ZERO overlap (pure vocabulary mismatch)."""
import json
import os
import statistics as st

from src.glossary import analyse
from src.retrieval import _VAGUE_STEMS, load_sections, tokenize

S = {s["id"]: s for s in load_sections("data/sections.jsonl")}
FILES = {"dev/test": "data/eval_queries.jsonl", "fresh": "data/eval_fresh.jsonl"}


def content(q):
    return [t for t in dict.fromkeys(tokenize(q)) if t not in _VAGUE_STEMS]


def overlap(terms, gold):
    if not terms:
        return None
    have = set()
    for g in gold:
        have |= set(tokenize(f"{S[g]['title']}. {S[g]['text']}"))
    return sum(t in have for t in terms) / len(terms)


rows = []
for name, path in FILES.items():
    if not os.path.exists(path):
        continue
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if not r.get("gold"):
            continue
        base = content(r["query"])
        ext = content(analyse(r["query"]).expanded)
        rows.append((r.get("split", name), overlap(base, r["gold"]), overlap(ext, r["gold"]),
                     len(base), len(set(ext) - set(base))))
print("| Split | n | queries' words found in answer section | + rewriting | zero-overlap queries | avg words added |")
print("|---|---|---|---|---|---|")
for split in ("dev", "test", "fresh"):
    r = [x for x in rows if x[0] == split and x[1] is not None]
    if r:
        print(f"| {split} | {len(r)} | {st.mean(x[1] for x in r):.0%} | {st.mean(x[2] for x in r):.0%} | "
              f"{sum(x[1] == 0 for x in r)} ({sum(x[1] == 0 for x in r)/len(r):.0%}) | {st.mean(x[4] for x in r):.1f} |")
print("\nReading: the lower the first column, the bigger the gap between how people ask and how the statute is written.")
