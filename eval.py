"""Compare retrieval systems on plain-language queries.

Usage:
  python eval.py --sections data/sections.jsonl --queries data/eval_queries.jsonl --split test
Query file (jsonl): {"query": "...", "gold": ["BNS-303"], "split": "dev" | "test"}
Write the rewriting rules (src/glossary.py) while looking at "dev" only; report "test".
"""
import argparse
import csv
import json
import os
import sys

from src.retrieval import DEFAULT_DENSE_MODEL, Retriever, load_sections

SYSTEMS = [
    ("TF-IDF (baseline)", "tfidf", False),
    ("BM25 (baseline)", "bm25", False),
    ("Dense embeddings (baseline)", "dense", False),
    ("BM25 + rewriting", "bm25", True),
    ("Hybrid (BM25 + dense, RRF)", "hybrid", False),
    ("Hybrid + rewriting = final system", "hybrid", True),
    ("Hybrid + rewriting + title match", "hybrid_title", True),
    ("Hybrid + pseudo-relevance feedback (no rules)", "hybrid_prf", False),
    ("Hybrid + passages", "hybrid_chunk", False),
    ("Hybrid + passages + rewriting", "hybrid_chunk", True),
]
FINAL = ("hybrid", True)   # chosen for demo quality with bge (see README); held-out numbers are reported for every system


def load_queries(path: str, split: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        qs = [json.loads(line) for line in f if line.strip()]
    return [q for q in qs if split == "all" or q.get("split") == split]


def run(ret: Retriever, queries: list[dict], system: str, rewrite: bool) -> dict:
    n = len(queries)
    hit = {1: 0, 3: 0, 5: 0}
    rr = 0.0
    conf_n = conf_hit3 = 0
    fails = []
    for q in queries:
        gold = set(q["gold"])
        res = ret.search(q["query"], k=10, system=system, rewrite=rewrite)
        ids = [ret.sections[h.idx]["id"] for h in res.hits]
        first = next((r for r, i in enumerate(ids, 1) if i in gold), None)
        if first is None or first > 3:
            fails.append((q["query"], q["gold"], ids[:3]))
        for k in hit:
            hit[k] += first is not None and first <= k
        rr += 1.0 / first if first else 0.0
        if (system, rewrite) == FINAL and Retriever.confident(res):
            conf_n += 1
            conf_hit3 += first is not None and first <= 3
    return {"n": n, "hit1": hit[1] / n, "hit3": hit[3] / n, "hit5": hit[5] / n, "mrr": rr / n,
            "coverage": conf_n / n, "conf_hit3": (conf_hit3 / conf_n) if conf_n else float("nan"), "fails": fails}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sections", default="data/sections.jsonl")
    ap.add_argument("--queries", default="data/eval_queries.jsonl")
    ap.add_argument("--split", default="test", choices=["dev", "test", "fresh", "all"])
    ap.add_argument("--dense-model", default=os.getenv("DENSE_MODEL", DEFAULT_DENSE_MODEL))
    ap.add_argument("--out", default="results")
    ap.add_argument("--show-fails", action="store_true", help="list misses for our system (dev split only)")
    a = ap.parse_args()
    if a.show_fails and a.split != "dev":
        print("--show-fails is only allowed with --split dev, so the test split is never used for tuning.")
        return 1

    sections = load_sections(a.sections)
    ids = {s["id"] for s in sections}
    queries = load_queries(a.queries, a.split)
    if not queries:
        print(f"No queries for split '{a.split}'.")
        return 1
    bad = sorted({g for q in queries for g in q["gold"] if g not in ids})
    if bad:
        print(f"Gold ids not in corpus (fix the labels): {bad}")
        return 1

    ret = Retriever(sections, dense_model=a.dense_model)
    print(f"Corpus: {len(sections)} sections | queries: {len(queries)} ({a.split}) | dense: {ret.dense_name}\n")
    print("| System | Hit@1 | Hit@3 | Hit@5 | MRR@10 |\n|---|---|---|---|---|")
    rows = []
    for name, system, rewrite in SYSTEMS:
        m = run(ret, queries, system, rewrite)
        fails = m.pop("fails")
        rows.append({"system": name, **m})
        print(f"| {name} | {m['hit1']:.1%} | {m['hit3']:.1%} | {m['hit5']:.1%} | {m['mrr']:.3f} |")
        if (system, rewrite) == FINAL:
            ours, ours_fails = m, fails
    print(f"\nWhen the bot answers directly (both retrievers agree): coverage {ours['coverage']:.1%}, "
          f"Hit@3 {ours['conf_hit3']:.1%} (vs {ours['hit3']:.1%} on all queries).")
    if a.show_fails:
        print(f"\nMisses for the final system (gold not in top 3): {len(ours_fails)}")
        for q, gold, top in ours_fails:
            print(f"  - {q}\n      gold {gold} | got {top}")
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"eval_{a.split}.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
