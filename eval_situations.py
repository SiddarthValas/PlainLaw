"""Routing accuracy: does a situation reach the right scenario playbook?

    python3 eval_situations.py                                   # rules + embeddings, data/eval_situations*.jsonl
    python3 eval_situations.py --compare                         # rules alone vs + LLM (fallback) vs LLM first
    python3 eval_situations.py --queries data/eval_situations3.jsonl --show

"expect" is a playbook id, a list of acceptable ids, or "none" (no plan should be shown: definitions, civil matters,
situations no playbook covers). Score = the first plan shown is an acceptable one (or no plan when "none").
eval_situations.jsonl / eval_situations2.jsonl were used while writing the rules (dev). eval_situations3.jsonl
(20 situations) was written as a blind test and run once with rules only before any LLM routing existed.
LLM routing needs Ollama running locally (llama3.2) or LLM_BASE + LLM_KEY for an API such as Groq.
"""
import argparse
import glob
import json
import os
import sys

import src.chatbot as cb
from src.llm_router import LLMRouter
from llm_compare import wilson
from src.retrieval import DEFAULT_DENSE_MODEL, Retriever, load_sections


def score(bot, qs, show=False, label=""):
    ok = 0
    for q in qs:
        exp = q["expect"] if isinstance(q["expect"], list) else [q["expect"]]
        r = bot.respond([q["query"]])
        got = [p["id"] for p in r.plans]
        good = (not got) if exp == ["none"] else bool(got) and (got[0] in exp or ("none" in exp and not got))
        if "none" in exp and not got:
            good = True
        ok += good
        if show and not good:
            print(f"  MISS [{label}] expect={exp} got={got or r.kind} | {q['query']}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", nargs="*", default=sorted(glob.glob("data/eval_situations*.jsonl")))
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()
    ret = Retriever(load_sections("data/sections.jsonl"), dense_model=os.getenv("DENSE_MODEL", DEFAULT_DENSE_MODEL))
    rules = cb.LegalBot(ret, router=False)
    configs = [("rules + embeddings", rules, "fallback")]
    router = None
    if a.compare:
        router = LLMRouter(rules.playbooks)
        if not router.available:
            print("No LLM available (start Ollama, or set LLM_BASE and LLM_KEY). Showing rules only.")
        else:
            llm_bot = cb.LegalBot(ret, router=router)
            configs += [(f"+ LLM when rules find nothing ({router.name})", llm_bot, "fallback"),
                        (f"LLM first, rules second ({router.name})", llm_bot, "first")]
    print(f"dense: {ret.dense_name}\n")
    header = "| Set | n | " + " | ".join(c[0] for c in configs) + " |"
    print(header + "\n|" + "---|" * (len(configs) + 2))
    for path in a.queries:
        qs = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
        cells = []
        for name, bot, mode in configs:
            cb.ROUTER_MODE = mode
            k = score(bot, qs, a.show, name)
            lo, hi = wilson(k, len(qs))
            cells.append(f"{k}/{len(qs)} ({k/len(qs):.0%}, CI {lo:.0%}-{hi:.0%})")
        print(f"| {os.path.basename(path)} | {len(qs)} | " + " | ".join(cells) + " |")
    if a.compare and router.available:
        print(f"\nLLM calls that failed: {len(router.errors)} recent" + (":\n  " + "\n  ".join(router.errors) if router.errors else " (none)"))
    print("\nScore = the first plan shown is acceptable (or no plan when none should be shown). CI = 95% Wilson interval.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
