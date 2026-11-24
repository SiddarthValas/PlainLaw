"""Print the top-3 sections for demo queries under each retrieval mode (uses the real bge model on your machine).
    python3 compare_demo.py
Queries are demo phrases and the unscored data/eval_unlabelled.jsonl set, so nothing here touches dev/test/fresh."""
import json
import os
import sys

from src.retrieval import DEFAULT_DENSE_MODEL, Retriever, load_sections

DEMO = ["someone stole my phone", "police won't register my FIR", "my roommate took money from my wallet",
        "I was cheated online by a fake seller", "my husband beats me and demands dowry", "how long can police keep me in lock-up",
        "can I get bail", "a man keeps following me and messaging me",
        "what is the punishment for murder", "punishment for theft", "what is cheating", "what is the punishment for rape"]
MODES = [("hybrid", False), ("hybrid", True), ("hybrid_title", True), ("hybrid_chunk", True)]

ret = Retriever(load_sections("data/sections.jsonl"), dense_model=os.getenv("DENSE_MODEL", DEFAULT_DENSE_MODEL))
qs = DEMO + ([json.loads(l)["query"] for l in open("data/eval_unlabelled.jsonl") if l.strip()] if os.path.exists("data/eval_unlabelled.jsonl") else [])
print(f"dense model: {ret.dense_name}\n")
for q in qs:
    print(q[:110])
    for system, rw in MODES:
        res = ret.search(q, k=3, system=system, rewrite=rw)
        tag = f"{system}{'+rewrite' if rw else ''}"
        row = " | ".join(f"{ret.sections[h.idx]['id']} {ret.sections[h.idx]['title'][:28]}" for h in res.hits)
        print(f"   {tag:<22} {'ANSWER' if Retriever.confident(res) else 'ask   '}  {row}")
    print()
