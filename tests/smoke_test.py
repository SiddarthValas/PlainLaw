"""Smoke test on the synthetic fixture: run from the project root with `python tests/smoke_test.py`."""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.chatbot import LegalBot, _grounded, find_references  # noqa: E402
from src.glossary import analyse  # noqa: E402
from src.retrieval import Retriever, load_sections  # noqa: E402

sections = load_sections(os.path.join(ROOT, "tests", "fixture_sections.jsonl"))
ret = Retriever(sections, dense_model="lsa")
bot = LegalBot(ret)
queries = [json.loads(l) for l in open(os.path.join(ROOT, "tests", "fixture_eval.jsonl")) if l.strip()]

fails = []

# 1. retrieval finds the gold section in the top 3 for plain-language queries
for q in queries:
    res = ret.search(q["query"], k=3)
    ids = [sections[h.idx]["id"] for h in res.hits]
    ok = set(q["gold"]) & set(ids)
    print(("ok  " if ok else "FAIL"), q["query"], "->", ids)
    if not ok:
        fails.append(f"retrieval: {q['query']}")

# 2. civil matters are routed out of scope; criminal wording with civil words is not
r = bot.respond(["landlord won't return my deposit"])
print("civil ->", r.kind)
if r.kind != "out_of_scope":
    fails.append("civil gate")
if analyse("my landlord threatened me with a knife").out_of_scope:
    fails.append("civil gate too aggressive")

# 3. exact lookups for old and new numbers
if find_references("what is section 420 of the IPC") != ([("IPC", "420")], []):
    fails.append("old-ref parsing")
if find_references("CrPC 154 and BNSS 173") != ([("CrPC", "154")], [("BNSS", "173")]):
    fails.append("ref parsing 2")
r = bot.respond(["IPC 9001"])
print("lookup ->", r.kind, r.text)
if r.kind != "mapped" or r.hits[0]["id"] != "TEST-1":
    fails.append("old->new lookup")

# 4. gibberish never produces cards
r = bot.respond(["qzxv wplk"])
print("gibberish ->", r.kind)
if r.hits:
    fails.append("gibberish returned cards")

# 5. grounding: excerpts are verbatim from the section, model lines are number-checked
for q in queries:
    reply = bot.respond([q["query"]])
    for card in reply.hits:
        original = " ".join(next(s for s in sections if s["id"] == card["id"])["text"].split())
        if not original.startswith(card["excerpt"].rstrip("…")):
            fails.append(f"excerpt not verbatim: {card['id']}")
if not _grounded("It covers taking property.", sections[0]):
    fails.append("grounded rejects a clean line")
if _grounded("This is section 999.", sections[0]):
    fails.append("grounded accepts a foreign number")

# 6. a finished answer starts a fresh question; a clarifying question keeps context
if bot.next_turns(["someone stole my phone"], "answer", "landlord won't return my deposit") != ["landlord won't return my deposit"]:
    fails.append("new question after answer kept old context")
if bot.next_turns(["help"], "clarify", "he hit me") != ["help", "he hit me"]:
    fails.append("clarify lost context")

print("\nFAILURES:" if fails else "\nALL PASSED", *fails, sep="\n  " if fails else "")
sys.exit(1 if fails else 0)
