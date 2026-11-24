"""Conditional laws in playbooks show only when the user's words mention the circumstance."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import os
os.environ.setdefault("LLM_ROUTER", "off")   # deterministic: rules + embeddings only
os.environ.setdefault("DENSE_MODEL", "lsa")
from src.chatbot import LegalBot
from src.retrieval import Retriever, load_sections

bot = LegalBot(Retriever(load_sections("data/sections.jsonl"), dense_model="lsa"))

def laws(q):
    r = bot.respond([q])
    assert r.kind == "plan", (q, r.kind)
    return {l["act"] + "-" + str(l["section"]) for p in r.plans for l in p["laws"]}

CASES = [  # (query, must include, must exclude)
    ("someone rear ended my car in traffic", {"BNS-281"}, {"BNS-106"}),
    ("someone rear ended my car and the driver died", {"BNS-281", "BNS-106"}, set()),
    ("I lost money in a UPI scam, a caller asked for my OTP", {"BNS-318"}, {"BNS-336", "BNS-340", "BNS-316"}),
    ("someone stole my bike from outside my house", {"BNS-303"}, {"BNS-309", "BNS-304"}),
    ("a man on a bike snatched my chain", {"BNS-303", "BNS-304"}, set()),
    ("My neighbour's son broke into our house at night and stole cash", {"BNS-303", "BNS-331"}, {"BNS-324"}),
    ("a dog bit me while I was walking", {"BNS-291"}, {"BNS-115"}),
    ("My friend took my scooter for a day and has not returned it in two weeks", {"BNS-316", "BNS-314"}, set()),
]
for q, inc, exc in CASES:
    got = laws(q)
    assert inc <= got, (q, "missing", inc - got, got)
    assert not (exc & got), (q, "should not show", exc & got, got)
print(f"conditional laws OK: {len(CASES)} cases")
