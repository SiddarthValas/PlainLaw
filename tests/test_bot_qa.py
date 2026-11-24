"""End-to-end behaviour checks on the real corpus (offline model is fine: none of these depend on ranking quality).
Run: DENSE_MODEL=lsa python3 tests/test_bot_qa.py"""
import os
os.environ.setdefault("LLM_ROUTER", "off")   # deterministic: rules + embeddings only
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.chatbot import LegalBot  # noqa: E402
from src.retrieval import Retriever, load_sections  # noqa: E402

bot = LegalBot(Retriever(load_sections("data/sections.jsonl"), dense_model=os.getenv("DENSE_MODEL", "lsa")))
CASES = [  # (message, expected kind, optional playbook id, text must contain)
    ("i want to kill myself", "crisis", None, "14416"),
    ("I don't want to live anymore", "crisis", None, "112"),
    ("hello", "smalltalk", None, None),
    ("thanks", "smalltalk", None, None),
    ("meri gaadi ko peeche se kisi ne thok diya", "language", None, "English"),
    ("मेरा फोन चोरी हो गया", "language", None, "English"),
    ("IPC 302", "mapped", None, "BNS 103"),
    ("what is section 420", "mapped", None, "BNS 318"),
    ("my landlord is not returning my deposit", "out_of_scope", None, None),
    ("someone is raping a girl near my house right now", "plan", "sexual_assault", "112"),
    ("I was driving my car in traffic, someone rear ended me, what shall I do", "plan", "accident", None),
    ("police are asking me to come to the station tomorrow", "plan", "police_call", None),
    ("how do I file an FIR online", "plan", "fir_howto", None),
    ("asdfgh", "clarify", None, None),
    ("", "clarify", None, None),
    ("a" * 5000, "clarify", None, None),
]
for msg, kind, pb, must in CASES:
    r = bot.respond([msg])
    assert r.kind == kind, (msg[:50], r.kind, kind)
    if pb:
        assert [p["id"] for p in r.plans][0] == pb, (msg[:50], [p["id"] for p in r.plans])
    if must:
        assert must in r.text, (msg[:50], r.text)
    if kind in ("crisis", "smalltalk", "language"):
        assert not r.hits and not r.plans
# newer behaviour
r = bot.respond(["What is the punishment for murder?"])
assert r.kind in ("answer", "closest") and not r.plans, r.kind          # a definition question gets sections, not a plan
r = bot.respond(["my cheque bounced"])
assert r.kind == "out_of_scope" and any("Negotiable Instruments" in n for n in r.notes), r
r = bot.respond(["someone is selling drugs in my colony"])
assert r.kind == "out_of_scope" and any("NDPS" in n for n in r.notes), r
r = bot.respond(["My landlord changed the locks while I was away and won\u2019t let me collect my belongings"])
assert r.kind == "plan" and any("civil" in n for n in r.notes), (r.kind, r.notes)   # curly quote + civil note
r = bot.respond(["what is cheating"], playbook="fraud")
assert r.kind == "plan" and r.plans[0]["id"] == "fraud"
r = bot.respond(["the man next door beats his wife"])
assert r.kind == "plan" and r.plans[0]["id"] == "domestic"
r = bot.respond(["someone is hurting me right now please help me"])
assert "112" in r.text or r.plans, r
print(f"bot QA OK: {len(CASES)+7} cases")
