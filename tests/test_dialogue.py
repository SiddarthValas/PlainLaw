"""Conversation layer: context is kept across turns, and every section cited in follow-up answers exists."""
import os, re, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("LLM_ROUTER", "off")
import src.chatbot as cb
from src.dialogue import Conversation, FOLLOW
from src.retrieval import DEFAULT_DENSE_MODEL, Retriever, load_sections

ret = Retriever(load_sections(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data/sections.jsonl")),
                dense_model=os.getenv("DENSE_MODEL", DEFAULT_DENSE_MODEL))
conv = Conversation(cb.LegalBot(ret, router=False))
ids = {s["id"] for s in ret.sections}

# every "BNS 75" / "BNSS 173(4)" mentioned in any follow-up answer must exist in the corpus
st = {"topic": "women_harassment", "laws": ["BNS-75", "BNS-64"], "story": "on a bus", "facts": {}}
for kind, _ in FOLLOW:
    say, cites = conv._follow(kind, st)
    for c in cites:
        assert c in ids, (kind, c)
    for act, num in re.findall(r"\b(BNSS|BNS) (\d+)", " ".join(say)):
        assert f"{act}-{num}" in ids, (kind, act, num)

if conv.bot.playbooks and conv.bot.semantic is not None:   # the full model (not the offline LSA fallback)
    script = [("Someone touched me inappropriately on a public bus.", "plan"), ("Yes, I'm safe now. This happened yesterday.", "plan"),
              ("What does Section 75 actually mean?", "explain"), ("What if I don't have any witnesses?", "followup")]
    s = {}
    for msg, kind in script:
        r, s = conv.reply(msg, s)
        assert r.kind == kind, (msg, r.kind)
    assert s["topic"] == "women_harassment" and s["focus"] == "BNS-75" and s["facts"].get("when") == "yesterday", s
    r, s2 = conv.reply("what is the punishment for it", s)
    assert r.explain and r.explain["id"] == "BNS-75" and r.explain["punishment"], r
    r, _ = conv.reply("my landlord won't return my deposit", s)
    assert r.kind == "out_of_scope", r.kind
    r, s3 = conv.reply("no, he is still here", {**s, "pending": "safety"})
    assert any("112" in t for t in r.say), r.say
print("dialogue OK")
