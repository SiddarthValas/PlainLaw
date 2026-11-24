"""Checks the situation playbooks against the real corpus. Run: python3 tests/test_playbooks.py"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.playbooks import load_playbooks, match, validate  # noqa: E402

ids = {json.loads(l)["id"] for l in open("data/sections.jsonl", encoding="utf-8") if l.strip()}
pbs = load_playbooks()
bad = validate(pbs, ids)
assert not bad, bad
for pb in pbs:
    assert pb.steps and pb.laws, pb.id
    assert any(s["src"] == "law" for s in pb.steps) or pb.id == "accident", pb.id
CASES = {
    "I was driving my car in traffic, someone rear ended me, what shall I do": "accident",
    "someone stole my phone": "theft",
    "police won't register my FIR": "fir_refused",
    "my brother was arrested yesterday": "arrest",
    "can I get bail in advance": "bail",
    "a man keeps following me and messaging me": "women_harassment",
    "my husband beats me and demands dowry": "domestic",
    "someone is blackmailing me": "threats",
    "I was cheated by a fake seller online": "fraud",
    "I got a call saying my account is blocked and lost 50000 through UPI": "fraud",
    "a man keeps sending me obscene messages": "women_harassment",
    "police took my brother without telling us": "arrest",
    "I hit a pedestrian with my car and he is injured": "accident",
    "someone is raping a girl near my house right now": "sexual_assault",
    "how do I file an FIR online": "fir_howto",
    "police are asking me to come to the station tomorrow": "police_call",
    "my wife is threatening to file a false case": "accused",
    "i was falsely accused of theft by my employer": "accused",
    "my child has been missing since this morning": "missing",
    "the cops beat me in custody": "custodial",
    "police seized my car and are not returning it": "seized",
    "someone is spreading false rumours about me": "defamation",
    "someone poisoned my dog": "animal",
    "my husband married another woman without divorcing me": "marriage",
    "a guy grabbed my purse and ran off": "theft",
    "auto driver is dropping me at a random place and is demanding me more money to drop": "transport",
    "the uber driver is asking for extra money and threatening me": "transport",
    "A car has been following me since 30 mins, what to do": "followed",
    "auto driver is dropping me at a random place and is demanding me more money to drop": "transport",
    "landlord entered my room when i wasnt there and took my laptop": "theft",
    "my mother in law locked me out of the house and kept my jewellery": "domestic",
    "someone filed a complaint saying I hit them but I was not even there": "accused",
    "I found out my business partner has been taking money from the company account for a year": "not_returned",
    "a girl in my college is getting rape threats on instagram": "threats",
    "the cops took my licence and asked for 2000 rs to give it back": "seized",
    "someone set my bike on fire": "fire",
    "I accidentally transferred money to the wrong person, and they are refusing to return it": "not_returned",
    "Someone borrowed my laptop for a week and has now stopped answering my calls or returning it": "not_returned",
    "My employer is refusing to return my original certificates after I resigned": "not_returned",
}
for q, want in CASES.items():
    got = [p.id for p in match(pbs, q)]
    assert want in got, (q, got)
for q in ["my dog bit a neighbour", "a stray dog bit my son", "hello", "IPC 420", "what is the weather"]:
    assert not match(pbs, q), q
assert "theft" not in [p.id for p in match(pbs, "A shopkeeper accused me of stealing")]
assert [p.id for p in match(pbs, "my laptop went missing from my hostel room")] == ["theft"]
assert "assault" not in [p.id for p in match(pbs, "my neighbour broke my car window on purpose")]
assert "arrest" not in [p.id for p in match(pbs, "police took my phone")]
assert "sexual_assault" not in [p.id for p in match(pbs, "a girl in my college is getting rape threats on instagram")]
assert "women_harassment" not in [p.id for p in match(pbs, "A car has been following me since 30 mins")]
assert "confinement" not in [p.id for p in match(pbs, "my mother in law locked me out of the house and kept my jewellery")]
print(f"playbooks OK: {len(pbs)} scenarios, all cited sections exist, {len(CASES)} trigger cases pass")
