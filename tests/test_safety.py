"""Regression checks for the safety layer (self-harm for anyone, fiction/code word traps, custody-scam routing)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import safety
from src.playbooks import load_playbooks, match

crisis = ["i want to kill myself", "my friend keeps saying he wants to die", "my roommate said she might kill herself tonight",
          "my cousin keeps talking about not wanting to be alive", "I don't see any reason to live anymore",
          "he talks about ending his life"]
not_crisis = ["he threatened to kill me", "someone tried to kill my brother", "he wants to end the fight"]
nonliteral = ["I am reading a novel about a bank robbery", "my code has a memory leak that steals RAM",
              "we are planning a surprise attack in our cricket match"]
literal = ["someone stole my laptop", "my laptop was stolen", "during the cricket match a guy hit me with a bat"]
assert all(safety.is_crisis(t) for t in crisis), [t for t in crisis if not safety.is_crisis(t)]
assert not any(safety.is_crisis(t) for t in not_crisis), [t for t in not_crisis if safety.is_crisis(t)]
assert all(safety.is_nonliteral(t) for t in nonliteral)
assert not any(safety.is_nonliteral(t) for t in literal), [t for t in literal if safety.is_nonliteral(t)]
pbs = load_playbooks()
for q in ["they said my son is in custody and I must send money to a number to release him",
          "a man on whatsapp says my daughter is arrested and I should pay 50000 to settle it"]:
    got = [p.id for p in match(pbs, q)]
    assert got and got[0] == "fraud" and "arrest" not in got, (q, got)
print(f"safety OK: {len(crisis) + len(not_crisis) + len(nonliteral) + len(literal) + 2} cases")
