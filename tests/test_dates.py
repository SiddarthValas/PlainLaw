"""The old-law notice: shown when the events may predate 1 July 2024 (BNS 358 / BNSS 531), not otherwise."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.dates import old_law_note

CASES = [("my phone was stolen in March 2023", True), ("he hit me 3 years ago", True), ("this happened in 2024", True),
         ("it happened in August 2024", False), ("someone stole my phone yesterday", False),
         ("i was cheated two years back", True), ("a case under the old IPC", True), ("my ID was stolen on 5 may 2024", True),
         ("I have 2023 rupees", False), ("he took ₹2020 from me", False), ("on 3 july 2024 my bike was stolen", False),
         ("in 2019 my uncle grabbed our land", True), ("last week someone threatened me", False)]
bad = [(t, e) for t, e in CASES if bool(old_law_note(t)) != e]
assert not bad, bad
print(f"dates OK: {len(CASES)} cases")
