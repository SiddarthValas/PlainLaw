"""The citation check on LLM-written chat replies (no network: the model is replaced by fixed answers)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.llm_chat import GroundedResponder

sec = {"id": "BNS-318", "act": "BNS", "section": "318", "title": "Cheating", "replaces": "IPC 415, IPC 417, IPC 418, IPC 420",
       "text": "(1) Whoever, by deceiving any person ... commits cheating."}
mat, allowed, old = GroundedResponder.material({"facts": {"when": "yesterday"}}, None, [sec])
g = GroundedResponder.__new__(GroundedResponder)
ok = ["Report it under BNS 318 and call 1930 or 112.", "Earlier this was IPC 420; it is now BNS 318.", "It happened in 2024."]
bad = ["Under BNS 999 you can sue.", "Call 155260 now.", "Call 9876543210.", "IPC 379 applies.", "See BNSS 35.",
       "You must sign it within three days.", "It is not too late to report.", "There is a deadline for this."]
for t in ok:
    assert g.check(t, allowed, old, mat) is None, t
for t in bad:
    assert g.check(t, allowed, old, mat) is not None, t
print(f"llm chat check OK: {len(ok) + len(bad)} cases")
