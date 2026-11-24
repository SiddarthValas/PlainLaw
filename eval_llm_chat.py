"""Runs the grounded chat replies against the real API using groq_key.txt (the key is read from the file, never printed)."""
import json, os, sys, time
ROOT = sys.argv[1]
sys.path.insert(0, ROOT)
key = open(os.path.join(ROOT, "groq_key.txt")).read().strip()
os.environ["LLM_BASE"] = "https://api.groq.com/openai/v1"; os.environ["LLM_KEY"] = key; os.environ["LOCAL_LLM_MODEL"] = "none"
os.environ["LLM_TIMEOUT"] = "20"
from src.llm_chat import GroundedResponder
g = GroundedResponder()
print("models (in order):", g.models if g.api_ok else "none")
cases = json.load(open(sys.argv[2]))
ok = rej = fail = 0
for c in cases:
    t = time.time()
    out = g.generate(c["history"], c["message"], c["material"], set(c["allowed"]), set(c["old"]))
    ms = (time.time() - t) * 1000
    print(f"\nYOU: {c['message']}")
    if out:
        ok += 1; print(f"  BOT [{g.last_model}] ({ms:.0f} ms): {out}")
    elif g.rejected and len(g.rejected) > rej:
        rej += 1; print(f"  REJECTED by citation check: {g.rejected[-1]}")
    else:
        fail += 1; print(f"  FAILED: {g.errors[-1] if g.errors else '?'}")
    time.sleep(5)   # free-tier rate limit
print(f"\nanswered {ok}, rejected by the check {rej}, failed {fail}")
