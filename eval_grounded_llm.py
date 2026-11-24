"""Baseline E: a GROUNDED LLM (retrieval-augmented) on the same corpus and the same frozen test questions.

    LLM_BASE=https://api.groq.com/openai/v1 LLM_KEY=... python3 eval_grounded_llm.py      # gpt-oss-120b (default)
    python3 eval_grounded_llm.py --local                                                  # llama3.2 via Ollama

For each test question with gold sections (data/eval_final.jsonl), the top-8 sections from system D (hybrid + glossary
expansion) are given to the LLM with their titles and opening text. The LLM must rank the sections that apply, using
only those ids. This is the fair comparison the earlier, ungrounded LLM test lacked: same corpus, same candidates.
Measures: Hit@1 / Hit@3 of the LLM's ranking vs D's own ranking (paired McNemar), citations outside the given
candidates or outside the corpus, latency, failed calls. Answers are cached in results/grounded_llm_cache.json, so a
rate-limited run can be resumed. Nothing is estimated: unanswered questions are reported as failures.
"""
import argparse
import json
import os
import re
import statistics
import sys
import time
import urllib.error

from eval_final import mcnemar_exact
from llm_compare import wilson
from src.llm_router import _post
from src.retrieval import DEFAULT_DENSE_MODEL, Retriever, load_sections

ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(ROOT, "results", "grounded_llm_cache.json")
K = 8
PROMPT = """You help ordinary people in India find which sections of the Bharatiya Nyaya Sanhita (BNS) or Bharatiya
Nagarik Suraksha Sanhita (BNSS), 2023 apply to their situation. Use ONLY the candidate sections below. Do not use the
old IPC or CrPC. Do not cite any section that is not in the list.

Situation: "{q}"

Candidate sections:
{cands}

Reply with JSON only: {{"sections": ["<id>", ...]}}, listing up to 3 candidate ids that apply, most relevant first.
If none applies, reply {{"sections": []}}."""


def ask(prompt: str, local: bool) -> str:
    if local:
        out = _post(os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/") + "/api/chat",
                    {"model": os.getenv("LOCAL_LLM_MODEL", "llama3.2"), "stream": False, "format": "json",
                     "options": {"temperature": 0}, "messages": [{"role": "user", "content": prompt}]}, {}, 120)
        return out["message"]["content"]
    base, key = os.environ["LLM_BASE"].rstrip("/"), os.environ["LLM_KEY"]
    model = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
    body = {"model": model, "temperature": 0, "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"}, "max_tokens": 700}
    if "gpt-oss" in model:
        body["reasoning_effort"] = "low"
    for attempt in range(4):
        try:
            return _post(base + "/chat/completions", body, {"Authorization": f"Bearer {key}"}, 60)["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            if e.code != 429 or attempt == 3:
                raise
            time.sleep(15)
    raise RuntimeError("rate limited")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", action="store_true")
    ap.add_argument("--sleep", type=float, default=float(os.getenv("LLM_SLEEP", "4")), help="pause between calls")
    a = ap.parse_args()
    model = os.getenv("LOCAL_LLM_MODEL", "llama3.2") if a.local else os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
    rows = [json.loads(l) for l in open(os.path.join(ROOT, "data/eval_final.jsonl"), encoding="utf-8")]
    rows = [r for r in rows if r["gold"]]
    ret = Retriever(load_sections(os.path.join(ROOT, "data/sections.jsonl")), dense_model=DEFAULT_DENSE_MODEL)
    corpus = {s["id"] for s in ret.sections}
    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    key = lambda q: f"{model}|{q}"
    d1, d3, e1, e3, outside, fake, failed, ms = [], [], [], [], 0, 0, 0, []
    for i, r in enumerate(rows, 1):
        res = ret.search(r["query"], k=K, system="hybrid", rewrite=True)
        cands = [ret.sections[h.idx] for h in res.hits]
        ids = [c["id"] for c in cands]
        d1.append(ids[0] in r["gold"]); d3.append(bool(set(ids[:3]) & set(r["gold"])))
        if key(r["query"]) not in cache:
            text = "\n".join(f'- {c["id"]}: {c["title"]}. {" ".join(c["text"].split())[:350]}' for c in cands)
            t = time.perf_counter()
            try:
                raw = ask(PROMPT.format(q=r["query"], cands=text), a.local)
                cache[key(r["query"])] = {"raw": raw, "ms": (time.perf_counter() - t) * 1000}
                json.dump(cache, open(CACHE, "w"), indent=0)
            except Exception as ex:
                print(f"  [{i}/{len(rows)}] call failed: {type(ex).__name__}: {str(ex)[:80]}")
            time.sleep(a.sleep)
        got = cache.get(key(r["query"]))
        if not got:
            failed += 1; e1.append(False); e3.append(False)
            continue
        ms.append(got["ms"])
        try:
            secs = [str(x).strip().upper().replace(" ", "-") for x in json.loads(got["raw"]).get("sections", [])]
        except Exception:
            secs = re.findall(r"BNSS?-\d+[A-Z]?", got["raw"].upper())
        secs = [re.sub(r"^(BNSS?)-?(\d+)", r"\1-\2", s) for s in secs]
        outside += sum(s not in ids for s in secs)
        fake += sum(s not in corpus for s in secs)
        e1.append(bool(secs) and secs[0] in r["gold"]); e3.append(bool(set(secs[:3]) & set(r["gold"])))
        if i % 10 == 0:
            print(f"  {i}/{len(rows)} done")
    n = len(rows)
    print(f"\nBaseline E: grounded LLM ({model}) re-ranking D's top {K} candidates | {n} questions with gold sections\n")
    print("| System | Hit@1 (95% CI) | Hit@3 (95% CI) |\n|---|---|---|")
    for name, h1, h3 in [("D  Hybrid + glossary expansion", d1, d3), (f"E  Grounded LLM ({model})", e1, e3)]:
        f = lambda k: f"{k}/{n} = {k/n:.0%} ({wilson(k, n)[0]:.0%}–{wilson(k, n)[1]:.0%})"
        print(f"| {name} | {f(sum(h1))} | {f(sum(h3))} |")
    print(f"\nPaired McNemar E vs D: Hit@1 p = {mcnemar_exact(e1, d1):.3f}, Hit@3 p = {mcnemar_exact(e3, d3):.3f}")
    print(f"Citations outside the given candidates: {outside}; not in the corpus at all: {fake}; failed calls: {failed}"
          + (" (failed calls count as misses above; see the answered-only line)" if failed else ""))
    answered = [i for i, r in enumerate(rows) if cache.get(key(r["query"]))]
    if failed and answered:
        m = len(answered)
        print(f"Answered only ({m}): D Hit@1 {sum(d1[i] for i in answered)/m:.0%}, Hit@3 {sum(d3[i] for i in answered)/m:.0%} | "
              f"E Hit@1 {sum(e1[i] for i in answered)/m:.0%}, Hit@3 {sum(e3[i] for i in answered)/m:.0%}")
    if ms:
        ms.sort()
        print(f"LLM latency: median {statistics.median(ms):.0f} ms, p95 {ms[int(.95 * len(ms)) - 1]:.0f} ms")
    json.dump({"model": model, "n": n, "D_hit1": sum(d1) / n, "D_hit3": sum(d3) / n, "E_hit1": sum(e1) / n,
               "E_hit3": sum(e3) / n, "outside": outside, "not_in_corpus": fake, "failed": failed,
               "mcnemar_hit1": mcnemar_exact(e1, d1), "mcnemar_hit3": mcnemar_exact(e3, d3)},
              open(os.path.join(ROOT, f"results/grounded_llm_{re.sub(r'[^a-z0-9]+', '_', model.lower())}.json"), "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
