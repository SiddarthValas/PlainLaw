"""Do general LLM chatbots cite the right sections of the NEW criminal laws? Compared with PlainLaw on the same queries.

    python3 llm_compare.py                                    # llama3.2 through local Ollama, asked for BNS/BNSS
    python3 llm_compare.py --style free                       # asked the way a citizen would ask a chatbot
    LLM_MODEL=llama-3.3-70b-versatile LLM_BASE=https://api.groq.com/openai/v1 LLM_KEY=... python3 llm_compare.py
    python3 llm_compare.py --ours-only                        # just PlainLaw

Any OpenAI-compatible API works (Groq, OpenAI, Gemini's OpenAI endpoint, ...). Raw model answers are saved in
results/llm_raw_<model>.jsonl so a rerun does not query again.

Queries: data/eval_fresh.jsonl (21) + the test split of data/eval_queries.jsonl (27), each with gold BNS/BNSS sections.
Every section a system cites is put in one bucket:
  correct      a gold section
  old code     an IPC / CrPC section (repealed for offences after 1 July 2024)
  not in Act   a BNS / BNSS number that does not exist (BNS has 358 sections, BNSS 531)
  old number   a real BNS/BNSS number that is actually the OLD IPC/CrPC number of a gold section ("BNS 302" for murder)
  other        a real section that is not gold (may be related, may be wrong)
Note: PlainLaw's rules were partly written while looking at these queries, so its hit rate here is optimistic.
The citation-validity columns (old code / not in Act / old number) do not depend on that.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.request

PROMPTS = {
    # asked directly for the new codes, in a fixed format (the example uses placeholders, not real numbers,
    # so the model cannot copy them)
    "json": ("You are a legal information assistant for India. A citizen describes a situation. "
             "List the sections of India's current criminal laws, the Bharatiya Nyaya Sanhita, 2023 (BNS) and the "
             "Bharatiya Nagarik Suraksha Sanhita, 2023 (BNSS), that apply. Give at most 5 sections, most relevant first. "
             'Reply with JSON only, in this form: {"sections": ["<BNS or BNSS> <section number>", ...]}\n\nSituation: '),
    # how a citizen would actually ask a chatbot
    "free": "I live in India. {q} Which laws and sections apply to my situation, and what should I do?",
}
ACTS = [
    (r"bharatiya\s+nagarik\s+suraksha\s+sanhita|\bbnss\b", "BNSS"),
    (r"bharatiya\s+nyaya\s+sanhita|\bbns\b", "BNS"),
    (r"indian\s+penal\s+code|\bipc\b", "IPC"),
    (r"code\s+of\s+criminal\s+procedure|criminal\s+procedure\s+code|\bcr\.?\s?p\.?\s?c\b", "CrPC"),
    (r"bharatiya\s+sakshya|\bbsa\b|evidence\s+act|information\s+technology|\bit\s+act\b|motor\s+vehicles|pocso|"
     r"dowry\s+prohibition|negotiable", "OTHER"),
]
_ACT_ANY = "|".join(f"(?:{p})" for p, _ in ACTS)
_NUM = r"(\d{1,3}[A-Z]{0,2})"
# act name, then optional ", 2023" / ", 1860" and "(BNS)", then optional "Section", then the number
_FILL = r"(?:\s*,?\s*\d{4})?(?:\s*\([A-Za-z.\s]{2,12}\))?[\s,:)*]*"
_PAT1 = re.compile(rf"({_ACT_ANY}){_FILL}(?:section|sec\.?|s\.)?\s*{_NUM}\b", re.I)
_PAT2 = re.compile(rf"(?:section|sec\.?|s\.)\s*{_NUM}(?:\s*\([^)]*\))?\s*(?:of\s+(?:the\s+)?)?({_ACT_ANY})", re.I)


def _act(name: str) -> str:
    for p, act in ACTS:
        if re.search(p, name, re.I):
            return act
    return "OTHER"


def parse_citations(text: str) -> list[tuple[str, str]]:
    """'BNS 303', 'Section 173 of the BNSS', 'IPC 420' -> [(act, number)], in order, no duplicates."""
    found: list[tuple[int, str, str]] = []
    for m in _PAT1.finditer(text):
        found.append((m.start(), _act(m.group(1)), m.group(2).upper()))
    for m in _PAT2.finditer(text):
        found.append((m.start(), _act(m.group(2)), m.group(1).upper()))
    out = []
    for _, act, num in sorted(found):
        if (act, num) not in out:
            out.append((act, num))
    return out


def make_prompt(query: str, style: str) -> str:
    p = PROMPTS[style]
    return p.replace("{q}", query) if "{q}" in p else p + query


def ask_llm(query: str, model: str, base: str | None, key: str | None, style: str = "json", timeout: float = 180) -> str:
    prompt = make_prompt(query, style)
    if base:   # OpenAI-compatible chat completions
        body = {"model": model, "temperature": 0, "messages": [{"role": "user", "content": prompt}],
                # free tiers limit tokens per minute: cap the answer length (sections are named early in an answer)
                "max_tokens": int(os.getenv("LLM_MAX_TOKENS", "700"))}
        if "gpt-oss" in model:
            body["reasoning_effort"] = "low"   # less hidden "thinking", far fewer tokens
        req = urllib.request.Request(base.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}",
                                              "User-Agent": "PlainLaw/1.0"})
        from src.llm_router import _SSL   # certifi certificates (python.org Python on macOS has none)
        with urllib.request.urlopen(req, timeout=timeout, context=_SSL) as r:
            return json.loads(r.read())["choices"][0]["message"]["content"]
    body = {"model": model, "stream": False, "options": {"temperature": 0},
            "messages": [{"role": "user", "content": prompt}]}
    host = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    req = urllib.request.Request(host + "/api/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "User-Agent": "PlainLaw/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())["message"]["content"]


def _words(title: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]+", title.lower()) if len(w) > 3} - {"with", "from", "under", "certain", "cases", "person", "offence", "offences"}


def classify(cites: list[tuple[str, str]], gold: set[str], ids: set[str], old_of: dict[str, set[str]],
             titles: dict[str, str] | None = None) -> list[str]:
    """One bucket per citation (see module docstring). 'old number' also needs the cited section to share no title
    word with the gold sections, so a related section whose number happens to match is not counted."""
    titles = titles or {}
    gold_words = set().union(*(_words(titles.get(g, "")) for g in gold)) if gold else set()
    out = []
    old_numbers = {"BNS": set(), "BNSS": set()}
    for g in gold:
        act = g.split("-")[0]
        old_numbers[act] |= old_of.get(g, set())
    for act, num in cites:
        sid = f"{act}-{num}"
        if act in ("IPC", "CrPC"):
            out.append("old code")
        elif act == "OTHER":
            out.append("other act")
        elif sid in gold:
            out.append("correct")
        elif sid not in ids:
            out.append("not in Act")
        elif num in old_numbers[act] and not (_words(titles.get(sid, "")) & gold_words):
            out.append("old number")
        else:
            out.append("other")
    return out


def summarise(name: str, rows: list[dict]) -> dict:
    n = len(rows)
    answered = [r for r in rows if r["cites"]]
    buckets = [b for r in rows for b in r["buckets"]]
    nb = len(buckets) or 1
    bad = lambda r: any(b in ("old code", "not in Act", "old number") for b in r["buckets"])
    return {
        "system": name, "queries": n,
        "answered": f"{len(answered) / n:.0%}",
        "gold cited (all queries)": f"{sum('correct' in r['buckets'] for r in rows) / n:.0%}",
        "gold cited (when answering)": f"{sum('correct' in r['buckets'] for r in answered) / max(len(answered), 1):.0%}",
        "citations": len(buckets),
        "old code": f"{buckets.count('old code') / nb:.0%}",
        "not in Act": f"{buckets.count('not in Act') / nb:.0%}",
        "old number": f"{buckets.count('old number') / nb:.0%}",
        "answers with an invalid citation": f"{sum(map(bad, answered)) / max(len(answered), 1):.0%}",
    }


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson confidence interval for a proportion k/n."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return max(0.0, c - h), min(1.0, c + h)


def mcnemar_p(a: list[bool], b: list[bool]) -> float:
    """Exact two-sided McNemar test on paired right/wrong outcomes (binomial on the discordant pairs)."""
    from math import comb
    n01 = sum(x and not y for x, y in zip(a, b))
    n10 = sum(y and not x for x, y in zip(a, b))
    n = n01 + n10
    if n == 0:
        return 1.0
    k = min(n01, n10)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def lenient_hit(row: dict, titles: dict[str, str]) -> bool:
    """Strict hit, OR a real BNS/BNSS section whose title shares a content word with a gold section's title
    (a related section, e.g. 'Punishment for theft' for gold 'Theft'). Kinder to the LLMs."""
    if "correct" in row["buckets"]:
        return True
    gold_words = set().union(*(_words(titles.get(g, "")) for g in row["gold"])) if row["gold"] else set()
    for (act, num), b in zip(row["cites"], row["buckets"]):
        if b in ("other", "old number") and _words(titles.get(f"{act}-{num}", "")) & gold_words:
            return True
    return False


def load_queries() -> list[dict]:
    qs = [json.loads(l) for l in open("data/eval_fresh.jsonl", encoding="utf-8") if l.strip()]
    qs += [q for q in (json.loads(l) for l in open("data/eval_queries.jsonl", encoding="utf-8") if l.strip())
           if q.get("split") == "test"]
    return qs


def ours(queries: list[dict]) -> list[list[tuple[str, str]]]:
    from src.chatbot import LegalBot
    from src.retrieval import DEFAULT_DENSE_MODEL, Retriever, load_sections
    # rules + embeddings only (no LLM router), so our side is reproducible and does not share the API rate limit
    bot = LegalBot(Retriever(load_sections("data/sections.jsonl"), dense_model=os.getenv("DENSE_MODEL", DEFAULT_DENSE_MODEL)),
                   router=False)
    out = []
    for q in queries:
        r = bot.respond([q["query"]])
        secs = [law for p in r.plans for law in p["laws"]] + list(r.hits)   # clarify / out of scope -> no citation
        out.append(list(dict.fromkeys((s["act"], str(s["section"]).upper()) for s in secs))[:5])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours-only", action="store_true")
    ap.add_argument("--show", action="store_true", help="print every query with both answers")
    ap.add_argument("--cached", action="store_true",
                    help="no API calls: compare every model whose answers are saved in results/ for this --style")
    ap.add_argument("--style", choices=["json", "free"], default="json",
                    help="json: asked for BNS/BNSS sections in a fixed format; free: asked the way a citizen would")
    a = ap.parse_args()
    model = os.getenv("LLM_MODEL", "llama3.2")
    base, key = os.getenv("LLM_BASE"), os.getenv("LLM_KEY")

    sections = [json.loads(l) for l in open("data/sections.jsonl", encoding="utf-8") if l.strip()]
    ids = {s["id"] for s in sections}
    titles = {s["id"]: s["title"] for s in sections}
    old_of = {s["id"]: set(re.findall(r"(?:IPC|CrPC)\s*(\d+[A-Z]*)", s.get("replaces", ""))) for s in sections}
    queries = load_queries()

    systems = {"PlainLaw (ours)": ours(queries)}
    if a.cached:
        import glob
        import hashlib
        tag = hashlib.sha1(PROMPTS[a.style].encode()).hexdigest()[:6]
        for path in sorted(glob.glob(f"results/llm_raw_*_{a.style}_{tag}.jsonl")):
            name = os.path.basename(path)[len("llm_raw_"):-len(f"_{a.style}_{tag}.jsonl")]
            saved = {}
            for l in open(path, encoding="utf-8"):
                c = json.loads(l)
                if c["answer"]:
                    saved[c["query"]] = c["answer"]
            systems[f"{name} ({a.style})"] = [parse_citations(saved.get(q["query"], ""))[:5] for q in queries]
    elif not a.ours_only:
        os.makedirs("results", exist_ok=True)
        import hashlib
        tag = hashlib.sha1(PROMPTS[a.style].encode()).hexdigest()[:6]   # a changed prompt never reuses old answers
        cache_path = f"results/llm_raw_{re.sub(r'[^A-Za-z0-9.]+', '_', model)}_{a.style}_{tag}.jsonl"
        cache = {}
        if os.path.exists(cache_path):
            for l in open(cache_path, encoding="utf-8"):
                c = json.loads(l)
                if c["answer"]:   # failed calls were saved empty: ask them again
                    cache[c["query"]] = c["answer"]
        cites = []
        with open(cache_path, "a", encoding="utf-8") as f:
            for i, q in enumerate(queries, 1):
                if q["query"] not in cache:
                    for attempt in range(4):   # free tiers rate-limit (HTTP 429): wait and retry
                        try:
                            cache[q["query"]] = ask_llm(q["query"], model, base, key, a.style)
                            break
                        except Exception as e:
                            if "429" in str(e) and attempt < 3:
                                time.sleep(15)
                                continue
                            print(f"  query {i}: {type(e).__name__}: {e}", file=sys.stderr)
                            cache[q["query"]] = ""   # counts as no answer now; not saved, so a rerun asks again
                    if cache[q["query"]]:
                        f.write(json.dumps({"query": q["query"], "answer": cache[q["query"]]}, ensure_ascii=False) + "\n")
                        f.flush()
                    print(f"  asked {model}: {i}/{len(queries)}", end="\r", file=sys.stderr)
                    time.sleep(float(os.getenv("LLM_SLEEP", "0")))
                cites.append(parse_citations(cache[q["query"]])[:5])
        systems[f"{model} ({a.style})"] = cites

    summary, all_rows = [], {}
    for name, cites in systems.items():
        rows = [{"query": q["query"], "gold": q["gold"], "cites": c,
                 "buckets": classify(c, set(q["gold"]), ids, old_of, titles)} for q, c in zip(queries, cites)]
        all_rows[name] = rows
        summary.append(summarise(name, rows))
        if a.show:
            print(f"\n== {name}")
            for r in rows:
                pairs = ", ".join(f"{act} {num} [{b}]" for (act, num), b in zip(r["cites"], r["buckets"])) or "(no citation)"
                print(f"- {r['query'][:90]}\n    gold {r['gold']} | {pairs}")

    cols = list(summary[0])
    print("\n| " + " | ".join(cols) + " |\n|" + "---|" * len(cols))
    for s in summary:
        print("| " + " | ".join(str(s[c]) for c in cols) + " |")
    # statistics: 95% confidence intervals, a kinder "related section" score, and McNemar's test against ours
    ours_rows = all_rows["PlainLaw (ours)"]
    ours_hit = ["correct" in r["buckets"] for r in ours_rows]
    print("\n| system | correct section (95% CI) | correct or related section (95% CI) | McNemar p vs ours |\n|---|---|---|---|")
    stats = []
    for name, rows in all_rows.items():
        n = len(rows)
        strict = ["correct" in r["buckets"] for r in rows]
        lenient = [lenient_hit(r, titles) for r in rows]
        lo, hi = wilson(sum(strict), n)
        llo, lhi = wilson(sum(lenient), n)
        p = "-" if name == "PlainLaw (ours)" else f"{mcnemar_p(ours_hit, strict):.2g}"
        print(f"| {name} | {sum(strict)/n:.0%} ({lo:.0%}-{hi:.0%}) | {sum(lenient)/n:.0%} ({llo:.0%}-{lhi:.0%}) | {p} |")
        stats.append({"system": name, "n": n, "strict": sum(strict), "lenient": sum(lenient), "mcnemar_p_vs_ours": p})
    with open("results/llm_compare.json", "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "stats": stats,
                   "rows": {k: [{**r, "cites": [f"{a} {b}" for a, b in r["cites"]]} for r in v] for k, v in all_rows.items()}},
                  f, indent=2)
    print("\nSaved results/llm_compare.json. Buckets are explained at the top of llm_compare.py.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
