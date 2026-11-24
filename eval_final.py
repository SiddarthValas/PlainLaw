"""Final evaluation on the frozen, held-out synthetic set (data/eval_final.jsonl, 150 questions).

    python3 eval_final.py                 # offline: retrieval baselines A-D, full system F (rules + trained classifier)
    python3 eval_final.py --llm           # also F with the LLM router (needs LLM_BASE/LLM_KEY or Ollama)

The system was frozen (results/frozen_config.json) before the set was written; nothing was changed after the first run.
The questions are SYNTHETIC: written by the AI assistant after the freeze, not collected from real users and not
checked by a lawyer. Results are reported for all 150 and for the "clean" subset whose questions are not
near-duplicates (cosine < 0.90) of any older evaluation question or playbook example (results/final_overlap.json).

Outputs: results/final_eval.json (all numbers + per-question rows), results/risk_coverage.png.
"""
import argparse
import json
import math
import os
import resource
import statistics
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from llm_compare import wilson                                                     # noqa: E402
import src.chatbot as cb                                                           # noqa: E402
from src.retrieval import DEFAULT_DENSE_MODEL, Retriever, load_sections            # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
BASELINES = [("A  BM25", "bm25", False), ("B  Dense (bge-small)", "dense", False),
             ("C  Hybrid BM25+dense (RRF)", "hybrid", False), ("D  Hybrid + glossary expansion", "hybrid", True)]


def mcnemar_exact(a: list[bool], b: list[bool]) -> float:
    """Two-sided exact McNemar test on paired binary outcomes."""
    n01 = sum(1 for x, y in zip(a, b) if x and not y)
    n10 = sum(1 for x, y in zip(a, b) if y and not x)
    n = n01 + n10
    if n == 0:
        return 1.0
    k = min(n01, n10)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def pct(k, n):
    lo, hi = wilson(k, n) if n else (0, 0)
    return f"{k}/{n} = {k / n:.0%} ({lo:.0%}–{hi:.0%})" if n else "–"


def route_outcome(reply, route: list[str]) -> str:
    """correct | abstained | wrong. 'none' = no specific playbook should be shown; 'clarify' = asking is right."""
    first = reply.plans[0]["id"] if reply.plans else None
    if "clarify" in route and reply.kind == "clarify":
        return "correct"
    if first and first != "generic" and first in route:
        return "correct"
    if "none" in route and (first is None or first == "generic") and reply.kind != "clarify":
        return "correct"
    if reply.kind in ("clarify", "empty"):
        return "abstained"
    if reply.kind == "crisis" and "none" in route:   # a crisis reply is a safe non-answer
        return "correct"
    return "wrong"


def shown_sections(reply) -> list[str]:
    ids = [law["id"] for p in reply.plans for law in p["laws"]] + [h["id"] for h in reply.hits]
    return list(dict.fromkeys(ids))


def run_system(bot, rows, label):
    out = []
    for r in rows:
        t = time.perf_counter()
        rep = bot.respond([r["query"]])
        ms = (time.perf_counter() - t) * 1000
        out.append({"query": r["query"], "category": r["category"], "route": r["route"], "gold": r["gold"],
                    "clean": r["clean"], "kind": rep.kind, "plan": rep.plans[0]["id"] if rep.plans else None,
                    "outcome": route_outcome(rep, r["route"]), "shown": shown_sections(rep),
                    "old_law_note": any("1 July 2024" in n for n in rep.notes), "ms": ms})
    return out


def summarise_routing(res, subset=lambda x: True):
    rs = [x for x in res if subset(x)]
    n = len(rs)
    c = sum(x["outcome"] == "correct" for x in rs)
    ab = sum(x["outcome"] == "abstained" for x in rs)
    w = sum(x["outcome"] == "wrong" for x in rs)
    answered = n - ab
    return {"n": n, "correct": c, "abstained": ab, "wrong": w, "coverage": answered / n if n else 0,
            "selective_acc": c / answered if answered else 0, "wrong_rate": w / n if n else 0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", action="store_true", help="also evaluate with the LLM router (needs an LLM backend)")
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(os.path.join(ROOT, "data/eval_final.jsonl"), encoding="utf-8")]
    overlap = json.load(open(os.path.join(ROOT, "results/final_overlap.json")))["max_cos_to_older"]
    for r, m in zip(rows, overlap):
        r["clean"] = m < 0.90
    frozen = json.load(open(os.path.join(ROOT, "results/frozen_config.json")))
    print(f"Frozen at {frozen['frozen_at']} | {len(rows)} synthetic questions, {sum(r['clean'] for r in rows)} clean "
          f"(no near-duplicate in older data)\n")
    t0 = time.perf_counter()
    ret = Retriever(load_sections(os.path.join(ROOT, "data/sections.jsonl")), dense_model=DEFAULT_DENSE_MODEL)
    load_s = time.perf_counter() - t0
    report = {"frozen": frozen["frozen_at"], "n": len(rows), "n_clean": sum(r["clean"] for r in rows)}

    # ---------------- 1. retrieval baselines A-D (questions with gold sections) ----------------
    G = [r for r in rows if r["gold"]]
    print(f"## 1. Retrieval: {len(G)} questions with gold sections (any listed section counts as relevant)\n")
    print("| System | Hit@1 (95% CI) | Hit@3 | Hit@5 | MRR@10 | Recall@5 | ms/query (median) |")
    print("|---|---|---|---|---|---|---|")
    per_sys, retr = {}, {}
    for name, system, rw in BASELINES:
        h1, h3, h5, rr, rec, ms, ranks = [], [], [], [], [], [], []
        for r in G:
            t = time.perf_counter()
            res = ret.search(r["query"], k=10, system=system, rewrite=rw)
            ms.append((time.perf_counter() - t) * 1000)
            ids = [ret.sections[h.idx]["id"] for h in res.hits]
            first = next((i + 1 for i, x in enumerate(ids) if x in r["gold"]), None)
            ranks.append(first)
            h1.append(first == 1); h3.append(bool(first and first <= 3)); h5.append(bool(first and first <= 5))
            rr.append(1 / first if first else 0)
            rec.append(len(set(ids[:5]) & set(r["gold"])) / len(r["gold"]))
            if (system, rw) == ("hybrid", True):
                retr[r["query"]] = res
        per_sys[name] = {"hit1": h1, "hit3": h3}
        n = len(G)
        print(f"| {name} | {pct(sum(h1), n)} | {sum(h3)/n:.0%} | {sum(h5)/n:.0%} | {sum(rr)/n:.3f} | {sum(rec)/n:.0%} | "
              f"{statistics.median(ms):.0f} |")
        report.setdefault("retrieval", {})[name] = {"hit1": sum(h1) / n, "hit3": sum(h3) / n, "hit5": sum(h5) / n,
                                                     "mrr10": sum(rr) / n, "recall5": sum(rec) / n,
                                                     "median_ms": statistics.median(ms)}
    print("\nPaired exact McNemar tests on Hit@3 (vs D, the default for situations):")
    for name in per_sys:
        if not name.startswith("D"):
            p = mcnemar_exact(per_sys[name]["hit3"], per_sys["D  Hybrid + glossary expansion"]["hit3"])
            print(f"  {name} vs D: p = {p:.3f}")
            report.setdefault("mcnemar_hit3_vs_D", {})[name] = p

    # ---------------- 2. abstention analysis on D ----------------
    print("\n## 2. Is the abstention gate (BM25 and dense agree in their top 3 + specific query) a good signal?\n")
    rows_d = []
    for r in G:
        res = retr[r["query"]]
        ids = [ret.sections[h.idx]["id"] for h in res.hits]
        top = res.hits[0]
        rows_d.append({"correct1": ids[0] in r["gold"], "correct3": bool(set(ids[:3]) & set(r["gold"])),
                       "gate": Retriever.confident(res),
                       "agree_rank": max(top.bm25_rank or 99, top.dense_rank or 99),
                       "score": top.score, "margin": top.score - (res.hits[1].score if len(res.hits) > 1 else 0)})
    for flag in (True, False):
        sub = [x for x in rows_d if x["gate"] == flag]
        if sub:
            k1, k3 = sum(x["correct1"] for x in sub), sum(x["correct3"] for x in sub)
            print(f"  gate {'ON (answers)' if flag else 'OFF (asks)  '}: n={len(sub):3d} | Hit@1 {pct(k1, len(sub))} | Hit@3 {k3/len(sub):.0%}")
    # correlated errors: do BM25 and dense fail on the same questions, and agree on the same wrong section?
    both_wrong = same_wrong = bm_only = de_only = 0
    for r in G:
        b = ret.search(r["query"], k=1, system="bm25", rewrite=True).hits
        d = ret.search(r["query"], k=1, system="dense", rewrite=False).hits
        bid = ret.sections[b[0].idx]["id"] if b else None
        did = ret.sections[d[0].idx]["id"] if d else None
        bw, dw = bid not in r["gold"], did not in r["gold"]
        both_wrong += bw and dw
        same_wrong += bw and dw and bid == did
        bm_only += bw and not dw
        de_only += dw and not bw
    print(f"\n  Top-1 errors: BM25+expansion wrong and dense right {bm_only}, dense wrong and BM25 right {de_only}, "
          f"both wrong {both_wrong} (of which the SAME wrong section: {same_wrong}).")
    report["errors_correlation"] = {"bm25_only_wrong": bm_only, "dense_only_wrong": de_only, "both_wrong": both_wrong,
                                    "both_wrong_same_section": same_wrong}
    # risk-coverage for three confidence signals
    signals = {"agreement rank (max of BM25/dense rank of top hit; lower = surer)": lambda x: -x["agree_rank"],
               "RRF score of top hit": lambda x: x["score"], "RRF margin top1 - top2": lambda x: x["margin"]}
    curves = {}
    print("\n  Risk–coverage (Hit@1 error rate when answering only the most confident fraction):")
    print("  | Signal | 25% | 50% | 75% | 100% | AURC (lower = better) |")
    print("  |---|---|---|---|---|---|")
    for sname, f in signals.items():
        order = sorted(rows_d, key=lambda x: -f(x))
        errs = np.cumsum([0 if x["correct1"] else 1 for x in order]) / np.arange(1, len(order) + 1)
        cov = np.arange(1, len(order) + 1) / len(order)
        curves[sname] = (cov, errs)
        at = lambda c: errs[max(0, int(round(c * len(order))) - 1)]
        aurc = float(np.mean(errs))
        print(f"  | {sname} | {at(.25):.0%} | {at(.5):.0%} | {at(.75):.0%} | {at(1):.0%} | {aurc:.3f} |")
        report.setdefault("risk_coverage", {})[sname] = {"risk_at": {str(c): float(at(c)) for c in (.25, .5, .75, 1)}, "aurc": aurc}
    rand = 1 - sum(x["correct1"] for x in rows_d) / len(rows_d)
    print(f"  (a useless signal would give {rand:.0%} at every coverage)")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.4, 4))
        for (sname, (cov, errs)), st in zip(curves.items(), ["-", "--", ":"]):
            ax.plot(cov * 100, errs * 100, st, lw=2, label=sname.split(" (")[0])
        ax.axhline(rand * 100, color="grey", lw=1, label="no signal")
        ax.set_xlabel("Coverage: % of questions answered directly"); ax.set_ylabel("Error rate (Hit@1 wrong), %")
        ax.set_title("Risk–coverage of confidence signals (D, frozen test set)", fontsize=10)
        ax.grid(alpha=.3); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(os.path.join(ROOT, "results/risk_coverage.png"), dpi=200)
    except Exception as e:
        print("  (plot skipped:", e, ")")

    # ---------------- 3. full system F ----------------
    configs = [("F1 Rules only (offline)", dict(router=False), "off"),
               ("F2 Rules + trained classifier (offline default)", dict(router=False), "on")]
    llm = None
    if a.llm:
        from src.llm_router import LLMRouter
        from src.playbooks import load_playbooks
        llm = LLMRouter(load_playbooks())
        if llm.available:
            configs.append((f"F3 Rules + LLM router ({llm.name}) + classifier", dict(router=llm), "on"))
        else:
            print("No LLM backend available; skipping F3.")
    print("\n## 3. Full system: situation handling (routing) and the sections it actually shows\n")
    print("| System | Subset | Correct | Asked instead (abstained) | Wrong | Coverage | Accuracy when answering |")
    print("|---|---|---|---|---|---|---|")
    allres = {}
    for name, kw, clf in configs:
        os.environ["SITUATION_CLF"] = clf
        bot = cb.LegalBot(ret, **kw)
        res = run_system(bot, rows, name)
        allres[name] = res
        for sub_name, sub in [("all 150", lambda x: True), ("clean", lambda x: x["clean"])]:
            s = summarise_routing(res, sub)
            print(f"| {name} | {sub_name} | {pct(s['correct'], s['n'])} | {s['abstained']} | {s['wrong']} "
                  f"({s['wrong_rate']:.0%}) | {s['coverage']:.0%} | {s['selective_acc']:.0%} |")
            report.setdefault("system", {}).setdefault(name, {})[sub_name] = s
    os.environ["SITUATION_CLF"] = "on"
    main_name = configs[-1][0] if len(configs) == 3 else configs[1][0]
    res = allres[main_name]
    print(f"\nBy category ({main_name}):")
    print("| Category | n | Correct | Asked | Wrong |")
    print("|---|---|---|---|---|")
    for cat in dict.fromkeys(r["category"] for r in rows):
        s = summarise_routing(res, lambda x, c=cat: x["category"] == c)
        print(f"| {cat} | {s['n']} | {s['correct']} ({s['correct']/s['n']:.0%}) | {s['abstained']} | {s['wrong']} |")
        report.setdefault("by_category", {})[cat] = s
    oos = [x for x in res if x["category"] == "out_of_scope"]
    strict = sum(x["kind"] == "out_of_scope" for x in oos)
    print(f"\nOut-of-scope rejection: strict (says 'civil / other law, go here') {pct(strict, len(oos))}; "
          f"lenient (no criminal guide shown) {pct(sum(x['outcome'] == 'correct' for x in oos), len(oos))}")
    od = [x for x in res if x["category"] == "old_date"]
    print(f"Old-date notice shown for pre-July-2024 events: {pct(sum(x['old_law_note'] for x in od), len(od))}; "
          f"false notices on other questions: {sum(x['old_law_note'] for x in res if x['category'] != 'old_date')}")
    # sections actually shown to the user (F), questions with gold sections
    gres = [x for x in res if x["gold"]]
    any_hit = sum(bool(set(x["shown"]) & set(x["gold"])) for x in gres)
    first_hit = sum(bool(x["shown"]) and x["shown"][0] in x["gold"] for x in gres)
    none_shown = sum(not x["shown"] for x in gres)
    avg = statistics.mean(len(x["shown"]) for x in gres)
    print(f"\nSections shown by the full system ({len(gres)} questions with gold): a gold section among those shown "
          f"{pct(any_hit, len(gres))}; first shown is gold {pct(first_hit, len(gres))}; nothing shown (asked) {none_shown}; "
          f"average shown {avg:.1f}")
    report["shown"] = {"any_gold": any_hit / len(gres), "first_gold": first_hit / len(gres), "none_shown": none_shown,
                       "avg_shown": avg, "n": len(gres)}
    # latency + memory (offline, one process, CPU)
    lat = sorted(x["ms"] for x in allres[configs[1][0]])
    p95 = lat[int(0.95 * len(lat)) - 1]
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 if sys.platform != "darwin" else 1024 * 1024)
    print(f"\nLatency per message (F2, offline, CPU): mean {statistics.mean(lat):.0f} ms, median {statistics.median(lat):.0f} ms, "
          f"p95 {p95:.0f} ms, max {lat[-1]:.0f} ms | start-up (corpus + model + embeddings) {load_s:.1f} s | peak memory {rss:.0f} MB")
    report["latency_ms"] = {"mean": statistics.mean(lat), "median": statistics.median(lat), "p95": p95, "max": lat[-1],
                            "startup_s": load_s, "peak_rss_mb": rss}
    if llm is not None and len(configs) == 3:
        l3 = sorted(x["ms"] for x in allres[configs[2][0]])
        print(f"Latency with LLM router: median {statistics.median(l3):.0f} ms, p95 {l3[int(.95*len(l3))-1]:.0f} ms; "
              f"LLM errors: {len(llm.errors)}")
    report["rows"] = {k: v for k, v in allres.items()}
    json.dump(report, open(os.path.join(ROOT, "results/final_eval.json"), "w"), indent=1)
    print("\nwrote results/final_eval.json, results/risk_coverage.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
