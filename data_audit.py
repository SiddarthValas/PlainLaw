"""Integrity audit of the statute corpus: completeness, numbering, text, old-code mappings, provenance.

    python3 data_audit.py        # prints a report and writes data/VERSION.json (source hashes, counts, dates)

Checks only; it never edits the legal text. Any correction must come from the source PDFs in data/source/.
"""
import collections, hashlib, json, os, re, sys

ROOT = os.path.dirname(os.path.abspath(__file__))
EXPECTED = {"BNS": 358, "BNSS": 531}          # number of sections in each Act as enacted


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    S = [json.loads(l) for l in open(os.path.join(ROOT, "data/sections.jsonl"), encoding="utf-8")]
    problems = []
    ids = collections.Counter(s["id"] for s in S)
    problems += [f"duplicate id {i}" for i, c in ids.items() if c > 1]
    report = {}
    for act, n in EXPECTED.items():
        nums = [s["section"] for s in S if s["act"] == act]
        ints = {int(re.match(r"\d+", x).group()) for x in nums}
        missing = sorted(set(range(1, n + 1)) - ints)
        report[act] = {"sections": len(nums), "expected": n, "missing": missing}
        if missing or len(nums) != n:
            problems.append(f"{act}: {len(nums)}/{n} sections, missing {missing[:10]}")
    for s in S:
        if not s["title"].strip():
            problems.append(f"{s['id']}: empty title")
        if len(s["text"]) < 40:
            problems.append(f"{s['id']}: text only {len(s['text'])} characters")
        if re.search(r"Subs\. by|Ins\. by|w\.e\.f\.|GAZETTE OF INDIA", s["text"]):
            problems.append(f"{s['id']}: amendment/gazette note inside section text")
    # mappings: one new section can replace several old ones, and one old section can be split
    old_to_new = collections.defaultdict(list)
    for s in S:
        for a, n in re.findall(r"\b(IPC|CrPC)\s*(\d+[A-Z]?)", s.get("replaces", ""), re.I):
            old_to_new[(a.upper(), n)].append(s["id"])
    many_old = sum(1 for s in S if len(re.findall(r"\d+[A-Z]*", s.get("replaces", ""))) > 1)
    split_old = {f"{a} {n}": v for (a, n), v in old_to_new.items() if len(v) > 1}
    no_old = [s["id"] for s in S if not s.get("replaces")]
    src = os.path.join(ROOT, "data", "source")
    version = {
        "built_from": {f: {"sha256": sha(os.path.join(src, f))} for f in sorted(os.listdir(src)) if f.endswith(".pdf")},
        "text_versions": {"BNS": "Act 45 of 2023, India Code text as on 6 October 2025",
                          "BNSS": "Act 46 of 2023, as enacted"},
        "old_numbers_from": "CAPT Bhopal comparison tables (BNS-IPC, BNSS-CrPC)",
        "in_force_from": "2024-07-01",
        "counts": {a: r["sections"] for a, r in report.items()},
        "sections_jsonl_sha256": sha(os.path.join(ROOT, "data/sections.jsonl")),
        "mapping": {"new_sections_with_old_number": len(S) - len(no_old), "new_sections_without_old_number": len(no_old),
                    "new_sections_replacing_several_old": many_old, "old_sections_split_into_several_new": len(split_old)},
        "known_gaps": ["BNS 106(2) (hit-and-run) not brought into force as of mid-2026; shown with a note",
                       "Amendments after the text dates above are not included",
                       "Mappings come from the comparison tables and are not one-to-one everywhere"],
    }
    json.dump(version, open(os.path.join(ROOT, "data/VERSION.json"), "w"), indent=2)
    print("Statute corpus audit")
    for act, r in report.items():
        print(f"  {act}: {r['sections']}/{r['expected']} sections, missing: {r['missing'] or 'none'}")
    print(f"  duplicate ids: {sum(c > 1 for c in ids.values())}")
    print(f"  sections with an old IPC/CrPC number: {len(S) - len(no_old)}; without (new provisions): {len(no_old)}")
    print(f"  new sections replacing several old ones: {many_old}; old sections split into several new: {len(split_old)}")
    for k, v in split_old.items():
        print(f"    {k} -> {', '.join(v)}")
    print(f"  problems: {len(problems)}" + ("".join(f"\n    - {p}" for p in problems)))
    print("  wrote data/VERSION.json")
    return 1 if any("missing" in p or "duplicate" in p for p in problems) else 0


if __name__ == "__main__":
    sys.exit(main())
