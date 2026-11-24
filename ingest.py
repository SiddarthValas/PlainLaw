"""Build data/<ACT>.jsonl from an official Act PDF (India Code layout) and its CAPT Bhopal comparison table.

  python ingest.py --act BNSS --pdf data/source/BNSS_Act46_2023.pdf --compare data/source/comparison_BNSS_to_CrPC.pdf
  python ingest.py --act BNS  --compare data/source/comparison_BNS_to_IPC.pdf --map-only     # mapping only
  python ingest.py --act BNS  --pdf data/source/BNS_Act45_2023.pdf --compare data/source/comparison_BNS_to_IPC.pdf
  python ingest.py --merge                                   # data/BNS.jsonl + data/BNSS.jsonl -> data/sections.jsonl

Section text comes only from the Act PDF. The comparison table supplies the old IPC/CrPC numbers and the
"what changed" note. Everything is validated: sequential numbering, titles against the Act's own contents list.
"""
import argparse
import difflib
import json
import os
import re
import sys

import pdfplumber

OLD_ACT = {"BNS": "IPC", "BNSS": "CrPC"}
# vertical column edges (x) of the comparison tables: section | subject | old section | summary
BOUNDS = {"BNSS": (37, 82, 235, 280, 582), "BNS": (45, 104, 301, 344, 580)}

SEC = re.compile(r"^(\d{1,3})\.[\s—–]*(\S.*)$")
TITLE_END = re.compile(r"\.?\s*[—–]+\s*")
SCHEDULE = re.compile(r"^((THE\s+)?(FIRST\s+|SECOND\s+)?SCHEDULE|STATEMENT OF OBJECTS)\b")
CHAPTER = re.compile(r"^C\s?HAPTER\s+[IVXLC]+\b")
SUBHEAD = re.compile(r"^[A-Z]\s?\.\s?[—–]")
ENDS_LIKE_TEXT = re.compile(r"[.;:,—)”\"'?]\s*$")


# ----------------------------------------------------------------------------- Act text
# per-Act page geometry: lowest y of body text (page numbers sit below it), watermark size
PAGE = {"BNSS": {"bottom": 722, "wm": 20}, "BNS": {"bottom": 770, "wm": 99}}
ACT = {"name": "BNSS"}


def keep_char(c) -> bool:
    g = PAGE[ACT["name"]]
    if c["size"] > g["wm"]:                 # diagonal "India Code" watermark
        return False
    if c["top"] > g["bottom"] or c["top"] < 45:     # page number / margins
        return False
    if c["size"] < 8:                       # footnote reference marks
        return False
    if c["size"] <= 9.5 and c["top"] > 650:  # footnote text
        return False
    return True


def page_lines(page) -> list[str]:
    text = page.filter(lambda o: o["object_type"] != "char" or keep_char(o)).extract_text(x_tolerance=2, y_tolerance=3) or ""
    return [ln.strip() for ln in text.split("\n") if ln.strip()]


def read_act(pdf_path: str, cache: str | None):
    if cache and os.path.exists(cache):
        d = json.load(open(cache, encoding="utf-8"))
        return d["toc"], d["body"]
    toc, body, started = [], [], False
    with pdfplumber.open(pdf_path) as pdf:
        for pg in pdf.pages:
            lines = page_lines(pg)
            if not started and any(re.match(r"^1\.\s+.*\.\s*[—–]+", ln) for ln in lines):
                started = True
            (body if started else toc).extend(lines)
    if cache:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        json.dump({"toc": toc, "body": body}, open(cache, "w", encoding="utf-8"))
    return toc, body


def parse_sections(lines: list[str], last: int):
    secs, cur, expected, i = [], None, 1, 0
    while i < len(lines):
        ln = lines[i]
        m = SEC.match(ln)
        if m and int(m.group(1)) == expected:
            buf, j = m.group(2), i
            while not TITLE_END.search(buf) and j < i + 2 and j + 1 < len(lines):
                j += 1
                buf += " " + lines[j]
            t = TITLE_END.search(buf)
            if t:
                rest = buf[t.end():].strip()
                cur = {"section": str(expected), "title": buf[:t.start()].strip(), "lines": [rest] if rest else []}
                secs.append(cur)
                expected += 1
                i = j + 1
                continue
        if cur is not None:
            if expected - 1 == last and SCHEDULE.match(ln):
                break
            cur["lines"].append(ln)
        i += 1
    return secs


def clean_tail(lines: list[str]):
    """Drop chapter headings and sub-headings that trail a section."""
    dropped: list[str] = []
    for k in range(len(lines) - 1, max(len(lines) - 7, -1), -1):
        if CHAPTER.match(lines[k]):
            dropped = lines[k:]
            lines = lines[:k]
            break
    while lines and (SUBHEAD.match(lines[-1]) or (len(lines[-1]) < 80 and not ENDS_LIKE_TEXT.search(lines[-1]))):
        dropped.insert(0, lines.pop())
    return lines, dropped


# ----------------------------------------------------------------------------- comparison table
def read_table(path: str, act: str):
    x = BOUNDS[act]
    rows, cur = [], None
    with pdfplumber.open(path) as pdf:
        for pg in pdf.pages:
            words = [w for w in pg.extract_words() if w["top"] < 705 and w["x0"] >= x[0] - 3]
            words.sort(key=lambda w: (round(w["top"]), w["x0"]))
            lines: list[tuple[float, list]] = []
            for w in words:
                if lines and abs(w["top"] - lines[-1][0]) <= 2.5:
                    lines[-1][1].append(w)
                else:
                    lines.append((w["top"], [w]))
            for _, ws in lines:
                cells = ["", "", "", ""]
                for w in sorted(ws, key=lambda w: w["x0"]):
                    ci = sum(w["x0"] >= b - 3 for b in x[1:4])
                    cells[ci] = (cells[ci] + " " + w["text"]).strip()
                if re.match(r"^\d", cells[0]) or re.match(r"^Explanati", cells[0]):
                    cur = list(cells)
                    rows.append(cur)
                elif cur is not None:
                    for k in range(4):
                        if cells[k]:
                            cur[k] = (cur[k] + " " + cells[k]).strip()
    return rows


def old_numbers(cell: str) -> list[str]:
    out = []
    for piece in re.split(r"\s*(?:&|\band\b|,)\s*", cell):
        m = re.match(r"^(\d+[A-Z]?)\b", piece.strip())
        if m:
            out.append(m.group(1).upper())
    return out


def build_mapping(rows, act: str):
    """base section -> {subject, old: [numbers], change}"""
    mp: dict[int, dict] = {}
    for first, subject, old, summary in rows:
        m = re.match(r"^(\d+)", first) or re.search(r"(\d+)", first)
        if not m:
            continue
        b = int(m.group(1))
        e = mp.setdefault(b, {"subject": "", "old": [], "change": "", "exact": False})
        for n in old_numbers(old):
            if n not in e["old"]:
                e["old"].append(n)
        if first.strip() == str(b) or not e["exact"] and not e["change"]:
            if first.strip() == str(b):
                e["subject"], e["change"], e["exact"] = subject, summary, True
            elif not e["subject"]:
                e["subject"], e["change"] = subject, summary
    return mp


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


# ----------------------------------------------------------------------------- main
def run(a) -> int:
    act = a.act.upper()
    rows = read_table(a.compare, act) if a.compare else []
    mp = build_mapping(rows, act) if rows else {}
    last = a.last or (max(mp) if mp else 0)
    if mp:
        missing = [n for n in range(1, max(mp) + 1) if n not in mp]
        print(f"[table] {len(rows)} rows, {len(mp)} sections mapped, max {max(mp)}, missing: {missing or 'none'}")

    os.makedirs("data", exist_ok=True)
    old = OLD_ACT[act]

    def mapping_fields(n: int) -> dict:
        e = mp.get(n)
        if not e:
            return {"replaces": "", "change": ""}
        return {"replaces": ", ".join(f"{old} {x}" for x in e["old"]), "change": e["change"][:400]}

    if a.map_only:
        with open(f"data/{act}_map.jsonl", "w", encoding="utf-8") as f:
            for n in sorted(mp):
                f.write(json.dumps({"act": act, "section": str(n), "title": mp[n]["subject"], **mapping_fields(n)}, ensure_ascii=False) + "\n")
        print(f"wrote data/{act}_map.jsonl ({len(mp)} sections)")
        return 0

    ACT["name"] = act
    toc, body = read_act(a.pdf, a.cache)
    secs = parse_sections(body, last)
    problems = []
    if len(secs) != last:
        problems.append(f"parsed {len(secs)} sections, expected {last}")
    toc_norm = norm(" ".join(toc))
    out, dropped_log, title_flags = [], [], []
    for s in secs:
        n = int(s["section"])
        lines, dropped = clean_tail(s["lines"])
        if dropped:
            dropped_log.append((n, dropped))
        text = "\n".join(lines).strip()
        if len(text) < 15:
            problems.append(f"section {n} has almost no text")
        if norm(f"{n} {s['title']}") not in toc_norm:
            title_flags.append((n, s["title"], "not found in the Act's contents list"))
        sub = mp.get(n, {}).get("subject", "")
        if sub and difflib.SequenceMatcher(None, norm(sub), norm(s["title"])).ratio() < 0.6:
            title_flags.append((n, s["title"], f"differs from table subject: {sub}"))
        title = s["title"].replace("summonscases", "summons-cases")  # typo in the official PDF's headings (s.282, s.309)
        out.append({"id": f"{act}-{n}", "act": act, "section": str(n), "title": title, "text": text, **mapping_fields(n)})

    with open(f"data/{act}.jsonl", "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[act] wrote data/{act}.jsonl: {len(out)} sections; chars: min {min(len(r['text']) for r in out)}, "
          f"median {sorted(len(r['text']) for r in out)[len(out)//2]}, max {max(len(r['text']) for r in out)}")
    print(f"[check] trailing headings removed from {len(dropped_log)} sections")
    for n, d in dropped_log[:int(a.show)]:
        print(f"    s.{n}: {d}")
    print(f"[check] title flags: {len(title_flags)}")
    for t in title_flags[:int(a.show)]:
        print("    ", t)
    for p in problems:
        print("[PROBLEM]", p)
    return 1 if problems else 0


def merge() -> int:
    n = 0
    with open("data/sections.jsonl", "w", encoding="utf-8") as out:
        for act in ("BNS", "BNSS"):
            p = f"data/{act}.jsonl"
            if os.path.exists(p):
                for line in open(p, encoding="utf-8"):
                    if line.strip():
                        out.write(line)
                        n += 1
                print(f"merged {p}")
            else:
                print(f"skipped {p} (not found)")
    print(f"data/sections.jsonl: {n} sections")
    return 0 if n else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--act", choices=["BNS", "BNSS"])
    ap.add_argument("--pdf")
    ap.add_argument("--compare")
    ap.add_argument("--last", type=int, default=0)
    ap.add_argument("--map-only", action="store_true")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--cache", default=None, help="cache extracted page text here (speeds up re-runs)")
    ap.add_argument("--show", default=15, help="how many flagged items to print")
    args = ap.parse_args()
    if args.merge:
        sys.exit(merge())
    if not args.act or (not args.pdf and not args.map_only):
        ap.error("need --act and --pdf (or --map-only)")
    sys.exit(run(args))
