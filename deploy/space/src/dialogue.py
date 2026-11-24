"""Conversation manager: turns the single-shot bot into an ongoing, context-aware legal information assistant.

The browser keeps a small JSON `state` and sends it back with every message (the server stays stateless):
    topic     the situation being discussed (playbook id) and its title
    story     the user's own words describing it (so later details can be added to it)
    laws      section ids shown so far, in order (for "section 75", "the first one", "it")
    focus     the section last explained (for "what is the punishment for it?")
    pending   a question the assistant asked and is waiting for ("safety")
    facts     what the user told us: safe (yes/no), when ("yesterday")

Each new message is read in this order:
  1. crisis / greeting / thanks / non-English          (src/safety.py, unchanged)
  2. an answer to the assistant's own question          ("Yes, I'm safe now. It happened yesterday.")
  3. a question about a section                         ("What does Section 75 mean?", "explain the first one")
  4. a follow-up question about the same situation      ("What if I have no witnesses?", "Can I report it online?")
  5. a new situation, or more detail on the current one  (the existing routing in src/chatbot.py)

Everything legal that is shown still comes from the corpus (section text) or from the hand-written guides and the
follow-up answers below, whose citations are checked by tests/test_dialogue.py against the corpus.
"""
from __future__ import annotations

import re

from . import safety
from .chatbot import Reply, normalize
from .playbooks import match as match_playbooks
from .llm_chat import GroundedResponder
from .intent import is_definition_question

SENSITIVE = {"women_harassment", "sexual_assault", "assault", "domestic", "threats", "followed", "confinement",
             "missing", "custodial", "transport", "accident"}
WOMEN_PROVISO = {"BNS-64", "BNS-65", "BNS-66", "BNS-67", "BNS-68", "BNS-69", "BNS-70", "BNS-71", "BNS-74", "BNS-75",
                 "BNS-76", "BNS-77", "BNS-78", "BNS-79", "BNS-124"}     # BNSS 173(1), first proviso
IDENTITY_PROTECTED = {"BNS-64", "BNS-65", "BNS-66", "BNS-67", "BNS-68", "BNS-69", "BNS-70", "BNS-71"}   # BNS 72

EMPATHY = {
    "women_harassment": "I'm sorry that happened. No one should have to deal with that.",
    "sexual_assault": "I'm so sorry this happened to you. It is not your fault.",
    "assault": "I'm sorry you were hurt.",
    "domestic": "I'm sorry you're going through this. You deserve to be safe at home.",
    "threats": "That sounds frightening. I'm sorry you're dealing with it.",
    "followed": "That sounds frightening. I'm sorry you're dealing with it.",
    "confinement": "That sounds very distressing.",
    "missing": "I'm sorry, that must be very worrying.",
    "custodial": "I'm sorry. That is serious, and there are protections for people in custody.",
    "accident": "I'm sorry to hear about the accident.",
    "theft": "I'm sorry, that's a horrible thing to deal with.",
    "fraud": "I'm sorry, losing money like that is really stressful.",
    "fire": "I'm sorry about the damage.",
    "transport": "I'm sorry, that sounds like a bad experience.",
    "trespass": "That sounds unsettling.",
}
AREA = {   # "... may fall under provisions of the BNS on <area>, depending on the circumstances"
    "women_harassment": "sexual harassment and outraging a woman's modesty", "sexual_assault": "sexual offences",
    "assault": "hurt and assault", "domestic": "cruelty by a husband or his relatives, and hurt",
    "threats": "criminal intimidation and extortion", "followed": "stalking", "confinement": "wrongful restraint and confinement",
    "missing": "kidnapping and missing persons", "custodial": "hurt and the rights of arrested persons",
    "transport": "extortion, intimidation and restraint", "accident": "rash driving and endangering life",
}
SAFE_YES = re.compile(r"\b(?:yes|yeah|yep|i'?m (?:safe|ok|okay|fine|alright)|i am (?:safe|ok|okay|fine)|safe now|now safe|"
                      r"at home|home now|it'?s over|not anymore|no longer|it (?:was|happened) (?:yesterday|earlier|last)|"
                      r"(?:he|she|they) (?:left|is gone|are gone|went away))\b", re.I)
SAFE_NO = re.compile(r"\b(?:no+|not (?:really )?safe|unsafe|still (?:here|there|following|outside|with me|happening)|"
                     r"right now|happening now|help|scared|he'?s (?:here|outside|still)|she'?s (?:here|outside|still)|"
                     r"they'?re (?:here|outside|still)|following me)\b", re.I)
WHEN = re.compile(r"\b(?:today|yesterday|last (?:night|week|month|year)|this (?:morning|evening|afternoon)|"
                  r"(?:a|one|two|three|four|five|six|\d+) (?:day|week|month|year)s? (?:ago|back)|a while ago|recently|"
                  r"(?:on|in) (?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*(?: \d{4})?)\b", re.I)

ADDITIVE = re.compile(r"^(?:and |also |plus |then |after that |later |moreover )|\b(?:also|as well|too|then he|then she|then they|"
                      r"after that|on top of (?:that|it))\b", re.I)
VAGUE = re.compile(r"^\s*(?:hi|hello|hey)?[\s,]*(?:i |we )?(?:need|want) (?:some |your )?help\W*$|^\s*(?:can you |please )?help(?: me| us)?(?: please)?\W*$|"
                   r"^\s*i have a (?:problem|question|doubt|legal (?:problem|issue|question))\W*$", re.I)
ACK = re.compile(r"^\s*(?:i'?m|i am|we'?re|we are|she'?s|he'?s)?\s*(?:ok|okay|fine|alright|safe|good)(?: now)?\W*$|^\s*(?:ok|okay|got it|i see|understood|hmm+)\W*$", re.I)
PRONOUN_START = re.compile(r"^\s*(?:and |also |then |but )?(?:he|she|they|it|him|her|them|the (?:man|guy|person|driver|thief|seller))\b", re.I)
HYPOTHETICAL = re.compile(r"^(?:what|and|but)?\s*(?:if|about)\b|^(?:suppose|hypothetically|is it (?:a crime|illegal|an offence)|can (?:someone|a person|i be))", re.I)

# ---- section references ----
_REF = re.compile(r"\b(?:(BNSS|BNS|IPC|CrPC)\s*(?:section|sec\.?|s\.?)?\s*(\d+[A-Z]?)|"
                  r"(?:section|sec\.?|s\.)\s*(\d+[A-Z]?)(?:\s*(?:of\s+(?:the\s+)?)?(BNSS|BNS|IPC|CrPC))?)\b", re.I)
_ORDINAL = re.compile(r"\b(?:the )?(first|1st|second|2nd|third|3rd|last) (?:one|section|law|card)\b", re.I)
_DEICTIC = re.compile(r"\b(?:this|that|the) (?:section|law|provision|one)\b|\b(?:it|this|that)\b", re.I)
_EXPLAIN = re.compile(r"\b(?:mean|means|meaning|explain|explanation|in simple|simple (?:words|language|terms)|what does|"
                      r"what is (?:in )?(?:it|this|that)|tell me (?:more )?about|break (?:it )?down|elaborate|say)\b", re.I)
_PUNISH = re.compile(r"\b(?:punish\w*|sentence|jail|prison|how (?:many|much) (?:years|time)|what (?:will|can) happen to "
                     r"(?:him|her|them|the (?:man|person|accused)))\b", re.I)

# ---- follow-up questions about the current situation ----
FOLLOW = [
    ("insurance", r"\binsur\w*|\bclaim (?:for|from) (?:the )?(?:damage|repair|accident)\b|\bmotor accident claims?\b"),
    ("workplace", r"\b(?:workplace|office|at work|colleague|co-?worker|manager|boss|employer|company)\b.{0,60}\b(?:law|act|committee|posh|separate|complain\w*|hr)\b|\b(?:law|act|committee|complain\w*|rule)\b.{0,40}\b(?:workplace|at work|office|job)\b|\bposh\b|\binternal (?:complaints? )?committee\b"),
    ("arrest_q", r"\b(?:will|would|can|do|does|should|must) (?:the )?(?:police|they|cops?)\b.{0,15}\barrest\b|\bget (?:him|her|them) arrested\b|\bwill (?:he|she|they) be arrested\b|\barrest (?:him|her|them) (?:immediately|now|right away|at once)\b"),
    ("which_law", r"\b(?:wh?at|which|under)\b.{0,25}\b(?:sec(?:tion)?s?|laws?|acts?|provisions?|ipc|bns)\b(?!\s*\d)|\bis (?:it|this|that) (?:a )?(?:crime|illegal|an offen[cs]e|punishable|legal)\b|\bwhat (?:crime|offen[cs]e)\b"),
    ("what_do", r"^\s*(?:so )?(?:what|wat|wht) (?:should|can|do|shall|must) (?:i|we|she|he|they) do\b|\bwhat (?:to|do i) do (?:first|now|next)\b|\bwhat (?:should|can) (?:i|we) do (?:first|now|next)\b|^\s*(?:what|wat) (?:now|next)\??\s*$|\bwhere do i start\b|\bfirst step\b"),
    ("no_witness", r"\b(?:no|without|don'?t have (?:any )?|not have (?:any )?|any) (?:witness\w*|proof|evidence)\b|\bnobody (?:saw|was there)\b|\bno one (?:saw|was there)\b|\bonly my word\b"),
    ("police_refuse", r"\b(?:police|polic|polis\w*|pulis\w*|they|cops?|officer|sho|thana)\b.{0,30}\b(?:refus\w*|don'?t|won'?t|didn'?t|not)\b.{0,20}\b(?:register|registr|file|write|take|listen|accept|help)\w*|\bif (?:the )?(?:police|they) (?:refuse|say no)\b"),
    ("woman_officer", r"\b(?:woman|women|female|lady|lady) (?:officer|police|constable|cop)\b"),
    ("online", r"\b(?:online|e-?fir|by email|email|website|app|without going|from home|phone)\b.{0,40}\b(?:report|complain\w*|file|fir|register)\b|\b(?:report|complain\w*|file|register)\b.{0,40}\b(?:online|e-?fir|by email|from home)\b"),
    ("where", r"\b(?:which|what|any) police station\b|\bwhere (?:do|should|can) i (?:go|report|complain|file)\b|\b(?:different|another|other) (?:city|area|state|station)\b"),
    ("what_next", r"\bwhat (?:happens|will happen) (?:next|after|then)\b|\bafter (?:the |i (?:file|report) )?(?:fir|complaint)\b|\bhow long (?:will|does) (?:it|the case|investigation)\b|\bnext steps?\b"),
    ("how_report", r"\bhow (?:do|can|should) i (?:report|complain|file|lodge|register)\b|\b(?:can|should|do) (?:i|we|she|he) (?:complain|report|go to the police|file)\b|\bwho (?:can|should) (?:file|report|complain)\b|\b(?:file|report|complain) (?:it )?(?:herself|himself|myself|ourselves|on (?:her|his) behalf)\b|\b(?:file|lodge|register|write) (?:an? |the )?(?:fir|complaint|case)\b|\bwhat (?:do|should) i (?:say|write|tell)\b|\bgo to the police\b|\bshould i (?:report|complain|go)\b"),
    ("delay", r"\b(?:too late|late to|delay\w*|time limit|deadline|how long do i have|after (?:a )?(?:few|many|some) (?:days|weeks|months)|still report)\b"),
    ("lawyer_cost", r"\b(?:lawyer|advocate|legal aid|free (?:legal|help|lawyer)|cost|fees?|pay (?:for|anything)|afford|money to file)\b"),
    ("privacy", r"\b(?:name|identity|privacy|private|confidential|secret|anonymous\w*|my family (?:know|find)|people (?:know|find out)|media|news)\b"),
    ("compensation", r"\b(?:compensation|money back|get (?:my )?money|damages|claim)\b"),
    ("safety", r"\b(?:protect\w*|safe(?:ty)?|scared|afraid|retaliat\w*|threaten\w* me (?:again|if))\b"),
    ("evidence", r"\b(?:what|which) (?:evidence|proof|documents?|papers)\b|\bcollect (?:evidence|proof)\b|\bcctv\b|\bscreenshots?\b|\bdelet\w*\b|\b(?:lose|lost) (?:the )?(?:messages?|proof|evidence|chats?)\b"),
]
FOLLOW = [(n, re.compile(p, re.I)) for n, p in FOLLOW]


def _lc(t: str) -> str:
    """Lower-case the first letter unless the word is an acronym (FIR, CCTV)."""
    return t if len(t) > 1 and t[1].isupper() else t[:1].lower() + t[1:]


def _simplify_punishment(s: str) -> str:
    s = " ".join(s.split())
    reps = [
        (r"imprisonment of either description", "prison"), (r"rigorous imprisonment", "rigorous imprisonment (with hard labour)"),
        (r"simple imprisonment", "simple imprisonment"),
        (r"for a term which shall not be less than ([\w\s-]+?) but which may extend to ([\w\s-]+?)(?=,| and| or|\.)", r"for \1 to \2"),
        (r"for a term which may extend to ([\w\s-]+?)(?=,| and| or|\.)", r"for up to \1"),
        (r"imprisonment for life, which shall mean imprisonment for the remainder of that person's natural life", "life imprisonment for the rest of the person's life"),
        (r"or with fine, or with both", "or a fine, or both"), (r"and shall also be liable to fine", "and a fine"),
        (r"shall be punished with", "can be punished with"), (r"shall also be liable to fine", "a fine as well"),
    ]
    for a, b in reps:
        s = re.sub(a, b, s, flags=re.I)
    return s


def explain_section(sec: dict, why: str | None = None) -> dict:
    """A plain-language explanation built only from the section's own text (no generation): what it covers
    (its clauses), who it applies to, and the punishment, with the old IPC/CrPC number."""
    text = " ".join(sec["text"].split())
    points, punish = [], []
    # the operative text only: Explanations and Illustrations are examples, not what the section covers
    main = re.split(r"\s(?:Explanation\s*\d*\s*\.?[—-]|Illustrations?\b)", text, maxsplit=1)[0]
    # clauses (i), (ii) ... or (a), (b) ... in the first sub-section: what the section covers
    first = re.split(r"\s\(2\)\s", main, maxsplit=1)[0]
    clauses = re.findall(r"\((?:i|ii|iii|iv|v|vi|vii|viii|ix|x|[a-h])\)\s+(.+?)(?=;|\s\((?:i|ii|iii|iv|v|vi|vii|viii|ix|x|[a-h])\)\s|,\s*shall|\.\s|$)", first)
    for c in clauses[:8]:
        c = re.sub(r"\s*(?:or|and)$", "", c.strip(" ;,:—-")).strip()
        if 3 <= len(c.split()) <= 40:
            points.append(c[0].upper() + c[1:])
    if not points:   # a definition ("Whoever ..., is said to commit theft.") is the clearest summary
        lead = re.sub(r"^\(\d+\)\s*", "", re.split(r"(?<=\.)\s+(?=\(\d+\)|[A-Z])", main, maxsplit=1)[0])
        if not re.search(r"\bpunished with\b", lead) and len(lead.split()) <= 130:
            points.append(re.sub(r"^Whoever\b", "Anyone who", lead))
    for sent in re.split(r"(?<=\.)\s+(?=\(\d+\)|[A-Z])", text):
        if re.search(r"\bpunished with\b|\bliable to fine\b", sent):
            body = re.sub(r"^\(\d+\)\s*", "", sent)
            body = re.split(r":\s*Provided\b", body)[0]
            body = re.sub(r"^Whoever\b", "Anyone who", body)
            # "Any man who commits the offence specified in clause (i) or clause (ii) of sub-section (1) shall ..." -> "For (i) or (ii): ..."
            m = re.match(r".*?specified in ((?:clause \([ivx]+\)(?: or |, )?)+) of sub-section \(\d+\)\s*(shall.*)", body)
            if m:
                body = "For " + re.sub(r"clause ", "", m.group(1)).strip() + ": " + m.group(2)
            punish.append(_simplify_punishment(body).rstrip(".") + ".")
    if not points and not punish:
        lead = re.split(r"(?<=[.;:])\s", text, maxsplit=1)[0]
        points.append(lead[:300])
    return {"id": sec["id"], "act": sec["act"], "section": sec["section"], "title": sec["title"],
            "replaces": sec.get("replaces"), "summary": why, "points": points, "punishment": punish[:4],
            "text": sec["text"]}


class Conversation:
    def __init__(self, bot, responder: GroundedResponder | None | bool = None):
        self.bot = bot
        # the LLM that writes free-form replies (None = auto-detect from LLM_BASE/LLM_KEY or Ollama; False = off)
        self.llm = (GroundedResponder() if responder is None else responder) or None
        if self.llm is not None and not self.llm.available:
            self.llm = None
        self.by_id = {s["id"]: s for s in bot.r.sections}
        self.pb = {p.id: p for p in bot.playbooks}

    # ---------------- helpers ----------------
    def _why(self, sid: str, topic: str | None) -> str | None:
        pb = self.pb.get(topic or "")
        for law in (pb.laws if pb else []):
            if law["id"] == sid:
                return law["why"]
        for p in self.bot.playbooks:
            for law in p.laws:
                if law["id"] == sid:
                    return law["why"]
        return None

    def _chips(self, st: dict) -> list[str]:
        chips = []
        laws = [l for l in st.get("laws", []) if l.startswith("BNS-")] or st.get("laws", [])
        if laws:
            a, n = laws[0].split("-")
            chips.append(f"What does {a} {n} mean?")
        chips += ["What if I don't have any witnesses?", "How do I report it?"]
        if any(l in WOMEN_PROVISO for l in st.get("laws", [])):
            chips.append("Can a woman officer record my statement?")
        else:
            chips.append("What if the police refuse?")
        return chips[:4]

    def _resolve_ref(self, msg: str, st: dict) -> str | None:
        shown = st.get("laws", [])
        m = _REF.search(msg)
        bare = re.fullmatch(r"\s*(\d+[A-Za-z]?)\s*\??\s*", msg)
        if m or bare:
            act = ((m.group(1) or m.group(4) or "") if m else "").upper()
            num = ((m.group(2) or m.group(3)) if m else bare.group(1)).upper()
            if act in ("IPC", "CRPC"):
                return None   # old numbers: handled by the normal lookup (IPC 420 -> BNS 318)
            if act:
                sid = f"{act}-{num}"
                return sid if sid in self.by_id else None
            for sid in shown:   # "section 75" while BNS 75 is on screen
                if sid.split("-")[1] == num:
                    return sid
            if shown and st.get("topic"):   # in a conversation about a crime, an unqualified number means BNS first
                for act in ("BNS", "BNSS"):
                    if f"{act}-{num}" in self.by_id and any(s.startswith(act + "-") for s in shown):
                        return f"{act}-{num}"
            return None
        m = _ORDINAL.search(msg)
        if m and shown:
            k = {"first": 0, "1st": 0, "second": 1, "2nd": 1, "third": 2, "3rd": 2, "last": -1}[m.group(1).lower()]
            try:
                return shown[k]
            except IndexError:
                return None
        if _DEICTIC.search(msg) and (st.get("focus") or shown):
            return st.get("focus") or shown[0]
        return None

    def _plan_reply(self, base: Reply, st: dict, say: list[str], collapsed=False) -> Reply:
        base.say = say
        base.collapsed = collapsed
        base.chips = self._chips(st)
        return base

    def _remember_plan(self, st: dict, r: Reply, story: str):
        if r.plans and r.plans[0]["id"] != "generic":
            p = r.plans[0]
            st["topic"], st["title"] = p["id"], p["title"]
            st["laws"] = list(dict.fromkeys([l["id"] for pl in r.plans for l in pl["laws"]]))
        elif r.plans:
            st["laws"] = list(dict.fromkeys([l["id"] for pl in r.plans for l in pl["laws"]]))
        elif r.hits:
            st["laws"] = [h["id"] for h in r.hits]
        st["story"] = story
        st.setdefault("facts", {})
        w = WHEN.search(story)
        if w:
            st["facts"]["when"] = w.group(0)

    # ---------------- follow-up answers (general legal information, grounded in cited sections) ----------------
    def _follow(self, kind: str, st: dict) -> tuple[list[str], list[str]]:
        laws, topic = st.get("laws", []), st.get("topic")
        women = any(l in WOMEN_PROVISO for l in laws)
        pb = self.pb.get(topic or "")
        gather = [g for g in (pb.gather if pb else [])]
        if kind == "what_do":
            pb = self.pb.get(topic or "")
            if pb:
                first = [stp["text"] for stp in pb.steps[:3]]
                return (["Here is where to start:"] + [f"{i}. {t}" for i, t in enumerate(first, 1)]
                        + ["The full list of steps and the sections that may apply are in the plan above."]), []
            return ["Tell me a little more about what happened, and I'll show you the steps that fit."], []
        if kind == "which_law":
            shown = [self.by_id[l] for l in laws if l in self.by_id][:4]
            if not shown:
                return [], []
            lines = []
            for sec in shown:
                why = self._why(sec["id"], topic)
                lines.append(f"{sec['act']} {sec['section']} ({sec['title']})" + (f": {why}" if why else ""))
            return (["Based on what you've told me, these provisions may apply, depending on the exact facts:"] + lines
                    + ["Only the police and the courts can decide which one actually applies. You can ask me what any of them means."]), [s_["id"] for s_ in shown]
        if kind == "arrest_q":
            return ["Not always. For offences punishable with up to seven years, the police may arrest without a warrant only if "
                    "certain conditions are met, for example to prevent further offences, for proper investigation, or to stop "
                    "evidence being destroyed or witnesses being threatened (BNSS 35(1)(b)).",
                    "When arrest is not needed, the police must instead give the person a notice to appear before them "
                    "(BNSS 35(3)). For more serious offences, arrest without a warrant is allowed on credible information.",
                    "Either way, the case continues: the police investigate and report to the Magistrate."], ["BNSS-35"]
        if kind == "workplace":
            return ["Yes. Besides a police complaint under the BNS, sexual harassment at work is covered by the Sexual Harassment "
                    "of Women at Workplace (Prevention, Prohibition and Redressal) Act, 2013 (the POSH Act).",
                    "Every employer with ten or more employees must have an Internal Committee; you can complain to it in writing, "
                    "usually within three months of the incident (it can be extended by another three months). If there is no "
                    "Internal Committee, the Local Committee set up by the District Officer handles it.",
                    "You can use the POSH complaint and a police complaint at the same time."], []
        if kind in ("no_witness", "evidence"):
            say = ["You can still approach the police. Witnesses are not the only possible source of evidence, and your own "
                   "account is evidence: when you report, it is written down, read back to you and signed by you, and you get "
                   "a free copy (BNSS 173(1)-(2))."]
            extra = []
            story = (st.get("story") or "").lower()
            if re.search(r"\b(?:bus|train|metro|auto|cab|taxi)\b", story):
                extra.append("the route or vehicle number, the time and stop, and your ticket or UPI payment showing you were on it")
            extra += ["CCTV at the place or nearby shops and stations (ask quickly: footage is often overwritten within days)",
                      "messages, photos, call logs or anything the person sent you",
                      "anyone you told soon afterwards, and any medical record if you were hurt"]
            say.append("Other things that can help: " + "; ".join(extra) + ".")
            if kind == "evidence" and re.search(r"delet|lose|lost", (st.get("last") or ""), re.I):
                say.insert(0, "Take screenshots now, with the date, time and the sender's number or profile visible, and keep a "
                              "copy somewhere safe. Even if messages are deleted later, the police can ask the platform or "
                              "telecom company for records.")
            say.append("Do you have anything like that, for example CCTV nearby, messages, or someone you told?")
            return say, ["BNSS-173"]
        if kind == "police_refuse":
            return ["If the police refuse to record it, you can send the substance of your complaint in writing and by post to "
                    "the Superintendent of Police (BNSS 173(4)), who can investigate or order an investigation.",
                    "You can also apply to the Magistrate, who can order the police to investigate (BNSS 175(3)).",
                    "Keep a copy of everything you send and note the date, the police station and the officer's name."], ["BNSS-173", "BNSS-175"]
        if kind == "woman_officer":
            if women:
                return ["Yes. When a woman reports offences such as sexual harassment (BNS 75), stalking (BNS 78) or outraging "
                        "her modesty (BNS 74), the information must be recorded by a woman police officer or a woman officer "
                        "(BNSS 173(1), first proviso). You can ask for this at the police station."], ["BNSS-173"]
            return ["For certain offences against women (for example BNS 74 to 79 and the sexual offences in BNS 64 to 71), "
                    "BNSS 173(1) requires the information to be recorded by a woman officer. For other offences there is no such rule, "
                    "but you can still ask."], ["BNSS-173"]
        if kind == "online":
            return ["Yes, information can be given orally or by electronic communication (for example email or a state "
                    "police e-FIR portal). If you report electronically, it is taken on record once you sign it within three days "
                    "(BNSS 173(1)(ii)).", "Many states also have online portals for complaints such as lost items or cyber fraud "
                    "(cybercrime.gov.in, helpline 1930)."], ["BNSS-173"]
        if kind == "where":
            return ["You can go to any police station. Information about a cognizable offence can be given to the officer in "
                    "charge 'irrespective of the area where the offence is committed' (BNSS 173(1)), which is often called a zero FIR; "
                    "it is then sent to the right station."], ["BNSS-173"]
        if kind == "how_report":
            say = ["You can report at any police station, orally or in writing. Tell them what happened, when and where, who was "
                   "involved (or a description), and what evidence you have. It is written down, read back to you, and you sign "
                   "it (BNSS 173(1)). You are entitled to a free copy (BNSS 173(2))."]
            if women:
                say.append("Because this is an offence against a woman, the information must be recorded by a woman officer "
                           "(BNSS 173(1), first proviso).")
            say.append("Anyone who knows about the offence can give the information, not only the victim, but it helps if the "
                       "person it happened to gives their own statement. Taking someone you trust along can help.")
            return say, ["BNSS-173"]
        if kind == "delay":
            return ["It is best to report as soon as you reasonably can, because evidence like CCTV may disappear, but a delay "
                    "does not stop you from reporting. Explain the reason for the delay when you report.",
                    "For less serious offences there is a time limit for a court to take up the case: six months if the "
                    "punishment is only a fine, one year if it is up to one year in prison, three years if it is up to three "
                    "years (BNSS 514, which has some exceptions). There is no such limit for more serious offences."], ["BNSS-514"]
        if kind == "lawyer_cost":
            return ["Reporting to the police is free, and you do not need a lawyer to file a complaint.",
                    "If you want advice or help later, free legal aid is available through your District Legal Services "
                    "Authority or the NALSA helpline 15100 (Tele-Law can also connect you to a lawyer)."], []
        if kind == "privacy":
            if any(l in IDENTITY_PROTECTED for l in laws):
                return ["For rape and related sexual offences, publishing the victim's name or anything that identifies them is "
                        "itself a crime (BNS 72), with narrow exceptions.",
                        "You can also ask the police to keep your details confidential."], ["BNS-72"]
            return ["A police complaint is not published. You can ask the officer to keep your details confidential, and you "
                    "can take someone you trust with you. For rape and related sexual offences, the law goes further: "
                    "publishing the victim's identity is itself a crime (BNS 72)."], ["BNS-72"]
        if kind == "what_next":
            return ["After your information is recorded, the police investigate: they can visit the place, question people and "
                    "record statements (BNSS 176, 180), and at the end they send a report to the Magistrate (BNSS 193).",
                    "For offences punishable with three to seven years, the police may first do a preliminary enquiry of up to "
                    "fourteen days, with a senior officer's permission, to see if there is a case (BNSS 173(3))."], ["BNSS-176", "BNSS-193"]
        if kind == "insurance":
            say = ["Insurance is outside criminal law, but the police complaint helps: insurers usually ask for a copy of the "
                   "FIR, which you get free of cost (BNSS 173(2)), along with photos, repair estimates and bills.",
                   "Tell your insurer as soon as you can; most policies have a time limit for informing them."]
            if topic in ("accident", "transport") or re.search(r"\b(?:car|bike|scooter|vehicle|accident|hit)\b", st.get("story") or "", re.I):
                say.append("For a road accident, claims for injury or death go to the Motor Accident Claims Tribunal under the "
                           "Motor Vehicles Act, 1988, and in hit-and-run cases where the vehicle is never traced there is a "
                           "government compensation scheme under that Act for death or grievous hurt.")
            return say, ["BNSS-173"]
        if kind == "compensation":
            say = ["Every state has a victim compensation scheme for people who suffered loss or injury from a crime and need "
                   "rehabilitation; the District or State Legal Services Authority decides the amount (BNSS 396)."]
            if topic == "fraud":
                say.append("For money lost in a fraud, report it quickly (for cyber fraud: helpline 1930) so the bank can try to freeze it.")
            return say, ["BNSS-396"]
        if kind == "safety":
            return ["If you are in danger at any point, call 112. Women can also call 181 or 1091 in many states.",
                    "If someone threatens you for reporting, tell the police: threatening someone is itself an offence (BNS 351), "
                    "and witnesses can ask about the witness protection scheme (BNSS 398)."], ["BNS-351", "BNSS-398"]
        return [], []

    # ---------------- main entry ----------------
    def reply(self, message: str, state: dict | None) -> tuple[Reply, dict]:
        """One turn of the conversation: the reply and the new state (with a short history for the LLM)."""
        history = list((state or {}).get("history") or [])[-6:]
        r, st = self._reply(message, state)
        msg = normalize(message or "").strip()
        if r.kind not in ("crisis",) and safety.FIRE.search(msg) and not any(safety.FIRE_TEXT in t for t in (r.say or [r.text])):
            rest = [t for t in (r.say or [r.text]) if not t.startswith(("Thanks, I've noted", "I'm not able to answer",
                                                                     "If this is happening right now", "That sounds like a separate"))]
            r.say = [safety.FIRE_TEXT] + (rest or ["Once everyone is safe, photograph the damage from a safe distance and report "
                                                  "it to the police; I can help with that when you're ready."])
        elif r.kind not in ("crisis",) and safety.is_urgent(msg) and st.get("topic") and \
                not any("112" in t for t in (r.say or [r.text])[:1]):
            r.say = ["If this is happening right now, call 112 immediately."] + (r.say or [r.text])
        said = " ".join(r.say or [r.text])
        st["history"] = (history + [{"u": msg[:400], "a": said[:600]}])[-6:]
        return r, st

    def _chat(self, msg: str, st: dict, new_laws: list[str]) -> Reply | None:
        """A free-form answer written by the LLM from the conversation, the guide and the sections shown (checked)."""
        if self.llm is None:
            return None
        pb = self.pb.get(st.get("topic") or "")
        secs = [self.by_id[l] for l in st.get("laws", []) if l in self.by_id][:4]
        text = self.llm.reply(st.get("history") or [], msg, st, pb, secs)
        if not text:
            return None
        r = Reply("chat", text)
        r.say = [text]
        r.hits = [self.bot._card(self.bot.r.sections.index(self.by_id[l])) for l in new_laws][:3] if new_laws else []
        r.chips = self._chips(st)[:3]
        return r

    def _reply(self, message: str, state: dict | None) -> tuple[Reply, dict]:
        st = dict(state or {})
        st.setdefault("laws", []); st.setdefault("facts", {})
        msg = normalize(message or "").strip()
        if not msg:
            return Reply("smalltalk", safety.GREETING_TEXT), st

        # 1. crisis, greeting, thanks, language
        if safety.is_crisis(msg):
            return Reply("crisis", safety.CRISIS_TEXT), st
        if safety.GREETING.match(msg) or safety.THANKS.match(msg) or safety.is_non_english(msg):
            return self.bot.respond([msg]), st

        explain_q = bool(_EXPLAIN.search(msg) or _PUNISH.search(msg) or re.fullmatch(r"(?:bns|bnss)?\s*(?:section|sec\.?|s\.)?\s*\d+[a-z]?\??", msg, re.I))
        follow_q = any(p.search(msg) for _, p in FOLLOW)

        # 2. the answer to our own question ("Are you safe right now?")
        if st.get("pending") == "safety" and st.get("topic"):
            yes, no = bool(SAFE_YES.search(msg)), bool(SAFE_NO.search(msg))
            other_q = explain_q or bool(_REF.search(msg) or _ORDINAL.search(msg)) or (follow_q and not (yes or no))
            looks_like_answer = (yes or no or bool(WHEN.search(msg))) and not (msg.rstrip().endswith("?") and not (yes or no))
            if not other_q and looks_like_answer:
                st["pending"] = None
            if looks_like_answer and not other_q:
                w = WHEN.search(msg)
                if w:
                    st["facts"]["when"] = w.group(0)
                story = st.get("story", "") + " " + msg
                r = self.bot.respond([story])
                if not r.plans:
                    r = self.bot.respond([st.get("story", "")])
                self._remember_plan(st, r, story)
                if st["topic"] == "missing":
                    say = ["Understood. You don't need to wait before reporting a missing person: go to the nearest police "
                           "station now, or call 112. Take a recent photo, what they were wearing, where they were last seen, "
                           "their phone number, and the names of friends they might be with.",
                           "Here are the steps, and the provisions that may matter if someone has taken them."]
                elif no and not (yes and not re.search(r"\bnot safe|unsafe|still\b", msg, re.I)):
                    st["facts"]["safe"] = False
                    say = ["Please call 112 now, or go somewhere with other people around you. If you can, call or message "
                           "someone you trust. Women can also call 181 or 1091 in many states.",
                           "When you are safe, here is what you can do and which laws may apply."]
                else:
                    st["facts"]["safe"] = True
                    when = st["facts"].get("when")
                    say = ["Understood, I'm glad you're safe."
                           + (f" Since it happened {when}, you can still report it; it is best to do it soon, while things like "
                              "CCTV footage are still available." if when else " You can still report it; it is best to do it "
                              "soon, while things like CCTV footage are still available.")]
                    pb = self.pb.get(st["topic"])
                    if pb and pb.gather:
                        say.append("It may also help to note down: " + "; ".join(_lc(g) for g in pb.gather) + ".")
                    say.append("Let's look at the steps and the provisions of the BNS that may apply. Whether they apply depends "
                               "on the exact facts.")
                return self._plan_reply(r, st, say), st

        # 3. a question about a section ("What does Section 75 mean?", "explain the first one", "punishment for it?")
        sid = self._resolve_ref(msg, st) if (explain_q or _REF.search(msg)) else None
        if not sid and _PUNISH.search(msg) and (st.get("focus") or st.get("laws")) and len(msg.split()) <= 10 \
                and not re.search(r"\b(?:for|of) (?!it\b|this\b|that\b|him\b|her\b|them\b)\w+", msg, re.I):
            sid = st.get("focus") or st["laws"][0]
        if not sid and re.search(r"\b(?:again|second time|repeat\w*|keeps? doing)\b", msg, re.I) and HYPOTHETICAL.search(msg) \
                and (st.get("focus") or st.get("laws")):
            sid = st.get("focus") or st["laws"][0]
            explain_q = True
        if sid and (explain_q or _REF.search(msg)) and not (_REF.search(msg) is None and not st.get("laws")):
            sec = self.by_id[sid]
            ex = explain_section(sec, self._why(sid, st.get("topic")))
            if _PUNISH.search(msg) and not _EXPLAIN.search(msg):
                ex["points"] = []
            st["focus"] = sid
            if sid not in st["laws"]:
                st["laws"].append(sid)
            only_punish = bool(_PUNISH.search(msg)) and not _EXPLAIN.search(msg)
            say = [f"{sec['act']} {sec['section']} deals with {sec['title'][0].lower() + sec['title'][1:]}."
                   + (" Here is the punishment it sets out." if only_punish else " Here it is in simple language.")]
            r = Reply("explain", say[0])
            r.say = say + ["This explains what the section says. Whether it applies to a particular case depends on the facts, "
                           "and only a court can decide that."]
            r.explain = ex
            r.chips = ([f"What is the punishment under {sec['act']} {sec['section']}?"] if not only_punish else []) + \
                      ["What if I don't have any witnesses?", "How do I report it?"]
            return r, st

        # 4. a follow-up question about the same situation
        st["last"] = msg
        if VAGUE.search(msg):
            r = Reply("clarify", "I'm here to help.")
            r.say = ["I'm here to help. Tell me in a sentence or two what happened: who did what, where and when.",
                     "If you or someone else is in danger right now, call 112."]
            r.chips = ["Someone stole my phone", "The police won't register my FIR", "I lost money in an online scam",
                       "Someone is harassing me"]
            return r, st
        if st.get("topic") and ACK.search(msg):
            r = Reply("followup", "Good to hear.")
            r.say = ["Good to hear. What would you like to know next? I can explain any of the sections, or how to report it."]
            r.chips = self._chips(st)
            return r, st
        if (st.get("topic") or st.get("laws")) and follow_q:
            # a different situation described in the same message wins over a follow-up (rules only: no LLM call)
            fresh = match_playbooks(self.bot.playbooks, msg)
            new_situation = bool(fresh) and fresh[0].id != st.get("topic") and len(msg.split()) > 12
            if not new_situation:
                for kind, pat in FOLLOW:
                    if pat.search(msg):
                        say, cites = self._follow(kind, st)
                        if say:
                            r = Reply("followup", say[0])
                            r.say = say
                            r.chips = [c for c in self._chips(st) if not pat.search(c)][:3]
                            r.understood = [st.get("title")] if st.get("title") else []
                            return r, st

        # 5. a "what if" about a law question just answered ("what is the punishment for murder" -> "what if it was by accident")
        if not st.get("topic") and st.get("question") and HYPOTHETICAL.search(msg):
            q = st["question"] + " " + msg
            r = self.bot._definition(q)
            if r.hits:
                st = {"laws": [h["id"] for h in r.hits], "facts": {}, "question": q}
                r.chips = self._chips(st)[:1] + ["What does the second one mean?"]
                return r, st

        # 6. a new situation, or more detail on the current one
        r = self.bot.respond([msg])
        own_plan = bool(r.plans) and r.plans[0]["id"] not in ("generic", st.get("topic"))
        additive = bool(st.get("topic")) and len(msg.split()) <= 30 and (
            bool(PRONOUN_START.search(msg)) or (bool(ADDITIVE.search(msg)) and not own_plan))
        vague_result = (not r.plans or r.plans[0]["id"] == "generic") and r.kind in ("clarify", "closest", "empty", "plan")
        in_topic = bool(st.get("topic")) and r.kind not in ("out_of_scope", "mapped") and not is_definition_question(msg) and not (
            own_plan and not PRONOUN_START.search(msg) and len(msg.split()) > 4)
        if in_topic and (additive or vague_result or (r.kind == "answer" and len(msg.split()) <= 14) or not own_plan):
            # more detail on the same story, or a free-form question about it: answer in the chat, never repeat the plan
            new = []
            if len(msg.split()) <= 30 and not msg.rstrip().endswith("?"):
                story = st.get("story", "") + " " + msg
                r2 = self.bot.respond([story])
                if r2.plans and (additive or st.get("topic") in [p["id"] for p in r2.plans]):
                    before = set(st.get("laws", []))
                    topic, title = st.get("topic"), st.get("title")
                    self._remember_plan(st, r2, story)
                    st["topic"], st["title"] = topic, title          # the conversation stays on its topic
                    st["laws"] = list(dict.fromkeys(list(before) + st["laws"]))
                    new = [l for l in st["laws"] if l not in before]
            hurt = re.search(r"\b(?:hospital|injur\w*|cut|bleed\w*|stitch\w*|fracture|broken|bruise\w*|wound\w*)\b", msg, re.I)
            chat = self._chat(msg, st, new)
            if chat is not None:
                return chat, st
            r0 = Reply("followup", "Thanks, I've noted that.")
            if new:
                r0.say = ["Thanks, I've added that. With that detail, these provisions may also be relevant:"]
                r0.hits = [self.bot._card(self.bot.r.sections.index(self.by_id[l])) for l in new][:3]
            elif msg.rstrip().endswith("?") or re.match(r"^\s*(?:shall|should|can|could|will|would|do|does|is|are|what|how|why|when|where)\b", msg, re.I):
                r0.say = ["I'm not able to answer that one reliably from the law. I can explain any of the sections, how to "
                          "report it, what evidence to keep, or what to do if the police refuse.",
                          "If anyone is in danger, call 112."]
            else:
                r0.say = ["Thanks, I've noted that. What would you like to know next?"]
            if hurt:
                r0.say.insert(0, "I'm sorry you were hurt. Get the injury examined and ask the hospital for the medical report; "
                                 "it is important evidence, and the police can arrange a medical examination.")
            r0.chips = self._chips(st)[:3]
            return r0, st
        if r.plans and r.plans[0]["id"] != "generic":
            switched = bool(st.get("topic")) and st.get("topic") != r.plans[0]["id"]
            st = {"laws": [], "facts": {}}
            self._remember_plan(st, r, msg)
            tid = st["topic"]
            say = []
            if switched:
                say.append("Got it, let's look at this.")
            if EMPATHY.get(tid):
                say.append(EMPATHY[tid])
            urgent = safety.is_urgent(msg)
            hypothetical = bool(HYPOTHETICAL.search(msg))
            if hypothetical:
                say = [s_ for s_ in say if not s_.startswith("I'm") and not s_.startswith("That sounds")]
            if tid in SENSITIVE and not urgent and not hypothetical:
                say.append(("Are you safe right now? " if tid != "missing" else "Have you been able to contact them or anyone "
                            "who might know where they are? ") + "I can help you understand your options.")
                if AREA.get(tid):
                    say.append(f"What you describe may fall under provisions of the BNS on {AREA[tid]}, depending on the "
                               "circumstances.")
                st["pending"] = "safety"
                r = self._plan_reply(r, st, say, collapsed=True)
                r.chips = (["Yes, I'm safe now", "No, it's happening now"] if tid != "missing"
                           else ["Not yet", "Yes, they're back"])
                return r, st
            if urgent:
                say.insert(0, "If this is happening right now, call 112 immediately.")
            say.append("Here is what you can do next, and the provisions that may apply depending on the exact facts.")
            return self._plan_reply(r, st, say), st
        # anything else (definitions, lookups, civil matters, generic plans, clarifying questions): as before
        if r.plans or r.hits:
            st = {"laws": [], "facts": {}}
            self._remember_plan(st, r, msg)
            if r.kind in ("answer", "closest") and not r.plans:
                st["question"] = msg      # a law question: a following "what if ..." refines it
        r.chips = self._chips(st) if (r.hits or r.plans) and r.kind not in ("clarify",) else []
        return r, st
