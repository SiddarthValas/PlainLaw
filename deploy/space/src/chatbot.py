"""Grounded legal-information bot.

Every section shown to the user comes from retrieval over the statute text, never from a
language model. An optional local LLM may add a plain-words line, and it is discarded unless
every number in it already appears in the section it explains.
"""
from __future__ import annotations

import json
import os
import re
import urllib.request
from dataclasses import dataclass, field

from .glossary import analyse, legal_terms
from .intent import is_definition_question
from . import safety
from .dates import old_law_note
from .llm_router import NONE as ROUTE_NONE, LLMRouter
from .playbooks import SemanticMatcher, SituationClassifier, load_generic, load_playbooks, match as match_playbooks, validate as validate_playbooks, visible_laws
from .retrieval import _VAGUE_STEMS, Retriever, tokenize

_OLD_NAMES = {"ipc": "IPC", "crpc": "CrPC"}
_ACT_PAT = r"(ipc|cr\.?\s?p\.?\s?c\.?)"
_SEC_PAT = r"(?:sec(?:tion)?\.?\s*|s\.?\s*)"
_RE_OLD = re.compile(rf"\b{_ACT_PAT}\s*(?:{_SEC_PAT})?(\d+[a-z]?)\b", re.I)
_RE_OLD2 = re.compile(rf"\b{_SEC_PAT}(\d+[a-z]?)\s*(?:of\s+(?:the\s+)?)?{_ACT_PAT}\b", re.I)
_RE_NEW = re.compile(rf"\b(bnss|bns)\s*(?:{_SEC_PAT})?(\d+[a-z]?)\b", re.I)
_RE_NEW2 = re.compile(rf"\b{_SEC_PAT}(\d+[a-z]?)\s*(?:of\s+(?:the\s+)?)?(bnss|bns)\b", re.I)
_RE_BARE = re.compile(r"\b(?:section|sec\.?)\s*(\d{1,3}[a-z]?)\b", re.I)

RETRIEVAL = os.getenv("RETRIEVAL", "hybrid+rewrite")   # hybrid | hybrid+rewrite | hybrid_chunk+rewrite
CONTINUING = {"clarify", "closest", "empty"}
ROUTER_MODE = os.getenv("ROUTER_MODE", "fallback")   # fallback | first  (see respond)   # reply kinds after which the next message adds detail

CHOICES = [   # (button label, keyword arguments for LegalBot.respond)
    ("Something was done to me or someone else", {"acts": ("BNS",)}),
    ("I need to know my rights or what to do next", {"acts": ("BNSS",)}),
]


_QUOTES = str.maketrans({"\u2018": "'", "\u2019": "'", "\u201b": "'", "\u2032": "'", "\u201c": '"', "\u201d": '"',
                         "\u2013": "-", "\u2014": "-", "\u00a0": " "})


def normalize(text: str) -> str:
    """Phone keyboards type curly quotes (won’t); the rules are written with straight ones (won't)."""
    return " ".join(text.translate(_QUOTES).split())


def _old_name(raw: str) -> str:
    return _OLD_NAMES[re.sub(r"[.\s]", "", raw.lower())]


def find_references(text: str) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Return (old_refs, new_refs) such as [("IPC","420")] and [("BNS","318")]."""
    old = [(_old_name(a), n.upper()) for a, n in _RE_OLD.findall(text)]
    old += [(_old_name(a), n.upper()) for n, a in _RE_OLD2.findall(text)]
    new = [(a.upper(), n.upper()) for a, n in _RE_NEW.findall(text)]
    new += [(a.upper(), n.upper()) for n, a in _RE_NEW2.findall(text)]
    return list(dict.fromkeys(old)), list(dict.fromkeys(new))


def _grounded(out: str | None, section: dict) -> bool:
    """Accept a model-written line only if every number in it already appears in the section."""
    if not out or len(out) > 500:
        return False
    allowed = set(re.findall(r"\d+", section["text"])) | set(re.findall(r"\d+", str(section["section"])))
    return all(n in allowed for n in re.findall(r"\d+", out))


class OllamaExplainer:
    """Optional one-line plain-words explanation from a local LLM (Ollama). Fails silent."""

    def __init__(self, model: str | None = None, host: str | None = None, timeout: float = 8.0):
        self.model = model or os.getenv("LLM_MODEL", "llama3.2:3b")
        self.host = (host or os.getenv("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
        self.timeout = timeout

    def explain(self, section: dict) -> str | None:
        prompt = (
            "Explain this provision of Indian law in one or two plain sentences for a non-lawyer. "
            "Use ONLY the text below. Do not mention any section numbers and add nothing that is not in the text.\n\n"
            f"{section['act']} {section['section']} - {section['title']}\n{section['text'][:1500]}"
        )
        body = json.dumps({"model": self.model, "prompt": prompt, "stream": False,
                           "options": {"temperature": 0}}).encode()
        try:
            req = urllib.request.Request(f"{self.host}/api/generate", data=body,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                out = json.loads(resp.read()).get("response", "").strip()
        except Exception:
            return None
        return out if _grounded(out, section) else None


@dataclass
class Reply:
    kind: str                     # crisis | smalltalk | language | plan | answer | closest | clarify | out_of_scope | mapped | empty
    text: str
    hits: list[dict] = field(default_factory=list)
    understood: list[str] = field(default_factory=list)
    choices: list[tuple[str, dict]] = field(default_factory=list)   # (button label, kwargs for respond)
    plans: list[dict] = field(default_factory=list)   # situation playbooks: laws that may apply + next steps
    notes: list[str] = field(default_factory=list)    # pointers to other Acts that also apply
    # conversation layer (src/dialogue.py)
    say: list[str] = field(default_factory=list)      # conversational paragraphs shown instead of `text`
    chips: list[str] = field(default_factory=list)    # suggested follow-up messages (sent as typed text)
    collapsed: bool = False                           # plan attached but folded (e.g. while asking "are you safe?")
    explain: dict | None = None                       # plain-language explanation of one section


class LegalBot:
    def __init__(self, retriever: Retriever, explainer: OllamaExplainer | None = None, max_cards: int = 3,
                 playbooks: bool | str = True, router: LLMRouter | bool | None = None):
        self.r = retriever
        self.explainer = explainer
        self.max_cards = max_cards
        self.acts = {s["act"] for s in retriever.sections}
        self.playbooks = []
        if playbooks:
            self.playbooks = load_playbooks(playbooks) if isinstance(playbooks, str) else load_playbooks()
            bad = validate_playbooks(self.playbooks, {s["id"] for s in retriever.sections})
            if bad:   # e.g. the synthetic test corpus has none of the cited sections
                self.playbooks = []
        # fallback plan: steps true for any offence, attached to whatever BNS offence sections retrieval finds
        self.generic = None
        if self.playbooks:
            g = load_generic(playbooks) if isinstance(playbooks, str) else load_generic()
            if g and not validate_playbooks([g], {s["id"] for s in retriever.sections}):
                self.generic = g
        # with a real sentence-embedding model, unclear messages get "is it one of these?" suggestions
        self.semantic = None
        model = getattr(retriever.dense, "model", None)
        if self.playbooks and model is not None:
            self.semantic = SemanticMatcher(self.playbooks, lambda t: model.encode(t, normalize_embeddings=True))
        # trained situation classifier (train_router.py), if it has been trained for this embedding model
        self.clf = None
        clf_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "situation_clf.npz")
        if self.semantic is not None and os.path.exists(clf_path) and os.getenv("SITUATION_CLF", "on") != "off":
            try:
                self.clf = SituationClassifier(self.playbooks, lambda t: model.encode(t, normalize_embeddings=True), clf_path)
            except Exception:
                self.clf = None
        # an LLM that only chooses which playbook fits (never writes legal content); None = auto-detect, False = off
        if router is None:
            router = LLMRouter(self.playbooks) if self.playbooks else False
        self.router = router if router and router.available else None
        # section id -> a playbook that lists it, so a law card can offer "what to do next"
        self.by_law = {}
        for pb in sorted(self.playbooks, key=lambda p: p.priority):   # the most general playbook wins a shared law
            for law in pb.laws:
                if not law.get("when"):   # conditional laws are side cases of the situation, not its core
                    self.by_law.setdefault(law["id"], pb)
        self.new_idx: dict[tuple[str, str], int] = {}
        self.old_idx: dict[tuple[str, str], list[int]] = {}
        for i, s in enumerate(retriever.sections):
            self.new_idx[(s["act"].upper(), str(s["section"]).upper())] = i
            for act, num in re.findall(r"\b(IPC|CrPC)\s*(\d+[A-Z]?)\b", s.get("replaces", ""), re.I):
                self.old_idx.setdefault((_OLD_NAMES[act.lower()], num.upper()), []).append(i)

    @staticmethod
    def next_turns(turns: list[str], last_kind: str | None, prompt: str) -> list[str]:
        """Keep earlier turns only while the bot is still working out what the user means."""
        return (turns if last_kind in CONTINUING else []) + [prompt]

    # ---- cards ----
    def _card(self, i: int) -> dict:
        s = dict(self.r.sections[i])
        text = " ".join(s["text"].split())
        s["excerpt"] = text if len(text) <= 360 else text[:360].rsplit(" ", 1)[0] + "…"
        s["plain"] = self.explainer.explain(s) if self.explainer else None
        return s

    def _plan(self, pb, query: str = "") -> dict:
        idx = {s["id"]: i for i, s in enumerate(self.r.sections)}
        laws = []
        for law in visible_laws(pb, query):
            card = self._card(idx[law["id"]])
            card["why"] = law["why"]
            laws.append(card)
        return {"id": pb.id, "title": pb.title, "intro": pb.intro, "laws": laws, "steps": pb.steps,
                "gather": pb.gather, "outside": pb.outside, "alert": pb.alert}

    # ---- exact lookups: "IPC 420", "section 302 IPC", "BNS 103" ----
    def _lookup(self, text: str) -> Reply | None:
        old, new = find_references(text)
        if not old and not new:
            m = _RE_BARE.search(text)
            if not m:
                return None
            num = m.group(1).upper()   # "section 420" with no Act named: people usually mean the old IPC/CrPC number
            found = [(a, self.old_idx[(a, num)]) for a in ("IPC", "CrPC") if (a, num) in self.old_idx]
            if not found:
                return None
            idxs = list(dict.fromkeys(i for _, f in found for i in f))
            say = "; ".join(f"{a} {num} is now " + ", ".join(f"{self.r.sections[i]['act']} {self.r.sections[i]['section']}" for i in f)
                            for a, f in found)
            return Reply("mapped", f"You didn't name an Act, so I assumed the old numbering: {say}. "
                                   f"If you meant BNS {num} or BNSS {num} directly, type it like “BNS {num}”.",
                         [self._card(i) for i in idxs[:3]])
        idxs: list[int] = []
        lines: list[str] = []
        for key in old:
            found = self.old_idx.get(key, [])
            if found:
                names = ", ".join(f"{self.r.sections[i]['act']} {self.r.sections[i]['section']}" for i in found)
                lines.append(f"{key[0]} {key[1]} is now {names}.")
                idxs += found
            else:
                lines.append(f"I don't have {key[0]} {key[1]} in my mapping.")
        for key in new:
            i = self.new_idx.get(key)
            if i is not None:
                lines.append(f"Here is {key[0]} {key[1]}.")
                idxs.append(i)
            else:
                lines.append(f"I don't have {key[0]} {key[1]}.")
        idxs = list(dict.fromkeys(idxs))
        if not idxs:
            return Reply("empty", " ".join(lines))
        return Reply("mapped", " ".join(lines), [self._card(i) for i in idxs])

    def _next_steps_choice(self, hits: list[dict]) -> list[tuple[str, dict]]:
        for h in hits[:2]:
            pb = self.by_law.get(h["id"])
            if pb:
                return [(f"What should I do next? ({pb.title})", {"playbook": pb.id})]
        if self.generic and hits and self._is_offence(hits[0]):
            return [("What should I do if this happened to me?", {"playbook": "generic", "laws": [hits[0]["id"]]})]
        return []

    @staticmethod
    def _is_offence(section: dict) -> bool:
        """BNS sections 63 onwards define offences (1-62 are definitions, exceptions, abetment, attempt)."""
        m = re.match(r"\d+", str(section["section"]))
        return section["act"] == "BNS" and bool(m) and int(m.group()) >= 63

    def _generic_plan(self, idxs: list[int], query: str) -> dict:
        laws = []
        for n, i in enumerate(idxs):
            card = self._card(i)
            card["why"] = ("The section that best matches what you described. Read it to check it fits." if n == 0
                           else "A related section that may also apply.")
            laws.append(card)
        g = self.generic
        return {"id": g.id, "title": g.title, "intro": g.intro, "laws": laws, "steps": g.steps,
                "gather": g.gather, "outside": g.outside, "alert": g.alert}

    def _definition(self, query: str) -> Reply:
        """'What is X / punishment for X': the user names a legal idea, so section titles matter most.
        Searched with the user's words plus the statute's names for common legal terms, no situation rewriting."""
        res = self.r.search(legal_terms(query), k=5, system="hybrid_title", rewrite=False)
        if not res.lexical_signal or not res.hits:
            return Reply("empty", "I couldn't find that in BNS or BNSS. Try naming the offence or the procedure in other words.")
        q_words = set(tokenize(legal_terms(query))) - _VAGUE_STEMS
        title = lambda h: set(tokenize(self.r.sections[h.idx]["title"]))
        # a section whose whole title the user typed ("punishment for murder", "cheating") goes first
        hits = sorted(res.hits, key=lambda h: not (title(h) and title(h) <= q_words))
        # related cards only if their title shares a word with the question, so unrelated sections are not shown
        hits = hits[:1] + [h for h in hits[1:] if title(h) & q_words]
        cards = [self._card(h.idx) for h in hits[:self.max_cards]]
        top = self.r.sections[hits[0].idx]
        title_words = set(tokenize(top["title"]))
        named = bool(title_words) and len(title_words & set(tokenize(query))) / len(title_words) >= 0.5
        sure = Retriever.confident(res) or named   # the user named the section's own title, e.g. "punishment for murder"
        more = {1: "", 2: ", with one related section", 3: ", with two related ones"}[len(cards)]
        text = (f"Here is the section that matches best{more}." if sure else
                "I'm not fully sure. These are the closest sections, so please check them.")
        return Reply("answer" if sure else "closest", text, cards, choices=self._next_steps_choice(cards))

    # ---- main entry ----
    def respond(self, turns: list[str], acts=None, playbook: str | None = None, laws: list[str] | None = None) -> Reply:
        reply = self._respond(turns, acts, playbook, laws)
        # the date of the events decides which code applies: flag messages that may predate 1 July 2024
        if reply.kind in ("plan", "answer", "closest", "mapped", "clarify"):
            note = old_law_note(" ".join(normalize(t) for t in turns))
            if note and note not in reply.notes:
                reply.notes = list(reply.notes) + [note]
        return reply

    def _respond(self, turns: list[str], acts=None, playbook: str | None = None, laws: list[str] | None = None) -> Reply:
        turns = [normalize(t) for t in turns]
        last = turns[-1]
        if playbook == "generic" and self.generic and laws:   # "What should I do if this happened to me?"
            idx = {s["id"]: i for i, s in enumerate(self.r.sections)}
            idxs = [idx[l] for l in laws if l in idx]
            if idxs:
                return Reply("plan", "Here is what you can do next.", understood=[self.generic.title],
                             plans=[self._generic_plan(idxs, " ".join(turns))])
        if playbook:   # the user pressed "What should I do next?" under a law card
            pb = next((p for p in self.playbooks if p.id == playbook), None)
            if pb:
                return Reply("plan", "Here is what you can do next.", understood=[pb.title],
                             plans=[self._plan(pb, " ".join(turns))])
        if safety.is_crisis(last):
            return Reply("crisis", safety.CRISIS_TEXT)
        if safety.GREETING.match(last):
            return Reply("smalltalk", safety.GREETING_TEXT)
        if safety.THANKS.match(last):
            return Reply("smalltalk", safety.THANKS_TEXT)
        if safety.is_non_english(last):
            return Reply("language", safety.LANGUAGE_TEXT)
        ref = self._lookup(last)
        if ref:
            return ref
        if len(turns) == 1 and not acts and safety.is_nonliteral(last) and not safety.is_urgent(last):
            return Reply("clarify", safety.NONLITERAL_TEXT)
        query = " ".join(turns)
        urgent = safety.URGENT_PREFIX if safety.is_urgent(query) else ""
        a = analyse(query)
        if len(turns) == 1 and not acts and is_definition_question(last):
            return self._definition(last)
        plans = match_playbooks(self.playbooks, query) if self.playbooks and not acts else []
        ai_label = None
        router_failed = False
        # ROUTER_MODE=first: the LLM's reading of the message decides, rule matches come second.
        # ROUTER_MODE=fallback (default): the LLM is asked only when no rule fires.
        if self.router and not acts and len(turns) <= 2 and not (a.out_of_scope and not plans) \
                and (ROUTER_MODE == "first" or not plans):
            n_err = len(self.router.errors)
            sid, conf = self.router.route(query)
            router_failed = len(self.router.errors) > n_err   # network/API error: behave as if offline
            pb = next((p for p in self.playbooks if p.id == sid), None)
            if pb and conf in ("high", "medium"):
                plans = [pb] + [p for p in plans if p.id != pb.id][:1]
                ai_label = pb.title
        clf_label = None
        # offline (no LLM router, or the LLM call failed): the trained classifier picks a plan when no rule fired
        # and it is confident enough
        if not plans and (not self.router or router_failed) and self.clf and not acts and len(turns) == 1 and not a.out_of_scope:
            plans = self.clf.match(query)
            clf_label = plans[0].title if plans else None
        if a.out_of_scope and not plans:
            if a.notes:   # criminal, but under another Act
                return Reply("out_of_scope", "BNS and BNSS don't cover this directly. Here is where to go instead.",
                             understood=a.note_labels, notes=a.notes)
            return Reply(
                "out_of_scope",
                f"This is mainly a civil matter ({', '.join(a.civil_labels)}), not a crime under BNS or BNSS. "
                "Here is where to go." if a.civil_notes else
                f"This sounds like a civil matter ({', '.join(a.civil_labels)}). BNS and BNSS are criminal laws, "
                "so they won't cover it. Free legal aid, such as Tele-Law or your District Legal Services "
                "Authority (NALSA helpline 15100), can point you the right way.",
                understood=a.civil_labels, notes=a.civil_notes + a.notes,
            )
        if plans:
            notes = list(a.notes)
            if a.civil_labels:
                notes += a.civil_notes or [f"Parts of this may also be a civil matter ({', '.join(a.civil_labels)}), which BNS "
                                           "and BNSS do not cover. Free legal aid (NALSA helpline 15100) can advise on that side."]
            understood = [f"{p.title} (understood by the AI model)" if p.title == ai_label else
                          f"{p.title} (picked by the trained classifier)" if p.title == clf_label else p.title for p in plans]
            return Reply("plan", urgent + "Here is what you can do next, and the sections that may apply depending on the exact facts.",
                         understood=understood, plans=[self._plan(p, query) for p in plans], notes=notes)
        # default: hybrid + plain-language rewriting (best cards on the demo phrases with bge; on held-out sets it was
        # neutral-to-negative, see README). Override with RETRIEVAL=hybrid or hybrid_chunk+rewrite.
        system, _, rw = RETRIEVAL.partition("+")
        res = self.r.search(query, k=5, system=system, rewrite=bool(rw), acts=acts)
        first_try = len(turns) == 1 and not acts
        if res.lexical_signal and Retriever.confident(res):
            top = res.hits[0]
            extra = [h for h in res.hits[1:self.max_cards]
                     if h.bm25_rank and h.bm25_rank <= 5 and h.dense_rank and h.dense_rank <= 5]
            cards = [self._card(h.idx) for h in [top] + extra]
            specific = [ch for ch in self._next_steps_choice(cards) if ch[1].get("playbook") != "generic"]
            if self.generic and self._is_offence(cards[0]) and not specific:
                # no detailed guide for this situation: matched sections + the steps that hold for any offence
                return Reply("plan", urgent + "I don't have a detailed guide for exactly this, but here are the sections "
                                              "that match and the steps that apply to any offence.",
                             understood=a.labels, plans=[self._generic_plan([h.idx for h in [top] + extra], query)], notes=a.notes)
            return Reply("answer", "This section seems closest to what you described. Whether it applies depends on the full facts.", cards, a.labels,
                         choices=self._next_steps_choice(cards), notes=a.notes)
        # both retrievers put the same section near the top, and no situation is a close match: show that section
        # (with the any-offence steps if it is an offence) instead of asking about unrelated situations
        top = res.hits[0] if res.hits else None
        strong = top is not None and top.bm25_rank and top.bm25_rank <= 5 and top.dense_rank and top.dense_rank <= 5
        near_situation = self.semantic is not None and self.semantic.scores(query)[0][0] >= 0.78
        # (only with the real embedding model; the offline LSA fallback ranks too loosely for this)
        if first_try and res.lexical_signal and strong and self.semantic is not None and not near_situation:
            agreed = [h for h in res.hits if h.bm25_rank and h.bm25_rank <= 10 and h.dense_rank and h.dense_rank <= 10]
            if self.generic and self._is_offence(self.r.sections[top.idx]):
                return Reply("closest", urgent + "I'm not fully sure which section fits. These are the closest, with the "
                                                 "steps that apply to any offence. Please check the sections before relying on them.",
                             understood=a.labels, plans=[self._generic_plan([h.idx for h in agreed[:self.max_cards]], query)],
                             notes=a.notes)
            cards = [self._card(h.idx) for h in agreed[:self.max_cards]]
            return Reply("closest", urgent + "I'm not fully sure. These are the closest sections, so please check them "
                                             "before relying on them.", cards, a.labels,
                         choices=self._next_steps_choice(cards), notes=a.notes)
        if first_try:
            if self.semantic:   # top 3 situations: trained classifier (94% top-3 on 65 blind situations) or nearest examples
                sug = ([pb for _, pb in self.clf.scores(query) if pb is not None][:3] if self.clf
                       else [pb for _, pb in self.semantic.scores(query)[:3]])
                return Reply("clarify", urgent + "I want to be sure I understood. Is your situation closest to one of these? "
                                                 "You can also type a little more detail.",
                             understood=a.labels,
                             choices=[(pb.title, {"playbook": pb.id}) for pb in sug]
                                     + [("None of these", {"acts": tuple(sorted(self.acts))})])
            if {"BNS", "BNSS"} <= self.acts:
                return Reply("clarify", urgent + "I want to point you to the right law. Which is closer to your situation? You can also type a little more detail.",
                             understood=a.labels, choices=list(CHOICES))
            return Reply("clarify", urgent + "Could you tell me a bit more about what happened?", understood=a.labels)
        # show "closest" sections only where both retrievers put them in their top 10; otherwise say so plainly
        agreed = [h for h in res.hits if h.bm25_rank and h.bm25_rank <= 10 and h.dense_rank and h.dense_rank <= 10]
        if res.lexical_signal and agreed and self.generic and self._is_offence(self.r.sections[agreed[0].idx]):
            return Reply("closest", "I'm not fully sure which section fits. These are the closest, with the steps that "
                                    "apply to any offence. Please check the sections before relying on them.",
                         understood=a.labels, plans=[self._generic_plan([h.idx for h in agreed[:self.max_cards]], query)],
                         notes=a.notes)
        if res.lexical_signal and agreed:
            return Reply("closest",
                         "I'm not fully sure. These are the closest matches, so please check them before relying on them.",
                         [self._card(h.idx) for h in agreed[:self.max_cards]], a.labels,
                         choices=self._next_steps_choice([self.r.sections[h.idx] for h in agreed[:2]]), notes=a.notes)
        return Reply("empty", "I couldn't find a section of BNS or BNSS that clearly fits. Try saying what was done in a "
                              "sentence or two (for example who did what, and whether anyone was hurt, threatened or cheated). "
                              "For personal advice, free legal aid is available on the NALSA helpline 15100.")
