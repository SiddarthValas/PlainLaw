"""Situation -> suggested laws + next steps.

data/playbooks.json holds hand-written, source-checked scenario playbooks. Every cited section must exist in the
corpus (validate() fails loudly otherwise). Steps are tagged:
  law      comes from a section of BNS/BNSS and carries its citation
  general  common practical guidance (call 112, keep evidence); shown as such, never presented as law
The retrieval model is not involved in the advice itself, so the next steps can never be invented at answer time.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

DEFAULT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "playbooks.json")


@dataclass
class Playbook:
    id: str
    title: str
    intro: str
    laws: list[dict]
    steps: list[dict]
    gather: list[str]
    outside: str | None
    patterns: list[re.Pattern]
    excludes: list[re.Pattern]
    priority: int = 0
    examples: list[str] = field(default_factory=list)
    alert: bool = True   # show the 'call 112' banner

    def score(self, query: str) -> int:
        if any(e.search(query) for e in self.excludes):
            return 0
        return sum(bool(p.search(query)) for p in self.patterns)


def load_playbooks(path: str = DEFAULT_PATH) -> list[Playbook]:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)["playbooks"]
    out = []
    for r in raw:
        out.append(Playbook(r["id"], r["title"], r["intro"], r["laws"], r["steps"], r.get("gather", []),
                            r.get("outside"), [re.compile(t, re.I) for t in r["triggers"]],
                            [re.compile(t, re.I) for t in r.get("excludes", [])], r.get("priority", 0), r.get("examples", []), r.get("alert", True)))
    return out


def load_generic(path: str = DEFAULT_PATH) -> Playbook | None:
    """The fallback plan: steps that hold for any offence; its laws are filled in from retrieval at answer time."""
    with open(path, encoding="utf-8") as f:
        r = json.load(f).get("generic")
    if not r:
        return None
    return Playbook(r["id"], r["title"], r["intro"], [], r["steps"], r.get("gather", []), r.get("outside"), [], [],
                    0, [], r.get("alert", True))


def validate(playbooks: list[Playbook], section_ids: set[str]) -> list[str]:
    """Return a list of problems (empty = fine)."""
    problems = []
    ids = [pb.id for pb in playbooks]
    problems += [f"duplicate playbook id {i}" for i in sorted({i for i in ids if ids.count(i) > 1})]
    for pb in playbooks:
        for law in pb.laws:
            if law["id"] not in section_ids:
                problems.append(f"{pb.id}: law {law['id']} not in corpus")
        for st in pb.steps:
            if st["src"] not in ("law", "general"):
                problems.append(f"{pb.id}: bad src {st['src']}")
            if st["src"] == "law" and (st.get("id") not in section_ids or not st.get("ref")):
                problems.append(f"{pb.id}: law step cites missing section {st.get('id')}")
            if st["src"] == "general" and st.get("id"):
                problems.append(f"{pb.id}: general step must not cite a section")
    return problems


def match(playbooks: list[Playbook], query: str, limit: int = 2) -> list[Playbook]:
    """Playbooks whose triggers fire, best first: higher priority (more specific situation), then more triggers
    fired, then file order."""
    scored = [(pb.priority, pb.score(query), -i, pb) for i, pb in enumerate(playbooks)]
    hits = [x for x in scored if x[1] > 0]
    hits.sort(key=lambda x: x[:3], reverse=True)
    return [x[3] for x in hits[:limit]]


def visible_laws(pb: Playbook, query: str) -> list[dict]:
    """Laws without a 'when' pattern always show; conditional ones show only when the query mentions that circumstance.
    'unless' hides a law when the query rules it out (an animal bite is not hurt caused by a person)."""
    out = [law for law in pb.laws
           if (not law.get("when") or re.search(law["when"], query, re.I))
           and not (law.get("unless") and re.search(law["unless"], query, re.I))]
    return out or pb.laws[:1]


class SemanticMatcher:
    """Fallback when no trigger fires: compare the query with each playbook's example situations using sentence
    embeddings (nearest neighbour, cosine similarity). Used only with a real embedding model, and only above a
    threshold set on development examples, so an unclear message still gets a clarifying question."""

    def __init__(self, playbooks: list[Playbook], encode, threshold: float = 0.80, margin: float = 0.02):
        import numpy as np
        self.np = np
        self.encode = encode
        self.threshold, self.margin = threshold, margin
        self.owner: list[Playbook] = []
        texts: list[str] = []
        for pb in playbooks:
            for ex in pb.examples:
                texts.append(ex)
                self.owner.append(pb)
        self.emb = encode(texts) if texts else np.zeros((0, 1))

    def scores(self, query: str) -> list[tuple[float, Playbook]]:
        """Best similarity per playbook, highest first."""
        if not self.owner:
            return []
        sims = self.emb @ self.encode([query])[0]
        best: dict[str, tuple[float, Playbook]] = {}
        for s, pb in zip(sims.tolist(), self.owner):
            if pb.id not in best or s > best[pb.id][0]:
                best[pb.id] = (s, pb)
        return sorted(best.values(), key=lambda x: -x[0])

    def match(self, query: str) -> list[Playbook]:
        ranked = self.scores(query)
        if not ranked or ranked[0][0] < self.threshold:
            return []
        if len(ranked) > 1 and ranked[0][0] - ranked[1][0] < self.margin:
            return []   # two situations are equally close: ask instead of guessing
        return [ranked[0][1]]


class SituationClassifier:
    """Trained situation classifier (train_router.py): multinomial logistic regression on bge-small embeddings,
    trained on the playbook examples + the development situation sets. Gives a probability for every playbook,
    used for the "is it one of these?" suggestions and, offline, to pick a plan when it is confident enough
    (threshold chosen by cross-validation on the training data)."""

    def __init__(self, playbooks: list[Playbook], encode, path: str):
        import numpy as np
        w = np.load(path)
        self.np, self.W, self.b = np, w["coef"], w["intercept"]
        self.classes, self.threshold = [str(c) for c in w["classes"]], float(w["threshold"])
        self.encode = encode
        self.by_id = {pb.id: pb for pb in playbooks}

    def scores(self, query: str) -> list[tuple[float, Playbook | None]]:
        """(probability, playbook) highest first; None stands for 'no playbook fits'."""
        z = self.np.asarray(self.encode([query]))[0] @ self.W.T + self.b      # multinomial logistic regression
        p = self.np.exp(z - z.max()); p /= p.sum()
        out = [(float(pr), self.by_id.get(c)) for pr, c in zip(p, self.classes) if c in self.by_id or c == "none"]
        return sorted(out, key=lambda x: -x[0])

    def match(self, query: str) -> list[Playbook]:
        ranked = self.scores(query)
        if ranked and ranked[0][1] is not None and ranked[0][0] >= self.threshold:
            return [ranked[0][1]]
        return []
