"""Hybrid retrieval over statute sections.

Systems (the `system` argument of Retriever.search):
  tfidf, bm25, dense   one ranker over whole sections (baselines)
  hybrid               BM25 + dense over whole sections, fused with Reciprocal Rank Fusion (RRF)
  hybrid_title         hybrid plus a third ranked list: BM25 over section titles only, on the user's own words.
                       Used for definition questions ("what is the punishment for murder"), where users name the title.
  hybrid_prf           hybrid after pseudo-relevance feedback (adds words from the top 3 sections; no hand rules)
  hybrid_chunk         hybrid over overlapping title-prefixed passages, scored by their best passage

rewrite=True adds the plain-language bridge (src/glossary.py): the query is searched as "your words + legal
vocabulary". (On the dev split, fusing the raw and expanded queries as two lists did worse than searching the
expanded query alone, because the expanded query already contains the user's words.)
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from nltk.stem import PorterStemmer
from rank_bm25 import BM25Okapi
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from .glossary import analyse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LOCAL_MODEL = os.path.join(ROOT, "models", "bge-small-en-v1.5")
# a local copy (models/) means no internet is needed; otherwise download from Hugging Face; "lsa" forces the offline fallback
DEFAULT_DENSE_MODEL = _LOCAL_MODEL if os.path.isdir(_LOCAL_MODEL) else "BAAI/bge-small-en-v1.5"
REQUIRED = ("id", "act", "section", "title", "text")
CACHE_DIR = os.getenv("EMB_CACHE", os.path.join(ROOT, ".cache"))   # embedding cache (EMB_CACHE overrides)

_STOP = frozenset(
    "a an the of to in and or for on with is are was were be been being it its this that these those "
    "at by from as my me i we you your they their his her he she him them not no do does did has have had "
    "if but so than then there into out up over under about after before again very can could would "
    "should will shall may might must".split()
)
_VAGUE = frozenset("how what when where who which why help need want get let know tell something anything thing happen happened".split())
_stem = lru_cache(maxsize=None)(PorterStemmer().stem)
_VAGUE_STEMS = frozenset(_stem(w) for w in _VAGUE)


def tokenize(text: str) -> list[str]:
    return [_stem(t) for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOP and len(t) > 1]


def load_sections(path: str) -> list[dict]:
    """Read sections.jsonl; fail loudly on missing fields or duplicate ids."""
    sections, seen = [], set()
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            s = json.loads(line)
            missing = [k for k in REQUIRED if not str(s.get(k, "")).strip()]
            if missing:
                raise ValueError(f"{path}:{n} missing fields {missing}")
            if s["id"] in seen:
                raise ValueError(f"{path}:{n} duplicate id {s['id']}")
            seen.add(s["id"])
            s.setdefault("replaces", "")
            s.setdefault("change", "")
            sections.append(s)
    if not sections:
        raise ValueError(f"{path} contains no sections")
    return sections


def make_chunks(sections: list[dict], size: int = 100, stride: int = 70) -> tuple[list[str], list[int]]:
    """Overlapping word windows, each prefixed with the section title. Returns (chunk texts, owner section index)."""
    docs, owner = [], []
    for i, s in enumerate(sections):
        words = " ".join(s["text"].split()).split(" ")
        start = 0
        while True:
            docs.append(f"{s['title']}. {' '.join(words[start:start + size])}")
            owner.append(i)
            if start + size >= len(words):
                break
            start += stride
    return docs, owner


class TfidfIndex:
    def __init__(self, docs: list[str]):
        self.vec = TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None, sublinear_tf=True)
        self.mat = self.vec.fit_transform(docs)

    def scores(self, q: str) -> np.ndarray:
        return (self.mat @ self.vec.transform([q]).T).toarray().ravel()


class BM25Index:
    def __init__(self, docs: list[str]):
        self.n = len(docs)
        self.bm25 = BM25Okapi([tokenize(d) for d in docs])

    def scores(self, q: str) -> np.ndarray:
        toks = tokenize(q)
        return np.asarray(self.bm25.get_scores(toks)) if toks else np.zeros(self.n)


class DenseIndex:
    """Sentence embeddings (cached on disk); falls back to TF-IDF + LSA when the model cannot be loaded."""

    def __init__(self, docs: list[str], model_name: str = DEFAULT_DENSE_MODEL):
        self.model = None
        reason = "forced"
        if model_name != "lsa":
            try:
                from sentence_transformers import SentenceTransformer

                self.model = SentenceTransformer(model_name)
                self.prefix = "Represent this sentence for searching relevant passages: " if "bge" in model_name.lower() else ""
                key = hashlib.sha1((os.path.basename(model_name.rstrip("/")) + "\n" + "\n".join(docs)).encode()).hexdigest()[:16]
                path = os.path.join(CACHE_DIR, f"emb_{key}.npy")
                if os.path.exists(path):
                    self.emb = np.load(path)
                else:
                    self.emb = self.model.encode(docs, normalize_embeddings=True, batch_size=32, show_progress_bar=False)
                    os.makedirs(CACHE_DIR, exist_ok=True)
                    np.save(path, self.emb)
                self.name = os.path.basename(model_name.rstrip("/"))
                return
            except Exception as e:  # offline, model missing, etc.
                self.model = None
                reason = type(e).__name__
        self.vec = TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None, sublinear_tf=True)
        X = self.vec.fit_transform(docs)
        k = max(1, min(128, X.shape[0] - 1, X.shape[1] - 1))
        self.svd = TruncatedSVD(n_components=k, random_state=0)
        self.emb = normalize(self.svd.fit_transform(X))
        self.name = f"lsa-fallback ({reason})"

    def scores(self, q: str) -> np.ndarray:
        if self.model is not None:
            qv = self.model.encode([self.prefix + q], normalize_embeddings=True)[0]
        else:
            qv = normalize(self.svd.transform(self.vec.transform([q])))[0]
        return self.emb @ qv


@dataclass
class Hit:
    idx: int
    score: float
    bm25_rank: int | None = None
    dense_rank: int | None = None


@dataclass
class SearchResult:
    hits: list[Hit]
    lexical_signal: bool   # did any section share vocabulary with the query?
    n_terms: int = 0       # content words in the user's own query (vague words excluded)
    support: float = 0.0   # share of those words found in the best section
    rule_matched: bool = False   # did a plain-language rule recognise the query?


class Retriever:
    def __init__(self, sections: list[dict], dense_model: str = DEFAULT_DENSE_MODEL,
                 pool: int = 30, rrf_k: int = 60, w_bm25: float = 1.0, w_dense: float = 1.0, w_title: float = 1.0):
        self.sections = sections
        self.acts = np.array([s["act"] for s in sections])
        docs = [f"{s['title']}. {s['text']}" for s in sections]
        self.tfidf = TfidfIndex(docs)
        self.bm25, self.dense = BM25Index(docs), DenseIndex(docs, dense_model)
        self._dense_model = dense_model
        self._chunks = None   # passage indexes, built on first use (only the hybrid_chunk system needs them)
        self.title_bm25 = BM25Index([s['title'] for s in sections])   # titles only, for the title-match signal
        self.pool, self.rrf_k, self.w_bm25, self.w_dense, self.w_title = pool, rrf_k, w_bm25, w_dense, w_title
        self._surface = self._build_surface(sections)

    @staticmethod
    def _build_surface(sections) -> dict[str, str]:
        """stem -> most frequent surface word, so expansion terms can be fed to the dense model as real words."""
        from collections import Counter, defaultdict
        counts: dict[str, Counter] = defaultdict(Counter)
        for s in sections:
            for w in re.findall(r"[a-z]+", f"{s['title']} {s['text']}".lower()):
                if w not in _STOP and len(w) > 2:
                    counts[_stem(w)][w] += 1
        return {st: c.most_common(1)[0][0] for st, c in counts.items()}

    def prf_expand(self, query: str, docs: int = 3, terms: int = 6) -> str:
        """Pseudo-relevance feedback: add the most distinctive words of the top-ranked sections to the query.
        No hand-written rules. Fixed settings (3 sections, 6 terms), not tuned."""
        first = self._search(query, docs, "hybrid", False, None)
        if not first.hits:
            return query
        rows = self.tfidf.mat[[h.idx for h in first.hits]].toarray().mean(axis=0)
        names = self.tfidf.vec.get_feature_names_out()
        have = set(tokenize(query))
        extra = []
        for j in np.argsort(-rows):
            if rows[j] <= 0 or len(extra) == terms:
                break
            st = names[j]
            if st not in have and st in self._surface:
                extra.append(self._surface[st])
        return f"{query} {' '.join(extra)}".strip()

    @property
    def dense_name(self) -> str:
        return self.dense.name

    def _mask(self, acts) -> np.ndarray:
        if not acts:
            return np.ones(len(self.sections), dtype=bool)
        return np.isin(self.acts, list(acts))

    @staticmethod
    def _top(scores: np.ndarray, mask: np.ndarray, n: int, positive_only: bool) -> list[int]:
        out: list[int] = []
        for i in np.argsort(-scores, kind="stable"):
            if not mask[i]:
                continue
            if positive_only and scores[i] <= 0:
                break
            out.append(int(i))
            if len(out) == n:
                break
        return out

    def _chunk_index(self):
        if self._chunks is None:
            chunk_docs, owner = make_chunks(self.sections)
            self._chunks = (np.array(owner), BM25Index(chunk_docs), DenseIndex(chunk_docs, self._dense_model))
        return self._chunks

    def _best_chunk(self, chunk_scores: np.ndarray) -> np.ndarray:
        out = np.full(len(self.sections), -np.inf)
        np.maximum.at(out, self._chunk_index()[0], chunk_scores)
        return out

    def _rankings(self, q: str, mask: np.ndarray, chunked: bool) -> tuple[list[int], list[int]]:
        if chunked:
            _, bm25_c, dense_c = self._chunk_index()
            bm, dn = self._best_chunk(bm25_c.scores(q)), self._best_chunk(dense_c.scores(q))
        else:
            bm, dn = self.bm25.scores(q), self.dense.scores(q)
        return self._top(bm, mask, self.pool, True), self._top(dn, mask, self.pool, False)

    def search(self, query: str, k: int = 5, system: str = "hybrid_chunk", rewrite: bool = True, acts=None) -> SearchResult:
        if system == "hybrid_prf":
            res = self._search(self.prf_expand(query), k, "hybrid", False, acts)
        else:
            res = self._search(query, k, system, rewrite, acts)
        terms = [t for t in dict.fromkeys(tokenize(query)) if t not in _VAGUE_STEMS]
        res.n_terms = len(terms)
        if res.hits and terms:
            s = self.sections[res.hits[0].idx]
            have = set(tokenize(f"{s['title']}. {s['text']}"))
            res.support = sum(t in have for t in terms) / len(terms)
        res.rule_matched = bool(analyse(query).labels)
        return res

    def _search(self, query: str, k: int, system: str, rewrite: bool, acts) -> SearchResult:
        expanded = analyse(query).expanded if rewrite else query
        mask = self._mask(acts)
        if system in ("tfidf", "bm25", "dense"):
            if system == "tfidf":
                order = self._top(self.tfidf.scores(expanded), mask, k, True)
                lexical = bool(self._top(self.bm25.scores(expanded), mask, 1, True))
                return SearchResult([Hit(i, 0.0) for i in order], lexical)
            if system == "bm25":
                order = self._top(self.bm25.scores(expanded), mask, k, True)
                return SearchResult([Hit(i, 0.0, bm25_rank=r) for r, i in enumerate(order, 1)], bool(order))
            order = self._top(self.dense.scores(expanded), mask, k, False)
            lexical = bool(self._top(self.bm25.scores(expanded), mask, 1, True))
            return SearchResult([Hit(i, 0.0, dense_rank=r) for r, i in enumerate(order, 1)], lexical)
        if system not in ("hybrid", "hybrid_chunk", "hybrid_title"):
            raise ValueError(f"unknown system {system!r}")
        chunked = system == "hybrid_chunk"
        title_rank: list[int] = []
        if system == "hybrid_title":
            # third ranked list: BM25 over section titles, scored on the user's ORIGINAL words (not the expansion),
            # so "punishment for murder" finds the section titled "Punishment for murder"
            title_rank = self._top(self.title_bm25.scores(query), mask, self.pool, True)
        variants = [expanded]   # the expanded query already contains the user's own words
        fused: dict[int, float] = {}
        bm_rank: dict[int, int] = {}
        dn_rank: dict[int, int] = {}
        lexical = False
        for v in variants:
            bm, dn = self._rankings(v, mask, chunked)
            lexical = lexical or bool(bm)
            for ranks, w, best in ((bm, self.w_bm25, bm_rank), (dn, self.w_dense, dn_rank)):
                for r, i in enumerate(ranks, 1):
                    fused[i] = fused.get(i, 0.0) + w / (self.rrf_k + r)
                    best[i] = min(best.get(i, r), r)
        for r, i in enumerate(title_rank, 1):
            fused[i] = fused.get(i, 0.0) + self.w_title / (self.rrf_k + r)
        order = sorted(fused, key=lambda i: (-fused[i], bm_rank.get(i, 10**6), i))[:k]
        return SearchResult([Hit(i, fused[i], bm_rank.get(i), dn_rank.get(i)) for i in order], lexical)

    @staticmethod
    def confident(result: SearchResult, top_n: int = 3) -> bool:
        """Answer directly only if (1) BM25 and the dense model both rank the best hit in their top-n, and
        (2) the query is specific: a plain-language rule recognised it, or it has 2+ content words, most of which
        appear in that section. Otherwise the bot asks a clarifying question instead of guessing."""
        if not result.hits:
            return False
        h = result.hits[0]
        agree = (h.bm25_rank is not None and h.bm25_rank <= top_n
                 and h.dense_rank is not None and h.dense_rank <= top_n)
        specific = result.rule_matched or (result.n_terms >= 2 and result.support >= 0.5)
        return agree and specific
