"""PlainLaw web API + the web UI (web/index.html).

Run locally (one command, opens the full UI at http://localhost:8000):
    uvicorn api:app --port 8000
Deployed: the same app runs on Hugging Face Spaces (Dockerfile); the UI can also be hosted on Vercel and pointed at
it via web/config.js.

POST /api/chat  {"message": "...", "turns": [...], "last_kind": "...", "acts": [...]}           a typed message
                {"action": {"playbook": "...", "laws": [...], "acts": [...]}, "turns": [...], ...} a pressed button
-> {"reply": {...}, "turns": [...], "acts": [...]}   (the browser keeps turns/acts and sends them back)
"""
from __future__ import annotations

import os
from dataclasses import asdict
from functools import lru_cache

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.chatbot import CONTINUING, LegalBot
from src.dialogue import Conversation
from src.retrieval import DEFAULT_DENSE_MODEL, Retriever, load_sections

ROOT = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.join(ROOT, "web")
SECTIONS_PATH = os.getenv("SECTIONS_PATH", os.path.join(ROOT, "data", "sections.jsonl"))
CARD_FIELDS = ("id", "act", "section", "title", "replaces", "excerpt", "text", "why", "plain")
MAX_LEN = 2000

app = FastAPI(title="PlainLaw API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])


@lru_cache(maxsize=1)
def bot() -> LegalBot:
    return LegalBot(Retriever(load_sections(SECTIONS_PATH), dense_model=os.getenv("DENSE_MODEL", DEFAULT_DENSE_MODEL)))


class Action(BaseModel):
    playbook: str | None = None
    laws: list[str] | None = None
    acts: list[str] | None = None


class ChatIn(BaseModel):
    message: str | None = Field(default=None, max_length=MAX_LEN)
    state: dict | None = None          # conversation state from the previous reply (topic, sections shown, ...)
    turns: list[str] = Field(default_factory=list, max_length=8)
    last_kind: str | None = None
    acts: list[str] | None = None
    action: Action | None = None


def _card(c: dict) -> dict:
    return {k: c.get(k) for k in CARD_FIELDS if c.get(k) is not None}


def _reply_json(r) -> dict:
    d = asdict(r)
    d["hits"] = [_card(h) for h in r.hits]
    d["plans"] = [{**p, "laws": [_card(l) for l in p["laws"]]} for p in r.plans]
    d["choices"] = [{"label": label, "action": {k: (list(v) if isinstance(v, tuple) else v) for k, v in act.items()}}
                    for label, act in r.choices]
    return d


@lru_cache(maxsize=1)
def convo() -> Conversation:
    return Conversation(bot())


def _clean_state(st: dict | None) -> dict:
    """Keep only the known keys, with bounded sizes: the state comes from the browser."""
    st = st or {}
    out = {"topic": st.get("topic") if isinstance(st.get("topic"), str) else None,
           "title": str(st.get("title") or "")[:200], "story": str(st.get("story") or "")[:MAX_LEN],
           "laws": [str(x)[:12] for x in (st.get("laws") or [])][:12],
           "focus": str(st.get("focus"))[:12] if st.get("focus") else None,
           "pending": st.get("pending") if st.get("pending") in ("safety",) else None,
           "facts": {k: v for k, v in (st.get("facts") or {}).items() if k in ("safe", "when")},
           "question": str(st.get("question") or "")[:MAX_LEN] or None,
           "history": [{"u": str(h.get("u") or "")[:400], "a": str(h.get("a") or "")[:600]}
                       for h in (st.get("history") or [])[-6:] if isinstance(h, dict)]}
    return out


@app.get("/api/health")
def health():
    b = bot()
    return {"ok": True, "sections": len(b.r.sections), "dense": b.r.dense_name,
            "router": b.router.name if b.router else "off",
            "online_ai": bool(b.router and b.router.api_ok)}


@app.post("/api/chat")
def chat(body: ChatIn):
    b = bot()
    turns = [t[:MAX_LEN] for t in body.turns][-6:]
    acts = tuple(body.acts) if body.acts else None
    if body.action:   # a pressed button: same conversation, maybe narrowed to one Act or a chosen playbook
        if body.action.acts is not None:
            acts = tuple(body.action.acts)
        turns = turns or [""]
        reply = b.respond(turns, acts, playbook=body.action.playbook, laws=body.action.laws)
        if body.state is not None:   # keep the conversation going from a pressed button
            st = _clean_state(body.state)
            if reply.plans or reply.hits:
                convo()._remember_plan(st, reply, " ".join(t for t in turns if t))
                reply.chips = convo()._chips(st)
            return {"reply": _reply_json(reply), "turns": turns, "acts": list(acts) if acts else None, "state": st}
    elif body.state is not None:   # conversational mode (the web UI): context-aware follow-ups
        msg = (body.message or "").strip()[:MAX_LEN]
        st = _clean_state(body.state)
        if body.last_kind in CONTINUING and not st.get("topic") and turns:   # still clarifying the first message
            turns = b.next_turns(turns, body.last_kind, msg)
            reply = b.respond(turns, acts)
            st2 = dict(st)
            if reply.plans or reply.hits:
                convo()._remember_plan(st2, reply, " ".join(turns))
                reply.chips = convo()._chips(st2)
            return {"reply": _reply_json(reply), "turns": turns, "acts": list(acts) if acts else None, "state": st2}
        reply, st = convo().reply(msg, st)
        turns = [msg] if reply.kind in CONTINUING else []
        return {"reply": _reply_json(reply), "turns": turns, "acts": None, "state": st}
    else:
        msg = (body.message or "").strip()
        if body.last_kind not in CONTINUING:   # a finished answer means this is a new question
            acts = None
        turns = b.next_turns(turns, body.last_kind, msg)
        reply = b.respond(turns, acts)
    return {"reply": _reply_json(reply), "turns": turns, "acts": list(acts) if acts else None}


@app.get("/")
def index():
    return FileResponse(os.path.join(WEB, "index.html"))


app.mount("/", StaticFiles(directory=WEB), name="web")
