"""LLM intent router: reads a free-text message and picks ONE of the scenario playbooks (or "none").

The model only chooses an id from a fixed list (constrained, validated JSON). It never writes legal content: the steps
and sections shown still come from data/playbooks.json and the Act text, so a wrong or repealed section cannot appear.

Backends, tried in order:
  1. LLM_BASE + LLM_KEY [+ LLM_MODEL]   any OpenAI-compatible API, e.g. Groq (https://api.groq.com/openai/v1)
  2. local Ollama (LOCAL_LLM_MODEL, default llama3.2) if it is running: used when the API is not set or fails
Set LLM_ROUTER=off to disable. If no backend answers, the bot silently falls back to rules + embeddings.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from functools import lru_cache

NONE = "none"


UA = "PlainLaw/1.0 (+https://github.com)"   # some API gateways (Cloudflare) reject Python's default user agent


def _ssl_context():
    """python.org Python on macOS ships without root certificates; use certifi's bundle when available."""
    import ssl
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


_SSL = _ssl_context()


def _post(url: str, body: dict, headers: dict, timeout: float) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "User-Agent": UA, **headers})
    with urllib.request.urlopen(req, timeout=timeout, context=_SSL if url.startswith("https") else None) as r:
        return json.loads(r.read())


class LLMRouter:
    def __init__(self, playbooks, timeout: float = float(os.getenv("LLM_TIMEOUT", "8"))):
        self.ids = [pb.id for pb in playbooks]
        self.titles = {pb.id: pb.title for pb in playbooks}
        lines = []
        for pb in playbooks:
            ex = "; ".join(f'"{e}"' for e in pb.examples[:2])
            lines.append(f"- {pb.id}: {pb.title}. Examples: {ex}")
        self.menu = "\n".join(lines)
        self.timeout = timeout
        self.base, self.key = os.getenv("LLM_BASE"), os.getenv("LLM_KEY")
        self.model = os.getenv("LLM_MODEL", "openai/gpt-oss-120b" if self.base else "llama3.2")
        self.local_model = os.getenv("LOCAL_LLM_MODEL", "llama3.2")
        self.host = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
        on = os.getenv("LLM_ROUTER", "on").lower() != "off"
        self.api_ok = on and bool(self.base and self.key)
        self.local_ok = on and self._ollama_has(self.local_model)
        self.available = self.api_ok or self.local_ok
        self.errors: list[str] = []   # last failures, so a broken key or URL is visible instead of silent

    @property
    def name(self) -> str:
        parts = ([f"{self.model} via API"] if self.api_ok else []) + ([f"{self.local_model} local"] if self.local_ok else [])
        return " -> ".join(parts) or "off"

    def _ollama_has(self, model: str) -> bool:
        try:   # is Ollama running and is the model pulled?
            with urllib.request.urlopen(self.host + "/api/tags", timeout=1.5) as r:
                names = {m["name"].split(":")[0] for m in json.loads(r.read()).get("models", [])}
            return model.split(":")[0] in names
        except Exception:
            return False

    def prompt(self, message: str) -> str:
        return (
            "You route messages for an Indian legal-information app. Read the user's message and choose the ONE "
            "situation below that best matches what happened to them. Understand the meaning, not just the words.\n"
            f"{self.menu}\n"
            f"- {NONE}: none of these fits, OR it is a question about what a law says (\"what is the punishment for X\"), "
            "OR it is a civil matter (rent or deposit, salary, divorce, property ownership, a loan not repaid).\n\n"
            'Reply with JSON only: {"situation": "<one id from the list>", "confidence": "high" | "medium" | "low"}\n\n'
            f"User message: {message}"
        )

    def _ask_api(self, prompt: str, retries: int = 2) -> str:
        time.sleep(float(os.getenv("LLM_SLEEP", "0")))   # pacing for free-tier rate limits during evaluation
        body = {"model": self.model, "temperature": 0, "messages": [{"role": "user", "content": prompt}],
                "response_format": {"type": "json_object"}}
        fallback = os.getenv("LLM_FALLBACK_MODEL", "openai/gpt-oss-20b")
        for attempt in range(retries + 1):
            try:
                out = _post(self.base.rstrip("/") + "/chat/completions", body,
                            {"Authorization": f"Bearer {self.key}"}, min(self.timeout, 15))
                return out["choices"][0]["message"]["content"]
            except urllib.error.HTTPError as e:
                if e.code == 429 and fallback and body["model"] != fallback and not os.getenv("LLM_SLEEP"):
                    body = {**body, "model": fallback}      # rate-limited: the smaller model has its own quota
                    continue
                if e.code != 429 or attempt == retries:
                    raise
                wait = e.headers.get("retry-after") if e.headers else None
                time.sleep(min(float(wait) if wait and wait.replace(".", "").isdigit() else 8.0, 20.0))   # rate limit: wait, retry
        raise RuntimeError("unreachable")

    def _ask_local(self, prompt: str) -> str:
        out = _post(self.host + "/api/chat",
                    {"model": self.local_model, "stream": False, "format": "json", "keep_alive": "30m",
                     "options": {"temperature": 0}, "messages": [{"role": "user", "content": prompt}]},
                    {}, self.timeout)
        return out["message"]["content"]

    def _ask(self, prompt: str) -> str:
        """API first (e.g. Groq); if it fails or is rate-limited, the local model; if both fail, raise."""
        if self.api_ok:
            try:
                return self._ask_api(prompt)
            except Exception as e:
                self._note(e)
                if not self.local_ok:
                    raise
        return self._ask_local(prompt)

    def _note(self, e: Exception) -> None:
        detail = ""
        if hasattr(e, "read"):
            try:
                detail = e.read().decode()[:200]
            except Exception:
                pass
        self.errors = (self.errors + [f"{type(e).__name__}: {e} {detail}".strip()])[-5:]

    def parse(self, raw: str) -> tuple[str, str]:
        """Validated (situation id or 'none', confidence). Anything not on the list becomes 'none'."""
        m = re.search(r"\{.*\}", raw or "", re.S)
        try:
            d = json.loads(m.group(0)) if m else {}
        except json.JSONDecodeError:
            d = {}
        sid = str(d.get("situation", NONE)).strip().lower()
        conf = str(d.get("confidence", "low")).strip().lower()
        return (sid if sid in self.ids else NONE), (conf if conf in ("high", "medium", "low") else "low")

    @lru_cache(maxsize=512)
    def route(self, message: str) -> tuple[str, str]:
        if not self.available:
            return NONE, "low"
        try:
            return self.parse(self._ask(self.prompt(message)))
        except Exception as e:   # timeout, network, model error: fall back to rules (recorded in self.errors)
            self._note(e)
            return NONE, "low"


if __name__ == "__main__":   # quick check:  python3 -m src.llm_router "someone is touching me in a bus"
    import sys
    from .playbooks import load_playbooks
    r = LLMRouter(load_playbooks())
    print("backend:", r.name)
    msg = " ".join(sys.argv[1:]) or "we got a call saying my son is in police custody and we must pay to release him"
    print("message:", msg)
    print("route:  ", r.route(msg))
    if r.errors:
        print("errors: ", *r.errors, sep="\n  ")
