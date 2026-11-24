"""Grounded conversational replies: an LLM writes the chat answer, but only from the material we give it, and every
legal citation in its answer is checked before it is shown.

Used by src/dialogue.py for messages that the rule-based layer cannot answer (open-ended follow-ups such as "shall I
take my valuables out?" or "is this covered by insurance?"). The model receives:
  * the conversation so far (last few turns),
  * the current situation guide (title, steps, what to keep, other laws),
  * the sections already shown, with their official text,
  * facts the user gave (safe or not, when it happened).
It must answer in a few plain sentences and may cite only those sections.

Checks before a reply is shown (otherwise the caller falls back to the rule-based answer):
  * every "BNS n" / "BNSS n" it mentions is one of the sections it was given;
  * it mentions no IPC/CrPC number except one listed as replaced by a given section;
  * every phone-like number (3 to 6 or 10 digits) is either in the material or a known national helpline;
  * it does not talk about deadlines or being too late to report (a fixed, checked answer covers that);
  * the reply is not empty and not too long.
Backends: the same as the router (LLM_BASE + LLM_KEY for an OpenAI-compatible API such as Groq; else local Ollama).
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error

from .llm_router import _post

HELPLINES = {"112", "100", "101", "102", "108", "181", "1091", "1098", "1930", "14416", "15100", "14567", "1915", "1076"}
_SEC = re.compile(r"\b(BNSS|BNS)\s*(?:section\s*)?(\d+[A-Z]?)", re.I)
_OLD = re.compile(r"\b(IPC|CrPC)\s*(?:section\s*)?(\d+[A-Z]?)", re.I)
_DEADLINE = re.compile(r"\b(?:deadline|too late|time limit|limitation period|within (?:\w+ )?(?:days?|weeks?|months?|hours?)|"
                       r"must (?:report|file|complain) (?:within|before|by))\b", re.I)
_NUM = re.compile(r"(?<![\d,.₹])\b(\d{3,6}|\d{10})\b(?!\s*(?:rupees|rs|years?|months?|days?|hours?|km|%|,\d))", re.I)

SYSTEM = """You are PlainLaw, a calm and kind legal information assistant for people in India. You explain the Bharatiya
Nyaya Sanhita (BNS) and Bharatiya Nagarik Suraksha Sanhita (BNSS), 2023, which replaced the IPC and CrPC on 1 July 2024.

Rules:
- Answer the user's latest message directly, in 2 to 5 short sentences of plain English, like a helpful person in a chat.
  Do not repeat the whole plan; refer to it ("the steps above") if useful.
- For anything about the law, use ONLY the MATERIAL below. Cite sections only from ALLOWED SECTIONS, written like
  "BNS 324" or "BNSS 173". Never cite any other section, and never use old IPC/CrPC numbers.
- Say a section "may apply"; never say someone is guilty or what a court will decide.
- Answer only the specific question. Do not list the steps again; say "see the steps above" if needed.
- Mention 112 only if the user may be in danger right now (fire 101, ambulance 108).
- Describe a section only by what its text says; do not stretch it to situations the text does not mention.
- Never state or imply a deadline for reporting a crime: there is none. Do not reuse procedural time limits from the
  material (such as signing an electronic report) as reporting deadlines.
- If the user may have done the act themselves, explain neutrally and suggest a lawyer or free legal aid (NALSA 15100);
  do not tell them to report themselves.
- Practical safety advice is fine.
- If the question is outside criminal law (insurance claims, civil disputes), say so briefly and give the practical next
  step if the material has one; otherwise suggest free legal aid (NALSA helpline 15100).
- If you do not know, say so honestly and suggest what the user can ask instead. Do not invent facts, numbers, deadlines.
Reply with JSON only: {"reply": "<your answer>"}"""


GENERAL_FACTS = [   # checked, always-true background the model may use
    "GENERAL: Reporting a crime to the police is free. Anyone who knows about an offence can give the information, not "
    "only the victim; witnesses can give statements to any police officer.",
    "GENERAL: The rule that a woman officer records the information applies when the woman herself reports certain offences "
    "against her (BNSS 173(1), first proviso); it is not a rule for witnesses.",
    "GENERAL: A lost or stolen mobile phone can be blocked using its IMEI on the government's Sanchar Saathi (CEIR) portal, "
    "sancharsaathi.gov.in; keep the police complaint number for that.",
    "GENERAL: Police seizure rules are about property the police take as evidence; they are not about tracing a lost item.",
]


class GroundedResponder:
    def __init__(self, timeout: float = float(os.getenv("LLM_TIMEOUT", "8"))):
        self.base, self.key = os.getenv("LLM_BASE"), os.getenv("LLM_KEY")
        # the big model first; on a rate limit or error the same request goes to the smaller one (its own quota on Groq)
        self.models = [m.strip() for m in os.getenv("LLM_CHAT_MODELS", "openai/gpt-oss-120b,openai/gpt-oss-20b").split(",")
                       if m.strip()] if self.base else ["llama3.2"]
        self.model = self.models[0]
        self.last_model = None
        self.local_model = os.getenv("LOCAL_LLM_MODEL", "llama3.2")
        self.host = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
        on = os.getenv("LLM_CHAT", "on").lower() != "off" and os.getenv("LLM_ROUTER", "on").lower() != "off"
        self.api_ok = on and bool(self.base and self.key)
        self.local_ok = on and self.local_model.lower() != "none" and self._ollama_up()
        self.available = self.api_ok or self.local_ok
        self.timeout = timeout
        self.errors: list[str] = []
        self.rejected: list[str] = []   # replies dropped by the citation check (for evaluation)

    def _ollama_up(self) -> bool:
        try:
            out = _post(self.host + "/api/show", {"model": self.local_model}, {}, 1.5)
            return bool(out)
        except Exception:
            return False

    def _ask(self, messages: list[dict]) -> str:
        """Each API model in turn (120b, then 20b on a rate limit or error), then the local model; raises if all fail."""
        if self.api_ok:
            last = None
            for model in self.models:
                body = {"model": model, "temperature": 0.2, "messages": messages, "max_tokens": 450,
                        "response_format": {"type": "json_object"}}
                if "gpt-oss" in model:
                    body["reasoning_effort"] = "low"
                try:
                    out = _post(self.base.rstrip("/") + "/chat/completions", body,
                                {"Authorization": f"Bearer {self.key}"}, self.timeout)
                    self.last_model = model
                    return out["choices"][0]["message"]["content"]
                except Exception as e:   # 429 rate limit, timeout, model error: try the next model
                    last = e
                    self.errors = (self.errors + [f"{model}: {type(e).__name__}: {str(e)[:100]}"])[-20:]
            if not self.local_ok:
                raise last
        out = _post(self.host + "/api/chat", {"model": self.local_model, "stream": False, "format": "json",
                                               "options": {"temperature": 0.2}, "messages": messages}, {}, self.timeout * 3)
        self.last_model = self.local_model
        return out["message"]["content"]

    @staticmethod
    def material(state: dict, playbook, sections: list[dict], extra: list[str] | None = None) -> tuple[str, set[str], set[str]]:
        """The text the model may use, the section ids it may cite, and the old numbers it may mention."""
        lines, allowed, old = [], set(), set()
        if playbook is not None:
            lines.append(f"SITUATION: {playbook.title}. {playbook.intro}")
            for i, s in enumerate(playbook.steps, 1):
                lines.append(f"STEP {i}: {s['text']}" + (f" [{s['ref']}]" if s.get("ref") else ""))
                if s.get("id"):
                    allowed.add(s["id"])
            if playbook.gather:
                lines.append("KEEP READY: " + "; ".join(playbook.gather))
            if playbook.outside:
                lines.append("OTHER LAWS: " + playbook.outside)
        for sec in sections:
            allowed.add(sec["id"])
            old |= {f"{a.upper()}-{n.upper()}" for a, n in _OLD.findall(sec.get("replaces") or "")}
            text = " ".join(sec["text"].split())
            lines.append(f"SECTION {sec['act']} {sec['section']} ({sec['title']}"
                         + (f"; replaces {sec['replaces']}" if sec.get("replaces") else "") + f"): {text[:600]}")
        lines += GENERAL_FACTS
        facts = state.get("facts") or {}
        if facts:
            lines.append("FACTS FROM THE USER: " + "; ".join(f"{k}: {v}" for k, v in facts.items()))
        for e in extra or []:
            lines.append(e)
        return "\n".join(lines), allowed, old

    def check(self, reply: str, allowed: set[str], old: set[str], material: str) -> str | None:
        """None if the reply passes; else the reason it is rejected."""
        if not reply or len(reply) > 1500:
            return "empty or too long"
        for act, num in _SEC.findall(reply):
            if f"{act.upper()}-{num.upper()}" not in allowed:
                return f"cites {act.upper()} {num}, which it was not given"
        for act, num in _OLD.findall(reply):
            if f"{act.upper()}-{num.upper()}" not in old:
                return f"mentions old {act.upper()} {num}"
        if _DEADLINE.search(reply):
            return "talks about a deadline or being too late (handled by a fixed answer instead)"
        for n in _NUM.findall(reply):
            if n not in HELPLINES and n not in material and not re.fullmatch(r"20\d\d|19\d\d", n):
                return f"number {n} not in the material"
        return None

    def reply(self, history: list[dict], message: str, state: dict, playbook, sections: list[dict],
              extra: list[str] | None = None) -> str | None:
        if not self.available:
            return None
        mat, allowed, old = self.material(state, playbook, sections, extra)
        return self.generate(history, message, mat, allowed, old)

    def generate(self, history: list[dict], message: str, mat: str, allowed: set[str], old: set[str]) -> str | None:
        msgs = [{"role": "system", "content": SYSTEM + "\n\nALLOWED SECTIONS: "
                 + (", ".join(sorted(a.replace("-", " ") for a in allowed)) or "none") + "\n\nMATERIAL:\n" + mat}]
        for h in history[-4:]:
            if h.get("u"):
                msgs.append({"role": "user", "content": h["u"][:300]})
            if h.get("a"):
                msgs.append({"role": "assistant", "content": json.dumps({"reply": h["a"][:300]})})
        msgs.append({"role": "user", "content": message[:1000]})
        try:
            raw = self._ask(msgs)
        except Exception as e:   # network, rate limit, timeout: the caller falls back to rules
            self.errors = (self.errors + [f"{type(e).__name__}: {str(e)[:120]}"])[-20:]
            return None
        try:
            text = str(json.loads(raw).get("reply", "")).strip()
        except Exception:
            m = re.search(r'"reply"\s*:\s*"(.*)"', raw or "", re.S)
            text = m.group(1).strip() if m else ""
        why = self.check(text, allowed, old, mat)
        if why:
            self.rejected = (self.rejected + [f"{why} | {text[:160]}"])[-50:]
            return None
        return text
