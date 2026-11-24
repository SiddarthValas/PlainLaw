"""PlainLaw: describe what happened, get the section of BNS / BNSS that applies.

Run:  streamlit run app.py
Env:  SECTIONS_PATH (default data/sections.jsonl), DENSE_MODEL (default BAAI/bge-small-en-v1.5, "lsa" = offline),
      USE_LLM=1 to add a plain-words line from a local Ollama model.
"""
import os
from html import escape

import streamlit as st

from src.chatbot import CONTINUING, LegalBot, OllamaExplainer
from src.retrieval import DEFAULT_DENSE_MODEL, Retriever, load_sections

SECTIONS_PATH = os.getenv("SECTIONS_PATH", "data/sections.jsonl")

st.set_page_config(page_title="PlainLaw", page_icon="⚖️", layout="centered")
st.markdown(
    """<style>
#MainMenu, footer, header {visibility: hidden;}
html, body, [class*="css"] {font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", Roboto, sans-serif;}
.block-container {max-width: 720px; padding-top: 2.2rem;}
h1 {font-weight: 700; letter-spacing: -0.02em;}
.card {border: 1px solid rgba(128,128,128,.25); border-radius: 16px; padding: 16px 18px; margin: 10px 0; background: rgba(128,128,128,.06);}
.sec {font-size: .78rem; font-weight: 600; letter-spacing: .05em; text-transform: uppercase; opacity: .65;}
.ttl {font-size: 1.15rem; font-weight: 650; margin: 2px 0 6px;}
.pill {display: inline-block; font-size: .75rem; padding: 2px 10px; border-radius: 999px; background: rgba(0,122,255,.14); color: #0a6cff; margin-bottom: 6px;}
.plain {margin: 6px 0; font-size: 1rem;}
.txt {margin: 6px 0 0; font-size: .92rem; opacity: .8;}
.why {margin: 4px 0 0; font-size: .95rem;}
.h2 {font-size: 1.05rem; font-weight: 650; margin: 18px 0 4px;}
.step {display: flex; gap: 12px; margin: 10px 0;}
.num {flex: 0 0 26px; height: 26px; border-radius: 50%; background: rgba(0,122,255,.14); color: #0a6cff; font-weight: 650; font-size: .85rem; display: flex; align-items: center; justify-content: center;}
.tag {display: inline-block; font-size: .7rem; padding: 1px 8px; border-radius: 999px; margin-left: 6px; vertical-align: 1px;}
.tag.law {background: rgba(52,199,89,.16); color: #1b8a3a;}
.tag.gen {background: rgba(128,128,128,.18); opacity: .85;}
.note {border-left: 3px solid rgba(255,159,10,.7); padding: 6px 12px; margin: 12px 0; font-size: .9rem; opacity: .9;}
.alert {border-radius: 14px; padding: 10px 14px; margin: 6px 0 4px; background: rgba(255,59,48,.12); font-size: .95rem;}
</style>""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading the law…")
def get_bot(stamp: tuple = ()) -> LegalBot:   # stamp = file times, so edited data reloads without a restart
    sections = load_sections(SECTIONS_PATH)
    retriever = Retriever(sections, dense_model=os.getenv("DENSE_MODEL", DEFAULT_DENSE_MODEL))
    explainer = OllamaExplainer() if os.getenv("USE_LLM") == "1" else None
    return LegalBot(retriever, explainer)


def plan_html(p: dict) -> str:
    laws = "".join(
        f'<div class="card"><div class="sec">{escape(h["act"])} {escape(h["section"])}</div>'
        f'<div class="ttl">{escape(h["title"])}</div>'
        + (f'<span class="pill">Replaces {escape(h["replaces"])}</span>' if h.get("replaces") else "")
        + f'<p class="why">{escape(h["why"])}</p><p class="txt">{escape(h["excerpt"])}</p></div>' for h in p["laws"])
    steps = ""
    for n, s in enumerate(p["steps"], 1):
        tag = (f'<span class="tag law">{escape(s["ref"])}</span>' if s["src"] == "law"
               else '<span class="tag gen">general guidance</span>')
        steps += f'<div class="step"><div class="num">{n}</div><div>{escape(s["text"])}{tag}</div></div>'
    gather = ("<ul>" + "".join(f"<li>{escape(g)}</li>" for g in p["gather"]) + "</ul>") if p["gather"] else ""
    outside = f'<div class="note">{escape(p["outside"])}</div>' if p.get("outside") else ""
    return (f'<div class="h2">{escape(p["title"])}</div><p>{escape(p["intro"])}</p>'
            + ('<div class="alert">If anyone is in danger or hurt, call <b>112</b> first.</div>' if p.get("alert", True) else "")
            + f'<div class="h2">What to do now</div>{steps}'
            + f'<div class="h2">Laws that may apply</div>{laws}'
            + (f'<div class="h2">Keep ready</div>{gather}' if gather else "") + outside)


def card_html(h: dict) -> str:
    pill = f'<span class="pill">Replaces {escape(h["replaces"])}</span><br>' if h.get("replaces") else ""
    plain = f'<p class="plain">{escape(h["plain"])}</p>' if h.get("plain") else ""
    return (f'<div class="card"><div class="sec">{escape(h["act"])} {escape(h["section"])}</div>'
            f'<div class="ttl">{escape(h["title"])}</div>{pill}{plain}<p class="txt">{escape(h["excerpt"])}</p></div>')


if not os.path.exists(SECTIONS_PATH):
    st.error(f"Statute data not found at {SECTIONS_PATH}. Run ingest.py first (see README).")
    st.stop()

_DATA = [SECTIONS_PATH, "data/playbooks.json"] + [os.path.join("src", f) for f in sorted(os.listdir("src")) if f.endswith(".py")]
bot = get_bot(stamp=tuple(os.path.getmtime(p) for p in _DATA if os.path.exists(p)))
ss = st.session_state
ss.setdefault("turns", [])
ss.setdefault("msgs", [])
ss.setdefault("acts", None)


def reply_with(action=None, label=None):
    action = action or {}
    if label:   # show the button the user pressed as their message
        ss.msgs.append({"role": "user", "text": label})
    if "acts" in action:
        ss.acts = action["acts"]
    ss.msgs.append({"role": "assistant", "reply": bot.respond(ss.turns, ss.acts, playbook=action.get("playbook"),
                                                              laws=action.get("laws"))})


def reset():
    ss.turns, ss.msgs, ss.acts = [], [], None


st.title("PlainLaw")
st.caption("Describe what happened in your own words. You'll get the laws that may apply and what to do next.")
if any(s["act"] == "TEST" for s in bot.r.sections):
    st.warning("Running on synthetic TEST data, not real law.")

EXAMPLES = [
    "Someone rear-ended my car. What should I do?",
    "I lost money in a UPI scam after sharing an OTP",
    "The police won't register my FIR",
    "My brother was arrested last night",
    "A man keeps following me and messaging me",
    "What is the punishment for murder?",
]


def ask(prompt: str):
    last = ss.msgs[-1]["reply"].kind if ss.msgs and ss.msgs[-1]["role"] == "assistant" else None
    if last not in CONTINUING:   # a finished answer means this is a new question
        ss.acts = None
    ss.turns = bot.next_turns(ss.turns, last, prompt)
    ss.msgs.append({"role": "user", "text": prompt})
    reply_with()


prompt = st.chat_input("Describe what happened…")
if prompt:
    ask(prompt)
if not ss.msgs:
    st.markdown('<div class="h2">Try one of these</div>', unsafe_allow_html=True)
    cols = st.columns(2)
    for i, ex in enumerate(EXAMPLES):
        cols[i % 2].button(ex, key=f"ex{i}", on_click=ask, args=(ex,), use_container_width=True)
    st.markdown('<div class="txt" style="margin-top:14px">Covers the Bharatiya Nyaya Sanhita (offences) and the Bharatiya Nagarik Suraksha Sanhita '
                '(police and court procedure), 2023, with the old IPC and CrPC numbers. Type “IPC 420” or “BNS 318” '
                'to look up a section directly.</div>', unsafe_allow_html=True)

for n, m in enumerate(ss.msgs):
    if m["role"] == "user":
        with st.chat_message("user"):
            st.write(m["text"])
        continue
    r = m["reply"]
    with st.chat_message("assistant"):
        st.write(r.text)
        if r.understood and r.kind in ("answer", "closest", "plan"):
            st.caption("Understood as: " + ", ".join(r.understood))
        for p in r.plans:
            st.markdown(plan_html(p), unsafe_allow_html=True)
        for h in r.hits:
            st.markdown(card_html(h), unsafe_allow_html=True)
        for note in r.notes:
            st.markdown(f'<div class="note">{escape(note)}</div>', unsafe_allow_html=True)
        if r.choices and n == len(ss.msgs) - 1:
            for j, (label, action) in enumerate(r.choices):
                st.button(label, key=f"choice{n}_{j}", on_click=reply_with, args=(action, label))

if ss.msgs:
    st.button("New question", on_click=reset)
if bot.router and bot.router.api_ok:
    st.caption("To understand your message, its text is sent to an online AI model. It only picks the matching "
               "situation; every law shown comes from the Act itself.")
st.caption("General information from the text of BNS and BNSS, not legal advice. Steps marked general guidance are practical tips, not law. Helpline numbers can change, so check locally. In danger? Call 112.")
