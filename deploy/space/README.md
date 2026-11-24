---
title: PlainLaw
emoji: ⚖️
colorFrom: gray
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# PlainLaw

Describe what happened in your own words and get the sections of India's new criminal laws (BNS and BNSS, 2023)
that may apply, what to do next, and the exact statute text with the old IPC / CrPC number.

Every law shown is retrieved from the official Act text, never generated. An optional AI model (Groq; set the
`LLM_KEY` secret) only picks which of 26 checked situation guides fits the message.

API: `POST /api/chat`, `GET /api/health`. General information, not legal advice. In danger? Call 112.
