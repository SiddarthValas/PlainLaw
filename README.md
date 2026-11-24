# PlainLaw

**Author:** Siddarth Valasubramanian · B.Tech CSE (Data Science), UPES Dehradun

**Live demo:** public link on Vercel coming soon.

## How to run it

You need Python 3.10 or newer ([python.org](https://www.python.org/downloads/)). The first start installs everything
PlainLaw needs into a private `.venv` folder and downloads the embedding model (a few minutes, one time only).

**macOS:** double-click `start.command` (or run `./start.command` in Terminal). The chat opens at http://localhost:8000.

**Windows:** double-click `start.bat`. The chat opens at http://localhost:8000.

**Any system, manually:**
```
python3 -m venv .venv
.venv/bin/pip install -r requirements-run.txt        # Windows: .venv\Scripts\pip install -r requirements-run.txt
.venv/bin/python -m uvicorn api:app --port 8000      # Windows: .venv\Scripts\python -m uvicorn api:app --port 8000
```
Then open http://localhost:8000.

**Optional, AI features:** create a free API key at [console.groq.com](https://console.groq.com) and save it as a
single line in a file named `groq_key.txt` in the project folder (it is in `.gitignore`, so it is never uploaded).
With the key, an AI model helps understand unusual messages and answers open-ended follow-up questions, and every
section it cites is checked against the official text. Without it, PlainLaw runs fully offline on its rules,
search and trained classifier.

Run the tests: `.venv/bin/python tests/test_playbooks.py` (see "Tests and evaluation" below for the full list).

Describe a problem in everyday words and get the laws of India's new criminal codes that may apply, what to do next, and
the exact statute text with the old IPC / CrPC number. Covers the Bharatiya Nyaya Sanhita (BNS, offences) and the Bharatiya
Nagarik Suraksha Sanhita (BNSS, police and court procedure), 2023. Built for CSEG4134.

## A conversation, not a one-off report
PlainLaw keeps the context of the conversation (`src/dialogue.py`). The browser holds a small state (the situation being
discussed, the sections already shown, the last section explained, a question the assistant asked, facts such as
"safe" and "when") and sends it back with each message. Example:

| You | PlainLaw |
|---|---|
| Someone touched me inappropriately on a public bus. | I'm sorry that happened. Are you safe right now? ... may fall under provisions of the BNS on sexual harassment, depending on the circumstances. *(buttons: "Yes, I'm safe now" / "No, it's happening now"; the steps are available folded)* |
| Yes, I'm safe now. This happened yesterday. | Understood, I'm glad you're safe. Since it happened yesterday, you can still report it... *(steps, what to note down, sections that may apply)* |
| What does Section 75 actually mean? | BNS 75 in simple language: what it covers (its four clauses) and the punishment, from the section's own text |
| What if I don't have any witnesses? | You can still approach the police; your own account is evidence (BNSS 173); bus route, ticket/UPI payment, CCTV... |

Questions no pattern covers ("shall I take my valuables out?") are answered by the LLM (`src/llm_chat.py`) from the
conversation, the current guide and the official text of the sections on screen; a checker rejects the reply if it cites
any section it was not given, an old IPC/CrPC number, or an unknown phone number, and the user then gets an honest "I
can't answer that reliably". Later turns never repeat the whole plan. "On fire", "trapped" and similar trigger safety
advice at any point. Test the live model with `.venv/bin/python eval_llm_chat.py "$PWD" results/llm_chat_cases.json`.

It understands: answers to its own question, "section 75" / "the first one" / "it" / a bare "118", follow-ups about
the same situation (no witnesses, online report, which station, woman officer, police refusal, delay, lawyer and cost,
privacy, what happens next, compensation, safety), extra details added to the story ("he also had a knife" adds
robbery and intimidation), "what if" refinements of a law question, and a switch to a new matter. Multi-turn check
(`eval_dialogue.py`, 12 scripted conversations, 42 turns): 38/42 on the first run, 40/42 after two bug fixes.

## What the user gets
| The user types | The bot replies with |
|---|---|
| A situation: "someone rear-ended my car, what should I do?" | A **plan** from one of 27 scenario playbooks: numbered steps (green tag = cites a BNSS/BNS section, grey tag = general practical guidance), the laws that may apply with the reason each applies, what to keep ready, and which other law covers the rest (e.g. Motor Vehicles Act). Laws that depend on a circumstance (a death, a weapon, a forged document) only show when the user mentions it. |
| A question about the law: "what is the punishment for murder?" | The matching **sections** (BNS 103 first), plus a "What should I do next?" button when a playbook covers that law. |
| An exact reference: "IPC 420", "BNS 318", "section 302" | The section itself, with the old-to-new mapping. |
| Something unclear | "Is your situation closest to one of these?" with the 3 nearest scenarios as buttons, or plain "closest matches" when it is not sure. |
| A civil matter or another Act (rent, salary, cheque bounce, drugs, bribery, caste abuse) | Says which law covers it and where to go, instead of showing an unrelated criminal section. |
| Self-harm, an emergency in progress, small talk, Hindi/Hinglish | A dedicated reply (Tele-MANAS 14416, 112, ask for English). |

## How it works (NLP pipeline)
1. **Normalisation and guards** (`src/safety.py`, `src/chatbot.py`): curly quotes and spacing normalised; crisis, urgency,
   greetings and non-English input detected with patterns and a Devanagari/Hinglish ratio.
2. **Intent detection** (`src/intent.py`): is this a definition question ("what is X", no personal story) or a situation?
3. **Scenario routing** (`src/playbooks.py`, `data/playbooks.json`): trigger patterns with exclusions and priorities pick
   up to two playbooks. If none fires, sentence embeddings (bge-small) compare the message with 115 example situations
   (nearest neighbour, cosine similarity) and offer the closest 3 as buttons instead of guessing.
4. **Retrieval** (`src/retrieval.py`): Porter-stemmed tokens, BM25 + dense embeddings (BAAI/bge-small-en-v1.5) fused with
   Reciprocal Rank Fusion. Situations use the plain-language bridge (`src/glossary.py`, everyday words -> statute vocabulary);
   definition questions add a third ranked list, BM25 over section titles, plus a table of common legal names
   ("anticipatory bail", "zero FIR", "mob lynching") mapped to the statute's own words.
5. **Grounding and abstention**: section numbers and section text come only from the curated corpus of the official
   Acts (a test checks every cited number exists), so a number outside BNS/BNSS cannot be displayed. That does **not**
   mean a shown section applies to the user's facts: replies say sections *may* apply depending on the facts, and the
   steps are hand-written guidance, not official text. The bot answers directly only when BM25 and the dense model
   agree and the query is specific (validated in EVALUATION.md: Hit@1 52% when it answers vs 24% when it asks).
6. **Trained situation classifier** (`train_router.py`): logistic regression on bge-small embeddings, used offline or
   when the LLM is unreachable (75% vs 49% top-1 for nearest-neighbour on 65 blind situations).
7. **Old dates** (`src/dates.py`): events before 1 July 2024 get a notice that the IPC/CrPC generally apply
   (BNS s.358, BNSS s.531).
8. Optional (`USE_LLM=1`, local Ollama): one plain-words line per card, dropped unless every number in it is in the section.

## Run
```
pip install -r requirements.txt
uvicorn api:app --port 8000          # new web UI + API at http://localhost:8000
streamlit run app.py                 # the earlier Streamlit UI (same logic)
```
Deployment: `deploy/space/` is a Hugging Face Docker Space (API + UI), `deploy/vercel/` is the same UI as a static
site on Vercel, pointed at the Space through `config.js`.
The embedding model loads from `models/bge-small-en-v1.5` if that folder exists (no internet needed), otherwise it downloads
from Hugging Face on first start (it is not in this repository: 129 MB). Optional AI features need a Groq API key in a
file `groq_key.txt` in the project folder (never commit it; it is in `.gitignore`). To create the local copy once:
`python3 -c "from sentence_transformers import SentenceTransformer as S; S('BAAI/bge-small-en-v1.5').save('models/bge-small-en-v1.5')"`.
The first start builds the section embeddings (about a minute) and caches them in `.cache/`. `DENSE_MODEL=lsa` forces an
offline TF-IDF + LSA fallback (no suggestions in that mode).

Tests and evaluation:
```
python3 data_audit.py                                # corpus integrity + data/VERSION.json
python3 train_router.py                              # trains models/situation_clf.npz and compares it on blind sets
python3 eval_final.py [--llm]                        # frozen 150-question test: baselines, abstention, full system
python3 eval_grounded_llm.py                         # baseline E: LLM given the same retrieved sections (needs LLM_KEY)
python3 eval_safety.py                               # 51 safety cases
python3 eval_dialogue.py                             # 12 multi-turn conversations (context handling)
python3 tests/test_dates.py; python3 tests/test_safety.py
python3 tests/test_playbooks.py                      # every cited section exists; 38 routing cases
DENSE_MODEL=lsa python3 tests/test_bot_qa.py         # 23 end-to-end behaviour checks
DENSE_MODEL=lsa python3 tests/test_conditional_laws.py
DENSE_MODEL=lsa python3 tests/smoke_test.py          # synthetic fixture
python3 eval.py --split test                         # retrieval systems compared (also --split dev, or the fresh file)
python3 eval_situations.py --queries data/eval_situations2.jsonl   # scenario routing
python3 vocab_gap.py                                 # how many query words appear in the gold sections
```

## Evaluation summary (full details, contamination ledger and limitations: [EVALUATION.md](EVALUATION.md))
Frozen held-out test (`data/eval_final.jsonl`, 150 **synthetic** questions written after the system was frozen; not
real users, not lawyer-checked), run once:

| Measure | Result (95% CI) |
|---|---|
| Retrieval, hybrid + expansion (119 with gold): Hit@1 / Hit@3 / MRR | 35% (27–44%) / 55% / 0.458 |
| Full system offline (rules + trained classifier): right guide or right non-answer | 79% (72–85%); wrong 10%; asked 11% |
| Same, 130 questions with no near-duplicate in older data | 76% (68–83%) |
| A gold section among the sections shown | 79% (71–85%) |
| Old-date notice | 5/5, no false notices |
| Safety cases (emergency, self-harm, civil, other Acts, scams, word traps) | 31/39 frozen → 49/51 after fixes (12 written after the fixes: 8/12 first run) |
| Latency (CPU) | median 1 ms, p95 90 ms; API p95 342 ms at 10 concurrent requests |

The earlier figures below come from development sets that were used while building the system, so they are optimistic.

## Results on development sets (bge-small, 889 sections)
Retrieval on plain-language queries (Hit@3 / MRR@10):

| System | dev (24) | test (27) | fresh (21) |
|---|---|---|---|
| BM25 | 37.5% / 0.334 | 70.4% / 0.574 | 28.6% / 0.218 |
| Dense (bge-small) | 58.3% / 0.539 | 81.5% / 0.713 | 19.0% / 0.220 |
| Hybrid (RRF) | 70.8% / 0.481 | 74.1% / 0.686 | 23.8% / 0.279 |
| Hybrid + rewriting (app default for situations) | 87.5% / 0.751 | 63.0% / 0.529 | 38.1% / 0.344 |
| Hybrid + passages + rewriting | 83.3% / 0.741 | 63.0% / 0.549 | 38.1% / 0.310 |

Reading it honestly: rewriting rules were written on dev, so dev is optimistic. On test (mostly procedure questions that already
use statute words) dense alone is best and rewriting hurts; on fresh (everyday offence descriptions with a large vocabulary
gap) rewriting is what helps. The abstain gate matters most: when the bot answers directly, Hit@3 rises to 78.9% on test
(coverage 70%) and 66.7% on fresh (coverage 29%).

Definition questions (`data/eval_definitions.jsonl`, 20): hybrid + title match + legal-name table puts the gold section in the
cards for 20/20, first for 18/20 (hybrid alone: top 3 for 17/20). A section whose whole title the user typed is put first.

Scenario routing: on a held-out list of 30 situations written after the rules, the right playbook came first for 25/30 (83%)
with patterns alone, and the 3 embedding suggestions contained the right playbook for all 28 situations that should have one. Those misses were then fixed, so
both situation lists are now development data.

## Data
`data/sections.jsonl`: one JSON object per section: `id` (e.g. `BNS-303`), `act`, `section`, `title`, `text`, `replaces`
(e.g. `IPC 378`). Text from the official Act PDFs (BNS as on 6 Oct 2025 from India Code; BNSS as enacted); old numbers
from the CAPT Bhopal comparison tables. See `data/README.md` for every file. `tests/fixture_sections.jsonl` is synthetic.

## Limits
Hand-written playbooks cover 27 common situations; anything else falls back to retrieval. English only. BNS 106(2)
(hit-and-run) is shown as not yet in force (it had not been notified as of mid-2026). Other Acts (Motor Vehicles, IT, POCSO,
NDPS, Domestic Violence) are only pointed to, not searched. Helpline numbers can change.

Not legal advice. In danger? Call 112.
