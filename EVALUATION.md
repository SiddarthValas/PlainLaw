# PlainLaw: audit, evaluation and limitations

This file records the audit of the project (24 Nov 2026), what was changed, how the system was evaluated, and what
the results do and do not show. Every number below was produced by a script in this repository. Where a result could
not be produced here (the grounded-LLM baseline needs an API key), the script is provided and no number is given.

## 1. Audit findings

| # | Issue raised | What the audit found | Priority | Status |
|---|---|---|---|---|
| 6 | Valid citation ≠ applicable law | UI said "may apply", but some replies read as conclusions ("the part of the law that fits best") | P0 | Fixed: qualified wording in replies and on every plan |
| 8 | Date of the events | Not handled: events before 1 July 2024 are generally governed by the IPC (BNS s.358) and pending matters by the CrPC (BNSS s.531) | P0 | Fixed: date detector + notice (`src/dates.py`, 13 tests) |
| 8 | Dataset integrity | Complete: 358/358 BNS, 531/531 BNSS, no duplicate ids, no gazette/amendment noise. Mappings are not one-to-one (54 new sections replace several old ones; 5 old sections split) | P0 | Audit script + `data/VERSION.json` (source hashes, text versions, known gaps) |
| 7 | Overconfident claims | "never generated", "every law shown is from the Act", "0% guaranteed", "nobody has a dataset", "better than ChatGPT" in README, UI, report, explainer | P0 | Rewritten (section 6) |
| 2 | Test contamination | Every situation set (1–5) was used to fix rules; the retrieval test split was used for model selection; 11 playbook examples are near-copies (cosine ≥ 0.90) of evaluation questions; 4 questions appear in two sets | P1 | Contamination ledger (section 3); new frozen test set |
| 1 | Small evaluation sets | 20–50 per set, no frozen final test | P1 | 150-question frozen synthetic set + 51 safety cases |
| 4 | Retrieval metrics | Hit@1/3/5 and MRR existed; recall@k, wrong-answer rate, coverage, out-of-scope rejection, CIs, paired tests missing | P1 | Added (`eval_final.py`) |
| 5 | Abstention unvalidated | The BM25/dense agreement gate had never been checked against correctness | P1 | Risk–coverage + error-correlation analysis |
| 3 | Unfair LLM comparison | LLMs were tested without the statute text | P1 | Baselines A–D, F run; E (grounded LLM) script provided; old comparison relabelled "ungrounded" |
| 9 | Safety testing | 2 self-harm tests only | P2 | 51-case safety evaluation; 3 safety bugs found and fixed |
| 10 | Performance | Never measured | P2 | Latency, memory, load test, LLM-failure behaviour |
| A | Expansion drift | "stole my phone" always added "snatching" | P1 | Fixed; drift measured per question (section 5.6) |

## 2. Changes made (and why)

| Change | Files | Reason |
|---|---|---|
| Old-date notice (BNS 358 / BNSS 531) | `src/dates.py`, `src/chatbot.py`, `tests/test_dates.py` | Legal correctness: the new codes do not govern earlier offences |
| Qualified wording ("may apply depending on the exact facts"; note under every list of laws; steps marked as app guidance, not official text) | `src/chatbot.py`, `web/index.html` | A valid section is not a legal conclusion |
| Theft rule no longer adds "snatching" | `src/glossary.py` | Semantic drift; fresh-set MRR 0.344 → 0.368, dev/test unchanged |
| Trained situation classifier (logistic regression on bge-small embeddings, 195 training texts, C by 5-fold CV, threshold by CV) used offline or when the LLM call fails; also ranks the "is it one of these?" suggestions | `train_router.py`, `src/playbooks.py`, `src/chatbot.py`, `models/situation_clf.npz` | Measured gain over 1-nearest-neighbour (section 5.2); weights stored as plain arrays, no pickle |
| **After** the frozen run (safety bugs it exposed): pronoun-independent self-harm detection; a story/game/code guard ("a novel about a robbery"); custody-scam trigger to the fraud guide (never the arrest guide); civil rules for inheritance, IP, name change, custody of kids, utility boards, banks | `src/safety.py`, `data/playbooks.json`, `src/glossary.py`, `tests/test_safety.py` | Safety. These changes are post-hoc: the frozen numbers remain the official ones |
| LLM failure → classifier fallback; LLM timeout 20 s → 8 s (`LLM_TIMEOUT`) | `src/chatbot.py`, `src/llm_router.py` | With the API configured but down, the app did worse than offline |
| **Conversation layer**: state kept by the browser (topic, sections shown, focus, pending question, facts); safety check-in for personal-harm situations; section explanations built from the section's own text (clauses + simplified punishment); 13 follow-up intents with answers citing BNSS 173, 175, 176, 193, 396, 398, 514 and BNS 72, 351; anaphora ("it", "the first one", bare numbers); story continuation; topic switch | `src/dialogue.py`, `api.py`, `web/index.html`, `tests/test_dialogue.py`, `eval_dialogue.py` | The faculty asked for an ongoing, context-aware assistant rather than a report generator |
| **Grounded chat replies**: open-ended in-conversation questions answered by the LLM from the conversation, the guide and the sections on screen; rejected unless every cited section was given, no unlisted IPC/CrPC number and no unknown phone number appear; honest fallback otherwise; continuation turns no longer repeat the plan; fire/"trapped" detected at any turn | `src/llm_chat.py`, `src/dialogue.py`, `src/safety.py`, `tests/test_llm_chat.py`, `eval_llm_chat.py` | User testing showed open-ended follow-ups ("shall I take my valuables out?") were not understood and long conversations repeated the plan. This changes the earlier design claim: the LLM now writes conversational text, but legal citations remain verified against the corpus |
| Dataset audit + version metadata | `data_audit.py`, `data/VERSION.json` | Provenance; corrections must be traceable to the source PDFs |

All tests pass after every change: `tests/test_dialogue.py` (follow-up citations exist; the faculty conversation), `tests/test_playbooks.py` (27 guides, 38 trigger cases, every cited section
exists), `test_bot_qa.py` (23), `test_conditional_laws.py` (8), `test_dates.py` (13), `test_safety.py` (17),
`smoke_test.py`.

## 3. Data splits and contamination ledger

| Data | Size | Used for | Status for reporting |
|---|---|---|---|
| `eval_queries.jsonl` dev | 24 | writing glossary rules | development only |
| `eval_queries.jsonl` test | 27 | model selection | **contaminated** (used to choose systems) |
| `eval_fresh.jsonl` | 21 | first run clean, then error analysis | first-run numbers only |
| `eval_definitions.jsonl` | 20 | title-match design | development |
| `eval_situations.jsonl`, `…2` | 50 + 30 | rule fixing; classifier training | development / training |
| `eval_situations3/4/5` | 20 + 20 + 25 | first run blind, then rule fixing; classifier test | first-run numbers only; later rule scores optimistic |
| `playbooks.json` examples | 115 | classifier training, LLM prompt | 11 are near-copies of evaluation questions |
| **`eval_final.jsonl`** | **150** | **final test, run once on the frozen system** | **official held-out result** |
| `eval_safety.py` cases | 39 + 12 holdout | safety measurement | 39 used for fixes; 12 holdout written after fixes |

**About `eval_final.jsonl`.** It is **synthetic**: written by the AI assistant (Claude) on 24 Nov 2026 *after* the
system was frozen (`results/frozen_config.json` holds the SHA-256 of every source file, rule file and model at
21:52:22), not collected from real users, and not checked by a lawyer. Gold sections were chosen from the statute
titles and text and validated to exist. The same author also wrote earlier playbook examples, so 20 of the 150
questions are near-duplicates (cosine ≥ 0.90) of older data (4 exact). Results are given for all 150 and for the
**130 clean** questions. Categories: easy 30, medium 30, hard 20, typo/informal 15, ambiguous 12, multi-issue 10,
not covered by any guide 13, out of scope 15, events before July 2024 5. 119 questions have gold sections
(several acceptable sections per question).

Reproducibility: bge-small-en-v1.5 (local copy in `models/`), RRF k = 60, classifier `random_state=0`,
`StratifiedKFold(5, shuffle=True, random_state=0)`, scikit-learn 1.9.1 (training only), Python 3.11+.

## 4. How to reproduce

```
python3 data_audit.py                       # corpus integrity, writes data/VERSION.json
python3 train_router.py                     # classifier: CV, blind-set comparison, saves models/situation_clf.npz
python3 eval_final.py                       # frozen test: baselines A-D, abstention analysis, full system F, latency
python3 eval_final.py --llm                 # + LLM router (needs LLM_BASE/LLM_KEY or Ollama)
python3 eval_grounded_llm.py                # baseline E: grounded LLM (needs LLM_BASE/LLM_KEY), or --local
python3 eval_safety.py                      # 51 safety cases
python3 eval_dialogue.py                    # 12 multi-turn conversations
python3 bench_api.py --url http://127.0.0.1:8000 --n 100 --c 10   # load test (start uvicorn first)
```

## 5. Results

### 5.1 Retrieval baselines on the frozen set (119 questions with gold sections)

| System | Hit@1 (95% CI) | Hit@3 | Hit@5 | MRR@10 | Recall@5 | Median ms | McNemar Hit@3 vs D |
|---|---|---|---|---|---|---|---|
| A  BM25 | 10% (6–17%) | 23% | 28% | 0.176 | 21% | 1 | p < 0.001 |
| B  Dense (bge-small) | 25% (18–34%) | 45% | 50% | 0.371 | 37% | 23 | p = 0.099 |
| C  Hybrid BM25 + dense (RRF) | 24% (17–32%) | 40% | 49% | 0.341 | 39% | 23 | p = 0.002 |
| **D  Hybrid + glossary expansion** | **35% (27–44%)** | **55%** | **57%** | **0.458** | **47%** | 24 | – |
| E  Grounded LLM over D's top 8 | *run `eval_grounded_llm.py`* | | | | | | |

Reading: everyday situation descriptions are hard for retrieval alone (Hit@1 35% for the best system). Glossary
expansion is the largest single gain (C → D, significant), but D is not significantly better than dense retrieval
alone on Hit@3 (p = 0.099). Fusion without expansion (C) did not beat dense alone.

### 5.2 Trained situation classifier (before the freeze; `train_router.py`)

Trained on 115 playbook examples + situation sets 1–2 (195 texts, 28 classes incl. "none"); tested on blind sets
3–5 (65 situations), which the classifier never saw.

| Method (same bge features) | Top-1 on 65 (95% CI) | Top-3 | Macro-F1 |
|---|---|---|---|
| 1-NN over playbook examples (previous suggestions) | 49% (37–61%) | 75% | 0.45 |
| 1-NN over examples + dev | 63% (51–74%) | 86% | 0.59 |
| Logistic regression, examples only | 63% (51–74%) | 78% | 0.58 |
| **Logistic regression, examples + dev** | **75% (64–84%)** | **94%** | **0.67** |

5-fold CV on training data chose C = 20 (91.0% CV accuracy). The confidence threshold (0.3) is the lowest at which
cross-validated accuracy when answering reaches 95%; a first rule (90% → threshold 0.2) answered vague messages such
as "something bad happened with my neighbour" and was rejected before the freeze.

### 5.3 Full system on the frozen set (official, run once)

"Correct" = the right guide first, or no guide when none should be shown, or a clarifying question for ambiguous
messages. "Asked" = the bot asked instead of answering (abstained). "Wrong" = a wrong guide or a crime guide for a
non-crime.

| System | Subset | Correct (95% CI) | Asked | Wrong | Coverage | Accuracy when answering |
|---|---|---|---|---|---|---|
| F1 Rules only (offline) | all 150 | 67% (59–74%) | 33 | 16 (11%) | 78% | 86% |
| F1 Rules only (offline) | clean 130 | 64% (55–72%) | 32 | 15 (12%) | 75% | 85% |
| **F2 Rules + trained classifier** | all 150 | **79% (72–85%)** | 16 | 15 (10%) | 89% | 89% |
| **F2 Rules + trained classifier** | clean 130 | **76% (68–83%)** | 16 | 15 (12%) | 88% | 87% |
| F3 + LLM router | | *run `eval_final.py --llm` on a machine with the key* | | | | |

By category (F2): easy 90%, medium 93%, hard 55%, typo/informal 73%, ambiguous 92% (asked as it should), multi-issue
90%, not covered 69%, out of scope 53%, old date 100%. Out-of-scope rejection: strict 7/15 (47%, CI 25–70%).
Old-date notice: 5/5 shown, 0 false notices on the other 145.

Sections actually shown (119 with gold): a gold section among those shown 79% (71–85%); the first shown is gold 66%
(58–74%); 12 times nothing was shown because the bot asked; 2.3 sections shown on average.

**After the post-freeze safety fixes** (not held-out any more, for information only): F2 84% correct, 8% wrong,
out-of-scope rejection 80%.

### 5.4 Abstention: does "BM25 and dense agree" predict correctness?

| Gate (system D) | n | Hit@1 | Hit@3 |
|---|---|---|---|
| ON: answers directly | 48 | 52% (38–66%) | 73% |
| OFF: asks | 71 | 24% (16–35%) | 42% |

Errors are mostly not shared: BM25+expansion and dense were both wrong at rank 1 on 65 questions, but agreed on the
*same* wrong section only 8 times, so agreement is informative. Risk–coverage (error rate at Hit@1 when answering the
most confident fraction; `results/risk_coverage.png`):

| Confidence signal | 25% | 50% | 75% | 100% | AURC |
|---|---|---|---|---|---|
| Agreement rank (max of BM25/dense rank) | 40% | 47% | 57% | 65% | 0.477 |
| RRF score of top hit | 40% | 45% | 56% | 65% | 0.474 |
| RRF margin top1 − top2 | 77% | 60% | 62% | 65% | 0.698 (worse than no signal) |

Conclusion: the gate halves the error rate of what it answers, but even at 25% coverage 40% of direct answers have a
wrong first section. Direct answers therefore show several sections with qualified wording; they are not conclusions.

### 5.5 Safety (`eval_safety.py`)

| Group | Frozen system | After fixes | 12 holdout cases written after the fixes (first run) |
|---|---|---|---|
| Emergency → 112 | 6/6 | 6/6 | 1/1 |
| Self-harm (incl. third person) | 3/6 | 6/6 | 1/3 → 3/3 after making detection pronoun-independent |
| Civil not shown as crime | 6/7 | 7/7 | 1/2 |
| Other Acts pointed to | 5/5 | 5/5 | 1/1 |
| Old dates | 4/4 | 4/4 | 1/1 |
| Crime words in stories/code | 3/6 | 6/6 | 2/3 |
| Custody / digital-arrest scams | 2/3 | 3/3 | 1/1 |
| Multi-issue | 2/2 | 2/2 | – |
| **Total** | **31/39** | **39/39** | **8/12 first run; 10/12 now** |

Still failing: "our startup is stealing customers" (theft guide) and "someone copied my song" (fraud guide, should
point to copyright law). Regex safety rules needed a second fix after the holdout, which shows their limits.

### 5.6 Query expansion: help vs drift (frozen set, C vs D)

Expansion fired on 57/119 questions: the gold section moved up in 31, down in 10, unchanged in 16. Drift comes from
word sense and negation: "my car was **hit** by a truck" and "an auto **hit** my mother" pull in assault sections;
"her husband **doesn't hit** her" (negated) does too; "**hurt** my community's feelings" pulls in physical hurt;
"my dog was **killed** by a car" pulls in homicide. Not fixed (the system is frozen); see section 7.

### 5.7 Engineering

| Measure (CPU, offline, one process) | Value |
|---|---|
| Start-up (corpus, model, cached embeddings) | 4.2 s |
| Latency per message: mean / median / p95 / max | 22 / 1 / 90 / 104 ms |
| Retrieval (system D) median | 24 ms |
| Peak memory | ~1.06 GB (PyTorch + model) |
| API load test: 60 req, concurrency 1 / 6; 100 req, concurrency 10 | 0 errors; p95 30 / 212 / 342 ms; ~120–130 req/s |
| LLM API unreachable | reply still returned in ~20–70 ms; classifier takes over; error recorded |
| LLM router latency | not measured here (needs the key): `eval_final.py --llm` prints it |

### 5.8 Conversation (multi-turn) handling (`eval_dialogue.py`)

12 scripted conversations (42 turns) written after the conversation layer was built (synthetic; a development check,
not held-out): the faculty example, details added to a story, a "not safe" answer, a safety question interrupted by
another question, police refusal then lawyer and delay questions, a topic switch, a "what if" after a law question,
an old IPC number mid-conversation, accident evidence, a missing person, a civil matter inside a criminal one, and a
bare section number.

| Run | Turns correct | Conversations fully correct |
|---|---|---|
| First run | 38/42 (90%) | 8/12 |
| After fixing two bugs ("what happens after I file" matched "how to file"; a bare "118" was not read as a section) | 40/42 (95%) | 10/12 |

A second batch of 15 realistic conversations (49 turns: cyber fraud, dowry, workplace harassment, road rage, a police
bribe, a false case, an elderly parent, online threats, typos, noise, a crisis mid-conversation, a definition chain,
Hinglish, a topic switch, vague openers) was run after that. Roughly half of its follow-up turns were mishandled on the
first run (`results/dialogue_batch2_before.txt`): generic follow-ups ("what should I do first", "which section is this
under") were not recognised, a question was taken as the answer to "are you safe?", an injury detail replaced the topic,
typos ("polise", "wat is the sec") were missed, an elderly parent was routed to cruelty-by-husband, and a police bribe
to the "police called you" guide. Four intents (what to do, which law, arrest, workplace/POSH Act), typo-tolerant
patterns, topic-continuation rules and two routing rules were added; the batch is now handled as intended
(`results/dialogue_batch2_after.txt`), but it is no longer unseen.

Remaining misses: "he broke the car window to take it" (mischief, BNS 324, is not part of the theft guide) and
"what if it was from a house" after "punishment for theft" (BNS 305 not ranked first).

### 5.9 LLM-written chat replies (`eval_llm_chat.py`, live Groq API, run on the developer's Mac)

17 open-ended follow-up messages from 5 conversations, each sent with the material the app would send (the harness
sends every message to the model; in the app, messages covered by fixed answers never reach it).

| Run | Answered | Blocked by the citation check | Failed (rate limit) | Problems found by reading every answer |
|---|---|---|---|---|
| 1 (gpt-oss-120b) | 13 | 0 | 4 | read the 3-day e-signature rule as a reporting deadline; stretched mischief and told the user to report themselves; mentioned 112 needlessly |
| 2 (after prompt rules + a hard block on deadline claims) | 5 | 0 | 12 | none in the 5 answered; daily quota exhausted |
| 3 (120b first, 20b on rate limit; smaller requests) | 15 (14 by 20b) | 0 | 2 (both models limited) | witness statements said to need a woman officer; "the cost of reporting" (reporting is free); seizure sections cited for IMEI tracking |

After run 3, four verified general facts were added to the material (reporting is free; anyone can report; the
woman-officer rule is for the woman reporting; lost phones can be blocked on Sanchar Saathi/CEIR). Not re-run, to save
quota. The citation check never had to fire in these runs; it was tested with fixed bad answers in
`tests/test_llm_chat.py` (11 cases).

### 5.10 Error analysis (F2 on the frozen set: 15 wrong, 16 asked)

| Category | Count | Representative example | Likely cause |
|---|---|---|---|
| Wrong guide from keyword rules | 7 | "they said my son is in custody and I must send money" → arrest guide | Rule for "custody" fired; scam rule needed "call" (fixed after freeze) |
| | | "someone got into my UPI and took 20000" → theft (fraud expected) | "took" matches the theft rule |
| | | "my car was hit…", "dog bit my kid" → assault | Word sense ("hit", "bit") |
| Civil/crime boundary | 4 | "electricity board sent a bill of 50000" → fraud; "can I patent my app" → fraud; rumour to landlord → tenancy (civil) | Missing civil patterns; "landlord" alone triggered tenancy |
| No guide fits, nearest guide forced | 2 | "printing fake 500 rupee notes" → fraud guide | Counterfeiting has no guide |
| Wrong retrieval | 1 | "the SHO told me to come back with proof" → BNSS 330 | Vocabulary gap (no "FIR" word) |
| Should have asked | 1 | "i want to complain" → generic plan | Retrieval agreement on a vague query |
| Asked instead of answering | 16 | typos ("polise not registring my complain"), indirect ("sugar in my bike's petrol tank"), civil matters with no rule | Rules do not handle misspellings; classifier below threshold |

Retrieval-level confusions seen in shown sections: neighbouring sections (stalking 78 vs sexual harassment 75 /
insult to modesty 79), theft 303 vs cheating 318 for digital payments, and vocabulary gaps for uncovered offences
(fake medicines → cheating instead of 276–278; insulting religion → hurt instead of 299).

## 6. Claims, corrected

| Old claim | Corrected statement |
|---|---|
| "PlainLaw can never show a fake or old section number" / "0% guaranteed" | Section numbers shown come only from the curated corpus and are validated against it by a test, so a number that is not in BNS/BNSS cannot be displayed. This says nothing about whether the section is relevant, correctly interpreted, or applicable to the user's facts or dates. |
| "Every answer / every law shown is from the Act" | Section text is quoted from the official Acts. Steps, explanations and "where to go" notes are hand-written guidance by the project team, not official text, and have not been reviewed by a lawyer. |
| "PlainLaw is better than LLMs" | In our test, *ungrounded* LLMs cited repealed or non-existent sections far more often than PlainLaw. That shows the value of grounding on this task with these prompts and models; it does not show PlainLaw is better than an LLM given the same statute text (baseline E, not yet run). |
| "Nobody has a dataset or tool for these laws" | We are not aware of a public benchmark for situation-to-section retrieval on BNS/BNSS; we built a corpus from the official Acts and small evaluation sets. |
| "73% correct vs 6%" | Measured on 48 development queries some rules were written on (optimistic). The held-out figure for retrieval-only top-1 is 35% (D) and the full system shows a gold section among its cards for 79%. |

Established techniques (not claimed as ours): BM25, dense embeddings, RRF, query expansion, logistic regression,
LLM classification. Project-specific work: the corpus and its old-code mapping, the plain-language glossary, routing
constrained to verified guides, abstention integration and its validation, the old-date and scope handling, and the
end-to-end assistant with its evaluation.

## 7. Limitations

* All evaluation data is synthetic or team-written, single-author, without lawyer-verified labels; n = 150 gives
  confidence intervals of about ±7 points.
* Retrieval on everyday wording is modest (Hit@1 35%); the app compensates by showing several sections and asking.
* Rules are brittle to misspellings and word sense; the post-freeze safety fixes are regex patches and were needed
  twice. A learned safety classifier would generalise better.
* Applicability: the system cannot check legally significant facts (intention, injury level, consent, value), and
  only conditional laws in guides react to some of them.
* The LLM-router (F3) and grounded-LLM (E) numbers on the frozen set have not been produced yet.
* Statute text is as of the dates in `data/VERSION.json`; later amendments are not included.

## 8. Final assessment

| Perspective | Score | Justification |
|---|---|---|
| Undergraduate NLP project | **8.5 / 10** | Covers the core NLP pipeline end to end (corpus building, lexical and dense retrieval, fusion, expansion, intent, supervised classification, LLM classification), with ablations, a frozen held-out test, CIs, paired tests, risk–coverage, error analysis and documented contamination. Loses points for single-author synthetic data and modest retrieval accuracy. |
| AI/ML engineering portfolio | **7.5 / 10** | Working FastAPI service with a dependency-free UI, graceful degradation (LLM → classifier → suggestions), 6 test files, latency/load numbers, reproducible scripts and portable model weights. Weaker: regex-heavy logic, ~1 GB memory, hosting depends on a laptop/tunnel, Windows script untested on real Windows. |
| Research-oriented IR system | **5 / 10** | The evaluation design is sound for its size, but there is no human-annotated benchmark, labels are not lawyer-verified, n is small, the grounded-LLM baseline is pending, and the methods are established ones applied to a new domain. A publishable version would need real user queries, multi-annotator gold labels and comparison with strong rerankers. |
