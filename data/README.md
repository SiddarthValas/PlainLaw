# data/

| File | What it is |
|---|---|
| `sections.jsonl` | The searchable corpus: 358 BNS + 531 BNSS sections (`id`, `act`, `section`, `title`, `text`, `replaces`, `change`). Built by `ingest.py`. |
| `BNS.jsonl`, `BNSS.jsonl` | The same, one file per Act. |
| `BNS_map.jsonl` | IPC to BNS mapping (title, old section numbers, "what changed" note) from the CAPT Bhopal comparison table. |
| `eval_queries.jsonl` | 51 plain-language BNSS queries, split dev (24) / test (27). Dev was used for tuning. **The test split has been used for selection and is no longer clean.** |
| `eval_fresh.jsonl` | 21 newer queries (5 BNSS, 16 BNS) with gold sections chosen from the statute text before any run. Run once: `python3 eval.py --queries data/eval_fresh.jsonl --split fresh`. |
| `eval_unlabelled.jsonl` | 5 queries that are ambiguous or civil (garbage in compound, certificates held by employer, ...). Not scored; used for qualitative checks in the app. |
| `eval_situations.jsonl` | 50 situations with the scenario playbook that should answer them (or `none`). Written before the routing changes were made (they were built on a separate dev list). First run after them: 90% right-playbook-first; then used to fix ordering, so it is now a dev set. |
| `eval_situations2.jsonl` | 30 more situations written after those changes. First (clean) run: 83% right-playbook-first with rules only; the right playbook was in the top-3 suggestions for 28/28. Its misses were then fixed, so it is spent too. |
| `eval_definitions.jsonl` | 20 definition questions ("what is extortion") with gold sections, for the title-match retrieval. |
| `playbooks.json` | 27 hand-written scenario playbooks (laws with reasons, numbered steps, what to keep ready, other laws). Checked against `sections.jsonl` by `tests/test_playbooks.py`. |
| `eval_situations3/4/5.jsonl` | 20 + 20 + 25 situations. First run blind (rules 55% / 40% / 56%), then used to fix rules; also the classifier's test sets. |
| `eval_final.jsonl` | **Frozen held-out test**: 150 synthetic questions written by the AI assistant after the system freeze (`results/frozen_config.json`), with route labels, gold sections (several allowed) and a category. Not real user queries; not lawyer-checked. Run once; see EVALUATION.md. |
| `VERSION.json` | Provenance: SHA-256 of the source PDFs and of `sections.jsonl`, text dates, mapping statistics, known gaps (`python3 data_audit.py`). |
| `source/` | Original PDFs: BNSS Act 46 of 2023, BNS Act 45 of 2023 (India Code, as on 6 Oct 2025), and the two comparison tables. |

Section text comes only from the Act PDFs. The comparison tables supply old IPC/CrPC numbers and the change note.
Rebuild: see the commands at the top of `ingest.py`.
