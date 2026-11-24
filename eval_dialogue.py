"""Multi-turn evaluation: does the assistant keep the context of a conversation?

    python3 eval_dialogue.py          # offline (rules + trained classifier); prints per-turn results and totals

Each conversation is a list of (user message, expected reply kind, text or section that must appear). Checks cover:
answering the assistant's own question, explaining a section by number / by "the first one" / by "it", follow-up
questions about the same situation, extra details added to the story, a switch to a new topic, "what if" refinements
of a law question, and old IPC numbers mid-conversation. Written by the AI assistant after src/dialogue.py was built
(synthetic; a development check, not a held-out benchmark). `ANY` accepts any reply kind.
"""
import sys
import warnings

warnings.filterwarnings("ignore")
import src.chatbot as cb                                                     # noqa: E402
from src.dialogue import Conversation                                       # noqa: E402
from src.retrieval import DEFAULT_DENSE_MODEL, Retriever, load_sections     # noqa: E402

ANY = None
CONVS = [
    ("faculty example", [
        ("Someone touched me inappropriately on a public bus.", "plan", "Are you safe"),
        ("Yes, I'm safe now. This happened yesterday.", "plan", "yesterday"),
        ("What does Section 75 actually mean?", "explain", "BNS-75"),
        ("What if I don't have any witnesses?", "followup", "not the only possible source"),
    ]),
    ("theft, details added", [
        ("my laptop was stolen from my car last night", "plan", "theft"),
        ("can I report it online?", "followup", "electronic"),
        ("explain the first one", "explain", "BNS-303"),
        ("and he broke the car window to take it", "plan", "BNS-324"),
    ]),
    ("domestic, not safe", [
        ("my husband hits me when he is drunk", "plan", "Are you safe"),
        ("no, he is still at home", "plan", "112"),
        ("what does BNS 85 say", "explain", "BNS-85"),
        ("will my family find out if I complain?", "followup", "confidential"),
    ]),
    ("stalking, interrupted safety question", [
        ("a man keeps following me to my office every day", "plan", "Are you safe"),
        ("what does the first one mean", "explain", "BNS-78"),
        ("yes I'm safe at home", "plan", "glad you're safe"),
        ("can a woman officer take my complaint?", "followup", "woman"),
    ]),
    ("police refuse, then lawyer and delay", [
        ("the police are not registering my complaint about my stolen bike", "plan", "fir_refused"),
        ("do I need a lawyer for this?", "followup", "15100"),
        ("it happened two weeks ago, is it too late?", "followup", "does not stop you"),
    ]),
    ("topic switch", [
        ("someone cheated me of 30000 rupees on a fake job offer", "plan", "fraud"),
        ("what happens after I file the complaint?", "followup", "BNSS 193"),
        ("also my neighbour broke into my house yesterday", "plan", ANY),
    ]),
    ("law question then what-if", [
        ("what is the punishment for theft", "answer", "BNS-303"),
        ("what if it was from a house", "answer", "BNS-305"),
        ("explain the first one", "explain", ANY),
    ]),
    ("old number mid-conversation", [
        ("someone threatened to kill me over a land dispute", "plan", "Are you safe"),
        ("yes", "plan", "BNS-351"),
        ("what was IPC 506", "mapped", "BNS-351"),
        ("what is the punishment for it", "explain", "BNS-351"),
    ]),
    ("accident, injury detail", [
        ("a bike hit me while I was crossing the road", "plan", "Are you safe"),
        ("yes I'm okay but my leg is broken", "plan", ANY),
        ("what evidence should I collect?", "followup", "CCTV"),
        ("how do I file the complaint?", "followup", "BNSS 173"),
    ]),
    ("missing person", [
        ("my brother hasn't come home for two days", "plan", "contact them"),
        ("no", "plan", "don't need to wait"),
        ("which police station should I go to?", "followup", "any police station"),
    ]),
    ("civil inside a conversation", [
        ("my phone was snatched near the market", "plan", "theft"),
        ("my landlord is also refusing to return my deposit", "out_of_scope", "civil"),
    ]),
    ("bare number and compensation", [
        ("a man attacked me with a rod and I needed stitches", "plan", "Are you safe"),
        ("yes I'm safe", "plan", ANY),
        ("118", "explain", "BNS-118"),
        ("can I get compensation?", "followup", "BNSS 396"),
    ]),
]


def check(r, kind, must) -> bool:
    if kind is not None and r.kind != kind:
        return False
    if must is None:
        return True
    blob = " ".join(r.say or [r.text]) + " " + " ".join(r.notes) + " " + " ".join(r.understood)
    ids = [h["id"] for h in r.hits] + [l["id"] for p in r.plans for l in p["laws"]] + [p["id"] for p in r.plans]
    if r.explain:
        ids.append(r.explain["id"])
    return must in ids or must.lower() in blob.lower()


def main() -> int:
    ret = Retriever(load_sections("data/sections.jsonl"), dense_model=DEFAULT_DENSE_MODEL)
    conv = Conversation(cb.LegalBot(ret, router=False))
    ok = total = conv_ok = 0
    for name, turns in CONVS:
        st, all_ok = {}, True
        print(f"\n## {name}")
        for msg, kind, must in turns:
            r, st = conv.reply(msg, st)
            good = check(r, kind, must)
            ok += good; total += 1; all_ok &= good
            print(f"  {'ok  ' if good else 'FAIL'} {msg[:60]:62} -> {r.kind}"
                  + ("" if good else f" (expected {kind} with '{must}'; got: {(' '.join(r.say) or r.text)[:90]})"))
        conv_ok += all_ok
    print(f"\nTurns handled correctly: {ok}/{total} ({ok/total:.0%}); conversations fully correct: {conv_ok}/{len(CONVS)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
