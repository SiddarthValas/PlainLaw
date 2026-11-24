"""Guards that run before any legal matching: self-harm, small talk, non-English input, urgency."""
from __future__ import annotations

import re

CRISIS = re.compile(   # pronoun-independent: the user or someone they know may be at risk
    r"\b(?:kill(?:ing|ed)? (?:myself|himself|herself|themselves|themself)|"
    r"(?:end|ending|ended|take|taking|took) (?:my|his|her|their) (?:own )?life|suicid\w*|"
    r"(?:want|wants|wanted|wanting|wanna) to die|(?:don'?t|doesn'?t|didn'?t|not|no longer|never) (?:want|wants|wanting) to (?:live|be alive|exist)|"
    r"(?:hurt|hurting|harm|harming|cut|cutting) (?:myself|himself|herself|themselves)|self[- ]?harm|"
    r"(?:no|any) reason to (?:live|go on)|better off dead)\b", re.I)
# messages that use crime words in a story, a game, sport or code, not a real incident
NONLITERAL = re.compile(
    r"\b(?:novel|book|movie|film|web ?series|tv (?:show|series)|story|plot|episode|video game)\b.{0,30}\b(?:about|on|with|had|has|where|in which|of)\b"
    r"|\b(?:surprise attack|attack(?:ing)? (?:strategy|plan)|killed it|nailed it|steal(?:ing)? the show)\b"
    r"|\b(?:code|program|script|software|ram|cpu|memory leak|process)\b.{0,40}\b(?:steal\w*|kill\w*|attack\w*|leak\w*)\b", re.I)
NONLITERAL_TEXT = ("This sounds like it may be about a story, a game, sport or code rather than something that happened to you. "
                   "If something did happen to you or someone you know, tell me in a sentence or two what was done, and I'll "
                   "show the laws that may apply and what to do next.")
GREETING = re.compile(r"^\s*(?:hi+|hello+|hey+|hii+|namaste|good (?:morning|afternoon|evening)|yo|sup)\W*$", re.I)
THANKS = re.compile(r"^\s*(?:thanks?(?: you)?|thank u|ty|ok(?:ay)?|cool|got it|bye|goodbye)\W*$", re.I)
URGENT = re.compile(
    r"\b(?:right now|happening now|at this moment|as we speak|is being (?:attacked|beaten|raped|assaulted|robbed|chased)|"
    r"(?:please )?help me|in danger|following me now|(?:for|since) (?:the (?:last|past) )?\d+ ?(?:min|mins|minutes)\b|"
    r"on fire|catch(?:ing|es)? fire|caught fire|(?:is|are) burning|smoke (?:is )?coming|trapped|explod\w*|blast)\b", re.I)
FIRE = re.compile(r"\b(?:on fire|catch(?:ing|es)? fire|caught fire|(?:is|are) burning|smoke (?:is )?coming|flames?|explod\w*)\b", re.I)
FIRE_TEXT = ("If there is a fire, move away from it, keep others away and call 101 (fire) or 112 now. Do not go back for "
             "belongings or try to start or move the vehicle; your safety matters more than anything inside.")
_DEVANAGARI = re.compile(r"[ऀ-ॿ]")
_HINGLISH = frozenset("mera meri mere mujhe nahi nahin kya hai hain gaya gayi kiya kisi bhai paise chori gaadi gadi kaise "
                      "mein humne unhone apna pati biwi ghar peeche thok maara mara".split())

CRISIS_TEXT = (
    "I'm really sorry you're going through this. You don't have to face it alone. "
    "If it is someone else who is at risk, stay with them if you can and get help now. "
    "Please call Tele-MANAS on 14416 (free, 24x7) or 112 right now, or tell someone you trust where you are and how you feel. "
    "This tool only explains criminal law, so it can't help with this, but talking to a person can."
)
GREETING_TEXT = ("Hi. Tell me what happened in your own words, for example “someone rear-ended my car” or "
                 "“the police won't register my FIR”. I'll show the laws that may apply and what to do next.")
THANKS_TEXT = "You're welcome. Take care. If you or anyone is in danger, call 112."
LANGUAGE_TEXT = ("I can only read English for now. Please describe what happened in English, in a sentence or two. "
                 "If it is an emergency, call 112.")
URGENT_PREFIX = "If this is happening right now, call 112 immediately. "


def is_crisis(text: str) -> bool:
    return bool(CRISIS.search(text))


def is_nonliteral(text: str) -> bool:
    """Crime words used about fiction, sport or software ("a novel about a robbery", "code that kills the process")."""
    return bool(NONLITERAL.search(text))


def is_urgent(text: str) -> bool:
    return bool(URGENT.search(text))


def is_non_english(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    if letters and len(_DEVANAGARI.findall(text)) / len(letters) > 0.3:
        return True
    words = set(re.findall(r"[a-z]+", text.lower()))
    return len(words & _HINGLISH) >= 2
