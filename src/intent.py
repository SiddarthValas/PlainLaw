"""Is the user describing their own situation, or asking what a law says?

"what is the punishment for murder" asks for a definition, so the bot shows the section itself.
"someone killed my dog, what do I do" describes a situation, so the bot shows a scenario plan.
"""
from __future__ import annotations

import re

_DEF_START = re.compile(
    r"^\s*(?:what(?:'s| is| are| does| do| counts? as| amounts? to)|define|definition of|meaning of|explain|which section|what section|"
    r"under which section|punishment (?:for|of)|penalty (?:for|of)|is (?:it )?(?:a |an )?(?:crime|offence|offense|illegal)|"
    r"is [\w\s-]{2,40} (?:a )?(?:crime|offence|offense|illegal|punishable|bailable|cognizable))\b", re.I)
_PERSONAL = re.compile(
    r"\b(?:i|i'm|im|i've|me|my|mine|we|our|us|myself|someone|somebody|he|she|they|him|her|them|his|their|"
    r"friend|husband|wife|son|daughter|brother|sister|father|mother|neighbou?r|boss|landlord)\b", re.I)


def is_definition_question(text: str) -> bool:
    """A question about the law itself, with no personal story in it."""
    return bool(_DEF_START.search(text)) and not _PERSONAL.search(text)
