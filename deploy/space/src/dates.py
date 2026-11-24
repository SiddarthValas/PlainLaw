"""Which law applies depends on when the act happened. BNS and BNSS came into force on 1 July 2024.
Offences committed before that date remain punishable under the Indian Penal Code (BNS section 358, repeal and
savings), and appeals, trials, inquiries and investigations pending on that date continue under the old CrPC
(BNSS section 531). This module only spots dates in the message; it never decides which law applies."""
import re

_NUM = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
_MONTH_H1 = r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?)"
_BEFORE = re.compile(
    r"(?<![₹\d,.])\b(?:19\d\d|200\d|201\d|202[0-3])\b(?!\s*(?:rupees|rs\b|inr|/-|km|kg|people|times))"   # a year before 2024
    rf"|\b{_MONTH_H1}\.?,?\s+(?:\d{{1,2}}(?:st|nd|rd|th)?,?\s+)?2024\b"         # Jan-June 2024
    rf"|\b\d{{1,2}}(?:st|nd|rd|th)?\s+{_MONTH_H1}\.?,?\s+2024\b"
    r"|\bbefore (?:1(?:st)? )?july,? 2024\b|\bbefore the new (?:laws?|codes?)\b|\bunder the (?:old )?ipc\b", re.I)
_YEARS_AGO = re.compile(r"\b(\d+|two|three|four|five|six|seven|eight|nine|ten)\s+years?\s+(?:ago|back)\b", re.I)
_MAYBE = re.compile(r"\b2024\b", re.I)
_H2_2024 = re.compile(r"\b(?:july|jul|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?,?\s+"
                      r"(?:\d{1,2}(?:st|nd|rd|th)?,?\s+)?2024\b|\b\d{1,2}(?:st|nd|rd|th)?\s+(?:july|jul|aug\w*|sep\w*|oct\w*|nov\w*|dec\w*)\.?,?\s+2024\b", re.I)

NOTE = ("When it happened matters. BNS and BNSS apply from 1 July 2024. If this happened before that date, the offence "
        "is generally dealt with under the old Indian Penal Code (BNS section 358 keeps the old penalties alive), and an "
        "investigation or case already pending on that date continues under the old CrPC (BNSS section 531). Each "
        "card shows the old IPC/CrPC number. A lawyer or free legal aid (NALSA 15100) can confirm which applies.")


def old_law_note(text: str) -> str | None:
    """A note when the message dates the events (possibly) before 1 July 2024, else None."""
    if _BEFORE.search(text):
        return NOTE
    m = _YEARS_AGO.search(text)
    if m:
        n = _NUM.get(m.group(1).lower()) or (int(m.group(1)) if m.group(1).isdigit() else 0)
        if n >= 2:
            return NOTE
    if _MAYBE.search(text) and not _H2_2024.search(text):   # 2024 without a month: could be either side of 1 July
        return NOTE
    return None
