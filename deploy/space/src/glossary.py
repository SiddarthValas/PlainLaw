"""Plain-language -> legal-vocabulary bridge (the query-rewriting layer).

Each rule maps everyday phrasing to the words statutes actually use. Rules hold
vocabulary only, never section numbers, so they can change what is retrieved but
cannot put a wrong citation in front of a user.

kind: "offence" (BNS-type), "process" (BNSS-type), "civil" (outside criminal law), "other" (criminal, but under
another Act such as NDPS or the NI Act). Civil and other rules add no search terms; they drive the out-of-scope
reply, and "other" rules also attach a pointer to the right Act.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# (regex, legal terms appended to the query, label shown to the user, kind)
RULES = [
    # ---------------- offences ----------------
    (r"\b(stole|stolen|steal\w*|theft|thie(?:f|ves)|pickpocket\w*|snatch\w*|shoplift\w*|(?:took|taken|taking)\b.{0,30}\bfrom (?:my|our|the) (?:wallet|pocket|bag|purse|locker|drawer|cupboard)|(?:took|taken|taking)\b.{0,40}\bwithout (?:my |our )?(?:permission|knowing|consent|asking))\b",
     "theft dishonestly movable property", "theft or snatching", "offence"),
    (r"\b(robbed|robbery|rob|mugged|mugging|looted|dacoit\w*|held me up)\b",
     "robbery dacoity theft extortion hurt fear", "robbery", "offence"),
    (r"\b(cheat\w*|fraud\w*|scam\w*|conned|duped|swindl\w*|fake (?:job|offer|website|seller))\b",
     "cheating dishonestly inducing delivery property deceive fraud", "cheating or fraud", "offence"),
    (r"\b(misappropriat\w*|embezzl\w*|entrust\w*|safe ?keeping|breach of trust)\b",
     "criminal breach of trust entrusted dominion dishonest misappropriation", "breach of trust", "offence"),
    (r"\b(beat|beaten|beating|hit|slapp\w*|punch\w*|kick\w*|attack\w*|assault\w*|injur\w*|hurt|stabb?\w*|thrash\w*)\b",
     "hurt grievous hurt voluntarily causing assault criminal force weapon", "physical assault or injury", "offence"),
    (r"\b(murder\w*|kill\w*|homicide|died|death)\b",
     "murder culpable homicide death punishment", "killing or death", "offence"),
    (r"\b(threat\w*|blackmail\w*|extort\w*|intimidat\w*|ransom)\b",
     "criminal intimidation threat injury extortion fear", "threats or blackmail", "offence"),
    (r"\b(eve[- ]?teas\w*|catcall\w*|molest\w*|grop\w*|sexual\w* harass\w*|touched me (?:inappropriately|wrongly)|outrag\w* (?:my )?modesty)\b",
     "sexual harassment assault criminal force woman outrage modesty", "sexual harassment or molestation", "offence"),
    (r"\b(rape\w*|sexual\w* assault\w*|forced (?:me )?(?:to )?(?:have )?sex)\b",
     "rape sexual intercourse without consent woman", "sexual offence", "offence"),
    (r"\b(stalk\w*|follow(?:s|ed|ing)? me|keeps? (?:calling|messaging|texting) me)\b",
     "stalking woman follows contacts monitors electronic communication", "stalking", "offence"),
    (r"\b(secretly (?:film\w*|record\w*|photograph\w*|click\w*)|hidden camera|spy ?cam|private (?:videos?|photos?|images?) (?:leak\w*|shar\w*|viral))\b",
     "voyeurism captures image woman private act watching", "private images or secret recording", "offence"),
    (r"\b(dowry|in[- ]?laws?|husband (?:beats?|harass\w*|tortur\w*)|domestic violence|cruelty)\b",
     "cruelty husband relatives woman dowry death harassment", "dowry or domestic cruelty", "offence"),
    (r"\b(kidnap\w*|abduct\w*|missing (?:child|person|girl|boy|daughter|son)|taken away forcibly)\b",
     "kidnapping abduction child minor lawful guardianship", "kidnapping or abduction", "offence"),
    (r"\b(defam\w*|false (?:rumou?rs?|allegations?|accusations?)|spread\w* (?:lies|rumou?rs?)|tarnish\w* (?:my )?(?:name|reputation)|bad[- ]?mouth\w*)\b",
     "defamation imputation reputation harm", "defamation", "offence"),
    (r"\b(trespass\w*|broke into|break[- ]?in|burglar\w*|entered (?:my )?(?:house|home|shop|property|land) (?:without|illegally))\b",
     "criminal trespass house-trespass house-breaking lurking entry", "trespass or break-in", "offence"),
    (r"\b(vandal\w*|damag\w*|destroy\w*|set fire|arson|smash\w*)\b",
     "mischief damage destroy property fire", "damage to property", "offence"),
    (r"\b(forger(?:y|ies)|forged|fake (?:documents?|certificates?|signatures?|ids?|aadhaar|degrees?)|counterfeit\w*)\b",
     "forgery false document making counterfeit", "forgery", "offence"),
    (r"\b(negligen\w*|by mistake|accidentally|carelessly|careless)\b",
     "rash negligent act not amounting to culpable homicide", "negligence", "offence"),
    (r"\b(acid)\b",
     "grievous hurt by use of acid throwing", "acid attack", "offence"),
    (r"\b(accident|rash(?:ly)? driv\w*|hit[- ]and[- ]run|ran over|drunk(?:en)? driv\w*)\b",
     "rash negligent act endangering life death negligence hurt", "accident or rash driving", "offence"),
    (r"\b(locked (?:me|him|her|us)|confined|held (?:me )?against (?:my|his|her) will|illegally detained|wrongful(?:ly)? (?:confine\w*|restrain\w*))\b",
     "wrongful confinement restraint", "wrongful confinement", "offence"),
    (r"\b(riot\w*|mob|unlawful assembly)\b",
     "rioting unlawful assembly common object force", "rioting or mob violence", "offence"),
    (r"\b(second (?:wife|marriage)|bigam\w*|married again|deceived into marriage)\b",
     "marrying again during lifetime of husband or wife bigamy deceit marriage", "second marriage", "offence"),
    # ---------------- process (BNSS-type) ----------------
    (r"\b(?:won'?t|refus\w*|not|didn'?t|denied|ignor\w*)\b.{0,25}\b(?:register\w*|fil(?:e|ing)|tak(?:e|ing)|accept\w*|lodg\w*)\b.{0,20}\b(?:complaint|fir|case|report)\b",
     "information cognizable offence first information report refusal Superintendent of Police senior officer",
     "police not registering a complaint", "process"),
    (r"(?<!\w)(?:fir|f\.i\.r\.?|(?:file|lodge|register)\s+(?:a\s+|an\s+)?(?:complaint|fir|case|report)|police complaint|report (?:it |this )?to (?:the )?police)(?!\w)",
     "first information report information cognizable offence police station written signed", "filing an FIR", "process"),
    (r"(?<!\w)(?:e-?fir|(?:online|electronic(?:ally)?|e-?mail)\s+(?:complaint|fir|report))(?!\w)",
     "information electronic communication signed police station", "online complaint", "process"),
    (r"\b(?:(?:different|another|other)\s+(?:city|state|district|police station|area)|jurisdiction|zero fir)\b",
     "information irrespective of area offence committed police station jurisdiction", "complaint in a different area", "process"),
    (r"\b(?:arrest\w*|detain\w*|picked up|taken (?:me |him |her )?to (?:the )?police station|police custody|judicial custody|in custody|remand)\b",
     "arrest grounds inform relative produce magistrate custody detention rights", "arrest or custody", "process"),
    (r"\b(?:bail\w*|anticipatory|released from (?:jail|custody)|get out of jail|surety)\b",
     "bail bailable non-bailable anticipatory bail bond surety release", "bail", "process"),
    (r"\b(?:summons?|warrant|notice from (?:the )?(?:police|court)|appear before (?:the )?(?:court|police|magistrate))\b",
     "summons warrant appear magistrate court service", "summons or warrant", "process"),
    (r"\b(?:police (?:searched|raided|seized)|search warrant|raid\w*|seiz\w*|confiscat\w*)\b",
     "search seizure warrant police officer witnesses list receipt", "search or seizure", "process"),
    (r"\b(?:maintenance|alimony|not (?:being )?(?:supported|maintained)|refus\w* to (?:support|maintain)|(?:support|maintain)\w* (?:my |his |her )?(?:wife|parents?|mother|father|children|child))\b",
     "maintenance wife children parents neglect refuse order magistrate monthly allowance", "maintenance", "process"),
    (r"\b(?:arrest\w*|detain\w*|custody)\b.{0,40}\b(?:reason|why|grounds?|told|inform\w*|know)\b|\b(?:reason|why|grounds?|told|inform\w*)\b.{0,40}\b(?:arrest\w*|detain\w*)\b",
     "arresting without warrant shall communicate full particulars of the offence grounds for arrest informed entitled bail", "grounds of arrest", "process"),
    (r"\b(?:lawyers?|advocates?|attorneys?|counsel)\b.{0,50}\b(?:police|question\w*|interrogat\w*|custody|arrest\w*)\b|\b(?:police|question\w*|interrogat\w*|custody|arrest\w*)\b.{0,50}\b(?:lawyers?|advocates?|attorneys?)\b",
     "arrested interrogated entitled to meet advocate of his choice during interrogation", "lawyer during questioning", "process"),
    (r"\b(?:handcuff\w*|shackl\w*|chain(?:ed|s)|tied up|roped)\b",
     "arrest no unnecessary restraint more restraint than necessary to prevent escape", "handcuffs or restraint", "process"),
    (r"\b(?:under-?trial\w*|(?:jail|prison|custody|detain\w*|locked up)\b.{0,60}\b(?:trial|waiting|without (?:being )?(?:convict\w*|sentenc\w*)|years))\b",
     "under trial prisoner detained maximum period one-half of maximum imprisonment released on bail", "long detention before trial", "process"),
    (r"\b(?:warrant)\b.{0,50}\b(?:valid\w*|how long|duration|expire\w*|in force)\b|\b(?:valid\w*|duration)\b.{0,30}\bwarrant\b",
     "warrant of arrest form duration in writing signed seal remain in force until cancelled or executed", "validity of a warrant", "process"),
    (r"^(?=.*\b(?:police|officers?|cops?|constable|inspector|SHO)\b).*?\b(?:took|taken|seized|confiscated|kept|holding|withh?eld?)\b.{0,25}\b(?:my |our )?(?:phone|mobile|laptop|car|vehicle|bike|scooter|jewel\w*|cash|money|documents?|property)\b",
     "police officer seize property suspicion report seizure custody release of property pending trial return to owner", "seized property", "process"),
    (r"\b(?:how long|how many (?:hours|days)|time limit|maximum (?:time|period))\b.{0,50}\b(?:police|lock-?up|custody|detain\w*|keep (?:me|him|her|them)|hold)\b|\block-?up\b",
     "person arrested not detained in custody longer than twenty-four hours produce before Magistrate without delay", "how long police can hold someone", "process"),
    (r"\b(?:woman|women|female|girl|wife|mother|daughter)\b.{0,60}\b(?:police station|questioning|questioned|interrogat\w*|summon\w*|call(?:ed)? (?:her |them )?to)\b",
     "police officer require attendance of witnesses woman not required to attend place other than residence", "woman called for questioning", "process"),
    (r"\b(?:plead\w* guilty|plea[- ]?bargain\w*|admit\w* (?:guilt|my guilt)|lesser (?:punishment|sentence)|reduced (?:punishment|sentence))\b",
     "plea bargaining application accused offence charge mutually satisfactory disposition lesser punishment", "plea bargaining", "process"),
    (r"\b(?:city|state|district|area|place)\s+(?:other than|different from)\b|\boutside\s+(?:my\s+|the\s+)?(?:city|state|district|area)\b|\bwhere (?:it|the (?:crime|incident|offence)) (?:happened|took place|occurred)\b",
     "information irrespective of the area where the offence is committed police station", "complaint in a different area", "process"),
    (r"\b(?:victim compensation|witness protection|compensation (?:for|to) (?:the )?victims?)\b",
     "compensation victim scheme witness protection", "victim compensation or witness protection", "process"),
    # ---------------- civil / other law: no search terms; drive the out-of-scope reply and say where to go ----------------
    # kind "civil": out of scope unless an offence word is also present; "civil!": out of scope even then
    # (e.g. "my father died and the bank won't release his savings" is about the bank, not a death)
    (r"\b(?:security deposit|deposit|rent|rental|landlord|tenant|tenancy|lease|evict\w*)\b",
     "Rent, deposits and eviction are civil matters: send the landlord or tenant a written notice first, then approach the "
     "Rent Authority or civil court in your state. If someone is locked out or threatened with force, the police can help. Free legal aid: NALSA helpline 15100.",
     "tenancy or rent", "civil"),
    (r"\b(?:divorce|child custody|custody of (?:my |our )?(?:child|children|kids?|son|daughter))\b",
     "Divorce and child custody are decided by the Family Court. Free legal aid: NALSA helpline 15100.", "divorce or child custody", "civil"),
    (r"\b(?:inherit\w*|partition of (?:the )?property|property (?:dispute|partition)|ancestral)\b|\b(?:late|deceased) (?:father|mother|parents?|grand\w+)'?s? (?:house|home|property|land|flat|plot)\b|\bbuilding on (?:part of )?(?:my|our) (?:land|plot|property)\b|\b(?:land|plot|field|farm|flat|house|property)\b.{0,60}\b(?:claim\w* (?:it|that it) (?:is|belongs)|belongs to them|claiming ownership|won'?t vacate|refus\w* to vacate|occupied|grabbed)\b",
     "Disputes over who owns or occupies land or a house go to the civil court; land records are with the Tehsildar or revenue "
     "office. If someone used force, threats or forged papers, that part is criminal. Free legal aid: NALSA helpline 15100.", "land or property dispute", "civil!"),
    (r"\b(?:salary|wages|notice period|termination|fired|laid off|gratuity|provident fund|overtime|unpaid work|not paid for (?:my )?work)\b",
     "Unpaid salary, overtime, dismissal and PF go to your state Labour Department (Labour Commissioner). Keep salary slips, "
     "attendance records and messages. Free legal aid: NALSA helpline 15100.", "employment or pay", "civil"),
    (r"\b(?:consumer|refund\w*|warranty|defective|insurance claim|overcharg\w*|extra charges?|hidden charges?|charged (?:us|me) (?:extra|for)|advance\b.{0,30}\b(?:refus|not return|won'?t return)|(?:isn'?t|not|refus\w* to be?) returning (?:the|our|my) money|cancelled\b.{0,50}\bmoney)",
     "For a refund, overcharging or poor service, you can file a consumer complaint with the District Consumer Commission "
     "(Consumer Protection Act, 2019), also online on the e-Daakhil portal. The National Consumer Helpline (1915) can often "
     "sort it out without a case.", "consumer complaint", "civil"),
    (r"\b(?:builder|developer)\b|\bpossession\b.{0,30}\b(?:flat|apartment|house)\b|\b(?:flat|apartment)\b.{0,40}\b(?:not (?:handed|given|delivered)|delay\w*|incomplete|still (?:not|under))",
     "Delays and broken promises by a builder go to your state's RERA authority (Real Estate (Regulation and Development) "
     "Act, 2016); complaints can be filed on its website. A consumer complaint is another route.", "builder or flat", "civil!"),
    (r"\b(?:electricity|power|water|gas)\s+bill\b|\b(?:electricity|power|water|gas)\s+(?:board|company|department|dept)\b.{0,40}\bbill\b|\bmeter reading\b|\bbill\b.{0,30}\b(?:increased|too high|wrong|inflated)",
     "Complain to the electricity (or water) company in writing first. If it is not fixed, go to its Consumer Grievance "
     "Redressal Forum and then the Electricity Ombudsman.", "utility bill", "civil!"),
    (r"\b(?:passed away|died|death|deceased)\b.{0,60}\b(?:bank|account|savings|fixed deposit|fd|locker|insurance)\b|\bbank\b.{0,40}\b(?:not allowing|refus\w*|won'?t|isn'?t (?:releasing|allowing|giving)|not releasing)\b.{0,30}\b(?:access|withdraw|release|close)",
     "With a nominee, the bank settles the claim on the death certificate and the nominee's ID. Without one, heirs may need a "
     "legal heir or succession certificate. If the bank delays, complain to the bank, then to the RBI Ombudsman (cms.rbi.org.in).",
     "bank or deceased's account", "civil!"),
    (r"\b(?:society|association|rwa|resident'?s? welfare|apartment association|housing society)\b.{0,50}\b(?:fees?|charges?|maintenance|dues|fine)\b",
     "Disputes with a housing society or apartment association go first to its managing committee in writing, then to the "
     "Registrar of Co-operative Societies (or of Societies) in your state, or the civil court.", "housing society", "civil!"),
    (r"\b(?:medical|hospital|treatment|discharge)\s+(?:records?|reports?|summary|papers|file)\b.{0,40}\b(?:refus|not (?:giv|releas)|won'?t)|\b(?:refus\w*|not (?:giving|releasing)|won'?t (?:give|release))\b.{0,40}\b(?:medical|hospital|treatment)\s+(?:records?|reports?|papers|file)",
     "Ask the hospital in writing for the records and keep a copy of your request. If it refuses, complain to the State "
     "Medical Council or file a consumer complaint.", "medical records", "civil!"),
    (r"\b(?:borrow\w*|lent|loan\w*)\b.{0,25}\b(?:money|rupees|rs\.?|lakhs?|thousand|cash|₹|\d{3,})\b.{0,60}\b(?:not (?:return\w*|repay\w*|paying|giving)|refus\w* to (?:return|repay|pay)|won'?t (?:return|repay|pay))\b",
     "A loan not repaid is usually a civil matter: send a written demand or legal notice, then file a recovery suit. It becomes "
     "criminal only if the person deceived you from the start. Free legal aid: NALSA helpline 15100.", "unpaid loan", "civil"),
    (r"\b(?:patent\w*|trademark\w*|copyright\w*)\b",
     "Patents, trademarks and copyright are intellectual property matters, not BNS/BNSS: patents and trademarks are filed "
     "with the Controller General of Patents, Designs and Trade Marks (ipindia.gov.in), copyright with the Copyright Office.",
     "intellectual property", "civil!"),
    (r"\bchange (?:my|his|her|our|the child'?s) (?:name|surname)\b|\bname change\b",
     "A legal name change is done by affidavit, a newspaper advertisement and publication in the Gazette (Department of "
     "Publication for central records, or your state's Gazette office).", "name change", "civil!"),
    (r"\b(?:cutting|cut|chopp\w*|fell\w*)\b.{0,25}\btrees?\b",
     "Cutting trees usually needs permission from the municipal corporation or forest department under your state's "
     "tree-protection law. Report it to them with photos and the location.", "tree cutting", "civil!"),
    (r"\b(?:old|elderly|aged|senior citizen)\s+(?:mother|father|parents?|man|woman)\b|\b(?:son|daughter)\b.{0,60}\b(?:pension|property)\b.{0,30}\b(?:mother|father|parents?)\b",
     "Senior citizens can also claim maintenance and protection from children or relatives under the Maintenance and Welfare "
     "of Parents and Senior Citizens Act, 2007, through the Maintenance Tribunal (apply to the SDM). Elder Line: 14567.",
     "senior citizen", "other"),
    (r"\b(?:construction site|factory|workplace|at work|on the job)\b.{0,60}\b(?:injur\w*|hurt|accident|died)\b|\b(?:injur\w*|hurt|accident)\b.{0,60}\b(?:construction site|factory|workplace|safety (?:equipment|gear))\b",
     "Injury at work also gives a right to compensation from the employer under the Employees' Compensation Act, 1923, "
     "through the Labour Department, in addition to any criminal case for negligence.", "workplace injury", "other"),
    # ---------------- criminal, but under another Act (not in this tool): note only ----------------
    (r"\b(?:cheque|check)\b.{0,20}\b(?:bounce\w*|dishonou?r\w*|returned)\b|\bbounced (?:cheque|check)\b",
     "A bounced cheque is handled under section 138 of the Negotiable Instruments Act, 1881, which is not part of this tool. "
     "A legal notice usually has to be sent within 30 days of the bank telling you the cheque bounced.", "cheque bounce", "other"),
    (r"^(?!.*\b(?:spreading|posting|saying|claiming|accus\w*|rumou?r\w*|telling (?:people|everyone))\b)(?:.*\b(?:bribe\w*|corrupt\w*|kickback|asking for money to (?:register|file|do))\b)|^(?=.*\b(?:police|cops?|constable|officer|inspector|SHO|traffic)\b).*\b(?:ask\w*|demand\w*|want\w*)\b.{0,15}\b(?:money|cash|rs\.?|rupees|₹\s?\d+|\d{3,})\b.{0,40}\b(?:to|for|or)\b",
     "Asking for or taking a bribe is an offence under the Prevention of Corruption Act, 1988, which is not part of this tool. "
     "You can complain to your state's Anti-Corruption Bureau or Vigilance department.", "bribery", "other"),
    (r"\b(?:drugs?|narcotic\w*|ganja|charas|heroin|smack|cocaine|mdma|brown sugar)\b.{0,30}\b(?:sell\w*|deal\w*|suppl\w*|peddl\w*)\b|\b(?:sell\w*|deal\w*|suppl\w*|peddl\w*)\b.{0,30}\b(?:drugs?|narcotic\w*|ganja|charas|heroin|smack|cocaine)\b",
     "Drug offences fall under the NDPS Act, 1985, which is not part of this tool. You can report drug selling to the police "
     "(112 or your local station) without giving your name.", "drugs", "other"),
    (r"\b(?:caste|casteist|jati|dalit|scheduled caste|scheduled tribe|sc/st)\b",
     "Insults, threats or violence against a member of a Scheduled Caste or Scheduled Tribe because of caste are covered by the "
     "SC/ST (Prevention of Atrocities) Act, 1989, which is not part of this tool. Report it to the police; the general "
     "offences (hurt, intimidation, insult) in BNS can apply too.", "caste-based abuse", "other"),
    (r"\bstray\b.{0,20}\b(?:dog|dogs|cattle|cow|bull|monkey|animal)s?\b.{0,30}\b(?:bit|bitten|bites|attack\w*|chased)\b|\b(?:bit|bitten|attacked)\b.{0,20}\bby (?:a )?stray\b",
     "For any animal bite, wash the wound with soap and running water for 15 minutes and get the anti-rabies vaccine "
     "at a hospital the same day. Stray animals are the responsibility of your municipal corporation or panchayat: "
     "complain to them. Criminal law applies only if an owner was negligent (BNS 291).", "stray animal bite", "other"),
    (r"\bchild labou?r\b|\b(?:\d{1,2}|ten|eleven|twelve|thirteen|fourteen)[- ]?(?:years?|yrs?)[- ]?old\b.{0,40}\b(?:work\w*|employ\w*|labou?r)\b|\b(?:child|kid|boy|girl|nephew|niece|son|daughter)\b.{0,30}\b(?:made to work|forced to work|working in)\b",
     "Child labour is covered by the Child and Adolescent Labour (Prohibition and Regulation) Act, 1986, which is not part of this tool. "
     "Call Childline 1098 or 112 to report it.", "child labour", "other"),
]

# Common names for legal ideas that the statute text itself never uses -> the words it does use.
# Applied to definition questions ("what is anticipatory bail") before searching titles and text.
TERMS = [
    (r"\banticipatory bail\b|\badvance bail\b|\bpre-?arrest bail\b", "bail person apprehending arrest direction"),
    (r"\bzero fir\b", "information cognizable offence irrespective of the area where the offence is committed"),
    (r"\be-?fir\b|\bonline fir\b", "information cognizable offence electronic communication"),
    (r"\b(?:mob )?lynch\w*", "group of five or more persons acting in concert commits murder race caste community"),
    (r"\bhit[- ]and[- ]run\b", "causing death rash negligent driving escapes without reporting"),
    (r"\bacid attack\b", "grievous hurt by use of acid"),
    (r"\b(?:murder|kill\w*|death|died|dies)\b.{0,60}\b(?:by accident|accidental\w*|unintentional\w*|by mistake|not on purpose|negligen\w*)\b|"
     r"\b(?:by accident|accidental\w*|unintentional\w*|by mistake|not on purpose)\b.{0,60}\b(?:murder|kill\w*|death|died)\b",
     "causing death by negligence rash or negligent act not amounting to culpable homicide"),
    (r"\bchain snatch\w*|\bsnatching\b", "snatching theft suddenly quickly forcibly seizes"),
    (r"\beve[- ]?teas\w*", "word gesture act intended to insult modesty of a woman"),
    (r"\bmolest\w*", "assault criminal force woman intent to outrage modesty"),
    (r"\bmarital rape\b", "rape husband wife exception sexual intercourse"),
    (r"\bfir\b", "information cognizable offence"),
    (r"\bcustodial (?:death|torture|violence)\b", "voluntarily causing hurt extort confession"),
    (r"\bthird degree\b", "voluntarily causing hurt extort confession"),
    (r"\bsuo moto\b|\bsuo motu\b", "cognizance"),
    (r"\bnon[- ]bailable\b", "bail non-bailable offence when bail may be taken"),
    (r"\bremand\b", "detention custody magistrate fifteen days"),
    (r"\bchargesheet\b|\bcharge[- ]sheet\b", "report of police officer on completion of investigation"),
]
_TERMS = [(re.compile(p, re.I), add) for p, add in TERMS]


def legal_terms(query: str) -> str:
    """The query plus the statute's own words for any common legal names in it."""
    extra = [add for pat, add in _TERMS if pat.search(query)]
    return f"{query} {' '.join(extra)}".strip() if extra else query


_COMPILED = [(re.compile(p, re.I), add, label, kind) for p, add, label, kind in RULES]


@dataclass
class Analysis:
    expanded: str                       # original query + legal vocabulary
    labels: list[str] = field(default_factory=list)       # what we understood (offence / process)
    civil_labels: list[str] = field(default_factory=list)
    civil_notes: list[str] = field(default_factory=list)  # where to go for the civil part
    notes: list[str] = field(default_factory=list)        # pointers to other Acts (criminal, but not in BNS/BNSS)
    note_labels: list[str] = field(default_factory=list)
    out_of_scope: bool = False          # civil or other-Act wording and nothing BNS/BNSS-related


def analyse(query: str) -> Analysis:
    terms: list[str] = []
    labels: list[str] = []
    civil: list[str] = []
    civil_notes: list[str] = []
    strong = False
    notes: list[str] = []
    note_labels: list[str] = []
    for pat, add, label, kind in _COMPILED:
        if not pat.search(query):
            continue
        if kind in ("civil", "civil!"):
            civil.append(label)
            if add:
                civil_notes.append(add)
            strong = strong or kind == "civil!"
        elif kind == "other":
            notes.append(add)
            note_labels.append(label)
        else:
            terms.append(add)
            labels.append(label)
    extra = " ".join(dict.fromkeys(" ".join(terms).split()))
    expanded = f"{query} {extra}".strip() if extra else query
    return Analysis(expanded, labels, civil, civil_notes, notes, note_labels,
                    out_of_scope=bool(civil or notes) and (not labels or strong))
