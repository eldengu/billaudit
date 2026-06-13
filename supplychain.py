"""
supplychain.py — AutoSwarm, second domain: shell-network detection.

The same engine as billaudit.py, pointed at a different synthetic problem:
spotting a hidden illicit network inside a company registry. Detectors see the
WHOLE registry and return the SET of entity ids they judge to be bad actors;
an F1 scorer grades against a held answer key; a union ensemble pools the lenses.

Synthetic data ONLY. No real company, person, address, or phone number. The
anomaly *typologies* are modeled on patterns documented publicly by bodies like
FinCEN and C4ADS -- shared registration address/phone, shared officers,
industry-facade mismatch, and clustering in an obscure jurisdiction -- but every
record here is invented.

Before running: put your real key in .env  ->  OPENAI_API_KEY=sk-...
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()  # reads .env in the working directory


# ---------------------------------------------------------------------------
# 1. Synthetic company registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Entity:
    id: int
    name: str
    declared_industry: str
    address: str
    phone: str
    officers: tuple[str, ...]
    product_keywords: tuple[str, ...]


# 40 companies. Most are ordinary, well-separated businesses (distinct cities,
# unique phones/officers, keywords that match their declared industry).
#
# Planted inside is an 8-entity illicit network (answer key below) tied together
# by VERIFIABLE shared signals, mixing obvious with subtle:
#
#   shared address  "Unit 7, 14 Harbour Mews, Port Kelvin"  -> ids 4, 9, 13, 31
#   shared phone    "+1-555-0142"                           -> ids 13, 18, 31
#   shared officer  "Dorian Vex"                            -> ids 4, 9, 18, 31
#   shared officer  "Marla Quint"                           -> ids 13, 22, 36
#   facade mismatch (apparel/textiles/electronics, but the
#                    product keywords read like solvents /
#                    drug precursors)                        -> ids 4, 9, 18, 27, 36
#   clustering in the obscure city "Port Kelvin"            -> ids 4, 9, 13, 22, 31, 36
#
#   - id 31 is the obvious one: shared address AND phone AND officer.
#   - id 22 is subtle: only Port Kelvin clustering + shared officer, generic goods.
#   - id 27 is the hardest: a big-city company with a unique phone/officer whose
#     ONLY tell is a textile facade hiding chemical-precursor keywords.
#
# Decoy noise: ids 7 and 25 are LEGIT firms that share a co-working address
# ("Suite 200, 1 Civic Plaza, Eastport") -- a benign shared-address false positive
# so precision actually has to be earned.
REGISTRY: list[Entity] = [
    Entity(1,  "Brightleaf Software Labs", "Software",    "220 Cedar Ave, Riverton",                  "+1-555-0101", ("Anita Roe",),    ("saas", "analytics", "dashboards")),
    Entity(2,  "Golden Crust Bakery",      "Bakery",      "14 Mill St, Hartwell",                     "+1-555-0102", ("Ben Saito",),    ("bread", "pastries", "sourdough")),
    Entity(3,  "Summit Dental Care",       "Dental",      "88 Park Rd, Lakeview",                     "+1-555-0103", ("Lena Ford",),    ("dentistry", "hygiene", "implants")),
    Entity(4,  "Meridian Textile Holdings","Textiles",    "Unit 7, 14 Harbour Mews, Port Kelvin",     "+1-555-0181", ("Dorian Vex",),   ("acetone", "toluene", "drum storage")),                  # BAD: facade + addr cluster + officer Vex (obvious)
    Entity(5,  "Hartwell Landscaping",     "Landscaping", "5 Garden Way, Hartwell",                   "+1-555-0105", ("Carlos Mund",),  ("lawn", "irrigation", "hedges")),
    Entity(6,  "Ledger & Vine Accounting", "Accounting",  "410 Finance Blvd, Metro City",             "+1-555-0106", ("Priya Nadar",),  ("tax", "audit", "bookkeeping")),
    Entity(7,  "Northwind Consulting",     "Consulting",  "Suite 200, 1 Civic Plaza, Eastport",       "+1-555-0107", ("Tom Reyes",),    ("strategy", "advisory")),                                # decoy: legit, shares co-working addr with #25
    Entity(8,  "Copper Kettle Brewery",    "Brewery",     "33 Barrel Ln, Riverton",                   "+1-555-0108", ("Greta Olsson",), ("beer", "ale", "brewing")),
    Entity(9,  "Kelvin Weave Trading",     "Textiles",    "Unit 7, 14 Harbour Mews, Port Kelvin",     "+1-555-0182", ("Dorian Vex",),   ("industrial solvents", "reagents", "fabric")),           # BAD: facade + addr cluster + officer Vex (obvious)
    Entity(10, "Open Page Bookstore",      "Retail",      "9 Read St, Lakeview",                      "+1-555-0110", ("Sam Holt",),     ("books", "stationery")),
    Entity(11, "Reliable Plumbing",        "Plumbing",    "77 Pipe Rd, Hartwell",                     "+1-555-0111", ("Dee Marsh",),    ("plumbing", "drains", "fixtures")),
    Entity(12, "Tiny Steps Pediatrics",    "Healthcare",  "120 Wellness Dr, Metro City",              "+1-555-0112", ("Omar Vance",),   ("pediatrics", "vaccines")),
    Entity(13, "Harbour Mews Imports",     "Logistics",   "Unit 7, 14 Harbour Mews, Port Kelvin",     "+1-555-0142", ("Marla Quint",),  ("freight", "containers", "transshipment")),              # BAD: addr cluster + phone 0142 + officer Quint
    Entity(14, "Apex Auto Repair",         "Automotive",  "250 Gear St, Riverton",                    "+1-555-0114", ("Rico Tan",),     ("auto repair", "brakes", "engine")),
    Entity(15, "SunPath Solar",            "Energy",      "60 Bright Rd, Lakeview",                   "+1-555-0115", ("Hana Kim",),     ("solar", "panels", "inverters")),
    Entity(16, "Roast Republic Coffee",    "Food",        "18 Bean Aly, Eastport",                    "+1-555-0116", ("Lou Park",),     ("coffee", "roasting", "espresso")),
    Entity(17, "Companion Vet Clinic",     "Veterinary",  "5 Paw Pl, Hartwell",                       "+1-555-0117", ("Iris Bell",),    ("veterinary", "pets")),
    Entity(18, "Anchor Fabric Supply",     "Apparel",     "3 Dockyard Rd, Brightbay",                 "+1-555-0142", ("Dorian Vex",),   ("precursor chemicals", "glassware", "apparel")),         # BAD: phone 0142 + officer Vex + facade (addr NOT clustered -> subtle)
    Entity(19, "Bloom & Stem Florist",     "Retail",      "22 Petal St, Metro City",                  "+1-555-0119", ("Nina Cole",),    ("flowers", "bouquets")),
    Entity(20, "Ironclad Hardware",        "Retail",      "140 Bolt Ave, Riverton",                   "+1-555-0120", ("Walt Greer",),   ("hardware", "tools")),
    Entity(21, "PulseFit Gym",             "Fitness",     "88 Rep Rd, Lakeview",                      "+1-555-0121", ("Tara Lin",),     ("gym", "fitness", "training")),
    Entity(22, "Saltmarsh Trading Co",     "Import/Export","41 Old Wharf, Port Kelvin",               "+1-555-0184", ("Marla Quint",),  ("general goods", "wholesale", "brokerage")),             # BAD: Port Kelvin cluster + officer Quint, generic goods -> subtle
    Entity(23, "Clearview Optometry",      "Healthcare",  "300 Vision Blvd, Metro City",              "+1-555-0123", ("Eli Frost",),    ("optometry", "glasses")),
    Entity(24, "Maple Catering Co",        "Food",        "7 Feast Ln, Hartwell",                     "+1-555-0124", ("Gabi Ruiz",),    ("catering", "events")),
    Entity(25, "Eastport Advisory Partners","Consulting", "Suite 200, 1 Civic Plaza, Eastport",       "+1-555-0125", ("Joan Pike",),    ("consulting", "advisory")),                              # decoy: legit, shares co-working addr with #7
    Entity(26, "Little Sprouts Daycare",   "Childcare",   "14 Cradle Ct, Riverton",                   "+1-555-0126", ("Mary Dunn",),    ("daycare", "childcare")),
    Entity(27, "Verdant Apparel Group",    "Textiles",    "500 Garment Row, Metro City",              "+1-555-0127", ("Stefan Auer",),  ("nitromethane", "acetic anhydride", "glass reactors")),  # BAD: facade ONLY, big city, unique phone/officer -> hardest
    Entity(28, "Topline Roofing",          "Construction","90 Shingle St, Hartwell",                  "+1-555-0128", ("Vic Stroud",),   ("roofing", "gutters")),
    Entity(29, "Cloudpeak Hosting",        "Software",    "700 Server Rd, Metro City",                "+1-555-0129", ("Ada Wynn",),     ("hosting", "cloud", "servers")),
    Entity(30, "Fresh Fork Diner",         "Food",        "11 Plate St, Lakeview",                    "+1-555-0130", ("Joe Banks",),    ("diner", "breakfast")),
    Entity(31, "Quay Side Logistics",      "Logistics",   "Unit 7, 14 Harbour Mews, Port Kelvin",     "+1-555-0142", ("Dorian Vex",),   ("freight forwarding", "containers")),                    # BAD: addr cluster + phone 0142 + officer Vex (obvious triple)
    Entity(32, "Stonebridge Law",          "Legal",       "410 Justice Ave, Metro City",              "+1-555-0132", ("Ruth Calder",),  ("legal", "litigation")),
    Entity(33, "GreenThumb Nursery",       "Retail",      "6 Sprout Rd, Hartwell",                    "+1-555-0133", ("Pete Salk",),    ("plants", "garden")),
    Entity(34, "Brightbay Marine",         "Marine",      "2 Harbor View, Brightbay",                 "+1-555-0134", ("Cory Lund",),    ("boats", "marine", "repair")),
    Entity(35, "Pixel Forge Studio",       "Software",    "15 Render St, Eastport",                   "+1-555-0135", ("Mona Ek",),      ("games", "design")),
    Entity(36, "Lowtide Components",       "Electronics", "9 Tidewater Rd, Port Kelvin",              "+1-555-0186", ("Marla Quint",),  ("solvent extraction reagents", "lab glassware")),        # BAD: Port Kelvin cluster + officer Quint + facade -> medium
    Entity(37, "Hearthstone Realty",       "Real Estate", "120 Home Ave, Lakeview",                   "+1-555-0137", ("Dan Voss",),     ("realty", "homes")),
    Entity(38, "Crisp Linen Laundry",      "Services",    "44 Wash St, Riverton",                     "+1-555-0138", ("Bea Knott",),    ("laundry", "linen")),
    Entity(39, "Trailhead Outfitters",     "Retail",      "88 Summit Rd, Eastport",                   "+1-555-0139", ("Kip Doran",),    ("outdoor", "gear", "camping")),
    Entity(40, "Quill & Press Printing",   "Services",    "17 Inkwell Ln, Metro City",                "+1-555-0140", ("Sol Mercer",),   ("printing", "signage")),
]

# Ground truth: the set of ids that belong to the illicit network (8 total).
ANSWER_KEY: set[int] = {4, 9, 13, 18, 22, 27, 31, 36}


# ---------------------------------------------------------------------------
# 2. F1 scorer  (same contract as billaudit.py)
# ---------------------------------------------------------------------------

@dataclass
class Score:
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int


def score(predicted_ids: set[int], truth_ids: set[int]) -> Score:
    """Precision / recall / F1 of predicted bad-actor ids vs the answer key.

    F1 keeps it honest: flagging every company tanks precision, so "flag all"
    can never win.
    """
    tp = len(predicted_ids & truth_ids)
    fp = len(predicted_ids - truth_ids)
    fn = len(truth_ids - predicted_ids)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return Score(precision, recall, f1, tp, fp, fn)


# ---------------------------------------------------------------------------
# 3. The SWARM of detectors (each sees the WHOLE registry)
# ---------------------------------------------------------------------------
#
# Signature contract:  detector(entities: list[Entity]) -> set[int]
#
# Each lens is gpt-4o-mini under a different system prompt. Returning a set of
# ids over the whole registry is what lets cross-entity signals (shared address,
# shared officer, clustering) be found at all -- they're invisible per-record.


def _format_registry(entities: list[Entity]) -> str:
    rows = []
    for e in entities:
        rows.append(
            f"  id {e.id}: \"{e.name}\" | industry: {e.declared_industry} "
            f"| address: {e.address} | phone: {e.phone} "
            f"| officers: {', '.join(e.officers)} "
            f"| products: {', '.join(e.product_keywords)}"
        )
    return "\n".join(rows)


# Six lenses, each biased toward one documented anomaly typology.
SYSTEM_PROMPTS: dict[str, str] = {
    "infra_overlap": (
        "You investigate a company registry for SHELL-COMPANY networks, specializing "
        "in INFRASTRUCTURE OVERLAP: distinct companies that share the same registered "
        "address or the same phone number. Compare records against each other across "
        "the whole registry and flag entities that share such identifiers."
    ),
    "shared_officers": (
        "You investigate a company registry for SHELL-COMPANY networks, specializing "
        "in SHARED OFFICERS / control: the same person listed as an officer of "
        "multiple otherwise-unrelated companies. Flag entities linked by a common officer."
    ),
    "facade_mismatch": (
        "You investigate a company registry for SHELL-COMPANY networks, specializing "
        "in INDUSTRY-FACADE MISMATCH: a company whose declared industry does not match "
        "its product keywords -- e.g. a 'textiles' or 'apparel' firm whose products read "
        "like industrial solvents or chemical precursors. Flag such mismatches."
    ),
    "reg_clustering": (
        "You investigate a company registry for SHELL-COMPANY networks, specializing "
        "in REGISTRATION CLUSTERING: an unusual concentration of companies registered "
        "in the same small or obscure city/jurisdiction, against a registry that is "
        "otherwise geographically spread out. Flag entities in such a cluster."
    ),
    "network": (
        "You investigate a company registry for SHELL-COMPANY networks. Think like a "
        "link analyst: trace any shared identifier (address, phone, officer) that "
        "connects companies into a CLUSTER, then flag the whole connected group."
    ),
    "general": (
        "You investigate a company registry for SHELL-COMPANY networks. Flag any entity "
        "that shows signs of being part of an illicit network: shared address or phone, "
        "shared officers, an industry facade that hides its real products, or clustering "
        "in an obscure jurisdiction."
    ),
}

_USER_TEMPLATE = """Here is a company registry. Review ALL entities together, comparing them \
against one another to find a hidden illicit / shell-company network.

{registry}

List the id numbers of every entity that is LIKELY part of the illicit network.
Respond with ONLY a comma-separated list of ids (e.g. "4, 9, 31").
If none look illicit, respond with the single word NONE."""


def make_openai_detector(system_prompt: str):
    """Build a detector(entities) -> set[int] backed by gpt-4o-mini under `system_prompt`."""

    def detector(entities: list[Entity]) -> set[int]:
        from openai import OpenAI

        client = OpenAI()  # reads OPENAI_API_KEY from the environment (.env)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            max_tokens=160,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": _USER_TEMPLATE.format(registry=_format_registry(entities))},
            ],
        )
        text = (resp.choices[0].message.content or "").strip()
        valid = {e.id for e in entities}
        return {int(tok) for tok in re.findall(r"\d+", text)} & valid

    return detector


SWARM: list[tuple[str, object]] = [
    (name, make_openai_detector(prompt)) for name, prompt in SYSTEM_PROMPTS.items()
]


# ---------------------------------------------------------------------------
# 4. Harness
# ---------------------------------------------------------------------------

def run_swarm(swarm, entities: list[Entity]) -> dict[str, set[int]]:
    """Run every detector over the whole registry; return {lens_name: flagged ids}."""
    return {name: det(entities) for name, det in swarm}


def union_ensemble(flags_by_detector: dict[str, set[int]]) -> set[int]:
    """Flag an entity if ANY single lens flags it. Costs no extra API calls."""
    flagged: set[int] = set()
    for ids in flags_by_detector.values():
        flagged |= ids
    return flagged


def _check_key() -> bool:
    key = os.getenv("OPENAI_API_KEY", "")
    if not key or key == "replace-me":
        print("=" * 64)
        print("  OPENAI_API_KEY is not set.")
        print("  Put your real key in the .env file before running:")
        print()
        print("      OPENAI_API_KEY=sk-...")
        print()
        print("  Then re-run:  python supplychain.py")
        print("=" * 64)
        return False
    return True


def main() -> None:
    print("AutoSwarm - shell-network detection (supply-chain domain)\n")
    print(f"Registry: {len(REGISTRY)} companies | planted network ({len(ANSWER_KEY)}): "
          f"{sorted(ANSWER_KEY)}")
    print(f"Lenses: {[name for name, _ in SWARM]}\n")

    if not _check_key():
        return

    print(f"Running {len(SWARM)} lenses (gpt-4o-mini); each sees the WHOLE registry...\n")
    flags = run_swarm(SWARM, REGISTRY)

    print("RESULTS (whole registry vs answer key)")
    print("=" * 50)
    print(f"{'LENS':<18}{'PREC':>8}{'RECALL':>8}{'F1':>8}")
    print("-" * 50)
    for name, _ in SWARM:
        s = score(flags[name], ANSWER_KEY)
        print(f"{name:<18}{s.precision:>8.2f}{s.recall:>8.2f}{s.f1:>8.2f}")
    print("-" * 50)
    uni = score(union_ensemble(flags), ANSWER_KEY)
    print(f"{'ENSEMBLE (union)':<18}{uni.precision:>8.2f}{uni.recall:>8.2f}{uni.f1:>8.2f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
