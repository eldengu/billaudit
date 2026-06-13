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
    ships_to: str  # downstream buyer / consignee


# 48 companies across two waves. Most are ordinary, well-separated businesses
# (distinct cities, unique phones/officers, keywords matching their industry, and
# varied downstream buyers -- with a couple of benign shared buyers as noise).
#
# COHORT 1 (ids 4,9,13,18,22,27,31,36) -- the original network, tied together by
# VERIFIABLE shared signals the lenses already catch, mixing obvious with subtle:
#   shared address  "Unit 7, 14 Harbour Mews, Port Kelvin"  -> 4, 9, 13, 31
#   shared phone    "+1-555-0142"                           -> 13, 18, 31
#   shared officer  "Dorian Vex"                            -> 4, 9, 18, 31
#   shared officer  "Marla Quint"                           -> 13, 22, 36
#   facade mismatch (textiles/apparel/electronics hiding solvent/precursor
#                    keywords)                               -> 4, 9, 18, 27, 36
#   clustering in obscure city "Port Kelvin"                -> 4, 9, 13, 22, 31, 36
#
# COHORT 2 (ids 41-48) -- the ADVERSARY ADAPTATION. After the first network was
# burned, these EVADE every cohort-1 signal: a UNIQUE address and phone each (no
# infrastructure overlap), UNIQUE officers (no shared control), and a switched
# facade to "home appliances / consumer electronics" with keywords that genuinely
# MATCH that facade (no mismatch to catch). To the current 6 lenses they look
# clean.
#   BUT a deeper invariant survives the disguise: all 8 consign to the SAME
#   downstream buyer, "Granite Bay Distribution Ltd". That signal is present in
#   the data (ships_to) and catchable -- but NO current lens is tuned to it.
#
# Decoy noise: ids 7 and 25 are LEGIT firms sharing a co-working address
# ("Suite 200, 1 Civic Plaza, Eastport"); ids 15, 20, 39 are LEGIT firms sharing
# the benign buyer "National Retail Group" -- so neither a shared address nor a
# shared buyer is, alone, proof of anything. Precision has to be earned.
REGISTRY: list[Entity] = [
    # ---- COHORT 1 network + legitimate wave (ids 1-40) ----
    Entity(1,  "Brightleaf Software Labs", "Software",    "220 Cedar Ave, Riverton",                  "+1-555-0101", ("Anita Roe",),    ("saas", "analytics", "dashboards"),                     "direct SaaS clients"),
    Entity(2,  "Golden Crust Bakery",      "Bakery",      "14 Mill St, Hartwell",                     "+1-555-0102", ("Ben Saito",),    ("bread", "pastries", "sourdough"),                      "Hartwell Grocers Co-op"),
    Entity(3,  "Summit Dental Care",       "Dental",      "88 Park Rd, Lakeview",                     "+1-555-0103", ("Lena Ford",),    ("dentistry", "hygiene", "implants"),                    "patients (direct)"),
    Entity(4,  "Meridian Textile Holdings","Textiles",    "Unit 7, 14 Harbour Mews, Port Kelvin",     "+1-555-0181", ("Dorian Vex",),   ("acetone", "toluene", "drum storage"),                  "Anchor Export Co"),       # BAD c1: facade + addr cluster + officer Vex (obvious)
    Entity(5,  "Hartwell Landscaping",     "Landscaping", "5 Garden Way, Hartwell",                   "+1-555-0105", ("Carlos Mund",),  ("lawn", "irrigation", "hedges"),                        "residential (direct)"),
    Entity(6,  "Ledger & Vine Accounting", "Accounting",  "410 Finance Blvd, Metro City",             "+1-555-0106", ("Priya Nadar",),  ("tax", "audit", "bookkeeping"),                         "clients (direct)"),
    Entity(7,  "Northwind Consulting",     "Consulting",  "Suite 200, 1 Civic Plaza, Eastport",       "+1-555-0107", ("Tom Reyes",),    ("strategy", "advisory"),                                "enterprise clients"),     # decoy: legit, shares co-working addr with #25
    Entity(8,  "Copper Kettle Brewery",    "Brewery",     "33 Barrel Ln, Riverton",                   "+1-555-0108", ("Greta Olsson",), ("beer", "ale", "brewing"),                              "Riverton Beverage Dist."),
    Entity(9,  "Kelvin Weave Trading",     "Textiles",    "Unit 7, 14 Harbour Mews, Port Kelvin",     "+1-555-0182", ("Dorian Vex",),   ("industrial solvents", "reagents", "fabric"),           "Meridian Wholesale"),     # BAD c1: facade + addr cluster + officer Vex (obvious)
    Entity(10, "Open Page Bookstore",      "Retail",      "9 Read St, Lakeview",                      "+1-555-0110", ("Sam Holt",),     ("books", "stationery"),                                 "walk-in retail"),
    Entity(11, "Reliable Plumbing",        "Plumbing",    "77 Pipe Rd, Hartwell",                     "+1-555-0111", ("Dee Marsh",),    ("plumbing", "drains", "fixtures"),                      "residential (direct)"),
    Entity(12, "Tiny Steps Pediatrics",    "Healthcare",  "120 Wellness Dr, Metro City",              "+1-555-0112", ("Omar Vance",),   ("pediatrics", "vaccines"),                              "patients (direct)"),
    Entity(13, "Harbour Mews Imports",     "Logistics",   "Unit 7, 14 Harbour Mews, Port Kelvin",     "+1-555-0142", ("Marla Quint",),  ("freight", "containers", "transshipment"),              "Pan-Ocean Freight"),      # BAD c1: addr cluster + phone 0142 + officer Quint
    Entity(14, "Apex Auto Repair",         "Automotive",  "250 Gear St, Riverton",                    "+1-555-0114", ("Rico Tan",),     ("auto repair", "brakes", "engine"),                     "vehicle owners (direct)"),
    Entity(15, "SunPath Solar",            "Energy",      "60 Bright Rd, Lakeview",                   "+1-555-0115", ("Hana Kim",),     ("solar", "panels", "inverters"),                        "National Retail Group"),  # decoy: benign shared buyer
    Entity(16, "Roast Republic Coffee",    "Food",        "18 Bean Aly, Eastport",                    "+1-555-0116", ("Lou Park",),     ("coffee", "roasting", "espresso"),                      "cafe wholesale"),
    Entity(17, "Companion Vet Clinic",     "Veterinary",  "5 Paw Pl, Hartwell",                       "+1-555-0117", ("Iris Bell",),    ("veterinary", "pets"),                                  "pet owners (direct)"),
    Entity(18, "Anchor Fabric Supply",     "Apparel",     "3 Dockyard Rd, Brightbay",                 "+1-555-0142", ("Dorian Vex",),   ("precursor chemicals", "glassware", "apparel"),         "Coastal Distributors"),   # BAD c1: phone 0142 + officer Vex + facade (addr NOT clustered -> subtle)
    Entity(19, "Bloom & Stem Florist",     "Retail",      "22 Petal St, Metro City",                  "+1-555-0119", ("Nina Cole",),    ("flowers", "bouquets"),                                 "walk-in retail"),
    Entity(20, "Ironclad Hardware",        "Retail",      "140 Bolt Ave, Riverton",                   "+1-555-0120", ("Walt Greer",),   ("hardware", "tools"),                                   "National Retail Group"),  # decoy: benign shared buyer
    Entity(21, "PulseFit Gym",             "Fitness",     "88 Rep Rd, Lakeview",                      "+1-555-0121", ("Tara Lin",),     ("gym", "fitness", "training"),                          "members (direct)"),
    Entity(22, "Saltmarsh Trading Co",     "Import/Export","41 Old Wharf, Port Kelvin",               "+1-555-0184", ("Marla Quint",),  ("general goods", "wholesale", "brokerage"),             "Saltmarsh Wholesale"),    # BAD c1: Port Kelvin cluster + officer Quint, generic goods -> subtle
    Entity(23, "Clearview Optometry",      "Healthcare",  "300 Vision Blvd, Metro City",              "+1-555-0123", ("Eli Frost",),    ("optometry", "glasses"),                                "patients (direct)"),
    Entity(24, "Maple Catering Co",        "Food",        "7 Feast Ln, Hartwell",                     "+1-555-0124", ("Gabi Ruiz",),    ("catering", "events"),                                  "event clients"),
    Entity(25, "Eastport Advisory Partners","Consulting", "Suite 200, 1 Civic Plaza, Eastport",       "+1-555-0125", ("Joan Pike",),    ("consulting", "advisory"),                              "enterprise clients"),     # decoy: legit, shares co-working addr with #7
    Entity(26, "Little Sprouts Daycare",   "Childcare",   "14 Cradle Ct, Riverton",                   "+1-555-0126", ("Mary Dunn",),    ("daycare", "childcare"),                                "families (direct)"),
    Entity(27, "Verdant Apparel Group",    "Textiles",    "500 Garment Row, Metro City",              "+1-555-0127", ("Stefan Auer",),  ("nitromethane", "acetic anhydride", "glass reactors"),  "Garment Exporters Ltd"),  # BAD c1: facade ONLY, big city, unique phone/officer -> hardest
    Entity(28, "Topline Roofing",          "Construction","90 Shingle St, Hartwell",                  "+1-555-0128", ("Vic Stroud",),   ("roofing", "gutters"),                                  "homeowners (direct)"),
    Entity(29, "Cloudpeak Hosting",        "Software",    "700 Server Rd, Metro City",                "+1-555-0129", ("Ada Wynn",),     ("hosting", "cloud", "servers"),                         "online customers"),
    Entity(30, "Fresh Fork Diner",         "Food",        "11 Plate St, Lakeview",                    "+1-555-0130", ("Joe Banks",),    ("diner", "breakfast"),                                  "walk-in retail"),
    Entity(31, "Quay Side Logistics",      "Logistics",   "Unit 7, 14 Harbour Mews, Port Kelvin",     "+1-555-0142", ("Dorian Vex",),   ("freight forwarding", "containers"),                    "Pan-Ocean Freight"),      # BAD c1: addr cluster + phone 0142 + officer Vex (obvious triple)
    Entity(32, "Stonebridge Law",          "Legal",       "410 Justice Ave, Metro City",              "+1-555-0132", ("Ruth Calder",),  ("legal", "litigation"),                                 "clients (direct)"),
    Entity(33, "GreenThumb Nursery",       "Retail",      "6 Sprout Rd, Hartwell",                    "+1-555-0133", ("Pete Salk",),    ("plants", "garden"),                                    "garden centers"),
    Entity(34, "Brightbay Marine",         "Marine",      "2 Harbor View, Brightbay",                 "+1-555-0134", ("Cory Lund",),    ("boats", "marine", "repair"),                           "boat owners (direct)"),
    Entity(35, "Pixel Forge Studio",       "Software",    "15 Render St, Eastport",                   "+1-555-0135", ("Mona Ek",),      ("games", "design"),                                     "publishers"),
    Entity(36, "Lowtide Components",       "Electronics", "9 Tidewater Rd, Port Kelvin",              "+1-555-0186", ("Marla Quint",),  ("solvent extraction reagents", "lab glassware"),        "Tidewater Supply"),       # BAD c1: Port Kelvin cluster + officer Quint + facade -> medium
    Entity(37, "Hearthstone Realty",       "Real Estate", "120 Home Ave, Lakeview",                   "+1-555-0137", ("Dan Voss",),     ("realty", "homes"),                                     "home buyers (direct)"),
    Entity(38, "Crisp Linen Laundry",      "Services",    "44 Wash St, Riverton",                     "+1-555-0138", ("Bea Knott",),    ("laundry", "linen"),                                    "hotels (local)"),
    Entity(39, "Trailhead Outfitters",     "Retail",      "88 Summit Rd, Eastport",                   "+1-555-0139", ("Kip Doran",),    ("outdoor", "gear", "camping"),                          "National Retail Group"),  # decoy: benign shared buyer
    Entity(40, "Quill & Press Printing",   "Services",    "17 Inkwell Ln, Metro City",                "+1-555-0140", ("Sol Mercer",),   ("printing", "signage"),                                 "local businesses"),

    # ---- COHORT 2: adversary adaptation (ids 41-48) ----
    # Unique address + phone + officer each; appliance/electronics facade WITH
    # matching keywords (no mismatch). Only invariant: all consign to the same
    # downstream buyer, "Granite Bay Distribution Ltd".
    Entity(41, "Northgate Appliance Imports","Home Appliances",     "12 Market St, Riverton",         "+1-555-0241", ("Glen Awe",),     ("refrigerators", "dishwashers", "ranges"),              "Granite Bay Distribution Ltd"),  # BAD c2
    Entity(42, "Crest Home Electronics",   "Consumer Electronics",  "60 Vale Rd, Lakeview",           "+1-555-0242", ("Pia Roth",),     ("televisions", "soundbars", "remotes"),                 "Granite Bay Distribution Ltd"),  # BAD c2
    Entity(43, "Bluepeak Domestic Goods",  "Home Appliances",       "7 Hill Ave, Hartwell",           "+1-555-0243", ("Sandro Beck",),  ("washing machines", "dryers", "spare parts"),           "Granite Bay Distribution Ltd"),  # BAD c2
    Entity(44, "Vantage Kitchenware Co",   "Home Appliances",       "210 Oak St, Metro City",         "+1-555-0244", ("Lara Finch",),   ("blenders", "microwaves", "toasters"),                  "Granite Bay Distribution Ltd"),  # BAD c2
    Entity(45, "Halcyon Electronics Trading","Consumer Electronics","4 Pier Rd, Brightbay",           "+1-555-0245", ("Dmitri Vale",),  ("laptops", "monitors", "cables"),                       "Granite Bay Distribution Ltd"),  # BAD c2
    Entity(46, "Stillwater Appliance Group","Home Appliances",      "33 Elm Way, Eastport",           "+1-555-0246", ("Owen Marsh",),   ("refrigerators", "freezers", "ice makers"),             "Granite Bay Distribution Ltd"),  # BAD c2
    Entity(47, "Granary Home Tech",        "Consumer Electronics",  "99 Bridge St, Riverton",         "+1-555-0247", ("Cleo Nash",),    ("smart home", "thermostats", "sensors"),                "Granite Bay Distribution Ltd"),  # BAD c2
    Entity(48, "Pinnacle White Goods",     "Home Appliances",       "15 Crown Rd, Lakeview",          "+1-555-0248", ("Reed Salt",),    ("ovens", "cooktops", "range hoods"),                    "Granite Bay Distribution Ltd"),  # BAD c2
]

# Ground truth, split into cohorts.
COHORT_1: set[int] = {4, 9, 13, 18, 22, 27, 31, 36}   # original signals (lenses catch these)
COHORT_2: set[int] = set(range(41, 49))               # disguised; only a shared downstream buyer
ANSWER_KEY: set[int] = COHORT_1 | COHORT_2


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
            f"| products: {', '.join(e.product_keywords)} "
            f"| ships_to: {e.ships_to}"
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


def evaluate_scope(flags_by_lens: dict[str, set[int]],
                   truth: set[int], exclude: set[int]) -> tuple[dict[str, Score], Score]:
    """Score every lens + the union on one cohort scope.

    `exclude` removes the OTHER cohort's bad ids from both predictions and truth,
    so e.g. the cohort-2 score isn't credited or penalised for cohort-1 hits.
    Legitimate decoys stay in scope, so precision is still real.
    """
    t = truth - exclude
    rows = {name: score(ids - exclude, t) for name, ids in flags_by_lens.items()}
    uni = score(union_ensemble(flags_by_lens) - exclude, t)
    return rows, uni


def main() -> None:
    print("AutoSwarm - shell-network detection (adversary adaptation)\n")
    print(f"Registry: {len(REGISTRY)} companies")
    print(f"Cohort 1 (original signals): {sorted(COHORT_1)}")
    print(f"Cohort 2 (disguised, shared buyer only): {sorted(COHORT_2)}")
    print(f"Lenses: {[name for name, _ in SWARM]}\n")

    if not _check_key():
        return

    print(f"Running {len(SWARM)} lenses (gpt-4o-mini); each sees the WHOLE registry...\n")
    flags = run_swarm(SWARM, REGISTRY)

    c1_rows, c1_uni = evaluate_scope(flags, COHORT_1, COHORT_2)   # cohort 1 only
    c2_rows, c2_uni = evaluate_scope(flags, COHORT_2, COHORT_1)   # cohort 2 only
    cb_rows, cb_uni = evaluate_scope(flags, ANSWER_KEY, set())    # combined

    print("F1 BY COHORT (current swarm, no recovery yet)")
    print("=" * 58)
    print(f"{'LENS':<18}{'COHORT 1':>12}{'COHORT 2':>12}{'COMBINED':>12}")
    print("-" * 58)
    for name, _ in SWARM:
        print(f"{name:<18}{c1_rows[name].f1:>12.2f}{c2_rows[name].f1:>12.2f}{cb_rows[name].f1:>12.2f}")
    print("-" * 58)
    print(f"{'ENSEMBLE (union)':<18}{c1_uni.f1:>12.2f}{c2_uni.f1:>12.2f}{cb_uni.f1:>12.2f}")
    print("=" * 58)


if __name__ == "__main__":
    main()
