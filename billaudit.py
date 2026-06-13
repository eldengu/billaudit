"""
billaudit.py — The SPINE of a medical-bill error detector.

One file, runnable, minimal. Synthetic data only.

Pipeline:
  synthetic bill (12 lines, 5 planted errors)
    -> train / held-out split (~60/40, each a mix of error + correct)
    -> F1 scorer (precision / recall / F1 so "flag everything" can't win)
    -> ONE detector backed by the OpenAI API (gpt-4o-mini)
    -> print train F1 and held-out F1 side by side

The detector is a plain function with a fixed signature, so it is trivially
swappable for a future ensemble / "swarm" without touching the harness.

Before running: put your real key in .env  ->  OPENAI_API_KEY=sk-...
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()  # reads .env in the working directory


# ---------------------------------------------------------------------------
# 1. Synthetic medical bill
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LineItem:
    id: int
    cpt_code: str
    description: str
    qty: int
    unit_price: float


# 12 line items. Synthetic, not from any real patient or claim.
#
# Planted errors (answer key below). A "context" line is included so some
# errors are only detectable by comparison (e.g. the duplicate).
#
#   id 2  duplicate charge      — same CPT + description as id 1 (obvious-ish)
#   id 5  wrong CPT code        — code says nerve block, description is a flu shot (subtle)
#   id 7  inflated price        — routine venipuncture billed at $450 vs ~$15 (obvious)
#   id 9  phantom service       — MRI w/ contrast that was never performed (subtle)
#   id 11 quantity error        — 90-min psychotherapy billed qty 8 in one visit (obvious)
BILL: list[LineItem] = [
    LineItem(1,  "99213", "Office/outpatient visit, established patient, 20-29 min", 1, 120.00),
    LineItem(2,  "99213", "Office/outpatient visit, established patient, 20-29 min", 1, 120.00),  # ERR: duplicate of #1
    LineItem(3,  "85025", "Complete blood count (CBC) with differential",            1,  35.00),
    LineItem(4,  "80053", "Comprehensive metabolic panel",                           1,  45.00),
    LineItem(5,  "64483", "Influenza vaccine, intramuscular",                        1,  40.00),  # ERR: CPT 64483 = transforaminal epidural injection, not a flu shot
    LineItem(6,  "93000", "Electrocardiogram (ECG), routine, with interpretation",   1,  55.00),
    LineItem(7,  "36415", "Routine venipuncture (blood draw)",                       1, 450.00),  # ERR: inflated (~$15 typical)
    LineItem(8,  "71046", "Chest X-ray, 2 views",                                    1,  90.00),
    LineItem(9,  "70553", "MRI brain with and without contrast",                     1, 1200.00), # ERR: phantom — never performed
    LineItem(10, "99214", "Office/outpatient visit, established patient, 30-39 min", 1, 175.00),
    LineItem(11, "90837", "Psychotherapy, 60 min",                                   8, 160.00),  # ERR: qty 8 in a single session is impossible
    LineItem(12, "82947", "Glucose, quantitative, blood",                           1,  18.00),
]

# Ground truth: the set of line ids that contain a planted error.
ANSWER_KEY: set[int] = {2, 5, 7, 9, 11}


# ---------------------------------------------------------------------------
# 2. Train / held-out split (~60/40), each keeping a mix of error + correct
# ---------------------------------------------------------------------------
#
# 12 lines -> 7 train / 5 held-out.
# Errors {2,5,7,9,11} split so neither side is all-error or all-correct:
#   train errors:    {2, 7, 11}   held-out errors: {5, 9}
TRAIN_IDS:    set[int] = {1, 2, 3, 6, 7, 10, 11}   # 3 of 5 errors, 4 correct
HELDOUT_IDS:  set[int] = {4, 5, 8, 9, 12}           # 2 of 5 errors, 3 correct

TRAIN   = [li for li in BILL if li.id in TRAIN_IDS]
HELDOUT = [li for li in BILL if li.id in HELDOUT_IDS]


# ---------------------------------------------------------------------------
# 3. F1 scorer
# ---------------------------------------------------------------------------

@dataclass
class Score:
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int


def score(predicted_error_ids: set[int], truth_error_ids: set[int]) -> Score:
    """Compare predicted error ids to truth. Precision/recall/F1.

    Recall alone is gamed by flagging everything; precision punishes that,
    so F1 keeps an honest balance.
    """
    tp = len(predicted_error_ids & truth_error_ids)
    fp = len(predicted_error_ids - truth_error_ids)
    fn = len(truth_error_ids - predicted_error_ids)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return Score(precision, recall, f1, tp, fp, fn)


# ---------------------------------------------------------------------------
# 4. The SWARM of detectors (SWAPPABLE)
# ---------------------------------------------------------------------------
#
# Signature contract for any detector:
#     detector(line: LineItem) -> bool      # True == "this line is an error"
#
# Each detector is the SAME model (gpt-4o-mini) seen through a DIFFERENT
# system-prompt "lens". A factory builds one detector per strategy, so they all
# share the signature and live in a plain list -> trivially swappable, and ready
# to grow (an ensemble now; evolution later).

_LINE_TEMPLATE = """Line item:
  CPT code:    {cpt_code}
  Description: {description}
  Quantity:    {qty}
  Unit price:  ${unit_price:.2f}

Is this line item LIKELY a billing error? Answer with a single word: YES or NO."""


def _line_block(line: LineItem) -> str:
    return _LINE_TEMPLATE.format(
        cpt_code=line.cpt_code,
        description=line.description,
        qty=line.qty,
        unit_price=line.unit_price,
    )


# Five strategies. Each one is biased toward a single failure mode and is told
# to stay quiet on lines outside its specialty, so the swarm is diverse rather
# than five copies of the same generalist.
SYSTEM_PROMPTS: dict[str, str] = {
    "duplicate": (
        "You audit medical-bill line items, specializing in DUPLICATE / redundant "
        "charges. Routine, low-cost services billed in standard units are the kind "
        "most often double-billed. Answer YES only if this line looks like a "
        "duplicate-prone or redundant charge; otherwise NO. Reply YES or NO only."
    ),
    "price": (
        "You audit medical-bill line items, specializing in PRICE plausibility. "
        "Compare the unit price against typical US rates for the described service. "
        "Answer YES only if the price is implausibly inflated for what was done; "
        "otherwise NO. Reply YES or NO only."
    ),
    "cpt_match": (
        "You audit medical-bill line items, specializing in CPT-code/description "
        "mismatches. Check whether the CPT code actually corresponds to the written "
        "description of the service. Answer YES only if the code and description do "
        "not match; otherwise NO. Reply YES or NO only."
    ),
    "phantom": (
        "You audit medical-bill line items, specializing in PHANTOM or "
        "never-performed services: high-cost procedures that look out of place or "
        "unlikely to have actually been delivered. Answer YES only if the service "
        "looks like it may not have been performed; otherwise NO. Reply YES or NO only."
    ),
    "general": (
        "You audit medical-bill line items for ANY billing error: duplicates, "
        "CPT/description mismatches, inflated prices, phantom services, or impossible "
        "quantities. Answer YES if this line is likely a billing error; otherwise NO. "
        "Reply YES or NO only."
    ),
}


def make_openai_detector(system_prompt: str):
    """Build a detector(line) -> bool backed by gpt-4o-mini under `system_prompt`."""

    def detector(line: LineItem) -> bool:
        from openai import OpenAI

        client = OpenAI()  # reads OPENAI_API_KEY from the environment (.env)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            max_tokens=3,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": _line_block(line)},
            ],
        )
        answer = (resp.choices[0].message.content or "").strip().upper()
        return answer.startswith("Y")

    return detector


# The swarm: a list of (name, detector) pairs, all matching the signature.
SWARM: list[tuple[str, object]] = [
    (name, make_openai_detector(prompt)) for name, prompt in SYSTEM_PROMPTS.items()
]

# Backwards-compatible single-detector handle (the general lens).
DETECTOR = SWARM[-1][1]


# ---------------------------------------------------------------------------
# 5. Harness
# ---------------------------------------------------------------------------

def run_detector(detector, lines: list[LineItem]) -> set[int]:
    """Run `detector` over each line; return the set of ids it flagged."""
    return {li.id for li in lines if detector(li)}


def _check_key() -> bool:
    key = os.getenv("OPENAI_API_KEY", "")
    if not key or key == "replace-me":
        print("=" * 64)
        print("  OPENAI_API_KEY is not set.")
        print("  Put your real key in the .env file before running:")
        print()
        print("      OPENAI_API_KEY=sk-...")
        print()
        print("  Then re-run:  python billaudit.py")
        print("=" * 64)
        return False
    return True


def run_swarm(swarm, lines: list[LineItem]) -> dict[str, set[int]]:
    """Run every detector over `lines`; return {detector_name: flagged ids}."""
    return {name: run_detector(det, lines) for name, det in swarm}


def majority_ensemble(flags_by_detector: dict[str, set[int]],
                      lines: list[LineItem]) -> set[int]:
    """Flag a line if a strict majority of detectors flagged it.

    Derived from the per-detector flags above, so the ensemble costs no extra
    API calls.
    """
    n = len(flags_by_detector)
    threshold = n // 2 + 1
    flagged = set()
    for li in lines:
        votes = sum(1 for ids in flags_by_detector.values() if li.id in ids)
        if votes >= threshold:
            flagged.add(li.id)
    return flagged


def union_ensemble(flags_by_detector: dict[str, set[int]],
                   lines: list[LineItem]) -> set[int]:
    """Flag a line if ANY single detector flagged it (1-vote threshold).

    The opposite bet from majority voting: trust a lone confident specialist.
    Should rescue subtle errors only one lens caught, at the cost of precision.
    Derived from the same per-detector flags -> no extra API calls.
    """
    line_ids = {li.id for li in lines}
    flagged = set()
    for ids in flags_by_detector.values():
        flagged |= ids
    return flagged & line_ids


def main() -> None:
    print("billaudit - medical-bill error detector (SWARM)\n")
    print(f"Bill: {len(BILL)} lines | planted errors: {sorted(ANSWER_KEY)}")
    print(f"Train ids:    {sorted(TRAIN_IDS)}")
    print(f"Held-out ids: {sorted(HELDOUT_IDS)}")
    print(f"Detectors:    {[name for name, _ in SWARM]}\n")

    if not _check_key():
        return

    train_truth   = ANSWER_KEY & TRAIN_IDS
    heldout_truth = ANSWER_KEY & HELDOUT_IDS

    print(f"Running {len(SWARM)} detectors (gpt-4o-mini) over each line...\n")
    train_flags   = run_swarm(SWARM, TRAIN)
    heldout_flags = run_swarm(SWARM, HELDOUT)

    # Per-detector scores.
    rows: list[tuple[str, Score, Score]] = []
    for name, _ in SWARM:
        ts = score(train_flags[name],   train_truth)
        hs = score(heldout_flags[name], heldout_truth)
        rows.append((name, ts, hs))

    # Ensembles, both derived from the same per-detector flags.
    maj_ts = score(majority_ensemble(train_flags,   TRAIN),    train_truth)
    maj_hs = score(majority_ensemble(heldout_flags, HELDOUT),  heldout_truth)
    uni_ts = score(union_ensemble(train_flags,   TRAIN),       train_truth)
    uni_hs = score(union_ensemble(heldout_flags, HELDOUT),     heldout_truth)

    # Table.
    print("RESULTS")
    print("=" * 52)
    print(f"{'DETECTOR':<22}{'TRAIN F1':>12}{'HELD-OUT F1':>16}")
    print("-" * 52)
    for name, ts, hs in rows:
        print(f"{name:<22}{ts.f1:>12.2f}{hs.f1:>16.2f}")
    print("-" * 52)
    print(f"{'ENSEMBLE (majority)':<22}{maj_ts.f1:>12.2f}{maj_hs.f1:>16.2f}")
    print(f"{'ENSEMBLE (union/any)':<22}{uni_ts.f1:>12.2f}{uni_hs.f1:>16.2f}")
    print("=" * 52)


if __name__ == "__main__":
    main()
