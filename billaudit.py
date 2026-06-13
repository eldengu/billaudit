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
# 4. The detector (SWAPPABLE)
# ---------------------------------------------------------------------------
#
# Signature contract for any detector:
#     detector(line: LineItem) -> bool      # True == "this line is an error"
#
# Swap `DETECTOR = openai_detector` for any function with this signature
# (rules engine, ensemble/swarm, a different model) without touching the harness.

_PROMPT = """You are auditing a single line item from a medical bill for billing errors.

Common billing errors include: duplicate charges, a CPT code that does not match \
the service description, an implausibly inflated unit price, a phantom service \
(billed but not medically performed), and an impossible quantity.

Line item:
  CPT code:    {cpt_code}
  Description: {description}
  Quantity:    {qty}
  Unit price:  ${unit_price:.2f}

Is this line item LIKELY a billing error? Answer with a single word: YES or NO."""


def openai_detector(line: LineItem) -> bool:
    """Ask gpt-4o-mini whether one line is likely a billing error. Returns bool."""
    from openai import OpenAI

    client = OpenAI()  # reads OPENAI_API_KEY from the environment (.env)
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        max_tokens=3,
        messages=[{
            "role": "user",
            "content": _PROMPT.format(
                cpt_code=line.cpt_code,
                description=line.description,
                qty=line.qty,
                unit_price=line.unit_price,
            ),
        }],
    )
    answer = (resp.choices[0].message.content or "").strip().upper()
    return answer.startswith("Y")


# Active detector. Point this at anything matching the signature contract.
DETECTOR = openai_detector


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


def _print_score(label: str, lines: list[LineItem], s: Score) -> None:
    truth = {li.id for li in lines if li.id in ANSWER_KEY}
    print(f"  {label}")
    print(f"    lines={len(lines)}  errors_present={len(truth)}  "
          f"(tp={s.tp} fp={s.fp} fn={s.fn})")
    print(f"    precision={s.precision:.2f}  recall={s.recall:.2f}  F1={s.f1:.2f}")


def main() -> None:
    print("billaudit - medical-bill error detector (SPINE)\n")
    print(f"Bill: {len(BILL)} lines | planted errors: {sorted(ANSWER_KEY)}")
    print(f"Train ids:    {sorted(TRAIN_IDS)}")
    print(f"Held-out ids: {sorted(HELDOUT_IDS)}\n")

    if not _check_key():
        return

    train_truth   = ANSWER_KEY & TRAIN_IDS
    heldout_truth = ANSWER_KEY & HELDOUT_IDS

    print("Running detector (gpt-4o-mini) over each line...\n")
    train_pred   = run_detector(DETECTOR, TRAIN)
    heldout_pred = run_detector(DETECTOR, HELDOUT)

    train_score   = score(train_pred,   train_truth)
    heldout_score = score(heldout_pred, heldout_truth)

    print("RESULTS")
    print("-" * 64)
    _print_score("TRAIN", TRAIN, train_score)
    print()
    _print_score("HELD-OUT", HELDOUT, heldout_score)
    print("-" * 64)
    print(f"  TRAIN F1 = {train_score.f1:.2f}   |   HELD-OUT F1 = {heldout_score.f1:.2f}")


if __name__ == "__main__":
    main()
