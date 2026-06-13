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
import re
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


# 35 line items. Synthetic, not from any real patient or claim.
#
# 10 planted errors (answer key below), 2 of each of the 5 types, mixing
# obvious with subtle. Several are only detectable WITH the whole bill in view
# (the duplicates compare to a far-away twin; the phantoms look out of place
# only against the rest of the visit):
#
#   duplicate:  id 3  (adjacent twin of #2, obvious) ; id 19 (twin of #4, far apart, subtle)
#   wrong CPT:  id 27 (71046 is a chest X-ray, billed as a lipid panel, obvious)
#               id 11 (64483 is a lumbar epidural injection, billed as a flu vaccine, subtle)
#   inflated:   id 7  (venipuncture $380 vs ~$15, obvious) ; id 23 (urinalysis $145 vs ~$15, subtle)
#   phantom:    id 31 (colonoscopy on an otherwise sick-visit bill, obvious-ish)
#               id 15 (brain MRI w/ & w/o contrast, never performed, subtle)
#   quantity:   id 9  (6 psychotherapy sessions in one day, obvious)
#               id 21 (12 immunization administrations, subtle-ish)
BILL: list[LineItem] = [
    LineItem(1,  "99214", "Office/outpatient visit, established patient, 30-39 min", 1,  175.00),
    LineItem(2,  "99213", "Office/outpatient visit, established patient, 20-29 min", 1,  120.00),
    LineItem(3,  "99213", "Office/outpatient visit, established patient, 20-29 min", 1,  120.00),  # ERR duplicate of #2 (obvious)
    LineItem(4,  "85025", "Complete blood count (CBC) with differential",           1,   35.00),
    LineItem(5,  "80053", "Comprehensive metabolic panel",                          1,   45.00),
    LineItem(6,  "80061", "Lipid panel",                                            1,   40.00),
    LineItem(7,  "36415", "Routine venipuncture (blood draw)",                      1,  380.00),  # ERR inflated (~$15 typical, obvious)
    LineItem(8,  "83036", "Hemoglobin A1c",                                         1,   30.00),
    LineItem(9,  "90837", "Psychotherapy, 60 min",                                  6,  160.00),  # ERR quantity (6 sessions in one day impossible, obvious)
    LineItem(10, "84443", "Thyroid stimulating hormone (TSH)",                      1,   45.00),
    LineItem(11, "64483", "Influenza vaccine, quadrivalent, intramuscular",         1,   40.00),  # ERR wrong CPT (64483 = lumbar transforaminal epidural injection, subtle)
    LineItem(12, "90471", "Immunization administration",                           1,   25.00),
    LineItem(13, "93000", "Electrocardiogram (ECG), routine, with interpretation",  1,   55.00),
    LineItem(14, "71046", "Chest X-ray, 2 views",                                   1,   90.00),
    LineItem(15, "70553", "MRI brain with and without contrast",                    1, 1200.00),  # ERR phantom (never performed, subtle)
    LineItem(16, "81001", "Urinalysis, automated, with microscopy",                 1,   15.00),
    LineItem(17, "99396", "Preventive visit, established patient, 40-64 yr",        1,  200.00),
    LineItem(18, "96372", "Therapeutic injection, subcutaneous/intramuscular",      1,   30.00),
    LineItem(19, "85025", "Complete blood count (CBC) with differential",           1,   35.00),  # ERR duplicate of #4 (far apart, subtle)
    LineItem(20, "87880", "Strep A rapid antigen test",                             1,   30.00),
    LineItem(21, "90471", "Immunization administration",                          12,   25.00),  # ERR quantity (12 admins, subtle-ish)
    LineItem(22, "20610", "Arthrocentesis, major joint",                           1,  150.00),
    LineItem(23, "81002", "Urinalysis, non-automated, without microscopy",         1,  145.00),  # ERR inflated (~$15 typical, subtle)
    LineItem(24, "12001", "Simple repair of superficial wound, 2.5 cm",            1,  150.00),
    LineItem(25, "17000", "Destruction of premalignant lesion (first lesion)",     1,  120.00),
    LineItem(26, "97110", "Therapeutic exercise, 15 min",                          2,   40.00),
    LineItem(27, "71046", "Lipid panel, fasting",                                   1,   40.00),  # ERR wrong CPT (71046 = chest X-ray, not a lipid panel, obvious)
    LineItem(28, "99000", "Specimen handling / transfer",                          1,   10.00),
    LineItem(29, "82947", "Glucose, quantitative, blood",                          1,   18.00),
    LineItem(30, "90686", "Influenza vaccine, quadrivalent, intramuscular",        1,   40.00),
    LineItem(31, "45378", "Colonoscopy, diagnostic",                               1,  950.00),  # ERR phantom (out of place on a sick visit, obvious-ish)
    LineItem(32, "97140", "Manual therapy techniques, 15 min",                     1,   40.00),
    LineItem(33, "99212", "Office/outpatient visit, established patient, 10-19 min", 1,  80.00),
    LineItem(34, "73721", "MRI, lower extremity joint, without contrast",          1,  700.00),
    LineItem(35, "85027", "Complete blood count (CBC), automated",                 1,   25.00),
]

# Ground truth: the set of line ids that contain a planted error.
ANSWER_KEY: set[int] = {3, 7, 9, 11, 15, 19, 21, 23, 27, 31}


# ---------------------------------------------------------------------------
# 2. Validation / test split (three-way discipline)
# ---------------------------------------------------------------------------
#
# Detectors always see the WHOLE bill (so cross-line context works); a "split"
# here just means which line ids count toward a given score.
#   VALIDATION (ids 1-20)  -> the ONLY selection signal during evolution
#   TEST       (ids 21-35) -> scored exactly ONCE, at the very end
# Errors per split: validation {3,7,9,11,15,19} (6) ; test {21,23,27,31} (4).
VALIDATION_IDS: set[int] = set(range(1, 21))    # ids 1-20
TEST_IDS:       set[int] = set(range(21, 36))   # ids 21-35


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
#     detector(lines: list[LineItem]) -> set[int]   # ids it judges to be errors
#
# Each detector now sees the WHOLE bill (numbered) and returns the SET of error
# ids -- so cross-line errors (duplicates, out-of-place phantom services) become
# detectable by comparison, which per-line judgement could never catch.
#
# Each detector is the SAME model (gpt-4o-mini) seen through a DIFFERENT
# system-prompt "lens". A factory builds one detector per strategy, so they all
# share the signature and live in a plain list -> trivially swappable.


def _format_bill(lines: list[LineItem]) -> str:
    return "\n".join(
        f"  id {li.id}: CPT {li.cpt_code} | {li.description} "
        f"| qty {li.qty} | ${li.unit_price:.2f}"
        for li in lines
    )


# Five strategies. Each one is biased toward a single failure mode, so the swarm
# is diverse rather than five copies of the same generalist.
SYSTEM_PROMPTS: dict[str, str] = {
    "duplicate": (
        "You audit a whole medical bill, specializing in DUPLICATE / redundant "
        "charges: the same service billed more than once for one encounter. "
        "Compare line items against each other across the entire bill."
    ),
    "price": (
        "You audit a whole medical bill, specializing in PRICE plausibility. "
        "Compare each unit price against typical US rates for the described "
        "service and flag prices that are implausibly inflated."
    ),
    "cpt_match": (
        "You audit a whole medical bill, specializing in CPT-code/description "
        "mismatches: a line whose CPT code does not correspond to the written "
        "description of the service."
    ),
    "phantom": (
        "You audit a whole medical bill, specializing in PHANTOM / never-performed "
        "services: high-cost or out-of-place procedures unlikely to have actually "
        "been delivered given the rest of the visit."
    ),
    "general": (
        "You audit a whole medical bill for ANY billing error: duplicates, "
        "CPT/description mismatches, inflated prices, phantom services, or "
        "impossible quantities."
    ),
}

_USER_TEMPLATE = """Here is a medical bill. Review ALL line items together, comparing them \
against one another where useful.

{bill}

List the id numbers of every line item that is LIKELY a billing error.
Respond with ONLY a comma-separated list of ids (e.g. "3, 7, 12").
If no line is an error, respond with the single word NONE."""


def make_openai_detector(system_prompt: str):
    """Build a detector(lines) -> set[int] backed by gpt-4o-mini under `system_prompt`."""

    def detector(lines: list[LineItem]) -> set[int]:
        from openai import OpenAI

        client = OpenAI()  # reads OPENAI_API_KEY from the environment (.env)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            max_tokens=120,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": _USER_TEMPLATE.format(bill=_format_bill(lines))},
            ],
        )
        text = (resp.choices[0].message.content or "").strip()
        valid = {li.id for li in lines}
        # Parse the integer ids the model returned; keep only ones on this bill.
        return {int(tok) for tok in re.findall(r"\d+", text)} & valid

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
    """Run every detector over the whole bill; return {detector_name: flagged ids}."""
    return {name: det(lines) for name, det in swarm}


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


# ---------------------------------------------------------------------------
# 6. Evolution (selection + mutation; no merge yet)
# ---------------------------------------------------------------------------

def mutate_prompt(system_prompt: str) -> str:
    """Ask gpt-4o-mini to rewrite a detector's system prompt into a child variant."""
    from openai import OpenAI

    client = OpenAI()
    meta = (
        "Below is a SYSTEM PROMPT giving an assistant a strategy for auditing a "
        "WHOLE medical bill: it is shown every line item and returns a "
        "comma-separated list of the ids that are billing errors.\n"
        "Rewrite it into an improved VARIANT that catches billing errors more "
        "reliably while staying concise. Keep it a strategy/lens description for "
        "reviewing the whole bill; do NOT change the output format and do NOT tell "
        "it to answer YES/NO.\n"
        "Return ONLY the rewritten system prompt, with no preamble or quotes.\n\n"
        f"SYSTEM PROMPT:\n{system_prompt}"
    )
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.7,  # some spread so children differ from the parent
        max_tokens=300,
        messages=[{"role": "user", "content": meta}],
    )
    return (resp.choices[0].message.content or "").strip()


def merge_prompts(prompt_a: str, prompt_b: str) -> str:
    """Ask gpt-4o-mini to combine two detector prompts into one covering BOTH lenses."""
    from openai import OpenAI

    client = OpenAI()
    meta = (
        "Below are TWO system prompts, each giving an assistant a strategy/lens for "
        "auditing a WHOLE medical bill and returning a comma-separated list of the "
        "ids that are billing errors.\n"
        "Combine them into ONE system prompt that covers BOTH strategies and both "
        "error types at once, while staying concise. Keep it a whole-bill strategy "
        "description; do NOT change the output format and do NOT tell it to answer "
        "YES/NO.\n"
        "Return ONLY the combined system prompt, with no preamble or quotes.\n\n"
        f"PROMPT A:\n{prompt_a}\n\nPROMPT B:\n{prompt_b}"
    )
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.7,
        max_tokens=350,
        messages=[{"role": "user", "content": meta}],
    )
    return (resp.choices[0].message.content or "").strip()


def greedy_ensemble_select(flags_by_name: dict[str, set[int]],
                           val_truth: set[int]) -> tuple[list[str], float]:
    """Greedily pick the subset of detectors that maximizes UNION validation F1.

    Start empty; repeatedly add the detector that most improves union validation
    F1; stop when nothing helps. This keeps low-solo-score-but-unique specialists
    (they earn their place by what they add to the ENSEMBLE, not their own F1).
    """
    chosen: list[str] = []
    union: set[int] = set()
    best = 0.0
    remaining = list(flags_by_name)
    while remaining:
        pick, pick_f1 = None, best
        for n in remaining:
            f1 = score(union | (flags_by_name[n] & VALIDATION_IDS), val_truth).f1
            if f1 > pick_f1:
                pick, pick_f1 = n, f1
        if pick is None:
            break  # nothing improves the union -> stop
        chosen.append(pick)
        union |= flags_by_name[pick] & VALIDATION_IDS
        best = pick_f1
        remaining.remove(pick)
    if not chosen:  # degenerate: no detector scores > 0; keep the best solo one
        chosen = [max(flags_by_name,
                      key=lambda n: score(flags_by_name[n] & VALIDATION_IDS, val_truth).f1)]
        best = score(flags_by_name[chosen[0]] & VALIDATION_IDS, val_truth).f1
    return chosen, best


def evolve(initial_prompts: dict[str, str],
           generations: int = 4, pop_size: int = 5) -> None:
    """Ensemble-aware evolution: greedy union-F1 selection + mutation + merge.

    The selection signal is the UNION ensemble's VALIDATION F1 (ids 1-20), NOT
    any individual detector's score -- so unique specialists survive on what they
    add to the ensemble. Each generation: greedily select the subset maximizing
    union validation F1, then refill to pop_size with a couple of merges (kept
    only if they raise union validation F1) plus mutations. The final selected
    ensemble is scored on TEST (ids 21-35) exactly ONCE, at the very end.
    """
    population = list(initial_prompts.items())  # [(name, system_prompt)]
    val_truth = ANSWER_KEY & VALIDATION_IDS

    print("\nEVOLUTION (ensemble-fitness: union VALIDATION F1 = sole signal; +mutation +merge)")
    print("=" * 64)
    print(f"{'GEN':<5}{'POP':>5}{'SEL':>5}{'UNION VAL F1':>15}   selected ensemble")
    print("-" * 64)

    flags: dict[str, set[int]] = {}
    chosen: list[str] = []
    for gen in range(generations):
        prompt_by_name = dict(population)
        swarm = [(name, make_openai_detector(p)) for name, p in population]
        flags = run_swarm(swarm, BILL)  # whole bill seen; scored on a subset below

        chosen, union_val = greedy_ensemble_select(flags, val_truth)
        print(f"{gen:<5}{len(population):>5}{len(chosen):>5}{union_val:>15.2f}   {chosen}")

        # No point evolving after the final generation's score is recorded.
        if gen == generations - 1:
            break

        survivors = [(name, prompt_by_name[name]) for name in chosen]
        next_pop = list(survivors)  # ensemble-level elitism

        # Baseline the operators must beat: the survivors' union on validation.
        surv_union: set[int] = set()
        for name in chosen:
            surv_union |= flags[name] & VALIDATION_IDS
        surv_f1 = score(surv_union, val_truth).f1

        # MERGE: try a couple of survivor pairs; keep a merge only if it raises union val F1.
        pairs = [(survivors[i], survivors[j])
                 for i in range(len(survivors))
                 for j in range(i + 1, len(survivors))][:2]
        for (na, pa), (nb, pb) in pairs:
            if len(next_pop) >= pop_size:
                break
            merged = merge_prompts(pa, pb)
            mflags = make_openai_detector(merged)(BILL) & VALIDATION_IDS
            if score(surv_union | mflags, val_truth).f1 > surv_f1:
                next_pop.append((f"g{gen + 1}_merge_{na}+{nb}", merged))
                surv_union |= mflags
                surv_f1 = score(surv_union, val_truth).f1

        # MUTATION: fill remaining slots with mutated survivors.
        k = 0
        while len(next_pop) < pop_size:
            _, pprompt = survivors[k % len(survivors)]
            next_pop.append((f"g{gen + 1}_mut{k + 1}", mutate_prompt(pprompt)))
            k += 1

        population = next_pop

    print("-" * 64)
    # TEST scored exactly ONCE, on the FINAL selected ensemble.
    test_truth = ANSWER_KEY & TEST_IDS
    test_pred: set[int] = set()
    for name in chosen:
        test_pred |= flags[name] & TEST_IDS
    union_test = score(test_pred, test_truth).f1
    print(f"FINAL selected ensemble: {chosen}")
    print(f"FINAL union TEST F1 (scored once): {union_test:.2f}")
    print("=" * 64)


def main() -> None:
    print("billaudit - medical-bill error detector (SWARM, whole-bill context)\n")
    print(f"Bill: {len(BILL)} lines | planted errors ({len(ANSWER_KEY)}): {sorted(ANSWER_KEY)}")
    print(f"Detectors: {[name for name, _ in SWARM]}\n")

    if not _check_key():
        return

    print(f"Running {len(SWARM)} detectors (gpt-4o-mini); each sees the WHOLE bill...\n")
    flags = run_swarm(SWARM, BILL)

    # Each detector + both ensembles, scored on the whole bill vs ANSWER_KEY.
    print("RESULTS (whole 35-line bill)")
    print("=" * 48)
    print(f"{'DETECTOR':<24}{'PREC':>8}{'RECALL':>8}{'F1':>8}")
    print("-" * 48)
    for name, _ in SWARM:
        s = score(flags[name], ANSWER_KEY)
        print(f"{name:<24}{s.precision:>8.2f}{s.recall:>8.2f}{s.f1:>8.2f}")
    print("-" * 48)
    maj = score(majority_ensemble(flags, BILL), ANSWER_KEY)
    uni = score(union_ensemble(flags, BILL),    ANSWER_KEY)
    print(f"{'ENSEMBLE (majority)':<24}{maj.precision:>8.2f}{maj.recall:>8.2f}{maj.f1:>8.2f}")
    print(f"{'ENSEMBLE (union/any)':<24}{uni.precision:>8.2f}{uni.recall:>8.2f}{uni.f1:>8.2f}")
    print("=" * 48)

    # Evolve: ensemble-fitness selection + mutation + merge over 4 generations,
    # selecting on union validation F1; score test once at the end.
    evolve(SYSTEM_PROMPTS, generations=4, pop_size=5)


if __name__ == "__main__":
    main()
