"""
inngest_app.py — AutoSwarm, STEP 1: prove the Inngest plumbing works locally.

Minimal on purpose. No swarm, no OpenAI. Just: an event triggers one Inngest
function, which runs ONE durable step that returns a small dict. If that shows
up green in the dev dashboard, the orchestration plumbing is real and we can
hang actual work off it later.

Built against the installed SDK: inngest 0.5.18 (sync API) + Flask 3.x.

------------------------------------------------------------------------------
HOW TO RUN (two terminals)
------------------------------------------------------------------------------
  Terminal 1 — serve this app on :8000
      python inngest_app.py

  Terminal 2 — start the Inngest dev server + dashboard, pointed at this app
      npx inngest-cli@latest dev -u http://127.0.0.1:8000/api/inngest

  Then:
   - open the dashboard URL the CLI prints (usually http://localhost:8288)
   - go to the "Functions" tab and confirm "autoswarm-ping" is registered
   - go to the dashboard's event tester and send an event:
       * autoswarm/ping               -> instant plumbing check ({} payload)
       * autoswarm/evaluate.requested -> REAL swarm run (one OpenAI call per
                                         lens over the whole registry, ~10-20s),
                                         returns {"f1",..,"precision",..,"recall",
                                         ..,"flagged_ids":[...],"n_errors":..}
       * autoswarm/data.arrived       -> the FINALE: durable cat-and-mouse loop,
                                         ~1-2s, observable step-by-step. Send
                                         with data:
                                           {"scenario":"baseline"}          (healthy)
                                           {"scenario":"adversary_adapted"} (crater
                                              -> explore -> evolve -> recover)
   - watch the run appear and the durable step(s) complete with the output.

  Only autoswarm/evaluate.requested needs OPENAI_API_KEY in .env; the finale
  replays proven values, so it runs offline.
------------------------------------------------------------------------------
"""

import os

from dotenv import load_dotenv

load_dotenv()  # OPENAI_API_KEY from .env, so the detector lenses can call the model

import inngest
import inngest.flask
from flask import Flask

# Reuse the swarm pieces from supplychain.py. Importing it is side-effect-safe:
# its main()/evolution only runs under `if __name__ == "__main__"`, and building
# SWARM just constructs detector closures (no API calls happen at import time).
import supplychain as sc

PORT = 8000

# 1. Inngest client. Dev mode by default; set AUTOSWARM_PRODUCTION=1 to flip it.
#    (is_production=False makes the SDK talk to the local dev server.)
_is_production = os.environ.get("AUTOSWARM_PRODUCTION", "").lower() in ("1", "true", "yes")

inngest_client = inngest.Inngest(
    app_id="autoswarm",
    is_production=_is_production,
)


# 2. ONE function, triggered by the "autoswarm/ping" event.
@inngest_client.create_function(
    fn_id="autoswarm-ping",
    name="AutoSwarm — ping (plumbing check)",
    trigger=inngest.TriggerEvent(event="autoswarm/ping"),
)
def ping(ctx: inngest.ContextSync) -> dict:
    """Run a single durable step that returns a tiny payload. No real work."""

    def _plumbing_check() -> dict:
        return {"status": "ok", "f1": 0.94, "note": "plumbing works"}

    # step.run makes the work durable: its result is checkpointed by Inngest.
    result = ctx.step.run("plumbing-check", _plumbing_check)
    return result


# 3. REAL work: run the union swarm over the registry and score it vs the key.
@inngest_client.create_function(
    fn_id="autoswarm-evaluate",
    name="AutoSwarm — evaluate swarm (union F1)",
    trigger=inngest.TriggerEvent(event="autoswarm/evaluate.requested"),
)
def evaluate_swarm(ctx: inngest.ContextSync) -> dict:
    """Run every lens over the whole registry, pool with the union ensemble, and
    score precision/recall/F1 against the answer key. Makes real OpenAI calls
    (one per lens), so the durable step takes ~10-20s."""

    def _evaluate() -> dict:
        flags = sc.run_swarm(sc.SWARM, sc.REGISTRY)   # one model call per lens
        flagged = sc.union_ensemble(flags)            # flag if ANY lens flags
        s = sc.score(flagged, sc.ANSWER_KEY)
        return {
            "f1": round(s.f1, 4),
            "precision": round(s.precision, 4),
            "recall": round(s.recall, 4),
            "flagged_ids": sorted(flagged),
            "n_errors": len(sc.ANSWER_KEY),
        }

    return ctx.step.run("evaluate", _evaluate)


# 4. THE FINALE: adaptive cat-and-mouse loop as a durable workflow. Each beat is
#    its own step.run, so the dashboard shows crater -> explore -> evolve -> recover.
@inngest_client.create_function(
    fn_id="autoswarm-monitor-adapt",
    name="AutoSwarm — monitor & adapt (cat-and-mouse)",
    trigger=inngest.TriggerEvent(event="autoswarm/data.arrived"),
)
def monitor_and_adapt(ctx: inngest.ContextSync) -> dict:
    """Durable monitor->detect->(explore->evolve->recover) loop.

    Built for a FAST, RELIABLE live demo: the steps REPLAY our already-proven
    supplychain.py results, so a run is ~1-2s with no live-API wait on stage.
    The orchestration is real Inngest; the F1 numbers are our experimental
    values. To run it live in production, swapping the evaluate step to a real
    swarm call is a one-liner:
        f1 = sc.score(sc.union_ensemble(sc.run_swarm(sc.SWARM, sc.REGISTRY)),
                      sc.ANSWER_KEY).f1
    """
    scenario = (ctx.event.data or {}).get("scenario", "baseline")
    THRESHOLD = 0.80

    # 1. Evaluate the swarm's current union F1 for this scenario.
    def _evaluate() -> dict:
        f1 = 0.64 if scenario == "adversary_adapted" else 0.94
        return {"scenario": scenario, "f1": f1}

    current = ctx.step.run("evaluate_current", _evaluate)

    # 2. Has performance dropped below the health threshold?
    def _detect() -> dict:
        return {"dropped": current["f1"] < THRESHOLD, "f1": current["f1"], "threshold": THRESHOLD}

    drop = ctx.step.run("detect_drop", _detect)

    # 3a. Healthy path: exploit, no evolution needed.
    if not drop["dropped"]:
        def _no_action() -> dict:
            return {"status": "healthy — swarm exploited, no evolution needed"}

        healthy = ctx.step.run("no_action", _no_action)
        return {
            "scenario": scenario,
            "dropped": False,
            "trajectory": [current["f1"]],
            "outcome": healthy["status"],
        }

    # 3b. Dropped: explore -> evolve -> confirm recovery, each a durable step.
    def _explore() -> dict:
        return {"discovered": "shared-consignee / downstream-buyer signal",
                "note": "probing fields no current lens covers"}

    discovered = ctx.step.run("explore", _explore)

    def _evolve() -> dict:
        return {"action": "coverage-aware selection + merge added a consignee lens",
                "new_f1": 1.0}

    evolved = ctx.step.run("evolve_and_select", _evolve)

    def _confirm() -> dict:
        return {"recovered": True, "f1_before": current["f1"], "f1_after": evolved["new_f1"]}

    recovery = ctx.step.run("confirm_recovery", _confirm)

    # 4. Final summary with the trajectory.
    return {
        "scenario": scenario,
        "dropped": True,
        "discovered": discovered["discovered"],
        "action": evolved["action"],
        "recovered": recovery["recovered"],
        "trajectory": [current["f1"], evolved["new_f1"]],
        "outcome": (f"recovered {current['f1']} -> {evolved['new_f1']} by "
                    "rediscovering the surviving consignee invariant"),
    }


# 5. Serve all three functions over Flask at the standard Inngest path (/api/inngest).
app = Flask(__name__)
inngest.flask.serve(app, inngest_client, [ping, evaluate_swarm, monitor_and_adapt])


def _print_instructions() -> None:
    mode = "PRODUCTION" if _is_production else "DEV"
    print("=" * 72)
    print(f"  AutoSwarm Inngest plumbing test  ({mode} mode)  serving on :{PORT}")
    print("=" * 72)
    print(f"  Endpoint:   http://127.0.0.1:{PORT}/api/inngest")
    print()
    print("  Terminal 1 (this one):")
    print("      python inngest_app.py")
    print()
    print("  Terminal 2 — Inngest dev server + dashboard:")
    print(f"      npx inngest-cli@latest dev -u http://127.0.0.1:{PORT}/api/inngest")
    print()
    print("  Then open the dashboard (usually http://localhost:8288) and send:")
    print("     'autoswarm/ping'                -> instant plumbing check")
    print("     'autoswarm/evaluate.requested'  -> real swarm run (~10-20s, OpenAI)")
    print("     'autoswarm/data.arrived'        -> FINALE cat-and-mouse loop (~1-2s)")
    print("         data {\"scenario\":\"baseline\"} or {\"scenario\":\"adversary_adapted\"}")
    print("=" * 72)


if __name__ == "__main__":
    _print_instructions()
    # Reloader off so the function registers exactly once.
    app.run(host="127.0.0.1", port=PORT, use_reloader=False)
