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
   - go to the dashboard's event tester and send an event ({} payload is fine):
       * autoswarm/ping               -> instant plumbing check
       * autoswarm/evaluate.requested -> REAL swarm run (one OpenAI call per
                                         lens over the whole registry, ~10-20s),
                                         returns {"f1",..,"precision",..,"recall",
                                         ..,"flagged_ids":[...],"n_errors":..}
   - watch the run appear and the durable step complete with the output.

  The evaluate run needs OPENAI_API_KEY set in .env.
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


# 4. Serve BOTH functions over Flask at the standard Inngest path (/api/inngest).
app = Flask(__name__)
inngest.flask.serve(app, inngest_client, [ping, evaluate_swarm])


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
    print("=" * 72)


if __name__ == "__main__":
    _print_instructions()
    # Reloader off so the function registers exactly once.
    app.run(host="127.0.0.1", port=PORT, use_reloader=False)
