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
   - go to the dashboard's event tester and send the event:  autoswarm/ping
       (an empty {} payload is fine — this function ignores the data)
   - watch the run appear, the step "plumbing-check" complete, and the
     function output show {"status":"ok","f1":0.94,"note":"plumbing works"}
------------------------------------------------------------------------------
"""

import os

import inngest
import inngest.flask
from flask import Flask

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


# 3. Serve the function over Flask at the standard Inngest path (/api/inngest).
app = Flask(__name__)
inngest.flask.serve(app, inngest_client, [ping])


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
    print("  Then open the dashboard (usually http://localhost:8288), send the")
    print("  event  'autoswarm/ping'  and watch the function run + step complete.")
    print("=" * 72)


if __name__ == "__main__":
    _print_instructions()
    # Reloader off so the function registers exactly once.
    app.run(host="127.0.0.1", port=PORT, use_reloader=False)
