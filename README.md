# AutoSwarm

A self-evolving swarm of AI auditors that catches **planted** errors via an **ungameable** metric — *we* plant the errors and hold the answer key, and everything is scored with F1 so "flag everything" can never win. The same engine runs across two synthetic domains — **medical bills** (`billaudit.py`) and **supply-chain anomalies** (`supplychain.py`) — and the adaptive cat-and-mouse loop (a swarm that craters when an adversary adapts, then *recovers* by rediscovering a surviving invariant) is orchestrated as a **durable Inngest workflow** (`inngest_app.py`) you can watch step-by-step in a dashboard. **Synthetic data only — no real patients, claims, or companies. It surfaces leads for a human analyst, never verdicts.**

---

## What's inside

| File | What it does |
|------|--------------|
| **`billaudit.py`** | Domain 1. A synthetic 56-line medical bill with planted errors (duplicate, wrong CPT, inflated price, phantom service, impossible quantity). 5 `gpt-4o-mini` "lens" detectors each see the whole bill and return error line-ids; an F1 scorer grades vs the answer key; majority + union ensembles; then a coverage-aware evolution loop (validation/test split, scored once). |
| **`supplychain.py`** | Domain 2. A synthetic 48-company registry with a planted illicit network (shared address/phone, shared officers, facade mismatch, registration clustering). 6 lenses + F1 scorer + union ensemble. Includes the **adversary adaptation**: a second cohort drops the old signals but keeps one surviving invariant (a shared downstream buyer), and a **recovery timeline** comparing exploit-only vs diversity+exploration arms. |
| **`inngest_app.py`** | Durable orchestration with the **Inngest** Python SDK + Flask. Three functions: a plumbing `ping`, a real `evaluate_swarm` (runs the supply-chain swarm via a durable step), and `monitor_and_adapt` (the cat-and-mouse finale, each beat its own durable step). Served at `/api/inngest` on port `8000`. |
| **`web/index.html`** | A self-contained dark-themed results page (inline SVG + vanilla JS, no build step): the research story for both domains and an **animated recovery chart** (Play/Replay, round-3 "adversary adapts" moment). |

---

## Prerequisites

- **Python 3.x** (developed on 3.13) with `pip`
- **Node.js** (provides `npx`) — only needed for the Inngest dev server / dashboard
- An **OpenAI API key** — needed to run the swarms and the live `evaluate` function (the cat-and-mouse finale replays proven values and runs offline)

---

## One-time setup

> Assumes **Windows / PowerShell** from the repo root (`C:\...\billaudit`).

**1. Create a `.env` with your OpenAI key.** `.env` is gitignored — **never commit it.**

```powershell
"OPENAI_API_KEY=your-key-here" | Out-File -Encoding utf8 .env
```

(Or create `.env` by hand with a single line: `OPENAI_API_KEY=your-key-here`.)

**2. Install dependencies.**

```powershell
# core deps for the two swarms (openai, python-dotenv)
pip install -r requirements.txt

# extra deps for the live Inngest demo (NOT in requirements.txt)
pip install inngest flask
```

---

## Run the swarms

Each script makes real `gpt-4o-mini` calls and reads `OPENAI_API_KEY` from `.env`. If the key is missing, the script prints a "set your key" notice and exits.

```powershell
python billaudit.py
```
Expected: a **RESULTS** table (per-lens precision/recall/F1 + `ENSEMBLE (majority)` and `ENSEMBLE (union/any)`), followed by an **EVOLUTION** block (4 generations of coverage-aware selection + merge) and a final **union TEST F1** scored once.

```powershell
python supplychain.py
```
Expected: a base swarm union-F1 line, then a **RECOVERY TIMELINE** table comparing **Arm A (exploit)** vs **Arm B (diversity+exploration)** across 6 rounds (the adversary adapts at round 3), ending with `ARM_A = [...]` / `ARM_B = [...]` arrays. This makes several dozen model calls and takes a couple of minutes.

---

## Run the live Inngest demo (cold start)

This is the centerpiece: a durable workflow you trigger from a dashboard and watch execute step-by-step. You need **two PowerShell windows**.

### Terminal 1 — serve the Inngest functions (Flask, port 8000)

```powershell
python inngest_app.py
```
Serves at **`http://127.0.0.1:8000/api/inngest`**. Leave it running.

### Terminal 2 — start the Inngest dev server + dashboard

```powershell
npx inngest-cli@latest dev -u http://127.0.0.1:8000/api/inngest
```
This launches the local Inngest dev server and prints a dashboard URL — usually **`http://localhost:8288`**. Open it in a browser. The **Functions** tab should list all three: `autoswarm-ping`, `autoswarm-evaluate`, `autoswarm-monitor-adapt`.

> Order matters: start **Terminal 1 first** so the app is reachable when the dev server introspects it.

### Trigger the functions from the dashboard

Use the dashboard's event/test sender. Exact event names and payloads:

**1. Plumbing check** — instant, no key needed.
- Event: `autoswarm/ping`
- Payload: `{}`
- Returns `{"status":"ok","f1":0.94,"note":"plumbing works"}` from one durable step.

**2. Real swarm evaluation** — makes one OpenAI call per lens over the whole registry (~10–20s); needs `OPENAI_API_KEY`.
- Event: `autoswarm/evaluate.requested`
- Payload: `{}`
- Returns `{"f1":…, "precision":…, "recall":…, "flagged_ids":[…], "n_errors":…}`.

**3. Cat-and-mouse finale** — durable, ~1–2s, replays proven values (runs offline). Send it **twice**, once per scenario:

Healthy path:
```json
{ "scenario": "baseline" }
```
→ steps `evaluate_current` (F1 0.94) → `detect_drop` (above the 0.80 threshold) → `no_action` ("healthy — swarm exploited, no evolution needed").

Adversary-adapted path (the full story):
```json
{ "scenario": "adversary_adapted" }
```
→ `evaluate_current` (F1 0.64) → `detect_drop` (below threshold → dropped) → `explore` (discovers the shared-consignee signal) → `evolve_and_select` (coverage-aware selection + merge adds a consignee lens, new F1 1.0) → `confirm_recovery` (0.64 → 1.0).

### What to expect in the Runs view

Open the **Runs** tab and click the run. You'll see the **step trace** — each `step.run` (`evaluate_current`, `detect_drop`, and then either `no_action`, or `explore` → `evolve_and_select` → `confirm_recovery`) as a separate, durably-checkpointed step with its own output, plus the function's final summary dict.

---

## View the web demo

It's a single self-contained file — just open it:

```powershell
Start-Process web\index.html
```

(Or serve the folder, e.g. `npx serve web`, if your browser restricts `file://` features.) The page is also deployed as a **static site on Vercel** (project root directory set to `web/` — no server, no API key, no build step).

---

## Troubleshooting

- **`npx` / `node` not found** → install **Node.js** from <https://nodejs.org>, then open a fresh PowerShell window so `npx` is on `PATH`.
- **Port already in use** → if `:8000` is taken, change `PORT` near the top of `inngest_app.py` (and update the `-u` URL in the Terminal 2 command to match). If the dashboard's `:8288` is taken, stop the other process using it.
- **`OPENAI_API_KEY is not set`** → the swarms and `autoswarm/evaluate.requested` need the key in `.env`. Confirm `.env` exists at the repo root with `OPENAI_API_KEY=...`. (The `autoswarm/data.arrived` finale does **not** need a key.)
- **Functions don't appear in the dashboard** → the Flask app (Terminal 1) must be running **before** you start the dev server, and reachable at `http://127.0.0.1:8000/api/inngest`. Restart Terminal 2 after Terminal 1 is up.

---

*Synthetic data throughout. AutoSwarm surfaces leads for human review — it does not make determinations about real people or entities.*
