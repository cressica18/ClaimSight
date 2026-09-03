# ClaimSight

**AI-Assisted Vehicle Insurance Claims Investigation Platform (Decision-Support Prototype)**

> ClaimSight is a **decision-support** prototype. It never adjudicates claims
> autonomously — a human claims officer always makes the final decision. All
> outputs are recommendations with attached evidence.

---

## Overview

Vehicle insurance claim handlers spend a large share of their day cross-checking
documents, photos, repair estimates, and historical claim records. ClaimSight
attempts to shorten that loop by:

1. **Reading** claim photos with a small fine-tuned vision model.
2. **Extracting** the structured fields of each uploaded document (claim form,
   repair estimate, prior-claim history).
3. **Cross-checking** the claim against nine deterministic consistency rules
   (R1–R9) and a frozen risk-scoring model.
4. **Summarising** the findings with an LLM (Gemini 2.5 Flash) that is strictly
   forbidden from inventing numbers or making the final call.
5. **Surfacing** the evidence, the rule firings, the risk band, and the LLM
   summary on a single investigation page, so a human officer can decide.

The system is **end-to-end runnable** on a laptop with PostgreSQL. A demo
mode (env-var toggles) lets the reviewer see the full pipeline without the
trained CV checkpoint or a live Gemini key.

---

## Features

- **Customer / policy / vehicle / claim CRUD** with FastAPI + SQLAlchemy 2.0.
- **Image upload + CV inference** (PyTorch, ResNet-50, dual-head: damage type +
  severity).
- **Document upload + stubbed DocIntel** that extracts a policy number from
  filenames and otherwise returns an empty (honest) field set with a
  `0.5` raw-confidence marker.
- **Deterministic Consistency Engine** with nine rules (R1–R9) covering
  unsupported damage, severity mismatches, repair-component mismatches,
  excessive cost, duplicate previous damage, policy-coverage mismatches, claim
  frequency, near-policy-boundary dates, and document-field conflicts.
- **Frozen Risk Engine** that produces a 0–10 explainable score and a
  Low / Medium / High band. Severity is bumped to `Medium` when a clean claim
  has no signals (low-data-confidence default).
- **Gemini Investigation Layer** (optional, mockable): a strict prompt that
  *narrates* the deterministic findings in 3–6 sentences, refuses to recompute
  scores, and overwrites any hallucinated recommendation with the
  deterministic value.
- **Nine-screen React UI**: Dashboard, Claims List, New Claim, Claim Analysis,
  Image Analysis, Document Viewer, Risk Signals, Investigation Summary,
  Decision Panel.
- **Demo mode** (`USE_DEMO_CV=1`, `USE_DEMO_GEMINI=1`) for offline review and
  CI.
- **Demo data generator** that seeds the five Section 3.3 scenarios
  (legitimate, inflated estimate, image/document mismatch, previous-claim
  overlap, multi-signal suspicious).

---

## Architecture

```
┌──────────────────┐     HTTP     ┌────────────────────────────────┐
│  React frontend  │ ───────────▶ │  FastAPI backend (uvicorn)     │
│  Vite + TS       │ ◀─────────── │  /api/uploads served statically│
│  9 screens       │              └────────────────────────────────┘
└──────────────────┘                       │
                                           │ SQLAlchemy 2.0
                                           ▼
                                    ┌──────────────────┐
                                    │  PostgreSQL 15+  │
                                    └──────────────────┘
                                           ▲
                                           │
        ┌──────────────────────────────────┴────────────────────────────┐
        │                                                               │
        │            Pipeline (per /claims/{id}/analyze run)            │
        │  ┌────────────┐  ┌──────────────┐  ┌────────────────────────┐ │
        │  │ CV module  │→ │ Document     │→ │ Consistency Engine     │ │
        │  │ ResNet-50  │  │ Intelligence │  │ R1…R9 deterministic    │ │
        │  │ (PyTorch)  │  │ (PyMuPDF +   │  │ rule firings           │ │
        │  │            │  │  stub)       │  │                        │ │
        │  └────────────┘  └──────────────┘  └────────────────────────┘ │
        │                                              │                  │
        │                                              ▼                  │
        │                                  ┌────────────────────────┐    │
        │                                  │  Risk Engine           │    │
        │                                  │  (frozen 5-feature     │    │
        │                                  │   weighted scoring)    │    │
        │                                  └────────────────────────┘    │
        │                                              │                  │
        │                                              ▼                  │
        │                                  ┌────────────────────────┐    │
        │                                  │  Gemini Investigation  │    │
        │                                  │  Layer (LLM narration, │    │
        │                                  │  no score computation) │    │
        │                                  └────────────────────────┘    │
        └────────────────────────────────────────────────────────────────┘
```

---

## Tech stack

| Layer        | Choice                                            |
| ------------ | ------------------------------------------------- |
| Frontend     | React 18, TypeScript 5.6, Vite 6, React Router 6  |
| Backend API  | FastAPI, Pydantic v2, pydantic-settings           |
| ORM          | SQLAlchemy 2.0                                    |
| Migrations   | Alembic                                           |
| Database     | PostgreSQL 15+                                    |
| CV model     | PyTorch, ResNet-50 (ImageNet pretrained, fine-tuned) |
| DocIntel     | PyMuPDF + deterministic filename-token stub       |
| Consistency  | Pure Python rule engine                           |
| Risk scoring | scikit-learn Isolation Forest (baseline fit) + frozen deterministic scoring |
| LLM layer    | Gemini 2.5 Flash via `httpx`                      |
| Tests        | pytest, pytest-asyncio, FastAPI `TestClient`      |

---

## Project structure

```
claimsight/
├── README.md                     # this file
├── DEVLOG.md                     # what we built, what we'd improve
├── claimsight_implementation.md  # long-form technical blueprint
├── pytest.ini
│
├── backend/
│   ├── alembic.ini
│   ├── alembic/                  # DB migrations (5 revisions)
│   ├── app/
│   │   ├── api/                  # routers: health, customers, policies,
│   │   │                         #   vehicles, claims, documents, images,
│   │   │                         #   cv, pipeline
│   │   ├── core/                 # config (pydantic-settings)
│   │   ├── db/                   # SQLAlchemy session + Base
│   │   ├── models/               # ORM models
│   │   ├── schemas/              # Pydantic v2 request/response models
│   │   └── services/             # CV, document intelligence, consistency,
│   │                             #   risk engine, Gemini client, pipeline
│   ├── requirements.txt
│   └── requirements-dev.txt
│
├── frontend/
│   ├── index.html
│   ├── package.json
│   ├── tsconfig.json
│   ├── vite.config.ts
│   └── src/
│       ├── api/                  # typed fetch client
│       ├── components/           # shared UI: Layout, PageShell,
│       │                         #   StatusPill, RiskBandPill, etc.
│       └── pages/                # 9 screens
│
├── ml/
│   ├── data_card.md
│   ├── inference/                # predictor + verification script
│   ├── training/                 # train.py, model.py, dataset.py, evaluate.py
│   ├── data/                     # processed CSVs and (gitignored) raw/
│   ├── results/                  # training_history.json
│   └── weights/                  # (gitignored) trained checkpoint
│
├── scripts/
│   └── generate_demo_data.py     # deterministic seed for 5 demo scenarios
│
├── tests/
│   ├── conftest.py
│   ├── backend/                  # ~260 backend tests
│   └── ml/                       # ~21 CV tests
│
└── data/
    ├── synthetic/.gitkeep
    ├── processed/.gitkeep
    └── uploads/.gitkeep          # runtime uploads land here
```

---

## Prerequisites

| Tool       | Version           |
| ---------- | ----------------- |
| Python     | 3.11+ (tested on 3.14) |
| Node.js    | 18+ (tested on 20) |
| npm        | 9+                |
| PostgreSQL | 15+               |

For the optional ML training path, a CUDA-capable PyTorch wheel is needed.
Inference on CPU works.

---

## Environment variables

All environment variables live in `backend/.env` (template:
`backend/.env.example`). **Never commit a real `.env`.**

| Name                       | Purpose                                                              | Default                                                            |
| -------------------------- | -------------------------------------------------------------------- | ------------------------------------------------------------------ |
| `APP_ENV`                  | `development` / `production`                                         | `development`                                                      |
| `APP_HOST`                 | uvicorn bind host                                                    | `0.0.0.0`                                                          |
| `APP_PORT`                 | uvicorn bind port                                                    | `8000`                                                             |
| `DATABASE_URL`             | SQLAlchemy connection string                                         | `postgresql+psycopg2://claimsight:claimsight@localhost:5432/claimsight_db` |
| `CORS_ORIGINS`             | Comma-separated list of allowed frontend origins                     | `http://localhost:5173,http://localhost:3000`                      |
| `UPLOAD_DIR`               | Where uploaded claim files are written                               | `../data/uploads` (relative to `backend/`)                         |
| `GEMINI_API_KEY`           | Gemini 2.5 Flash API key (only required when `USE_DEMO_GEMINI=0`)    | unset                                                              |
| `GEMINI_MODEL`             | Gemini model name                                                    | `gemini-2.5-flash`                                                 |
| `GEMINI_BASE_URL`          | Gemini API base URL                                                  | `https://generativelanguage.googleapis.com`                        |
| `GEMINI_TIMEOUT_SECONDS`   | Per-request timeout                                                  | `15.0`                                                             |
| `GEMINI_RETRY_BACKOFF_SECONDS` | Wait between retry attempts                                      | `2.0`                                                              |
| `USE_DEMO_CV`              | Use deterministic `_DemoCVPredictor` instead of the trained model    | `false`                                                            |
| `USE_DEMO_GEMINI`          | Use canned Gemini response (no network call)                         | `false`                                                            |

### Gemini setup

1. Generate an API key at
   <https://aistudio.google.com/apikey>.
2. Set `GEMINI_API_KEY=<your-key>` in `backend/.env`.
3. Leave `USE_DEMO_GEMINI=0` (or unset) to use the real model.

If you don't have a key, set `USE_DEMO_GEMINI=1` and the system will
return a deterministic canned response so the rest of the pipeline can
still be exercised offline.

---

## Installation

```bash
# 1. Clone
git clone https://github.com/cressica18/ClaimSight
cd claimsight

# 2. Backend
cd backend
python3 -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r requirements.txt -r requirements-dev.txt
cp .env.example .env                # then edit with your values
cd ..

# 3. Frontend
cd frontend
npm install
cd ..

# 4. ML (only if you want to run the local predictor directly)
# The backend uses the `ml` package via sys.path; no separate install
# is required for inference or tests.
```

---

## Database setup

```bash
# Make sure PostgreSQL is running and you can connect as a superuser
createdb claimsight_db
createuser -P claimsight             # set a password when prompted

# Apply the migrations
cd backend
source .venv/bin/activate
alembic upgrade head
```

`backend/.env` should reference the same user/database/password you
just created, e.g.:

```
DATABASE_URL=postgresql+psycopg2://claimsight:<your-password>@localhost:5432/claimsight_db
```

---

## Running the app

### Backend

```bash
cd backend
source .venv/bin/activate
uvicorn app.main:app --reload --port 8000
```

The API is served at <http://localhost:8000>.
- OpenAPI docs: <http://localhost:8000/docs>
- Health check: `GET /health` → `{"status":"ok"}`
- Mode check:   `GET /mode`   → `{app_env, demo_mode, use_demo_cv, use_demo_gemini}`

### Frontend

```bash
cd frontend
npm run dev
```

The dev server runs at <http://localhost:5173>.

### Demo data

```bash
cd backend
source .venv/bin/activate
python3 ../scripts/generate_demo_data.py --reset
```

This seeds five demo claims (CLM-DEMO-S1-LEGIT through
CLM-DEMO-S5-MULTI) with full customer / vehicle / policy / accident /
images / documents / repair-estimate / previous-claim graphs. Run with
`--analyze` to also kick off the analysis pipeline for each.

---

## Tests

```bash
# From the repo root
source backend/.venv/bin/activate
python3 -m pytest tests/backend tests/ml
```

This is the full suite we run before every release. As of the last
clean run it is **281 passed, 1 known failure**.

### Known failure: `test_scenario_1_legitimate_claim_low`

The blueprint expects a legitimate claim to score `Low`. The frozen
risk engine bumps a clean claim to `Medium` via the
*low-data-confidence* default. The Phase 13 progress doc
explains this is a calibration question, not a pipeline defect,
and is intentionally left as-is.

### Frontend type-check and build

```bash
cd frontend
npx tsc --noEmit
npm run build
```

Both should run clean. The build emits `frontend/dist/`.

---

## Basic demo workflow

1. **Backend on, database migrated, demo data seeded.**
   ```bash
   # terminal 1
   cd backend && source .venv/bin/activate && uvicorn app.main:app --reload --port 8000
   # terminal 2
   cd frontend && npm run dev
   # terminal 3
   cd backend && source .venv/bin/activate && python3 ../scripts/generate_demo_data.py --reset
   ```
2. **Open the app at <http://localhost:5173>.** The dashboard shows
   the five seeded claims; the sidebar shows a "Demo data" badge if
   the backend is in demo mode.
3. **Click into CLM-DEMO-S1-LEGIT.** The Claim Analysis page shows
   the pipeline stages. Click *Start analysis* (or hit
   `POST /claims/40/analyze` via `curl`); the page polls until the
   analysis completes.
4. **Walk the four evidence screens** — Image Analysis, Document
   Viewer, Risk Signals, Investigation Summary — then open the
   Decision Panel to record the final human decision.
5. **Repeat for the four riskier scenarios** to see the consistency
   rules fire (R1–R9) and the risk band climb.

---

## Demo mode

To run the entire stack on a laptop with no CV checkpoint and no
Gemini key, set in `backend/.env`:

```
USE_DEMO_CV=1
USE_DEMO_GEMINI=1
```

The CV service will return deterministic predictions derived from
the image filename (e.g. `small-dent.jpg` → `dent` / `minor`), and
the Gemini client will return a canned investigation summary built
from the deterministic risk signals. The pipeline, evidence
rendering, and decision flow are otherwise unchanged.

`GET /mode` exposes the active flags so the frontend can show a
"Demo data" badge.

---

## Known limitations / future improvements

These are honestly held limitations as of the current state of the
codebase. They are **not** addressed in this drop and are documented
as future work.

- **Document Intelligence is a stub.** It extracts a `POL-XXXX` token
  from the filename of `policy` documents and otherwise writes an
  honest empty `extracted_fields` set with `raw_confidence=0.5`.
  Real OCR / DocIntel is a future task.
- **Single-process concurrency.** Pipeline runs execute in a thread
  inside the same Python process as the API. Multi-process
  deployments (gunicorn workers) would need `SELECT … FOR UPDATE` or
  an external queue.
- **Synchronous-with-202 pattern.** If the process dies mid-run, an
  Analysis row stays `running` until the next request triggers the
  `_init_state` guard. A startup sweeper that flips long-running rows
  to `failed (interrupted)` is a future task.
- **Demo-mode CV is a small helper, not production code.** The
  filename-based predictor sits behind an env-var toggle and is
  intended for the demo laptop only.
- **No live Gemini verification in CI.** All Gemini tests use mocks
  or the demo stub; live integration was not run as part of the
  automated test suite.
- **Schema-drift risk.** A model change that ships without a
  matching alembic revision will be silent in tests (which use
  `Base.metadata.create_all`) and broken in production. A CI check
  that diffs the model against the head migration is a future
  improvement.
- **No Docker / production deployment configuration.** The repo is
  runnable as documented above; containerisation and a reverse-proxy
  setup are intentionally out of scope.

---

## License

Internal prototype — no license declared. Add a `LICENSE` file
before any external distribution.
