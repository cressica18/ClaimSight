# ClaimSight

**AI-Assisted Vehicle Insurance Claims Investigation Platform — Decision-Support Prototype**

> ClaimSight never adjudicates claims on its own. A human claims officer always makes the final decision. Every AI output is a recommendation backed by traceable evidence.

---

## What is ClaimSight?

Vehicle insurance claims handling is a manual, evidence-heavy job. A claims officer typically reads a claim form, examines accident photos, cross-checks a repair estimate against what the photos show, reviews the customer's prior claim history, and then decides whether to approve, flag, or investigate. That process is time-consuming and easy to get wrong in either direction — a missed red flag costs money; an unfair flag costs customer trust.

ClaimSight aims to shorten that loop by doing the cross-checking automatically and surfacing the results clearly, so a human officer can make a faster and better-informed decision. It never replaces that decision.

The full pipeline:

1. **Images** — accident photos are analysed by a fine-tuned ResNet-50 computer-vision model that predicts damage type (8 classes) and severity (minor / moderate / severe).
2. **Documents** — uploaded claim forms, repair estimates, and policy documents pass through a limited deterministic extraction layer. Currently this reads a policy-number token from filenames; full OCR-based field extraction is a future task.
3. **Consistency rules (R1–R9)** — nine deterministic rules cross-check all the evidence: do the photos match the claimed damage? Does the repair estimate cost match the baseline for this type of damage? Has the same vehicle had the same damage repaired before? And so on.
4. **Risk score** — a five-feature, fixed-weight scoring formula (no machine learning) turns the rule firings into a 0–100 score and a Low / Medium / High band.
5. **Gemini narrative** — Gemini 2.5 Flash writes a 3–6 sentence investigation summary, citing only the rule firings and evidence already computed. The recommendation (approve / review / investigate) is computed deterministically from the risk band; Gemini's value is overwritten.
6. **Human decision** — an officer reviews all the evidence on a single page and records one of four verdicts: Approve, Manual review, Investigate, or Deny.

---

## Key features

- **CRUD for the full claims entity graph** — customers, vehicles, policies, claims, accidents, images, documents, repair estimates, and previous claims.
- **Image upload + CV inference** — PyTorch ResNet-50 with dual heads (damage type + severity). A demo predictor (filename-based, no checkpoint required) is available for offline review.
- **Document upload + extraction** — PyMuPDF is included. The current extraction layer is a deterministic stub: it reads a `POL-XXXX` token from the filename of policy documents and otherwise returns an honest empty field set. Real OCR is a documented future task.
- **Nine deterministic consistency rules (R1–R9)** — pure Python, unit-testable, no LLM calls.
- **Frozen risk engine** — five named, fixed-weight features, 0–100 score, Low / Medium / High bands. Fully explainable; each contributing factor is labelled with the underlying risk signals and claim data that drove it.
- **Gemini investigation layer** — optional, mockable. Strict prompt that forbids inventing numbers or making the final call. Fails gracefully (summary set to null; pipeline still completes).
- **Nine-screen React UI** — Dashboard, Claims List, New Claim, Claim Analysis, Image Analysis, Document Viewer, Risk Signals, Investigation Summary, Decision Panel.
- **Demo mode** — two env-var toggles replace the CV model and Gemini API with deterministic stubs so the full pipeline can run offline.
- **Demo data generator** — seeds five repeatable scenario claims (legitimate, inflated estimate, image/document mismatch, previous-claim overlap, multi-signal suspicious).

---

## Architecture

```
┌──────────────────┐     HTTP/JSON    ┌────────────────────────────────┐
│  React frontend  │ ──────────────▶ │  FastAPI backend (uvicorn)     │
│  Vite + TS       │ ◀────────────── │  /api/* routers                │
│  9 screens       │                  │  /api/uploads  (static files)  │
└──────────────────┘                  └────────────────────────────────┘
                                                    │
                                                    │ SQLAlchemy 2.0
                                                    ▼
                                           ┌──────────────────┐
                                           │  PostgreSQL 15+  │
                                           └──────────────────┘
                                                    ▲
                           ┌────────────────────────┴──────────────────────┐
                           │         Pipeline  (POST /claims/{id}/analyze)  │
                           │  CV (ResNet-50)                                │
                           │    → Document Intelligence (stub + PyMuPDF)    │
                           │    → Consistency Engine (R1–R9)                │
                           │    → Risk Engine (5-feature weighted scoring)  │
                           │    → Gemini Investigation (LLM narration only) │
                           └───────────────────────────────────────────────┘
```

The pipeline runs in a background thread in the same Python process. `POST /claims/{id}/analyze` returns `202 Accepted` immediately with an `analysis_id`, and the frontend polls `GET /claims/{id}/analysis/{analysis_id}` until the status is `completed` or `failed`.

---

## Tech stack

| Layer          | Choice                                                         |
| -------------- | -------------------------------------------------------------- |
| Frontend       | React 18, TypeScript 5.6, Vite 6, React Router 6              |
| Backend API    | FastAPI, Pydantic v2, pydantic-settings                        |
| ORM            | SQLAlchemy 2.0                                                 |
| Migrations     | Alembic (5 revisions)                                          |
| Database       | PostgreSQL 15+                                                 |
| CV model       | PyTorch, ResNet-50 (ImageNet pretrained, dual-head fine-tuned) |
| DocIntel       | PyMuPDF + deterministic filename-token stub                    |
| Consistency    | Pure Python rule engine (no external dependencies)             |
| Risk scoring   | Deterministic 5-feature weighted formula (no sklearn at runtime)|
| LLM layer      | Gemini 2.5 Flash via `httpx`                                   |
| Tests          | pytest, FastAPI `TestClient`, SQLite in-memory                 |

---

## Project structure

```
claimsight/
├── README.md
├── DEVLOG.md
├── pytest.ini
│
├── backend/
│   ├── alembic.ini
│   ├── alembic/versions/          # 5 DB migration revisions
│   ├── app/
│   │   ├── api/                   # routers: health, customers, policies,
│   │   │                          #   vehicles, claims, documents, images,
│   │   │                          #   cv, pipeline
│   │   ├── core/                  # config (pydantic-settings)
│   │   ├── db/                    # SQLAlchemy session + Base
│   │   ├── models/                # ORM models
│   │   ├── schemas/               # Pydantic v2 request/response models
│   │   └── services/              # CV, document intelligence, consistency,
│   │                              #   risk engine, Gemini client, pipeline
│   ├── requirements.txt
│   └── requirements-dev.txt
│
├── frontend/
│   ├── index.html
│   ├── package.json
│   ├── vite.config.ts
│   └── src/
│       ├── api/                   # typed fetch client (no extra libs)
│       ├── components/            # PageShell, StatusPill, RiskBandPill, etc.
│       └── pages/                 # 9 screens
│
├── ml/
│   ├── inference/                 # predictor.py + demo predictor
│   ├── training/                  # train.py, model.py, dataset.py, config.py
│   ├── data/processed/            # train/val/test CSVs (images gitignored)
│   ├── results/training_history.json
│   └── weights/                   # trained checkpoint (gitignored)
│
├── scripts/
│   └── generate_demo_data.py      # deterministic seed for 5 demo scenarios
│
├── tests/
│   ├── backend/                   # backend test suite (~16 test files)
│   └── ml/                        # CV model tests
│
└── data/uploads/                  # runtime file uploads (gitignored)
```

---

## Prerequisites

| Tool       | Version                     |
| ---------- | --------------------------- |
| Python     | 3.11+ (tested on 3.14)      |
| Node.js    | 18+                         |
| npm        | 9+                          |
| PostgreSQL | 15+                         |

For the optional ML training path, a PyTorch wheel is needed (CPU inference works; CUDA speeds training significantly).

---

## Environment variables

All variables live in `backend/.env`. **Never commit a `.env` file with real secrets.**

| Name                           | Purpose                                                               | Default                                                                      |
| ------------------------------ | --------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| `APP_ENV`                      | `development` / `production`                                          | `development`                                                                |
| `APP_HOST`                     | uvicorn bind host                                                     | `0.0.0.0`                                                                    |
| `APP_PORT`                     | uvicorn bind port                                                     | `8000`                                                                       |
| `DATABASE_URL`                 | SQLAlchemy connection string                                          | `postgresql+psycopg2://claimsight:claimsight@localhost:5432/claimsight_db`   |
| `CORS_ORIGINS`                 | Comma-separated list of allowed frontend origins                      | `http://localhost:5173,http://localhost:3000`                                 |
| `UPLOAD_DIR`                   | Where uploaded claim files are written                                | `../data/uploads` (relative to `backend/`)                                   |
| `GEMINI_API_KEY`               | Gemini 2.5 Flash API key (not required when `USE_DEMO_GEMINI=true`)   | unset                                                                        |
| `GEMINI_MODEL`                 | Gemini model name                                                     | `gemini-2.5-flash`                                                           |
| `GEMINI_BASE_URL`              | Gemini API base URL                                                   | `https://generativelanguage.googleapis.com`                                  |
| `GEMINI_TIMEOUT_SECONDS`       | Per-request timeout                                                   | `15.0`                                                                       |
| `GEMINI_RETRY_BACKOFF_SECONDS` | Wait between retry attempts                                           | `2.0`                                                                        |
| `USE_DEMO_CV`                  | Use filename-based stub instead of the trained model                  | `false`                                                                      |
| `USE_DEMO_GEMINI`              | Use a canned Gemini response (no network call)                        | `false`                                                                      |

### Getting a Gemini API key

1. Visit <https://aistudio.google.com/apikey> and generate a key.
2. Add `GEMINI_API_KEY=<your-key>` to `backend/.env`.
3. Leave `USE_DEMO_GEMINI` unset (defaults to `false`).

If you don't have a key, set `USE_DEMO_GEMINI=true` — the pipeline will complete with a deterministic canned summary.

---

## Installation

```bash
# 1. Clone
git clone https://github.com/cressica18/ClaimSight.git
cd ClaimSight

# 2. Backend
cd backend
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt -r requirements-dev.txt
cp .env.example .env             # edit the values inside
cd ..

# 3. Frontend
cd frontend
npm install
cd ..
```

---

## Database setup

```bash
# Make sure PostgreSQL is running, then run as a superuser (e.g. postgres)
psql -U postgres -c "CREATE USER claimsight WITH PASSWORD 'claimsight';"
psql -U postgres -c "CREATE DATABASE claimsight_db OWNER claimsight;"
psql -U postgres -c "GRANT ALL PRIVILEGES ON DATABASE claimsight_db TO claimsight;"

# Apply migrations
cd backend
source .venv/bin/activate
alembic upgrade head
```

Set the matching connection string in `backend/.env`:

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

- API: <http://localhost:8000>
- OpenAPI docs: <http://localhost:8000/docs>
- Health check: `GET /health` → `{"status":"ok"}`
- Mode flags: `GET /mode` → `{app_env, demo_mode, use_demo_cv, use_demo_gemini}`

### Frontend

```bash
cd frontend
npm run dev
```

Dev server at <http://localhost:5173>.

### Seed demo data

```bash
cd backend
source .venv/bin/activate
python3 ../scripts/generate_demo_data.py --reset
# or to also run the analysis pipeline on each demo claim:
python3 ../scripts/generate_demo_data.py --reset --analyze
```

This seeds five scenario claims (`CLM-DEMO-S1-LEGIT` through `CLM-DEMO-S5-MULTI`) with a complete entity graph for each. The script is deterministic and idempotent — `--reset` wipes existing demo data before reseeding.

---

## Demo workflow

1. **Start everything** (three terminals):
   ```bash
   # terminal 1 — backend
   cd backend && source .venv/bin/activate && uvicorn app.main:app --reload --port 8000

   # terminal 2 — frontend
   cd frontend && npm run dev

   # terminal 3 — seed
   cd backend && source .venv/bin/activate && python3 ../scripts/generate_demo_data.py --reset
   ```

2. **Open <http://localhost:5173>**. The dashboard shows the five seeded claims. A "Demo data" badge appears in the sidebar when the backend is in demo mode (either `USE_DEMO_CV` or `USE_DEMO_GEMINI` is set).

3. **Click into any demo claim**. On the Claim Analysis page, click **Start analysis**. The page polls the pipeline in real time and updates each stage as it completes.

4. **Walk the evidence screens** — Image Analysis, Document Viewer, Risk Signals, Investigation Summary — then open the Decision Panel to record the human decision.

5. **Try all five scenarios** to see different rule combinations fire:
   - S1-LEGIT: legitimate claim, no signals
   - S2-INFLATED: repair estimate well above the cost baseline (R4 fires)
   - S3-MISMATCH: claimed damage area does not match any CV-detected damage (R1 fires)
   - S4-PREV: same vehicle region already claimed within 6 months (R5 fires)
   - S5-MULTI: multiple signals in combination

---

## Demo mode (offline)

To run with no CV checkpoint and no Gemini key, add to `backend/.env`:

```
USE_DEMO_CV=true
USE_DEMO_GEMINI=true
```

- **Demo CV**: predictions are derived from the uploaded image's filename (e.g. `small-dent.jpg` → `dent` / `minor`). The consistency rules, risk scoring, and evidence generation all run unchanged.
- **Demo Gemini**: a canned investigation summary is built from the actual deterministic risk signals. It passes through the same validator that a real Gemini response would.

`GET /mode` exposes which flags are active so the frontend can surface the demo badge.

---

## Tests

```bash
# From the repo root
source backend/.venv/bin/activate
python3 -m pytest tests/backend tests/ml -q
```

Current result: **281 passed, 1 known failure**.

### Known failure: `test_scenario_1_legitimate_claim_low`

The test asserts that a clean, legitimate claim produces a `Low` risk band. The risk engine's low-data-confidence default bumps a claim with zero signals to `Medium` rather than leaving it at `Low`. This is a calibration issue (the weight or threshold for the default needs tuning), not a pipeline defect. It is intentionally left as-is rather than hidden.

### Frontend

```bash
cd frontend
npx tsc --noEmit   # type check only
npm run build      # full production build
```

Both run clean. There is no JS unit test framework in the repository; the real logic lives in the backend services.

---

## Known limitations

These are honest, known limitations as of the current codebase. None are hidden or worked around.

- **Document intelligence is a stub.** The extraction layer reads a `POL-XXXX` token from the filename of policy documents and otherwise returns an empty field set. PyMuPDF is included but no real OCR or document-intelligence provider is wired in. Rules that depend on extracted fields (especially R9 — document field conflicts) have limited signal in the current implementation.
- **Single-process concurrency.** The pipeline runs in a thread inside the same uvicorn process. A partial unique index (`uq_analyses_one_running_per_claim`) provides an additional guard against concurrent running analyses, but the overall design is single-process. Running multiple uvicorn workers would need a cross-process lock (e.g. `SELECT … FOR UPDATE` on the claim row) or an external queue.
- **No startup sweeper.** If the process dies mid-pipeline, the `Analysis` row stays in `running` state. A startup job that flips stale `running` rows to `failed` is a future task.
- **No authentication.** The prototype assumes a single trusted user on a local machine. There is no session management, per-user audit trail, or claim-level locking.
- **CV model performance is limited.** The model was trained on two small public Kaggle datasets with a masked-loss strategy (each dataset provides labels for only one head). Damage-type macro-F1 on the validation set reached ~0.19; severity accuracy reached ~0.70. These are honest numbers — the model is suitable for demo purposes but not production use.
- **No live Gemini calls in CI.** All Gemini tests use mocks or the demo stub.
- **No Docker / production configuration.** The repository is runnable as documented above. Containerisation is out of scope.

---

## License

Internal prototype — no license declared. Add a `LICENSE` file before any external distribution.
