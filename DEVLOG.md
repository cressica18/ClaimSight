# ClaimSight — Devlog

A retrospective on what we built, why we built it this way, where we
ran into trouble, and what we'd improve with more time.

---

## Why we chose ClaimSight

Vehicle insurance claim handling is a high-volume, evidence-heavy
process where the cost of a missed red flag is paid in fraudulent
payouts, and the cost of a false flag is paid in customer trust. The
job today is largely manual: a handler reads a claim form, looks at
photos, cross-checks a repair estimate against a parts catalogue,
checks whether the customer has a recent claim for the same damage,
and only then makes a call.

That workflow has three things going for it as an AI-assist target:

1. **Structured inputs already exist.** Insurance claim forms, repair
   estimates, prior-claim tables, vehicle IDs, and policy records are
   all structured or semi-structured data. That makes it possible to
   build a deterministic, rule-based layer on top of which an LLM
   can narrate.
2. **The "adjudication" decision is auditable.** A claims officer
   already has to justify an approval or a denial to internal
   compliance. A system that ships *evidence* (rule firings, the
   photos, the document fields) instead of a verdict slots in
   naturally.
3. **The cost of an LLM getting it wrong is bounded by the human
   gate.** Because a human has to click *Approve* or *Investigate*
   anyway, the LLM can be optimistically helpful (write a 4-sentence
   summary, list the concerns) without ever being allowed to push
   the final button.

We chose to scope the prototype to vehicle claims specifically (not
home, life, health) because (a) the data is visually rich and
publicly available, (b) the entity graph (customer → vehicle →
policy → accident → images → documents → estimate) is rich enough to
exercise the consistency engine, and (c) the fraud surface is
well-documented in the public research literature.

---

## What we implemented

We built the system in 13 numbered phases over a single working
stretch, each phase ending in a green test run. The deliverables
below are the *integrated* end product; per-phase logs are not
included in this repository.

### Backend (FastAPI + SQLAlchemy 2.0 + PostgreSQL)

- **API surface.** `customers`, `policies`, `vehicles`, `claims`,
  `documents`, `images`, `cv`, `pipeline` routers mounted under
  `/customers`, `/policies`, `/vehicles`, `/claims`. `GET /health`
  for liveness; `GET /mode` for the demo-mode flag surface.
- **Data model.** Customers, vehicles, policies, claims, accidents,
  images, documents, repair estimates, previous claims, evidence,
  risk signals, investigations, decisions, and a `claim_id` →
  `analysis_id` pipeline state machine. All migrations live in
  `backend/alembic/versions/`.
- **Static file serving.** `app.mount("/api/uploads", StaticFiles(...))`
  so the frontend can show uploaded images directly.
- **Test rig.** `pytest` + FastAPI `TestClient`. SQLite in-memory for
  most tests; PostgreSQL probed by a single opt-in test that is
  skipped gracefully when PG is unavailable.

### Computer vision module (`ml/`)

- ResNet-50 backbone, ImageNet-pretrained, dual-head fine-tune: head
  A predicts damage type (8 classes), head B predicts severity
  (minor / moderate / severe).
- Trained on two public Kaggle datasets
  (car-damage-assessment, car-damage-severity-dataset) for
  ~30 epochs across two stages (frozen backbone, then unfrozen).
- Inference path lives in `ml/inference/predictor.py`; the
  `_DemoCVPredictor` (filename-based) is the demo-mode fallback.
- ML test suite (`tests/ml/`, 21 tests) covers happy path, corrupted
  image, missing file, tiny image, missing checkpoint, batch predict,
  and API integration.

### Document intelligence

- A deterministic, *honest* stub in
  `app/services/document_intelligence.py`. For `policy` documents
  with a `POL-XXXX` token in the filename it extracts the policy
  number; for everything else it writes an empty
  `extracted_fields` set and `raw_confidence=0.5`. It never claims
  to have read a file it didn't read.
- PyMuPDF is included for future real extraction, but no real OCR /
  DocIntel provider is wired in (see limitations below).

### Consistency engine (deterministic, rule-based)

Nine rules (R1–R9) implemented in pure Python:

- **R1 unsupported_damage** — damage not visible in any photo.
- **R2 severity_mismatch** — claim severity vs. CV-predicted severity.
- **R3 repair_component_mismatch** — estimate components vs. damaged
  parts.
- **R4 excessive_repair_cost** — estimate vs. baseline range.
- **R5 duplicate_previous_damage** — same damage already on a prior
  claim.
- **R6 policy_coverage_mismatch** — claim type outside policy.
- **R7 claim_frequency** — too many claims in a window.
- **R8 near_policy_boundary** — incident date suspiciously close to
  policy start or renewal.
- **R9 document_field_conflict** — extracted fields disagree across
  documents.

Each rule is small, has a fixed `rule_id`, and returns a structured
`RiskSignal`-shaped payload (severity, weight, region_ref, evidence).

### Risk engine (frozen)

- 5-feature weighted scoring: signal severity, count of high-band
  signals, evidence quality, low-data-confidence default, and
  policy/customer history.
- Bands: `Low` (<2.0), `Medium` (2.0–4.5), `High` (≥4.5).
- Low-data-confidence default: a claim with zero signals and zero CV
  confidence is bumped to `Medium` — *not* left at `Low`. This is
  what causes the known test failure documented in the README.
- The Isolation Forest was fit at training time on synthetic
  embeddings and is used as a *baseline*, not a primary signal.
  Primary scoring is the deterministic weighted sum.

### Gemini / LLM investigation layer

- `gemini-2.5-flash` via `httpx` (no SDK lock-in).
- Strict prompt: 3–6 sentence summary citing only rule_ids that
  exist in the input. Gemini's `recommendation` is **always
  overwritten** with the deterministic value (`Low → normal`,
  `Medium → manual_review`, `High → investigate`). The disclaimer is
  a fixed constant.
- Banned-phrase filter (fraud accusations, definitive verdicts)
  causes a single repair attempt; second failure returns `None` and
  the caller persists with `summary_text=None`.
- One retry on timeout / 5xx with `GEMINI_RETRY_BACKOFF_SECONDS`
  between attempts. Second failure returns `None`.

### Pipeline orchestrator

- `POST /claims/{id}/analyze` returns `202` immediately and runs the
  pipeline in a thread inside the same Python process.
- A partial unique index
  (`uq_analyses_one_running_per_claim`) plus an in-process
  `pipeline_locks` dict prevent two concurrent analyses on the same
  claim.
- The orchestrator fans out to: CV → document intelligence →
  consistency rules → risk scoring → Gemini summary → persistence.

### Frontend (React 18 + TypeScript + Vite)

Nine screens, all functional:

1. **Dashboard** — three clickable stat cards driving filtered lists.
2. **Claims List** — filter by status, risk band, search, sort.
3. **New Claim** — start a new claim, attach images / documents.
4. **Claim Analysis** — pipeline progress + per-stage results.
5. **Image Analysis** — per-image CV damage + severity.
6. **Document Viewer** — extracted fields and raw document.
7. **Risk Signals** — R1–R9 firings with evidence, sorted by severity.
8. **Investigation Summary** — Gemini summary + concerns + disclaimer.
9. **Decision Panel** — human's approve / investigate / deny + notes.

A "Demo data" badge in the sidebar appears when the backend is in
demo mode (driven by `GET /mode`).

### Demo data generator

`scripts/generate_demo_data.py` is deterministic and idempotent.
It seeds the five Section 3.3 scenarios (legitimate, inflated
estimate, image/document mismatch, previous-claim overlap,
multi-signal suspicious), with a complete entity graph for each
and consistent claim numbers `CLM-DEMO-S1-LEGIT` through
`CLM-DEMO-S5-MULTI`. Supports `--reset`, `--analyze`, and `--seed`
flags.

### Tests

| Surface         | Test count (last run) |
| --------------- | --------------------: |
| `tests/backend` | 260 (after final pass) |
| `tests/ml`      | 21                   |
| **Total**       | **281 passed, 1 known failure** |
| Frontend tsc    | clean                |
| `npm run build` | clean                |

The one known failure is `test_scenario_1_legitimate_claim_low`,
which asserts a `Low` band on a clean claim; the frozen risk
engine's low-data-confidence default bumps it to `Medium`. The
discrepancy is documented and intentionally left as-is per the
"don't change risk scoring" rule.

---

## Major challenges and how we solved them

### 1. Keeping the LLM from inventing

LLMs are fluent liars. A naïve prompt that asks Gemini to "summarise
the claim" will gladly invent amounts, dates, and damage types. The
fix was multi-layered:

- A **strict input schema** (`InvestigationInput`) listing only the
  numbers, dates, and rule_ids that are allowed in the output.
- A **banned-phrase filter** in the validation layer — any response
  containing fraud accusations or definitive verdicts is rejected.
- **One repair attempt** with the specific validation error in the
  prompt; second failure returns `None` and the caller persists
  without a summary.
- **Hard override** of the recommendation, summary length, and
  disclaimer — Gemini's values are discarded.

This won't catch every hallucination, but it bounds the surface and
keeps the prototype useful for reviewer walkthroughs.

### 2. Stubbed document intelligence without lying

The DocIntel stub had a defect in an earlier iteration: it wrote a
`_phase11_stub: true` marker into the `extracted_fields` JSON, which
the UI then surfaced as if it were real. The fix was to make the
stub return an *honest* empty payload derived from the file's
metadata (filename + doc_type) and have the UI render a meaningful
"no structured fields extracted" state instead of the marker. The
stub's existence is documented at the API/UI level via the
"AI-generated" disclaimer on the investigation page.

### 3. Schema-drift between model and migrations

The first end-to-end demo failed because `claims.decision_notes`
existed on the SQLAlchemy model but had no migration. Tests passed
because they use `Base.metadata.create_all`; the production-shaped
database (alembic-managed) didn't have the column. The fix was to
add the missing migration and make the constraint-drop migration
idempotent for databases where the constraint was never created.
We have a future task to add a CI check that diffs the model
against the head migration.

### 4. Concurrency on the same claim

A second click on "Start analysis" while the first was still
running would have produced a mess. The fix was two-layered: an
in-process `pipeline_locks` dict for the same Python interpreter,
and a partial unique index on the analyses table
(`uq_analyses_one_running_per_claim`) for cross-process safety. A
multi-process production deployment will still need either
`SELECT ... FOR UPDATE` on the claim row or an external queue.

### 5. Frozen risk scoring + demo scenario calibration

The blueprint's expected demo bands (Section 3.3) are not all
reachable from the current rule severities and the frozen 5-feature
weights. In particular, the legitimate scenario (S1) is bumped to
`Medium` by the low-data-confidence default. We considered
recalibrating the weights but the user's hard rule was "do not
change risk scoring" — so we documented the mismatch instead and
left the test failure as a known-acceptable state. This is the
honest answer rather than a fudged one.

### 6. The frontend placeholder row

The Claim Analysis page originally derived its "image analysis
pending" state from the `signals` array, which is the wrong field
(it represents rule firings, not CV status). The fix was to read
the per-image `Damage` rows directly *and* have the pipeline
delete the placeholder `damage_type="pending"` row when CV runs
successfully. Both layers were needed — the UI fix alone would
still show the placeholder.

---

## Testing and reliability

- **Per-commit tests.** Every phase ended in a green `pytest`
  run. The suite grew from 39 tests at the end of Phase 3 to 281
  tests at the end of the final bug-fix pass.
- **One known failure, documented.** See the README — it is
  intentionally not fixed.
- **Test isolation.** SQLite in-memory per test, `StaticPool` for
  cross-connection persistence, FastAPI dependency override for the
  `get_db` route. PostgreSQL is exercised by a single opt-in test.
- **Frontend.** `npx tsc --noEmit` and `npm run build` are both
  clean. There is no JS test framework in the repo (the frontend is
  a thin view over the typed API client; the real logic lives in
  the backend services).
- **No live LLM calls in tests.** All Gemini tests mock the
  transport or use the demo stub. This is a known gap, not a
  hidden one.

---

## What we'd improve with more time

In rough priority order:

1. **Real DocIntel.** The stub gets us through the demo but the
   whole consistency engine leans on extracted fields. A
   per-document-type extraction pipeline (PDF + image) using a
   real provider would unlock the R9 conflicts meaningfully.
2. **CI check for model-vs-migration drift.** The
   `decision_notes` miss was silent in tests; a diff check would
   catch the next one.
3. **Frontend test framework.** A small Vitest + React Testing
   Library setup covering the filter logic, the stage tracker,
   and the demo-mode badge.
4. **External queue for pipeline runs.** Replace the in-process
   thread with an RQ / Celery worker so a long analysis doesn't
   pin a uvicorn worker.
5. **Auth.** The current prototype assumes a single trusted user.
   Adding real session management, per-user audit trails, and
   per-claim locking for the decision would be the next big
   ticket.
6. **Model retraining + data drift monitoring.** The current CV
   model is a one-shot ResNet-50 fine-tune. Production needs a
   retraining pipeline and a way to monitor the per-class F1
   over time.
7. **Live Gemini verification in CI.** A nightly integration test
   that exercises the real Gemini endpoint (with a hard budget
   cap) would catch drift in the prompt.
8. **Reachable demo bands.** Recalibrate the rule severities or
   the low-data-confidence default so the five Section 3.3
   scenarios actually produce their expected bands. This is a
   calibration question, not a code refactor.

---

## Files

- `README.md` — installation, run, demo, and limitations.
- `claimsight_implementation.md` — the long-form technical
  blueprint that drove the build.
- `DEVLOG.md` — this file.
