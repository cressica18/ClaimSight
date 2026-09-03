# ClaimSight — Devlog

A record of what was built, why decisions were made the way they were, what went wrong, and what remains incomplete.

---

## Motivation

Vehicle insurance claim handling is a process where both kinds of error are costly. Miss a fraudulent claim and the insurer pays out money it shouldn't. Flag a legitimate claim incorrectly and a customer who had a genuine accident waits weeks for a payment they are owed. The job today is largely manual — a handler reads a form, looks at photos, checks an estimate, reviews history — and the cognitive load scales with claim volume.

The goal of ClaimSight was to explore whether a small, transparent AI system could do the evidence-gathering and cross-checking mechanically, and surface the results clearly enough that a human officer could make a better and faster decision without having to trust a black box.

Three properties drove every design decision:

1. **Auditability over accuracy.** A system that can explain exactly why it flagged a claim is more useful than one that is slightly more accurate but opaque. Every rule firing has a `rule_id`, a `description`, and at least one linked evidence row.
2. **The human is the decision.** The code is architecturally prevented from making the final call. The `Decision` row in the database is written only by the officer clicking a button.
3. **Honest about limits.** Where the system cannot do something real (document OCR, production-grade CV), it returns an honest empty state rather than inventing a plausible-looking output.

---

## What was built

### Backend (FastAPI + SQLAlchemy 2.0 + PostgreSQL)

The backend was built incrementally. The data model covers the full claims entity graph: `Customer → Vehicle → Policy → Claim → Accident`, with `Damage` rows for image CV results, `Document` rows for uploaded files, `RepairEstimate` / `RepairItem` for cost data, `PreviousClaim` for history, and `RiskSignal` / `Evidence` / `Investigation` for the analysis output.

The API surface is conventional REST:

- `POST /claims` to create a claim
- `POST /claims/{id}/images` and `POST /claims/{id}/documents` for evidence upload
- `POST /claims/{id}/analyze` to kick off the pipeline (returns `202 Accepted` with an `analysis_id`)
- `GET /claims/{id}/analysis/{analysis_id}` to poll status
- `GET /claims/{id}/evidence`, `GET /claims/{id}/investigation` to read results
- `POST /claims/{id}/decision` for the human verdict

Migrations are managed with Alembic (5 revisions). File uploads are written to disk under `data/uploads/` and served back as static files via `app.mount("/api/uploads", StaticFiles(...))` — a simple, workable solution for a prototype that avoids a separate file service.

### Computer vision model (`ml/`)

The CV module uses a ResNet-50 backbone (ImageNet pretrained) with two output heads: head A classifies damage type across 8 categories (scratch, dent, crack, shattered glass, bumper damage, panel damage, headlight damage, no damage), and head B classifies severity into three levels (minor, moderate, severe).

**Datasets.** Two public Kaggle datasets were used:
- `hamzamanssor/car-damage-assessment` — provides damage-type labels per image
- `prajwalbhamere/car-damage-severity-dataset` — provides genuine severity labels per image

The two datasets do not overlap, so each image only has labels for one head. A masked-loss strategy was used: the damage-type loss is only backpropagated on images from the first dataset, and the severity loss is only backpropagated on images from the second. This avoids fabricating labels that do not exist.

**Training.** Two stages: stage 1 freezes the backbone and trains only the two heads (5 epochs, LR 1e-3); stage 2 unfreezes layers 3 and 4 of the backbone and fine-tunes everything at LR 1e-5 (15 epochs). Total: 20 epochs.

**Actual results (from `ml/results/training_history.json`):**
- Best validation damage-type macro-F1: ~0.19 (at epoch 3 of stage 1)
- Best validation severity accuracy: ~0.70 (at epoch 7 of stage 2)

The damage-type F1 is low. The damage-type dataset is small and the class distribution is uneven; the model struggles to distinguish visually similar categories (e.g., dent vs. panel damage). The severity head does much better because the three classes are visually more distinct. These are honest numbers — the checkpoint works for demo purposes, but the CV output is a weak signal in the current state.

The `_DemoCVPredictor` is a fallback that returns predictions derived from the uploaded image's filename rather than loading the checkpoint. It exists purely to let the demo run on a laptop with no GPU and no model file available.

### Document intelligence

The document intelligence layer in `app/services/document_intelligence.py` is a deterministic stub. It extracts a `POL-XXXX` token from the filename of documents uploaded as type `policy`, and otherwise returns an empty `extracted_fields` dict with `raw_confidence=0.5`. PyMuPDF is included as a dependency for future real extraction, but no OCR or structured extraction provider is wired in.

This was a deliberate scoping decision. Real document intelligence (OCR + structured field extraction from PDFs) is a non-trivial integration, and building a convincing fake would have been dishonest. The stub returns an honest signal that no fields were read. The UI renders "No structured fields were extracted from this document" rather than fake data.

An earlier iteration of the stub wrote a `_phase11_stub: true` key into the `extracted_fields` JSON, which the UI then rendered as if it were a real extracted field. That was fixed — the marker is now invisible to the UI.

### Consistency engine — R1 through R9

Nine rules, each a pure Python function `rule(ctx: ClaimContext) -> RiskSignal | None`. The `ClaimContext` dataclass is assembled by the pipeline before any rule is evaluated, so rules never touch the database directly and are fully unit-testable in isolation.

| Rule | What it checks |
|------|---------------|
| R1 `unsupported_damage` | A damage area in the claim form has no corresponding CV detection in the photos |
| R2 `severity_mismatch` | The claim description implies a different severity level than what CV detected |
| R3 `repair_component_mismatch` | Repair line items reference parts that are not plausibly linked to any detected damage |
| R4 `excessive_repair_cost` | Repair estimate total exceeds the synthetic baseline range for this damage type and vehicle segment |
| R5 `duplicate_previous_damage` | The same vehicle has a prior claim for overlapping damage within the last 6 months |
| R6 `policy_coverage_mismatch` | The claimed damage type is not covered under the policy's coverage type |
| R7 `claim_frequency` | The customer has filed 3 or more claims in the trailing 12 months |
| R8 `near_policy_boundary` | The incident date falls within 14 days of the policy start or end date |
| R9 `document_field_conflict` | Key fields (policy number, plate number) differ across uploaded documents |

Each fired rule writes a `RiskSignal` row and at least one `Evidence` row. The evidence references the specific image, document, or computed value that triggered the rule, so the officer can inspect the raw source.

### Risk engine

The risk engine (`app/services/risk_engine.py`) is a 5-feature, fixed-weight scoring formula:

| Feature | What it measures | Weight |
|---------|-----------------|--------|
| f1 | Count of High-severity signals (capped at 3, normalised /3) | 0.35 |
| f2 | Count of Medium-severity signals (capped at 5, normalised /5) | 0.15 |
| f3 | Repair cost ratio vs baseline upper bound (capped at 3.0) | 0.25 |
| f4 | Previous-claim overlap score from R5 (0–1) | 0.15 |
| f5 | Anomaly feature — not implemented (weight redistributed) | 0.10 |

`score = 100 × Σ(weight_i × f_i)`, clamped to [0, 100]. Bands: Low (0–34), Medium (35–64), High (65–100).

**The Isolation Forest is not implemented.** The architecture document mentioned it as a potential future feature. The actual code has a comment in the docstring explaining this and states that f5 is a no-op with its weight redistributed proportionally. No sklearn Isolation Forest is fitted, loaded, or called at any point.

**Low-data-confidence default.** A claim with zero fired signals and low-confidence CV output is bumped to at least Medium rather than left at Low. The intent is to avoid automatically clearing claims that simply lack evidence rather than being genuinely clean. This is what causes the one known test failure (`test_scenario_1_legitimate_claim_low`), which asserts that a clean demo claim scores Low.

The repair-cost baseline is built from a small synthetic dataset embedded in the module itself. The docstring is explicit: "illustrative, not industry-validated." The numbers were constructed to produce plausible-looking results for demo scenarios, not fit to real insurer data.

### Gemini investigation layer

`app/services/gemini_client.py` calls `gemini-2.5-flash` via `httpx` (no Gemini SDK, to avoid lock-in). The input is a structured JSON object containing the claim's risk score, risk band, the list of fired `RiskSignal` rows, and the linked `Evidence` rows.

The prompt instructs the model to write a 3–6 sentence summary that:
- cites only `rule_id` values that exist in the input
- does not recompute any scores or arithmetic
- does not invent amounts, dates, or damage not present in the input
- does not use fraud accusation language

After parsing the response, the code:
1. Strips any `key_concerns` bullet that references a `rule_id` not in the original signal list
2. **Overwrites** `recommendation` with the deterministic value derived from the risk band (`Low → normal`, `Medium → manual_review`, `High → investigate`) — Gemini's own recommendation is discarded
3. Appends a fixed disclaimer: "AI-generated, human decision required"

On failure (timeout, 5xx, or parse error), the client makes one retry. If the second attempt also fails, it returns `None`. The pipeline then persists an `Investigation` row with `summary_text=None` and still completes normally — the rest of the evidence is not discarded just because the narrative failed.

The demo stub builds a canned summary from the actual deterministic risk signals and passes it through the same validator.

### Pipeline orchestrator

`POST /claims/{id}/analyze` is handled by `app/api/pipeline.py`. It validates the claim, creates an `Analysis` row in `running` status, returns `202 Accepted` with the `analysis_id`, then hands off to a thread that runs the actual pipeline:

```
CV on all pending images
→ document extraction on all pending documents
→ build ClaimContext
→ delete any existing RiskSignals for the claim (prevents duplicates on rerun)
→ evaluate R1–R9
→ persist RiskSignals + Evidence
→ compute risk score
→ persist risk score to Claim
→ generate Gemini investigation
→ persist Investigation
→ mark Analysis + Claim as completed
```

Duplicate runs on the same claim are blocked by two mechanisms: an in-process `pipeline_locks` dict (for the same Python interpreter) and a partial unique index `uq_analyses_one_running_per_claim` on the `analyses` table (which serves as an additional guard against concurrent running analyses, though the overall pipeline remains single-process).

A rerun removes the previous `RiskSignal` rows and their associated evidence before generating the new current set, so the database always holds exactly one current set of signals per claim. The `Investigation` row is upserted (one per claim).

The `Analysis` table retains a history of every run (started_at, finished_at, status, error_message) for auditing, even though the signals themselves reflect only the most recent run.

### Frontend

Nine screens, built in React 18 + TypeScript 5.6 + Vite 6. No component library — all styles are vanilla CSS with a custom design-system token file. The API client is a typed fetch wrapper with no extra HTTP library.

1. **Dashboard** — stat cards for claim counts by status, clickable to filtered list views
2. **Claims List** — filterable by status and risk band
3. **New Claim** — four-step form: customer/policy → vehicle/incident → documents → images
4. **Claim Analysis** — pipeline stage tracker with live polling
5. **Image Analysis** — per-image CV results (damage type + severity + confidence)
6. **Document Viewer** — document tabs with extracted fields panel; upload additional documents
7. **Risk Signals** — all fired rules sorted by severity with linked evidence detail
8. **Investigation Summary** — Gemini narrative + key concerns + disclaimer
9. **Decision Panel** — approve / investigate / deny + optional notes

The frontend does not make any risk decisions. It is a view over the backend API.

### Demo data generator

`scripts/generate_demo_data.py` is deterministic (fixed random seed, default 42). It creates five scenario claims with the complete entity graph (customer, vehicle, policy, accident, images, documents, repair estimate, and previous claims where relevant). Claim numbers are fixed: `CLM-DEMO-S1-LEGIT` through `CLM-DEMO-S5-MULTI`. The script is idempotent: `--reset` drops the existing demo data before reseeding.

---

## Major challenges and how they were handled

### Keeping the LLM from inventing

The first risk with using an LLM to summarise a claim is that it will confidently invent amounts, dates, and damage types that were not in the input. The mitigation was layered:

- The input schema (`InvestigationInput`) only contains data already computed deterministically. There is no natural-language claim description in the prompt — only structured fields.
- A post-parse validator strips any `key_concerns` that reference a rule ID not in the original signal list.
- The `recommendation` field is computed by a deterministic function and written over whatever Gemini returns.
- The disclaimer is a fixed constant the model is instructed not to change.

This doesn't eliminate all hallucination risk, but it substantially reduces the surface and contains the damage.

### The stub that pretended to do OCR

An early iteration of the document intelligence service wrote `_phase11_stub: true` into the `extracted_fields` JSON to mark rows as stub output. The frontend iterated over `extracted_fields` and rendered each key as a real extracted field — so users would see `_phase11_stub: true` displayed as if it were a genuine extraction result.

The fix was to make the stub return an honest empty dict with no marker key, and update the frontend to render a clearly labelled "no structured fields extracted" state.

### Schema drift between SQLAlchemy model and Alembic

The first full end-to-end demo failed because the `decision_notes` column existed on the `Claim` SQLAlchemy model but had no corresponding migration. Tests passed throughout development because they use `Base.metadata.create_all`, which builds the schema from the current model definition. The production-shaped database (built from Alembic revisions) was missing the column.

The fix was to add the missing migration and verify it was idempotent for databases where the relevant constraint might not have existed. A CI check that diffs the SQLAlchemy model against the Alembic head migration would catch this class of bug automatically — this remains a future task.

### Concurrent analysis runs on the same claim

Without a guard, two simultaneous requests to `POST /claims/{id}/analyze` would both succeed, both read the same claim state, and both write overlapping results. The fix was two-layered: an in-process `pipeline_locks` dictionary keyed on `claim_id` and a partial unique index on the `analyses` table that constrains one `running` row per claim. This index only guards concurrent running analyses; the overall pipeline remains single-process.

### Duplicate RiskSignals on rerun

A related bug: running analysis twice on the same claim accumulated risk signal rows instead of replacing them. The `consistency.persist()` function simply added new `RiskSignal` rows without clearing old ones. The fix was to delete existing signals for the claim before generating new ones (step 6 of the pipeline). Reruns remove the previous `RiskSignal`s and their associated evidence before generating the new current set. The `Investigation` row is already upserted, so no other cleanup was needed.

### CV model — damage type F1 is low

The damage-type classification head ended training with a validation macro-F1 of ~0.19. The primary causes:

- The dataset is small (roughly 2,200 training images split across 8 classes).
- Several classes are visually similar (dent vs. panel damage, scratch vs. bumper damage).
- The masked-loss training means the damage-type head only sees images from one dataset; it never sees images from the severity dataset, which may have had useful diversity.

Improving this would require a larger, better-curated dataset and more training time. For the prototype, the demo predictor (filename-based) is more reliable for walkthroughs because it produces predictable, deterministic results.

### Known calibration issue with demo scenario S1

The blueprint describes five demo scenarios with expected risk bands. Scenario S1 (legitimate claim, no red flags) is expected to score `Low`. The risk engine's low-data-confidence default bumps any claim with zero signals to `Medium`, so S1 scores `Medium`. Recalibrating the weights or the default threshold would fix this, but modifying risk scoring was out of scope. The mismatch is documented, the test is left as a known failure, and the table in the DEVLOG is honest about it.

---

## Testing

The test suite at the end of the project: **281 passed, 1 known failure**.

Backend tests cover: API routes (CRUD and error paths), the consistency engine rules (each rule in isolation and in combinations), the risk engine (scoring formula, band mapping, low-confidence default), the Gemini client (happy path, parse failure, retry, banned-phrase filter), the pipeline orchestrator (happy path, concurrent run guard, missing inputs, failure isolation), and several edge cases (corrupted image, missing checkpoint, empty claim). One opt-in test exercises a real PostgreSQL connection; all others use SQLite in-memory via `StaticPool`.

ML tests (21 tests) cover: happy path prediction, corrupted image handling, missing checkpoint, tiny image, batch prediction, and the API integration with a mock predictor.

Gemini tests use mocks or the demo stub. No live Gemini API calls are made in CI.

Frontend: `npx tsc --noEmit` and `npm run build` run clean. There is no JS test framework.

---

## Current state: what works, what is limited, what is future

| Area | Status |
|------|--------|
| Full CRUD for claims entity graph | Working |
| Image upload + CV inference (demo mode) | Working |
| Image upload + CV inference (trained checkpoint) | Working but low damage-type accuracy |
| Document upload | Working |
| Document field extraction | Stub only (filename-based) |
| R1–R9 consistency rules | Working |
| Risk scoring (deterministic formula) | Working |
| Gemini investigation narrative | Working (with demo stub available) |
| Human decision recording | Working |
| Demo data + demo mode | Working |
| Tests (backend + ML) | 281 passed, 1 known failure |
| Frontend type check + build | Clean |
| Authentication | Not implemented |
| Multi-process concurrency | Not safe (single-process only) |
| Docker / production configuration | Not implemented |

---

## Future improvements

In rough priority order:

1. **Real document intelligence.** The whole consistency engine benefits most from being able to actually read the fields on a claim form or repair estimate. A real OCR + structured extraction pipeline (PDF text extraction + Gemini vision for scanned documents) would unlock R9 meaningfully and make R3/R4 much stronger.

2. **CI model-vs-migration drift check.** A script that generates SQL from the current SQLAlchemy models and diffs it against the Alembic head migration would catch the `decision_notes`-class of bug before it reaches a deployed database.

3. **Authentication and per-user audit trail.** The prototype assumes a single trusted local user. Real deployment would need session management, per-user claim assignment, and an audit log of who made which decision.

4. **External queue for pipeline runs.** Replacing the in-process thread with an RQ or Celery worker would allow the API to serve requests during long analysis runs without pinning a worker.

5. **CV model improvement.** A larger, more balanced dataset for the damage-type head, combined with a longer training run and proper hyperparameter tuning, would improve the damage-type F1 substantially.

6. **Startup sweeper.** A startup task that flips `Analysis` rows stuck in `running` to `failed (interrupted)` would handle the case where the process dies mid-pipeline.

7. **Demo scenario calibration.** The S1 (legitimate) scenario currently scores `Medium` instead of the expected `Low`. Recalibrating the low-data-confidence default or the feature weights would fix this without changing the risk logic's fundamental approach.

8. **Frontend test framework.** A Vitest + React Testing Library setup for the filter logic, stage tracker, and demo-mode badge would give the frontend more coverage than type-checking alone.

---

## Files in this repo

- `README.md` — installation, configuration, run instructions, and limitations.
- `DEVLOG.md` — this document.
- `backend/` — FastAPI application, Alembic migrations, services, tests.
- `frontend/` — React + TypeScript + Vite application.
- `ml/` — CV model training code, inference module, training history.
- `scripts/generate_demo_data.py` — deterministic five-scenario demo data seeder.
- `tests/` — backend and ML test suites.
