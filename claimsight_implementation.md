# ClaimSight — Implementation Blueprint
**AI-Assisted Vehicle Insurance Claims Investigation Platform (Decision-Support Prototype)**

Version 1.0 — Technical Specification for AI Coding Agents (Antigravity / Claude Code)

> This is a decision-support prototype. It never adjudicates claims autonomously; a human claims officer always makes the final decision. All outputs are recommendations with attached evidence.

---

## 0. Legend

- **MUST** = required for MVP acceptance.
- **NICE** = explicitly out of MVP scope; implement only after MUST items are complete and stable.

---

## 1. Complete Architecture

```
┌─────────────┐     ┌──────────────┐     ┌────────────────────────┐
│  Frontend   │────▶│   Backend    │────▶│  Analysis Orchestrator  │
│  React/TS   │◀────│   FastAPI    │◀────│  (async pipeline runner)│
└─────────────┘     └──────────────┘     └───────────┬─────────────┘
                                                       │
                     ┌─────────────────────────────────┼─────────────────────────────────┐
                     ▼                 ▼               ▼                ▼               ▼
              ┌────────────┐   ┌──────────────┐  ┌─────────────┐ ┌────────────┐ ┌───────────────┐
              │  CV Module │   │  Document    │  │ Consistency  │ │  Anomaly/  │ │  Gemini LLM   │
              │ (CNN infer)│   │  Intelligence│  │   Engine     │ │  Risk Eng. │ │  Summary Layer│
              └─────┬──────┘   └──────┬───────┘  └──────┬───────┘ └─────┬──────┘ └───────┬───────┘
                    │                 │                 │               │                │
                    └─────────────────┴────────┬────────┴───────────────┴────────────────┘
                                                ▼
                                     ┌───────────────────────┐
                                     │ PostgreSQL + File Store│
                                     │ (evidence, results)    │
                                     └───────────┬────────────┘
                                                 ▼
                                     ┌───────────────────────┐
                                     │       Frontend         │
                                     │  (results, evidence)   │
                                     └───────────────────────┘
```

### 1.1 Component Table

| Component | Technology | Responsibility | Input | Output | Why |
|---|---|---|---|---|---|
| Frontend | React 18 + TypeScript + Vite | Upload UX, dashboards, evidence viewer, decision panel | User actions, API JSON | UI renders, API calls | Type-safe, fast dev, matches known stack |
| Backend API | FastAPI (Python 3.11) | REST endpoints, auth stub, request validation, orchestration trigger | HTTP requests | JSON responses | Async-native, Pydantic-native, matches known stack |
| Orchestrator | Python async module inside FastAPI (`services/pipeline.py`), backed by a task queue for long-running analysis | Sequences CV → Doc → Consistency → Anomaly → Risk → LLM → persistence | Claim ID | Pipeline result object | Keeps stages decoupled, testable in isolation |
| CV Module | PyTorch + torchvision (ResNet-50 transfer learning) | Damage detection/classification + severity estimate from images | Image files | Structured damage predictions + confidence | Best fit for transfer learning on small dataset |
| Document Intelligence | PyMuPDF + Gemini (structured extraction) + Pydantic | Extract structured fields from claim form, policy, estimate, invoice | PDF/image files | Validated Pydantic objects | PyMuPDF for text/layout, Gemini for unstructured/handwritten content |
| Consistency Engine | Pure Python, rule-based (`services/consistency.py`) | Deterministic cross-checks (image vs claim vs estimate vs policy vs history) | Normalized structured data | List of `RiskSignal` objects | Must be deterministic, auditable, no LLM |
| Anomaly/Risk Engine | scikit-learn (Isolation Forest) + Pandas/NumPy + deterministic weighted scoring | Financial anomaly detection + explainable composite risk score | Claim features + risk signals | Risk score (0–100), band, contributing factors | Explainability required for insurance domain |
| Gemini LLM Layer | Gemini API (structured JSON mode) | Investigation summary, evidence-grounded narrative, Q&A on analyzed claim | Risk signals + evidence + extracted docs | Structured summary object | Only component that produces free text; never computes numbers |
| Database | PostgreSQL 15 + SQLAlchemy 2.0 | Persist claims, evidence, signals, results | ORM writes | Rows/relations | Relational integrity for evidence traceability |
| File Storage | Local filesystem in dev (`/data/uploads/{claim_id}/`), swappable to S3-compatible store in prod | Store raw uploads + annotated images | Files | File paths/URLs | Simplicity for prototype, clean upgrade path |
| Evidence Store | Rows in `evidence` table + file references | Bind every risk signal to concrete proof | Signal + source refs | Evidence bundle | Core product differentiator |

### 1.2 Communication

- Frontend ↔ Backend: REST + JSON, `multipart/form-data` for uploads.
- Backend ↔ Orchestrator: in-process async function calls (FastAPI `BackgroundTasks` for MVP; NICE: Celery + Redis for real async queueing/retries at scale).
- Orchestrator ↔ CV/Document/Consistency/Anomaly/LLM modules: direct Python function calls (all in-process for MVP — no microservices, avoids unneeded network hops).
- All modules ↔ Database: SQLAlchemy sessions, one transaction per claim analysis run.

---

## 2. Computer Vision / CNN

### 2.1 MVP Decision

**MUST**: Multi-label **classification** (damage type presence) + **severity classification** (per detected damage region: minor/moderate/severe), using **transfer learning**. No object detection/segmentation in MVP — bounding boxes add annotation cost without improving the risk narrative significantly at prototype scale.

**NICE**: Object detection (YOLOv8) for bounding-box localization if time allows, purely for UI polish (highlighted damage regions).

### 2.2 Model

- **Architecture**: ResNet-50 pretrained on ImageNet, fine-tuned. Replace final FC layer with two heads:
  - Head A (multi-label, sigmoid): damage type — `{scratch, dent, crack, shattered_glass, bumper_damage, panel_damage, headlight_damage, no_damage}`.
  - Head B (single-label, softmax): severity — `{minor, moderate, severe}` (only meaningful when damage detected).
- **Framework**: PyTorch + torchvision.
- **Training strategy**: Freeze early ResNet blocks (layer1–layer2), fine-tune layer3–layer4 + both heads. Two-stage: (1) train heads only 5 epochs, lr=1e-3; (2) unfreeze layer3–4, fine-tune 10–15 epochs, lr=1e-5.

### 2.3 Dataset

- **Name**: Car Damage Detection dataset (publicly available on Kaggle/Roboflow — e.g. "Car Damage Detection" / "Vehicle damage severity" datasets).
- **Source**: Kaggle (`https://www.kaggle.com/datasets` search "car damage detection"), Roboflow Universe.
- **Classes/labels**: as listed in 2.2, mapped to the closest available labels in the chosen public dataset (do not fabricate classes not present in source data — audit and remap during data prep script).
- **Note**: exact dataset selection must be verified at implementation time (Kaggle datasets change availability); the coding agent should document the exact dataset name/version used in `ML/data_card.md`.

### 2.4 Preprocessing & Augmentation

- Resize to 224×224, normalize with ImageNet mean/std.
- Augmentation (train only): random horizontal flip, random rotation (±15°), color jitter (brightness/contrast ±0.2), random crop with padding.
- No augmentation on validation/test.

### 2.5 Split & Metrics

- Split: 70% train / 15% validation / 15% test, stratified by damage class where possible.
- Metrics: per-class F1 and macro-F1 for Head A (multi-label); accuracy + confusion matrix for Head B (severity); report both on held-out test set.
- Target (documented expectation, not guaranteed): macro-F1 ≥ 0.65 for damage type on this dataset size — realistic for a prototype, not production-grade.

### 2.6 Inference Pipeline

1. Load image → validate format/size → preprocess.
2. Forward pass → sigmoid probabilities per damage type (threshold 0.5, configurable) + severity softmax.
3. Output: `{damage_types: [{label, confidence}], severity: {label, confidence}, low_confidence: bool}`.
4. `low_confidence = true` when top damage confidence < 0.4 or severity confidence < 0.5 → flagged distinctly in UI ("model uncertain — manual review recommended") and excluded from automatic risk deductions.

### 2.7 UI Surfacing

- Per-image card: detected damage chips with confidence %, severity badge, low-confidence warning banner when applicable.
- Aggregate claim-level view: union of all detected damage types across images, worst severity found.

---

## 3. Datasets + Demo Data

### 3.1 Required Datasets

| Purpose | Name | Source | Fields/Classes | Usage |
|---|---|---|---|---|
| Vehicle damage classification | Car Damage Detection (Kaggle/Roboflow) | Kaggle/Roboflow Universe | scratch, dent, crack, glass shatter, bumper/panel damage | Train CV Module (Section 2) |
| Repair cost baselines | Synthetic (see 3.3) — no reliable public granular repair-cost dataset exists at prototype scale | Generated in-house | part, labor hours, cost per severity | Baseline for anomaly detection / excessive-cost rule |
| Fraud indicators reference | Public insurance-fraud research summaries (used only as *reference for rule design*, not as training data — do not claim a real fraud-detection model trained on proprietary industry data) | N/A (informs rule design only) | claim frequency, mismatch patterns | Informs Consistency Engine rule set |

Do not fabricate a dataset that "trains" fraud detection on real insurer data — this does not exist publicly at accessible scale. The Anomaly/Risk Engine (Section 6) uses statistical/rule-based methods on **synthetic + input claim data**, which is honest and defensible for a prototype.

### 3.2 Entity Schema (used across DB + demo data)

```
Customer(id, name, email, phone, created_at)
Vehicle(id, customer_id, make, model, year, vin, plate_number)
Policy(id, customer_id, vehicle_id, policy_number, coverage_type, coverage_limit, deductible, start_date, end_date, status)
Claim(id, policy_id, vehicle_id, claim_number, incident_date, reported_date, claimed_amount, status, risk_band, risk_score, created_at)
Accident(id, claim_id, description, location, incident_type)
Damage(id, claim_id, source [image|claim_form], damage_type, severity, confidence, region_ref)
Document(id, claim_id, doc_type [claim_form|policy|estimate|invoice|previous_claim], file_path, extraction_status, raw_confidence)
RepairEstimate(id, claim_id, document_id, shop_name, total_cost, currency, issued_date)
RepairItem(id, repair_estimate_id, part_name, operation [replace|repair|paint], cost, labor_hours)
PreviousClaim(id, customer_id, vehicle_id, claim_number, incident_date, damage_summary, claimed_amount, overlap_score)
RiskSignal(id, claim_id, rule_id, category, severity [low|medium|high], description, created_at)
Evidence(id, risk_signal_id, evidence_type [image|document|field|computed], reference, detail_json)
Investigation(id, claim_id, summary_text, recommendation [normal|manual_review|investigate], generated_at, model_version)
```

### 3.3 Synthetic Demo Scenarios (MUST build all 5)

1. **Legitimate claim** — image damage matches claim description matches repair estimate; cost within baseline range; no previous claim overlap. Expected risk: **Low**.
2. **Inflated repair estimate** — image shows minor/moderate damage; repair estimate cost is 2.5–4× the baseline range for that damage type/vehicle segment. Expected risk: **Medium–High**, triggers "excessive repair cost" rule.
3. **Image/document mismatch** — claim form describes rear-end collision damage; uploaded images show only front bumper scratch (or no matching damage detected). Expected risk: **High**, triggers "unsupported damage" rule.
4. **Previous-claim overlap** — same vehicle/customer has a previous claim within the last 6 months for damage in the same vehicle region (e.g., same panel). Expected risk: **Medium–High**, triggers "duplicate/overlapping damage" rule.
5. **Multi-signal suspicious claim** — combination of ≥3 signals: inflated cost + slight image/claim mismatch + recent previous claim + claim filed near policy end date. Expected risk: **High**, multiple contributing signals shown in evidence.

Demo data generation script: `scripts/generate_demo_data.py`, seeds all tables above with referential integrity, deterministic (fixed random seed) so evaluators see reproducible results.

---

## 4. Document Intelligence

### 4.1 Pipeline (per document)

1. Detect file type (PDF vs image) → route accordingly.
2. **PDF**: PyMuPDF extracts raw text + layout blocks. If text layer is empty/sparse (scanned doc) → rasterize pages to images, pass to Gemini vision for OCR + extraction in one call.
3. **Image documents** (photographed forms): pass directly to Gemini vision.
4. Gemini is prompted with the raw text/image **plus** a strict JSON schema (see 4.2) and instructed to extract only what is present, marking missing fields as `null` — never inferring values not in the source.
5. Gemini's JSON response is parsed and validated against a Pydantic model. On validation failure → one retry with an error-correction prompt containing the validation error. On second failure → mark document `extraction_status = "failed"`, surface to UI for manual entry.
6. Each extracted field carries a `confidence` (`high|medium|low`) supplied by Gemini in the schema; UI flags `low` confidence fields for officer verification.

### 4.2 Structured Schemas (Pydantic, illustrative)

```python
class ClaimFormExtraction(BaseModel):
    claimant_name: str | None
    policy_number: str | None
    vehicle_plate: str | None
    incident_date: date | None
    incident_description: str | None
    claimed_damage_areas: list[str]
    claimed_amount: float | None
    confidence: Literal["high", "medium", "low"]

class PolicyExtraction(BaseModel):
    policy_number: str | None
    coverage_type: str | None
    coverage_limit: float | None
    deductible: float | None
    start_date: date | None
    end_date: date | None
    confidence: Literal["high", "medium", "low"]

class RepairEstimateExtraction(BaseModel):
    shop_name: str | None
    total_cost: float | None
    items: list[RepairItemExtraction]
    issued_date: date | None
    confidence: Literal["high", "medium", "low"]

class RepairItemExtraction(BaseModel):
    part_name: str | None
    operation: Literal["replace", "repair", "paint"] | None
    cost: float | None
    labor_hours: float | None
```

### 4.3 Failure Handling

- Corrupted/unreadable PDF → `extraction_status = "failed"`, claim still proceeds with other documents; UI shows "document unreadable — please re-upload".
- Gemini API timeout/error → retry once with backoff (2s), then fail gracefully (see Section 13).
- Partial extraction (some fields null) → proceed; downstream rules treat missing fields as "insufficient data" (not as a violation).

---

## 5. Consistency / Investigation Engine (Deterministic, Rule-Based, No LLM)

| Rule ID | Compares | Logic | Output Severity |
|---|---|---|---|
| `R1_unsupported_damage` | Claim form damage areas vs CV-detected damage | If claimed area has no corresponding detected damage type (and image confidence is not low) | High |
| `R2_severity_mismatch` | Claim description severity language vs CV severity class | Simple keyword severity extraction (e.g. "totaled", "minor scratch") compared to CV severity label; mismatch by ≥2 levels | Medium |
| `R3_repair_component_mismatch` | Repair items vs CV-detected damage regions | Repair item part not plausibly linked to any detected/claimed damage area (lookup table: damage_type → plausible parts) | Medium |
| `R4_excessive_repair_cost` | Repair estimate total vs baseline cost range (Section 6) | `total_cost > baseline_upper * 1.5` | High (if >2×), Medium (if 1.5–2×) |
| `R5_duplicate_previous_damage` | Previous claims (same vehicle) vs current claim region/date | Same vehicle, damage region overlap, incident dates within 6 months | High |
| `R6_policy_coverage_mismatch` | Claimed damage type vs policy coverage type | Claimed damage type not covered under policy's coverage type | High |
| `R7_claim_frequency` | Customer's claim history | ≥3 claims in trailing 12 months | Medium |
| `R8_near_policy_boundary` | Incident date vs policy start/end date | Incident within 14 days of policy start or end | Medium |
| `R9_document_field_conflict` | Cross-document field agreement | Policy number / plate number differ across documents for same claim | High |

Each rule implementation: pure function `rule(claim_context: ClaimContext) -> RiskSignal | None`, unit-testable in isolation, no external calls. `ClaimContext` is a single dataclass assembled by the orchestrator after CV + document extraction, so rules never touch raw files.

---

## 6. Anomaly + Risk Engine

### 6.1 Method Split

- **Rules** (Section 5) → binary/categorical signals.
- **Statistical baseline** → repair cost baseline range per `(vehicle_segment, damage_type, severity)` computed from the synthetic demo dataset (mean ± 1.5×IQR as MUST; documented as illustrative, not industry-validated).
- **Isolation Forest** (scikit-learn) — **NICE**, trained on synthetic claim feature vectors (claimed_amount, days_to_report, previous_claim_count, cost_per_damage_ratio) to catch multivariate outliers rules don't explicitly cover. If implemented, its output is one additional weighted feature into the score (6.2), never a standalone verdict — keeps it auditable.

### 6.2 Explainable Risk Score

- **Features** (each normalized 0–1):
  - `f1`: count of High-severity RiskSignals (capped at 3, /3)
  - `f2`: count of Medium-severity RiskSignals (capped at 5, /5)
  - `f3`: repair cost ratio vs baseline upper bound (capped at 3.0, /3.0)
  - `f4`: previous-claim overlap score (0–1, from `R5`)
  - `f5`: (NICE) Isolation Forest anomaly score, min-max normalized
- **Weights** (MUST, fixed & documented — not learned, for explainability):
  - High signals: 0.35, Medium signals: 0.15, Cost ratio: 0.25, Previous overlap: 0.15, Anomaly (if used): 0.10 (redistribute proportionally if `f5` unused)
- **Score** = `100 × Σ(weight_i × f_i)`, clamped to [0, 100].
- **Bands**: Low = 0–34, Medium = 35–64, High = 65–100.
- **Confidence**: derived from proportion of documents/images with `low` confidence extraction — if >30% of inputs are low-confidence, score is shown with a "low data confidence" qualifier and defaults toward Medium review rather than automatic Low clearance.
- **Evidence mapping**: risk score response includes `contributing_factors: [{feature, weight, value, linked_signal_ids}]` so every point of the score traces to specific RiskSignal/Evidence rows.

---

## 7. Gemini / LLM Layer

### 7.1 Usage Scope

**Used for**: (a) document field extraction (Section 4), (b) investigation summary generation, (c) answering investigator follow-up questions about an already-analyzed claim.

**Never used for**: arithmetic, risk score computation, rule evaluation, or unsupported fraud accusations. The prompt explicitly instructs Gemini to only narrate signals/evidence already computed deterministically, and to phrase findings as "recommend review" language, never as accusations ("this claim is fraudulent" is disallowed phrasing — enforced via prompt instruction + a post-generation regex/keyword check that rejects and regenerates if banned phrasing appears).

### 7.2 Input Structure (Investigation Summary)

```json
{
  "claim_id": "...",
  "risk_score": 72,
  "risk_band": "High",
  "risk_signals": [{"rule_id": "R4_excessive_repair_cost", "severity": "High", "description": "..."}],
  "evidence": [{"signal_id": "...", "type": "computed", "detail": {"baseline_range": [800, 1400], "claimed": 3200}}],
  "extracted_documents_summary": {"claim_form": {...}, "policy": {...}, "repair_estimate": {...}},
  "cv_findings": [{"image_id": "...", "damage_types": [...], "severity": "moderate"}]
}
```

### 7.3 Output Schema

```python
class InvestigationSummary(BaseModel):
    summary: str            # 3-6 sentences, evidence-grounded
    key_concerns: list[str] # bullet list, each tied to a risk_signal id
    recommendation: Literal["normal", "manual_review", "investigate"]
    disclaimer: str         # fixed: "AI-generated, human decision required"
```

### 7.4 Grounding & Hallucination Prevention

- Prompt includes explicit instruction: "Only reference facts present in the provided JSON. Do not invent amounts, dates, or damage not listed."
- Every sentence in `key_concerns` must cite a `rule_id` present in the input — post-generation validation checks each cited `rule_id` exists in the original signal list; if not, that bullet is stripped before saving.
- `recommendation` is **not** chosen freely by Gemini — it is computed deterministically from the risk band (Low→normal, Medium→manual_review, High→investigate) and only passed to Gemini for narrative consistency; if Gemini's returned value disagrees with the deterministic mapping, the deterministic value wins (backend overwrites it before persistence).

### 7.5 Failure/Retry

- Timeout or 5xx from Gemini → retry once (backoff 2s) → on second failure, persist claim with `investigation.summary = null`, UI shows deterministic risk score/signals with a "narrative summary unavailable — retry" button (calls the summary endpoint alone, not the whole pipeline).
- Malformed JSON response → one repair attempt (re-prompt with the parse error) → same fallback as above on second failure.

---

## 8. Evidence System

- Every `RiskSignal` row has ≥1 `Evidence` row.
- `Evidence.evidence_type` ∈ `{image, document, field, computed}`:
  - `image` → `reference` = image_id, `detail_json` = bounding/region info (if available) + CV confidence.
  - `document` → `reference` = document_id, `detail_json` = page number + extracted field name/value.
  - `field` → `detail_json` = the two conflicting field values + their sources.
  - `computed` → `detail_json` = the calculation inputs (e.g., baseline range, claimed value, ratio).
- Frontend Evidence Viewer (Section 11): clicking a risk signal opens a side panel showing all linked evidence — annotated image thumbnail, document page snippet, or computed value breakdown, side-by-side.

---

## 9. Database (PostgreSQL)

- All entities from Section 3.2 become tables with standard `id BIGSERIAL PRIMARY KEY`, `created_at TIMESTAMPTZ DEFAULT now()`.
- Foreign keys: `Vehicle.customer_id → Customer.id`, `Policy.customer_id/vehicle_id`, `Claim.policy_id/vehicle_id`, `Accident.claim_id`, `Damage.claim_id`, `Document.claim_id`, `RepairEstimate.claim_id/document_id`, `RepairItem.repair_estimate_id`, `PreviousClaim.customer_id/vehicle_id`, `RiskSignal.claim_id`, `Evidence.risk_signal_id`, `Investigation.claim_id`.
- Indexes: `Claim.policy_id`, `Claim.vehicle_id`, `Claim.status`, `Document.claim_id`, `RiskSignal.claim_id`, `Evidence.risk_signal_id`, `PreviousClaim.vehicle_id`.
- Constraints: `Claim.risk_score` CHECK between 0–100; `RiskSignal.severity` and `Investigation.recommendation` as Postgres ENUM types.
- No over-engineering: no partitioning, no read replicas, no event sourcing for MVP.

---

## 10. Backend API

| Method | Path | Purpose | Request | Response | Key Errors |
|---|---|---|---|---|---|
| POST | `/customers` | Create customer | `CustomerCreate` | `Customer` | 400 validation |
| POST | `/claims` | Create claim (links policy/vehicle) | `ClaimCreate` | `Claim` | 404 policy not found |
| POST | `/claims/{id}/documents` | Upload a document (claim form/policy/estimate/invoice) | multipart file + `doc_type` | `Document` | 400 unsupported type, 413 too large |
| POST | `/claims/{id}/images` | Upload accident image(s) | multipart files | `list[Damage-pending]` | 400 unsupported format |
| POST | `/claims/{id}/analyze` | Trigger full pipeline (Section 12) | — | `202 {analysis_id}` | 409 already analyzing, 400 missing required docs |
| GET | `/claims/{id}/analysis/{analysis_id}` | Poll analysis status/result | — | `{status, risk_score?, risk_band?}` | 404 |
| GET | `/claims/{id}` | Get full claim detail | — | `ClaimDetail` | 404 |
| GET | `/claims/{id}/evidence` | Get all risk signals + evidence | — | `list[RiskSignalWithEvidence]` | 404 |
| GET | `/claims/{id}/investigation` | Get LLM investigation summary | — | `InvestigationSummary` | 404, 202 if pending |
| POST | `/claims/{id}/decision` | Officer records final decision | `{decision, notes}` | `Claim` (updated) | 400 invalid decision, 409 already decided |
| GET | `/claims` | List/filter claims (dashboard) | query params: status, risk_band | `list[ClaimSummary]` | — |
| GET | `/claims/{id}/previous-claims` | Get linked previous claims for the vehicle/customer | — | `list[PreviousClaim]` | 404 |

All endpoints: Pydantic request/response models, standard FastAPI error handling (`HTTPException`), CORS enabled for frontend origin.

---

## 11. Frontend + UX

### 11.1 Screens (MUST)

1. **Dashboard** — claim volume by risk band, recent claims needing review, quick stats (not vanity metric cards — functional counts tied to action, e.g. "7 claims awaiting review" is clickable).
2. **Claims List** — filterable/sortable table by status, risk band, date.
3. **New Claim / Upload** — stepped form: customer/policy lookup → vehicle/incident details → document upload → image upload → submit for analysis.
4. **Claim Analysis** — pipeline progress (stage-by-stage status), then results summary once complete.
5. **Image Analysis** — image gallery with per-image damage/severity chips, low-confidence warnings.
6. **Document/Evidence Viewer** — tabbed document previews with extracted-field overlay, confidence indicators.
7. **Risk Signals** — list of triggered rules, severity-colored, expandable to evidence.
8. **Investigation Summary** — Gemini narrative, key concerns list, recommendation badge, disclaimer.
9. **Decision Panel** — officer selects final action (approve/manual review/investigate/deny), notes field, submit.

### 11.2 Design Direction

- Typography: a serif or slab-serif display face for headings (e.g. "Source Serif 4" / "Newsreader") paired with a clean sans for body/UI (e.g. "Inter") — signals "investigation/legal" seriousness rather than generic SaaS.
- Palette: restrained — deep navy/charcoal base, muted amber/gold for medium risk, muted red (not neon) for high risk, forest green for low risk. No purple-blue gradients.
- Evidence-first layout: risk signal ↔ evidence is always a two-pane or expandable layout, never a modal that hides context.
- Image annotation: canvas overlay (or SVG) drawing bounding boxes/highlight regions on detected damage, toggleable.
- Motion: subtle (150–200ms ease) for panel expand/collapse and pipeline stage transitions only — no decorative animation.
- Empty/loading/error states designed per-screen, not generic spinners: pipeline stages show a real step tracker ("Analyzing images… Extracting documents… Running consistency checks…").
- Accessibility: semantic HTML, focus states, color not the sole indicator of risk band (icon + text label always paired with color).

Reference the `frontend-design` skill for concrete design-token and Tailwind implementation guidance when building this.

---

## 12. Analysis Pipeline — `POST /claims/{id}/analyze`

1. **Validate**: claim exists, status not already `analyzing`/`completed`, required inputs present (≥1 image, claim form, policy, repair estimate) → else 400 listing missing items.
2. **Set status** = `analyzing`, create `analysis_id`, return `202` immediately; remaining steps run in background task.
3. **Image processing**: for each image → CV inference (Section 2.6) → persist `Damage` rows.
4. **Document extraction**: for each unprocessed document → Section 4 pipeline → persist extracted structured data.
5. **Normalization**: assemble unified `ClaimContext` dataclass from CV + document outputs + previous claims lookup.
6. **Consistency checks**: run all rules (Section 5) against `ClaimContext` → persist `RiskSignal` rows.
7. **Anomaly detection**: compute cost-ratio + (NICE) Isolation Forest feature → feed into risk score.
8. **Risk scoring**: compute score/band/contributing factors (Section 6.2) → persist on `Claim`.
9. **Evidence generation**: for each `RiskSignal`, persist linked `Evidence` rows (Section 8).
10. **Gemini investigation summary**: call Section 7 pipeline → persist `Investigation` row.
11. **Persistence**: set claim `status = "completed"`, `analysis` marked complete.
12. **Frontend**: polls `GET /claims/{id}/analysis/{analysis_id}` until `completed`, then fetches full detail/evidence/investigation.

### Failure at any stage
- CV failure on one image → that image marked `failed`, others continue; claim proceeds with partial CV data (flagged in UI).
- Document extraction failure → Section 4.3 handling; pipeline continues with remaining documents.
- Consistency/Anomaly/Risk stages are pure Python — failure here is a bug, not an expected runtime state; still wrapped in try/except that marks claim `status = "analysis_failed"` with an error log reference, never leaves claim stuck in `analyzing`.
- Gemini failure → Section 7.5 handling; pipeline still reaches `completed` with a null summary (summary is not on the pipeline's critical path — risk score is).

---

## 13. Reliability

| Failure | Handling |
|---|---|
| Invalid file type | Reject at upload endpoint (400), whitelist extensions/MIME types |
| Corrupted PDF | Caught in PyMuPDF open call → document marked `failed`, pipeline continues |
| Unsupported image format | Reject at upload (400) |
| Low-quality image | CV still runs; `low_confidence` flag set if applicable (Section 2.6) |
| Failed CNN inference | Try/except around inference call → image `Damage` marked `failed`, others unaffected |
| Low-confidence predictions | Surfaced explicitly, excluded from automatic score boosts, prompts manual review |
| OCR/extraction failure | Section 4.3 — retry once, then manual-entry fallback |
| Gemini failure | Section 7.5 — retry once, then null summary with retry button |
| Malformed LLM output | Repair-prompt retry once, then fallback (7.5) |
| Database failure | Transaction rollback per claim-analysis run; claim status reverts to pre-analysis state, error surfaced to UI with retry action |
| Duplicate submissions | Unique constraint on `(policy_id, incident_date, claim_number)`; duplicate `/claims` POST returns 409 with existing claim id |
| Partial analysis | Every stage's persistence is independent — a partially-completed claim always shows what succeeded, never silently discards results |

---

## 14. Project Structure

```
claimsight/
├── frontend/                # React + TypeScript app
│   ├── src/
│   │   ├── pages/           # Dashboard, ClaimsList, NewClaim, ClaimAnalysis, etc.
│   │   ├── components/      # shared UI (RiskBadge, EvidencePanel, ImageAnnotator...)
│   │   ├── api/              # typed API client
│   │   └── types/            # shared TS types mirroring backend Pydantic models
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── api/               # route modules per resource
│   │   ├── models/            # SQLAlchemy models
│   │   ├── schemas/           # Pydantic schemas
│   │   ├── services/          # pipeline.py, consistency.py, risk_engine.py, gemini_client.py
│   │   └── db/                 # session, migrations (alembic)
├── ml/
│   ├── training/               # train_cv_model.py, dataset prep
│   ├── inference/               # inference.py used by backend
│   └── data_card.md             # documents exact dataset used, class mapping
├── data/
│   └── synthetic/               # generated demo data + generator script output
├── scripts/
│   └── generate_demo_data.py
├── tests/
│   ├── backend/                 # unit + API tests
│   └── ml/                       # inference tests
├── docs/
│   └── CLAIMSIGHT_IMPLEMENTATION_PLAN.md   # this file
└── docker-compose.yml
```

---

## 15. Testing

**Priority order (MUST for first four):**

1. **Risk scoring** — unit tests per feature/weight combination, boundary tests for band thresholds (34/35, 64/65).
2. **Consistency rules** — one unit test per rule (R1–R9), each with a triggering and non-triggering `ClaimContext` fixture.
3. **Evidence mapping** — every generated `RiskSignal` in tests must have ≥1 `Evidence` row; assert no orphan signals.
4. **API behavior** — integration tests per endpoint (happy path + key error cases from Section 10 table).
5. Document extraction — schema validation tests with sample extracted JSON (valid + intentionally malformed).
6. ML inference — smoke test that inference pipeline returns well-formed output shape on a fixed test image.
7. Complete claim-analysis workflow — one end-to-end test per demo scenario (Section 3.3), asserting expected risk band.
8. Failure cases — corrupted PDF upload, Gemini timeout (mocked), duplicate claim submission.

---

## 17. Implementation Roadmap

| Phase | Build | Depends On | Files/Modules | Acceptance Criteria |
|---|---|---|---|---|
| 1 | Foundation — repo scaffold, FastAPI hello-world, React scaffold, DB connection | — | root structure, `docker-compose.yml` | backend and database connection work; frontend dev server runs |
| 2 | Database — all tables, migrations, seed script skeleton | Phase 1 | `backend/app/models`, `alembic/` | Migrations apply cleanly; tables match Section 9 |
| 3 | Backend core API — CRUD endpoints for customer/policy/claim/document/image upload | Phase 2 | `backend/app/api` | All non-analysis endpoints in Section 10 pass integration tests |
| 4 | CV pipeline — model training script, inference module | Phase 1 (independent of DB) | `ml/training`, `ml/inference` | Model trained, saved artifact loads and infers on a sample image |
| 5 | Document intelligence — PyMuPDF + Gemini extraction, Pydantic schemas | Phase 3 | `backend/app/services/document_intel.py` | Extraction returns valid schema on sample documents (Section 4) |
| 6 | Consistency engine — all 9 rules | Phase 3, 4, 5 | `backend/app/services/consistency.py` | Unit tests for R1–R9 pass |
| 7 | Risk engine — baseline cost calc, scoring, (NICE) Isolation Forest | Phase 6 | `backend/app/services/risk_engine.py` | Score/band computed correctly on demo scenarios |
| 8 | Gemini investigation layer | Phase 7 | `backend/app/services/gemini_client.py` | Summary generated, grounded, recommendation matches deterministic band |
| 9 | Frontend — all 9 screens, API client, types | Phase 3 (can start in parallel with 4–8 against mocked data) | `frontend/src` | All screens navigable, forms submit, results render |
| 10 | Evidence UI — annotated images, document viewer, evidence panel | Phase 8, 9 | `frontend/src/components` | Clicking a risk signal shows correct linked evidence |
| 11 | Integration — wire full `/analyze` pipeline end-to-end | Phases 4–10 | `backend/app/services/pipeline.py` | Full pipeline runs on a real upload set without manual intervention |
| 12 | Testing — full suite per Section 15 | Phase 11 | `tests/` | All MUST-priority tests pass in CI |
| 13 | Final polish — UX refinement, empty/error states, demo data walkthrough | Phase 12 | frontend + `scripts/generate_demo_data.py` | All 5 demo scenarios produce expected, presentable results |

---

## 18. Antigravity / Claude Code Prompt Sequence

**Prompt 1 — Foundation**
Implement: repo scaffold per Section 14, `docker-compose.yml` with backend+db services, FastAPI app returning `{"status": "ok"}` on `/health`, React app scaffold with routing stubs for the 9 screens.
Acceptance: `docker-compose up` succeeds; `/health` returns 200; frontend renders empty routed pages.

**Prompt 2 — Database**
Implement: SQLAlchemy models for all entities (Section 3.2/9), Alembic migration setup, initial migration.
Acceptance: `alembic upgrade head` creates all tables with correct FKs/indexes/constraints.

**Prompt 3 — Backend Core API**
Implement: Pydantic schemas + REST endpoints for customer, policy, claim, document upload, image upload (Section 10, excluding `/analyze` and downstream).
Acceptance: All listed endpoints pass integration tests including error cases.

**Prompt 4 — CV Pipeline**
Implement: dataset prep script (`ml/training/prepare_data.py`), training script per Section 2.2–2.5, inference module per Section 2.6, save trained weights + `data_card.md`.
Acceptance: Training completes, test-set macro-F1 reported and documented; inference module returns correctly-shaped output on a sample image.

**Prompt 5 — Document Intelligence**
Implement: PyMuPDF text/layout extraction, Gemini structured-extraction calls, Pydantic validation + retry logic (Section 4).
Acceptance: Given sample claim form/policy/estimate files, extraction returns valid schema instances; malformed-response retry path covered by test.

**Prompt 6 — Consistency Engine**
Implement: `ClaimContext` assembly + all 9 rules (Section 5) as pure functions.
Acceptance: Unit test per rule passes for both triggering and non-triggering fixtures.

**Prompt 7 — Risk Engine**
Implement: baseline cost range computation from synthetic data, feature normalization, weighted scoring, band assignment, contributing-factor breakdown (Section 6).
Acceptance: Scores for all 5 demo scenarios (Section 3.3) fall into their expected bands; boundary unit tests pass.

**Prompt 8 — Gemini Investigation Layer**
Implement: input assembly, prompt template, structured output parsing, grounding validation (rule_id cross-check), recommendation override logic, retry/failure handling (Section 7).
Acceptance: Summary generated for each demo scenario references only signals present in input; deterministic recommendation always wins over any LLM disagreement; failure path returns null summary without crashing pipeline.

**Prompt 9 — Frontend Screens**
Implement: all 9 screens (Section 11.1) wired to the API client, using mocked/real data as available; design direction per Section 11.2.
Acceptance: Full navigation flow works; forms validate and submit; loading/error/empty states implemented per screen.

**Prompt 10 — Evidence UI**
Implement: image annotation overlay, document viewer with field overlay, evidence side-panel linked from risk signals (Section 8, 11).
Acceptance: Each risk signal in the UI opens the correct evidence bundle (image region, document snippet, or computed breakdown).

**Prompt 11 — Integration**
Implement: `services/pipeline.py` orchestrating Sections 4–10 in sequence per Section 12, wired to `POST /claims/{id}/analyze` and status-polling endpoint.
Acceptance: Uploading a full demo scenario's inputs and calling `/analyze` produces a completed claim with correct risk score, signals, evidence, and (if Gemini available) summary — with zero manual steps.

**Prompt 12 — Testing**
Implement: full test suite per Section 15, wired into a CI config (GitHub Actions).
Acceptance: All MUST-priority test categories pass; CI green on a clean checkout.

**Prompt 13 — Final Polish**
Implement: UX refinement pass (loading states, empty states, error copy), run `generate_demo_data.py` to seed all 5 scenarios, verify presentable results across the full UI.
Acceptance: A fresh evaluator can open the deployed app, browse the 5 seeded demo claims, and see coherent, evidence-backed results for each without any setup steps.
