/**
 * Shared TypeScript types mirroring backend Pydantic models.
 *
 * Phase 9: shapes are tightened to match the actual backend responses.
 * The "stub" interfaces from earlier phases have been replaced with
 * real ones (e.g. InvestigationSummary now has key_concerns, disclaimer,
 * and model_version; ClaimSummary is the list-view schema).
 */

// ─── Health ────────────────────────────────────────────────────────────────

export interface HealthResponse {
  status: string;
}

// ─── Risk / Status enums ───────────────────────────────────────────────────

export type RiskBand = "Low" | "Medium" | "High";
export type ClaimStatus =
  | "pending"
  | "analyzing"
  | "completed"
  | "analysis_failed"
  | "decided";
export type DocType =
  | "claim_form"
  | "policy"
  | "estimate"
  | "invoice"
  | "previous_claim";
export type ExtractionStatus = "pending" | "completed" | "failed";
export type Recommendation = "normal" | "manual_review" | "investigate";
export type EvidenceType = "image" | "document" | "field" | "computed";
export type SignalSeverity = "low" | "medium" | "high";
export type Decision = "approve" | "deny" | "investigate" | "manual_review";

// ─── CV Analysis Types (unchanged from Phase 5) ───────────────────────────

export interface DamageTypeResult {
  label: string;
  confidence: number;
}

export interface SeverityResult {
  label: string;
  confidence: number;
}

export interface CVAnalysisResult {
  damage_id: number;
  claim_id: number;
  damage_types: DamageTypeResult[];
  severity: SeverityResult;
  low_confidence: boolean;
  source_image: string | null;
  model_version: string;
  timestamp: string | null;
  error: string | null;
}

export interface CVAnalysisBatchResult {
  claim_id: number;
  analyzed: number;
  results: CVAnalysisResult[];
}

// ─── Damage/Image Types (unchanged from Phase 5) ───────────────────────────

export interface DamageResponse {
  id: number;
  claim_id: number;
  source: string;
  damage_type: string | null;
  severity: string | null;
  confidence: number | null;
  region_ref: string | null;
}

// ─── Claim types (Phase 9, real backend shapes) ───────────────────────────

export interface ClaimSummary {
  id: number;
  claim_number: string;
  status: ClaimStatus;
  risk_band: RiskBand | null;
  risk_score: number | null;
  incident_date: string;
  created_at: string;
}

export interface Claim extends ClaimSummary {
  policy_id: number;
  vehicle_id: number;
  reported_date: string | null;
  claimed_amount: number | null;
  decision_notes: string | null;
}

export interface ClaimCreate {
  claim_number: string;
  policy_id: number;
  vehicle_id: number;
  incident_date: string;
  reported_date?: string | null;
  claimed_amount?: number | null;
}

// ─── Risk Signal + Evidence (Phase 9, real shape) ─────────────────────────

export interface RiskSignal {
  id: number;
  claim_id: number;
  rule_id: string;
  category: string;
  severity: SignalSeverity;
  description: string;
  created_at: string;
}

export interface Evidence {
  id: number;
  risk_signal_id: number;
  evidence_type: EvidenceType;
  reference: string | null;
  detail_json: Record<string, unknown> | null;
  created_at: string;
}

export interface RiskSignalWithEvidence extends RiskSignal {
  evidence: Evidence[];
}

// ─── Investigation (Phase 9) ───────────────────────────────────────────────

export interface InvestigationSummary {
  summary: string;
  key_concerns: string[];
  recommendation: Recommendation;
  disclaimer: string;
  model_version: string | null;
}

export interface InvestigationRecord {
  id: number;
  claim_id: number;
  summary_text: string | null;
  recommendation: Recommendation;
  model_version: string | null;
  generated_at: string | null;
  created_at: string;
}

// ─── Document (Phase 9) ────────────────────────────────────────────────────

export interface DocumentListItem {
  id: number;
  doc_type: DocType;
  extraction_status: ExtractionStatus;
  file_path: string;
}

export interface DocumentDetail extends DocumentListItem {
  claim_id: number;
  raw_confidence: number | null;
  extracted_fields: Record<string, unknown> | null;
  created_at: string;
}

// ─── Customer / Policy / Vehicle (Phase 9) ─────────────────────────────────

export interface CustomerSummary {
  id: number;
  name: string;
  email: string;
}

export interface CustomerCreate {
  name: string;
  email: string;
  phone?: string | null;
}

export interface PolicySummary {
  id: number;
  policy_number: string;
  coverage_type: string;
  status: string;
  end_date: string;
}

export interface PolicyCreate {
  customer_id: number;
  vehicle_id: number;
  policy_number: string;
  coverage_type:
    | "comprehensive"
    | "third_party"
    | "collision"
    | "fire_theft";
  coverage_limit: number;
  deductible: number;
  start_date: string;
  end_date: string;
  status?: "active" | "expired" | "cancelled";
}

export interface VehicleSummary {
  id: number;
  make: string;
  model: string;
  year: number;
  plate_number: string | null;
}

export interface VehicleCreate {
  customer_id: number;
  make: string;
  model: string;
  year: number;
  vin?: string | null;
  plate_number?: string | null;
}

// ─── Previous claims (existing endpoint, used by the audit/future phases) ──

export interface PreviousClaim {
  id: number;
  customer_id: number;
  vehicle_id: number | null;
  incident_date: string;
  damage_summary: string | null;
  payout_amount: number | null;
}

// ─── Decision (Phase 9) ────────────────────────────────────────────────────

export interface DecisionRequest {
  decision: Decision;
  notes?: string | null;
}

// ─── Analysis (Phase 11) ───────────────────────────────────────────────────

export type AnalysisStatus = "pending" | "running" | "completed" | "failed";

export interface AnalysisStartResponse {
  analysis_id: number;
  status: AnalysisStatus;
  claim_id: number;
}

export interface AnalysisResultSummary {
  risk_score: number | null;
  risk_band: RiskBand | null;
  signal_count: number;
  evidence_count: number;
  investigation_id: number | null;
}

export interface AnalysisStatusResponse {
  analysis_id: number;
  claim_id: number;
  status: AnalysisStatus;
  current_step: string | null;
  started_at: string;
  finished_at: string | null;
  error_message: string | null;
  claim_status: ClaimStatus;
  result: AnalysisResultSummary | null;
}
