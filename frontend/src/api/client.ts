/**
 * Typed API client for the ClaimSight backend.
 *
 * Phase 9: the client now covers every endpoint the 9 screens need
 * (claims, evidence, investigation, decision, documents, lookup).
 * Phase 1 + Phase 5 helpers (health, images, CV analysis) are kept
 * unchanged.
 *
 * The client uses a Vite proxy in development (see vite.config.ts):
 *   /api/* → http://localhost:8000/*
 *
 * In production the VITE_API_BASE_URL env var overrides the base URL.
 */

import type {
  HealthResponse,
  DamageResponse,
  CVAnalysisResult,
  CVAnalysisBatchResult,
  Claim,
  ClaimSummary,
  ClaimCreate,
  ClaimStatus,
  RiskBand,
  RiskSignalWithEvidence,
  InvestigationSummary,
  DocumentListItem,
  DocumentDetail,
  Decision,
  DecisionRequest,
  CustomerSummary,
  CustomerCreate,
  PolicySummary,
  PolicyCreate,
  VehicleSummary,
  VehicleCreate,
  PreviousClaim,
  AnalysisStartResponse,
  AnalysisStatusResponse,
} from "../types";

const API_BASE =
  import.meta.env.VITE_API_BASE_URL !== undefined
    ? String(import.meta.env.VITE_API_BASE_URL)
    : "/api";

class ApiError extends Error {
  constructor(
    public readonly status: number,
    message: string
  ) {
    super(message);
    this.name = "ApiError";
  }
}

// Default request timeout. Without this, a hung request (e.g. backend
// not responding, network blip, proxy hang) leaves the caller in a
// permanent "loading" state with no way to surface an error to the
// user. 15s is generous for a local FastAPI on a healthy network.
const DEFAULT_TIMEOUT_MS = 15000;

export class TimeoutError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "TimeoutError";
  }
}

// Combine an external AbortSignal with a timeout so the fetch aborts
// when either fires first. Returns the combined signal + a cleanup
// function the caller must invoke to clear the timer and detach the
// listener (avoiding a stray setTimeout firing on a long-lived signal).
function withTimeout(
  externalSignal: AbortSignal | undefined,
  timeoutMs: number
): { signal: AbortSignal; cleanup: () => void; timedOut: () => boolean } {
  const controller = new AbortController();
  let timedOut = false;
  let timer: ReturnType<typeof setTimeout> | undefined;

  const onExternalAbort = () => {
    if (timer !== undefined) clearTimeout(timer);
    controller.abort(externalSignal?.reason);
  };

  timer = setTimeout(() => {
    timedOut = true;
    controller.abort(new DOMException("Request timed out", "TimeoutError"));
  }, timeoutMs);

  if (externalSignal) {
    if (externalSignal.aborted) {
      onExternalAbort();
    } else {
      externalSignal.addEventListener("abort", onExternalAbort, { once: true });
    }
  }

  return {
    signal: controller.signal,
    cleanup: () => {
      if (timer !== undefined) clearTimeout(timer);
      if (externalSignal) {
        externalSignal.removeEventListener("abort", onExternalAbort);
      }
    },
    timedOut: () => timedOut,
  };
}

interface RequestOptions extends Omit<RequestInit, "signal"> {
  signal?: AbortSignal;
  timeoutMs?: number;
}

async function request<T>(
  path: string,
  options: RequestOptions = {}
): Promise<T> {
  const { signal: externalSignal, timeoutMs = DEFAULT_TIMEOUT_MS, ...fetchOptions } = options;
  const url = `${API_BASE}${path}`;
  const { signal, cleanup, timedOut } = withTimeout(externalSignal, timeoutMs);

  let response: Response;
  try {
    response = await fetch(url, {
      headers: {
        "Content-Type": "application/json",
        ...fetchOptions.headers,
      },
      ...fetchOptions,
      signal,
    });
  } catch (err) {
    cleanup();
    if (timedOut()) {
      throw new TimeoutError(
        `Request to ${path} timed out after ${timeoutMs}ms`
      );
    }
    throw err;
  }
  cleanup();

  if (!response.ok) {
    const text = await response.text().catch(() => response.statusText);
    throw new ApiError(response.status, text);
  }

  // 202 Accepted may carry a body (POST /analyze returns 202 with the
  // analysis_id payload). Try to JSON-decode; if the body is empty
  // (e.g. an upstream 202 with no payload), fall through to undefined.
  if (response.status === 202) {
    const text = await response.text();
    if (!text) {
      return undefined as T;
    }
    try {
      return JSON.parse(text) as T;
    } catch {
      return undefined as T;
    }
  }

  return response.json() as Promise<T>;
}

// ─── Health ────────────────────────────────────────────────────────────────

export async function getHealth(): Promise<HealthResponse> {
  return request<HealthResponse>("/health");
}

// ─── Images / CV Analysis (Phase 5, unchanged) ────────────────────────────

export async function listImages(claimId: number): Promise<DamageResponse[]> {
  const response = await fetch(`${API_BASE}/claims/${claimId}/images`);
  if (!response.ok) {
    const text = await response.text().catch(() => response.statusText);
    throw new ApiError(response.status, text);
  }
  return response.json() as Promise<DamageResponse[]>;
}

export async function uploadImages(
  claimId: number,
  files: File[]
): Promise<DamageResponse[]> {
  const formData = new FormData();
  for (const file of files) {
    formData.append("files", file);
  }

  const response = await fetch(`${API_BASE}/claims/${claimId}/images`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const text = await response.text().catch(() => response.statusText);
    throw new ApiError(response.status, text);
  }

  return response.json() as Promise<DamageResponse[]>;
}

export async function analyzeImage(
  claimId: number,
  damageId: number
): Promise<CVAnalysisResult> {
  const response = await fetch(
    `${API_BASE}/claims/${claimId}/damages/${damageId}/analyze`,
    { method: "POST" }
  );

  if (!response.ok) {
    const text = await response.text().catch(() => response.statusText);
    throw new ApiError(response.status, text);
  }

  return response.json() as Promise<CVAnalysisResult>;
}

export async function analyzeAllImages(
  claimId: number
): Promise<CVAnalysisBatchResult> {
  const response = await fetch(
    `${API_BASE}/claims/${claimId}/analyze-images`,
    { method: "POST" }
  );

  if (!response.ok) {
    const text = await response.text().catch(() => response.statusText);
    throw new ApiError(response.status, text);
  }

  return response.json() as Promise<CVAnalysisBatchResult>;
}

// ─── Claims (Phase 9) ─────────────────────────────────────────────────────

export interface ListClaimsOptions {
  status?: ClaimStatus;
  riskBand?: RiskBand;
  skip?: number;
  limit?: number;
}

export async function listClaims(
  opts: ListClaimsOptions = {}
): Promise<ClaimSummary[]> {
  const params = new URLSearchParams();
  if (opts.status) params.set("status", opts.status);
  if (opts.riskBand) params.set("risk_band", opts.riskBand);
  if (typeof opts.skip === "number") params.set("skip", String(opts.skip));
  if (typeof opts.limit === "number") params.set("limit", String(opts.limit));
  const qs = params.toString();
  return request<ClaimSummary[]>(`/claims${qs ? `?${qs}` : ""}`);
}

export async function getClaim(claimId: number): Promise<Claim> {
  return request<Claim>(`/claims/${claimId}`);
}

export async function createClaim(payload: ClaimCreate): Promise<Claim> {
  return request<Claim>("/claims", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function getPreviousClaims(
  claimId: number
): Promise<PreviousClaim[]> {
  return request<PreviousClaim[]>(`/claims/${claimId}/previous-claims`);
}

export async function getEvidence(
  claimId: number
): Promise<RiskSignalWithEvidence[]> {
  return request<RiskSignalWithEvidence[]>(`/claims/${claimId}/evidence`);
}

/**
 * Fetch the investigation summary for a claim.
 *
 * Returns `null` if the backend returns 404 (no investigation row)
 * or 202 (investigation pending). Both states are valid display
 * states for the Investigation Summary screen and the caller can
 * handle them by passing the `null` to the empty-state UI.
 */
export async function getInvestigation(
  claimId: number
): Promise<InvestigationSummary | null> {
  const url = `${API_BASE}/claims/${claimId}/investigation`;
  const response = await fetch(url);
  if (response.status === 404 || response.status === 202) {
    return null;
  }
  if (!response.ok) {
    const text = await response.text().catch(() => response.statusText);
    throw new ApiError(response.status, text);
  }
  return response.json() as Promise<InvestigationSummary>;
}

export async function recordDecision(
  claimId: number,
  decision: Decision,
  notes?: string
): Promise<Claim> {
  const payload: DecisionRequest = { decision };
  if (notes !== undefined && notes !== null && notes !== "") {
    payload.notes = notes;
  }
  return request<Claim>(`/claims/${claimId}/decision`, {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

// ─── Documents (Phase 9) ──────────────────────────────────────────────────

export async function listDocuments(
  claimId: number,
  options?: { signal?: AbortSignal; timeoutMs?: number }
): Promise<DocumentListItem[]> {
  return request<DocumentListItem[]>(`/claims/${claimId}/documents`, options);
}

export async function getDocument(
  claimId: number,
  documentId: number,
  options?: { signal?: AbortSignal; timeoutMs?: number }
): Promise<DocumentDetail> {
  return request<DocumentDetail>(
    `/claims/${claimId}/documents/${documentId}`,
    options
  );
}

export async function uploadDocument(
  claimId: number,
  file: File,
  docType: string
): Promise<DocumentDetail> {
  const formData = new FormData();
  formData.append("doc_type", docType);
  formData.append("file", file);

  const response = await fetch(`${API_BASE}/claims/${claimId}/documents`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const text = await response.text().catch(() => response.statusText);
    throw new ApiError(response.status, text);
  }

  return response.json() as Promise<DocumentDetail>;
}

// ─── Customer / Policy / Vehicle lookup (Phase 9) ─────────────────────────

export async function listCustomers(
  opts: { skip?: number; limit?: number } = {},
  options?: { signal?: AbortSignal; timeoutMs?: number }
): Promise<CustomerSummary[]> {
  const params = new URLSearchParams();
  if (typeof opts.skip === "number") params.set("skip", String(opts.skip));
  if (typeof opts.limit === "number") params.set("limit", String(opts.limit));
  const qs = params.toString();
  return request<CustomerSummary[]>(`/customers${qs ? `?${qs}` : ""}`, options);
}

export async function createCustomer(
  payload: CustomerCreate
): Promise<CustomerSummary & { id: number; phone: string | null }> {
  return request("/customers", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function listPolicies(
  opts: { customerId?: number; skip?: number; limit?: number } = {},
  options?: { signal?: AbortSignal; timeoutMs?: number }
): Promise<PolicySummary[]> {
  const params = new URLSearchParams();
  if (typeof opts.customerId === "number") {
    params.set("customer_id", String(opts.customerId));
  }
  if (typeof opts.skip === "number") params.set("skip", String(opts.skip));
  if (typeof opts.limit === "number") params.set("limit", String(opts.limit));
  const qs = params.toString();
  return request<PolicySummary[]>(`/policies${qs ? `?${qs}` : ""}`, options);
}

export async function createPolicy(
  payload: PolicyCreate
): Promise<PolicySummary & { id: number; customer_id: number; vehicle_id: number }> {
  return request("/policies", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function listVehicles(
  opts: { customerId?: number; skip?: number; limit?: number } = {},
  options?: { signal?: AbortSignal; timeoutMs?: number }
): Promise<VehicleSummary[]> {
  const params = new URLSearchParams();
  if (typeof opts.customerId === "number") {
    params.set("customer_id", String(opts.customerId));
  }
  if (typeof opts.skip === "number") params.set("skip", String(opts.skip));
  if (typeof opts.limit === "number") params.set("limit", String(opts.limit));
  const qs = params.toString();
  return request<VehicleSummary[]>(`/vehicles${qs ? `?${qs}` : ""}`, options);
}

export async function createVehicle(
  payload: VehicleCreate
): Promise<VehicleSummary & { id: number; customer_id: number }> {
  return request("/vehicles", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

// ─── Analysis (Phase 11) ───────────────────────────────────────────────────

/**
 * Start a full analysis run for a claim.
 *
 * Returns 202 + the analysis_id. The frontend then polls
 * `getAnalysisStatus` to track the run. If the response status is
 * 409 (already running), the caller should fetch the existing
 * analysis via the latest endpoint and start polling that one.
 */
export async function startAnalysis(
  claimId: number
): Promise<AnalysisStartResponse> {
  return request<AnalysisStartResponse>(`/claims/${claimId}/analyze`, {
    method: "POST",
  });
}

/**
 * Poll the status of an analysis run. Use every 2 seconds while the
 * run is `running`. When status becomes `completed` or `failed`, the
 * caller stops polling and refreshes the rest of the claim data.
 */
export async function getAnalysisStatus(
  claimId: number,
  analysisId: number
): Promise<AnalysisStatusResponse> {
  return request<AnalysisStatusResponse>(
    `/claims/${claimId}/analysis/${analysisId}`
  );
}

/**
 * Fetch the most recent analysis (any status) for a claim. Used by
 * the frontend when it lands on a claim whose status is already
 * "analyzing" (e.g. after a page refresh mid-run), so the UI can
 * immediately resume polling without a second POST.
 */
export async function getLatestAnalysis(
  claimId: number
): Promise<AnalysisStatusResponse> {
  return request<AnalysisStatusResponse>(
    `/claims/${claimId}/analysis/latest`
  );
}

export { ApiError };
