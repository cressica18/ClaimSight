/**
 * Screen 4: Claim Analysis
 *
 * Blueprint (Section 11.1):
 * - Pipeline progress (stage-by-stage status tracker)
 * - Results summary once analysis is complete
 *
 * Phase 11: the "Run full analysis" button is now wired to the real
 * pipeline. Click → POST /claims/{id}/analyze (202 + analysis_id) →
 * poll GET /claims/{id}/analysis/{id} every 2s. When the analysis
 * reaches a terminal state, the existing `load()` is re-run so the
 * stage tracker, risk band, signal/evidence/investigation panels all
 * reflect the new data.
 *
 * Route: /claims/:id
 */
import { useCallback, useEffect, useRef, useState } from "react";
import { Link, useParams } from "react-router-dom";
import PageShell from "../components/PageShell";
import EmptyState from "../components/EmptyState";
import ErrorState from "../components/ErrorState";
import LoadingState from "../components/LoadingState";
import StageTracker, { Stage, StageStatus } from "../components/StageTracker";
import { RiskBandPill, ClaimStatusPill } from "../components/StatusPill";
import {
  getClaim,
  getEvidence,
  listDocuments,
  listImages,
  getInvestigation,
  startAnalysis,
  getAnalysisStatus,
  getLatestAnalysis,
} from "../api/client";
import { ApiError } from "../api/client";
import type {
  AnalysisStatus,
  AnalysisStatusResponse,
  Claim,
  DamageResponse,
  DocumentListItem,
  InvestigationSummary,
  RiskSignalWithEvidence,
} from "../types";
import sharedStyles from "../components/shared.module.css";

const POLL_INTERVAL_MS = 2000;
type RunState = "idle" | "starting" | "polling" | "completed" | "failed";

interface ClaimAnalysisData {
  claim: Claim;
  signals: RiskSignalWithEvidence[];
  documents: DocumentListItem[];
  // `images` is the list of Damage rows with `source="image"`. The
  // CV stage is derived from this list (not from `signals`, which
  // are R1–R9 consistency-rule outputs). A row whose `damage_type`
  // is still `"pending"` indicates CV has not run on that image yet;
  // a row with a real damage type or `"cv_error"` indicates CV has
  // already produced a result for that image.
  images: DamageResponse[];
  investigation: InvestigationSummary | null;
}

function deriveStages(data: ClaimAnalysisData | null): Stage[] {
  if (!data) {
    return [
      { key: "upload", title: "Upload", description: "Claim and evidence uploaded.", status: "complete" },
      { key: "cv", title: "Image analysis", description: "CV damage detection on uploaded images.", status: "pending" },
      { key: "doc", title: "Document extraction", description: "Extract fields from uploaded documents.", status: "pending" },
      { key: "rules", title: "Consistency checks", description: "Run the 9 consistency rules over the claim context.", status: "pending" },
      { key: "risk", title: "Risk scoring + investigation", description: "Compute risk score, derive a recommendation, and (later) generate the AI narrative.", status: "pending" },
    ];
  }

  const claim = data.claim;

  // ── CV stage ────────────────────────────────────────────────────────
  // The source of truth for "did CV run?" is the persisted Damage
  // rows. A row is "pending" until cv_service fills in a damage
  // type; a row with damage_type="cv_error" means the model was
  // invoked and produced an error; a row with any other damage_type
  // means CV produced a real result.
  //
  // Note: the Phase 11 pipeline currently does NOT delete the
  // original "pending" row when it writes a new analyzed row, so
  // after a run completes a claim may have BOTH a leftover pending
  // row AND a new analyzed row. We treat that as "complete" because
  // the pipeline has terminated: re-deriving "complete" purely from
  // "no pending rows" would mis-classify finished claims. The
  // pipeline status (`claim.status`) is the authoritative signal —
  // if the claim is `completed` or `decided` and we have any image
  // rows, the CV stage has run.
  const images = data.images;
  const hasImages = images.length > 0;
  const pendingImages = images.filter((img) => img.damage_type === "pending");
  const failedImages = images.filter((img) => img.damage_type === "cv_error");
  const analyzedImages = images.filter(
    (img) =>
      img.damage_type !== null &&
      img.damage_type !== "pending" &&
      img.damage_type !== "cv_error"
  );
  const pipelineFinished =
    claim.status === "completed" ||
    claim.status === "decided" ||
    claim.status === "analysis_failed";
  const cvStatus: StageStatus = hasImages && (analyzedImages.length > 0 || failedImages.length > 0) && pipelineFinished
    ? "complete"
    : hasImages && pendingImages.length > 0 && claim.status === "analyzing"
    ? "running"
    : hasImages && analyzedImages.length > 0
    ? "complete"
    : hasImages
    ? "pending"
    : "pending";
  const cvComplete = cvStatus === "complete";

  // ── Document extraction stage ───────────────────────────────────────
  const hasDocuments = data.documents.length > 0;
  const pendingDocuments = data.documents.filter(
    (d) => d.extraction_status === "pending"
  );
  const failedDocuments = data.documents.filter(
    (d) => d.extraction_status === "failed"
  );
  const completedDocuments = data.documents.filter(
    (d) => d.extraction_status === "completed"
  );
  const docComplete =
    hasDocuments && pendingDocuments.length === 0;
  const docRunning =
    !docComplete && (claim.status === "analyzing" || claim.status === "completed");
  const docStatus: StageStatus = docComplete
    ? "complete"
    : docRunning
    ? "running"
    : "pending";

  // ── Rules + risk stages ─────────────────────────────────────────────
  const hasAnySignal = data.signals.length > 0;
  const ruleStatus: StageStatus = hasAnySignal ? "complete" : "pending";
  const hasRiskScoring = Boolean(claim.risk_band || data.investigation);
  const riskStatus: StageStatus = hasRiskScoring ? "complete" : "pending";

  // Build richer per-stage descriptions that surface the underlying
  // counts so the user can see at a glance why a stage is "complete"
  // vs. "partial". These descriptions replace the static strings the
  // page had when `data === null` (the initial render before any
  // fetches have resolved).
  let cvDescription: string;
  if (!hasImages) {
    cvDescription = "No images uploaded for this claim.";
  } else if (cvComplete) {
    const parts: string[] = [`${analyzedImages.length} analyzed`];
    if (failedImages.length > 0) {
      parts.push(`${failedImages.length} failed`);
    }
    cvDescription = `CV damage detection complete — ${parts.join(", ")}.`;
  } else if (pendingImages.length > 0) {
    cvDescription = `CV damage detection in progress — ${pendingImages.length} of ${images.length} image(s) remaining.`;
  } else {
    cvDescription = "CV damage detection on uploaded images.";
  }

  let docDescription: string;
  if (!hasDocuments) {
    docDescription = "No documents uploaded for this claim.";
  } else if (docComplete) {
    const parts: string[] = [`${completedDocuments.length} extracted`];
    if (failedDocuments.length > 0) {
      parts.push(`${failedDocuments.length} failed`);
    }
    docDescription = `Document extraction complete — ${parts.join(", ")}.`;
  } else if (pendingDocuments.length > 0) {
    docDescription = `Extracting fields — ${pendingDocuments.length} of ${data.documents.length} document(s) remaining.`;
  } else {
    docDescription = "Extract fields from uploaded documents.";
  }

  let rulesDescription: string;
  if (hasAnySignal) {
    rulesDescription = `Consistency checks complete — ${data.signals.length} signal(s) fired.`;
  } else if (claim.risk_band || data.investigation) {
    rulesDescription = "Consistency checks complete — 0 signals fired.";
  } else {
    rulesDescription = "Run the 9 consistency rules over the claim context.";
  }

  let riskDescription: string;
  if (claim.risk_band) {
    riskDescription = data.investigation
      ? `Risk scoring + investigation complete — band ${claim.risk_band}, score ${claim.risk_score?.toFixed(1) ?? "—"}.`
      : `Risk scoring complete — band ${claim.risk_band}, score ${claim.risk_score?.toFixed(1) ?? "—"}.`;
  } else {
    riskDescription = "Compute risk score, derive a recommendation, and (later) generate the AI narrative.";
  }

  return [
    {
      key: "upload",
      title: "Upload",
      description: "Claim and evidence uploaded.",
      status: "complete",
    },
    {
      key: "cv",
      title: "Image analysis",
      description: cvDescription,
      status: cvStatus,
    },
    {
      key: "doc",
      title: "Document extraction",
      description: docDescription,
      status: docStatus,
    },
    {
      key: "rules",
      title: "Consistency checks",
      description: rulesDescription,
      status: ruleStatus,
    },
    {
      key: "risk",
      title: "Risk scoring + investigation",
      description: riskDescription,
      status: riskStatus,
    },
  ];
}

function formatCurrency(n: number | null | undefined): string {
  if (n === null || n === undefined) return "—";
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 0,
  }).format(n);
}

function formatDate(iso: string | null | undefined): string {
  if (!iso) return "—";
  try {
    return new Date(iso).toLocaleDateString("en-US", {
      year: "numeric",
      month: "short",
      day: "numeric",
    });
  } catch {
    return iso;
  }
}

export default function ClaimAnalysis() {
  const { id } = useParams<{ id: string }>();
  const claimId = Number(id);

  const [data, setData] = useState<ClaimAnalysisData | null>(null);
  const [error, setError] = useState<string | null>(null);
  // Phase 11: state machine for the Run button + polling loop.
  const [runState, setRunState] = useState<RunState>("idle");
  const [analysisId, setAnalysisId] = useState<number | null>(null);
  const [analysisStatus, setAnalysisStatus] = useState<AnalysisStatus | null>(null);
  const [analysisError, setAnalysisError] = useState<string | null>(null);
  const pollTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const pollFailuresRef = useRef(0);

  const load = async () => {
    if (!claimId) return;
    try {
      setError(null);
      const [claim, signals, documents, images, investigation] = await Promise.all([
        getClaim(claimId),
        getEvidence(claimId).catch(() => [] as RiskSignalWithEvidence[]),
        listDocuments(claimId).catch(() => [] as DocumentListItem[]),
        listImages(claimId).catch(() => [] as DamageResponse[]),
        getInvestigation(claimId).catch(() => null),
      ]);
      setData({ claim, signals, documents, images, investigation });
    } catch (err) {
      const message = err instanceof Error ? err.message : "Failed to load claim.";
      setError(message);
    }
  };

  // Stop polling on unmount.
  useEffect(() => {
    return () => {
      if (pollTimerRef.current !== null) {
        clearTimeout(pollTimerRef.current);
      }
    };
  }, []);

  // On first load, if the claim is already `analyzing`, attach to
  // the in-flight run via /analysis/latest and resume polling.
  useEffect(() => {
    if (!claimId) return;
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [claimId]);

  // When the claim data first arrives and the claim is currently
  // `analyzing`, hook into the in-flight analysis so a page refresh
  // mid-run does not require the user to click Run again.
  useEffect(() => {
    if (!data || runState !== "idle") return;
    if (data.claim.status !== "analyzing") return;
    let cancelled = false;
    (async () => {
      try {
        const latest = await getLatestAnalysis(claimId);
        if (cancelled) return;
        if (latest.status === "running" || latest.status === "pending") {
          setAnalysisId(latest.analysis_id);
          setAnalysisStatus(latest.status);
          setRunState("polling");
          schedulePoll(latest.analysis_id);
        }
      } catch {
        // No analysis yet; the user can click Run.
      }
    })();
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data?.claim.status, data?.claim.id, runState]);

  // ─── Run-button handlers ───────────────────────────────────────────────

  const stopPolling = useCallback(() => {
    if (pollTimerRef.current !== null) {
      clearTimeout(pollTimerRef.current);
      pollTimerRef.current = null;
    }
  }, []);

  const handlePollResult = useCallback(
    async (aid: number) => {
      try {
        const status: AnalysisStatusResponse = await getAnalysisStatus(claimId, aid);
        pollFailuresRef.current = 0;
        setAnalysisStatus(status.status);
        if (status.status === "completed") {
          stopPolling();
          setRunState("completed");
          await load();
        } else if (status.status === "failed") {
          stopPolling();
          setAnalysisError(status.error_message || "Analysis failed.");
          setRunState("failed");
          await load();
        } else {
          schedulePoll(aid);
        }
      } catch {
        // Network blip: retry up to 3 times, then give up.
        pollFailuresRef.current += 1;
        if (pollFailuresRef.current >= 3) {
          stopPolling();
          setAnalysisError("Lost connection to the analysis. Refresh to check status.");
          setRunState("failed");
        } else {
          schedulePoll(aid);
        }
      }
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [claimId]
  );

  const schedulePoll = useCallback(
    (aid: number) => {
      stopPolling();
      pollTimerRef.current = setTimeout(() => {
        handlePollResult(aid);
      }, POLL_INTERVAL_MS);
    },
    [handlePollResult, stopPolling]
  );

  const handleRun = async () => {
    if (!claimId) return;
    setRunState("starting");
    setAnalysisError(null);
    try {
      const start = await startAnalysis(claimId);
      setAnalysisId(start.analysis_id);
      setAnalysisStatus(start.status);
      setRunState("polling");
      schedulePoll(start.analysis_id);
    } catch (err) {
      // 409 means a run is already in flight — attach to it.
      if (err instanceof ApiError && err.status === 409) {
        try {
          const latest = await getLatestAnalysis(claimId);
          setAnalysisId(latest.analysis_id);
          setAnalysisStatus(latest.status);
          setRunState("polling");
          schedulePoll(latest.analysis_id);
        } catch {
          setRunState("idle");
          setAnalysisError("An analysis is already running, but we could not attach to it. Refresh to try again.");
        }
        return;
      }
      const message = err instanceof Error ? err.message : "Failed to start analysis.";
      setAnalysisError(message);
      setRunState("failed");
    }
  };

  const handleReset = () => {
    setRunState("idle");
    setAnalysisId(null);
    setAnalysisStatus(null);
    setAnalysisError(null);
    pollFailuresRef.current = 0;
  };

  if (!claimId) {
    return (
      <PageShell title="Claim Analysis" description="Select a claim to view its analysis.">
        <EmptyState
          title="No claim selected"
          description="Navigate to a claim from the Claims list to see its analysis."
          action={
            <Link to="/claims" className={sharedStyles.retryButton}>
              Go to Claims
            </Link>
          }
        />
      </PageShell>
    );
  }

  if (error) {
    return (
      <PageShell title={`Claim Analysis — Claim #${claimId}`} description="Pipeline progress and analysis results.">
        <ErrorState message={error} onRetry={load} />
      </PageShell>
    );
  }

  if (!data) {
    return (
      <PageShell title={`Claim Analysis — Claim #${claimId}`} description="Pipeline progress and analysis results.">
        <LoadingState label="Loading claim analysis…" />
      </PageShell>
    );
  }

  const stages = deriveStages(data);

  return (
    <PageShell
      title={`Claim Analysis — Claim #${data.claim.id}`}
      description={`Claim ${data.claim.claim_number} — incident ${formatDate(data.claim.incident_date)}`}
    >
      <div style={{ display: "flex", flexDirection: "column", gap: "var(--space-6)" }}>
        {/* Summary panel */}
        <section
          aria-label="Claim summary"
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))",
            gap: "var(--space-4)",
            padding: "var(--space-4)",
            backgroundColor: "var(--color-surface)",
            border: "1px solid var(--color-border)",
            borderRadius: "var(--radius-md)",
          }}
        >
          <div>
            <div style={{ fontSize: "var(--text-xs)", color: "var(--color-text-muted)", textTransform: "uppercase", letterSpacing: "0.05em" }}>
              Status
            </div>
            <div style={{ marginTop: "var(--space-1)" }}>
              <ClaimStatusPill status={data.claim.status} />
            </div>
          </div>
          <div>
            <div style={{ fontSize: "var(--text-xs)", color: "var(--color-text-muted)", textTransform: "uppercase", letterSpacing: "0.05em" }}>
              Risk band
            </div>
            <div style={{ marginTop: "var(--space-1)" }}>
              <RiskBandPill band={data.claim.risk_band} />
            </div>
          </div>
          <div>
            <div style={{ fontSize: "var(--text-xs)", color: "var(--color-text-muted)", textTransform: "uppercase", letterSpacing: "0.05em" }}>
              Risk score
            </div>
            <div
              style={{
                marginTop: "var(--space-1)",
                fontSize: "var(--text-lg)",
                fontFamily: "var(--font-serif)",
                color: "var(--color-text-primary)",
              }}
            >
              {data.claim.risk_score !== null ? data.claim.risk_score.toFixed(1) : "—"}
            </div>
          </div>
          <div>
            <div style={{ fontSize: "var(--text-xs)", color: "var(--color-text-muted)", textTransform: "uppercase", letterSpacing: "0.05em" }}>
              Claimed amount
            </div>
            <div
              style={{
                marginTop: "var(--space-1)",
                fontSize: "var(--text-lg)",
                fontFamily: "var(--font-serif)",
                color: "var(--color-text-primary)",
              }}
            >
              {formatCurrency(data.claim.claimed_amount)}
            </div>
          </div>
          <div>
            <div style={{ fontSize: "var(--text-xs)", color: "var(--color-text-muted)", textTransform: "uppercase", letterSpacing: "0.05em" }}>
              Signals · Documents
            </div>
            <div
              style={{
                marginTop: "var(--space-1)",
                fontSize: "var(--text-lg)",
                fontFamily: "var(--font-serif)",
                color: "var(--color-text-primary)",
              }}
            >
              {data.signals.length} · {data.documents.length}
            </div>
          </div>
        </section>

        {/* Stage tracker */}
        <section aria-label="Pipeline stages">
          <h2
            style={{
              fontSize: "var(--text-lg)",
              margin: 0,
              marginBottom: "var(--space-3)",
              color: "var(--color-text-primary)",
              fontFamily: "var(--font-serif)",
            }}
          >
            Pipeline
          </h2>
          <StageTracker stages={stages} />
          <p
            style={{
              fontSize: "var(--text-xs)",
              color: "var(--color-text-muted)",
              marginTop: "var(--space-3)",
            }}
          >
            Stage status is derived from claim data. The full pipeline runs through
            image analysis, document extraction, consistency checks, risk scoring,
            and an AI summary.
          </p>
          <div style={{ display: "flex", alignItems: "center", gap: "var(--space-3)", marginTop: "var(--space-3)", flexWrap: "wrap" }}>
            <button
              type="button"
              onClick={handleRun}
              disabled={runState === "starting" || runState === "polling"}
              style={{
                padding: "var(--space-2) var(--space-4)",
                borderRadius: "var(--radius-md)",
                backgroundColor: runState === "starting" || runState === "polling"
                  ? "var(--color-surface-raised)"
                  : "var(--color-accent)",
                color: runState === "starting" || runState === "polling"
                  ? "var(--color-text-muted)"
                  : "white",
                border: "1px solid var(--color-border)",
                fontSize: "var(--text-sm)",
                fontWeight: 500,
                cursor: runState === "starting" || runState === "polling"
                  ? "wait"
                  : "pointer",
              }}
            >
              {runState === "starting"
                ? "Starting…"
                : runState === "polling"
                ? "Running…"
                : runState === "completed"
                ? "Re-run analysis"
                : runState === "failed"
                ? "Retry analysis"
                : "Start analysis"}
            </button>
            {runState === "polling" && analysisStatus && (
              <span
                aria-live="polite"
                style={{
                  fontSize: "var(--text-xs)",
                  color: "var(--color-text-muted)",
                }}
              >
                Status: {analysisStatus}
                {analysisId !== null && ` (analysis #${analysisId})`}
              </span>
            )}
            {runState === "failed" && analysisError && (
              <span
                role="alert"
                style={{
                  fontSize: "var(--text-xs)",
                  color: "var(--color-text-muted)",
                  maxWidth: "60ch",
                }}
              >
                {analysisError}
              </span>
            )}
            {(runState === "completed" || runState === "failed") && (
              <button
                type="button"
                onClick={handleReset}
                style={{
                  padding: "var(--space-1) var(--space-3)",
                  borderRadius: "var(--radius-sm)",
                  border: "1px solid var(--color-border)",
                  backgroundColor: "var(--color-surface-raised)",
                  color: "var(--color-text-muted)",
                  fontSize: "var(--text-xs)",
                  cursor: "pointer",
                }}
              >
                Dismiss
              </button>
            )}
          </div>
        </section>

        {/* Sub-screen links */}
        <section aria-label="Sub-screens">
          <h2
            style={{
              fontSize: "var(--text-lg)",
              margin: 0,
              marginBottom: "var(--space-3)",
              color: "var(--color-text-primary)",
              fontFamily: "var(--font-serif)",
            }}
          >
            View sub-screens
          </h2>
          <ul
            style={{
              listStyle: "none",
              padding: 0,
              margin: 0,
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))",
              gap: "var(--space-3)",
            }}
          >
            {[
              { to: "images", label: "Image analysis" },
              { to: "documents", label: "Documents" },
              { to: "signals", label: "Risk signals" },
              { to: "investigation", label: "Investigation summary" },
              { to: "decision", label: "Decision panel" },
            ].map((link) => (
              <li key={link.to}>
                <Link
                  to={link.to}
                  style={{
                    display: "block",
                    padding: "var(--space-3) var(--space-4)",
                    backgroundColor: "var(--color-surface)",
                    border: "1px solid var(--color-border)",
                    borderRadius: "var(--radius-md)",
                    color: "var(--color-text-primary)",
                    fontSize: "var(--text-sm)",
                    fontWeight: 500,
                    transition: "border-color var(--transition-base)",
                  }}
                >
                  {link.label} →
                </Link>
              </li>
            ))}
          </ul>
        </section>
      </div>
    </PageShell>
  );
}
