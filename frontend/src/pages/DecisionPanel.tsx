/**
 * Screen 9: Decision Panel
 *
 * Blueprint (Section 11.1):
 * - Officer selects final action: approve / manual_review / investigate / deny
 * - Notes field
 * - Submit records decision via POST /claims/{id}/decision
 *
 * Implementation: Phase 9. Notes are persisted via the new
 * `decision_notes` column on the Claim model (Phase 9 backend gap).
 *
 * Route: /claims/:id/decision
 */
import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import PageShell from "../components/PageShell";
import ErrorState from "../components/ErrorState";
import LoadingState from "../components/LoadingState";
import EmptyState from "../components/EmptyState";
import Banner from "../components/Banner";
import {
  DecisionPill,
  RecommendationPill,
  RiskBandPill,
} from "../components/StatusPill";
import { getClaim, getInvestigation, recordDecision } from "../api/client";
import type { Claim, Decision, InvestigationSummary } from "../types";
import sharedStyles from "../components/shared.module.css";

const DECISION_OPTIONS: { value: Decision; label: string; description: string }[] = [
  {
    value: "approve",
    label: "Approve",
    description: "Claim is valid; process the payout as filed.",
  },
  {
    value: "manual_review",
    label: "Manual review",
    description: "Route to a human reviewer for closer inspection.",
  },
  {
    value: "investigate",
    label: "Investigate",
    description: "Escalate for fraud / SIU investigation.",
  },
  {
    value: "deny",
    label: "Deny",
    description: "Claim is rejected; no payout.",
  },
];

const inputStyle: React.CSSProperties = {
  width: "100%",
  padding: "var(--space-3) var(--space-4)",
  borderRadius: "var(--radius-md)",
  backgroundColor: "var(--color-surface-raised)",
  color: "var(--color-text-primary)",
  border: "1px solid var(--color-border)",
  fontSize: "var(--text-sm)",
  fontFamily: "inherit",
  resize: "vertical",
};

const buttonStyle: React.CSSProperties = {
  padding: "var(--space-3) var(--space-6)",
  borderRadius: "var(--radius-md)",
  backgroundColor: "var(--color-accent)",
  color: "#fff",
  border: "none",
  fontSize: "var(--text-sm)",
  fontWeight: 500,
  cursor: "pointer",
  transition: "background-color var(--transition-base)",
};

interface DecisionState {
  claim: Claim;
  investigation: InvestigationSummary | null;
}

export default function DecisionPanel() {
  const { id } = useParams<{ id: string }>();
  const claimId = Number(id);

  const [state, setState] = useState<DecisionState | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [decision, setDecision] = useState<Decision | "">("");
  const [notes, setNotes] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [submittedDecision, setSubmittedDecision] = useState<Decision | null>(null);

  const load = async () => {
    if (!claimId) return;
    try {
      setError(null);
      const [claim, investigation] = await Promise.all([
        getClaim(claimId),
        getInvestigation(claimId).catch(() => null),
      ]);
      setState({ claim, investigation });
    } catch (err) {
      const message = err instanceof Error ? err.message : "Failed to load claim.";
      setError(message);
    }
  };

  useEffect(() => {
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [claimId]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!decision) return;
    setSubmitting(true);
    setSubmitError(null);
    try {
      const updated = await recordDecision(claimId, decision, notes);
      setSubmittedDecision(decision);
      // Refresh local claim state to reflect server-side status change.
      setState((s) => (s ? { ...s, claim: updated } : s));
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Failed to record decision.";
      setSubmitError(message);
    } finally {
      setSubmitting(false);
    }
  };

  if (!claimId) {
    return (
      <PageShell title="Decision Panel" description="Officer decision panel.">
        <EmptyState
          title="No claim selected"
          description="Navigate to a claim from the Claims list to record a decision."
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
      <PageShell title={`Decision Panel — Claim #${claimId}`} description="Officer decision panel.">
        <ErrorState message={error} onRetry={load} />
      </PageShell>
    );
  }

  if (!state) {
    return (
      <PageShell title={`Decision Panel — Claim #${claimId}`} description="Officer decision panel.">
        <LoadingState label="Loading claim…" />
      </PageShell>
    );
  }

  const { claim, investigation } = state;
  const alreadyDecided = claim.status === "decided";
  const isReadOnly = alreadyDecided || submittedDecision !== null;

  return (
    <PageShell
      title={`Decision Panel — Claim #${claim.id}`}
      description={`Officer decision panel for claim ${claim.claim_number}.`}
    >
      <div style={{ display: "flex", flexDirection: "column", gap: "var(--space-6)" }}>
        {/* Context: what the AI said */}
        <section
          aria-label="Claim context"
          style={{
            padding: "var(--space-4)",
            backgroundColor: "var(--color-surface)",
            border: "1px solid var(--color-border)",
            borderRadius: "var(--radius-md)",
            display: "flex",
            flexDirection: "column",
            gap: "var(--space-3)",
          }}
        >
          <h2
            style={{
              fontSize: "var(--text-base)",
              margin: 0,
              fontFamily: "var(--font-serif)",
              color: "var(--color-text-primary)",
            }}
          >
            Claim context
          </h2>
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))",
              gap: "var(--space-3)",
            }}
          >
            <div>
              <div
                style={{
                  fontSize: "var(--text-xs)",
                  color: "var(--color-text-muted)",
                  textTransform: "uppercase",
                  letterSpacing: "0.05em",
                }}
              >
                Risk band
              </div>
              <div style={{ marginTop: "var(--space-1)" }}>
                <RiskBandPill band={claim.risk_band} />
              </div>
            </div>
            <div>
              <div
                style={{
                  fontSize: "var(--text-xs)",
                  color: "var(--color-text-muted)",
                  textTransform: "uppercase",
                  letterSpacing: "0.05em",
                }}
              >
                AI recommendation
              </div>
              <div style={{ marginTop: "var(--space-1)" }}>
                {investigation ? (
                  <RecommendationPill recommendation={investigation.recommendation} />
                ) : (
                  <span style={{ fontSize: "var(--text-sm)", color: "var(--color-text-muted)" }}>
                    Not available
                  </span>
                )}
              </div>
            </div>
          </div>
        </section>

        {/* Disclaimer — Phase 8 invariant */}
        <Banner tone="warning">
          The decision below is <strong>final</strong> and is the officer's, not the
          AI's. The recommendation above is informational. The claim is closed to
          further changes once a decision is submitted.
        </Banner>

        {/* Decision form */}
        <form
          onSubmit={handleSubmit}
          aria-label="Decision form"
          style={{ display: "flex", flexDirection: "column", gap: "var(--space-4)" }}
        >
          <fieldset
            disabled={isReadOnly || submitting}
            style={{
              border: "1px solid var(--color-border)",
              borderRadius: "var(--radius-md)",
              padding: "var(--space-4)",
              margin: 0,
              display: "flex",
              flexDirection: "column",
              gap: "var(--space-3)",
            }}
          >
            <legend
              style={{
                fontSize: "var(--text-xs)",
                color: "var(--color-text-muted)",
                textTransform: "uppercase",
                letterSpacing: "0.05em",
                padding: "0 var(--space-2)",
              }}
            >
              Final decision
            </legend>
            {DECISION_OPTIONS.map((opt) => (
              <label
                key={opt.value}
                style={{
                  display: "flex",
                  gap: "var(--space-3)",
                  padding: "var(--space-3) var(--space-4)",
                  backgroundColor:
                    decision === opt.value
                      ? "var(--color-surface-raised)"
                      : "var(--color-surface)",
                  border:
                    decision === opt.value
                      ? "1px solid var(--color-accent)"
                      : "1px solid var(--color-border)",
                  borderRadius: "var(--radius-sm)",
                  cursor: isReadOnly ? "default" : "pointer",
                }}
              >
                <input
                  type="radio"
                  name="decision"
                  value={opt.value}
                  checked={decision === opt.value}
                  onChange={() => setDecision(opt.value)}
                  disabled={isReadOnly}
                  style={{ marginTop: "3px" }}
                />
                <div>
                  <div
                    style={{
                      fontSize: "var(--text-sm)",
                      fontWeight: 500,
                      color: "var(--color-text-primary)",
                    }}
                  >
                    {opt.label}
                  </div>
                  <div
                    style={{
                      fontSize: "var(--text-xs)",
                      color: "var(--color-text-secondary)",
                    }}
                  >
                    {opt.description}
                  </div>
                </div>
              </label>
            ))}
          </fieldset>

          <div>
            <label
              htmlFor="decision-notes"
              style={{
                display: "block",
                fontSize: "var(--text-xs)",
                color: "var(--color-text-muted)",
                textTransform: "uppercase",
                letterSpacing: "0.05em",
                marginBottom: "var(--space-2)",
              }}
            >
              Officer notes (optional)
            </label>
            <textarea
              id="decision-notes"
              rows={4}
              value={notes}
              onChange={(e) => setNotes(e.target.value)}
              placeholder="Justify the decision or leave a comment for the audit log."
              disabled={isReadOnly}
              style={inputStyle}
            />
          </div>

          {submitError && (
            <div role="alert" style={{ color: "var(--color-risk-high)", fontSize: "var(--text-sm)" }}>
              {submitError}
            </div>
          )}

          {submittedDecision && (
            <div
              role="status"
              style={{
                padding: "var(--space-3) var(--space-4)",
                backgroundColor: "var(--color-risk-low-bg)",
                border: "1px solid var(--color-risk-low)",
                borderRadius: "var(--radius-md)",
                fontSize: "var(--text-sm)",
                color: "var(--color-risk-low)",
                display: "flex",
                alignItems: "center",
                gap: "var(--space-2)",
                flexWrap: "wrap",
              }}
            >
              <span>Decision recorded:</span>
              <DecisionPill decision={submittedDecision} />
              <span style={{ color: "var(--color-text-secondary)" }}>
                The claim is now closed to further changes.
              </span>
            </div>
          )}

          <div
            style={{
              display: "flex",
              justifyContent: "flex-end",
              gap: "var(--space-3)",
            }}
          >
            <Link
              to={`/claims/${claim.id}/investigation`}
              className={sharedStyles.retryButton}
            >
              ← Back to investigation
            </Link>
            <button
              type="submit"
              disabled={!decision || isReadOnly || submitting}
              style={{
                ...buttonStyle,
                opacity: !decision || isReadOnly || submitting ? 0.5 : 1,
                cursor:
                  !decision || isReadOnly || submitting
                    ? "not-allowed"
                    : "pointer",
              }}
            >
              {submitting ? "Submitting…" : alreadyDecided ? "Already decided" : "Submit decision"}
            </button>
          </div>
        </form>

        {/* If already decided on a previous page load, show the recorded notes */}
        {alreadyDecided && claim.decision_notes && (
          <section
            aria-label="Recorded notes"
            style={{
              padding: "var(--space-4)",
              backgroundColor: "var(--color-surface)",
              border: "1px solid var(--color-border)",
              borderRadius: "var(--radius-md)",
            }}
          >
            <h2
              style={{
                fontSize: "var(--text-base)",
                margin: 0,
                marginBottom: "var(--space-2)",
                fontFamily: "var(--font-serif)",
                color: "var(--color-text-primary)",
              }}
            >
              Recorded notes
            </h2>
            <p
              style={{
                margin: 0,
                fontSize: "var(--text-sm)",
                color: "var(--color-text-primary)",
                whiteSpace: "pre-wrap",
              }}
            >
              {claim.decision_notes}
            </p>
          </section>
        )}
      </div>
    </PageShell>
  );
}
