/**
 * Screen 8: Investigation Summary
 *
 * Blueprint (Section 11.1):
 * - Gemini narrative
 * - Key concerns list, each tied to a rule_id
 * - Recommendation badge
 * - AI disclaimer
 *
 * Phase 11: investigation generation is part of the run-analysis
 * pipeline. There is no separate "Run investigation" button — the
 * summary is produced (or not) when the user clicks "Start analysis"
 * on the Claim Analysis screen. This page reads the existing
 * `GET /claims/{id}/investigation` response and displays the
 * generated narrative. If the pipeline has not run, the page
 * shows an honest "Run analysis to generate an investigation
 * summary" message and a link to the Claim Analysis screen —
 * never a fake "Run investigation" button.
 *
 * Route: /claims/:id/investigation
 */
import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import PageShell from "../components/PageShell";
import EmptyState from "../components/EmptyState";
import ErrorState from "../components/ErrorState";
import LoadingState from "../components/LoadingState";
import Banner from "../components/Banner";
import { RecommendationPill } from "../components/StatusPill";
import { getInvestigation } from "../api/client";
import type { InvestigationSummary } from "../types";
import sharedStyles from "../components/shared.module.css";

const DISCLAIMER_FALLBACK = "AI-generated, human decision required";

// Extract the leading "[R<n>_<rule_id>]" token from a key_concern string,
// if present. The backend produces concerns in the form
// "[R4_excessive_repair_cost] Repair cost is 2.5x baseline.".
const RULE_ID_RE = /^\[(R\d+_[a-z_]+)\]\s*(.*)$/;

interface ParsedConcern {
  ruleId: string | null;
  text: string;
}

function parseConcern(raw: string): ParsedConcern {
  const m = RULE_ID_RE.exec(raw);
  if (m) {
    return { ruleId: m[1], text: m[2] };
  }
  return { ruleId: null, text: raw };
}

export default function InvestigationSummaryPage() {
  const { id } = useParams<{ id: string }>();
  const claimId = Number(id);

  const [data, setData] = useState<InvestigationSummary | null>(null);
  const [pending, setPending] = useState<boolean>(false);
  const [missing, setMissing] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const load = async () => {
    if (!claimId) return;
    setError(null);
    setMissing(false);
    setPending(false);
    try {
      const result = await getInvestigation(claimId);
      if (result === null) {
        // Distinguish "404 no row" from "202 pending" via the
        // response status; ApiError carries it.
        // We use a single helper that returns null for both; the
        // empty-state UI handles both.
        setMissing(true);
      } else {
        setData(result);
      }
    } catch (err) {
      if (err instanceof Error && /202/.test(err.message)) {
        setPending(true);
      } else {
        const message =
          err instanceof Error ? err.message : "Failed to load investigation.";
        setError(message);
      }
    }
  };

  useEffect(() => {
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [claimId]);

  if (!claimId) {
    return (
      <PageShell
        title="Investigation Summary"
        description="AI-generated investigation narrative and recommendation."
      >
        <EmptyState
          title="No claim selected"
          description="Navigate to a claim from the Claims list to view its investigation."
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
      <PageShell
        title={`Investigation Summary — Claim #${claimId}`}
        description="AI-generated investigation narrative and recommendation."
      >
        <ErrorState message={error} onRetry={load} />
      </PageShell>
    );
  }

  if (!data && !missing && !pending) {
    return (
      <PageShell
        title={`Investigation Summary — Claim #${claimId}`}
        description="AI-generated investigation narrative and recommendation."
      >
        <LoadingState label="Loading investigation…" />
      </PageShell>
    );
  }

  if (pending) {
    return (
      <PageShell
        title={`Investigation Summary — Claim #${claimId}`}
        description="AI-generated investigation narrative and recommendation."
      >
        <EmptyState
          title="Investigation pending"
          description="The analysis pipeline is still running. The investigation summary is generated as part of the analysis."
          action={
            <button
              type="button"
              onClick={load}
              className={sharedStyles.retryButton}
            >
              Refresh
            </button>
          }
        />
        <div
          style={{
            marginTop: "var(--space-4)",
            padding: "var(--space-3) var(--space-4)",
            backgroundColor: "var(--color-surface)",
            border: "1px solid var(--color-border)",
            borderRadius: "var(--radius-md)",
            fontSize: "var(--text-sm)",
            color: "var(--color-text-secondary)",
          }}
        >
          <strong>Generated as part of analysis.</strong> There is no
          separate &ldquo;run investigation&rdquo; action — the
          summary is written when the full pipeline finishes.
        </div>
      </PageShell>
    );
  }

  if (missing) {
    return (
      <PageShell
        title={`Investigation Summary — Claim #${claimId}`}
        description="AI-generated investigation narrative and recommendation."
      >
        <EmptyState
          title="No investigation yet"
          description="This claim has not been analyzed. Run the analysis pipeline to generate an investigation summary."
          action={
            <Link to={`/claims/${claimId}`} className={sharedStyles.retryButton}>
              Go to Claim Analysis
            </Link>
          }
        />
        <div
          style={{
            marginTop: "var(--space-4)",
            padding: "var(--space-3) var(--space-4)",
            backgroundColor: "var(--color-surface)",
            border: "1px solid var(--color-border)",
            borderRadius: "var(--radius-md)",
            fontSize: "var(--text-sm)",
            color: "var(--color-text-secondary)",
          }}
        >
          <strong>Generated as part of analysis.</strong> The
          investigation summary is written when the full pipeline
          finishes. Open the Claim Analysis screen and click
          &ldquo;Start analysis&rdquo; to generate it.
        </div>
      </PageShell>
    );
  }

  if (!data) return null; // unreachable, but keeps TS happy

  const disclaimer = data.disclaimer || DISCLAIMER_FALLBACK;
  const concerns = data.key_concerns.map(parseConcern);

  return (
    <PageShell
      title={`Investigation Summary — Claim #${claimId}`}
      description="AI-generated investigation narrative and recommendation. Human decision required."
    >
      <div style={{ display: "flex", flexDirection: "column", gap: "var(--space-6)" }}>
        {/* Recommendation + origin status */}
        <section
          aria-label="Recommendation"
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            gap: "var(--space-3)",
            padding: "var(--space-4)",
            backgroundColor: "var(--color-surface)",
            border: "1px solid var(--color-border)",
            borderRadius: "var(--radius-md)",
            flexWrap: "wrap",
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: "var(--space-3)" }}>
            <span
              style={{
                fontSize: "var(--text-xs)",
                color: "var(--color-text-muted)",
                textTransform: "uppercase",
                letterSpacing: "0.05em",
              }}
            >
              Recommendation
            </span>
            <RecommendationPill recommendation={data.recommendation} />
          </div>
          <span
            aria-label="Investigation origin"
            title="The investigation summary is produced by the analysis pipeline. There is no separate 'run investigation' action."
            style={{
              display: "inline-flex",
              alignItems: "center",
              gap: "var(--space-2)",
              padding: "var(--space-1) var(--space-3)",
              borderRadius: "var(--radius-sm)",
              backgroundColor: "var(--color-surface-raised)",
              border: "1px solid var(--color-border)",
              fontSize: "var(--text-xs)",
              color: "var(--color-text-muted)",
            }}
          >
            <span
              aria-hidden="true"
              style={{
                width: 6,
                height: 6,
                borderRadius: "50%",
                backgroundColor: "var(--color-text-muted)",
              }}
            />
            Generated as part of analysis
          </span>
          {data.model_version && (
            <span
              aria-label="Investigation model"
              title={
                data.model_version.toLowerCase().startsWith("demo")
                  ? "This summary was produced by the deterministic demo summariser, not the real Gemini model. The model is identified by its model_version string."
                  : "The model that produced this investigation summary."
              }
              style={{
                display: "inline-flex",
                alignItems: "center",
                gap: "var(--space-2)",
                padding: "var(--space-1) var(--space-3)",
                borderRadius: "var(--radius-sm)",
                backgroundColor: "var(--color-surface-raised)",
                border: "1px solid var(--color-border)",
                fontSize: "var(--text-xs)",
                color: "var(--color-text-muted)",
              }}
            >
              Model: {data.model_version}
            </span>
          )}
        </section>

        {/* Summary paragraph */}
        <section aria-label="Summary narrative">
          <h2
            style={{
              fontSize: "var(--text-lg)",
              margin: 0,
              marginBottom: "var(--space-3)",
              fontFamily: "var(--font-serif)",
              color: "var(--color-text-primary)",
            }}
          >
            Summary
          </h2>
          <p
            style={{
              fontSize: "var(--text-base)",
              lineHeight: 1.7,
              color: "var(--color-text-primary)",
              margin: 0,
              whiteSpace: "pre-wrap",
            }}
          >
            {data.summary}
          </p>
        </section>

        {/* Key concerns */}
        <section aria-label="Key concerns">
          <h2
            style={{
              fontSize: "var(--text-lg)",
              margin: 0,
              marginBottom: "var(--space-3)",
              fontFamily: "var(--font-serif)",
              color: "var(--color-text-primary)",
            }}
          >
            Key concerns
          </h2>
          {concerns.length === 0 ? (
            <p
              style={{
                fontSize: "var(--text-sm)",
                color: "var(--color-text-muted)",
                margin: 0,
              }}
            >
              No specific concerns were raised.
            </p>
          ) : (
            <ul
              style={{
                listStyle: "none",
                margin: 0,
                padding: 0,
                display: "flex",
                flexDirection: "column",
                gap: "var(--space-3)",
              }}
            >
              {concerns.map((c, idx) => (
                <li
                  key={`${c.ruleId ?? "concern"}-${idx}`}
                  style={{
                    display: "flex",
                    gap: "var(--space-3)",
                    padding: "var(--space-3) var(--space-4)",
                    backgroundColor: "var(--color-surface)",
                    border: "1px solid var(--color-border)",
                    borderRadius: "var(--radius-md)",
                  }}
                >
                  {c.ruleId ? (
                    <Link
                      to={`/claims/${claimId}/signals`}
                      style={{
                        flexShrink: 0,
                        textDecoration: "none",
                      }}
                      aria-label={`View signals for rule ${c.ruleId}`}
                    >
                      <code
                        style={{
                          display: "inline-block",
                          padding: "2px 8px",
                          borderRadius: "var(--radius-sm)",
                          backgroundColor: "var(--color-surface-raised)",
                          border: "1px solid var(--color-border)",
                          fontSize: "var(--text-xs)",
                          color: "var(--color-accent)",
                          fontFamily: "ui-monospace, SFMono-Regular, monospace",
                        }}
                      >
                        {c.ruleId}
                      </code>
                    </Link>
                  ) : (
                    <span
                      style={{
                        display: "inline-block",
                        padding: "2px 8px",
                        borderRadius: "var(--radius-sm)",
                        backgroundColor: "var(--color-surface-raised)",
                        border: "1px solid var(--color-border)",
                        fontSize: "var(--text-xs)",
                        color: "var(--color-text-muted)",
                        fontFamily: "ui-monospace, SFMono-Regular, monospace",
                      }}
                    >
                      (no rule id)
                    </span>
                  )}
                  <span
                    style={{
                      fontSize: "var(--text-sm)",
                      color: "var(--color-text-primary)",
                      lineHeight: 1.5,
                    }}
                  >
                    {c.text}
                  </span>
                </li>
              ))}
            </ul>
          )}
        </section>

        {/* Disclaimer banner — fixed text per blueprint 11.2 + Phase 8 rules */}
        <Banner tone="warning">
          <strong>AI-generated, human decision required.</strong> {disclaimer}
        </Banner>

        {/* CTA: route to the Decision Panel */}
        <div
          style={{
            display: "flex",
            justifyContent: "flex-end",
            gap: "var(--space-3)",
          }}
        >
          <Link to={`/claims/${claimId}/decision`} className={sharedStyles.retryButton}>
            Continue to decision →
          </Link>
        </div>
      </div>
    </PageShell>
  );
}
