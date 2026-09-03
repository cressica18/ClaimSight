/**
 * Screen 1: Dashboard
 *
 * Blueprint (Section 11.1) calls for "functional stat cards" — not
 * vanity metrics. The three cards here are clickable and each drives
 * the Claims List filtered appropriately.
 */
import { useEffect, useMemo, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import PageShell from "../components/PageShell";
import EmptyState from "../components/EmptyState";
import ErrorState from "../components/ErrorState";
import LoadingState from "../components/LoadingState";
import { listClaims } from "../api/client";
import type { ClaimStatus, ClaimSummary, RiskBand } from "../types";
import { ClaimStatusPill, RiskBandPill } from "../components/StatusPill";
import sharedStyles from "../components/shared.module.css";

const RELEVANT_STATUSES: ClaimStatus[] = ["pending", "analyzing"];

interface Stat {
  key: string;
  label: string;
  count: number;
  filter: { status?: ClaimStatus; riskBand?: RiskBand; statusIn?: ClaimStatus[] };
}

function formatDate(d: string | null | undefined): string {
  if (!d) return "—";
  try {
    return new Date(d).toLocaleDateString();
  } catch {
    return d;
  }
}

export default function Dashboard() {
  const navigate = useNavigate();
  const [claims, setClaims] = useState<ClaimSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = async () => {
    try {
      setError(null);
      // The default page size is 100; we want the counts to be honest,
      // so we ask for a generous limit.
      const data = await listClaims({ limit: 200 });
      setClaims(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load dashboard.");
    }
  };

  useEffect(() => {
    load();
  }, []);

  const stats: Stat[] = useMemo(() => {
    if (!claims) return [];
    const awaiting = claims.filter((c) => RELEVANT_STATUSES.includes(c.status)).length;
    const highRisk = claims.filter((c) => c.risk_band === "High").length;
    const decided = claims.filter((c) => c.status === "decided").length;
    return [
      {
        key: "awaiting",
        label: "Awaiting review",
        count: awaiting,
        filter: { statusIn: RELEVANT_STATUSES },
      },
      {
        key: "high",
        label: "High risk",
        count: highRisk,
        filter: { riskBand: "High" },
      },
      {
        key: "decided",
        label: "Decided",
        count: decided,
        filter: { status: "decided" },
      },
    ];
  }, [claims]);

  if (error) {
    return (
      <PageShell title="Dashboard" description="Claims awaiting your review.">
        <ErrorState message={error} onRetry={load} />
      </PageShell>
    );
  }

  if (claims === null) {
    return (
      <PageShell title="Dashboard" description="Claims awaiting your review.">
        <LoadingState label="Loading claims…" />
      </PageShell>
    );
  }

  // Build the URL for each stat card
  function urlFor(stat: Stat): string {
    const params = new URLSearchParams();
    if (stat.filter.status) params.set("status", stat.filter.status);
    if (stat.filter.riskBand) params.set("risk_band", stat.filter.riskBand);
    const qs = params.toString();
    return `/claims${qs ? `?${qs}` : ""}`;
  }

  // The "recent claims needing review" table is a curated subset of
  // pending + analyzing claims, ordered by recency, capped at 10.
  const needsReview = claims
    .filter((c) => RELEVANT_STATUSES.includes(c.status))
    .sort((a, b) => (a.created_at < b.created_at ? 1 : -1))
    .slice(0, 10);

  return (
    <PageShell
      title="Dashboard"
      description="Claims awaiting your review. Click a stat to see the underlying list."
    >
      <div style={{ display: "flex", flexDirection: "column", gap: "var(--space-6)" }}>
        {/* Stat cards */}
        <section
          aria-label="Summary statistics"
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))",
            gap: "var(--space-4)",
          }}
        >
          {stats.map((s) => (
            <button
              key={s.key}
              type="button"
              onClick={() => navigate(urlFor(s))}
              style={{
                padding: "var(--space-4) var(--space-5)",
                backgroundColor: "var(--color-surface)",
                border: "1px solid var(--color-border)",
                borderRadius: "var(--radius-md)",
                display: "flex",
                flexDirection: "column",
                gap: "var(--space-2)",
                cursor: "pointer",
                textAlign: "left",
                transition: "transform 160ms ease, border-color 160ms ease",
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.borderColor = "var(--color-accent)";
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.borderColor = "var(--color-border)";
              }}
            >
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "space-between",
                }}
              >
                <span style={{ fontSize: "var(--text-xs)", color: "var(--color-text-muted)", textTransform: "uppercase", letterSpacing: "0.05em" }}>
                  {s.label}
                </span>
              </div>
              <span
                style={{
                  fontSize: "2rem",
                  fontWeight: 600,
                  color: "var(--color-text-primary)",
                  fontFamily: "var(--font-serif)",
                }}
              >
                {s.count}
              </span>
              <span
                style={{ fontSize: "var(--text-xs)", color: "var(--color-accent)" }}
              >
                View list →
              </span>
            </button>
          ))}
        </section>

        {/* Recent claims needing review */}
        <section
          aria-label="Claims awaiting review"
          style={{
            display: "flex",
            flexDirection: "column",
            gap: "var(--space-3)",
          }}
        >
          <h2
            style={{
              fontSize: "var(--text-lg)",
              fontFamily: "var(--font-serif)",
              color: "var(--color-text-primary)",
              margin: 0,
            }}
          >
            Recent claims awaiting review
          </h2>
          {needsReview.length === 0 ? (
            <EmptyState
              title="No claims awaiting review"
              description="All caught up. New claims will appear here when they land in pending or analyzing."
              action={
                <Link to="/claims/new" className={sharedStyles.retryButton}>
                  + Create claim
                </Link>
              }
            />
          ) : (
            <div
              style={{
                backgroundColor: "var(--color-surface)",
                border: "1px solid var(--color-border)",
                borderRadius: "var(--radius-md)",
                overflowX: "auto",
              }}
            >
              <table
                style={{
                  width: "100%",
                  borderCollapse: "collapse",
                  fontSize: "var(--text-sm)",
                }}
              >
                <thead>
                  <tr>
                    {["Claim #", "Status", "Risk band", "Incident date", "Created"].map(
                      (h) => (
                        <th
                          key={h}
                          scope="col"
                          style={{
                            textAlign: "left",
                            padding: "var(--space-3) var(--space-4)",
                            color: "var(--color-text-muted)",
                            textTransform: "uppercase",
                            letterSpacing: "0.05em",
                            fontSize: "var(--text-xs)",
                            fontWeight: 500,
                            borderBottom: "1px solid var(--color-border)",
                          }}
                        >
                          {h}
                        </th>
                      )
                    )}
                  </tr>
                </thead>
                <tbody>
                  {needsReview.map((c) => (
                    <tr
                      key={c.id}
                      style={{ borderBottom: "1px solid var(--color-border)" }}
                    >
                      <td style={{ padding: "var(--space-3) var(--space-4)" }}>
                        <Link
                          to={`/claims/${c.id}`}
                          style={{ color: "var(--color-accent)", textDecoration: "none" }}
                        >
                          {c.claim_number}
                        </Link>
                      </td>
                      <td style={{ padding: "var(--space-3) var(--space-4)" }}>
                        <ClaimStatusPill status={c.status} />
                      </td>
                      <td style={{ padding: "var(--space-3) var(--space-4)" }}>
                        {c.risk_band ? <RiskBandPill band={c.risk_band} /> : "—"}
                      </td>
                      <td style={{ padding: "var(--space-3) var(--space-4)", color: "var(--color-text-secondary)" }}>
                        {formatDate(c.incident_date)}
                      </td>
                      <td style={{ padding: "var(--space-3) var(--space-4)", color: "var(--color-text-secondary)" }}>
                        {formatDate(c.created_at)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>
      </div>
    </PageShell>
  );
}
