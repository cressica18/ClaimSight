/**
 * Screen 2: Claims List
 *
 * Blueprint (Section 11.1):
 * - Filterable/sortable table by status, risk band, date
 *
 * Implementation: Phase 9. Backed by GET /claims (with status, risk_band,
 * skip, limit query params).
 */
import { useEffect, useMemo, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import PageShell from "../components/PageShell";
import EmptyState from "../components/EmptyState";
import ErrorState from "../components/ErrorState";
import LoadingState from "../components/LoadingState";
import { ClaimStatusPill, RiskBandPill } from "../components/StatusPill";
import { listClaims } from "../api/client";
import type { ClaimStatus, ClaimSummary, RiskBand } from "../types";
import sharedStyles from "../components/shared.module.css";

type SortKey = "newest" | "oldest";

const STATUS_FILTERS: { value: "all" | ClaimStatus; label: string }[] = [
  { value: "all", label: "All statuses" },
  { value: "pending", label: "Pending" },
  { value: "analyzing", label: "Analyzing" },
  { value: "completed", label: "Completed" },
  { value: "analysis_failed", label: "Analysis failed" },
  { value: "decided", label: "Decided" },
];

const RISK_FILTERS: { value: "all" | RiskBand; label: string }[] = [
  { value: "all", label: "All bands" },
  { value: "Low", label: "Low" },
  { value: "Medium", label: "Medium" },
  { value: "High", label: "High" },
];

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

const inputStyle: React.CSSProperties = {
  padding: "var(--space-2) var(--space-3)",
  borderRadius: "var(--radius-sm)",
  backgroundColor: "var(--color-surface-raised)",
  color: "var(--color-text-primary)",
  border: "1px solid var(--color-border)",
  fontSize: "var(--text-sm)",
  minWidth: "140px",
};

export default function ClaimsList() {
  const [searchParams, setSearchParams] = useSearchParams();
  const statusFilter = (searchParams.get("status") as ClaimStatus | "all" | null) ?? "all";
  const riskFilter = (searchParams.get("risk_band") as RiskBand | "all" | null) ?? "all";
  const sort = (searchParams.get("sort") as SortKey | null) ?? "newest";
  const query = searchParams.get("q") ?? "";

  const [claims, setClaims] = useState<ClaimSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = async () => {
    try {
      setError(null);
      const opts: { status?: ClaimStatus; riskBand?: RiskBand } = {};
      if (statusFilter !== "all") opts.status = statusFilter;
      if (riskFilter !== "all") opts.riskBand = riskFilter;
      const data = await listClaims(opts);
      setClaims(data);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Failed to load claims.";
      setError(message);
    }
  };

  useEffect(() => {
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [statusFilter, riskFilter]);

  const filtered = useMemo(() => {
    if (!claims) return [];
    let rows = claims;
    if (query.trim()) {
      const needle = query.trim().toLowerCase();
      rows = rows.filter((c) =>
        c.claim_number.toLowerCase().includes(needle)
      );
    }
    rows = [...rows].sort((a, b) => {
      const ta = new Date(a.created_at).getTime();
      const tb = new Date(b.created_at).getTime();
      return sort === "newest" ? tb - ta : ta - tb;
    });
    return rows;
  }, [claims, query, sort]);

  function updateParam(key: string, value: string) {
    const next = new URLSearchParams(searchParams);
    if (value === "" || value === "all" || value === "newest") {
      next.delete(key);
    } else {
      next.set(key, value);
    }
    setSearchParams(next, { replace: true });
  }

  return (
    <PageShell
      title="Claims"
      description="Filterable and sortable list of all claims by status, risk band, and date."
    >
      {/* Filter bar */}
      <div
        style={{
          display: "flex",
          gap: "var(--space-3)",
          flexWrap: "wrap",
          alignItems: "center",
          padding: "var(--space-3) var(--space-4)",
          backgroundColor: "var(--color-surface)",
          border: "1px solid var(--color-border)",
          borderRadius: "var(--radius-md)",
          marginBottom: "var(--space-4)",
        }}
      >
        <label
          style={{
            display: "flex",
            alignItems: "center",
            gap: "var(--space-2)",
            fontSize: "var(--text-xs)",
            color: "var(--color-text-muted)",
            textTransform: "uppercase",
            letterSpacing: "0.05em",
          }}
        >
          Status
          <select
            value={statusFilter}
            onChange={(e) => updateParam("status", e.target.value)}
            style={inputStyle}
            aria-label="Filter by status"
          >
            {STATUS_FILTERS.map((f) => (
              <option key={f.value} value={f.value}>
                {f.label}
              </option>
            ))}
          </select>
        </label>
        <label
          style={{
            display: "flex",
            alignItems: "center",
            gap: "var(--space-2)",
            fontSize: "var(--text-xs)",
            color: "var(--color-text-muted)",
            textTransform: "uppercase",
            letterSpacing: "0.05em",
          }}
        >
          Risk band
          <select
            value={riskFilter}
            onChange={(e) => updateParam("risk_band", e.target.value)}
            style={inputStyle}
            aria-label="Filter by risk band"
          >
            {RISK_FILTERS.map((f) => (
              <option key={f.value} value={f.value}>
                {f.label}
              </option>
            ))}
          </select>
        </label>
        <label
          style={{
            display: "flex",
            alignItems: "center",
            gap: "var(--space-2)",
            fontSize: "var(--text-xs)",
            color: "var(--color-text-muted)",
            textTransform: "uppercase",
            letterSpacing: "0.05em",
          }}
        >
          Sort
          <select
            value={sort}
            onChange={(e) => updateParam("sort", e.target.value)}
            style={inputStyle}
            aria-label="Sort by date"
          >
            <option value="newest">Newest first</option>
            <option value="oldest">Oldest first</option>
          </select>
        </label>
        <label
          style={{
            display: "flex",
            alignItems: "center",
            gap: "var(--space-2)",
            fontSize: "var(--text-xs)",
            color: "var(--color-text-muted)",
            textTransform: "uppercase",
            letterSpacing: "0.05em",
            flex: "1 1 200px",
          }}
        >
          Search
          <input
            type="search"
            value={query}
            onChange={(e) => updateParam("q", e.target.value)}
            placeholder="Claim number…"
            style={{ ...inputStyle, flex: 1 }}
            aria-label="Search by claim number"
          />
        </label>
        <Link to="/claims/new" className={sharedStyles.retryButton}>
          + New claim
        </Link>
      </div>

      {/* States */}
      {error && <ErrorState message={error} onRetry={load} />}
      {claims === null && !error && <LoadingState label="Loading claims…" />}
      {claims && filtered.length === 0 && !error && (
        <EmptyState
          title="No claims match your filters"
          description={
            claims.length === 0
              ? "No claims have been filed yet. Create the first one to get started."
              : "Try adjusting the status, risk-band, or search filters."
          }
          action={
            <Link to="/claims/new" className={sharedStyles.retryButton}>
              + Create claim
            </Link>
          }
        />
      )}

      {/* Table */}
      {filtered.length > 0 && (
        <div
          style={{
            overflowX: "auto",
            border: "1px solid var(--color-border)",
            borderRadius: "var(--radius-md)",
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
              <tr style={{ backgroundColor: "var(--color-surface)" }}>
                {["Claim #", "Status", "Risk band", "Incident date", "Created"].map(
                  (h) => (
                    <th
                      key={h}
                      style={{
                        textAlign: "left",
                        padding: "var(--space-3) var(--space-4)",
                        borderBottom: "1px solid var(--color-border)",
                        color: "var(--color-text-muted)",
                        fontSize: "var(--text-xs)",
                        textTransform: "uppercase",
                        letterSpacing: "0.05em",
                        fontWeight: 600,
                      }}
                    >
                      {h}
                    </th>
                  )
                )}
              </tr>
            </thead>
            <tbody>
              {filtered.map((c) => (
                <tr
                  key={c.id}
                  style={{
                    backgroundColor: "var(--color-bg)",
                    borderBottom: "1px solid var(--color-border)",
                    transition: "background-color var(--transition-base)",
                  }}
                >
                  <td style={{ padding: "var(--space-3) var(--space-4)" }}>
                    <Link
                      to={`/claims/${c.id}`}
                      style={{
                        color: "var(--color-accent)",
                        fontWeight: 500,
                      }}
                    >
                      {c.claim_number}
                    </Link>
                  </td>
                  <td style={{ padding: "var(--space-3) var(--space-4)" }}>
                    <ClaimStatusPill status={c.status} />
                  </td>
                  <td style={{ padding: "var(--space-3) var(--space-4)" }}>
                    <RiskBandPill band={c.risk_band} />
                  </td>
                  <td
                    style={{
                      padding: "var(--space-3) var(--space-4)",
                      color: "var(--color-text-secondary)",
                    }}
                  >
                    {formatDate(c.incident_date)}
                  </td>
                  <td
                    style={{
                      padding: "var(--space-3) var(--space-4)",
                      color: "var(--color-text-muted)",
                      fontSize: "var(--text-xs)",
                    }}
                  >
                    {formatDate(c.created_at)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </PageShell>
  );
}
