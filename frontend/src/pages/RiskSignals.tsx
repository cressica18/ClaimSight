/**
 * Screen 7: Risk Signals
 *
 * Blueprint (Section 11.1):
 * - List of triggered rules, severity-coloured
 * - Expandable to evidence (evidence side-panel)
 *
 * Route: /claims/:id/signals
 *
 * Phase 9 implementation: the per-signal evidence was rendered as
 * a flat list of {type, reference, detail_json} rows. Phase 10
 * replaces that with the new `EvidenceViewer` component, which
 * renders image / document / field / computed evidence with
 * type-specific subrenderers and graceful fallbacks.
 *
 * Two ways to inspect evidence:
 *  1. Inline (default): click "Show evidence" on a card. The
 *     EvidenceViewer renders inside the card.
 *  2. Side panel: click "Open side panel" to dock a persistent
 *     right-hand panel for the active signal. Useful when the
 *     reviewer wants to compare evidence across signals.
 *
 * We deliberately avoid modal-only workflows: the evidence remains
 * visible alongside the rule/signal context.
 */
import { useEffect, useMemo, useState } from "react";
import { useParams, Link } from "react-router-dom";
import PageShell from "../components/PageShell";
import EmptyState from "../components/EmptyState";
import ErrorState from "../components/ErrorState";
import LoadingState from "../components/LoadingState";
import { SeverityChip } from "../components/SeverityChip";
import EvidenceViewer from "../components/EvidenceViewer";
import { getEvidence } from "../api/client";
import type { RiskSignalWithEvidence } from "../types";
import sharedStyles from "../components/shared.module.css";

const SEVERITY_RANK: Record<string, number> = {
  high: 0,
  medium: 1,
  low: 2,
};

function RuleIdBadge({ ruleId }: { ruleId: string }) {
  return (
    <code
      style={{
        display: "inline-block",
        padding: "2px 8px",
        borderRadius: "var(--radius-sm)",
        backgroundColor: "var(--color-surface-raised)",
        border: "1px solid var(--color-border)",
        fontSize: "var(--text-xs)",
        color: "var(--color-text-primary)",
        fontFamily: "ui-monospace, SFMono-Regular, monospace",
      }}
    >
      {ruleId}
    </code>
  );
}

interface SignalCardProps {
  signal: RiskSignalWithEvidence;
  isActive: boolean;
  claimId: number;
  onOpenSidePanel: () => void;
  onSelect: () => void;
}

function SignalCard({ signal, isActive, claimId, onOpenSidePanel, onSelect }: SignalCardProps) {
  const [open, setOpen] = useState(false);
  const hasEvidence = signal.evidence && signal.evidence.length > 0;

  return (
    <article
      onClick={onSelect}
      style={{
        backgroundColor: isActive ? "var(--color-surface-raised)" : "var(--color-surface)",
        border: isActive
          ? "1px solid var(--color-accent)"
          : "1px solid var(--color-border)",
        borderRadius: "var(--radius-md)",
        padding: "var(--space-4)",
        display: "flex",
        flexDirection: "column",
        gap: "var(--space-3)",
        cursor: "pointer",
        transition: "border-color 160ms ease, background-color 160ms ease",
      }}
    >
      <header
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          gap: "var(--space-3)",
          flexWrap: "wrap",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: "var(--space-3)", flexWrap: "wrap" }}>
          <RuleIdBadge ruleId={signal.rule_id} />
          <span style={{ fontSize: "var(--text-xs)", color: "var(--color-text-muted)" }}>
            {signal.category}
          </span>
        </div>
        <SeverityChip severity={signal.severity} />
      </header>

      <p
        style={{
          fontSize: "var(--text-sm)",
          color: "var(--color-text-primary)",
          lineHeight: 1.5,
          margin: 0,
        }}
      >
        {signal.description}
      </p>

      <div
        onClick={(e) => e.stopPropagation()}
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          gap: "var(--space-2)",
          borderTop: "1px solid var(--color-border)",
          paddingTop: "var(--space-2)",
        }}
      >
        <span style={{ fontSize: "var(--text-xs)", color: "var(--color-text-muted)" }}>
          {hasEvidence
            ? `${signal.evidence.length} evidence item${
                signal.evidence.length !== 1 ? "s" : ""
              } linked`
            : "No evidence items linked yet"}
        </span>
        {hasEvidence && (
          <div style={{ display: "flex", gap: "var(--space-2)" }}>
            <button
              type="button"
              onClick={() => setOpen((o) => !o)}
              aria-expanded={open}
              style={{
                padding: "var(--space-1) var(--space-3)",
                borderRadius: "var(--radius-sm)",
                border: "1px solid var(--color-border)",
                backgroundColor: "var(--color-surface-raised)",
                color: "var(--color-text-primary)",
                fontSize: "var(--text-xs)",
                cursor: "pointer",
              }}
            >
              {open ? "Hide evidence" : "Show evidence"}
            </button>
            <button
              type="button"
              onClick={onOpenSidePanel}
              style={{
                padding: "var(--space-1) var(--space-3)",
                borderRadius: "var(--radius-sm)",
                border: "1px solid var(--color-border)",
                backgroundColor: "transparent",
                color: "var(--color-accent)",
                fontSize: "var(--text-xs)",
                cursor: "pointer",
              }}
            >
              Open side panel →
            </button>
          </div>
        )}
      </div>

      {open && hasEvidence && (
        <div onClick={(e) => e.stopPropagation()}>
          <EvidenceViewer claimId={claimId} evidence={signal.evidence} inline />
        </div>
      )}
    </article>
  );
}

export default function RiskSignals() {
  const { id } = useParams<{ id: string }>();
  const claimId = Number(id);

  const [signals, setSignals] = useState<RiskSignalWithEvidence[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [activeSignalId, setActiveSignalId] = useState<number | null>(null);
  const [sidePanelOpen, setSidePanelOpen] = useState(false);

  const load = async () => {
    if (!claimId) return;
    try {
      setError(null);
      const data = await getEvidence(claimId);
      setSignals(data);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Failed to load risk signals.";
      setError(message);
    }
  };

  useEffect(() => {
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [claimId]);

  const sorted = useMemo(() => {
    if (!signals) return [];
    return [...signals].sort((a, b) => {
      const ra = SEVERITY_RANK[a.severity] ?? 99;
      const rb = SEVERITY_RANK[b.severity] ?? 99;
      if (ra !== rb) return ra - rb;
      return a.rule_id.localeCompare(b.rule_id);
    });
  }, [signals]);

  const activeSignal = useMemo(() => {
    if (!signals || activeSignalId === null) return null;
    return signals.find((s) => s.id === activeSignalId) ?? null;
  }, [signals, activeSignalId]);

  if (!claimId) {
    return (
      <PageShell title="Risk Signals" description="Select a claim to view risk signals.">
        <EmptyState
          title="No claim selected"
          description="Navigate to a claim from the Claims list to see its risk signals."
          action={
            <Link to="/claims" className={sharedStyles.retryButton}>
              Go to Claims
            </Link>
          }
        />
      </PageShell>
    );
  }

  return (
    <PageShell
      title={`Risk Signals — Claim #${claimId}`}
      description="Consistency rules and anomaly signals triggered for this claim. Click a signal to inspect its evidence."
    >
      {error && <ErrorState message={error} onRetry={load} />}
      {signals === null && !error && <LoadingState label="Loading risk signals…" />}
      {signals && signals.length === 0 && !error && (
        <EmptyState
          title="No risk signals"
          description="No consistency rules or anomaly signals have been generated for this claim yet."
        />
      )}
      {signals && signals.length > 0 && (
        <div
          style={{
            display: "grid",
            gridTemplateColumns: sidePanelOpen
              ? "minmax(0, 1fr) minmax(320px, 420px)"
              : "minmax(0, 1fr)",
            gap: "var(--space-4)",
            alignItems: "start",
          }}
        >
          <div style={{ display: "flex", flexDirection: "column", gap: "var(--space-3)" }}>
            <p
              style={{
                fontSize: "var(--text-sm)",
                color: "var(--color-text-secondary)",
                margin: 0,
              }}
            >
              {signals.length} signal{signals.length !== 1 ? "s" : ""} · ordered by severity
            </p>
            {sorted.map((signal) => (
              <SignalCard
                key={signal.id}
                signal={signal}
                isActive={signal.id === activeSignalId}
                claimId={claimId}
                onSelect={() => setActiveSignalId(signal.id)}
                onOpenSidePanel={() => {
                  setActiveSignalId(signal.id);
                  setSidePanelOpen(true);
                }}
              />
            ))}
          </div>

          {sidePanelOpen && (
            <aside
              aria-label="Evidence side panel"
              style={{
                position: "sticky",
                top: "var(--space-4)",
                padding: "var(--space-4)",
                backgroundColor: "var(--color-surface)",
                border: "1px solid var(--color-border)",
                borderRadius: "var(--radius-md)",
                display: "flex",
                flexDirection: "column",
                gap: "var(--space-3)",
                maxHeight: "calc(100vh - 120px)",
                overflowY: "auto",
              }}
            >
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "space-between",
                  gap: "var(--space-2)",
                  borderBottom: "1px solid var(--color-border)",
                  paddingBottom: "var(--space-2)",
                }}
              >
                <div
                  style={{
                    display: "flex",
                    flexDirection: "column",
                    gap: "var(--space-1)",
                  }}
                >
                  {activeSignal ? (
                    <>
                      <RuleIdBadge ruleId={activeSignal.rule_id} />
                      <span
                        style={{
                          fontSize: "var(--text-xs)",
                          color: "var(--color-text-muted)",
                        }}
                      >
                        {activeSignal.category}
                      </span>
                    </>
                  ) : (
                    <span
                      style={{
                        fontSize: "var(--text-sm)",
                        color: "var(--color-text-muted)",
                      }}
                    >
                      No signal selected
                    </span>
                  )}
                </div>
                <button
                  type="button"
                  onClick={() => setSidePanelOpen(false)}
                  aria-label="Close evidence side panel"
                  style={{
                    padding: "var(--space-1) var(--space-3)",
                    borderRadius: "var(--radius-sm)",
                    border: "1px solid var(--color-border)",
                    backgroundColor: "var(--color-surface-raised)",
                    color: "var(--color-text-primary)",
                    fontSize: "var(--text-xs)",
                    cursor: "pointer",
                  }}
                >
                  ×
                </button>
              </div>

              {activeSignal ? (
                <>
                  <SeverityChip severity={activeSignal.severity} />
                  <p
                    style={{
                      fontSize: "var(--text-sm)",
                      color: "var(--color-text-primary)",
                      lineHeight: 1.5,
                      margin: 0,
                    }}
                  >
                    {activeSignal.description}
                  </p>
                  <EvidenceViewer claimId={claimId} evidence={activeSignal.evidence} />
                </>
              ) : (
                <p
                  style={{
                    fontSize: "var(--text-sm)",
                    color: "var(--color-text-muted)",
                    margin: 0,
                  }}
                >
                  Click a signal on the left to load its evidence here.
                </p>
              )}
            </aside>
          )}
        </div>
      )}
    </PageShell>
  );
}
