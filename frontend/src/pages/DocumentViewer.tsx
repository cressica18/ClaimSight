/**
 * Screen 6: Document / Evidence Viewer
 *
 * Blueprint (Section 11.1):
 * - Tabbed document previews with extracted-field overlay
 * - Confidence indicators per field
 *
 * Phase 9: a tab strip + metadata panel + extracted-fields side panel
 * (when fields are populated). PDF-canvas overlay is a Phase 10 task
 * (Section 17 row 10).
 *
 * Route: /claims/:id/documents
 */
import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import PageShell from "../components/PageShell";
import EmptyState from "../components/EmptyState";
import ErrorState from "../components/ErrorState";
import LoadingState from "../components/LoadingState";
import { listDocuments, uploadDocument, getDocument } from "../api/client";
import type { DocumentDetail, DocumentListItem, DocType, ExtractionStatus } from "../types";
import sharedStyles from "../components/shared.module.css";

const DOC_TYPES: DocType[] = [
  "claim_form",
  "policy",
  "estimate",
  "invoice",
  "previous_claim",
];

const DOC_TYPE_LABEL: Record<DocType, string> = {
  claim_form: "Claim form",
  policy: "Policy",
  estimate: "Estimate",
  invoice: "Invoice",
  previous_claim: "Previous claim",
};

const EXTRACTION_STATUS_LABEL: Record<ExtractionStatus, string> = {
  pending: "Pending",
  completed: "Completed",
  failed: "Failed",
};

function formatConfidence(c: number | null | undefined): string {
  if (c === null || c === undefined) return "—";
  return `${Math.round(c * 100)}%`;
}

function confidenceTone(c: number | null | undefined): string {
  if (c === null || c === undefined) return "var(--color-text-muted)";
  if (c >= 0.7) return "var(--color-risk-low)";
  if (c >= 0.4) return "var(--color-risk-medium)";
  return "var(--color-risk-high)";
}

const inputStyle: React.CSSProperties = {
  padding: "var(--space-2) var(--space-3)",
  borderRadius: "var(--radius-sm)",
  backgroundColor: "var(--color-surface-raised)",
  color: "var(--color-text-primary)",
  border: "1px solid var(--color-border)",
  fontSize: "var(--text-sm)",
};

const buttonStyle: React.CSSProperties = {
  padding: "var(--space-2) var(--space-4)",
  borderRadius: "var(--radius-md)",
  backgroundColor: "var(--color-accent)",
  color: "#fff",
  border: "none",
  fontSize: "var(--text-sm)",
  fontWeight: 500,
  cursor: "pointer",
};

function ExtractedFields({
  fields,
  rawConfidence,
}: {
  fields: Record<string, unknown> | null | undefined;
  rawConfidence: number | null | undefined;
}) {
  // "Visible" = the count of fields the UI would actually render.
  // Internal keys (leading underscore) are filtered out at the render
  // boundary, so a payload of only internal keys should also show the
  // honest empty state rather than an empty list of cards.
  const visibleEntries = fields
    ? Object.entries(fields).filter(([key]) => !key.startsWith("_"))
    : [];
  if (!fields || visibleEntries.length === 0) {
    return (
      <div
        style={{
          padding: "var(--space-3) var(--space-4)",
          backgroundColor: "var(--color-surface-raised)",
          border: "1px dashed var(--color-border)",
          borderRadius: "var(--radius-md)",
          fontSize: "var(--text-sm)",
          color: "var(--color-text-muted)",
        }}
      >
        No structured fields were extracted from this document. The
        extraction is heuristic — only fields that can be derived
        from the file (e.g. a policy number encoded in a policy
        document's filename) are surfaced.
      </div>
    );
  }
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "var(--space-2)" }}>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          fontSize: "var(--text-xs)",
          color: "var(--color-text-muted)",
          textTransform: "uppercase",
          letterSpacing: "0.05em",
        }}
      >
        <span>Extracted fields</span>
        <span>
          Confidence:{" "}
          <span style={{ color: confidenceTone(rawConfidence), fontWeight: 500 }}>
            {formatConfidence(rawConfidence)}
          </span>
        </span>
      </div>
      <ul
        style={{
          listStyle: "none",
          margin: 0,
          padding: 0,
          display: "grid",
          gridTemplateColumns: "1fr",
          gap: "var(--space-2)",
        }}
      >
        {/* Filter out internal/debug keys (anything starting with `_`)
            at the UI boundary as defense-in-depth — the backend is
            responsible for not emitting them, but the UI should never
            display a leading-underscore field to the user regardless. */}
        {visibleEntries.map(([key, value]) => (
          <li
            key={key}
            style={{
              display: "grid",
              gridTemplateColumns: "minmax(140px, 1fr) 2fr",
              gap: "var(--space-3)",
              padding: "var(--space-2) var(--space-3)",
              backgroundColor: "var(--color-surface)",
              border: "1px solid var(--color-border)",
              borderRadius: "var(--radius-sm)",
              fontSize: "var(--text-sm)",
            }}
          >
            <span style={{ color: "var(--color-text-muted)" }}>{key}</span>
            <span
              style={{
                color: "var(--color-text-primary)",
                wordBreak: "break-word",
                fontFamily: "ui-monospace, SFMono-Regular, monospace",
                fontSize: "var(--text-xs)",
              }}
            >
              {typeof value === "object" && value !== null
                ? JSON.stringify(value)
                : String(value)}
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}

export default function DocumentViewer() {
  const { id } = useParams<{ id: string }>();
  const claimId = Number(id);

  const [docs, setDocs] = useState<DocumentListItem[] | null>(null);
  const [active, setActive] = useState<number | null>(null);
  const [activeDetail, setActiveDetail] = useState<DocumentDetail | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Upload state
  const [docType, setDocType] = useState<DocType>("claim_form");
  const [pendingFile, setPendingFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);

  const load = async (signal?: AbortSignal) => {
    if (!claimId) return;
    try {
      setError(null);
      const data = await listDocuments(claimId, { signal });
      if (signal?.aborted) return;
      setDocs(data);
      // Use the functional setter so we always see the current `active`
      // value at the time the state is committed, not the value at the
      // time `load` was called. This avoids a stale closure that would
      // fail to auto-select the first doc on a new claim.
      setActive((currentActive) => {
        if (data.length > 0 && currentActive === null) {
          return data[0].id;
        }
        return currentActive;
      });
    } catch (err) {
      if (signal?.aborted) return;
      const message =
        err instanceof Error ? err.message : "Failed to load documents.";
      setError(message);
    }
  };

  // Load documents when the claim changes. We use an AbortController so
  // navigating between claims cancels the in-flight request (prevents
  // a slow response for the old claim from clobbering the new one) and
  // so the underlying fetch is actually cancelled, not just orphaned.
  // The API client also enforces a default timeout, so a request that
  // hangs forever surfaces as a timeout error and we leave the loading
  // state with a clear message + retry rather than spinning forever.
  useEffect(() => {
    if (!claimId) return;
    const controller = new AbortController();
    // Reset stale state from a previous claim so we never briefly show
    // the previous claim's documents/selection while the new request
    // is in flight.
    setActive(null);
    setError(null);
    setDocs(null);
    load(controller.signal);
    return () => {
      controller.abort();
    };
    // We intentionally only re-run when claimId changes; `load` uses
    // a functional setState updater for `active` so it always sees the
    // current value at commit time. This avoids the user's tab
    // selection being clobbered on a re-render.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [claimId]);

  useEffect(() => {
    if (active === null) {
      setActiveDetail(null);
      return;
    }
    const controller = new AbortController();
    let cancelled = false;
    setDetailLoading(true);
    getDocument(claimId, active, { signal: controller.signal })
      .then((detail) => {
        if (cancelled) return;
        setActiveDetail(detail);
      })
      .catch((err) => {
        if (cancelled || controller.signal.aborted) return;
        // Surface a non-abort failure rather than silently clearing the
        // detail; the UI shows the existing detail until the user
        // picks another tab. This avoids an unhandled-promise landmine
        // where the active tab silently becomes empty.
        // eslint-disable-next-line no-console
        console.error("Failed to load document detail", err);
        setActiveDetail(null);
      })
      .finally(() => {
        if (!cancelled) setDetailLoading(false);
      });
    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [active, claimId]);

  const handleUpload = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!pendingFile) return;
    setUploading(true);
    setUploadError(null);
    try {
      await uploadDocument(claimId, pendingFile, docType);
      setPendingFile(null);
      // Reset the file input
      const input = document.getElementById("doc-file") as HTMLInputElement | null;
      if (input) input.value = "";
      await load();
    } catch (err) {
      const message = err instanceof Error ? err.message : "Upload failed.";
      setUploadError(message);
    } finally {
      setUploading(false);
    }
  };

  if (!claimId) {
    return (
      <PageShell title="Documents" description="Document viewer.">
        <EmptyState
          title="No claim selected"
          description="Navigate to a claim from the Claims list to view its documents."
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
        title={`Documents — Claim #${claimId}`}
        description="Tabbed document previews with extracted field overlays and confidence indicators."
      >
        <ErrorState message={error} onRetry={load} />
      </PageShell>
    );
  }

  if (docs === null) {
    return (
      <PageShell
        title={`Documents — Claim #${claimId}`}
        description="Tabbed document previews with extracted field overlays and confidence indicators."
      >
        <LoadingState label="Loading documents…" />
      </PageShell>
    );
  }

  const activeDoc = docs.find((d) => d.id === active) ?? null;

  return (
    <PageShell
      title={`Documents — Claim #${claimId}`}
      description="Tabbed document previews with extracted field overlays and confidence indicators."
    >
      <div style={{ display: "flex", flexDirection: "column", gap: "var(--space-4)" }}>
        {/* Upload control */}
        <form
          onSubmit={handleUpload}
          style={{
            display: "flex",
            gap: "var(--space-3)",
            padding: "var(--space-3) var(--space-4)",
            backgroundColor: "var(--color-surface)",
            border: "1px solid var(--color-border)",
            borderRadius: "var(--radius-md)",
            alignItems: "center",
            flexWrap: "wrap",
          }}
        >
          <label
            style={{
              fontSize: "var(--text-xs)",
              color: "var(--color-text-muted)",
              textTransform: "uppercase",
              letterSpacing: "0.05em",
            }}
          >
            Doc type
            <select
              value={docType}
              onChange={(e) => setDocType(e.target.value as DocType)}
              style={{ ...inputStyle, marginLeft: "var(--space-2)" }}
              aria-label="Document type"
            >
              {DOC_TYPES.map((t) => (
                <option key={t} value={t}>
                  {DOC_TYPE_LABEL[t]}
                </option>
              ))}
            </select>
          </label>
          <input
            id="doc-file"
            type="file"
            accept="application/pdf,image/jpeg,image/png"
            onChange={(e) => setPendingFile(e.target.files?.[0] ?? null)}
            disabled={uploading}
            style={{ color: "var(--color-text-primary)", fontSize: "var(--text-sm)" }}
          />
          <button type="submit" disabled={!pendingFile || uploading} style={buttonStyle}>
            {uploading ? "Uploading…" : "Upload"}
          </button>
          {uploadError && (
            <span role="alert" style={{ color: "var(--color-risk-high)", fontSize: "var(--text-sm)" }}>
              {uploadError}
            </span>
          )}
        </form>

        {/* Tabs */}
        {docs.length === 0 ? (
          <EmptyState
            title="No documents uploaded yet"
            description="Use the form above to attach a claim form, policy, estimate, or invoice."
          />
        ) : (
          <>
            <div
              role="tablist"
              style={{
                display: "flex",
                gap: "var(--space-1)",
                borderBottom: "1px solid var(--color-border)",
                overflowX: "auto",
              }}
            >
              {docs.map((d) => {
                const isActive = d.id === active;
                return (
                  <button
                    key={d.id}
                    role="tab"
                    aria-selected={isActive}
                    onClick={() => setActive(d.id)}
                    style={{
                      padding: "var(--space-2) var(--space-4)",
                      background: "transparent",
                      border: "none",
                      borderBottom: isActive
                        ? "2px solid var(--color-accent)"
                        : "2px solid transparent",
                      color: isActive
                        ? "var(--color-text-primary)"
                        : "var(--color-text-secondary)",
                      fontSize: "var(--text-sm)",
                      fontWeight: isActive ? 500 : 400,
                      cursor: "pointer",
                      whiteSpace: "nowrap",
                    }}
                  >
                    {DOC_TYPE_LABEL[d.doc_type] ?? d.doc_type}
                  </button>
                );
              })}
            </div>

            {activeDoc && (
              <div
                role="tabpanel"
                style={{
                  display: "grid",
                  gridTemplateColumns: "minmax(0, 2fr) minmax(0, 1fr)",
                  gap: "var(--space-4)",
                }}
              >
                {/* File metadata + link */}
                <section
                  aria-label="Document metadata"
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
                  <div
                    style={{
                      display: "grid",
                      gridTemplateColumns: "140px 1fr",
                      gap: "var(--space-3)",
                      fontSize: "var(--text-sm)",
                    }}
                  >
                    <span style={{ color: "var(--color-text-muted)" }}>Type</span>
                    <span style={{ color: "var(--color-text-primary)" }}>
                      {DOC_TYPE_LABEL[activeDoc.doc_type] ?? activeDoc.doc_type}
                    </span>
                    <span style={{ color: "var(--color-text-muted)" }}>Extraction</span>
                    <span style={{ color: "var(--color-text-primary)" }}>
                      {EXTRACTION_STATUS_LABEL[activeDoc.extraction_status]}
                    </span>
                    <span style={{ color: "var(--color-text-muted)" }}>File path</span>
                    <code
                      style={{
                        fontSize: "var(--text-xs)",
                        color: "var(--color-text-primary)",
                        wordBreak: "break-all",
                        fontFamily: "ui-monospace, SFMono-Regular, monospace",
                      }}
                    >
                      {activeDoc.file_path}
                    </code>
                  </div>
                  <a
                    href={`/api/${activeDoc.file_path}`}
                    target="_blank"
                    rel="noreferrer"
                    className={sharedStyles.retryButton}
                    style={{ alignSelf: "flex-start" }}
                  >
                    Open file in new tab →
                  </a>
                </section>

                {/* Extracted fields side panel */}
                <section
                  aria-label="Extracted fields"
                  style={{
                    padding: "var(--space-4)",
                    backgroundColor: "var(--color-surface)",
                    border: "1px solid var(--color-border)",
                    borderRadius: "var(--radius-md)",
                  }}
                >
                  {detailLoading ? (
                    <p
                      style={{
                        fontSize: "var(--text-sm)",
                        color: "var(--color-text-muted)",
                        margin: 0,
                      }}
                    >
                      Loading extracted fields…
                    </p>
                  ) : (
                    <ExtractedFields
                      fields={activeDetail?.extracted_fields ?? null}
                      rawConfidence={activeDetail?.raw_confidence ?? null}
                    />
                  )}
                  <p
                    style={{
                      marginTop: "var(--space-3)",
                      fontSize: "var(--text-xs)",
                      color: "var(--color-text-muted)",
                    }}
                  >
                    PDF canvas overlay is a later-phase enhancement. For now the
                    file is opened in a new tab and the extracted fields surface
                    here as the analysis pipeline runs.
                  </p>
                </section>
              </div>
            )}
          </>
        )}
      </div>
    </PageShell>
  );
}
