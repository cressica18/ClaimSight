/**
 * EvidenceViewer — render a bundle of Evidence rows with type-specific
 * subrenderers. Designed for the Risk Signals side panel
 * (Screen 7) and any other screen that needs to surface evidence
 * linked to a claim object.
 *
 * Blueprint Section 8: every RiskSignal must have >=1 Evidence row.
 * The four evidence types (image / document / field / computed) each
 * get their own subrenderer. We never invent bounding boxes, fields,
 * or calculation inputs that aren't in `detail_json`; missing data
 * falls back to an explicit "not available" message.
 *
 * For image evidence, the actual image file path lives on the linked
 * `Damage` row's `region_ref` (a JSON `{"image_path": "..."}`). We
 * fetch the claim's image list once and look the referenced damage
 * up by id; if the row can't be found we still render the evidence
 * row's metadata (confidence, bounding-box) but show an honest
 * "image not found" placeholder.
 *
 * Human-in-the-loop: every subrenderer shows the source id/reference
 * and the raw payload, so a reviewer can always see what the system
 * claimed and where the data came from.
 */
import { useEffect, useState } from "react";
import type { DamageResponse, Evidence, EvidenceType } from "../types";
import { listImages } from "../api/client";

interface EvidenceViewerProps {
  claimId: number;
  evidence: Evidence[];
  /**
   * Optional inline mode: when true, evidence rows render directly
   * (no per-evidence wrapper). Useful for embedding inside an
   * existing card. When false (default), a list with subtle dividers
   * is rendered.
   */
  inline?: boolean;
}

// ─── helpers ────────────────────────────────────────────────────────────────

function parseImagePath(regionRef: string | null | undefined): string {
  if (!regionRef) return "";
  try {
    const parsed = JSON.parse(regionRef);
    if (parsed && typeof parsed === "object" && "image_path" in parsed) {
      return String(parsed.image_path);
    }
  } catch {
    // Fall through — regionRef might already be a raw path.
  }
  return regionRef;
}

function parseBoundingBox(
  detail: Record<string, unknown> | null | undefined
): [number, number, number, number] | null {
  if (!detail) return null;
  const raw = detail.bounding_box;
  if (!Array.isArray(raw) || raw.length !== 4) return null;
  const [x1, y1, x2, y2] = raw;
  if (
    ![x1, y1, x2, y2].every(
      (n) => typeof n === "number" && Number.isFinite(n)
    )
  ) {
    return null;
  }
  return [x1, y1, x2, y2];
}

function formatNumber(n: unknown): string {
  if (typeof n === "number" && Number.isFinite(n)) {
    return Number.isInteger(n) ? n.toString() : n.toFixed(3);
  }
  if (n === null || n === undefined) return "—";
  return String(n);
}

function formatPercent(c: unknown): string {
  if (typeof c !== "number" || !Number.isFinite(c)) return "—";
  return `${Math.round(c * 100)}%`;
}

function EvidenceTypeBadge({ kind }: { kind: EvidenceType }) {
  const palette: Record<EvidenceType, { bg: string; fg: string; label: string }> = {
    image: { bg: "var(--color-surface-raised)", fg: "var(--color-text-primary)", label: "Image" },
    document: { bg: "var(--color-surface-raised)", fg: "var(--color-text-primary)", label: "Document" },
    field: { bg: "var(--color-surface-raised)", fg: "var(--color-text-primary)", label: "Field" },
    computed: { bg: "var(--color-surface-raised)", fg: "var(--color-text-primary)", label: "Computed" },
  };
  const p = palette[kind];
  return (
    <span
      style={{
        display: "inline-block",
        padding: "2px 8px",
        borderRadius: "var(--radius-sm)",
        backgroundColor: p.bg,
        color: p.fg,
        fontSize: "var(--text-xs)",
        fontWeight: 500,
        border: "1px solid var(--color-border)",
      }}
    >
      {p.label}
    </span>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "minmax(110px, 1fr) 2fr",
        gap: "var(--space-3)",
        fontSize: "var(--text-sm)",
        padding: "var(--space-2) 0",
        borderBottom: "1px solid var(--color-border)",
      }}
    >
      <span style={{ color: "var(--color-text-muted)" }}>{label}</span>
      <span
        style={{
          color: "var(--color-text-primary)",
          fontFamily: "ui-monospace, SFMono-Regular, monospace",
          fontSize: "var(--text-xs)",
          wordBreak: "break-word",
        }}
      >
        {children}
      </span>
    </div>
  );
}

function NotAvailableHint({ what }: { what: string }) {
  return (
    <p
      style={{
        fontSize: "var(--text-xs)",
        color: "var(--color-text-muted)",
        margin: "var(--space-2) 0 0 0",
        fontStyle: "italic",
      }}
    >
      {what} not available on this evidence row.
    </p>
  );
}

// ─── image subrenderer ─────────────────────────────────────────────────────

interface ImageEvidenceProps {
  ev: Evidence;
  /** Damage list fetched for the parent claim. */
  damages: DamageResponse[];
}

function ImageEvidence({ ev, damages }: ImageEvidenceProps) {
  // The reference is the Damage.id (stored as a string on the
  // Evidence row). Look it up in the per-claim image list.
  const damageId = ev.reference ? Number(ev.reference) : null;
  const damage =
    damageId !== null && Number.isFinite(damageId)
      ? damages.find((d) => d.id === damageId) ?? null
      : null;

  const imagePath = damage ? parseImagePath(damage.region_ref) : "";
  const imageUrl = imagePath ? `/api/${imagePath}` : "";
  const bbox = parseBoundingBox(ev.detail_json);
  const detailObj =
    ev.detail_json && typeof ev.detail_json === "object"
      ? (ev.detail_json as Record<string, unknown>)
      : null;
  const cvConfidence =
    detailObj && typeof detailObj.confidence === "number"
      ? detailObj.confidence
      : damage && typeof damage.confidence === "number"
      ? damage.confidence
      : null;
  const cvDamageType = damage ? damage.damage_type : null;
  const cvSeverity = damage ? damage.severity : null;

  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "minmax(0, 220px) minmax(0, 1fr)",
        gap: "var(--space-4)",
        padding: "var(--space-3)",
        backgroundColor: "var(--color-surface-raised)",
        border: "1px solid var(--color-border)",
        borderRadius: "var(--radius-sm)",
      }}
    >
      <div
        style={{
          position: "relative",
          minHeight: "120px",
          backgroundColor: "var(--color-surface)",
          border: "1px dashed var(--color-border)",
          borderRadius: "var(--radius-sm)",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          overflow: "hidden",
        }}
      >
        {imageUrl ? (
          <>
            <img
              src={imageUrl}
              alt={`Damage #${damageId ?? "unknown"}`}
              style={{
                maxWidth: "100%",
                maxHeight: "220px",
                objectFit: "contain",
                display: "block",
              }}
              onError={(e) => {
                e.currentTarget.style.display = "none";
              }}
            />
            {bbox && (
              <div
                aria-hidden="true"
                style={{
                  position: "absolute",
                  left: `${bbox[0]}px`,
                  top: `${bbox[1]}px`,
                  width: `${Math.max(0, bbox[2] - bbox[0])}px`,
                  height: `${Math.max(0, bbox[3] - bbox[1])}px`,
                  border: "2px solid var(--color-accent)",
                  backgroundColor: "rgba(0, 0, 0, 0.04)",
                  pointerEvents: "none",
                }}
                title={`Region ${bbox.join(", ")}`}
              />
            )}
          </>
        ) : (
          <span
            style={{
              fontSize: "var(--text-xs)",
              color: "var(--color-text-muted)",
              padding: "var(--space-3)",
              textAlign: "center",
            }}
          >
            Image not available
            <br />
            <span style={{ fontSize: "0.85em" }}>
              {damageId !== null
                ? `damage #${damageId} not found in this claim's image list`
                : "no reference id on this evidence row"}
            </span>
          </span>
        )}
      </div>

      <div style={{ display: "flex", flexDirection: "column", gap: "var(--space-1)" }}>
        <Field label="Damage id">{damageId !== null ? `#${damageId}` : "—"}</Field>
        <Field label="CV damage type">{cvDamageType ?? "—"}</Field>
        <Field label="CV severity">{cvSeverity ?? "—"}</Field>
        <Field label="CV confidence">{formatPercent(cvConfidence)}</Field>
        <Field label="Bounding box">
          {bbox ? `[${bbox.map((n) => Math.round(n)).join(", ")}]` : "—"}
        </Field>
        <Field label="Evidence id">#{ev.id}</Field>
      </div>
    </div>
  );
}

// ─── document subrenderer ──────────────────────────────────────────────────

function DocumentEvidence({ ev }: { ev: Evidence }) {
  const detailObj =
    ev.detail_json && typeof ev.detail_json === "object"
      ? (ev.detail_json as Record<string, unknown>)
      : null;

  // Detail shapes we look at (whichever is present):
  //   { field_name, value, page } — single extracted field
  //   { page, fields: [{name, value, confidence?}, ...] } — multiple fields
  //   { field_path } — bare path
  const docId = ev.reference;
  const page = detailObj && typeof detailObj.page === "number" ? detailObj.page : null;
  const fieldName =
    detailObj && typeof detailObj.field_name === "string"
      ? detailObj.field_name
      : detailObj && typeof detailObj.field_path === "string"
      ? detailObj.field_path
      : null;
  const value =
    detailObj && "value" in detailObj ? (detailObj.value as unknown) : null;
  const confidence =
    detailObj && typeof detailObj.confidence === "number" ? detailObj.confidence : null;
  const fields =
    detailObj && Array.isArray(detailObj.fields) ? (detailObj.fields as unknown[]) : null;

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        gap: "var(--space-1)",
        padding: "var(--space-3)",
        backgroundColor: "var(--color-surface-raised)",
        border: "1px solid var(--color-border)",
        borderRadius: "var(--radius-sm)",
      }}
    >
      <Field label="Document id">{docId ?? "—"}</Field>
      {page !== null && <Field label="Page">{`#${page}`}</Field>}
      {fieldName && <Field label="Field">{fieldName}</Field>}
      {value !== null && value !== undefined && (
        <Field label="Extracted value">{String(value)}</Field>
      )}
      {confidence !== null && <Field label="Confidence">{formatPercent(confidence)}</Field>}

      {fields && fields.length > 0 && (
        <div style={{ marginTop: "var(--space-2)" }}>
          <p
            style={{
              fontSize: "var(--text-xs)",
              color: "var(--color-text-muted)",
              textTransform: "uppercase",
              letterSpacing: "0.05em",
              margin: 0,
              marginBottom: "var(--space-1)",
            }}
          >
            Extracted fields
          </p>
          <ul
            style={{
              listStyle: "none",
              margin: 0,
              padding: 0,
              display: "flex",
              flexDirection: "column",
              gap: "var(--space-1)",
            }}
          >
            {fields.map((f, i) => {
              const obj = f && typeof f === "object" ? (f as Record<string, unknown>) : null;
              const name = obj && typeof obj.name === "string" ? obj.name : `field_${i + 1}`;
              const val = obj && "value" in obj ? (obj.value as unknown) : f;
              const conf = obj && typeof obj.confidence === "number" ? obj.confidence : null;
              return (
                <li
                  key={i}
                  style={{
                    display: "grid",
                    gridTemplateColumns: "1fr 1fr auto",
                    gap: "var(--space-2)",
                    fontSize: "var(--text-xs)",
                    padding: "var(--space-2)",
                    backgroundColor: "var(--color-surface)",
                    border: "1px solid var(--color-border)",
                    borderRadius: "var(--radius-sm)",
                  }}
                >
                  <span style={{ color: "var(--color-text-muted)" }}>{name}</span>
                  <span
                    style={{
                      color: "var(--color-text-primary)",
                      fontFamily: "ui-monospace, SFMono-Regular, monospace",
                      wordBreak: "break-word",
                    }}
                  >
                    {String(val)}
                  </span>
                  <span style={{ color: "var(--color-text-muted)" }}>
                    {conf !== null ? formatPercent(conf) : "—"}
                  </span>
                </li>
              );
            })}
          </ul>
        </div>
      )}

      {!fieldName && value === null && !fields && <NotAvailableHint what="Document field details" />}
    </div>
  );
}

// ─── field subrenderer ─────────────────────────────────────────────────────

function FieldEvidence({ ev }: { ev: Evidence }) {
  const detailObj =
    ev.detail_json && typeof ev.detail_json === "object"
      ? (ev.detail_json as Record<string, unknown>)
      : null;

  const fieldName =
    detailObj && typeof detailObj.field_name === "string"
      ? detailObj.field_name
      : ev.reference ?? null;
  const expected = detailObj && "expected" in detailObj ? detailObj.expected : null;
  const actual = detailObj && "actual" in detailObj ? detailObj.actual : null;
  const sourceA = detailObj && typeof detailObj.source_a === "string" ? detailObj.source_a : null;
  const sourceB = detailObj && typeof detailObj.source_b === "string" ? detailObj.source_b : null;
  const claimed = detailObj && "claimed" in detailObj ? detailObj.claimed : null;

  const hasConflict = expected !== null && actual !== null && expected !== actual;

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        gap: "var(--space-1)",
        padding: "var(--space-3)",
        backgroundColor: "var(--color-surface-raised)",
        border: "1px solid var(--color-border)",
        borderRadius: "var(--radius-sm)",
      }}
    >
      <Field label="Field">{fieldName ?? "—"}</Field>

      {(expected !== null || actual !== null) && (
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "1fr 1fr",
            gap: "var(--space-3)",
            marginTop: "var(--space-2)",
          }}
        >
          <div
            style={{
              padding: "var(--space-3)",
              backgroundColor: "var(--color-surface)",
              border: "1px solid var(--color-border)",
              borderRadius: "var(--radius-sm)",
            }}
          >
            <p
              style={{
                fontSize: "var(--text-xs)",
                color: "var(--color-text-muted)",
                textTransform: "uppercase",
                letterSpacing: "0.05em",
                margin: 0,
                marginBottom: "var(--space-1)",
              }}
            >
              {sourceA ?? "Source A"}
            </p>
            <p
              style={{
                margin: 0,
                fontSize: "var(--text-sm)",
                color: "var(--color-text-primary)",
                fontFamily: "ui-monospace, SFMono-Regular, monospace",
                wordBreak: "break-word",
              }}
            >
              {expected !== null ? String(expected) : "—"}
            </p>
          </div>
          <div
            style={{
              padding: "var(--space-3)",
              backgroundColor: "var(--color-surface)",
              border: hasConflict
                ? "1px solid var(--color-risk-high)"
                : "1px solid var(--color-border)",
              borderRadius: "var(--radius-sm)",
            }}
          >
            <p
              style={{
                fontSize: "var(--text-xs)",
                color: "var(--color-text-muted)",
                textTransform: "uppercase",
                letterSpacing: "0.05em",
                margin: 0,
                marginBottom: "var(--space-1)",
              }}
            >
              {sourceB ?? "Source B"}
            </p>
            <p
              style={{
                margin: 0,
                fontSize: "var(--text-sm)",
                color: hasConflict
                  ? "var(--color-risk-high)"
                  : "var(--color-text-primary)",
                fontFamily: "ui-monospace, SFMono-Regular, monospace",
                wordBreak: "break-word",
                fontWeight: hasConflict ? 500 : 400,
              }}
            >
              {actual !== null ? String(actual) : claimed !== null ? String(claimed) : "—"}
            </p>
          </div>
        </div>
      )}

      {hasConflict && (
        <p
          role="status"
          style={{
            margin: 0,
            marginTop: "var(--space-2)",
            fontSize: "var(--text-xs)",
            color: "var(--color-risk-high)",
            display: "flex",
            alignItems: "center",
            gap: "var(--space-1)",
          }}
        >
          Conflict: the two sources disagree on this field.
        </p>
      )}

      {!fieldName && expected === null && actual === null && claimed === null && (
        <NotAvailableHint what="Field comparison details" />
      )}
    </div>
  );
}

// ─── computed subrenderer ──────────────────────────────────────────────────

function ComputedEvidence({ ev }: { ev: Evidence }) {
  const detailObj =
    ev.detail_json && typeof ev.detail_json === "object"
      ? (ev.detail_json as Record<string, unknown>)
      : null;

  const baselineRange =
    detailObj && Array.isArray(detailObj.baseline_range)
      ? (detailObj.baseline_range as unknown[])
      : null;
  const claimed =
    detailObj && "claimed" in detailObj ? (detailObj.claimed as unknown) : null;
  const ratio =
    detailObj && typeof detailObj.ratio === "number" ? detailObj.ratio : null;
  const score =
    detailObj && typeof detailObj.score === "number" ? detailObj.score : null;
  const components =
    detailObj && Array.isArray(detailObj.components)
      ? (detailObj.components as unknown[])
      : null;

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        gap: "var(--space-1)",
        padding: "var(--space-3)",
        backgroundColor: "var(--color-surface-raised)",
        border: "1px solid var(--color-border)",
        borderRadius: "var(--radius-sm)",
      }}
    >
      <Field label="Reference">{ev.reference ?? "—"}</Field>
      {baselineRange && (
        <Field label="Baseline range">
          {baselineRange.map((n) => formatNumber(n)).join(" – ")}
        </Field>
      )}
      {claimed !== null && claimed !== undefined && (
        <Field label="Claimed">{formatNumber(claimed)}</Field>
      )}
      {ratio !== null && <Field label="Ratio">{formatNumber(ratio)}×</Field>}
      {score !== null && <Field label="Score">{formatNumber(score)}</Field>}

      {components && components.length > 0 && (
        <div style={{ marginTop: "var(--space-2)" }}>
          <p
            style={{
              fontSize: "var(--text-xs)",
              color: "var(--color-text-muted)",
              textTransform: "uppercase",
              letterSpacing: "0.05em",
              margin: 0,
              marginBottom: "var(--space-1)",
            }}
          >
            Inputs / breakdown
          </p>
          <ul
            style={{
              listStyle: "none",
              margin: 0,
              padding: 0,
              display: "flex",
              flexDirection: "column",
              gap: "var(--space-1)",
            }}
          >
            {components.map((c, i) => {
              const obj = c && typeof c === "object" ? (c as Record<string, unknown>) : null;
              const label = obj && typeof obj.name === "string" ? obj.name : `input_${i + 1}`;
              const val = obj && "value" in obj ? (obj.value as unknown) : c;
              return (
                <li
                  key={i}
                  style={{
                    display: "grid",
                    gridTemplateColumns: "1fr 1fr",
                    gap: "var(--space-2)",
                    fontSize: "var(--text-xs)",
                    padding: "var(--space-2)",
                    backgroundColor: "var(--color-surface)",
                    border: "1px solid var(--color-border)",
                    borderRadius: "var(--radius-sm)",
                  }}
                >
                  <span style={{ color: "var(--color-text-muted)" }}>{label}</span>
                  <span
                    style={{
                      color: "var(--color-text-primary)",
                      fontFamily: "ui-monospace, SFMono-Regular, monospace",
                      wordBreak: "break-word",
                    }}
                  >
                    {typeof val === "object" ? JSON.stringify(val) : String(val)}
                  </span>
                </li>
              );
            })}
          </ul>
        </div>
      )}

      {!baselineRange && claimed === null && ratio === null && score === null && !components && (
        <NotAvailableHint what="Calculation breakdown" />
      )}
    </div>
  );
}

// ─── dispatcher ────────────────────────────────────────────────────────────

function EvidenceItem({ ev, damages }: { ev: Evidence; damages: DamageResponse[] }) {
  const detailObj =
    ev.detail_json && typeof ev.detail_json === "object"
      ? (ev.detail_json as Record<string, unknown>)
      : null;
  let body: React.ReactNode;
  switch (ev.evidence_type) {
    case "image":
      body = <ImageEvidence ev={ev} damages={damages} />;
      break;
    case "document":
      body = <DocumentEvidence ev={ev} />;
      break;
    case "field":
      body = <FieldEvidence ev={ev} />;
      break;
    case "computed":
      body = <ComputedEvidence ev={ev} />;
      break;
    default:
      body = (
        <div
          style={{
            padding: "var(--space-3)",
            backgroundColor: "var(--color-surface-raised)",
            border: "1px solid var(--color-border)",
            borderRadius: "var(--radius-sm)",
            fontSize: "var(--text-sm)",
            color: "var(--color-text-muted)",
          }}
        >
          Unsupported evidence type <code>{String(ev.evidence_type)}</code>.
          <pre
            style={{
              fontSize: "var(--text-xs)",
              margin: "var(--space-2) 0 0 0",
              padding: "var(--space-2)",
              backgroundColor: "var(--color-surface)",
              border: "1px solid var(--color-border)",
              borderRadius: "var(--radius-sm)",
              whiteSpace: "pre-wrap",
              wordBreak: "break-word",
              fontFamily: "ui-monospace, SFMono-Regular, monospace",
            }}
          >
            {detailObj ? JSON.stringify(detailObj, null, 2) : "(no detail_json)"}
          </pre>
        </div>
      );
  }
  return (
    <li
      style={{
        display: "flex",
        flexDirection: "column",
        gap: "var(--space-2)",
        listStyle: "none",
      }}
    >
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: "var(--space-2)",
          fontSize: "var(--text-xs)",
          color: "var(--color-text-muted)",
        }}
      >
        <EvidenceTypeBadge kind={ev.evidence_type} />
        <span>#{ev.id}</span>
        {ev.reference && (
          <span>
            · ref <code style={{ fontFamily: "ui-monospace, SFMono-Regular, monospace" }}>{ev.reference}</code>
          </span>
        )}
      </div>
      {body}
    </li>
  );
}

export default function EvidenceViewer({ claimId, evidence, inline = false }: EvidenceViewerProps) {
  const [damages, setDamages] = useState<DamageResponse[] | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);

  useEffect(() => {
    // The image subrenderer needs the claim's damage list to look up
    // damage rows by id. If there's no image evidence, we still run
    // the fetch (cheap) so the state is consistent.
    let cancelled = false;
    setLoadError(null);
    listImages(claimId)
      .then((data) => {
        if (!cancelled) setDamages(data);
      })
      .catch((err) => {
        if (!cancelled) {
          setDamages([]);
          setLoadError(err instanceof Error ? err.message : "Failed to load image list.");
        }
      });
    return () => {
      cancelled = true;
    };
  }, [claimId]);

  if (evidence.length === 0) {
    return (
      <p
        style={{
          fontSize: "var(--text-sm)",
          color: "var(--color-text-muted)",
          margin: 0,
          fontStyle: "italic",
        }}
      >
        No evidence rows are linked to this signal yet.
      </p>
    );
  }

  const containerStyle: React.CSSProperties = inline
    ? { display: "flex", flexDirection: "column", gap: "var(--space-3)" }
    : {
        listStyle: "none",
        margin: 0,
        padding: 0,
        display: "flex",
        flexDirection: "column",
        gap: "var(--space-3)",
      };

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        gap: "var(--space-2)",
      }}
    >
      {loadError && (
        <p
          role="status"
          style={{
            fontSize: "var(--text-xs)",
            color: "var(--color-text-muted)",
            margin: 0,
            fontStyle: "italic",
          }}
        >
          Note: could not load the claim's image list ({loadError}). Image
          evidence will show metadata only.
        </p>
      )}
      {damages === null ? (
        <p
          role="status"
          aria-busy="true"
          style={{
            fontSize: "var(--text-xs)",
            color: "var(--color-text-muted)",
            margin: 0,
            fontStyle: "italic",
          }}
        >
          Loading linked images…
        </p>
      ) : (
        <ul style={containerStyle}>
          {evidence.map((ev) => (
            <EvidenceItem key={ev.id} ev={ev} damages={damages} />
          ))}
        </ul>
      )}
    </div>
  );
}
