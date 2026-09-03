/**
 * SeverityChip — small pill for signal / damage severity.
 * Visually identical to the per-image severity badges in ImageAnalysis
 * (Phase 5) so the two screens feel like one product.
 */
import styles from "./shared.module.css";

type Severity = "low" | "medium" | "high" | "minor" | "moderate" | "severe" | string;

interface SeverityChipProps {
  severity: Severity;
}

function classFor(s: string): string {
  const lower = s.toLowerCase();
  if (lower === "minor" || lower === "low") return styles.minor!;
  if (lower === "moderate" || lower === "medium") return styles.moderate!;
  if (lower === "severe" || lower === "high") return styles.severe!;
  return styles.unknown!;
}

export function SeverityChip({ severity }: SeverityChipProps) {
  return (
    <span
      className={`${styles.severity} ${classFor(severity)}`}
      aria-label={`Severity: ${severity}`}
    >
      {severity}
    </span>
  );
}
