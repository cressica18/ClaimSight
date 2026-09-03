/**
 * StatusPill — colored pill with paired icon + text label.
 *
 * Blueprint Section 11.2: "color not the sole indicator of risk band
 * (icon + text label always paired with color)". Every variant below
 * renders the band/status as visible text — the color is decoration.
 */
import styles from "./shared.module.css";

export type RiskBandValue = "Low" | "Medium" | "High";

interface RiskBandPillProps {
  band: RiskBandValue | null | undefined;
  /** Optional aria-label override; defaults to the band name. */
  label?: string;
}

const BAND_ICON: Record<RiskBandValue, string> = {
  Low: "●",
  Medium: "▲",
  High: "■",
};

const BAND_CLASS: Record<RiskBandValue, string> = {
  Low: styles.low!,
  Medium: styles.medium!,
  High: styles.high!,
};

export function RiskBandPill({ band, label }: RiskBandPillProps) {
  if (!band) {
    return (
      <span className={`${styles.pill} ${styles.neutral}`} aria-label="No risk band assigned">
        — not scored
      </span>
    );
  }
  return (
    <span
      className={`${styles.pill} ${BAND_CLASS[band]}`}
      aria-label={label ?? `Risk band: ${band}`}
    >
      <span aria-hidden="true">{BAND_ICON[band]}</span>
      {band}
    </span>
  );
}

type ClaimStatusValue =
  | "pending"
  | "analyzing"
  | "completed"
  | "analysis_failed"
  | "decided";

interface ClaimStatusPillProps {
  status: ClaimStatusValue;
}

const STATUS_CLASS: Record<ClaimStatusValue, string> = {
  pending: styles.pending!,
  analyzing: styles.analyzing!,
  completed: styles.completed!,
  analysis_failed: styles.analysis_failed!,
  decided: styles.decided!,
};

export function ClaimStatusPill({ status }: ClaimStatusPillProps) {
  // Replace underscore-separated words with spaces; show original case.
  const display = status.replace(/_/g, " ");
  return (
    <span
      className={`${styles.pill} ${STATUS_CLASS[status]}`}
      aria-label={`Status: ${display}`}
    >
      {display}
    </span>
  );
}

type RecommendationValue = "normal" | "manual_review" | "investigate";

interface RecommendationPillProps {
  recommendation: RecommendationValue;
}

const REC_CLASS: Record<RecommendationValue, string> = {
  normal: styles.normal!,
  manual_review: styles.manual_review!,
  investigate: styles.investigate!,
};

const REC_LABEL: Record<RecommendationValue, string> = {
  normal: "Normal processing",
  manual_review: "Manual review",
  investigate: "Investigate",
};

export function RecommendationPill({ recommendation }: RecommendationPillProps) {
  return (
    <span
      className={`${styles.pill} ${REC_CLASS[recommendation]}`}
      aria-label={`Recommendation: ${REC_LABEL[recommendation]}`}
    >
      {REC_LABEL[recommendation]}
    </span>
  );
}

type DecisionValue = "approve" | "deny" | "investigate" | "manual_review";

interface DecisionPillProps {
  decision: DecisionValue;
}

const DECISION_CLASS: Record<DecisionValue, string> = {
  approve: styles.approve!,
  deny: styles.deny!,
  investigate: styles.investigate!,
  manual_review: styles.manual_review!,
};

const DECISION_LABEL: Record<DecisionValue, string> = {
  approve: "Approve",
  deny: "Deny",
  investigate: "Investigate",
  manual_review: "Manual review",
};

export function DecisionPill({ decision }: DecisionPillProps) {
  return (
    <span
      className={`${styles.pill} ${DECISION_CLASS[decision]}`}
      aria-label={`Decision: ${DECISION_LABEL[decision]}`}
    >
      {DECISION_LABEL[decision]}
    </span>
  );
}
