/**
 * StageTracker — per-stage text tracker for the Claim Analysis
 * scaffolding. Blueprint 11.2 explicitly calls out a real step
 * tracker ("Analyzing images… Extracting documents… Running
 * consistency checks…") rather than a generic spinner.
 *
 * The stages are derived from data the frontend can already fetch
 * (claim + risk signals + documents + investigation). The actual
 * `/analyze` orchestration is Phase 11; this component is a read-only
 * display of progress.
 */
import styles from "./shared.module.css";

export type StageStatus = "pending" | "running" | "complete";

export interface Stage {
  key: string;
  title: string;
  description: string;
  status: StageStatus;
}

interface StageTrackerProps {
  stages: Stage[];
}

const STATUS_LABEL: Record<StageStatus, string> = {
  pending: "Pending",
  running: "In progress",
  complete: "Complete",
};

const MARKER: Record<StageStatus, string> = {
  pending: "·",
  running: "…",
  complete: "·",
};

export default function StageTracker({ stages }: StageTrackerProps) {
  return (
    <ol className={styles.stageList} aria-label="Analysis pipeline stages">
      {stages.map((stage) => {
        const rowClass =
          stage.status === "complete"
            ? `${styles.stageRow} ${styles.complete}`
            : stage.status === "running"
            ? `${styles.stageRow} ${styles.running}`
            : styles.stageRow;
        const markerClass =
          stage.status === "complete"
            ? `${styles.stageMarker} ${styles.complete}`
            : stage.status === "running"
            ? `${styles.stageMarker} ${styles.running}`
            : styles.stageMarker;
        const statusClass =
          stage.status === "complete"
            ? `${styles.stageStatus} ${styles.complete}`
            : stage.status === "running"
            ? `${styles.stageStatus} ${styles.running}`
            : styles.stageStatus;
        return (
          <li key={stage.key} className={rowClass}>
            <span className={markerClass} aria-hidden="true">
              {MARKER[stage.status]}
            </span>
            <div className={styles.stageBody}>
              <span className={styles.stageTitle}>{stage.title}</span>
              <span className={styles.stageDescription}>{stage.description}</span>
              <span className={statusClass}>{STATUS_LABEL[stage.status]}</span>
            </div>
          </li>
        );
      })}
    </ol>
  );
}
