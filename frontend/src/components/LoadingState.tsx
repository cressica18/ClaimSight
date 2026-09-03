/**
 * LoadingState — minimal spinner with a label. Used by every screen
 * while a request is in flight. Per blueprint 11.2, the spinner is
 * small and accompanied by a real label, not a generic full-page
 * loader.
 */
import styles from "./shared.module.css";

interface LoadingStateProps {
  label?: string;
}

export default function LoadingState({
  label = "Loading…",
}: LoadingStateProps) {
  return (
    <div
      className={styles.loadingState}
      role="status"
      aria-live="polite"
      aria-busy="true"
    >
      <div className={styles.loadingSpinner} aria-hidden="true" />
      <p className={styles.errorStateDescription}>{label}</p>
    </div>
  );
}
