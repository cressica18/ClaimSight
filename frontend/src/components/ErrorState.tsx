/**
 * ErrorState — reusable error block with optional retry.
 *
 * No decorative glyph. The errorState CSS class paints a left-border
 * accent and uses the high-risk text color to communicate severity;
 * adding an emoji on top would be redundant.
 */
import styles from "./shared.module.css";

interface ErrorStateProps {
  message: string;
  title?: string;
  onRetry?: () => void;
}

export default function ErrorState({
  message,
  title = "Something went wrong",
  onRetry,
}: ErrorStateProps) {
  return (
    <div className={styles.errorState} role="alert">
      <h3 className={styles.errorStateTitle}>{title}</h3>
      <p className={styles.errorStateDescription}>{message}</p>
      {onRetry && (
        <button
          type="button"
          className={styles.retryButton}
          onClick={onRetry}
        >
          Retry
        </button>
      )}
    </div>
  );
}
