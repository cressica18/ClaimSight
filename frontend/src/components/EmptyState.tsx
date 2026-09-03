/**
 * EmptyState — reusable empty-state block for screens that have no
 * data to render. Caller passes a short title and an optional body
 * line. Blueprint 11.2: "per-screen empty/loading/error states
 * designed per-screen, not generic spinners" — this is the skeleton;
 * each page styles its own copy.
 *
 * The `icon` slot is intentionally absent: we do not use decorative
 * emoji glyphs in the empty state. The title + border-left accent
 * (in `.emptyState`) carry the visual hierarchy instead.
 */
import type { ReactNode } from "react";
import styles from "./shared.module.css";

interface EmptyStateProps {
  title: string;
  description?: string;
  action?: ReactNode;
}

export default function EmptyState({
  title,
  description,
  action,
}: EmptyStateProps) {
  return (
    <div className={styles.emptyState} role="status">
      <h3 className={styles.emptyStateTitle}>{title}</h3>
      {description && (
        <p className={styles.emptyStateDescription}>{description}</p>
      )}
      {action && <div>{action}</div>}
    </div>
  );
}
