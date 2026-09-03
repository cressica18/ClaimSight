import type { ReactNode } from "react";
import styles from "./PageShell.module.css";

interface PageShellProps {
  title: string;
  description?: string;
  children?: ReactNode;
}

/**
 * Reusable page shell used by Phase 1 placeholder pages.
 * Phase 9 will replace or augment this with full page implementations.
 */
export default function PageShell({
  title,
  description,
  children,
}: PageShellProps) {
  return (
    <section className={styles.page}>
      <header className={styles.header}>
        <h1 className={styles.title}>{title}</h1>
        {description && (
          <p className={styles.description}>{description}</p>
        )}
      </header>
      {children && <div className={styles.content}>{children}</div>}
    </section>
  );
}
