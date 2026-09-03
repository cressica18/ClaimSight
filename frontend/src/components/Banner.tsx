/**
 * Banner — a small callout used to surface the human-decision-required
 * disclaimer on the Investigation Summary and Decision Panel screens.
 * Two tones: `info` (cool accent) and `warning` (muted amber, for the
 * AI disclaimer specifically).
 *
 * Tone is carried entirely by the border/background/text color in the
 * CSS; the banner deliberately has no decorative emoji glyph. Earlier
 * revisions used a `⚠` icon here; that glyph was removed in the final
 * bug-fix pass because the system does not ship an icon library and
 * the visual tone is already unambiguous.
 */
import type { ReactNode } from "react";
import styles from "./shared.module.css";

interface BannerProps {
  children: ReactNode;
  tone?: "info" | "warning";
}

export default function Banner({
  children,
  tone = "info",
}: BannerProps) {
  return (
    <div
      className={`${styles.banner} ${tone === "warning" ? styles.warning : styles.info}`}
      role="note"
    >
      <div>{children}</div>
    </div>
  );
}
