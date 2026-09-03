import { useEffect, useState } from "react";
import { NavLink, Outlet } from "react-router-dom";
import styles from "./Layout.module.css";

interface NavLinkItem {
  to: string;
  label: string;
  end?: boolean;
}

const NAV_LINKS: NavLinkItem[] = [
  { to: "/", label: "Dashboard", end: true },
  { to: "/claims", label: "Claims" },
  { to: "/claims/new", label: "New Claim" },
];

interface BackendMode {
  app_env: string;
  demo_mode: boolean;
  use_demo_cv: boolean;
  use_demo_gemini: boolean;
}

/**
 * Layout shell for the whole app.
 *
 * When the backend is running with `use_demo_cv=True` or
 * `use_demo_gemini=True`, a small "Demo data" badge appears in
 * the sidebar so the reviewer can tell at a glance that the
 * system is using the deterministic stub (no real CV model,
 * no real Gemini calls). The badge is purely cosmetic and does
 * not gate any functionality — production deployments simply
 * don't see it.
 */
export default function Layout() {
  const [mode, setMode] = useState<BackendMode | null>(null);

  useEffect(() => {
    // Read backend mode once on mount. The endpoint is cheap and
    // public; failure is non-fatal (the badge just stays hidden).
    let cancelled = false;
    fetch("/api/mode")
      .then((r) => (r.ok ? r.json() : null))
      .then((data: BackendMode | null) => {
        if (!cancelled) setMode(data);
      })
      .catch(() => {
        if (!cancelled) setMode(null);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <div className={styles.shell}>
      <nav className={styles.sidebar} aria-label="Main navigation">
        <div className={styles.logo}>
          <span className={styles.logoMark}>CS</span>
          <span className={styles.logoText}>ClaimSight</span>
        </div>

        <ul className={styles.navList} role="list">
          {NAV_LINKS.map(({ to, label, end }) => (
            <li key={to}>
              <NavLink
                to={to}
                end={end}
                className={({ isActive }) =>
                  [styles.navLink, isActive ? styles.navLinkActive : ""].join(
                    " "
                  )
                }
              >
                {label}
              </NavLink>
            </li>
          ))}
        </ul>

        {mode?.demo_mode && (
          <div className={styles.demoBadge} role="status" title={
            mode.use_demo_cv && mode.use_demo_gemini
              ? "Demo CV + demo Gemini are active. All signals come from the deterministic stub."
              : mode.use_demo_cv
              ? "Demo CV is active. Image analysis uses the deterministic stub."
              : "Demo Gemini is active. Investigation summaries use the deterministic stub."
          }>
            <span className={styles.demoBadgeDot} aria-hidden="true" />
            Demo data
          </div>
        )}
      </nav>

      <main className={styles.main}>
        <Outlet />
      </main>
    </div>
  );
}
