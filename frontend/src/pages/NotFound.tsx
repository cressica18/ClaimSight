/**
 * 404 Not Found page.
 */
import { Link } from "react-router-dom";
import PageShell from "../components/PageShell";

export default function NotFound() {
  return (
    <PageShell
      title="Page Not Found"
      description="The page you requested does not exist."
    >
      <Link to="/" style={{ color: "var(--color-accent)" }}>
        ← Back to Dashboard
      </Link>
    </PageShell>
  );
}
