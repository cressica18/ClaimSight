import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import Layout from "./components/Layout";
import Dashboard from "./pages/Dashboard";
import ClaimsList from "./pages/ClaimsList";
import NewClaim from "./pages/NewClaim";
import ClaimAnalysis from "./pages/ClaimAnalysis";
import ImageAnalysis from "./pages/ImageAnalysis";
import DocumentViewer from "./pages/DocumentViewer";
import RiskSignals from "./pages/RiskSignals";
import InvestigationSummary from "./pages/InvestigationSummary";
import DecisionPanel from "./pages/DecisionPanel";
import NotFound from "./pages/NotFound";

/**
 * Application routing.
 *
 * All 9 screens from blueprint Section 11.1 are registered here.
 * Phase 1: pages are shells (no data fetching). Phase 9 implements the full UX.
 *
 * Claim-scoped pages live under /claims/:id so the claim context is available
 * via useParams() without extra state management.
 */
export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          {/* 1. Dashboard */}
          <Route index element={<Dashboard />} />

          {/* 2. Claims List */}
          <Route path="claims" element={<ClaimsList />} />

          {/* 3. New Claim / Upload */}
          <Route path="claims/new" element={<NewClaim />} />

          {/* Claim-scoped screens (4–9) */}
          <Route path="claims/:id">
            {/* 4. Claim Analysis — pipeline progress + results */}
            <Route index element={<ClaimAnalysis />} />

            {/* 5. Image Analysis */}
            <Route path="images" element={<ImageAnalysis />} />

            {/* 6. Document / Evidence Viewer */}
            <Route path="documents" element={<DocumentViewer />} />

            {/* 7. Risk Signals */}
            <Route path="signals" element={<RiskSignals />} />

            {/* 8. Investigation Summary */}
            <Route path="investigation" element={<InvestigationSummary />} />

            {/* 9. Decision Panel */}
            <Route path="decision" element={<DecisionPanel />} />
          </Route>

          {/* Catch-all */}
          <Route path="404" element={<NotFound />} />
          <Route path="*" element={<Navigate to="/404" replace />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
