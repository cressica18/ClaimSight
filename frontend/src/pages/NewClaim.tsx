/**
 * Screen 3: New Claim / Upload
 *
 * Blueprint (Section 11.1):
 * - Stepped form: customer/policy lookup → vehicle/incident details →
 *   document upload → image upload → submit for analysis
 *
 * Phase 9 implementation:
 * 1. Customer & policy — pick existing OR create new
 * 2. Vehicle & incident details — pick existing vehicle OR create new,
 *    then the claim form
 * 3. Documents — upload via POST /claims/{id}/documents
 * 4. Images — upload via POST /claims/{id}/images
 *
 * After step 4 we redirect to /claims/{newId} so the user lands on
 * the Claim Analysis scaffolding screen. The claim itself is created
 * in step 2 (so document/image uploads have a valid claim_id).
 *
 * The "Run full analysis" button on the next page is intentionally
 * disabled — that's Phase 11.
 */
import { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import PageShell from "../components/PageShell";
import ErrorState from "../components/ErrorState";
import LoadingState from "../components/LoadingState";
import {
  createClaim,
  createCustomer,
  createVehicle,
  listCustomers,
  listPolicies,
  listVehicles,
  uploadDocument,
  uploadImages,
} from "../api/client";
import type {
  Claim,
  CustomerSummary,
  DocType,
  PolicySummary,
  VehicleSummary,
} from "../types";
import sharedStyles from "../components/shared.module.css";

const inputStyle: React.CSSProperties = {
  width: "100%",
  padding: "var(--space-2) var(--space-3)",
  borderRadius: "var(--radius-sm)",
  backgroundColor: "var(--color-surface-raised)",
  color: "var(--color-text-primary)",
  border: "1px solid var(--color-border)",
  fontSize: "var(--text-sm)",
  fontFamily: "inherit",
};

const selectStyle: React.CSSProperties = {
  ...inputStyle,
  appearance: "auto",
};

const labelStyle: React.CSSProperties = {
  display: "block",
  fontSize: "var(--text-xs)",
  color: "var(--color-text-muted)",
  textTransform: "uppercase",
  letterSpacing: "0.05em",
  marginBottom: "var(--space-1)",
};

const primaryButton: React.CSSProperties = {
  padding: "var(--space-3) var(--space-6)",
  borderRadius: "var(--radius-md)",
  backgroundColor: "var(--color-accent)",
  color: "#fff",
  border: "none",
  fontSize: "var(--text-sm)",
  fontWeight: 500,
  cursor: "pointer",
};

const secondaryButton: React.CSSProperties = {
  padding: "var(--space-3) var(--space-6)",
  borderRadius: "var(--radius-md)",
  backgroundColor: "var(--color-surface-raised)",
  color: "var(--color-text-primary)",
  border: "1px solid var(--color-border)",
  fontSize: "var(--text-sm)",
  fontWeight: 500,
  cursor: "pointer",
};

/**
 * Inline form-level error. Used for per-step validation messages and
 * transition errors that should NOT replace the whole page (those go
 * to `lookupError` instead). Dismissing it on the next interaction
 * happens via the calling code clearing `formError`.
 */
const formErrorBox: React.CSSProperties = {
  padding: "var(--space-2) var(--space-3)",
  borderRadius: "var(--radius-sm)",
  backgroundColor: "var(--color-surface-raised)",
  border: "1px solid var(--color-risk-high)",
  color: "var(--color-risk-high)",
  fontSize: "var(--text-sm)",
};

const DOC_TYPES: DocType[] = [
  "claim_form",
  "policy",
  "estimate",
  "invoice",
  "previous_claim",
];

const DOC_TYPE_LABEL: Record<DocType, string> = {
  claim_form: "Claim form",
  policy: "Policy",
  estimate: "Estimate",
  invoice: "Invoice",
  previous_claim: "Previous claim",
};

type Step = 1 | 2 | 3 | 4;

interface StepperProps {
  current: Step;
}

const STEPS: { id: Step; label: string }[] = [
  { id: 1, label: "Customer & policy" },
  { id: 2, label: "Vehicle & incident" },
  { id: 3, label: "Documents" },
  { id: 4, label: "Images" },
];

function Stepper({ current }: StepperProps) {
  return (
    <ol
      style={{
        display: "flex",
        gap: "var(--space-2)",
        listStyle: "none",
        margin: 0,
        padding: 0,
        flexWrap: "wrap",
      }}
      aria-label="Form progress"
    >
      {STEPS.map((s) => {
        const isActive = s.id === current;
        const isDone = s.id < current;
        return (
          <li
            key={s.id}
            style={{
              flex: "1 1 0",
              minWidth: "120px",
              padding: "var(--space-3) var(--space-4)",
              backgroundColor: isActive
                ? "var(--color-surface-raised)"
                : "var(--color-surface)",
              border: isActive
                ? "1px solid var(--color-accent)"
                : isDone
                ? "1px solid var(--color-risk-low)"
                : "1px solid var(--color-border)",
              borderRadius: "var(--radius-md)",
              display: "flex",
              alignItems: "center",
              gap: "var(--space-2)",
              fontSize: "var(--text-sm)",
              color: isActive
                ? "var(--color-text-primary)"
                : "var(--color-text-secondary)",
            }}
          >
            <span
              aria-hidden="true"
              style={{
                display: "inline-flex",
                alignItems: "center",
                justifyContent: "center",
                width: "20px",
                height: "20px",
                borderRadius: "50%",
                backgroundColor: isDone
                  ? "var(--color-risk-low)"
                  : isActive
                  ? "var(--color-accent)"
                  : "var(--color-border)",
                color: "#fff",
                fontSize: "var(--text-xs)",
                fontWeight: 600,
                flexShrink: 0,
              }}
            >
              {s.id}
            </span>
            <span>{s.label}</span>
          </li>
        );
      })}
    </ol>
  );
}

export default function NewClaim() {
  const navigate = useNavigate();
  const [step, setStep] = useState<Step>(1);

  // Step 1 — customer & policy
  const [customerMode, setCustomerMode] = useState<"existing" | "new">("existing");
  // New-policy creation needs a vehicle_id (FK), so we always require
  // the user to pick an existing policy in step 1. (Policy creation
  // happens out-of-band.)
  const [customers, setCustomers] = useState<CustomerSummary[]>([]);
  const [policies, setPolicies] = useState<PolicySummary[]>([]);
  const [selectedCustomerId, setSelectedCustomerId] = useState<number | "">("");
  const [selectedPolicyId, setSelectedPolicyId] = useState<number | "">("");
  const [newCustomer, setNewCustomer] = useState({
    name: "",
    email: "",
    phone: "",
  });

  // Step 2 — vehicle & claim
  const [vehicles, setVehicles] = useState<VehicleSummary[]>([]);
  const [vehicleMode, setVehicleMode] = useState<"existing" | "new">("existing");
  const [selectedVehicleId, setSelectedVehicleId] = useState<number | "">("");
  const [newVehicle, setNewVehicle] = useState({
    make: "",
    model: "",
    year: new Date().getFullYear().toString(),
    vin: "",
    plate_number: "",
  });
  const [claimForm, setClaimForm] = useState({
    claim_number: `CLM-${Date.now()}`,
    incident_date: new Date().toISOString().slice(0, 10),
    reported_date: new Date().toISOString().slice(0, 10),
    claimed_amount: "",
  });
  const [createdClaim, setCreatedClaim] = useState<Claim | null>(null);

  // Step 3 — documents
  const [pendingDocFile, setPendingDocFile] = useState<File | null>(null);
  const [pendingDocType, setPendingDocType] = useState<DocType>("claim_form");
  const [docUploading, setDocUploading] = useState(false);
  const [docError, setDocError] = useState<string | null>(null);

  // Step 4 — images
  const [pendingImages, setPendingImages] = useState<FileList | null>(null);
  const [imageUploading, setImageUploading] = useState(false);
  const [imageError, setImageError] = useState<string | null>(null);

  // Two distinct error states:
  //  - `lookupError` replaces the whole page (lookup fetch failed)
  //  - `formError` is an inline, per-step validation/transition error
  //    that the user can dismiss by interacting with the form. We keep
  //    these separate so a "pick a customer" validation message never
  //    erases the entire form, and a real network failure is clearly
  //    surfaced as a page-level error with a Retry.
  const [lookupError, setLookupError] = useState<string | null>(null);
  const [formError, setFormError] = useState<string | null>(null);
  const [loadingLookups, setLoadingLookups] = useState(true);

  // Re-run lookups (used by the Retry button on a lookup error).
  const loadLookups = async (signal?: AbortSignal) => {
    setLookupError(null);
    try {
      const [custs, pols, vehs] = await Promise.all([
        listCustomers({ limit: 100 }, { signal }),
        listPolicies({ limit: 100 }, { signal }),
        listVehicles({ limit: 100 }, { signal }),
      ]);
      if (signal?.aborted) return;
      setCustomers(custs);
      setPolicies(pols);
      setVehicles(vehs);
    } catch (err) {
      if (signal?.aborted) return;
      setLookupError(
        err instanceof Error ? err.message : "Failed to load lookup data."
      );
    } finally {
      if (!signal?.aborted) setLoadingLookups(false);
    }
  };

  useEffect(() => {
    // Reset to a clean loading state on mount so navigating away and
    // back doesn't briefly show stale data from the previous visit.
    setCustomers([]);
    setPolicies([]);
    setVehicles([]);
    setSelectedCustomerId("");
    setSelectedPolicyId("");
    setSelectedVehicleId("");
    setLookupError(null);
    setFormError(null);
    setLoadingLookups(true);

    const controller = new AbortController();
    loadLookups(controller.signal);
    return () => controller.abort();
    // loadLookups reads its own state via setters; we intentionally only
    // re-run on mount.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // PolicySummary does not embed a customer id, so we can't filter
  // by customer in memory. The user picks from all active policies;
  // the create flow will fail loudly if the policy doesn't belong
  // to the chosen customer.
  const filteredPolicies = policies;

  // ─── step transitions ────────────────────────────────────────────────────

  async function goFromStep1ToStep2() {
    setFormError(null);
    try {
      let customerId: number;

      if (customerMode === "existing") {
        if (!selectedCustomerId) {
          setFormError("Pick a customer, or switch to Create new.");
          return;
        }
        customerId = selectedCustomerId;
      } else {
        if (!newCustomer.name || !newCustomer.email) {
          setFormError("Customer name and email are required.");
          return;
        }
        const created = await createCustomer({
          name: newCustomer.name,
          email: newCustomer.email,
          phone: newCustomer.phone || null,
        });
        customerId = created.id;
      }

      if (!selectedPolicyId) {
        setFormError("Pick a policy.");
        return;
      }
      const policyId: number = selectedPolicyId;

      setStep(2);
      // Stash the chosen ids for step 2
      window.localStorage.setItem("cs_new_claim_customer", String(customerId));
      window.localStorage.setItem("cs_new_claim_policy", String(policyId));
    } catch (err) {
      setFormError(
        err instanceof Error ? err.message : "Failed to advance to step 2."
      );
    }
  }

  async function goFromStep2ToStep3() {
    setFormError(null);
    try {
      const customerIdStr = window.localStorage.getItem("cs_new_claim_customer");
      const policyIdStr = window.localStorage.getItem("cs_new_claim_policy");
      const customerId = customerIdStr ? Number(customerIdStr) : null;
      const policyId = policyIdStr ? Number(policyIdStr) : null;
      if (!customerId || !policyId) {
        setFormError("Step 1 must be completed first.");
        return;
      }

      let vehicleId: number;
      if (vehicleMode === "existing") {
        if (!selectedVehicleId) {
          setFormError("Pick a vehicle, or switch to Create new.");
          return;
        }
        vehicleId = selectedVehicleId;
      } else {
        if (!newVehicle.make || !newVehicle.model || !newVehicle.year) {
          setFormError("Make, model, and year are required for a new vehicle.");
          return;
        }
        const created = await createVehicle({
          customer_id: customerId,
          make: newVehicle.make,
          model: newVehicle.model,
          year: Number(newVehicle.year),
          vin: newVehicle.vin || null,
          plate_number: newVehicle.plate_number || null,
        });
        vehicleId = created.id;
      }

      if (!claimForm.claim_number || !claimForm.incident_date) {
        setFormError("Claim number and incident date are required.");
        return;
      }
      const claimed = claimForm.claimed_amount
        ? Number(claimForm.claimed_amount)
        : null;

      const claim = await createClaim({
        claim_number: claimForm.claim_number,
        policy_id: policyId,
        vehicle_id: vehicleId,
        incident_date: claimForm.incident_date,
        reported_date: claimForm.reported_date || null,
        claimed_amount: claimed,
      });
      setCreatedClaim(claim);
      setStep(3);
    } catch (err) {
      setFormError(
        err instanceof Error ? err.message : "Failed to create the claim."
      );
    }
  }

  async function uploadOneDocument(e: React.FormEvent) {
    e.preventDefault();
    if (!pendingDocFile || !createdClaim) return;
    setDocUploading(true);
    setDocError(null);
    try {
      await uploadDocument(createdClaim.id, pendingDocFile, pendingDocType);
      setPendingDocFile(null);
      const input = document.getElementById("doc-file-input") as HTMLInputElement | null;
      if (input) input.value = "";
    } catch (err) {
      setDocError(err instanceof Error ? err.message : "Document upload failed.");
    } finally {
      setDocUploading(false);
    }
  }

  async function uploadImagesStep() {
    if (!pendingImages || pendingImages.length === 0 || !createdClaim) {
      setStep(4);
      // No images? Skip to done.
      if (createdClaim) navigate(`/claims/${createdClaim.id}`);
      return;
    }
    setImageUploading(true);
    setImageError(null);
    try {
      await uploadImages(createdClaim.id, Array.from(pendingImages));
      navigate(`/claims/${createdClaim.id}`);
    } catch (err) {
      setImageError(
        err instanceof Error ? err.message : "Image upload failed."
      );
    } finally {
      setImageUploading(false);
    }
  }

  if (loadingLookups) {
    return (
      <PageShell
        title="New Claim"
        description="Step-by-step claim submission."
      >
        <LoadingState label="Loading lookup data…" />
      </PageShell>
    );
  }

  if (lookupError) {
    return (
      <PageShell
        title="New Claim"
        description="Step-by-step claim submission."
      >
        <ErrorState
          message={lookupError}
          onRetry={() => {
            // Re-fetch the lookups in place rather than reloading the
            // whole page, which would discard any work the user had done.
            setLoadingLookups(true);
            loadLookups();
          }}
        />
      </PageShell>
    );
  }

  return (
    <PageShell
      title="New Claim"
      description="Step-by-step claim submission: customer & policy, vehicle & incident, documents, and images."
    >
      <div style={{ display: "flex", flexDirection: "column", gap: "var(--space-6)" }}>
        <Stepper current={step} />

        {step === 1 && (
          <section
            aria-label="Step 1: customer and policy"
            style={{
              display: "flex",
              flexDirection: "column",
              gap: "var(--space-4)",
              padding: "var(--space-4)",
              backgroundColor: "var(--color-surface)",
              border: "1px solid var(--color-border)",
              borderRadius: "var(--radius-md)",
            }}
          >
            <h2
              style={{
                fontSize: "var(--text-lg)",
                margin: 0,
                fontFamily: "var(--font-serif)",
                color: "var(--color-text-primary)",
              }}
            >
              Customer & policy
            </h2>

            {/* Customer */}
            <fieldset
              style={{
                border: "1px solid var(--color-border)",
                borderRadius: "var(--radius-md)",
                padding: "var(--space-3) var(--space-4)",
                display: "flex",
                flexDirection: "column",
                gap: "var(--space-3)",
                margin: 0,
              }}
            >
              <legend
                style={{
                  fontSize: "var(--text-xs)",
                  color: "var(--color-text-muted)",
                  textTransform: "uppercase",
                  letterSpacing: "0.05em",
                  padding: "0 var(--space-2)",
                }}
              >
                Customer
              </legend>
              <div style={{ display: "flex", gap: "var(--space-3)" }}>
                <label
                  style={{ display: "flex", alignItems: "center", gap: "var(--space-2)", fontSize: "var(--text-sm)", color: "var(--color-text-primary)" }}
                >
                  <input
                    type="radio"
                    name="customer-mode"
                    value="existing"
                    checked={customerMode === "existing"}
                    onChange={() => setCustomerMode("existing")}
                  />
                  Pick existing
                </label>
                <label
                  style={{ display: "flex", alignItems: "center", gap: "var(--space-2)", fontSize: "var(--text-sm)", color: "var(--color-text-primary)" }}
                >
                  <input
                    type="radio"
                    name="customer-mode"
                    value="new"
                    checked={customerMode === "new"}
                    onChange={() => setCustomerMode("new")}
                  />
                  Create new
                </label>
              </div>
              {customerMode === "existing" ? (
                <div>
                  <label htmlFor="customer-pick" style={labelStyle}>
                    Existing customer
                  </label>
                  <select
                    id="customer-pick"
                    value={selectedCustomerId}
                    onChange={(e) =>
                      setSelectedCustomerId(
                        e.target.value === "" ? "" : Number(e.target.value)
                      )
                    }
                    style={selectStyle}
                  >
                    <option value="">— pick one —</option>
                    {customers.map((c) => (
                      <option key={c.id} value={c.id}>
                        {c.name} ({c.email})
                      </option>
                    ))}
                  </select>
                </div>
              ) : (
                <div
                  style={{
                    display: "grid",
                    gridTemplateColumns: "1fr 1fr",
                    gap: "var(--space-3)",
                  }}
                >
                  <div>
                    <label style={labelStyle} htmlFor="new-cust-name">Name</label>
                    <input
                      id="new-cust-name"
                      type="text"
                      value={newCustomer.name}
                      onChange={(e) =>
                        setNewCustomer({ ...newCustomer, name: e.target.value })
                      }
                      style={inputStyle}
                    />
                  </div>
                  <div>
                    <label style={labelStyle} htmlFor="new-cust-email">Email</label>
                    <input
                      id="new-cust-email"
                      type="email"
                      value={newCustomer.email}
                      onChange={(e) =>
                        setNewCustomer({ ...newCustomer, email: e.target.value })
                      }
                      style={inputStyle}
                    />
                  </div>
                  <div style={{ gridColumn: "1 / -1" }}>
                    <label style={labelStyle} htmlFor="new-cust-phone">Phone (optional)</label>
                    <input
                      id="new-cust-phone"
                      type="tel"
                      value={newCustomer.phone}
                      onChange={(e) =>
                        setNewCustomer({ ...newCustomer, phone: e.target.value })
                      }
                      style={inputStyle}
                    />
                  </div>
                </div>
              )}
            </fieldset>

            {/* Policy */}
            <fieldset
              style={{
                border: "1px solid var(--color-border)",
                borderRadius: "var(--radius-md)",
                padding: "var(--space-3) var(--space-4)",
                display: "flex",
                flexDirection: "column",
                gap: "var(--space-3)",
                margin: 0,
              }}
            >
              <legend
                style={{
                  fontSize: "var(--text-xs)",
                  color: "var(--color-text-muted)",
                  textTransform: "uppercase",
                  letterSpacing: "0.05em",
                  padding: "0 var(--space-2)",
                }}
              >
                Policy
              </legend>
              <p
                style={{
                  margin: 0,
                  fontSize: "var(--text-xs)",
                  color: "var(--color-text-muted)",
                }}
              >
                Pick the policy this claim is filed against. If you don't see the
                policy you need, ask an admin to add it before submitting.
              </p>
              <div>
                <label htmlFor="policy-pick" style={labelStyle}>
                  Existing policy
                </label>
                <select
                  id="policy-pick"
                  value={selectedPolicyId}
                  onChange={(e) =>
                    setSelectedPolicyId(
                      e.target.value === "" ? "" : Number(e.target.value)
                    )
                  }
                  style={selectStyle}
                >
                  <option value="">— pick one —</option>
                  {filteredPolicies.map((p) => (
                    <option key={p.id} value={p.id}>
                      {p.policy_number} · {p.coverage_type} · ends {p.end_date}
                    </option>
                  ))}
                </select>
              </div>
            </fieldset>

            <div style={{ display: "flex", justifyContent: "flex-end", gap: "var(--space-3)" }}>
              <Link to="/claims" className={sharedStyles.retryButton}>
                Cancel
              </Link>
              <button type="button" onClick={goFromStep1ToStep2} style={primaryButton}>
                Next: vehicle & incident →
              </button>
            </div>
            {formError && (
              <div role="alert" style={formErrorBox}>
                {formError}
              </div>
            )}
          </section>
        )}

        {step === 2 && (
          <section
            aria-label="Step 2: vehicle and claim"
            style={{
              display: "flex",
              flexDirection: "column",
              gap: "var(--space-4)",
              padding: "var(--space-4)",
              backgroundColor: "var(--color-surface)",
              border: "1px solid var(--color-border)",
              borderRadius: "var(--radius-md)",
            }}
          >
            <h2
              style={{
                fontSize: "var(--text-lg)",
                margin: 0,
                fontFamily: "var(--font-serif)",
                color: "var(--color-text-primary)",
              }}
            >
              Vehicle & incident
            </h2>

            <fieldset
              style={{
                border: "1px solid var(--color-border)",
                borderRadius: "var(--radius-md)",
                padding: "var(--space-3) var(--space-4)",
                display: "flex",
                flexDirection: "column",
                gap: "var(--space-3)",
                margin: 0,
              }}
            >
              <legend
                style={{
                  fontSize: "var(--text-xs)",
                  color: "var(--color-text-muted)",
                  textTransform: "uppercase",
                  letterSpacing: "0.05em",
                  padding: "0 var(--space-2)",
                }}
              >
                Vehicle
              </legend>
              <div style={{ display: "flex", gap: "var(--space-3)" }}>
                <label
                  style={{ display: "flex", alignItems: "center", gap: "var(--space-2)", fontSize: "var(--text-sm)", color: "var(--color-text-primary)" }}
                >
                  <input
                    type="radio"
                    name="vehicle-mode"
                    value="existing"
                    checked={vehicleMode === "existing"}
                    onChange={() => setVehicleMode("existing")}
                  />
                  Pick existing
                </label>
                <label
                  style={{ display: "flex", alignItems: "center", gap: "var(--space-2)", fontSize: "var(--text-sm)", color: "var(--color-text-primary)" }}
                >
                  <input
                    type="radio"
                    name="vehicle-mode"
                    value="new"
                    checked={vehicleMode === "new"}
                    onChange={() => setVehicleMode("new")}
                  />
                  Create new
                </label>
              </div>
              {vehicleMode === "existing" ? (
                <div>
                  <label htmlFor="vehicle-pick" style={labelStyle}>
                    Existing vehicle
                  </label>
                  <select
                    id="vehicle-pick"
                    value={selectedVehicleId}
                    onChange={(e) =>
                      setSelectedVehicleId(
                        e.target.value === "" ? "" : Number(e.target.value)
                      )
                    }
                    style={selectStyle}
                  >
                    <option value="">— pick one —</option>
                    {vehicles.map((v) => (
                      <option key={v.id} value={v.id}>
                        {v.year} {v.make} {v.model}
                        {v.plate_number ? ` (${v.plate_number})` : ""}
                      </option>
                    ))}
                  </select>
                </div>
              ) : (
                <div
                  style={{
                    display: "grid",
                    gridTemplateColumns: "1fr 1fr",
                    gap: "var(--space-3)",
                  }}
                >
                  <div>
                    <label style={labelStyle} htmlFor="new-veh-make">Make</label>
                    <input
                      id="new-veh-make"
                      type="text"
                      value={newVehicle.make}
                      onChange={(e) =>
                        setNewVehicle({ ...newVehicle, make: e.target.value })
                      }
                      style={inputStyle}
                    />
                  </div>
                  <div>
                    <label style={labelStyle} htmlFor="new-veh-model">Model</label>
                    <input
                      id="new-veh-model"
                      type="text"
                      value={newVehicle.model}
                      onChange={(e) =>
                        setNewVehicle({ ...newVehicle, model: e.target.value })
                      }
                      style={inputStyle}
                    />
                  </div>
                  <div>
                    <label style={labelStyle} htmlFor="new-veh-year">Year</label>
                    <input
                      id="new-veh-year"
                      type="number"
                      min="1900"
                      max="2100"
                      value={newVehicle.year}
                      onChange={(e) =>
                        setNewVehicle({ ...newVehicle, year: e.target.value })
                      }
                      style={inputStyle}
                    />
                  </div>
                  <div>
                    <label style={labelStyle} htmlFor="new-veh-plate">Plate (optional)</label>
                    <input
                      id="new-veh-plate"
                      type="text"
                      value={newVehicle.plate_number}
                      onChange={(e) =>
                        setNewVehicle({ ...newVehicle, plate_number: e.target.value })
                      }
                      style={inputStyle}
                    />
                  </div>
                  <div style={{ gridColumn: "1 / -1" }}>
                    <label style={labelStyle} htmlFor="new-veh-vin">VIN (optional)</label>
                    <input
                      id="new-veh-vin"
                      type="text"
                      value={newVehicle.vin}
                      onChange={(e) =>
                        setNewVehicle({ ...newVehicle, vin: e.target.value })
                      }
                      style={inputStyle}
                    />
                  </div>
                </div>
              )}
            </fieldset>

            <fieldset
              style={{
                border: "1px solid var(--color-border)",
                borderRadius: "var(--radius-md)",
                padding: "var(--space-3) var(--space-4)",
                display: "flex",
                flexDirection: "column",
                gap: "var(--space-3)",
                margin: 0,
              }}
            >
              <legend
                style={{
                  fontSize: "var(--text-xs)",
                  color: "var(--color-text-muted)",
                  textTransform: "uppercase",
                  letterSpacing: "0.05em",
                  padding: "0 var(--space-2)",
                }}
              >
                Claim details
              </legend>
              <div
                style={{
                  display: "grid",
                  gridTemplateColumns: "1fr 1fr",
                  gap: "var(--space-3)",
                }}
              >
                <div>
                  <label style={labelStyle} htmlFor="claim-num">Claim number</label>
                  <input
                    id="claim-num"
                    type="text"
                    value={claimForm.claim_number}
                    onChange={(e) =>
                      setClaimForm({ ...claimForm, claim_number: e.target.value })
                    }
                    style={inputStyle}
                  />
                </div>
                <div>
                  <label style={labelStyle} htmlFor="claim-amount">Claimed amount (USD)</label>
                  <input
                    id="claim-amount"
                    type="number"
                    min="0"
                    step="0.01"
                    value={claimForm.claimed_amount}
                    onChange={(e) =>
                      setClaimForm({ ...claimForm, claimed_amount: e.target.value })
                    }
                    style={inputStyle}
                  />
                </div>
                <div>
                  <label style={labelStyle} htmlFor="claim-incident">Incident date</label>
                  <input
                    id="claim-incident"
                    type="date"
                    value={claimForm.incident_date}
                    onChange={(e) =>
                      setClaimForm({ ...claimForm, incident_date: e.target.value })
                    }
                    style={inputStyle}
                  />
                </div>
                <div>
                  <label style={labelStyle} htmlFor="claim-reported">Reported date</label>
                  <input
                    id="claim-reported"
                    type="date"
                    value={claimForm.reported_date}
                    onChange={(e) =>
                      setClaimForm({ ...claimForm, reported_date: e.target.value })
                    }
                    style={inputStyle}
                  />
                </div>
              </div>
            </fieldset>

            <div style={{ display: "flex", justifyContent: "space-between" }}>
              <button type="button" onClick={() => { setFormError(null); setStep(1); }} style={secondaryButton}>
                ← Back
              </button>
              <button type="button" onClick={goFromStep2ToStep3} style={primaryButton}>
                Create claim & continue →
              </button>
            </div>
            {formError && (
              <div role="alert" style={formErrorBox}>
                {formError}
              </div>
            )}
          </section>
        )}

        {step === 3 && createdClaim && (
          <section
            aria-label="Step 3: documents"
            style={{
              display: "flex",
              flexDirection: "column",
              gap: "var(--space-4)",
              padding: "var(--space-4)",
              backgroundColor: "var(--color-surface)",
              border: "1px solid var(--color-border)",
              borderRadius: "var(--radius-md)",
            }}
          >
            <h2
              style={{
                fontSize: "var(--text-lg)",
                margin: 0,
                fontFamily: "var(--font-serif)",
                color: "var(--color-text-primary)",
              }}
            >
              Documents
            </h2>
            <p
              style={{
                margin: 0,
                fontSize: "var(--text-sm)",
                color: "var(--color-text-secondary)",
              }}
            >
              Claim <code>{createdClaim.claim_number}</code> was created. You can
              upload supporting documents now, or skip and add them later from the
              Documents screen.
            </p>
            <form
              onSubmit={uploadOneDocument}
              style={{ display: "flex", gap: "var(--space-3)", alignItems: "center", flexWrap: "wrap" }}
            >
              <select
                value={pendingDocType}
                onChange={(e) => setPendingDocType(e.target.value as DocType)}
                style={selectStyle}
                aria-label="Document type"
              >
                {DOC_TYPES.map((t) => (
                  <option key={t} value={t}>
                    {DOC_TYPE_LABEL[t]}
                  </option>
                ))}
              </select>
              <input
                id="doc-file-input"
                type="file"
                accept="application/pdf,image/jpeg,image/png"
                onChange={(e) => setPendingDocFile(e.target.files?.[0] ?? null)}
                disabled={docUploading}
                style={{ color: "var(--color-text-primary)", fontSize: "var(--text-sm)" }}
              />
              <button type="submit" disabled={!pendingDocFile || docUploading} style={primaryButton}>
                {docUploading ? "Uploading…" : "Upload"}
              </button>
              {docError && (
                <span role="alert" style={{ color: "var(--color-risk-high)", fontSize: "var(--text-sm)" }}>
                  {docError}
                </span>
              )}
            </form>
            <div style={{ display: "flex", justifyContent: "space-between" }}>
              <button type="button" onClick={() => setStep(2)} style={secondaryButton}>
                ← Back
              </button>
              <button type="button" onClick={() => setStep(4)} style={primaryButton}>
                Next: images →
              </button>
            </div>
          </section>
        )}

        {step === 4 && createdClaim && (
          <section
            aria-label="Step 4: images"
            style={{
              display: "flex",
              flexDirection: "column",
              gap: "var(--space-4)",
              padding: "var(--space-4)",
              backgroundColor: "var(--color-surface)",
              border: "1px solid var(--color-border)",
              borderRadius: "var(--radius-md)",
            }}
          >
            <h2
              style={{
                fontSize: "var(--text-lg)",
                margin: 0,
                fontFamily: "var(--font-serif)",
                color: "var(--color-text-primary)",
              }}
            >
              Images
            </h2>
            <p
              style={{
                margin: 0,
                fontSize: "var(--text-sm)",
                color: "var(--color-text-secondary)",
              }}
            >
              Upload one or more accident photos. You can analyze them after the
              claim is created.
            </p>
            <input
              type="file"
              accept="image/jpeg,image/png"
              multiple
              onChange={(e) => setPendingImages(e.target.files)}
              disabled={imageUploading}
              style={{ color: "var(--color-text-primary)", fontSize: "var(--text-sm)" }}
            />
            {imageError && (
              <span role="alert" style={{ color: "var(--color-risk-high)", fontSize: "var(--text-sm)" }}>
                {imageError}
              </span>
            )}
            <div style={{ display: "flex", justifyContent: "space-between" }}>
              <button type="button" onClick={() => setStep(3)} style={secondaryButton}>
                ← Back
              </button>
              <button type="button" onClick={uploadImagesStep} disabled={imageUploading} style={primaryButton}>
                {imageUploading ? "Uploading…" : "Finish & open claim →"}
              </button>
            </div>
          </section>
        )}
      </div>
    </PageShell>
  );
}
