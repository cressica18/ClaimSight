/**
 * Screen 5: Image Analysis
 *
 * Blueprint (Section 11.1):
 * - Image gallery with per-image damage/severity chips
 * - Low-confidence warnings
 * - Upload and analyze flow
 *
 * Route: /claims/:id/images
 *
 * Implementation: Phase 5
 */
import { useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import PageShell from "../components/PageShell";
import {
  listImages,
  uploadImages,
  analyzeImage,
  analyzeAllImages,
} from "../api/client";
import { DamageResponse, CVAnalysisResult, DamageTypeResult } from "../types";
import styles from "./ImageAnalysis.module.css";

interface ImageWithAnalysis extends DamageResponse {
  analysis?: CVAnalysisResult;
  analyzing?: boolean;
  error?: string;
}

export default function ImageAnalysis() {
  const { id } = useParams<{ id: string }>();
  const claimId = Number(id);

  const [images, setImages] = useState<ImageWithAnalysis[]>([]);
  const [analyzingAll, setAnalyzingAll] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Load images on mount
  useEffect(() => {
    if (claimId) {
      loadImages();
    }
  }, [claimId]);

  const loadImages = async () => {
    try {
      setError(null);
      const data = await listImages(claimId);
      setImages(data.map((img) => ({ ...img, analysis: deriveAnalysis(img) })));
    } catch (err) {
      const message = err instanceof Error ? err.message : "Failed to load images";
      setError(message);
    }
  };

  const handleUpload = async (files: FileList | File[]) => {
    if (!files.length) return;
    setUploading(true);
    setError(null);
    try {
      const newImages = await uploadImages(claimId, Array.from(files));
      setImages((prev) => [...prev, ...newImages.map((img) => ({ ...img }))]);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Upload failed";
      setError(message);
    } finally {
      setUploading(false);
    }
  };

  const handleAnalyze = async (damageId: number, index: number) => {
    setImages((prev) =>
      prev.map((img, i) => (i === index ? { ...img, analyzing: true } : img))
    );
    try {
      const result = await analyzeImage(claimId, damageId);
      setImages((prev) =>
        prev.map((img, i) =>
          i === index ? { ...img, analysis: result, analyzing: false } : img
        )
      );
    } catch (err) {
      const message = err instanceof Error ? err.message : "Analysis failed";
      setImages((prev) =>
        prev.map((img, i) =>
          i === index ? { ...img, error: message, analyzing: false } : img
        )
      );
    }
  };

  const handleAnalyzeAll = async () => {
    setAnalyzingAll(true);
    setError(null);
    try {
      await analyzeAllImages(claimId);
      // The backend creates new Damage rows for each analyzed image rather
      // than updating the pending rows in place, so the batch response's
      // damage_id values do not match the original pending record ids in the
      // client state. Refetch the authoritative list to pick up the new
      // analyzed rows and drop the now-stale pending ones.
      await loadImages();
    } catch (err) {
      const message = err instanceof Error ? err.message : "Batch analysis failed";
      setError(message);
    } finally {
      setAnalyzingAll(false);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      handleUpload(e.target.files);
    }
  };

  const parseImagePath = (regionRef: string | null): string => {
    if (!regionRef) return "";
    try {
      const parsed = JSON.parse(regionRef);
      return parsed.image_path || "";
    } catch {
      return regionRef;
    }
  };

  const parseRegionMeta = (regionRef: string | null): Record<string, unknown> => {
    if (!regionRef) return {};
    try {
      const parsed = JSON.parse(regionRef);
      return typeof parsed === "object" && parsed !== null ? parsed : {};
    } catch {
      return {};
    }
  };

  // Build a CVAnalysisResult-shaped object from a single persisted Damage
  // row, so the UI can render the same analysis panel after a page reload
  // or after a batch re-analysis refetches the damage list. Returns
  // undefined for rows that are still in the `pending` state (no CV run yet).
  const deriveAnalysis = (img: ImageWithAnalysis): CVAnalysisResult | undefined => {
    if (!img.damage_type || img.damage_type === "pending" || img.damage_type === "cv_error") {
      return undefined;
    }
    const meta = parseRegionMeta(img.region_ref);
    return {
      damage_id: img.id,
      claim_id: img.claim_id,
      damage_types: [
        {
          label: img.damage_type,
          confidence: img.confidence ?? 0,
        },
      ],
      severity: {
        label: img.severity ?? "unknown",
        confidence: typeof meta.severity_confidence === "number" ? meta.severity_confidence : 0,
      },
      low_confidence: meta.low_confidence === true,
      source_image: typeof meta.image_path === "string" ? meta.image_path : null,
      model_version: typeof meta.model_version === "string" ? meta.model_version : "claimsight_cv_v1",
      timestamp: typeof meta.timestamp === "string" ? meta.timestamp : null,
      error: null,
    };
  };

  const getSeverityClass = (severity: string | null): string => {
    switch (severity?.toLowerCase()) {
      case "minor":
        return "minor";
      case "moderate":
        return "moderate";
      case "severe":
        return "severe";
      default:
        return "unknown";
    }
  };

  const getConfidenceClass = (confidence: number): "high" | "medium" | "low" => {
    if (confidence >= 0.7) return "high";
    if (confidence >= 0.4) return "medium";
    return "low";
  };

  const formatConfidence = (conf: number): string => {
    return `${Math.round(conf * 100)}%`;
  };

  if (!claimId) {
    return (
      <PageShell title="Image Analysis" description="Select a claim to view image analysis.">
        <div className={styles.emptyState}>
          <h3 className={styles.emptyStateTitle}>No Claim Selected</h3>
          <p className={styles.emptyStateDescription}>
            Navigate to a claim from the Claims list to analyze its images.
          </p>
        </div>
      </PageShell>
    );
  }

  return (
    <PageShell
      title={`Image Analysis — Claim #${claimId}`}
      description="Upload accident images and run CV damage detection. View per-image results with confidence scores."
    >
      {/* Error Banner */}
      {error && (
        <div className={styles.errorState} role="alert">
          <p className={styles.errorMessage}>{error}</p>
          <button className={`${styles.analyzeButton} ${styles.retryButton}`} onClick={loadImages}>
            Retry
          </button>
        </div>
      )}

      {/* Batch Actions */}
      {images.length > 0 && (
        <div className={styles.batchActions}>
          <span className={styles.batchActionsLabel}>
            {images.length} image{images.length !== 1 ? "s" : ""} uploaded
          </span>
          <button
            className={styles.analyzeButtonPrimary}
            onClick={handleAnalyzeAll}
            disabled={analyzingAll || images.every((img) => img.analysis)}
          >
            {analyzingAll ? (
              <>
                <span className={styles.loadingSpinner} />
                Analyzing all…
              </>
            ) : (
              "Analyze All"
            )}
          </button>
          <label className={`${styles.analyzeButtonSecondary} ${uploading ? styles.disabledLabel : ""}`}>
            {uploading ? (
              <>
                <span className={styles.loadingSpinner} />
                Uploading…
              </>
            ) : (
              "Add Images"
            )}
            <input type="file" accept="image/jpeg,image/png" multiple onChange={handleFileChange} hidden disabled={uploading} />
          </label>
        </div>
      )}

      {/* Empty State */}
      {images.length === 0 && !error && (
        <div className={styles.emptyState}>
          <h3 className={styles.emptyStateTitle}>No Images Yet</h3>
          <p className={styles.emptyStateDescription}>
            Upload accident photos to begin damage analysis. The CV model will detect
            damage types and assess severity for each image.
          </p>
          <label className={`${styles.analyzeButtonPrimary} ${uploading ? styles.disabledLabel : ""}`}>
            {uploading ? (
              <>
                <span className={styles.loadingSpinner} />
                Uploading…
              </>
            ) : (
              "Upload First Image"
            )}
            <input type="file" accept="image/jpeg,image/png" multiple onChange={handleFileChange} hidden disabled={uploading} />
          </label>
        </div>
      )}

      {/* Image Grid */}
      {images.length > 0 && (
        <div className={styles.imageGrid} role="list">
          {images.map((img, index) => {
            const imagePath = parseImagePath(img.region_ref);
            const imageUrl = imagePath ? `/api/${imagePath}` : "";
            const isAnalyzed = !!img.analysis;
            const hasError = !!img.error;
            const isPending = !isAnalyzed && img.damage_type === "pending";

            return (
              <article
                key={img.id}
                className={`${styles.imageCard} ${
                  img.analyzing
                    ? styles.analyzing
                    : hasError
                    ? styles.error
                    : isAnalyzed
                    ? styles.analyzed
                    : ""
                }`}
                role="listitem"
              >
                <img
                  src={imageUrl}
                  alt={`Claim ${claimId} image ${index + 1}`}
                  className={styles.imagePreview}
                  onError={(e) => {
                    e.currentTarget.style.display = "none";
                  }}
                />
                {!imageUrl && (
                  <div
                    className={styles.imagePreview}
                    style={{
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      backgroundColor: "var(--color-surface-raised)",
                      color: "var(--color-text-muted)",
                    }}
                  >
                    Image not available
                  </div>
                )}

                <div className={styles.imageCardContent}>
                  <div className={styles.imageMeta}>
                    <span className={styles.imageFilename} title={imagePath}>
                      {imagePath.split("/").pop() || `Image ${index + 1}`}
                    </span>
                    {isPending && !img.analyzing && (
                      <button
                        className={styles.analyzeButtonPrimary}
                        onClick={() => handleAnalyze(img.id, index)}
                        disabled={img.analyzing}
                      >
                        {img.analyzing ? (
                          <>
                            <span className={styles.loadingSpinner} />
                            Analyzing…
                          </>
                        ) : (
                          "Analyze"
                        )}
                      </button>
                    )}
                  </div>

                  {/* Analysis Results */}
                  {isAnalyzed && img.analysis && (
                    <div className={styles.resultsSection}>
                      <div className={styles.resultHeader}>
                        <span
                          className={`${styles.severityBadge} ${getSeverityClass(
                            img.analysis.severity.label
                          )}`}
                        >
                          {img.analysis.severity.label}
                        </span>
                        {img.analysis.low_confidence && (
                          <span className={styles.lowConfidenceBadge}>
                            Low Confidence
                          </span>
                        )}
                      </div>

                      {/* Damage Types */}
                      {img.analysis.damage_types.length > 0 && (
                        <div className={styles.damageTypesList}>
                          {img.analysis.damage_types.map((dt: DamageTypeResult) => (
                            <div
                              key={dt.label}
                              className={`${styles.damageChip} ${
                                dt.label === "no_damage" ? styles.noDamage : ""
                              }`}
                            >
                              <span>
                                {dt.label.replace(/_/g, " ")}
                              </span>
                              <span className={`${styles.damageChip} ${styles.confidenceChip}`}>
                                {formatConfidence(dt.confidence)}
                              </span>
                            </div>
                          ))}
                        </div>
                      )}

                      {/* Confidence Bars for Damage Types */}
                      {img.analysis.damage_types.length > 0 && (
                        <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
                          {img.analysis.damage_types.map((dt: DamageTypeResult) => (
                            <div key={dt.label} style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                              <span style={{ fontSize: "0.75rem", color: "var(--color-text-secondary)", minWidth: "100px" }}>
                                {dt.label.replace(/_/g, " ")}
                              </span>
                              <div className={styles.confidenceBar}>
                                <div
                                  className={`${styles.confidenceBarFill} ${getConfidenceClass(dt.confidence)}`}
                                  style={{ width: `${dt.confidence * 100}%` }}
                                />
                              </div>
                              <span style={{ fontSize: "0.75rem", fontWeight: 500, color: "var(--color-text-primary)", minWidth: "45px" }}>
                                {formatConfidence(dt.confidence)}
                              </span>
                            </div>
                          ))}
                        </div>
                      )}

                      {/* Severity Confidence */}
                      <div className={styles.detailRow}>
                        <span className={styles.detailLabel}>Severity Confidence</span>
                        <span className={styles.detailValue}>
                          {formatConfidence(img.analysis.severity.confidence)}
                        </span>
                      </div>

                      {/* Model Info */}
                      <div className={styles.modelInfo}>
                        <span className={styles.modelInfoItem}>
                          Model: {img.analysis.model_version}
                          {img.analysis.model_version
                            ?.toLowerCase()
                            .startsWith("demo") && (
                            <span
                              className={styles.demoTag}
                              title="This result was produced by the deterministic demo predictor, not a trained model. The model is identified by its model_version string."
                            >
                              demo
                            </span>
                          )}
                        </span>
                        {img.analysis.timestamp && (
                          <span className={styles.modelInfoItem}>
                            Analyzed: {new Date(img.analysis.timestamp).toLocaleString()}
                          </span>
                        )}
                      </div>

                      <button
                        className={styles.analyzeButtonSecondary}
                        onClick={() => handleAnalyze(img.id, index)}
                        style={{ marginTop: "var(--space-2)" }}
                      >
                        Re-analyze
                      </button>
                    </div>
                  )}

                  {/* Analysis Error */}
                  {hasError && (
                    <div className={styles.errorState} style={{ padding: "var(--space-4)" }}>
                      <p className={styles.errorMessage}>{img.error}</p>
                      <button
                        className={styles.analyzeButtonSecondary}
                        onClick={() => handleAnalyze(img.id, index)}
                      >
                        Retry
                      </button>
                    </div>
                  )}

                  {/* Pending State */}
                  {isPending && !hasError && (
                    <p style={{ color: "var(--color-text-muted)", fontSize: "var(--text-sm)" }}>
                      Uploaded — click "Analyze" to run CV detection
                    </p>
                  )}
                </div>
              </article>
            );
          })}
        </div>
      )}
    </PageShell>
  );
}