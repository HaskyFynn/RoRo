# Measurement Definitions

One observation is one distinct camera frame analyzed once. Every expected wearer has a fixed, independently labelled region. Every selected tag must remain in that region throughout the timed trial.

| Output | Numerator / denominator |
|---|---|
| QR localization | Frames with QR corner points / scene samples |
| QR raw recovery | Exact expected payload from original QR crop / scene samples |
| QR enhanced recovery | Exact expected payload after fixed CLAHE / scene samples |
| QR any recovery | Raw OR enhanced exact recovery / scene samples |
| Per-user correct ID | Region's selected prediction equals its physical user / ID samples |
| Per-user miss | No prediction centered in region / ID samples |
| Per-user wrong ID | Prediction exists but has a different user label / ID samples |
| Accuracy given detection | Correct / non-missed region predictions |
| All correct | Every expected user's region is correctly labelled / ID samples |
| Exact frame | All correct and zero remaining predictions / ID samples |
| Extra detections/sample | Unassigned and duplicate predictions / ID samples |
| Bright region rate | At least 5 connected green-channel pixels >= threshold / samples |

Miss is a detector failure, not proof the physical tag disappeared. Bright region rate is a diagnostic, not identification. Extras are not an exhaustive false-positive measure: a wrong prediction inside a region is captured by wrong-ID rate. No full precision/recall or bounding-box accuracy is claimed without independent per-frame box annotations. Confusion counts include `miss`. Correct + wrong + miss = 1 for each evaluated user. No model or no observations produces null/N/A, not a fabricated 0%.

Latency median/p95 measures preprocessing + QR probes + tag inference + scoring. `frame_age_ms` is host read-completion to analysis completion, excluding later saving/display. It is not sensor-to-display latency. `observed_sample_hz` uses distinct sample timestamps, while `camera_start/end.delivered_hz` describes camera reads. Target analysis Hz is an upper bound, especially on slower laptops. Do not interpret reduced analysis throughput as a tag miss rate.

Every report preserves settings, model hash and software versions. JSONL contains complete predictions, regions via metadata, QR payloads and timestamps. CSV is a convenient flat view; JSONL is authoritative. Rebuilding after a crash excludes malformed lines with explicit warnings and marks missing completion as interrupted. Keep originals backed up before analysis.

For paper figures, report the mean and spread across three repetitions at each matched condition, with each run as one unit. Inspect repetition values; three repetitions do not establish broad population generalization. Do not mix distances, lighting, confidence, resolution, firmware, motion, or different camera control readbacks. The automatic comparison deliberately plots each repetition separately and does not silently pool unlike settings. QR and ID use the same scene-sample frames. Multi-user runs omit QR metrics (N/A).
