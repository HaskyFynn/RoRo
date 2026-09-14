# RoRo: RoEmotion Experiments

Standalone webcam console for **scene suppression** and **simultaneous tag identification**. Reads an existing QR code in the scene and includes the 9.9 MB tag model, transmitter sketches, automatic metrics/reports, and figure snapshots. No Android app, cloud service, training dataset, or emotion model is needed.

## Start

On Windows with Python 3.11-3.13 installed, double-click **install.cmd** once, then **launch.cmd**. The installer checks the bundled model and QR asset. For another OS, create a virtual environment, install `requirements.txt`, then run `python roemotion_pc_tool.py` (Tk must be installed).

1. **Camera:** run `launch.cmd`. The default camera is index 0; use `launch.cmd --camera 1` if needed.
2. **Exposure/gain:** enter the intended exposure label, camera exposure value, intended ISO/gain label and camera gain value, then click **Apply Camera Settings**. Blank values leave that camera control alone. The app logs camera readback values separately from the intended labels.
3. **Scene QR:** place an existing QR code in view and enter its expected payload. The integrated reader logs QR detection and exact payload match while exposure changes.
4. **Tags:** select the expected users. The bundled YOLO/TFLite receiver is enabled by default and classifies the visible rolling-shutter/OOK tag pattern as `User_1`, `User_2`, or `User_3`.
5. **Run:** choose duration, sampling interval and condition, then click **Start Timed Run**. Use **Snapshot** for figure frames.

Use any existing QR that the integrated reader can decode at baseline. The GUI displays the decoded value and logs exact matches as camera settings change. Keep that physical QR, its size and location fixed within an exposure sweep. There is no QR-generation step.

Exposure/gain controls, detector confidence, green threshold, expected users, QR payload, duration, sampling interval and condition notes are adjustable before recording. The 9.9 MB model retains its trained input size; changing its architecture would require a separately validated model.

The default console is intentionally simple and fast for experiments. `app.py` remains in the repository as the more structured region/reporting console, but `launch.cmd` opens `roemotion_pc_tool.py`.

`--qr-only` allows QR/blob pilots but disables YOLO tag IDs. The tag model is bundled and enabled by default; `--yolo` is accepted as a compatibility flag, and `--model-path PATH` is only needed if someone deliberately keeps the `.tflite` file outside `models/`.

## Quick Experiments

**1. Scene suppression:** place an existing QR in the scene. Fix the camera, QR, tag, lighting, gain, focus, white balance, resolution and frame rate. Measure QR width. First record a readable baseline, then 4-5 supported exposure settings spanning readable to suppressed scenes. Use **45 s x 3 repetitions** each; keep one known tag inside its region. Also record an LED-off control at baseline and the selected short exposure. Keep confidence and green threshold fixed after the pilot.

Primary outputs: exact QR payload recovery on original pixels and on original-or-CLAHE pixels, plus correct tag ID rate in the labelled region. The fixed contrast recovery probe helps distinguish a dark appearance from a QR that this reader cannot recover. Bright green region rate is a diagnostic and can respond to unrelated bright objects.

**2. Multi-user:** use the chosen exposure and verify all three tags individually on the webcam. Record each single user, each pair, and all three: **45 s x 3 repetitions** per configuration. Use the same gentle wrist movement within marked regions; log still and moving conditions separately. Rotate physical left/right positions between repetitions and redraw regions accordingly. Crossing or leaving regions invalidates this simple ground-truth protocol.

The 45-second default allows for slower laptops. A local software benchmark needed about 1.2 seconds per model invocation. Use a short pilot to check achieved analysis Hz, then choose a fixed duration per experiment that yields at least 30 analyzed samples per repetition. This is a practical floor, not a statistical power guarantee; a faster PC may allow shorter trials. The GUI flags runs below 30 samples. Live preview remains separate from inference.

Primary outputs: per-user correct, missed and wrong ID rates; all users correct on the same analyzed frame; stricter exact-frame success with no extra predictions; extra predictions per sample; processing latency and achieved analysis rate. The app evaluates these directly from region assignments.

## Outputs

Each simple-console run contains `metadata.json`, `log.csv`, saved sampled frames, snapshots and `session_summary.json`. The advanced console additionally produces JSONL/HTML reports and plots. All saved observations include intended exposure labels, camera exposure readback, intended ISO/gain label, camera gain readback, QR payload status, LED blob diagnostics and YOLO user detections.

Rebuild after interruption or on another PC: `python reporting.py runs`. All necessary observations are in each run folder. No unrecorded or reused prediction enters the denominator. Interrupted runs remain labelled. For paper statistics, use repetitions as the independent units; do not treat neighboring video frames as independent participants.

## Before Collection

- Read [one-day checklist](docs/ONE_DAY.md), [measurement definitions](docs/METRICS.md), and [transmitter/receiver audit](docs/LOGIC_AUDIT.md).
- Webcam controls are device-specific. The exposure label is an operator annotation, not a command in seconds. Verify manual exposure physically and record its source; gain control is not necessarily ISO. Some webcams cannot resolve these optical patterns.
- QR non-recovery only supports scene suppression for the selected QR, reader and recovery procedure. It does not establish privacy preservation or failure of other recovery methods.
- Original User 1 firmware emits a different green pattern from its label. Keep the original firmware for initial model compatibility checks; do not silently switch to the canonical pattern.

## Repository Contents

`roemotion_pc_tool.py` simple default GUI; `app.py` advanced GUI/capture; `core.py` detector/metrics; `reporting.py` reports; `assets/` static software-test/demo images; `firmware/original/` unchanged upstream sketches; `firmware/RoEmotionTransmitter/` configurable equivalent; `reference/` Android receiver/exposure source; `models/` actual model bytes and provenance.

Run software checks with `python check_install.py` and `python -m unittest discover -s tests -v`. See [verification record](docs/VERIFICATION.md) for what was actually tested.

Upload this directory to your separate GitHub repository. The model is small enough for ordinary Git; no LFS or external Drive download is needed. Runs and local settings are ignored by Git. The package contains no measured experiment results. Read [third-party notices](THIRD_PARTY_NOTICES.md) before choosing a public repository license.
