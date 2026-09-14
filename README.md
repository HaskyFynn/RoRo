# RoEmotion PC Experiment Console

This tool is tailored for the two quick RoEmotion experiments:

- Experiment 1: scene suppression versus active LED tag utility
- Experiment 2: simultaneous multi-user tag identification

The GUI includes the QR reader, LED visibility detector, optional RoEmotion YOLO user detector, run timer, condition labels, and CSV logging. No separate QR reader is needed.

## Install

```powershell
cd pc_experiment_tool
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Optional YOLO user identification:

```powershell
pip install -r requirements-yolo.txt
```

If `ai-edge-litert` does not work on the experiment laptop:

```powershell
pip install tensorflow
```

## Print The QR Board

```powershell
python make_scene_board.py
```

Print `scene_board.png`. The QR payload is:

```text
ROEMOTION_SCENE_SUPPRESSION_TEST
```

Keep the same printed board for all exposure settings.

## Run The GUI

Without YOLO:

```powershell
python roemotion_pc_tool.py
```

With YOLO:

```powershell
python roemotion_pc_tool.py --model models\RoEmotion_LED_Recognition_Model_YOLOv26n.tflite
```

Stricter validation-style YOLO threshold:

```powershell
python roemotion_pc_tool.py --model models\RoEmotion_LED_Recognition_Model_YOLOv26n.tflite --confidence 0.25
```

If the wrong camera opens:

```powershell
python roemotion_pc_tool.py --camera 1
```

## Experiment 1 Workflow

1. Select `1 Scene suppression`.
2. Place one LED wristband in front of the printed board without covering the QR code.
3. Enter the intended exposure label, e.g. `1/1000 s`.
4. Enter the webcam exposure value if the camera supports manual exposure.
5. Keep distance, lighting, frame rate, board, and wristband fixed.
6. Confirm the expected QR payload is `ROEMOTION_SCENE_SUPPRESSION_TEST`.
7. Use a 10-second duration and three repetitions per exposure setting.
8. Start the timed run.

Interpretation:

- QR match rate estimates scene recoverability.
- LED visibility and YOLO user detection estimate active-tag utility.
- Good settings have low QR match rate and usable LED/tag detection.

## Experiment 2 Workflow

1. Select `2 Multi-user tags`.
2. Check the expected users present in the camera view.
3. Use the exposure setting chosen from Experiment 1.
4. Use 30 seconds per trial.
5. Record single-user trials, all two-user pairs, then all three users.
6. During each run: 10 seconds still, 10 seconds guided small wrist motion, 10 seconds natural motion.

Interpretation:

- `yolo_all_expected_rate` is the main multi-user metric.
- Per-user rates show which tag is weak.
- The CSV keeps full per-frame user predictions and bounding boxes.

## Outputs

Each run creates a folder under `runs/` containing:

- `metadata.json`: trial settings
- `log.csv`: per-frame detections
- `session_summary.json`: quick run-level rates
- `frames/`: sampled frames for paper figures
- `snapshots/`: manual snapshots

Summarize all runs:

```powershell
python summarize_runs.py
```

This writes `runs_summary.csv`.

## Camera Note

Webcam exposure values are device-specific in OpenCV. Report both:

- intended exposure duration, e.g. `1/6000 s`
- camera-reported exposure value from the GUI/logs

Do not call exposure duration, frame rate, LED modulation frequency, and rolling-shutter row timing the same thing.
# RoRo
