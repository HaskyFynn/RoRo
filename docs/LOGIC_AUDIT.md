# Transmitter / Receiver Audit

Upstream revision: `06221e23ea8d24df7656069b3ec3d3974019e62d` of https://github.com/denizkarya1999/RoEmotion.

## Transmitters

The original sketches use XIAO ESP32C3 aliases D2 (red) and D4 (green). They write the first slot to **red**, not green. Red stays LOW throughout; at steady state green holds its previous LOW in that first slot. Assuming active-high LEDs, the five green slots are:

| Physical user | Model label | Original green slots |
|---|---|---|
| 1 | ook_10101 | 01010 |
| 2 | ook_01110 | 01110 |
| 3 | ook_00110 | 00110 |

User 1 is a real comment/waveform mismatch. The code alone cannot establish which firmware produced the training images. Repeating codewords have no explicit start marker, so observed phase is arbitrary. Preserve the upstream sketches as an initial reference. `RoEmotionTransmitter.ino` explicitly writes the equivalent steady-state green sequence by default; set `USER_ID` for each board. It initializes both pins LOW. Its loop overhead differs slightly from the original, so record firmware choice and measure actual timing before comparisons. The optional `CANONICAL_LABEL_PATTERN=1` changes User 1 to 10101 and requires renewed model validation; it is off by default.

Each slot requests 166 microseconds: ideal symbol rate is about 6024 symbols/s, and five-slot repetition about 1205 cycles/s. GPIO calls, loop overhead and scheduling add time/jitter. This is a nominal OOK symbol rate, not camera frame rate or an exact 6 kHz square-wave frequency. No timer-accuracy claim is made. Verify green output using a photodiode/oscilloscope or logic analyzer when available. LED polarity and wiring must match the actual hardware; use appropriate current limiting.

The simple configurable sketch is supplied for inspection/convenience. It was not flashed or hardware-timed in this software verification.

## Receiver

The upstream receiver classifies optical images; it does not recover bits through a protocol decoder. The bundled model has float32 input `[1,3,960,960]`, RGB/255 with centered 114-valued letterboxing, and output `[1,300,6]` containing xyxy, confidence, class index. The PC implementation checks this contract, reverses letterboxing and applies per-class NMS at IoU 0.45. It retains duplicate-class boxes in different locations so they can be counted. The old per-class maximum would conceal duplicates.

Default confidence is 0.25 (validation-style), compared with 0.001 in the Android live settings. This is a deliberate PC evaluation setting; tune only on pilots, freeze before trials, and log it. The code retains the Android score sanitization convention for values outside [0,1], rejects nonfinite rows, and accepts pixel or normalized boxes using the upstream heuristic. The supplied model checksum is recorded per run.

Ground truth comes from manually labelled, nonoverlapping wrist regions. Highest-confidence detection centered in a region is its prediction, irrespective of predicted class. Wrong labels therefore remain errors. Remaining detections are extras. This is a controlled region experiment, not a general tracking or occlusion benchmark; do not allow crossings. Background inside a region can still fool detection, which is why regions should be tight and LED-off controls are included.

Camera readout runs continuously in its own thread. Analysis consumes the latest distinct frame at up to the requested analysis rate. No cached inference is reused as a new sample. Capture delivery, requested frame rate, achieved analysis rate and processing latency are recorded separately. Timestamps are host read-completion times, not sensor exposure timestamps. Frames skipped between analyses are not scored; this is sampled real-time evaluation, not exhaustive frame accuracy. PNG interval snapshots are selected independently of success.

## Camera Terminology

Exposure duration is the integration time of an individual row. Frame rate is complete frames delivered per second. LED modulation here is a five-symbol OOK sequence with a nominal symbol interval. Rolling-shutter row timing is the time offset between row starts; it requires sensor documentation or calibration. Neither frame rate nor exposure duration determines row timing on its own. The Android 6000 shutter setting requests approximately 1/6000 s exposure, not a 6000 fps camera.

Webcam control units and manual/auto encodings vary by backend/device. The PC app accepts raw controls and records set-return values and readback; it does not assert they represent seconds, ISO, or successful manual operation. Use the manufacturer's control panel/docs and a visible exposure sweep. Lock focus/white balance/gain where supported. Log unsupported controls and unknown row timing. A wider view may shrink the LED enough to lose stripe resolution, and a mobile-trained model may fail on the webcam. Verify this before collecting all conditions.

QR detection/decoding uses [OpenCV QRCodeDetector](https://docs.opencv.org/4.13.0/de/dc3/classcv_1_1QRCodeDetector.html) on a fixed QR crop. Localization is separate from successful payload recovery. The enhancement probe always applies the same grayscale CLAHE configuration; it is only one recovery attempt, not an adversarial privacy test.
