import argparse
import csv
import datetime as dt
import json
import time
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk


USER_LABELS = ["User_1", "User_2", "User_3"]
MODEL_LABELS = ["ook_10101", "ook_01110", "ook_00110"]
SCENE_QR_PAYLOAD = "ROEMOTION_SCENE_SUPPRESSION_TEST"


class OptionalYoloDetector:
    def __init__(self, model_path: Path, confidence_threshold: float = 0.001):
        self.confidence_threshold = confidence_threshold
        self.input_size = 960
        self.interpreter = self._load_interpreter(model_path)
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        input_shape = list(self.input_details[0]["shape"])
        if len(input_shape) == 4:
            self.input_size = int(max(input_shape[2], input_shape[3]))

    @staticmethod
    def _load_interpreter(model_path: Path):
        try:
            from tflite_runtime.interpreter import Interpreter
        except ImportError:
            try:
                from ai_edge_litert.interpreter import Interpreter
            except ImportError:
                try:
                    from tensorflow.lite.python.interpreter import Interpreter
                except ImportError as error:
                    raise RuntimeError(
                        "No TFLite runtime found. Install ai-edge-litert or TensorFlow to enable YOLO inference."
                    ) from error

        interpreter = Interpreter(model_path=str(model_path))
        interpreter.allocate_tensors()
        return interpreter

    def detect(self, frame_bgr):
        original_h, original_w = frame_bgr.shape[:2]
        input_tensor, meta = self._preprocess(frame_bgr)
        self.interpreter.set_tensor(self.input_details[0]["index"], input_tensor)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]["index"])

        detections = []
        for row in np.asarray(output).reshape((-1, 6)):
            x1, y1, x2, y2, score, class_id = row.tolist()
            score = self._sanitize_probability(score)
            class_id = int(class_id) if np.isfinite(class_id) else -1
            if score < self.confidence_threshold or class_id < 0 or class_id >= len(USER_LABELS):
                continue
            if x2 <= x1 or y2 <= y1:
                continue
            detections.append({
                "class_id": class_id,
                "user": USER_LABELS[class_id],
                "ook": MODEL_LABELS[class_id],
                "confidence": score,
                "bbox": self._rescale_box((x1, y1, x2, y2), meta, original_w, original_h),
            })

        return self._pick_one_per_class(sorted(detections, key=lambda det: det["confidence"], reverse=True))

    def _preprocess(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        scale = min(self.input_size / w, self.input_size / h)
        resized_w = max(1, int(round(w * scale)))
        resized_h = max(1, int(round(h * scale)))
        pad_left = (self.input_size - resized_w) // 2
        pad_top = (self.input_size - resized_h) // 2

        resized = cv2.resize(frame_bgr, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        canvas[pad_top:pad_top + resized_h, pad_left:pad_left + resized_w] = resized
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        nchw = np.transpose(rgb, (2, 0, 1))[None, ...].astype(np.float32)
        return nchw, {"scale": scale, "pad_left": pad_left, "pad_top": pad_top}

    def _rescale_box(self, box, meta, original_w, original_h):
        x1, y1, x2, y2 = box
        if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.5:
            x1 *= self.input_size
            x2 *= self.input_size
            y1 *= self.input_size
            y2 *= self.input_size

        x1 = int(np.clip((x1 - meta["pad_left"]) / meta["scale"], 0, original_w))
        x2 = int(np.clip((x2 - meta["pad_left"]) / meta["scale"], 0, original_w))
        y1 = int(np.clip((y1 - meta["pad_top"]) / meta["scale"], 0, original_h))
        y2 = int(np.clip((y2 - meta["pad_top"]) / meta["scale"], 0, original_h))
        return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)

    @staticmethod
    def _pick_one_per_class(detections):
        best = {}
        for det in detections:
            current = best.get(det["class_id"])
            if current is None or det["confidence"] > current["confidence"]:
                best[det["class_id"]] = det
        return [best[class_id] for class_id in sorted(best)]

    @staticmethod
    def _sanitize_probability(value):
        if not np.isfinite(value):
            return 0.0
        if value < 0.0 or value > 1.0:
            value = 1.0 / (1.0 + np.exp(-np.clip(value, -30.0, 30.0)))
        return float(np.clip(value, 0.0, 1.0))


class SessionStats:
    def __init__(self):
        self.reset()

    def reset(self):
        self.frames = 0
        self.qr_detected = 0
        self.qr_matched = 0
        self.led_frames = 0
        self.yolo_all_expected = 0
        self.yolo_user_hits = {user: 0 for user in USER_LABELS}

    def pct(self, count):
        return 0.0 if self.frames == 0 else 100.0 * count / self.frames


class RoEmotionPCTool:
    def __init__(
        self,
        camera_index: int,
        output_dir: Path,
        model_path: Path | None,
        confidence_threshold: float,
    ):
        self.camera_index = camera_index
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.confidence_threshold = confidence_threshold
        self.yolo = self._load_yolo(model_path)

        self.cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera index {camera_index}")

        self.qr = cv2.QRCodeDetector()
        self.root = tk.Tk()
        self.root.title("RoEmotion Experiment Console")

        self.mode_var = tk.StringVar(value="scene")
        self.trial_var = tk.StringVar(value="1")
        self.exposure_label_var = tk.StringVar(value="1/6000 s")
        self.camera_exposure_var = tk.StringVar(value="")
        self.iso_label_var = tk.StringVar(value="ISO100")
        self.camera_gain_var = tk.StringVar(value="")
        self.distance_var = tk.StringVar(value="1.5 m")
        self.angle_var = tk.StringVar(value="front")
        self.motion_var = tk.StringVar(value="still")
        self.qr_expected_var = tk.StringVar(value=SCENE_QR_PAYLOAD)
        self.duration_var = tk.IntVar(value=10)
        self.threshold_var = tk.IntVar(value=230)
        self.sample_every_var = tk.IntVar(value=15)
        self.yolo_every_var = tk.IntVar(value=3)
        self.expected_user_vars = {
            user: tk.BooleanVar(value=(user == "User_1"))
            for user in USER_LABELS
        }
        self.condition_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="Ready")
        self.metrics_var = tk.StringVar(value="Frames 0 | QR match 0.0% | LED 0.0% | YOLO all-users 0.0%")
        self.yolo_status_var = tk.StringVar(
            value="YOLO enabled" if self.yolo else "YOLO unavailable; using LED blobs"
        )

        self.session_dir = None
        self.csv_file = None
        self.csv_writer = None
        self.logging = False
        self.frame_index = 0
        self.preview_index = 0
        self.session_started_at = None
        self.stop_at = None
        self.last_frame = None
        self.photo = None
        self.last_yolo_detections = []
        self.stats = SessionStats()

        self._build_ui()
        self._sync_mode_defaults()
        self._update_frame()

    def _load_yolo(self, model_path: Path | None):
        if not model_path:
            return None
        if not model_path.exists():
            raise RuntimeError(f"YOLO model not found: {model_path}")
        try:
            return OptionalYoloDetector(model_path, confidence_threshold=self.confidence_threshold)
        except Exception as error:
            print(f"YOLO disabled: {error}")
            return None

    def _build_ui(self):
        self.video_label = tk.Label(self.root)
        self.video_label.grid(row=0, column=0, columnspan=2, padx=8, pady=8, sticky="nsew")

        panel = ttk.Frame(self.root)
        panel.grid(row=0, column=2, padx=8, pady=8, sticky="nsew")

        experiment = ttk.LabelFrame(panel, text="Experiment")
        experiment.grid(row=0, column=0, sticky="we", pady=(0, 8))
        ttk.Radiobutton(
            experiment,
            text="1 Scene suppression",
            value="scene",
            variable=self.mode_var,
            command=self._sync_mode_defaults,
        ).grid(row=0, column=0, sticky="w", padx=6, pady=3)
        ttk.Radiobutton(
            experiment,
            text="2 Multi-user tags",
            value="multi",
            variable=self.mode_var,
            command=self._sync_mode_defaults,
        ).grid(row=1, column=0, sticky="w", padx=6, pady=3)
        ttk.Label(experiment, text="Condition").grid(row=2, column=0, sticky="w", padx=6)
        ttk.Entry(experiment, textvariable=self.condition_var, width=34).grid(
            row=3, column=0, sticky="we", padx=6, pady=(0, 6)
        )

        capture = ttk.LabelFrame(panel, text="Camera And Trial")
        capture.grid(row=1, column=0, sticky="we", pady=(0, 8))
        self._add_labeled_entry(capture, "Intended exposure", self.exposure_label_var, 0)
        self._add_labeled_entry(capture, "Camera exposure value", self.camera_exposure_var, 1)
        self._add_labeled_entry(capture, "Intended ISO/gain", self.iso_label_var, 2)
        self._add_labeled_entry(capture, "Camera gain value", self.camera_gain_var, 3)
        self._add_labeled_entry(capture, "Trial", self.trial_var, 4)
        self._add_labeled_entry(capture, "Distance", self.distance_var, 5)
        self._add_labeled_entry(capture, "Angle", self.angle_var, 6)
        self._add_labeled_entry(capture, "Motion", self.motion_var, 7)
        ttk.Button(capture, text="Apply Camera Settings", command=self.apply_camera_settings).grid(
            row=8, column=0, columnspan=2, sticky="we", padx=6, pady=6
        )

        scene = ttk.LabelFrame(panel, text="Experiment 1 QR Probe")
        scene.grid(row=2, column=0, sticky="we", pady=(0, 8))
        ttk.Label(scene, text="Expected QR payload").grid(row=0, column=0, sticky="w", padx=6, pady=(4, 0))
        ttk.Entry(scene, textvariable=self.qr_expected_var, width=34).grid(
            row=1, column=0, sticky="we", padx=6, pady=(0, 6)
        )
        ttk.Label(
            scene,
            text="Use the same printed QR code across all exposure settings. QR match rate is the scene-recoverability metric.",
            wraplength=280,
        ).grid(row=2, column=0, sticky="we", padx=6, pady=(0, 6))

        users = ttk.LabelFrame(panel, text="Experiment 2 Expected Users")
        users.grid(row=3, column=0, sticky="we", pady=(0, 8))
        for index, user in enumerate(USER_LABELS):
            ttk.Checkbutton(
                users,
                text=f"{user} ({MODEL_LABELS[index]})",
                variable=self.expected_user_vars[user],
            ).grid(row=index, column=0, sticky="w", padx=6, pady=2)

        logging = ttk.LabelFrame(panel, text="Logging")
        logging.grid(row=4, column=0, sticky="we", pady=(0, 8))
        ttk.Label(logging, text="Duration seconds").grid(row=0, column=0, sticky="w", padx=6)
        ttk.Entry(logging, textvariable=self.duration_var, width=8).grid(row=0, column=1, sticky="w", padx=6)
        ttk.Label(logging, text="Save every N frames").grid(row=1, column=0, sticky="w", padx=6)
        ttk.Entry(logging, textvariable=self.sample_every_var, width=8).grid(row=1, column=1, sticky="w", padx=6)
        ttk.Label(logging, text="YOLO every N frames").grid(row=2, column=0, sticky="w", padx=6)
        ttk.Entry(logging, textvariable=self.yolo_every_var, width=8).grid(row=2, column=1, sticky="w", padx=6)
        ttk.Label(logging, text="LED threshold").grid(row=3, column=0, sticky="w", padx=6)
        tk.Scale(logging, from_=120, to=255, orient="horizontal", variable=self.threshold_var).grid(
            row=3, column=1, sticky="we", padx=6
        )
        ttk.Button(logging, text="Start Timed Run", command=self.start_logging).grid(
            row=4, column=0, sticky="we", padx=6, pady=6
        )
        ttk.Button(logging, text="Stop", command=self.stop_logging).grid(row=4, column=1, sticky="we", padx=6, pady=6)
        ttk.Button(logging, text="Snapshot", command=self.save_snapshot).grid(
            row=5, column=0, columnspan=2, sticky="we", padx=6, pady=(0, 6)
        )

        ttk.Label(panel, textvariable=self.yolo_status_var).grid(row=5, column=0, sticky="we", pady=(0, 4))
        ttk.Label(panel, textvariable=self.metrics_var, wraplength=300).grid(row=6, column=0, sticky="we", pady=(0, 4))
        ttk.Label(panel, textvariable=self.status_var, wraplength=300).grid(row=7, column=0, sticky="we")

        for col in range(3):
            self.root.grid_columnconfigure(col, weight=1 if col < 2 else 0)
        self.root.grid_rowconfigure(0, weight=1)
        panel.grid_columnconfigure(0, weight=1)

    @staticmethod
    def _add_labeled_entry(parent, label, variable, row):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=6, pady=2)
        ttk.Entry(parent, textvariable=variable, width=18).grid(row=row, column=1, sticky="we", padx=6, pady=2)

    def _sync_mode_defaults(self):
        if self.mode_var.get() == "scene":
            self.duration_var.set(10)
            self.motion_var.set("still")
            self.expected_user_vars["User_1"].set(True)
            self.expected_user_vars["User_2"].set(False)
            self.expected_user_vars["User_3"].set(False)
        else:
            self.duration_var.set(30)
            self.motion_var.set("10s still + 10s guided + 10s natural")
        self._update_condition_label()

    def _update_condition_label(self):
        mode = "E1_scene" if self.mode_var.get() == "scene" else "E2_multi"
        expected = "-".join(self.expected_users()) or "no_users"
        exposure = self.exposure_label_var.get().replace("/", "over").replace(" ", "")
        trial = self.trial_var.get().strip() or "1"
        self.condition_var.set(f"{mode}_{exposure}_{self.iso_label_var.get()}_{expected}_r{trial}")

    def expected_users(self):
        return [user for user, var in self.expected_user_vars.items() if var.get()]

    def apply_camera_settings(self):
        exposure_text = self.camera_exposure_var.get().strip()
        gain_text = self.camera_gain_var.get().strip()

        if exposure_text:
            try:
                self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
                self.cap.set(cv2.CAP_PROP_EXPOSURE, float(exposure_text))
            except ValueError:
                messagebox.showerror("Invalid exposure", "Camera exposure value must be a number.")
                return

        if gain_text:
            try:
                self.cap.set(cv2.CAP_PROP_GAIN, float(gain_text))
            except ValueError:
                messagebox.showerror("Invalid gain", "Camera gain value must be a number.")
                return

        self.status_var.set(
            "Camera reports "
            f"exposure={self.cap.get(cv2.CAP_PROP_EXPOSURE):.3f}, "
            f"gain={self.cap.get(cv2.CAP_PROP_GAIN):.3f}, "
            f"fps={self.cap.get(cv2.CAP_PROP_FPS):.3f}"
        )

    def start_logging(self):
        if self.logging:
            return

        self._update_condition_label()
        duration = max(1, int(self.duration_var.get()))
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_condition = "".join(c if c.isalnum() or c in "-_" else "_" for c in self.condition_var.get())
        self.session_dir = self.output_dir / f"{stamp}_{safe_condition}"
        (self.session_dir / "frames").mkdir(parents=True, exist_ok=True)
        (self.session_dir / "snapshots").mkdir(parents=True, exist_ok=True)

        metadata = self._session_metadata()
        metadata["started_at"] = dt.datetime.now().isoformat()
        metadata["duration_seconds"] = duration
        (self.session_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        self.csv_file = (self.session_dir / "log.csv").open("w", newline="", encoding="utf-8")
        self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=self._csv_fields())
        self.csv_writer.writeheader()
        self.logging = True
        self.frame_index = 0
        self.stats.reset()
        self.session_started_at = time.time()
        self.stop_at = self.session_started_at + duration
        self.status_var.set(f"Recording {duration}s to {self.session_dir}")

    def stop_logging(self):
        was_logging = self.logging
        self.logging = False
        if self.csv_file:
            self.csv_file.close()
            self.csv_file = None
        self.csv_writer = None
        if was_logging and self.session_dir:
            self._write_summary()
        self.status_var.set("Stopped")

    def _session_metadata(self):
        return {
            "experiment_mode": self.mode_var.get(),
            "condition": self.condition_var.get(),
            "trial": self.trial_var.get(),
            "intended_exposure_duration": self.exposure_label_var.get(),
            "camera_exposure_value": self.cap.get(cv2.CAP_PROP_EXPOSURE),
            "intended_iso_or_gain": self.iso_label_var.get(),
            "camera_gain_value": self.cap.get(cv2.CAP_PROP_GAIN),
            "camera_fps": self.cap.get(cv2.CAP_PROP_FPS),
            "camera_width": self.cap.get(cv2.CAP_PROP_FRAME_WIDTH),
            "camera_height": self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
            "distance": self.distance_var.get(),
            "angle": self.angle_var.get(),
            "motion": self.motion_var.get(),
            "expected_qr_payload": self.qr_expected_var.get(),
            "expected_users": self.expected_users(),
            "led_threshold": self.threshold_var.get(),
            "yolo_confidence_threshold": self.confidence_threshold,
            "yolo_every_n_frames": self.yolo_every_var.get(),
            "yolo_available": bool(self.yolo),
        }

    @staticmethod
    def _csv_fields():
        return [
            "timestamp",
            "elapsed_seconds",
            "frame_index",
            "experiment_mode",
            "condition",
            "trial",
            "intended_exposure_duration",
            "camera_exposure",
            "intended_iso_or_gain",
            "camera_gain",
            "camera_fps",
            "distance",
            "angle",
            "motion",
            "qr_detected",
            "qr_payload",
            "qr_match",
            "led_blob_count",
            "largest_blob_area",
            "largest_blob_bbox",
            "expected_users",
            "yolo_detection_count",
            "yolo_users",
            "yolo_ooks",
            "yolo_confidences",
            "yolo_bboxes",
            "yolo_all_expected_detected",
            "scene_mean_brightness",
            "saved_frame",
        ]

    def _write_summary(self):
        summary = {
            **self._session_metadata(),
            "frames": self.stats.frames,
            "qr_detect_rate": self._rate(self.stats.qr_detected),
            "qr_match_rate": self._rate(self.stats.qr_matched),
            "led_visible_rate": self._rate(self.stats.led_frames),
            "yolo_all_expected_rate": self._rate(self.stats.yolo_all_expected),
            "per_user_yolo_rates": {
                user: self._rate(count) for user, count in self.stats.yolo_user_hits.items()
            },
            "finished_at": dt.datetime.now().isoformat(),
        }
        (self.session_dir / "session_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    def _rate(self, count):
        return 0.0 if self.stats.frames == 0 else count / self.stats.frames

    def save_snapshot(self):
        if self.last_frame is None:
            return
        target_dir = self.session_dir / "snapshots" if self.session_dir else self.output_dir / "snapshots"
        target_dir.mkdir(parents=True, exist_ok=True)
        path = target_dir / f"snapshot_{dt.datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
        cv2.imwrite(str(path), self.last_frame)
        self.status_var.set(f"Saved snapshot: {path}")

    def _detect_qr(self, frame):
        payloads = []
        points = []
        try:
            ok, decoded_info, qr_points, _straight = self.qr.detectAndDecodeMulti(frame)
            if ok:
                payloads = [p for p in decoded_info if p]
                if qr_points is not None:
                    points = [np.asarray(point_set, dtype=np.int32) for point_set in qr_points]
        except cv2.error:
            payload, qr_points, _straight = self.qr.detectAndDecode(frame)
            if payload:
                payloads = [payload]
                if qr_points is not None:
                    points = [np.asarray(qr_points, dtype=np.int32)]
        return payloads, points

    def _detect_led_blobs(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, self.threshold_var.get(), 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        blobs = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 5:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            blobs.append((area, x, y, w, h))
        blobs.sort(reverse=True)
        return blobs

    def _should_run_yolo(self):
        if not self.yolo:
            return False
        every = max(1, int(self.yolo_every_var.get()))
        return self.preview_index % every == 0 or not self.last_yolo_detections

    def _draw_overlay(self, frame, payloads, qr_points, blobs, yolo_detections):
        overlay = frame.copy()
        for point_set in qr_points:
            cv2.polylines(overlay, [point_set.reshape((-1, 1, 2))], True, (0, 255, 0), 2)

        for det in yolo_detections:
            x1, y1, x2, y2 = det["bbox"]
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 0), 2)
            label = f"{det['user']} {det['confidence']:.2f}"
            cv2.putText(overlay, label, (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)

        for _area, x, y, w, h in blobs[:10]:
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 255), 2)

        qr_ok = "yes" if payloads else "no"
        expected = set(self.expected_users())
        seen = {det["user"] for det in yolo_detections}
        all_expected = bool(expected) and expected.issubset(seen)
        lines = [
            f"Mode: {'Scene suppression' if self.mode_var.get() == 'scene' else 'Multi-user tags'}",
            f"QR decoded: {qr_ok} | YOLO users: {','.join(sorted(seen)) or '-'}",
            f"Expected users present: {'yes' if all_expected else 'no'}",
        ]
        if self.logging and self.stop_at:
            remaining = max(0, self.stop_at - time.time())
            lines.append(f"REC {remaining:.1f}s left")

        y = 24
        for line in lines:
            cv2.putText(overlay, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 0), 2)
            y += 25
        return overlay

    def _log_frame(self, frame, payloads, blobs, yolo_detections):
        if not self.logging or not self.csv_writer:
            return

        elapsed = time.time() - self.session_started_at
        sample_every = max(1, int(self.sample_every_var.get()))
        saved_frame = ""
        if self.frame_index % sample_every == 0:
            saved_frame = f"frames/frame_{self.frame_index:06d}.jpg"
            cv2.imwrite(str(self.session_dir / saved_frame), frame)

        expected_qr = self.qr_expected_var.get().strip()
        qr_match = bool(expected_qr and expected_qr in payloads)
        expected_users = self.expected_users()
        seen_users = [det["user"] for det in yolo_detections]
        all_expected = bool(expected_users) and set(expected_users).issubset(set(seen_users))
        largest = blobs[0] if blobs else None

        self._update_stats(payloads, qr_match, blobs, yolo_detections, all_expected)

        row = {
            "timestamp": time.time(),
            "elapsed_seconds": f"{elapsed:.3f}",
            "frame_index": self.frame_index,
            "experiment_mode": self.mode_var.get(),
            "condition": self.condition_var.get(),
            "trial": self.trial_var.get(),
            "intended_exposure_duration": self.exposure_label_var.get(),
            "camera_exposure": self.cap.get(cv2.CAP_PROP_EXPOSURE),
            "intended_iso_or_gain": self.iso_label_var.get(),
            "camera_gain": self.cap.get(cv2.CAP_PROP_GAIN),
            "camera_fps": self.cap.get(cv2.CAP_PROP_FPS),
            "distance": self.distance_var.get(),
            "angle": self.angle_var.get(),
            "motion": self.motion_var.get(),
            "qr_detected": int(bool(payloads)),
            "qr_payload": "|".join(payloads),
            "qr_match": int(qr_match),
            "led_blob_count": len(blobs),
            "largest_blob_area": largest[0] if largest else 0,
            "largest_blob_bbox": f"{largest[1]},{largest[2]},{largest[3]},{largest[4]}" if largest else "",
            "expected_users": "|".join(expected_users),
            "yolo_detection_count": len(yolo_detections),
            "yolo_users": "|".join(seen_users),
            "yolo_ooks": "|".join(det["ook"] for det in yolo_detections),
            "yolo_confidences": "|".join(f"{det['confidence']:.6f}" for det in yolo_detections),
            "yolo_bboxes": "|".join(",".join(str(v) for v in det["bbox"]) for det in yolo_detections),
            "yolo_all_expected_detected": int(all_expected),
            "scene_mean_brightness": float(np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))),
            "saved_frame": saved_frame,
        }
        self.csv_writer.writerow(row)
        self.frame_index += 1

    def _update_stats(self, payloads, qr_match, blobs, yolo_detections, all_expected):
        self.stats.frames += 1
        self.stats.qr_detected += int(bool(payloads))
        self.stats.qr_matched += int(qr_match)
        self.stats.led_frames += int(bool(blobs))
        self.stats.yolo_all_expected += int(all_expected)
        seen = {det["user"] for det in yolo_detections}
        for user in USER_LABELS:
            self.stats.yolo_user_hits[user] += int(user in seen)

        self.metrics_var.set(
            f"Frames {self.stats.frames} | "
            f"QR match {self.stats.pct(self.stats.qr_matched):.1f}% | "
            f"QR detect {self.stats.pct(self.stats.qr_detected):.1f}% | "
            f"LED {self.stats.pct(self.stats.led_frames):.1f}% | "
            f"YOLO all-users {self.stats.pct(self.stats.yolo_all_expected):.1f}%"
        )

    def _update_frame(self):
        ok, frame = self.cap.read()
        if ok:
            self.last_frame = frame
            self.preview_index += 1
            payloads, qr_points = self._detect_qr(frame)
            blobs = self._detect_led_blobs(frame)
            if self._should_run_yolo():
                self.last_yolo_detections = self.yolo.detect(frame)
            yolo_detections = self.last_yolo_detections

            self._log_frame(frame, payloads, blobs, yolo_detections)
            overlay = self._draw_overlay(frame, payloads, qr_points, blobs, yolo_detections)

            rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (960, 540), interpolation=cv2.INTER_AREA)
            image = Image.fromarray(rgb)
            self.photo = ImageTk.PhotoImage(image=image)
            self.video_label.configure(image=self.photo)

        if self.logging and self.stop_at and time.time() >= self.stop_at:
            self.stop_logging()

        self.root.after(20, self._update_frame)

    def run(self):
        try:
            self.root.mainloop()
        finally:
            self.stop_logging()
            self.cap.release()


def main():
    parser = argparse.ArgumentParser(description="RoEmotion PC webcam experiment console")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index")
    parser.add_argument("--output", type=Path, default=Path("runs"), help="Output directory")
    parser.add_argument("--model", type=Path, default=None, help="Optional RoEmotion YOLO .tflite model")
    parser.add_argument("--confidence", type=float, default=0.001, help="YOLO confidence threshold")
    args = parser.parse_args()

    app = RoEmotionPCTool(args.camera, args.output, args.model, args.confidence)
    app.run()


if __name__ == "__main__":
    main()
