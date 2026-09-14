"""Frame-local measurements shared by the GUI, reports, and regression checks."""
import hashlib
import importlib.metadata
import json
import os
import platform
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
MODEL_FILENAME = 'RoEmotion_LED_Recognition_Model_YOLOv26n.tflite'
MODEL = ROOT / 'models' / MODEL_FILENAME
MODEL_SHA256 = 'c2d6109d8b1c795ddac2112b38c449a1dd1371c6c6d9553f111c43edd6f01018'
USERS = ['User_1', 'User_2', 'User_3']
LABELS = ['ook_10101', 'ook_01110', 'ook_00110']
PAYLOAD = 'ROEMOTION_SCENE_SUPPRESSION_TEST'


def resolve_model_path(path=None):
    candidates = []
    if path:
        candidates.append(Path(path))
    env_path = os.environ.get('ROEMOTION_MODEL_PATH')
    if env_path:
        candidates.append(Path(env_path))
    candidates.extend([MODEL, ROOT / MODEL_FILENAME, Path.cwd() / 'models' / MODEL_FILENAME])
    checked = []
    for candidate in candidates:
        candidate = candidate.expanduser()
        if not candidate.is_absolute():
            candidate = (ROOT / candidate).resolve()
        else:
            candidate = candidate.resolve()
        checked.append(candidate)
        if candidate.exists():
            return candidate
    expected = '\n'.join('  - ' + str(candidate) for candidate in checked)
    raise FileNotFoundError(
        'Tag model not found. Put the .tflite file in models/ or launch with '
        '--model-path /path/to/RoEmotion_LED_Recognition_Model_YOLOv26n.tflite.\n'
        'Checked:\n' + expected
    )


def write_json(path, data):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def environment():
    versions = {}
    for name in ['opencv-python', 'numpy', 'Pillow', 'matplotlib', 'ai-edge-litert']:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {'python': platform.python_version(), 'platform': platform.platform(), 'packages': versions}


def iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2]-a[0]) * (a[3]-a[1]) + (b[2]-b[0]) * (b[3]-b[1]) - intersection
    return intersection / union if union > 0 else 0.0


class Detector:
    """Android-compatible letterbox/RGB/NCHW with full per-class NMS output."""
    def __init__(self, path=MODEL, confidence=0.25, nms_iou=0.45):
        if not 0 < confidence <= 1:
            raise ValueError('Confidence must be in (0, 1].')
        path = resolve_model_path(path)
        from ai_edge_litert.interpreter import Interpreter
        self.path = path
        self.sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        self.confidence, self.nms_iou = confidence, nms_iou
        self.threads = min(4, os.cpu_count() or 1)
        self.interpreter = Interpreter(model_path=str(path), num_threads=self.threads)
        self.interpreter.allocate_tensors()
        self.input = self.interpreter.get_input_details()[0]
        self.output = self.interpreter.get_output_details()[0]
        if list(self.input['shape']) != [1, 3, 960, 960] or self.input['dtype'] != np.float32:
            raise ValueError('Expected float32 model input [1, 3, 960, 960].')
        if list(self.output['shape']) != [1, 300, 6] or self.output['dtype'] != np.float32:
            raise ValueError('Expected float32 model output [1, 300, 6].')

    @staticmethod
    def preprocess(frame):
        h, w = frame.shape[:2]
        scale = min(960 / w, 960 / h)
        rw, rh = int(w * scale + 0.5), int(h * scale + 0.5)
        left, top = (960-rw)//2, (960-rh)//2
        canvas = np.full((960, 960, 3), 114, np.uint8)
        canvas[top:top+rh, left:left+rw] = cv2.resize(frame, (rw, rh))
        tensor = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
        return np.ascontiguousarray(tensor.transpose(2, 0, 1)[None]), (scale, left, top)

    def parse(self, output, meta, shape):
        scale, left, top = meta
        height, width = shape[:2]
        candidates = []
        for row in output.reshape(-1, 6):
            if not np.isfinite(row).all():
                continue
            x1, y1, x2, y2, score, cls = map(float, row)
            if score < 0 or score > 1:
                score = float(1 / (1 + np.exp(-np.clip(score, -30, 30))))
            if score < self.confidence or cls != int(cls) or int(cls) not in range(3):
                continue
            if x2 <= x1 or y2 <= y1:
                continue
            if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.5:
                x1, y1, x2, y2 = [v * 960 for v in (x1, y1, x2, y2)]
            box = [float(np.clip((x1-left)/scale, 0, width)),
                   float(np.clip((y1-top)/scale, 0, height)),
                   float(np.clip((x2-left)/scale, 0, width)),
                   float(np.clip((y2-top)/scale, 0, height))]
            if box[2] > box[0] and box[3] > box[1]:
                candidates.append({'user': USERS[int(cls)], 'class_id': int(cls),
                                   'confidence': score, 'bbox': box})
        picked = []
        for det in sorted(candidates, key=lambda d: d['confidence'], reverse=True):
            if not any(det['class_id'] == old['class_id'] and iou(det['bbox'], old['bbox']) > self.nms_iou for old in picked):
                picked.append(det)
        return picked

    def detect(self, frame):
        tensor, meta = self.preprocess(frame)
        self.interpreter.set_tensor(self.input['index'], tensor)
        self.interpreter.invoke()
        return self.parse(self.interpreter.get_tensor(self.output['index']), meta, frame.shape)


def pixels(region, shape):
    h, w = shape[:2]
    x1, y1, x2, y2 = region
    return (max(0, int(x1*w)), max(0, int(y1*h)), min(w, int(x2*w)), min(h, int(y2*h)))


def validate_regions(regions, expected, scene=False):
    for key in expected + (['QR'] if scene else []):
        if key not in regions:
            raise ValueError(f'Draw the {key} region first.')
        a = regions[key]
        if len(a) != 4 or not all(np.isfinite(a)) or not (0 <= a[0] < a[2] <= 1 and 0 <= a[1] < a[3] <= 1):
            raise ValueError(f'Invalid region: {key}')
    keys = expected + (['QR'] if scene else [])
    for index, key in enumerate(keys):
        for other in keys[index+1:]:
            if iou(regions[key], regions[other]) > 0:
                raise ValueError(f'{key} and {other} regions overlap.')


def assign_regions(detections, regions, shape, expected):
    # Regions are independent, operator-labelled ground truth. Never assign by predicted class.
    selected, used = {}, set()
    for user in expected:
        x1, y1, x2, y2 = pixels(regions[user], shape)
        candidates = []
        for index, det in enumerate(detections):
            a, b, c, d = det['bbox']
            if x1 <= (a+c)/2 < x2 and y1 <= (b+d)/2 < y2:
                candidates.append((index, det))
        if candidates:
            index, det = max(candidates, key=lambda pair: pair[1]['confidence'])
            selected[user] = det['user']
            used.add(index)
        else:
            selected[user] = 'miss'
    return selected, len(detections) - len(used)


class Analyzer:
    def __init__(self, detector=None):
        self.detector = detector
        self.qr = cv2.QRCodeDetector()
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def measure(self, frame, config):
        start = time.perf_counter()
        result = {'qr_localized': None, 'qr_match': None, 'qr_enhanced_match': None,
                  'qr_any_match': None, 'qr_payload': '', 'qr_enhanced_payload': '',
                  'scene_mean': None, 'scene_std': None}
        if config['mode'] == 'scene':
            a, b, c, d = pixels(config['regions']['QR'], frame.shape)
            gray = cv2.cvtColor(frame[b:d, a:c], cv2.COLOR_BGR2GRAY)
            payload, points, _ = self.qr.detectAndDecode(gray)
            enhanced, _, _ = self.qr.detectAndDecode(self.clahe.apply(gray))
            expected = config['qr_payload']
            result.update(qr_localized=points is not None, qr_payload=payload,
                          qr_enhanced_payload=enhanced, qr_match=(payload == expected) if expected else None,
                          qr_enhanced_match=(enhanced == expected) if expected else None,
                          qr_any_match=(expected in (payload, enhanced)) if expected else None,
                          scene_mean=float(gray.mean()), scene_std=float(gray.std()))
        blobs = {}
        for user in config['expected']:
            a, b, c, d = pixels(config['regions'][user], frame.shape)
            green = frame[b:d, a:c, 1]
            mask = (green >= config['led_threshold']).astype(np.uint8)
            count, _, stats, _ = cv2.connectedComponentsWithStats(mask)
            area = int(stats[1:, cv2.CC_STAT_AREA].max()) if count > 1 else 0
            blobs[user] = {'visible': area >= 5, 'area_px': area}
        detections = self.detector.detect(frame) if self.detector else None
        selected, extras = (assign_regions(detections, config['regions'], frame.shape, config['expected'])
                            if detections is not None else ({}, None))
        result.update(detections=detections, assignments=selected, extra_detections=extras, blobs=blobs,
                      processing_ms=(time.perf_counter()-start)*1000)
        return result


def summarize(rows, config):
    n = len(rows)
    def rate(values):
        values = [v for v in values if v is not None]
        return sum(values)/len(values) if values else None
    identified = [r for r in rows if r['detections'] is not None]
    users = {}
    for user in config['expected']:
        choices = [r['assignments'][user] for r in identified]
        confusion = {label: choices.count(label) for label in USERS + ['miss']}
        detected = [c for c in choices if c != 'miss']
        users[user] = {'correct_rate': rate([c == user for c in choices]),
                       'miss_rate': rate([c == 'miss' for c in choices]),
                       'wrong_id_rate': rate([c not in (user, 'miss') for c in choices]),
                       'accuracy_given_detection': rate([c == user for c in detected]),
                       'bright_region_rate': rate([r['blobs'][user]['visible'] for r in rows]),
                       'confusion_counts': confusion}
    all_correct = [all(r['assignments'][u] == u for u in config['expected']) for r in identified]
    latency = [r['processing_ms'] for r in rows]
    span = rows[-1]['elapsed_s'] - rows[0]['elapsed_s'] if n > 1 else 0
    return {'samples': n, 'id_samples': len(identified),
            'observed_sample_hz': (n-1)/span if span > 0 else None,
            'qr_localized_rate': rate([r['qr_localized'] for r in rows]),
            'qr_raw_match_rate': rate([r['qr_match'] for r in rows]),
            'qr_enhanced_match_rate': rate([r['qr_enhanced_match'] for r in rows]),
            'qr_any_match_rate': rate([r['qr_any_match'] for r in rows]),
            'all_correct_rate': rate(all_correct),
            'exact_frame_rate': rate([ok and r['extra_detections'] == 0 for ok, r in zip(all_correct, identified)]),
            'extra_detections_per_sample': rate([r['extra_detections'] for r in identified]),
            'processing_ms_median': float(np.median(latency)) if latency else None,
            'processing_ms_p95': float(np.percentile(latency, 95)) if latency else None,
            'per_user': users}


def overlay(frame, result, config):
    canvas = frame.copy()
    for key, region in config['regions'].items():
        if key not in config['expected'] and key != 'QR':
            continue
        a, b, c, d = pixels(region, frame.shape)
        cv2.rectangle(canvas, (a, b), (c, d), (80, 200, 120), 2)
        cv2.putText(canvas, f'Truth: {key}', (a, max(18, b-6)), cv2.FONT_HERSHEY_SIMPLEX, .55, (80, 200, 120), 2)
    for det in result.get('detections') or []:
        a, b, c, d = map(int, det['bbox'])
        cv2.rectangle(canvas, (a, b), (c, d), (0, 180, 255), 2)
        cv2.putText(canvas, f"{det['user']} {det['confidence']:.2f}", (a, min(canvas.shape[0]-8, d+18)), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 180, 255), 1)
    return canvas


def save_snapshot(directory, stem, frame, result, config):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = {'raw': f'{stem}_raw.png', 'annotated': f'{stem}_annotated.png'}
    for kind, image in [('raw', frame), ('annotated', overlay(frame, result, config))]:
        if not cv2.imwrite(str(directory / paths[kind]), image):
            raise OSError(f'Cannot save {directory / paths[kind]}')
    for key in config['expected'] + (['QR'] if config['mode'] == 'scene' else []):
        a, b, c, d = pixels(config['regions'][key], frame.shape)
        name = f'{stem}_{key}.png'
        if not cv2.imwrite(str(directory / name), frame[b:d, a:c]):
            raise OSError(f'Cannot save crop {name}')
        paths[key] = name
    write_json(directory / f'{stem}.json', {'config': config, 'measurement': result, 'files': paths,
               'image_note': 'Unmodified decoded camera pixels in raw PNG; not sensor RAW.'})
    return paths
