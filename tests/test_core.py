import json
import re
from pathlib import Path
import sys
import uuid
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np

from core import (Analyzer, Detector, PAYLOAD, ROOT, assign_regions, save_snapshot,
                  summarize, validate_regions, write_json)
from reporting import report_all, report_run


def config(mode='multi'):
    return {'mode': mode, 'condition': 'synthetic_test', 'repetition': '1', 'exposure_label': 'test only',
            'regions': {'User_1': [0, 0, .45, 1], 'User_2': [.55, 0, 1, 1]},
            'expected': ['User_1', 'User_2'], 'led_threshold': 230, 'qr_payload': PAYLOAD}


def det(user, box, score=.9):
    return {'user': user, 'bbox': box, 'confidence': score, 'class_id': int(user[-1])-1}


def sample(predictions, index=0):
    c = config()
    assignments, extras = assign_regions(predictions, c['regions'], (100, 100, 3), c['expected'])
    return {'detections': predictions, 'assignments': assignments, 'extra_detections': extras,
            'elapsed_s': index*.2, 'processing_ms': 10, 'qr_localized': None, 'qr_match': None,
            'qr_enhanced_match': None, 'qr_any_match': None,
            'blobs': {u: {'visible': True} for u in c['expected']}, 'frame_id': index}


class MetricsTests(unittest.TestCase):
    def test_swapped_ids_are_errors_despite_all_labels_present(self):
        row = sample([det('User_2', [5, 5, 15, 15]), det('User_1', [60, 5, 70, 15])])
        result = summarize([row], config())
        self.assertEqual(result['all_correct_rate'], 0)
        self.assertEqual(result['per_user']['User_1']['wrong_id_rate'], 1)

    def test_duplicate_is_counted_not_hidden(self):
        row = sample([det('User_1', [5, 5, 15, 15]), det('User_2', [60, 5, 70, 15]), det('User_1', [20, 20, 30, 30], .7)])
        result = summarize([row], config())
        self.assertEqual(result['all_correct_rate'], 1)
        self.assertEqual(result['exact_frame_rate'], 0)
        self.assertEqual(result['extra_detections_per_sample'], 1)

    def test_misses_and_wrong_ids_have_separate_denominators(self):
        rows = [sample([det('User_1', [5, 5, 15, 15])], 0), sample([], 1), sample([det('User_2', [5, 5, 15, 15])], 2)]
        result = summarize(rows, config())
        user = result['per_user']['User_1']
        self.assertEqual(user['correct_rate'], 1/3)
        self.assertEqual(user['miss_rate'], 1/3)
        self.assertEqual(user['wrong_id_rate'], 1/3)
        self.assertEqual(user['accuracy_given_detection'], .5)
        self.assertAlmostEqual(result['observed_sample_hz'], 5)

    def test_absent_model_is_na_not_zero(self):
        row = sample([])
        row.update(detections=None, assignments={}, extra_detections=None)
        result = summarize([row], config())
        self.assertIsNone(result['all_correct_rate'])
        self.assertEqual(result['id_samples'], 0)
        self.assertIsNone(summarize([], config())['qr_raw_match_rate'])

    def test_regions_require_independent_ground_truth(self):
        c = config()
        validate_regions(c['regions'], c['expected'])
        c['regions']['User_2'] = [.2, .1, .7, .7]
        with self.assertRaises(ValueError):
            validate_regions(c['regions'], c['expected'])

    def test_qr_decodes_expected_target_and_rejects_wrong_payload(self):
        qr = cv2.imread(str(ROOT / 'assets' / 'qr_only.png'))
        h, w = qr.shape[:2]
        frame = np.zeros((h, w+100, 3), np.uint8)
        frame[:, 100:] = qr
        frame[20:30, 10:20, 1] = 255
        c = config('scene')
        c.update(expected=['User_1'], regions={'QR': [100/(w+100), 0, 1, 1], 'User_1': [0, 0, 90/(w+100), 1]})
        analyzer = Analyzer()
        result = analyzer.measure(frame, c)
        self.assertTrue(result['qr_match'])
        self.assertTrue(result['qr_localized'])
        self.assertTrue(result['blobs']['User_1']['visible'])
        c['qr_payload'] = 'wrong'
        self.assertFalse(analyzer.measure(frame, c)['qr_any_match'])
        self.assertFalse(analyzer.measure(np.zeros_like(frame), c)['qr_match'])
        c['qr_payload'] = ''
        uncalibrated = analyzer.measure(frame, c)
        self.assertEqual(uncalibrated['qr_payload'], PAYLOAD)
        self.assertIsNone(uncalibrated['qr_match'])


class ReceiverTests(unittest.TestCase):
    def detector(self):
        detector = Detector.__new__(Detector)
        detector.confidence, detector.nms_iou = .25, .45
        return detector

    def test_preprocessing_and_reverse_letterbox(self):
        frame = np.zeros((720, 1280, 3), np.uint8)
        frame[:, :, 2] = 255
        tensor, meta = Detector.preprocess(frame)
        self.assertEqual(tensor.shape, (1, 3, 960, 960))
        self.assertEqual(float(tensor[0, 0, 300, 300]), 1)
        self.assertEqual(float(tensor[0, 2, 300, 300]), 0)
        output = np.array([[[75, 285, 150, 360, .9, 0]]])
        boxes = self.detector().parse(output, meta, frame.shape)
        np.testing.assert_allclose(boxes[0]['bbox'], [100, 100, 200, 200])

    def test_invalid_and_duplicate_rows(self):
        rows = np.array([[[100, 100, 200, 200, .9, 0], [101, 101, 201, 201, .8, 0],
                          [300, 100, 400, 200, .8, 0], [100, 100, 200, 200, .7, 1],
                          [np.nan, 100, 200, 200, .9, 0], [2, 2, 1, 1, .9, 0]]])
        result = self.detector().parse(rows, (1, 0, 0), (960, 960, 3))
        self.assertEqual(len(result), 3)


class FirmwareTests(unittest.TestCase):
    def test_original_green_waveforms_after_one_warmup_loop(self):
        for user, expected in [(1, '01010'), (2, '01110'), (3, '00110')]:
            path = next((ROOT / 'firmware' / 'original').glob(f'*User_{user}/*.ino'))
            loop = path.read_text().split('void loop()')[1]
            tokens = re.findall(r'digitalWrite\((\w+),\s*(HIGH|LOW)\)|delayMicroseconds\((\d+)\)', loop)
            green = 0
            for _ in range(2):
                bits = []
                for pin, value, duration in tokens:
                    if pin == 'greenLed4':
                        green = int(value == 'HIGH')
                    if duration:
                        self.assertEqual(int(duration), 166)
                        bits.append(str(green))
            self.assertEqual(''.join(bits), expected)


class ArtifactTests(unittest.TestCase):
    def test_snapshot_is_lossless_and_report_rebuilds(self):
        c = config()
        row = sample([det('User_1', [5, 5, 15, 15])])
        frame = np.random.default_rng(17).integers(0, 256, (100, 100, 3), dtype=np.uint8)
        scratch = ROOT / 'verification-artifacts'
        scratch.mkdir(exist_ok=True)
        folder = scratch / ('report_check_' + uuid.uuid4().hex)
        folder.mkdir()
        if folder.is_relative_to(scratch):
            directory = Path(folder) / 'test'
            directory.mkdir()
            write_json(directory / 'metadata.json', {'config': c})
            (directory / 'samples.jsonl').write_text(json.dumps(row)+'\n{broken', encoding='utf-8')
            paths = save_snapshot(directory / 'snapshots', 'test', frame, row, c)
            np.testing.assert_array_equal(cv2.imread(str(directory / 'snapshots' / paths['raw'])), frame)
            result = report_run(directory)
            self.assertEqual(result['completion']['status'], 'interrupted')
            self.assertEqual(len(result['warnings']), 1)
            self.assertEqual(result['samples'], 1)
            self.assertTrue((directory / 'confusion.png').exists())
            self.assertTrue(report_all(Path(folder)).exists())
            self.assertIn('User_1_wrong_id_rate', (Path(folder) / 'runs_summary.csv').read_text())


if __name__ == '__main__':
    unittest.main()
