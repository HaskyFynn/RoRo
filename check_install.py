import hashlib
import json
import time
import tkinter

import cv2
import numpy as np

from core import Detector, MODEL, MODEL_SHA256, PAYLOAD, ROOT, environment


def main():
    assert hashlib.sha256(MODEL.read_bytes()).hexdigest() == MODEL_SHA256, 'Bundled model checksum mismatch'
    detector = Detector()
    started = time.perf_counter()
    detections = detector.detect(np.zeros((720, 1280, 3), np.uint8))
    print(f'Model loaded and invoked in {time.perf_counter()-started:.2f}s; blank-image detections: {len(detections)}')
    board = cv2.imread(str(ROOT / 'assets' / 'qr_only.png'))
    assert board is not None and cv2.QRCodeDetector().detectAndDecode(board)[0] == PAYLOAD
    print('Static QR test image decoded successfully. Tk version:', tkinter.TkVersion)
    print(json.dumps(environment(), indent=2))
    print('Software checks passed. Camera and transmitter checks still require the physical setup.')


if __name__ == '__main__':
    main()
