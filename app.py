"""Standalone desktop console. Run: python app.py"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime as dt
import json
import queue
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk
import webbrowser

import cv2
import numpy as np
from PIL import Image, ImageTk

from core import (ROOT, Detector, Analyzer, USERS, environment,
                  overlay, save_snapshot, summarize, validate_regions, write_json)


class Camera:
    def __init__(self, index, width, height, fps, demo=False):
        self.latest = None
        self.info = {}
        self.error = None
        self.stop = threading.Event()
        self.commands = queue.Queue()
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._read, args=(index, width, height, fps, demo), daemon=True)
        self.thread.start()

    def _read(self, index, width, height, fps, demo):
        cap = None
        try:
            if demo:
                image = cv2.imread(str(ROOT / 'assets' / 'scene_board.png'))
                if image is None:
                    raise RuntimeError('Bundled demo image is missing from assets/scene_board.png.')
                image = cv2.resize(image, (width, height))
                self.info = {'backend': 'DEMO STILL IMAGE', 'width': width, 'height': height, 'reported_fps': fps}
            else:
                cap = cv2.VideoCapture(index, cv2.CAP_DSHOW) if __import__('os').name == 'nt' else cv2.VideoCapture(index)
                if not cap.isOpened():
                    cap.release()
                    cap = cv2.VideoCapture(index)
                if not cap.isOpened():
                    raise RuntimeError(f'Cannot open camera {index}. Close other camera apps or change Camera index.')
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                cap.set(cv2.CAP_PROP_FPS, fps)
            frame_id, failures = 0, 0
            started = time.monotonic()
            properties = {'exposure_raw': cv2.CAP_PROP_EXPOSURE, 'gain_raw': cv2.CAP_PROP_GAIN,
                          'auto_exposure_raw': cv2.CAP_PROP_AUTO_EXPOSURE, 'focus_raw': cv2.CAP_PROP_FOCUS,
                          'auto_focus_raw': cv2.CAP_PROP_AUTOFOCUS, 'auto_wb_raw': cv2.CAP_PROP_AUTO_WB,
                          'reported_fps': cv2.CAP_PROP_FPS, 'width': cv2.CAP_PROP_FRAME_WIDTH,
                          'height': cv2.CAP_PROP_FRAME_HEIGHT}
            while not self.stop.is_set():
                while not self.commands.empty():
                    settings = self.commands.get_nowait()
                    accepted = {key: bool(cap.set(properties[key], value)) if cap else False for key, value in settings.items()}
                    self.info = {**self.info, 'requested_controls': settings, 'control_set_return': accepted,
                                 'controls_applied_monotonic': time.monotonic()}
                ok, frame = (True, image.copy()) if demo else cap.read()
                if not ok:
                    failures += 1
                    if failures >= 30:
                        raise RuntimeError('Camera stopped delivering frames. Run was interrupted.')
                    self.stop.wait(.05)
                    continue
                failures = 0
                frame_id += 1
                now = time.monotonic()
                if cap and (frame_id == 1 or frame_id % 15 == 0):
                    self.info = {**self.info, **{k: cap.get(p) for k, p in properties.items()}, 'backend': cap.getBackendName()}
                self.info = {**self.info, 'delivered_hz': frame_id / max(now-started, .001)}
                with self.lock:
                    self.latest = (frame_id, now, time.time(), frame)
                if demo:
                    self.stop.wait(1 / fps)
        except Exception as error:
            self.error = str(error)
        finally:
            if cap is not None:
                cap.release()

    def get(self):
        with self.lock:
            return self.latest

    def close(self):
        self.stop.set()
        self.thread.join(timeout=2)


class App:
    def __init__(self, demo=False, model=True, model_path=None):
        self.root = tk.Tk()
        self.root.title('RoEmotion | Experiment Console' + (' | DEMO' if demo else ''))
        self.root.geometry('1260x820')
        self.root.minsize(850, 620)
        self.root.configure(bg='#edf0ef')
        self.demo = demo
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.future = None
        self.future_context = None
        self.camera = None
        self.detector = None
        self.model_error = ''
        if model:
            try:
                self.detector = Detector(path=model_path)
            except Exception as error:
                self.model_error = str(error)
        self.analyzer = Analyzer(self.detector)
        self.regions = {}
        self.last_result = None
        self.last_frame_id = -1
        self.last_sample_time = 0
        self.photo = None
        self.running = False
        self.closing = False
        self.run_dir = None
        self.rows = []
        self.log = None
        self.next_snapshot = 0
        self.drag = None
        self.display_box = None
        self.run_config = None
        self.epoch = 0
        self.vars = {}
        defaults = {'mode': 'scene', 'condition': 'baseline', 'repetition': '1', 'camera': '0',
                    'width': '1280', 'height': '720', 'fps': '30', 'exposure_label': 'unverified',
                    'exposure_raw': '', 'gain_raw': '', 'auto_exposure_raw': '', 'focus_raw': '',
                    'auto_focus_raw': '', 'auto_wb_raw': '', 'distance': '1.5 m', 'angle': '0 deg',
                    'motion': 'still', 'lighting': 'fixed room lighting', 'qr_width': '',
                    'camera_name': '', 'firmware': 'original', 'row_timing': 'unknown',
                    'exposure_evidence': 'unverified', 'gain_note': 'unknown',
                    'symbol_us': '166', 'qr_payload': '', 'duration': '45', 'sample_hz': '5',
                    'snapshot_s': '2', 'led_threshold': '230', 'confidence': '0.25', 'region': 'QR'}
        self.vars = {k: tk.StringVar(value=v) for k, v in defaults.items()}
        self.expected = {u: tk.BooleanVar(value=u == 'User_1') for u in USERS}
        self.status = tk.StringVar(value='Ready')
        self.metrics = tk.StringVar(value='No run recorded')
        self.camera_status = tk.StringVar(value='Camera disconnected')
        self.region_status = tk.StringVar(value='Regions: none')
        self.verified = tk.BooleanVar(value=False)
        self._build()
        self.root.protocol('WM_DELETE_WINDOW', self.close)
        self.root.after(80, self.tick)

    def _build(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TLabel', font=('Segoe UI', 10))
        style.configure('TButton', padding=6)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        left = ttk.Frame(self.root, padding=10)
        left.grid(row=0, column=0, sticky='nsew')
        left.rowconfigure(1, weight=1)
        left.columnconfigure(0, weight=1)
        ttk.Label(left, text='RoEmotion / ' + ('DEMO' if self.demo else 'Live capture'), font=('Segoe UI', 18, 'bold')).grid(row=0, column=0, sticky='w', pady=(0, 8))
        self.canvas = tk.Canvas(left, bg='#191d1c', highlightthickness=0)
        self.canvas.grid(row=1, column=0, sticky='nsew')
        self.canvas.bind('<ButtonPress-1>', self.drag_start)
        self.canvas.bind('<B1-Motion>', self.drag_move)
        self.canvas.bind('<ButtonRelease-1>', self.drag_end)
        ttk.Label(left, textvariable=self.metrics, wraplength=700, font=('Segoe UI', 11)).grid(row=2, column=0, sticky='w', pady=8)
        ttk.Label(left, textvariable=self.status, wraplength=700).grid(row=3, column=0, sticky='w')
        right = ttk.Frame(self.root, padding=10, width=370)
        right.grid(row=0, column=1, sticky='ns')
        right.grid_propagate(False)
        right.rowconfigure(1, weight=1)
        right.columnconfigure(0, weight=1)
        model_text = 'Tag model ready (9.9 MB)' if self.detector else 'No tag model loaded; QR / bright-region mode only'
        if self.model_error:
            model_text += '\n' + self.model_error
        ttk.Label(right, text=model_text, wraplength=340).grid(row=0, column=0, sticky='w', pady=(0, 10))
        notebook = ttk.Notebook(right)
        notebook.grid(row=1, column=0, sticky='nsew')
        self.editable = []
        pages = {}
        for name in ['Run', 'Camera', 'Regions', 'Notes']:
            shell = ttk.Frame(notebook)
            notebook.add(shell, text=name)
            canvas = tk.Canvas(shell, highlightthickness=0, width=325)
            scroll = ttk.Scrollbar(shell, orient='vertical', command=canvas.yview)
            canvas.configure(yscrollcommand=scroll.set)
            scroll.pack(side='right', fill='y')
            canvas.pack(side='left', fill='both', expand=True)
            body = ttk.Frame(canvas, padding=6)
            item = canvas.create_window((0, 0), window=body, anchor='nw')
            body.bind('<Configure>', lambda event, c=canvas: c.configure(scrollregion=c.bbox('all')))
            canvas.bind('<Configure>', lambda event, c=canvas, i=item: c.itemconfigure(i, width=event.width))
            body.columnconfigure(1, weight=1)
            pages[name] = body

        def entry(page, label, key):
            row = page.grid_size()[1]
            ttk.Label(page, text=label).grid(row=row, column=0, sticky='w', pady=4)
            field = ttk.Entry(page, textvariable=self.vars[key], width=18)
            field.grid(row=row, column=1, sticky='ew', padx=(5, 0), pady=4)
            self.editable.append(field)

        page = pages['Run']
        for index, (label, mode) in enumerate([('1 Scene suppression', 'scene'), ('2 Multi-user IDs', 'multi')]):
            control = ttk.Radiobutton(page, text=label, variable=self.vars['mode'], value=mode, command=self.mode_changed)
            control.grid(row=index, column=0, columnspan=2, sticky='w', pady=4)
            self.editable.append(control)
        for label, key in [('Condition', 'condition'), ('Repetition', 'repetition'), ('Duration (s)', 'duration'),
                           ('Analysis target (Hz)', 'sample_hz'), ('Snapshot interval (s)', 'snapshot_s'),
                           ('ID confidence', 'confidence'), ('Green threshold', 'led_threshold')]:
            entry(page, label, key)
        for user in USERS:
            control = ttk.Checkbutton(page, text=user, variable=self.expected[user])
            control.grid(row=page.grid_size()[1], column=0, columnspan=2, sticky='w', pady=5)
            self.editable.append(control)
        control = ttk.Checkbutton(page, text='Setup / baseline checked', variable=self.verified)
        control.grid(row=page.grid_size()[1], column=0, columnspan=2, sticky='w', pady=8)
        self.editable.append(control)

        page = pages['Camera']
        for label, key in [('Camera index', 'camera'), ('Width (px)', 'width'), ('Height (px)', 'height'), ('Frame rate request', 'fps')]:
            entry(page, label, key)
        self.connect_button = ttk.Button(page, text='Connect camera', command=self.connect)
        self.connect_button.grid(row=page.grid_size()[1], column=0, columnspan=2, sticky='ew', pady=7)
        self.editable.append(self.connect_button)
        for label, key in [('Exposure duration label', 'exposure_label'), ('Exposure evidence', 'exposure_evidence'),
                           ('Exposure control', 'exposure_raw'), ('Gain control', 'gain_raw'),
                           ('Auto exposure control', 'auto_exposure_raw'), ('Focus control', 'focus_raw'),
                           ('Autofocus control', 'auto_focus_raw'), ('Auto WB control', 'auto_wb_raw')]:
            entry(page, label, key)
        button = ttk.Button(page, text='Apply controls', command=self.apply_controls)
        button.grid(row=page.grid_size()[1], column=0, columnspan=2, sticky='ew', pady=8)
        self.editable.append(button)
        ttk.Label(page, textvariable=self.camera_status, wraplength=310).grid(row=page.grid_size()[1], column=0, columnspan=2, sticky='w', pady=8)

        page = pages['Regions']
        selector = ttk.Combobox(page, textvariable=self.vars['region'], values=['QR'] + USERS, state='readonly', width=24)
        selector.grid(row=0, column=0, columnspan=2, sticky='ew', pady=5)
        self.region_selector = selector
        self.editable.append(selector)
        ttk.Label(page, textvariable=self.region_status, wraplength=310).grid(row=1, column=0, columnspan=2, sticky='w', pady=8)
        for label, command in [('Clear regions', self.clear_regions), ('Save setup', self.save_setup),
                               ('Load setup', self.load_setup)]:
            button = ttk.Button(page, text=label, command=command)
            button.grid(row=page.grid_size()[1], column=0, columnspan=2, sticky='ew', pady=5)
            self.editable.append(button)
        entry(page, 'Expected QR value', 'qr_payload')
        button = ttk.Button(page, text='Use currently read QR value', command=self.use_read_qr)
        button.grid(row=page.grid_size()[1], column=0, columnspan=2, sticky='ew', pady=8)
        self.editable.append(button)
        page = pages['Notes']
        for label, key in [('Camera name', 'camera_name'), ('Distance', 'distance'), ('Angle', 'angle'),
                           ('Motion', 'motion'), ('Lighting', 'lighting'), ('Printed QR width', 'qr_width'),
                           ('Gain / ISO note', 'gain_note'), ('Firmware', 'firmware'),
                           ('Symbol interval (us)', 'symbol_us'), ('Row timing (us)', 'row_timing')]:
            entry(page, label, key)
        actions = ttk.Frame(right)
        actions.grid(row=2, column=0, sticky='ew', pady=(10, 0))
        actions.columnconfigure((0, 1), weight=1)
        self.start_button = ttk.Button(actions, text='Start run', command=self.start_run)
        self.start_button.grid(row=0, column=0, sticky='ew', padx=2, pady=3)
        ttk.Button(actions, text='Stop', command=lambda: self.finish_run('stopped_early')).grid(row=0, column=1, sticky='ew', padx=2, pady=3)
        ttk.Button(actions, text='Snapshot', command=self.snapshot).grid(row=1, column=0, sticky='ew', padx=2, pady=3)
        ttk.Button(actions, text='Open report', command=self.open_report).grid(row=1, column=1, sticky='ew', padx=2, pady=3)
        self.report_button = ttk.Button(actions, text='Build comparison', command=self.build_comparison)
        self.report_button.grid(row=2, column=0, columnspan=2, sticky='ew', padx=2, pady=3)

    def mode_changed(self):
        self.vars['duration'].set('45')
        self.verified.set(False)

    def config(self, preview=False):
        config = {key: var.get().strip() for key, var in self.vars.items()}
        for key in ['duration', 'sample_hz', 'snapshot_s', 'confidence']:
            config[key] = float(config[key])
            if not np.isfinite(config[key]) or config[key] <= 0:
                raise ValueError(f'{key} must be positive.')
        if config['confidence'] > 1 or config['sample_hz'] > 60:
            raise ValueError('Confidence must be <= 1 and sample rate <= 60 Hz.')
        config['led_threshold'] = int(config['led_threshold'])
        if not 1 <= config['led_threshold'] <= 255:
            raise ValueError('Green threshold must be 1..255.')
        if not config['condition']:
            raise ValueError('Condition cannot be blank.')
        if not preview and config['mode'] == 'scene' and not config['qr_payload']:
            raise ValueError('Read the QR at baseline and select Use currently read QR value, or enter its known value.')
        config['expected'] = [u for u, var in self.expected.items() if var.get()]
        if not config['expected'] and not preview:
            raise ValueError('Select at least one expected user.')
        config['regions'] = {k: list(v) for k, v in self.regions.items()}
        if preview:
            config['regions'].setdefault('QR', [0, 0, 1, 1])
            config['expected'] = [u for u in config['expected'] if u in self.regions]
        else:
            validate_regions(config['regions'], config['expected'], config['mode'] == 'scene')
        config.update(demo=self.demo, model_enabled=self.detector is not None, setup_checked=self.verified.get(),
                      model_sha256=self.detector.sha256 if self.detector else None,
                      model_cpu_threads=self.detector.threads if self.detector else None,
                      nms_iou=0.45, qr_enhancement='grayscale CLAHE clip=2 tile=8x8')
        return config

    def connect(self):
        if self.running:
            return
        try:
            index = int(self.vars['camera'].get())
            width, height = int(self.vars['width'].get()), int(self.vars['height'].get())
            fps = float(self.vars['fps'].get())
            if min(width, height) < 64 or not 0 < fps <= 240:
                raise ValueError('Invalid resolution or frame rate.')
            if self.camera:
                self.camera.close()
            self.camera = Camera(index, width, height, fps, self.demo)
            self.last_frame_id = -1
            self.last_result = None
            self.epoch += 1
            self.verified.set(False)
            self.status.set('Connecting camera...')
        except Exception as error:
            messagebox.showerror('Camera', str(error))

    def apply_controls(self):
        if not self.camera or self.running:
            return
        try:
            settings = {key: float(self.vars[key].get()) for key in ['exposure_raw', 'gain_raw', 'auto_exposure_raw', 'focus_raw', 'auto_focus_raw', 'auto_wb_raw'] if self.vars[key].get().strip()}
            if not all(np.isfinite(v) for v in settings.values()):
                raise ValueError('Camera controls must be finite numbers.')
            self.camera.commands.put(settings)
            self.verified.set(False)
            self.status.set('Controls requested; inspect readback and allow at least 2 seconds to settle.')
        except ValueError as error:
            messagebox.showerror('Camera controls', str(error))

    def set_locked(self, locked):
        for widget in self.editable:
            widget.configure(state='disabled' if locked else ('readonly' if widget is self.region_selector else 'normal'))
        self.start_button.configure(state='disabled' if locked else 'normal')
        self.report_button.configure(state='disabled' if locked else 'normal')

    def start_run(self):
        if self.running:
            return
        try:
            config = self.config()
            if not self.camera or self.camera.error or not self.camera.get():
                raise ValueError('Connect a working camera first.')
            if time.monotonic() - self.camera.get()[1] > 1:
                raise ValueError('No recent camera frame.')
            if not self.camera.commands.empty() or time.monotonic()-self.camera.info.get('controls_applied_monotonic', 0) < 2:
                raise ValueError('Allow camera settings to settle for at least 2 seconds.')
            if not self.verified.get():
                raise ValueError('Check setup, baseline QR reading and labelled tag regions; then tick Setup / baseline checked.')
            if config['mode'] == 'multi' and not self.detector:
                raise ValueError('Multi-user identification needs the tag model. ' + self.model_error)
            stamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            safe = ''.join(c if c.isascii() and (c.isalnum() or c in '-_') else '_' for c in config['condition'])[:80]
            self.run_dir = ROOT / ('demo_runs' if self.demo else 'runs') / f'{stamp}_{safe}'
            self.run_dir.mkdir(parents=True)
            self.run_config = config
            self.last_result = None
            self.run_start = time.monotonic()
            self.deadline = self.run_start + config['duration']
            self.next_snapshot = 0
            self.rows = []
            self.epoch += 1
            write_json(self.run_dir / 'metadata.json', {'schema_version': 1, 'started_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
                       'config': config, 'camera_start': dict(self.camera.info), 'environment': environment()})
            self.log = (self.run_dir / 'samples.jsonl').open('w', encoding='utf-8', buffering=1)
            self.running = True
            self.set_locked(True)
            self.status.set('Recording ' + config['condition'])
        except Exception as error:
            messagebox.showerror('Run setup', str(error))

    def finish_run(self, status='completed', error=None):
        if not self.running:
            return
        self.running = False
        self.log.close()
        self.log = None
        self.set_locked(False)
        if status == 'completed' and not self.rows:
            status = 'no_samples'
        completion = {'status': 'demo_' + status if self.demo else status, 'error': error,
                      'low_sample_count': len(self.rows) < 30,
                      'actual_duration_s': time.monotonic()-self.run_start, 'camera_end': dict(self.camera.info)}
        write_json(self.run_dir / 'completion.json', completion)
        write_json(self.run_dir / 'summary.json', summarize(self.rows, self.run_config))
        suffix = ' Fewer than 30 samples; consider a longer repetition.' if len(self.rows) < 30 else ''
        self.status.set(f'{status}: {len(self.rows)} samples saved. Building report...' + suffix)
        self.executor.submit(self._report, self.run_dir)
        try:
            self.vars['repetition'].set(str(int(self.vars['repetition'].get())+1))
        except ValueError:
            pass

    @staticmethod
    def _report(directory):
        try:
            from reporting import report_run
            report_run(directory)
        except Exception as error:
            write_json(directory / 'report_error.json', {'error': str(error)})

    def save_setup(self):
        write_json(ROOT / 'local_setup.json', {'settings': {k: v.get() for k, v in self.vars.items()},
                   'regions': self.regions, 'expected': [u for u, v in self.expected.items() if v.get()]})
        self.status.set('Setup saved')

    def load_setup(self):
        try:
            data = json.loads((ROOT / 'local_setup.json').read_text())
            for key, value in data['settings'].items():
                if key in self.vars:
                    self.vars[key].set(value)
            self.regions = data['regions']
            for user, var in self.expected.items():
                var.set(user in data['expected'])
            self.verified.set(False)
            self.region_status.set('Regions: ' + ', '.join(self.regions))
            self.status.set('Setup loaded; check regions against the current camera view')
        except Exception as error:
            messagebox.showerror('Load setup', str(error))

    def clear_regions(self):
        self.regions.clear()
        self.region_status.set('Regions: none')
        self.verified.set(False)

    def use_read_qr(self):
        if self.last_result:
            result = self.last_result[1]
            payload = result['qr_payload'] or result['qr_enhanced_payload']
            if payload and time.time()-result['capture_unix_s'] < 5:
                self.vars['qr_payload'].set(payload)
                self.verified.set(False)
                self.status.set('Expected QR value set from baseline reading.')
                return
        self.status.set('No recent QR value. Make the scene QR readable at baseline first.')

    def open_report(self):
        if self.run_dir and (self.run_dir / 'report.html').exists():
            webbrowser.open((self.run_dir / 'report.html').as_uri())
        elif self.run_dir and (self.run_dir / 'report_error.json').exists():
            messagebox.showerror('Report', (self.run_dir / 'report_error.json').read_text())
        else:
            self.status.set('Report unavailable yet; finish a run first.')

    def build_comparison(self):
        def build():
            from reporting import report_all
            root = ROOT / ('demo_runs' if self.demo else 'runs')
            try:
                report_all(root)
            except Exception as error:
                root.mkdir(exist_ok=True)
                write_json(root / 'comparison_error.json', {'error': str(error)})
        self.executor.submit(build)
        self.status.set('Comparison queued. Output: runs/index.html (demo_runs in demo mode).')

    def snapshot(self):
        if not self.last_result:
            self.status.set('Define regions and wait for a fresh analysis before taking a snapshot.')
            return
        frame, result, config = self.last_result
        target = self.run_dir / 'snapshots' if self.running else ROOT / 'snapshots'
        stamp = dt.datetime.now().strftime('manual_%Y%m%d_%H%M%S_%f')
        try:
            save_snapshot(target, stamp, frame, result, config)
            self.status.set('Snapshot saved: ' + str(target / (stamp + '_raw.png')))
        except Exception as error:
            messagebox.showerror('Snapshot', str(error))

    def point(self, event):
        if not self.display_box:
            return None
        x, y, w, h = self.display_box
        return (float(np.clip((event.x-x)/w, 0, 1)), float(np.clip((event.y-y)/h, 0, 1)))

    def drag_start(self, event):
        if not self.running:
            self.drag = self.point(event)

    def drag_move(self, event):
        if self.drag and self.display_box:
            point = self.point(event)
            x, y, w, h = self.display_box
            self.canvas.delete('selection')
            self.canvas.create_rectangle(x+self.drag[0]*w, y+self.drag[1]*h, x+point[0]*w, y+point[1]*h, outline='#ffcf52', width=2, tags='selection')

    def drag_end(self, event):
        if self.drag and not self.running:
            point = self.point(event)
            region = [min(self.drag[0], point[0]), min(self.drag[1], point[1]), max(self.drag[0], point[0]), max(self.drag[1], point[1])]
            if region[2]-region[0] >= .02 and region[3]-region[1] >= .02:
                self.regions[self.vars['region'].get()] = region
                self.region_status.set('Regions: ' + ', '.join(self.regions))
                self.verified.set(False)
        self.drag = None
        self.canvas.delete('selection')

    def analyze(self, packet, config):
        frame_id, captured, unix, frame = packet
        if self.detector:
            self.detector.confidence = config['confidence']
        result = self.analyzer.measure(frame, config)
        result.update(frame_id=frame_id, capture_unix_s=unix,
                      frame_age_ms=(time.monotonic()-captured)*1000)
        return frame, result, config

    def _consume(self):
        if not self.future or not self.future.done():
            return
        epoch, captured = self.future_context
        future, self.future = self.future, None
        frame, result, config = future.result()
        if epoch != self.epoch:
            return
        result['elapsed_s'] = captured-self.run_start if self.running else None
        self.last_result = frame, result, config
        if self.running and self.run_start <= captured <= self.deadline:
            result['elapsed_s'] = captured-self.run_start
            if result['elapsed_s'] >= self.next_snapshot:
                stem = f'sample_{result["frame_id"]:08d}'
                result['snapshot'] = stem
                save_snapshot(self.run_dir / 'snapshots', stem, frame, result, config)
                self.next_snapshot += config['snapshot_s']
            self.log.write(json.dumps(result, allow_nan=False) + '\n')
            self.log.flush()
            self.rows.append(result)
            summary = summarize(self.rows, config)
            def pct(v):
                return 'N/A' if v is None else f'{v*100:.1f}%'
            self.metrics.set(f"Samples {len(self.rows)} | QR raw {pct(summary['qr_raw_match_rate'])} | QR recovered {pct(summary['qr_any_match_rate'])}\nAll IDs correct {pct(summary['all_correct_rate'])} | Analysis {summary['observed_sample_hz'] or 0:.1f} Hz | {max(0, self.deadline-time.monotonic()):.1f}s left")
        elif not self.running:
            payload = result['qr_payload'] or result['qr_enhanced_payload'] or '(not decoded)'
            shown_payload = payload if len(payload) <= 85 else payload[:82] + '...'
            self.metrics.set(f"QR read: {shown_payload}\nMatch: {result['qr_match']} | Enhanced match: {result['qr_enhanced_match']} | IDs: {result['assignments']} | {result['processing_ms']:.0f} ms")

    def tick(self):
        if self.closing:
            return
        try:
            self._consume()
            if self.running and time.monotonic() >= self.deadline and self.future is None:
                self.finish_run()
            if self.camera:
                if self.camera.error:
                    self.finish_run('camera_error', self.camera.error)
                    self.status.set(self.camera.error)
                info = self.camera.info
                self.camera_status.set('\n'.join(f'{k}: {v}' for k, v in info.items() if k != 'controls_applied_monotonic'))
                packet = self.camera.get()
                if packet:
                    self.draw(packet[3])
                    if self.running and time.monotonic()-packet[1] > 2:
                        self.finish_run('camera_stalled', 'No fresh frame for 2 seconds')
                    if not self.future and packet[0] != self.last_frame_id and (not self.running or time.monotonic() < self.deadline):
                        try:
                            config = self.run_config if self.running else self.config(preview=True)
                        except (ValueError, tk.TclError) as error:
                            self.status.set(str(error))
                        else:
                            if time.monotonic()-self.last_sample_time >= 1/config['sample_hz']:
                                self.last_frame_id = packet[0]
                                self.last_sample_time = time.monotonic()
                                self.future_context = self.epoch, packet[1]
                                self.future = self.executor.submit(self.analyze, packet, config)
        except Exception as error:
            self.finish_run('analysis_error', str(error))
            self.status.set('Analysis error: ' + str(error))
        self.root.after(40, self.tick)

    def draw(self, frame):
        # Live preview has only ground-truth regions; predictions appear on their exact source frame in snapshots.
        config = {'regions': self.regions, 'expected': [u for u, v in self.expected.items() if v.get()]}
        image = overlay(frame, {}, config)
        h, w = image.shape[:2]
        cw, ch = max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())
        scale = min(cw/w, ch/h)
        dw, dh = max(1, int(w*scale)), max(1, int(h*scale))
        x, y = (cw-dw)//2, (ch-dh)//2
        self.display_box = x, y, dw, dh
        image = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), (dw, dh))
        self.photo = ImageTk.PhotoImage(Image.fromarray(image))
        self.canvas.delete('preview')
        self.canvas.create_image(x, y, image=self.photo, anchor='nw', tags='preview')
        self.canvas.tag_lower('preview')

    def close(self):
        self.closing = True
        self.finish_run('window_closed')
        if self.camera:
            self.camera.close()
        self.executor.shutdown(wait=True, cancel_futures=False)
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo', action='store_true', help='Still-image rehearsal; stored separately from experiments')
    parser.add_argument('--yolo', action='store_true', help='Enable tag model; kept for compatibility and already the default')
    parser.add_argument('--qr-only', action='store_true', help='Disable tag model for QR-only pilots')
    parser.add_argument('--model-path', default=None, help='Optional path to RoEmotion_LED_Recognition_Model_YOLOv26n.tflite')
    args = parser.parse_args()
    app = App(demo=args.demo, model=not args.qr_only, model_path=args.model_path)
    app.root.mainloop()


if __name__ == '__main__':
    main()
