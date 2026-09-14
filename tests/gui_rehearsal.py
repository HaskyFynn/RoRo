"""Opt-in real Tk rehearsal with a synthetic board; never experimental data."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import ImageGrab
from app import App
from core import ROOT
from core import PAYLOAD


def main():
    app = App(demo=True, model=True)
    failures = []
    run_dirs = []
    app.root.report_callback_exception = lambda kind, error, trace: failures.append(str(error))
    app.vars['duration'].set('3')
    app.vars['snapshot_s'].set('1')
    app.vars['condition'].set('SOFTWARE_REHEARSAL_ONLY')
    app.regions = {'QR': [.58, .17, .98, .72], 'User_1': [.04, .64, .47, .88]}
    app.region_status.set('Regions: QR, User_1')
    app.connect()
    artifacts = ROOT / 'verification-artifacts'
    artifacts.mkdir(exist_ok=True)
    def start():
        app.use_read_qr()
        if not app.vars['qr_payload'].get():
            app.vars['qr_payload'].set(PAYLOAD)
        app.verified.set(True)
        app.start_run()
    def capture():
        app.root.update_idletasks()
        x, y = app.root.winfo_rootx(), app.root.winfo_rooty()
        try:
            ImageGrab.grab(bbox=(x, y, x+app.root.winfo_width(), y+app.root.winfo_height())).save(artifacts / 'desktop.png')
        except OSError as error:
            print('Desktop screenshot unavailable:', error)
        assert app.canvas.winfo_width() > 400
        assert app.start_button.winfo_rooty()+app.start_button.winfo_height() <= y+app.root.winfo_height()
        app.root.geometry('900x650')
    def compact():
        app.root.update_idletasks()
        x, y = app.root.winfo_rootx(), app.root.winfo_rooty()
        try:
            ImageGrab.grab(bbox=(x, y, x+app.root.winfo_width(), y+app.root.winfo_height())).save(artifacts / 'compact.png')
        except OSError as error:
            print('Compact screenshot unavailable:', error)
        assert app.canvas.winfo_width() > 350
        assert app.report_button.winfo_rooty()+app.report_button.winfo_height() <= y+app.root.winfo_height()
    def multi():
        if app.running:
            failures.append('Scene run did not finish on time')
            return
        run_dirs.append(app.run_dir)
        app.vars['mode'].set('multi')
        app.vars['condition'].set('SOFTWARE_MULTI_REHEARSAL_ONLY')
        app.vars['duration'].set('2')
        app.regions['User_2'] = [.52, .76, .72, .98]
        app.regions['User_3'] = [.76, .76, .98, .98]
        for var in app.expected.values():
            var.set(True)
        app.verified.set(True)
        app.start_run()
    app.root.after(1500, start)
    app.root.after(3000, capture)
    app.root.after(5000, compact)
    app.root.after(6500, multi)
    app.root.after(12000, app.close)
    app.root.mainloop()
    assert not failures, failures
    assert app.run_dir and (app.run_dir / 'report.html').exists(), 'Missing GUI-generated report'
    assert app.rows, 'GUI produced no measurements'
    assert run_dirs and (run_dirs[0] / 'report.html').exists(), 'Missing scene report'
    assert all(set(r['assignments']) == {'User_1', 'User_2', 'User_3'} for r in app.rows)
    ids = [r['frame_id'] for r in app.rows]
    assert len(ids) == len(set(ids)), 'Repeated frame IDs'
    assert all(r['elapsed_s'] >= 0 for r in app.rows)
    assert not app.running
    print(f'GUI rehearsal passed: {len(app.rows)} fresh samples, report and snapshots: {app.run_dir}')


if __name__ == '__main__':
    main()
