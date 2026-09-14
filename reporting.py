"""Rebuild reports from durable logs; repetitions remain separate observations."""
import argparse
import csv
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from core import ROOT, summarize, write_json


def load_rows(path):
    rows = []
    warnings = []
    with Path(path).open(encoding='utf-8') as source:
        for index, line in enumerate(source, 1):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                warnings.append(f'Unreadable log line {index}; excluded.')
    return rows, warnings


def number(value):
    return 'N/A' if value is None else f'{value:.3f}'


def report_run(directory):
    directory = Path(directory)
    metadata = json.loads((directory / 'metadata.json').read_text(encoding='utf-8'))
    rows, warnings = load_rows(directory / 'samples.jsonl')
    summary = summarize(rows, metadata['config'])
    completion_path = directory / 'completion.json'
    completion = json.loads(completion_path.read_text()) if completion_path.exists() else {'status': 'interrupted'}
    summary.update(completion=completion, warnings=warnings)
    write_json(directory / 'summary.json', summary)
    flat = ['frame_id', 'elapsed_s', 'capture_unix_s', 'frame_age_ms', 'processing_ms',
            'qr_localized', 'qr_match', 'qr_enhanced_match', 'qr_any_match', 'extra_detections']
    users = metadata['config']['expected']
    with (directory / 'confusion.csv').open('w', newline='', encoding='utf-8') as target:
        labels = ['User_1', 'User_2', 'User_3', 'miss']
        writer = csv.writer(target)
        writer.writerow(['physical_user'] + labels)
        for user in users:
            writer.writerow([user] + [summary['per_user'][user]['confusion_counts'][label] for label in labels])
    if summary['id_samples']:
        matrix = np.array([[summary['per_user'][u]['confusion_counts'][label] for label in labels] for u in users])
        fig, ax = plt.subplots(figsize=(6, 3), layout='constrained')
        ax.imshow(matrix, cmap='Greens', vmin=0)
        for (y, x), value in np.ndenumerate(matrix):
            ax.text(x, y, str(value), ha='center', va='center', color='white' if value > matrix.max()/2 else 'black')
        ax.set_xticks(range(4), labels)
        ax.set_yticks(range(len(users)), users)
        ax.set_xlabel('Predicted label (region assignment)')
        ax.set_ylabel('Physical user')
        fig.savefig(directory / 'confusion.png', dpi=200)
        fig.savefig(directory / 'confusion.pdf')
        plt.close(fig)
    with (directory / 'samples.csv').open('w', newline='', encoding='utf-8') as target:
        writer = csv.DictWriter(target, fieldnames=flat + [f'{u}_prediction' for u in users])
        writer.writeheader()
        for row in rows:
            writer.writerow({**{k: row.get(k) for k in flat},
                             **{f'{u}_prediction': row['assignments'].get(u) for u in users}})
    if rows:
        fig, ax = plt.subplots(figsize=(8, 3.4), layout='constrained')
        series = []
        if metadata['config']['mode'] == 'scene':
            series += [('QR raw', [r['qr_match'] for r in rows], '#287b4b'),
                       ('QR raw or enhanced', [r['qr_any_match'] for r in rows], '#aa536d')]
        if summary['id_samples']:
            series.append(('All IDs correct in regions', [all(r['assignments'].get(u) == u for u in users) for r in rows], '#236d9e'))
        for index, (label, values, color) in enumerate(series):
            for success, marker, outcome_color in [(True, 'o', '#287b4b'), (False, 'x', '#aa354d')]:
                times = [r['elapsed_s'] for r, value in zip(rows, values) if bool(value) == success]
                ax.scatter(times, [index] * len(times), s=20, marker=marker, color=outcome_color,
                           label=('Success' if success else 'Failure') if index == 0 else None)
        ax.set_yticks(range(len(series)), [s[0] for s in series])
        ax.set_ylim(-.5, max(.5, len(series)-.5))
        ax.invert_yaxis()
        ax.set_xlabel('Elapsed time (s) / one mark per analyzed frame')
        ax.legend(loc='upper right')
        ax.set_title(metadata['config']['condition'])
        fig.savefig(directory / 'timeline.png', dpi=200)
        fig.savefig(directory / 'timeline.pdf')
        plt.close(fig)
    values = {k: v for k, v in summary.items() if isinstance(v, (float, int)) or v is None}
    table = ''.join(f'<tr><td>{html.escape(k)}</td><td>{number(v)}</td></tr>' for k, v in values.items())
    per_user = ''.join(f'<tr><td>{u}</td><td>{number(v["correct_rate"])}</td><td>{number(v["miss_rate"])}</td><td>{number(v["wrong_id_rate"])}</td></tr>' for u, v in summary['per_user'].items())
    links = ''.join(f'<a href="{p.relative_to(directory).as_posix()}">{html.escape(p.stem)}</a><br>' for p in sorted((directory / 'snapshots').glob('*_raw.png')))
    page = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>RoEmotion run report</title>
<style>body{{font:15px system-ui;max-width:1050px;margin:32px auto;padding:0 20px;color:#202526}}td,th{{text-align:left;padding:6px 18px 6px 0;border-bottom:1px solid #ddd}}img{{max-width:100%}}pre{{white-space:pre-wrap}}a{{color:#236d9e}}</style>
<h1>{html.escape(metadata['config']['condition'])}</h1>
<p>Status: {html.escape(completion['status'])}. Rates are fractions of fresh analyzed samples. N/A means no eligible samples.</p>
{'<p>Fewer than 30 analyzed samples: consider a longer repetition. This threshold is a practical collection check, not a power calculation.</p>' if summary['samples'] < 30 else ''}
<table>{table}</table><h2>Wearer regions</h2><table><tr><th>User</th><th>Correct ID</th><th>Miss</th><th>Wrong ID</th></tr>{per_user}</table>
<p>Ground truth assumes one physical tag remains inside each labelled region. Extra detections include unassigned and duplicate predictions.</p>
{'<img src="timeline.png" alt="Per-sample timeline">' if rows else ''}
{'<img src="confusion.png" alt="Region confusion counts">' if summary['id_samples'] else ''}<h2>Unmodified snapshots</h2>{links}
<h2>Settings and provenance</h2><pre>{html.escape(json.dumps(metadata, indent=2))}</pre></html>'''
    (directory / 'report.html').write_text(page, encoding='utf-8')
    return summary


def report_all(root):
    root = Path(root)
    records = []
    for path in sorted(root.glob('*/metadata.json')):
        if not (path.parent / 'samples.jsonl').exists():
            continue
        metadata = json.loads(path.read_text())
        summary = report_run(path.parent)
        config = metadata['config']
        records.append({'run': path.parent.name, 'mode': config['mode'], 'condition': config['condition'],
                        'repetition': config['repetition'], 'status': summary['completion']['status'],
                        'exposure_label': config['exposure_label'], 'expected': '|'.join(config['expected']),
                        **{f'{user}_{metric}': summary['per_user'].get(user, {}).get(metric)
                           for user in ['User_1', 'User_2', 'User_3'] for metric in ['correct_rate', 'miss_rate', 'wrong_id_rate']},
                        **{k: summary[k] for k in ['samples', 'id_samples', 'observed_sample_hz',
                            'qr_raw_match_rate', 'qr_any_match_rate', 'all_correct_rate', 'exact_frame_rate',
                            'extra_detections_per_sample', 'processing_ms_median', 'processing_ms_p95']}})
    if not records:
        raise ValueError('No runs found.')
    with (root / 'runs_summary.csv').open('w', newline='', encoding='utf-8') as target:
        writer = csv.DictWriter(target, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    # Plot each run independently: unlike conditions are never silently pooled.
    fig, ax = plt.subplots(figsize=(max(8, len(records)*.4), 4.5), layout='constrained')
    x = np.arange(len(records))
    for key, label, color in [('qr_raw_match_rate', 'QR raw', '#287b4b'),
                               ('qr_any_match_rate', 'QR raw or enhanced', '#aa536d'),
                               ('all_correct_rate', 'All IDs correct', '#236d9e')]:
        ax.plot(x, [np.nan if r[key] is None else r[key] for r in records], 'o', label=label, color=color)
    ax.set_xticks(x, [f'{i+1}: {r["condition"]} / r{r["repetition"]}' for i, r in enumerate(records)], rotation=60, ha='right', fontsize=8)
    ax.set_ylim(-.04, 1.04)
    ax.set_ylabel('Fraction of analyzed samples')
    ax.legend(loc='upper left', bbox_to_anchor=(0, 1.25), ncol=3)
    fig.savefig(root / 'comparison.png', dpi=220)
    fig.savefig(root / 'comparison.pdf')
    plt.close(fig)
    links = ''.join(f'<li><a href="{r["run"]}/report.html">{html.escape(r["run"])}</a> ({r["status"]})</li>' for r in records)
    (root / 'index.html').write_text(f'<!doctype html><html lang="en"><meta charset="utf-8"><title>RoEmotion runs</title><h1>RoEmotion runs</h1><p>Each point is one repetition; interrupted runs are retained and marked in CSV.</p><a href="runs_summary.csv">Summary CSV</a><br><img style="max-width:100%" src="comparison.png" alt="Run comparison"><ul>{links}</ul></html>', encoding='utf-8')
    return root / 'index.html'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', nargs='?', type=Path, default=ROOT / 'runs')
    args = parser.parse_args()
    if (args.directory / 'metadata.json').exists():
        report_run(args.directory)
        print(args.directory / 'report.html')
    else:
        print(report_all(args.directory))
