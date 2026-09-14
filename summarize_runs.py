import argparse
import csv
import json
from pathlib import Path


def summarize_run(run_dir: Path):
    log_path = run_dir / "log.csv"
    if not log_path.exists():
        return None

    rows = list(csv.DictReader(log_path.open(newline="", encoding="utf-8")))
    if not rows:
        return None

    frames = len(rows)
    qr_detected = sum(int(row.get("qr_detected") or 0) for row in rows)
    qr_match = sum(int(row.get("qr_match") or 0) for row in rows)
    led_visible = sum(1 for row in rows if int(float(row.get("led_blob_count") or 0)) > 0)
    all_expected = sum(int(row.get("yolo_all_expected_detected") or 0) for row in rows)

    user_hits = {user: 0 for user in ["User_1", "User_2", "User_3"]}
    for row in rows:
        seen = set(filter(None, (row.get("yolo_users") or "").split("|")))
        for user in user_hits:
            user_hits[user] += int(user in seen)

    def rate(count):
        return 0.0 if frames == 0 else count / frames

    metadata = {}
    metadata_path = run_dir / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    return {
        "run": run_dir.name,
        "mode": metadata.get("experiment_mode", ""),
        "condition": metadata.get("condition", ""),
        "intended_exposure_duration": metadata.get("intended_exposure_duration", ""),
        "expected_users": "|".join(metadata.get("expected_users", [])),
        "frames": frames,
        "qr_detect_rate": rate(qr_detected),
        "qr_match_rate": rate(qr_match),
        "led_visible_rate": rate(led_visible),
        "yolo_all_expected_rate": rate(all_expected),
        "user_1_rate": rate(user_hits["User_1"]),
        "user_2_rate": rate(user_hits["User_2"]),
        "user_3_rate": rate(user_hits["User_3"]),
    }


def main():
    parser = argparse.ArgumentParser(description="Summarize RoEmotion PC experiment runs")
    parser.add_argument("--runs", type=Path, default=Path("runs"), help="Runs directory")
    parser.add_argument("--output", type=Path, default=Path("runs_summary.csv"), help="Summary CSV path")
    args = parser.parse_args()

    summaries = []
    for run_dir in sorted(args.runs.iterdir()) if args.runs.exists() else []:
        if run_dir.is_dir():
            summary = summarize_run(run_dir)
            if summary:
                summaries.append(summary)

    if not summaries:
        print("No runs found.")
        return

    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)

    print(args.output.resolve())


if __name__ == "__main__":
    main()
