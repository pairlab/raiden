#!/usr/bin/env python
"""Per-episode QC of real demos while they are being recorded (checks: ``raiden/qc.py``).

For every finished episode of a task (metadata.json with complete=true) it prints one line, PASS or the reasons,
then the running tally and the croissant coverage of the blue box. Results go to data/qc/<task>/ (qc_log.txt,
qc_results.json); ``rd record --ui`` writes the same files. Deleted episodes drop out of the tally.

    uv run python scripts/qc_episodes.py croissant_oven_real_newrig --watch
"""

import argparse
import json
import time
import traceback
from pathlib import Path

from raiden import qc
from raiden._config import CAMERA_CONFIG
from raiden.camera_config import CameraConfig

REPO = Path(__file__).resolve().parents[1]


def main():
    d = qc.QCSettings()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task")
    ap.add_argument("--watch", action="store_true", help="keep checking new episodes every 5 s")
    ap.add_argument("--vla", default=str(qc.VLA_DEFAULT), help="vla-benchmark checkout (rig, task JSON)")
    ap.add_argument("--ref-tips", type=int, nargs=2, default=list(d.ref_tips), help="wrist fingertip rows (blue, black), arm at home")
    ap.add_argument("--max-tip-px", type=int, default=d.max_tip_px)
    ap.add_argument("--min-wrist-mean", type=float, default=d.min_wrist_mean)
    ap.add_argument("--scene-mean", type=float, nargs=2, default=list(d.scene_mean))
    ap.add_argument("--max-pause", type=float, default=d.max_pause)
    ap.add_argument("--max-closes", type=int, default=d.max_closes)
    ap.add_argument("--min-s", type=float, default=d.min_s)
    ap.add_argument("--max-s", type=float, default=d.max_s)
    ap.add_argument("--out", default="", help="default data/qc/<task>")
    args = ap.parse_args()
    from raiden.db.database import get_db

    st = qc.QCSettings(tuple(args.ref_tips), args.max_tip_px, args.min_wrist_mean, tuple(args.scene_mean),
                       args.max_pause, args.max_closes, args.min_s, args.max_s)
    raw = REPO / "data/raw" / args.task
    out = Path(args.out) if args.out else REPO / "data/qc" / args.task
    out.mkdir(parents=True, exist_ok=True)
    state_file, log_file = out / "qc_results.json", out / "qc_log.txt"
    results = json.load(open(state_file)) if state_file.exists() else {}
    geo, cfg = qc.load_geometry(Path(args.vla)), CameraConfig(CAMERA_CONFIG)

    def say(msg):
        print(msg, flush=True)
        with open(log_file, "a") as f:
            f.write(msg + "\n")

    say(f"QC {args.task}: {raw}  (blue box x {geo.blue[0]:.3f}-{geo.blue[1]:.3f}, y {geo.blue[2]:+.3f}..{geo.blue[3]:+.3f};"
        f" white x {geo.white[0]:.3f}-{geo.white[1]:.3f}, y ±{geo.white[3]:.3f}; wrist tips {list(st.ref_tips)})")
    while True:
        changed = False
        eps = sorted(p for p in raw.iterdir() if p.is_dir() and p.name.isdigit()) if raw.exists() else []
        for gone in set(results) - {e.name for e in eps}:
            say(f"{gone} deleted: dropped from the tally")
            del results[gone]
            changed = True
        for ep in eps:
            mf = ep / "metadata.json"
            if not mf.exists() or not json.load(open(mf)).get("complete", False):
                continue
            stamp = mf.stat().st_mtime
            if (ep.name in results and results[ep.name].get("_stamp") == stamp
                    and results[ep.name].get("_qc_version") == qc.QC_VERSION):
                continue
            try:
                r = qc.check(ep, args.task, geo, cfg, st, get_db())
            except Exception as e:  # one bad episode must not stop the watcher
                r = dict(episode=ep.name, reasons=[f"QC error: {e!r}"], duration_s=None)
                traceback.print_exc()
            r["_stamp"] = stamp
            results[ep.name] = r
            changed = True
            say(qc.line(r))
        if changed:
            qc.apply_duration_outliers(results)
            say(qc.tally_text(results, geo))
            json.dump(results, open(state_file, "w"), indent=1)
        if not args.watch:
            break
        time.sleep(5)


if __name__ == "__main__":
    main()
