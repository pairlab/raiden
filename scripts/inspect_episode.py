#!/usr/bin/env python3
"""Summarise one raw recording: timing, arm motion, gripper behaviour and DB status.

Read-only. Nothing is written, so it is safe to run against an episode while a
recording session is in progress.

    cd ~/robot/raiden
    uv run python scripts/inspect_episode.py data/raw/cube_final_real/0157
    uv run python scripts/inspect_episode.py data/raw/cube_final_real/01[5-9]* --brief

The gripper section reports which ``_GRIPPER_SAFETY_THRESHOLD`` the episode was
recorded under.  The clamp saturates during a grasp, so the lag between commanded
and measured gripper position identifies the constant that was live at the time —
useful for telling pre- and post-2026-09-12 episodes apart when checking sim/real
parity.
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

DB_PATH = Path.home() / ".config" / "raiden" / "db" / "demonstrations.json"

# 0.9 x the constant, which is what the control loop actually applies.
CLAMPS = {
    "old (6.0/71.0, pre-2026-09-12)": 0.9 * 6.0 / 71.0,
    "new (2.0/95.0)": 0.9 * 2.0 / 95.0,
}


def db_status(ep: Path) -> str:
    """Look the episode up in the demonstrations DB by its recorded path."""
    if not DB_PATH.exists():
        return "no DB"
    try:
        rows = json.loads(DB_PATH.read_text())["data"]
    except Exception as exc:  # a half-written DB should not kill the report
        return f"unreadable ({type(exc).__name__})"
    # Paths are stored relative to the repo root; match on the tail so the script
    # works whether the caller passed a relative or absolute path.
    hits = [r for r in rows if Path(r["raw_data_path"]).parts[-2:] == ep.parts[-2:]]
    if not hits:
        return "no DB row"
    if len(hits) > 1:
        return f"{len(hits)} DB ROWS: {', '.join(r['status'] for r in hits)}"
    return hits[0]["status"] + ("  (converted)" if hits[0].get("converted") else "")


def arms(data) -> list:
    return sorted({k.split("_joint_pos")[0] for k in data if k.endswith("_joint_pos")})


def report(ep: Path, brief: bool) -> list:
    """Print the summary; return a list of warning strings."""
    warn = []
    meta = json.loads((ep / "metadata.json").read_text()) if (ep / "metadata.json").exists() else {}

    status = db_status(ep)
    dur = meta.get("duration_s")
    print(f"\n{'=' * 66}")
    print(f"  {ep}")
    print(f"{'=' * 66}")
    print(f"  task         {meta.get('task_name', '?')}   \"{meta.get('task_instruction', '?')}\"")
    print(f"  recorded     {meta.get('timestamp', '?')}   control={meta.get('control', '?')}")
    print(f"  status       {status}")
    print(f"  duration     {dur} s   {meta.get('robot_frames', '?')} frames @ {meta.get('robot_hz', '?')} Hz")
    print(f"  complete     {meta.get('complete')}    converted={meta.get('converted')}")

    if not meta.get("complete", True):
        warn.append("metadata says complete=False — this recording was cut short")
    if isinstance(dur, (int, float)) and dur < 15:
        warn.append(f"only {dur:.1f} s long — short for a full demo, check it is not an aborted take")
    if "PENDING" in status.upper() or status.startswith("pending"):
        warn.append("no verdict recorded (pending) — it will be skipped by success filters")
    if "DB ROWS" in status:
        warn.append("duplicate DB rows for this path — statuses conflict")

    # ── cameras ──────────────────────────────────────────────────────────
    cams = sorted((ep / "cameras").glob("*.bag")) if (ep / "cameras").is_dir() else []
    print(f"\n  cameras      {len(cams)}")
    for c in cams:
        print(f"    {c.name:26s} {c.stat().st_size / 1e9:6.2f} GB")
    for name in meta.get("cameras", []):
        if not any(c.stem == name for c in cams):
            warn.append(f"metadata lists camera {name!r} but no {name}.bag on disk")

    npz = ep / "robot_data.npz"
    if not npz.exists():
        warn.append("no robot_data.npz — the episode has no telemetry at all")
        return warn
    d = np.load(npz, allow_pickle=True)

    # ── timing ───────────────────────────────────────────────────────────
    ts = np.asarray(d["timestamps"]).astype(np.int64)
    if len(ts) > 1:
        dt = np.diff(ts) / 1e6  # ms
        span = (ts[-1] - ts[0]) / 1e9
        print(f"\n  telemetry    {len(ts)} samples over {span:.2f} s   "
              f"median {np.median(dt):.2f} ms   max gap {dt.max():.1f} ms")
        if dt.max() > 100:
            warn.append(f"{dt.max():.0f} ms gap in robot telemetry — interpolation will be poor there")

    # ── per arm ──────────────────────────────────────────────────────────
    for arm in arms(d):
        pos = np.asarray(d[f"{arm}_joint_pos"], dtype=float)
        vel = np.asarray(d[f"{arm}_joint_vel"], dtype=float)
        eff = np.asarray(d[f"{arm}_joint_eff"], dtype=float)
        print(f"\n  {arm}")
        if not brief:
            print("    joint    min      max     range    max|vel|   max|eff|")
            for j in range(pos.shape[1]):
                print(f"      j{j}   {pos[:, j].min():+7.3f}  {pos[:, j].max():+7.3f}  "
                      f"{np.ptp(pos[:, j]):7.3f}   {np.abs(vel[:, j]).max():8.3f}  "
                      f"{np.abs(eff[:, j]).max():9.3f}")
        if np.ptp(pos, axis=0).max() < 0.05:
            warn.append(f"{arm}: the arm barely moved (max joint range < 0.05 rad)")

        # ── gripper ──────────────────────────────────────────────────────
        cmd_key, meas_key = f"{arm}_joint_cmd", f"{arm}_joint_pos_7d"
        if cmd_key not in d or meas_key not in d:
            continue
        cmd = np.asarray(d[cmd_key], dtype=float)[:, 6]
        meas = np.asarray(d[meas_key], dtype=float)[:, 6]
        geff = np.abs(np.asarray(d[f"{arm}_gripper_eff"], dtype=float).ravel())
        lag = meas - cmd  # > 0 means the command is closing past the measured jaws
        closed = cmd < 0.6
        print(f"    gripper  opening {meas.min():.3f}..{meas.max():.3f}   "
              f"closed-command frames {100 * closed.mean():.0f}%   |eff| p99 {np.percentile(geff, 99):.2f}")

        if closed.sum() < 10:
            print("             never commanded closed — no grasp in this episode")
            warn.append(f"{arm}: gripper never commanded closed, so nothing was grasped")
            continue
        p99 = np.percentile(lag[closed], 99)
        at = {k: (lag[closed] > 0.9 * v).mean() for k, v in CLAMPS.items()}
        match = min(CLAMPS, key=lambda k: abs(p99 - CLAMPS[k]))
        print(f"             lag p99 {p99:+.4f}  ->  clamp looks like {match}")
        if not brief:
            for k, v in CLAMPS.items():
                print(f"               {k:32s} ceiling {v:.4f}   {100 * at[k]:5.1f}% of closed frames pinned")
        if abs(p99 - CLAMPS[match]) > 0.25 * CLAMPS[match]:
            warn.append(f"{arm}: gripper lag p99 {p99:+.4f} matches no known clamp ceiling")
    return warn


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("episodes", nargs="+", type=Path, help="raw episode directories")
    ap.add_argument("--brief", action="store_true", help="skip the per-joint and per-clamp tables")
    args = ap.parse_args()

    missing = [e for e in args.episodes if not e.is_dir()]
    for e in missing:
        print(f"not a directory: {e}", file=sys.stderr)
    warned = 0
    for ep in args.episodes:
        if not ep.is_dir():
            continue
        warn = report(Path(ep), args.brief)
        if warn:
            warned += 1
            print("\n  WARNINGS")
            for w in warn:
                print(f"    - {w}")
    print()
    return 1 if (missing or warned) else 0


if __name__ == "__main__":
    raise SystemExit(main())
