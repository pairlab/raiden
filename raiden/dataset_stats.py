"""Accumulated stats of one recorded dataset (a task), over every session.

``rd record --ui`` and ``scripts/qc_episodes.py`` rebuild ``data/raw/<task>/README.md`` whenever an episode is checked,
relabelled or deleted, from the QC results (``data/qc/<task>/qc_results.json``), the DB labels and the session log
(``data/qc/<task>/sessions.jsonl``, one line per ``rd record --ui`` start). Everything above the Notes section is
overwritten; the notes are kept.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np

from raiden import qc

SESSION_GAP_S = 30 * 60  # this long without a recording also starts a new session (sessions recorded without --ui)
NOTES_BEGIN = "<!-- notes: kept when this file is rebuilt -->"
NOTES_END = "<!-- end of notes -->"
LABELS = ("success", "failure", "unlabelled")
# (text in a QC reason, the check it belongs to), first match wins
FLAG_KINDS = (
    ("oven moved", "oven moved during the episode"),
    ("duration outlier", "duration far from the median"), ("duration", "duration out of range"),
    ("pause", "pause"), ("gripper closes", "gripper closes not 1 (regrasp, handle grasp or no grasp)"),
    ("croissant centre", "croissant centre outside the blue box"), ("croissant (", "croissant may cross the white box"),
    ("croissant not found", "croissant not found"), ("oven x", "oven x off the sim's"),
    ("oven y", "oven y outside the sim's range"), ("oven (", "oven crosses the white box"),
    ("oven knobs", "oven not found"), ("wrist:", "wrist fingertips off (mount)"),
    ("wrist brightness", "wrist camera dark"), ("scene brightness", "scene brightness"),
    ("color_controls", "colour lock missing or different"), ("unreadable", "camera bag unreadable"),
    ("QC error", "QC error"),
)


def label_of(r: Dict) -> str:
    return r["label"] if r.get("label") in ("success", "failure") else "unlabelled"


def refresh(results: Dict[str, Dict], task: str, demos: List[Dict], teachers: Dict, raw_dir: Path) -> None:
    """Update each result's label, save time and teacher from the DB rows; drop episodes whose folder is gone."""
    rows = {d.get("raw_data_path"): d for d in demos}  # a later row for the same path wins
    for name in list(results):
        ep = Path(raw_dir) / name
        if not ep.is_dir():
            del results[name]
            continue
        r, d = results[name], rows.get(f"data/raw/{task}/{name}")
        if d is not None:
            r.update(label=d.get("status"), saved_at=d.get("created_at"), teacher=teachers.get(d.get("teacher_id"), "?"))
            continue
        r["label"] = "no DB row"
        if not r.get("saved_at") and (ep / "metadata.json").exists():
            r["saved_at"] = json.load(open(ep / "metadata.json")).get("timestamp")


def log_session(qc_dir: Path, teacher: str) -> datetime:
    """Append a session start to ``sessions.jsonl``; returns the start time."""
    now = datetime.now().replace(microsecond=0)
    qc_dir.mkdir(parents=True, exist_ok=True)
    with open(qc_dir / "sessions.jsonl", "a") as f:
        f.write(json.dumps(dict(started=now.isoformat(), teacher=teacher)) + "\n")
    return now


def logged_starts(qc_dir: Path) -> List[datetime]:
    f = qc_dir / "sessions.jsonl"
    out = []
    for line in f.read_text().splitlines() if f.exists() else []:
        try:
            out.append(datetime.fromisoformat(json.loads(line)["started"]))
        except (ValueError, KeyError, TypeError):
            pass
    return sorted(out)


def sessions(results: Dict[str, Dict], starts: List[datetime]) -> List[List[Dict]]:
    """Episodes grouped by session, oldest first: a new session at each logged start, and after SESSION_GAP_S
    without a recording."""
    eps = sorted((r for r in results.values() if r.get("saved_at")), key=lambda r: r["saved_at"])
    groups: List[List[Dict]] = []
    prev_end: Optional[datetime] = None
    k = 0
    for r in eps:
        end = datetime.fromisoformat(r["saved_at"])
        start = end - timedelta(seconds=r.get("duration_s") or 0.0)
        new = prev_end is None or (start - prev_end).total_seconds() > SESSION_GAP_S
        while k < len(starts) and starts[k] <= start:
            new = new or starts[k] > prev_end
            k += 1
        if new:
            groups.append([])
        groups[-1].append(r)
        prev_end = end
    return groups


def _counts(rs: Iterable[Dict]) -> Dict[str, int]:
    c = dict.fromkeys(LABELS, 0)
    for r in rs:
        c[label_of(r)] += 1
    return c


def summary(results: Dict[str, Dict], starts: List[datetime], since: Optional[datetime] = None) -> Dict:
    """Headline numbers for the recording page; ``session`` counts the episodes saved since ``since``."""
    rs = list(results.values())
    succ = [r for r in rs if label_of(r) == "success"]
    durs = [r["duration_s"] for r in succ if r.get("duration_s")]
    checked = [r for r in succ if not r.get("qc_pending")]
    out = dict(_counts(rs), total=len(rs), success_min=round(sum(durs) / 60, 1),
               median_s=round(float(np.median(durs)), 1) if durs else None, sessions=len(sessions(results, starts)),
               first=min((r["saved_at"] for r in rs if r.get("saved_at")), default=None),
               succ_pass=sum(not r.get("reasons") for r in checked), succ_checked=len(checked))
    if since is not None:
        out["session"] = _counts(r for r in rs if r.get("saved_at") and datetime.fromisoformat(r["saved_at"]) >= since)
    return out


def flag_kind(reason: str) -> str:
    return next((kind for key, kind in FLAG_KINDS if key in reason), reason)


def _when(iso: Optional[str]) -> str:
    return datetime.fromisoformat(iso).strftime("%m-%d %H:%M") if iso else "?"


def _xy(p) -> str:
    return f"{p[0]:.3f}, {p[1]:+.3f}" if p else "–"


def read_notes(path: Path) -> str:
    if not path.exists():
        return ""
    old = path.read_text()
    a, b = old.find(NOTES_BEGIN), old.find(NOTES_END)
    return old[a + len(NOTES_BEGIN):b].strip("\n") if 0 <= a < b else ""


def markdown(task: str, instruction: str, results: Dict[str, Dict], geo: qc.Geometry, st: qc.QCSettings,
             color_controls: Dict[str, Optional[Dict]], starts: List[datetime], notes: str = "") -> str:
    rs = [results[k] for k in sorted(results)]
    by = {lab: [r for r in rs if label_of(r) == lab] for lab in LABELS}
    groups = sessions(results, starts)
    minutes = lambda xs: sum(r.get("duration_s") or 0.0 for r in xs) / 60
    out = [f"# {task}", "",
           f"\"{instruction}\" · episodes in `data/raw/{task}/` · rebuilt {datetime.now():%Y-%m-%d %H:%M} by "
           "`rd record --ui` or `scripts/qc_episodes.py` after every checked, relabelled or deleted episode. "
           "Everything above Notes is overwritten.", "",
           "## Totals", "", "| label | episodes | minutes |", "|---|---:|---:|"]
    out += [f"| {lab} | {len(by[lab])} | {minutes(by[lab]):.1f} |" for lab in LABELS]
    out += [f"| all | {len(rs)} | {minutes(rs):.1f} |", ""]
    labelled = len(by["success"]) + len(by["failure"])
    if labelled:
        out.append(f"- Success rate: {100 * len(by['success']) / labelled:.0f}% of the {labelled} labelled episodes.")
    durs = [r["duration_s"] for r in by["success"] if r.get("duration_s")]
    if durs:
        out.append(f"- Successful demos: median {np.median(durs):.0f} s, {min(durs):.0f}-{max(durs):.0f} s.")
    checked = [r for r in by["success"] if not r.get("qc_pending")]
    if checked:
        out.append(f"- QC: {sum(not r.get('reasons') for r in checked)} of {len(checked)} successes pass every "
                   "check (flags below).")
    if by["unlabelled"]:
        out.append(f"- {len(by['unlabelled'])} unlabelled ({', '.join(r['episode'] for r in by['unlabelled'])}): "
                   "label them in the recording page's episode list (✓ / ✗).")
    if groups:
        teachers = sorted({r.get("teacher") or "?" for r in rs})
        out.append(f"- Recorded {_when(groups[0][0]['saved_at'])} to {_when(groups[-1][-1]['saved_at'])} in "
                   f"{len(groups)} session(s); teacher(s): {', '.join(teachers)}.")

    G, outside = qc.coverage(results, geo)
    n_found = sum(1 for r in by["success"] if r.get("croissant"))
    bx0, bx1, by0, by1 = geo.blue
    xs, ys = np.linspace(bx0, bx1, qc.GRID_X + 1), np.linspace(by1, by0, qc.GRID_Y + 1)
    out += ["", "## Coverage of the successful demos", "",
            f"Croissant start (centre of mass, first scene frame) over the blue box. Top row = far from the robot, "
            f"left = the robot's left (+y). {len(by['success'])} successes: {n_found - outside} in the box, "
            f"{outside} outside it, {len(by['success']) - n_found} with no croissant found.", "",
            "| x (m) \\ y (m) | " + " | ".join(f"{(ys[j] + ys[j + 1]) / 2:+.2f}" for j in range(qc.GRID_Y)) + " |",
            "|---" * (qc.GRID_Y + 1) + "|"]
    for i in reversed(range(qc.GRID_X)):
        out.append(f"| {xs[i]:.2f}-{xs[i + 1]:.2f} | " + " | ".join(str(n) if n else "·" for n in G[i]) + " |")
    out += ["", f"{int((G == 0).sum())} of {G.size} cells have no successful demo yet.", ""]
    O, o_out = qc.oven_coverage(results, geo)
    oy = np.linspace(geo.oven_region[3], geo.oven_region[2], qc.OVEN_GRID + 1)
    out += [f"Oven origin y over the sim's range (left = the robot's left); {o_out} outside it. The sim's oven x is "
            f"{sum(geo.oven_region[:2]) / 2:.3f} (QC allows ±{qc.OVEN_X_TOL * 1000:.0f} mm).", "",
            "| " + " | ".join(f"{(oy[j] + oy[j + 1]) / 2:+.3f}" for j in range(qc.OVEN_GRID)) + " |",
            "|---" * qc.OVEN_GRID + "|", "| " + " | ".join(str(n) if n else "·" for n in O) + " |"]

    out += ["", "## Sessions", "",
            f"A new session starts at each `rd record --ui` start, or after {SESSION_GAP_S // 60} min without a recording.",
            "", "| # | first episode | teacher | episodes | success | failure | unlabelled | success min |",
            "|---:|---|---|---:|---:|---:|---:|---:|"]
    for k, g in enumerate(groups, 1):
        c = _counts(g)
        out.append(f"| {k} | {_when(g[0]['saved_at'])} | {', '.join(sorted({r.get('teacher') or '?' for r in g}))} "
                   f"| {len(g)} | {c['success']} | {c['failure']} | {c['unlabelled']} "
                   f"| {minutes([r for r in g if label_of(r) == 'success']):.1f} |")

    kinds: Dict[str, List[int]] = {}
    for r in rs:
        for kind in {flag_kind(x) for x in r.get("reasons", [])}:
            kinds.setdefault(kind, [0, 0])
            kinds[kind][0] += 1
            kinds[kind][1] += label_of(r) == "success"
    out += ["", "## QC flags", "", "Episodes with at least one flag of each kind (details in the episode table).", ""]
    if kinds:
        out += ["| check | episodes | successes |", "|---|---:|---:|"]
        out += [f"| {kind} | {n} | {s} |" for kind, (n, s) in sorted(kinds.items(), key=lambda kv: -kv[1][0])]
    else:
        out.append("None.")

    out += ["", "## Episodes", "", "Oven moved: from where its knobs are first seen to where they are last seen (mm).", "",
            "| # | saved | teacher | length | label | croissant x, y | oven x, y | oven moved | QC |",
            "|---|---|---|---:|---|---|---|---|---|"]
    for r in rs:
        qc_text = "QC running" if r.get("qc_pending") else "; ".join(r.get("reasons", [])).replace("|", "/") or "pass"
        length = f"{r['duration_s']:.0f} s" if r.get("duration_s") else "?"
        m = r.get("oven_moved")
        moved = f"{m[0] * 1000:+.0f}, {m[1] * 1000:+.0f}" if m else "–"
        out.append(f"| {r['episode']} | {_when(r.get('saved_at'))} | {r.get('teacher') or '?'} | {length} "
                   f"| {r.get('label')} | {_xy(r.get('croissant'))} | {_xy(r.get('oven'))} | {moved} | {qc_text} |")

    wx0, wx1, wy0, wy1 = geo.white
    ox0, ox1, oy0, oy1 = geo.oven_region
    locks = "; ".join(f"{name} " + (f"{c['white_balance']:.0f} K, exposure {c['exposure']:.0f}, gain {c['gain']:.0f}"
                                    if c else "auto") for name, c in color_controls.items())
    out += ["", "## Setup", "",
            "- Placement limits, from vla-benchmark (`rig.json`, `layout.json` and the task JSON's regions):",
            f"  - croissant centre of mass (blue box): x {bx0:.3f}-{bx1:.3f}, y {by0:+.3f}..{by1:+.3f}",
            f"  - whole croissant and oven (white box): x {wx0:.3f}-{wx1:.3f}, y {wy0:+.3f}..{wy1:+.3f}",
            f"  - oven origin: x {(ox0 + ox1) / 2:.3f} ± {qc.OVEN_X_TOL:.3f}, y {oy0:+.3f}..{oy1:+.3f}",
            f"- Colour lock in camera.json now: {locks}. Episodes recorded with other values are flagged.",
            f"- Wrist fingertip reference rows {st.ref_tips[0]} / {st.ref_tips[1]} (arm at home), ±{st.max_tip_px} px.",
            f"- Gripper: exactly {st.closes} close per episode (the croissant grasp; the door is hooked open, gripper open).",
            f"- Oven: moves at most {st.max_oven_move * 1000:.0f} mm during an episode (first to last seen knobs).",
            "", "## Notes", "", NOTES_BEGIN] + ([notes] if notes else []) + [NOTES_END, ""]
    return "\n".join(out)


def write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)
