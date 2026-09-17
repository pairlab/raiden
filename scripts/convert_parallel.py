#!/usr/bin/env python3
"""Convert a raw task directory to UnifiedDataset format across several processes.

``rd convert`` walks the recordings one at a time, which leaves 15 of 16 cores idle
on a RealSense capture — the bag decode is pure CPU, and episodes never touch each
other's output.  This runs :func:`convert_recording` in a process pool and then
writes the dataset-level files (``split_all.json``, ``metadata_shared.json``,
``calibration_results.json``) and the DB updates itself, exactly as
:func:`convert_task` would.

    cd ~/robot/raiden
    uv run python scripts/convert_parallel.py data/raw/crossaint_oven_real
    uv run python scripts/convert_parallel.py data/raw/*_real --workers 12

Episodes are numbered by position in the sorted list of successful recordings, not
by completion order, so a run that is interrupted and restarted maps each raw
directory to the same episode number.  Already-finished episodes are skipped, which
makes the script safe to re-run to pick up where a previous pass stopped.
"""
import argparse
import json
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from raiden.converter import ConversionError, convert_recording


def successful(task_path: Path) -> list:
    """The recordings `rd convert` would take, in the order it would number them."""
    recordings = sorted(
        d for d in task_path.iterdir() if d.is_dir() and (d / "cameras").exists()
    )
    try:
        from raiden.db.database import get_db

        db = get_db()
    except Exception:
        db = None

    keep = []
    for rec in recordings:
        status = "unknown"
        if db is not None:
            try:
                demo = db.get_demonstration_by_raw_path(str(rec))
                if demo is not None:
                    status = demo.get("status", "pending")
            except Exception:
                pass
        # No DB row means the recording predates the DB; convert it rather than
        # silently dropping it, which is what convert_task does.
        if status in ("success", "unknown"):
            keep.append(rec)
        else:
            print(f"  skipping {rec.name} (status={status})")
    return keep


def one(rec_dir: str, ep_dir: str) -> tuple:
    """Worker: convert a single recording. Returns (ep_name, frames, error)."""
    name = Path(ep_dir).name
    try:
        counts = convert_recording(rec_dir, episode_dir=ep_dir)
    except ConversionError as e:
        shutil.rmtree(ep_dir, ignore_errors=True)
        return name, 0, str(e)
    except Exception as e:  # a bad bag should not take the whole pool down
        shutil.rmtree(ep_dir, ignore_errors=True)
        return name, 0, f"{type(e).__name__}: {e}"
    return name, (max(counts.values()) if counts else 0), None


def convert(task_path: Path, out_base: Path, workers: int) -> int:
    recs = successful(task_path)
    if not recs:
        print(f"no successful recordings in {task_path}")
        return 0
    out_base.mkdir(parents=True, exist_ok=True)

    # Position in this list *is* the episode number, so a resumed run reproduces
    # the mapping a single sequential pass would have produced.
    planned = [(rec, out_base / f"{i:04d}") for i, rec in enumerate(recs)]
    todo = [(r, e) for r, e in planned if not (e / "metadata.json").exists()]
    done = len(planned) - len(todo)
    # Drop anything half-written by an interrupted run before re-converting it.
    for _, ep in todo:
        shutil.rmtree(ep, ignore_errors=True)

    print(f"{task_path.name}: {len(planned)} episodes, {done} already done, "
          f"{len(todo)} to convert on {workers} workers")

    frames, failed = {}, []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(one, str(rec), str(ep)): (rec, ep)
            for rec, ep in todo
        }
        for n, fut in enumerate(as_completed(futures), 1):
            rec, ep = futures[fut]
            name, count, err = fut.result()
            if err:
                print(f"  [{n}/{len(todo)}] {rec.name} -> {name}  FAILED: {err}",
                      flush=True)
                failed.append(rec.name)
                continue
            frames[name] = count
            print(f"  [{n}/{len(todo)}] {rec.name} -> {name}  {count} frames",
                  flush=True)
            mark_converted(rec, ep)

    # Episodes finished by an earlier pass still need to land in split_all.json.
    for _, ep in planned:
        meta = ep / "metadata.json"
        if ep.name not in frames and meta.exists():
            frames[ep.name] = json.loads(meta.read_text()).get("num_frames", 0)

    write_dataset_files(task_path, out_base, frames)
    if failed:
        print(f"\n{len(failed)} recording(s) failed: {', '.join(failed)}")
    return len(failed)


def mark_converted(rec: Path, ep: Path) -> None:
    """Flip the DB row for *rec*. Done in the parent so writes stay serialised."""
    try:
        from raiden.db.database import get_db

        db = get_db()
        demo = db.get_demonstration_by_raw_path(str(rec))
        if demo is not None:
            db.update_demonstration(demo["id"], converted=True,
                                    converted_data_path=str(ep))
    except Exception:
        pass


def write_dataset_files(task_path: Path, out_base: Path, frames: dict) -> None:
    if not frames:
        return
    ordered = {k: frames[k] for k in sorted(frames)}
    split = {
        "filters": {},
        "size": {"n_seqs": len(ordered), "n_samples": len(ordered),
                 "n_frames": sum(ordered.values())},
        "files": ordered,
    }
    (out_base / "split_all.json").write_text(json.dumps(split, indent=2))
    print(f"\nsplit_all.json ({len(ordered)} episodes, {sum(ordered.values())} frames)")

    first_meta = out_base / next(iter(ordered)) / "metadata.json"
    if first_meta.exists():
        shutil.copy(first_meta, out_base / "metadata_shared.json")
        print("metadata_shared.json")
    for rec in sorted(task_path.iterdir()):
        calib = rec / "calibration_results.json"
        if calib.is_dir() or not calib.exists():
            continue
        shutil.copy(calib, out_base / "calibration_results.json")
        print("calibration_results.json")
        break
    print(f"ready: {out_base}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tasks", nargs="+", type=Path, help="raw task directories")
    ap.add_argument("--workers", type=int, default=10,
                    help="parallel conversions (default 10)")
    ap.add_argument("--data-dir", default="data",
                    help="root data directory; output goes to <data-dir>/processed")
    args = ap.parse_args()

    bad = 0
    for task in args.tasks:
        if not task.is_dir():
            print(f"not a directory: {task}", file=sys.stderr)
            bad += 1
            continue
        out = Path(args.data_dir) / "processed" / task.name
        print(f"\n{'=' * 66}\n  {task.name}\n{'=' * 66}")
        bad += convert(task, out, args.workers)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
