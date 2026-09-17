#!/usr/bin/env python3
"""Export converted task(s) to LeRobot datasets without the fzf picker.

``rd lerobot`` routes through :func:`select_processed_task`, which always opens fzf
and so cannot run unattended.  This calls :func:`export_task_to_lerobot` straight on
named task directories, which is what a scripted or background export needs.

    cd ~/robot/raiden
    uv run python scripts/export_lerobot.py data/processed/crossaint_oven_real
    uv run python scripts/export_lerobot.py data/processed/*_real --repo-id frankchang1000/3D_sim2real

Each task is written to ``<output-dir>/<task_name>``.  Tasks are exported one after
another; pass ``--jobs`` to overlap them, which mostly buys video-encode throughput.
"""
import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from raiden.lerobot_export import export_task_to_lerobot


def episodes(task_dir: Path) -> list:
    return sorted(
        ep for ep in task_dir.iterdir()
        if ep.is_dir() and (ep / "metadata.json").exists()
    )


def one(task_dir: str, output_dir: str, repo_id, vcodec: str,
        threads: int, overwrite: bool) -> tuple:
    task = Path(task_dir)
    try:
        root = export_task_to_lerobot(
            task,
            episodes(task),
            output_dir=Path(output_dir),
            repo_id=repo_id,
            vcodec=vcodec,
            image_writer_threads=threads,
            overwrite=overwrite,
        )
    except Exception as e:
        return task.name, None, f"{type(e).__name__}: {e}"
    return task.name, str(root), None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tasks", nargs="+", type=Path, help="processed task directories")
    ap.add_argument("--output-dir", default="data/lerobot")
    ap.add_argument("--repo-id", default=None,
                    help="repo id recorded in the dataset metadata "
                         "(default raiden/<task_name>)")
    ap.add_argument("--vcodec", default="libsvtav1")
    ap.add_argument("--image-writer-threads", type=int, default=8)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--jobs", type=int, default=1,
                    help="tasks to export concurrently (default 1)")
    args = ap.parse_args()

    tasks = []
    for t in args.tasks:
        if not t.is_dir():
            print(f"not a directory: {t}", file=sys.stderr)
            return 1
        n = len(episodes(t))
        if not n:
            print(f"no converted episodes in {t}", file=sys.stderr)
            return 1
        print(f"{t.name}: {n} episodes")
        tasks.append(t)

    bad = 0
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = [
            pool.submit(one, str(t), args.output_dir, args.repo_id, args.vcodec,
                        args.image_writer_threads, args.overwrite)
            for t in tasks
        ]
        for fut in as_completed(futures):
            name, root, err = fut.result()
            if err:
                print(f"{name}: FAILED {err}", flush=True)
                bad += 1
            else:
                print(f"{name}: exported -> {root}", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
