#!/usr/bin/env python
"""Delete recorded episodes cleanly: the raw folder, the converted copy if there is one, and the DB row.

The console's Delete removes only the DB row and leaves the files. This takes episode folders, shows what it
will remove, asks once, backs up demonstrations.json, then removes both.

    uv run python scripts/delete_episode.py data/raw/croissant_oven_real_newrig/0003 [more episode dirs] [--yes]

Deleting the newest episode frees its number for the next recording; deleting an older one leaves a gap
(conversion numbers the successful episodes in order, so gaps do not matter there).
"""

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

from raiden._config import DB_DIR
from raiden.db.database import get_db

REPO = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("episodes", nargs="+", help="episode dirs, e.g. data/raw/<task>/0003")
    ap.add_argument("--yes", action="store_true", help="do not ask")
    args = ap.parse_args()

    db = get_db()
    plan = []
    for e in args.episodes:
        p = Path(e).resolve()
        try:
            rel = p.relative_to(REPO)
        except ValueError:
            sys.exit(f"{p} is not inside {REPO}")
        if rel.parts[:2] != ("data", "raw") or len(rel.parts) != 4 or not rel.parts[3].isdigit():
            sys.exit(f"{rel}: expected data/raw/<task>/<NNNN>")
        demo = db.get_demonstration_by_raw_path(str(rel))
        conv = REPO / demo["converted_data_path"] if demo and demo.get("converted_data_path") else None
        plan.append((rel, p, demo, conv))
        print(f"{rel}: {'folder' if p.exists() else 'no folder'}, "
              f"{'DB row #%s (%s)' % (demo['id'], demo['status']) if demo else 'no DB row'}"
              f"{', converted copy ' + str(conv.relative_to(REPO)) if conv and conv.exists() else ''}")
    if not args.yes and input("Delete all of this? type yes: ").strip().lower() != "yes":
        print("nothing deleted")
        return
    backup = Path(DB_DIR) / f"demonstrations.json.bak-{datetime.now():%Y%m%d-%H%M%S}-pre-delete"
    shutil.copy2(Path(DB_DIR) / "demonstrations.json", backup)
    for rel, p, demo, conv in plan:
        if demo:
            db.delete_demonstration(demo["id"])
        if conv and conv.exists():
            shutil.rmtree(conv)
        if p.exists():
            shutil.rmtree(p)
        print(f"  deleted {rel}")
    print(f"  DB backup: {backup}")


if __name__ == "__main__":
    main()
