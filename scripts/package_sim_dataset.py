#!/usr/bin/env python3
"""Build a local simulation release from an explicit episode selection."""

import argparse
from pathlib import Path

from raiden.sim.bundle import package_sim_episodes, read_episode_list


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes-file", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--verify-assets",
        action="store_true",
        help="Verify extracted USD dependencies; run with the pinned Isaac/pxr interpreter",
    )
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        metavar="DESTINATION=SOURCE",
        help="Attach an explicit config/document below metadata/ or sim/; repeatable",
    )
    args = parser.parse_args()
    attachments = {}
    for item in args.include:
        destination, separator, source = item.partition("=")
        if not separator or destination in attachments:
            parser.error("--include requires a unique DESTINATION=SOURCE")
        attachments[destination] = Path(source).expanduser().resolve()
    manifest = package_sim_episodes(
        read_episode_list(args.episodes_file),
        args.out,
        asset_root=args.asset_root,
        attachments=attachments,
        verify_assets=args.verify_assets,
    )
    print(
        f"Packaged {manifest['episode_count']} episodes into {args.out}; nothing uploaded."
    )


if __name__ == "__main__":
    main()
