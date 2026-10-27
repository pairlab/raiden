"""Package selected Isaac recordings and their recorded scene dependencies.

The bundle preserves recording bytes, uses relative asset paths, and contains no
application source or installed simulator binaries. Publication is a separate step.
"""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
import tarfile
import tempfile
from pathlib import Path


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def read_episode_list(path: Path) -> list[Path]:
    """Read absolute paths or paths relative to a selection file."""
    paths = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        value = Path(line).expanduser()
        paths.append((value if value.is_absolute() else path.parent / value).resolve())
    if not paths or len(paths) != len(set(paths)):
        raise ValueError("episode selection must be nonempty and contain no duplicates")
    return paths


def _runtime(metadata: dict) -> dict:
    keys = (
        "backend",
        "physics_backend",
        "state_version",
        "physics_dt",
        "physics_substeps",
        "solver_dt",
        "versions",
        "arms",
        "arm_joints",
        "action_dim",
        "controller",
        "finger_coupling",
        "solver_capacity",
        "grasp_contact",
        "calibration_sha256",
        "asset_input_sha256",
        "implementation_sha256",
    )
    result = {key: metadata[key] for key in keys if key in metadata}
    result["scene"] = {
        key: metadata["scene"][key] for key in ("descriptor", "sha256", "teleop")
    }
    return result


def _relative(path: Path, root: Path) -> Path:
    if path.is_symlink():
        raise ValueError(f"symlink inputs are unsupported: {path}")
    try:
        return path.resolve().relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"input is outside declared root {root}: {path}") from error


def _archive(target: Path, entries: list[tuple[Path, str]]) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(target, "w") as archive:
        for source, name in entries:
            info = archive.gettarinfo(str(source), arcname=name)
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mtime = 0
            info.mode = 0o644
            with source.open("rb") as stream:
                archive.addfile(info, stream)


def _usd_paths(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"usd_path", "housing_usd_path"}:
                yield Path(item)
            else:
                yield from _usd_paths(item)
    elif isinstance(value, list):
        for item in value:
            yield from _usd_paths(item)


def verify_scene_assets(bundle: Path) -> dict:
    """Check extracted USD dependencies using the pinned Isaac/pxr interpreter."""
    from pxr import UsdUtils

    environment = bundle / "sim/environment"
    entrypoints = json.loads((environment / "entrypoints.json").read_text())
    with tempfile.TemporaryDirectory(prefix="raiden-scene-check-") as directory:
        root = Path(directory)
        assets = root / "assets"
        assets.mkdir()
        with tarfile.open(environment / entrypoints["asset_archive"]) as archive:
            archive.extractall(assets, filter="data")
        descriptor = root / entrypoints["scene"]
        checked = set()
        for relative in _usd_paths(json.loads(descriptor.read_text())):
            if relative.is_absolute() or ":" in str(relative):
                raise ValueError(f"scene asset path must be relative: {relative}")
            source = descriptor.parent / relative
            _relative(source, assets)
            if not source.is_file():
                raise ValueError(f"missing scene asset: {source}")
            layers, payloads, unresolved = UsdUtils.ComputeAllDependencies(str(source))
            if unresolved:
                raise ValueError(f"unresolved scene dependencies: {unresolved}")
            dependencies = [Path(layer.realPath) for layer in layers if layer.realPath]
            dependencies.extend(Path(str(path)) for path in payloads)
            for dependency in dependencies:
                checked.add(_relative(dependency, assets).as_posix())
                if not dependency.is_file():
                    raise ValueError(f"missing scene dependency: {dependency}")
        return {"status": "passed", "resolved_dependency_files": sorted(checked)}


def package_sim_episodes(
    episodes: list[Path],
    output: Path,
    *,
    asset_root: Path,
    attachments: dict[str, Path] | None = None,
    verify_assets: bool = False,
) -> dict:
    """Build a new, immutable local bundle; never connects to a remote service.

    A release has one task and one recorded runtime profile. Raw episodes remain
    unchanged, including timestamps and original provenance paths. Assets and the
    scene descriptor retain their relative layout under ``sim/environment/assets``.
    Optional attachments are explicit files, never recursively copied directories.
    """
    if not episodes or len(episodes) != len(set(episodes)):
        raise ValueError("expected a nonempty, unique episode selection")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite an existing bundle: {output}")
    asset_root = asset_root.resolve()
    inputs, common, task, identifiers = [], None, None, set()
    data_suffixes = {
        ".json",
        ".jsonl",
        ".npz",
        ".npy",
        ".png",
        ".jpg",
        ".jpeg",
        ".mp4",
        ".csv",
        ".txt",
        ".md",
    }
    for episode in episodes:
        episode = episode.resolve()
        label = json.loads((episode / "metadata.json").read_text())
        metadata = json.loads((episode / "sim_metadata.json").read_text())
        if not label.get("complete") or metadata.get("backend") != "isaac":
            raise ValueError(f"episode must be a completed Isaac recording: {episode}")
        if episode.name in identifiers:
            raise ValueError(f"duplicate episode identifier: {episode.name}")
        identifiers.add(episode.name)
        profile = _runtime(metadata)
        if common is None:
            common, task = profile, label["task_name"]
        elif profile != common or label["task_name"] != task:
            raise ValueError(
                "each release must contain one task and one recorded runtime profile"
            )
        files = sorted(path for path in episode.rglob("*") if path.is_file())
        for path in files:
            relative = _relative(path, episode)
            if (
                any(part.startswith(".") for part in relative.parts)
                or path.suffix.lower() not in data_suffixes
            ):
                raise ValueError(
                    f"unexpected recording file; explicit review required: {path}"
                )
        inputs.append((episode, label, files))

    sources, asset_hashes = {}, {}
    for original, expected in common["asset_input_sha256"].items():
        source = Path(original)
        relative = _relative(source, asset_root)
        if sha256(source) != expected:
            raise ValueError(f"asset differs from recorded bytes: {source}")
        sources[relative] = source
        asset_hashes[source] = expected
    descriptor = Path(common["scene"]["descriptor"])
    descriptor_relative = _relative(descriptor, asset_root)
    if sha256(descriptor) != common["scene"]["sha256"]:
        raise ValueError("scene descriptor differs from the recording")
    sources[descriptor_relative] = descriptor
    asset_hashes[descriptor] = common["scene"]["sha256"]
    for relative in _usd_paths(json.loads(descriptor.read_text())):
        if relative.is_absolute() or ":" in str(relative):
            raise ValueError(f"scene asset path must be relative: {relative}")
        _relative(descriptor.parent / relative, asset_root)

    attachments = attachments or {}
    for name, source in attachments.items():
        relative = Path(name)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or any(p.startswith(".") for p in relative.parts)
        ):
            raise ValueError(f"unsafe attachment destination: {name}")
        if (
            source.is_symlink()
            or not source.is_file()
            or source.name.startswith(".")
            or source.suffix.lower()
            not in {".md", ".json", ".txt", ".toml", ".lock", ".csv", ".yaml", ".yml"}
        ):
            raise ValueError(
                f"attachments must be explicit data/configuration files: {source}"
            )
        if not relative.parts or relative.parts[0] not in {"metadata", "sim"}:
            raise ValueError("attachments must live below metadata/ or sim/")

    output.mkdir(parents=True)
    environment = output / "sim/environment"
    asset_map = {}
    asset_entries = []
    for relative, source in sorted(sources.items()):
        asset_entries.append((source, relative.as_posix()))
        asset_map[str(source)] = f"assets/{relative.as_posix()}"
    _archive(environment / "scene_assets.tar", asset_entries)
    for source, expected in asset_hashes.items():
        if sha256(source) != expected:
            raise RuntimeError(f"scene asset changed while packaging: {source}")
    _json(environment / "asset_paths.json", asset_map)
    _json(environment / "recorded_runtime.json", common)
    task_config = copy.deepcopy(common["scene"]["teleop"])
    contact = task_config.get("contact_profile")
    if isinstance(contact, dict):
        _json(environment / "config/contact.json", contact["content"])
        task_config["contact_profile"] = "contact.json"
    _json(environment / "config/task.json", task_config)
    _json(
        environment / "entrypoints.json",
        {
            "scene": "assets/" + descriptor_relative.as_posix(),
            "task_config": "config/task.json",
            "asset_archive": "scene_assets.tar",
            "extract_assets_into": "assets/",
        },
    )
    if verify_assets:
        _json(environment / "verification.json", verify_scene_assets(output))

    inventory = []
    for episode, label, files in inputs:
        archive_path = output / "sim/teleop/train" / (episode.name + ".tar")
        records = [
            {
                "path": path.relative_to(episode).as_posix(),
                "size": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in files
        ]
        _archive(
            archive_path,
            [
                (
                    path,
                    f"{episode.parent.name}/{episode.name}/{path.relative_to(episode).as_posix()}",
                )
                for path in files
            ],
        )
        # Detect a writer changing a recording while it is packaged.
        for path, record in zip(files, records):
            if (
                path.stat().st_size != record["size"]
                or sha256(path) != record["sha256"]
            ):
                raise RuntimeError(f"recording changed while packaging: {path}")
        if sorted(path for path in episode.rglob("*") if path.is_file()) != files:
            raise RuntimeError(f"recording files changed while packaging: {episode}")
        inventory.append(
            {
                "episode_id": episode.name,
                "task": task,
                "duration_s": label["duration_s"],
                "archive": archive_path.relative_to(output).as_posix(),
                "files": records,
            }
        )

    _json(
        output / "metadata/sim_teleop/episodes.json",
        {"task": task, "episode_count": len(inventory), "episodes": inventory},
    )
    for name, source in attachments.items():
        target = output / name
        if target.exists():
            raise FileExistsError(
                f"attachment would overwrite a generated file: {name}"
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    payloads = sorted(path for path in output.rglob("*") if path.is_file())
    manifest = {
        "schema": "raiden-sim-bundle/1",
        "task": task,
        "episode_count": len(inventory),
        "original_episode_ids": [item["episode_id"] for item in inventory],
        "files": [
            {
                "path": path.relative_to(output).as_posix(),
                "size": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in payloads
        ],
    }
    _json(output / "manifest.json", manifest)
    return manifest
