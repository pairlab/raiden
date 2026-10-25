"""Find saved native RealSense files without importing a camera SDK."""

from pathlib import Path

REALSENSE_RECORDING_FORMATS = ("bag", "db3")
REALSENSE_RECORDING_SUFFIXES = tuple(f".{format}" for format in REALSENSE_RECORDING_FORMATS)


def list_realsense_recordings(directory: Path) -> list[Path]:
    """List native recordings by camera name; reject ambiguous duplicate formats."""
    directory = Path(directory)
    if not directory.is_dir():
        return []
    recordings = sorted(path for path in directory.iterdir()
                        if path.is_file() and path.suffix in REALSENSE_RECORDING_SUFFIXES)
    by_name = {}
    for path in recordings:
        if path.stem in by_name:
            raise ValueError(
                f"Duplicate RealSense recordings for camera {path.stem!r}: "
                f"{by_name[path.stem].name}, {path.name}"
            )
        by_name[path.stem] = path
    return recordings


def find_realsense_recording(directory: Path, name: str) -> Path | None:
    """Find the saved file for one camera, returning None when absent."""
    return next((path for path in list_realsense_recordings(directory) if path.stem == name), None)
