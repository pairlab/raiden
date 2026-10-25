# Installation

## Requirements

- Python 3.11
- [uv](https://docs.astral.sh/uv/) package manager — install with `curl -LsSf https://astral.sh/uv/install.sh | sh`
- [ZED SDK](https://www.stereolabs.com/developers/) (if using ZED cameras)

## Install

Clone the repository with submodules:

```bash
git clone --recurse-submodules git@github.com:TRI-ML/raiden.git
cd raiden
```

If you already cloned without `--recurse-submodules`, initialize the submodule manually:

```bash
git submodule update --init
```

## Hardware SDKs

**ZED cameras** - after installing the ZED SDK, run the helper script to download
the matching Python wheel:

```bash
uv run python scripts/install_pyzed.py
```

This downloads `pyzed-*.whl` into `packages/` and updates `pyproject.toml` to
reference it.

**Intel RealSense** (not recommended) - no additional SDK installation is required
for most distributions; `pyrealsense2` is included in the base dependencies.

## Install `rd`

Install `rd` as a shell command, picking the extras that match your hardware:

```bash
uv tool install -e .                                                    # base install
uv tool install -e ".[zed]"                                             # + ZED cameras
uv tool install -e ".[realsense]"                                       # + RealSense cameras (not recommended)
rd --help
```

!!! note
    Re-run `uv tool install --reinstall -e ".[<extras>]"` whenever you add
    or change extras, or after pulling updates from the repository.

### Run from an installed checkout environment

If Raiden is installed in the checkout's `.venv`, invoke its executable
directly or activate that environment before using the bare `rd` command:

```bash
cd /path/to/raiden
.venv/bin/rd --help
source .venv/bin/activate
rd record_calibration_poses --help
```

From a parent workspace that contains the `raiden/` checkout, use:

```bash
./raiden/.venv/bin/rd record_calibration_poses --help
# Equivalent: explicitly select the installed project's environment.
uv run --project ./raiden --no-sync rd record_calibration_poses --help
```

`uv run rd` in the parent workspace does not discover a nested project.
The `--no-sync` form uses the existing environment without installing or
changing packages; it requires an already installed, working environment.
An environment created only for tests may have neither the `rd` entry point
nor all runtime dependencies. A `command not found` error means the shell
cannot find the executable; a Python traceback naming a missing module means
the selected environment still lacks a runtime dependency.

Help commands do not initialize the robot. Before running a recording
command, use the host that has access to the intended CAN interfaces,
cameras, Quest/ADB connection, and `~/.config/raiden/` profiles. Having a
checkout on another workstation does not make those devices available there.
