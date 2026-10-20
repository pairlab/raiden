#!/usr/bin/env python3
"""Replay portable held commands in Newton without cameras, RPC or hardware.

Run with mesa_isaac/.venv/bin/python from its project directory. Uses the same
validated schedule, initial state, trace writer and comparisons as parity_replay.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

# The Isaac Python3.12 environment is separate from Raiden Python3.11 installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from parity_replay import replay

from raiden.sim.command_replay import load_bundle


class DirectIsaac:
    """Small in-process subset of simulator RPC, with no renderer callbacks."""

    def __init__(self, adapter, contact_audit=None):
        self.adapter = adapter
        self.cmd = np.r_[np.zeros(6), 1.0]
        self.paused = True
        self.ticks = 0
        self.contact_audit = contact_audit
        self.drop_rows = []
        self.drop_active = False
        self.replay_start = None

    def snapshot(self):
        state, obs, _ = self.adapter.snapshot()
        state.update(
            observations=obs,
            cmd=self.cmd.copy(),
            requested_cmd=self.cmd.copy(),
            kp=np.r_[self.adapter.kp[:6], 20.0],
            kd=np.r_[self.adapter.kd[:6], 0.5],
        )
        return state

    def call(self, name, **kwargs):
        if name == "get_metadata":
            return self.adapter.metadata
        if name == "is_paused":
            return self.paused
        if name == "set_paused":
            self.paused = bool(kwargs["paused"])
            return self.paused
        if name == "get_state":
            return self.snapshot()
        if name == "reset":
            self.adapter.reset()
            self.cmd = np.r_[np.zeros(6), 1.0]
            return self.snapshot()
        if name == "reset_to":
            state = kwargs["state"]
            self.adapter.reset_to(state)
            self.cmd = np.asarray(state["cmd"], dtype=float).copy()
            self.adapter.update_gains(
                np.asarray(state.get("kp", self.adapter.kp))[:6],
                np.asarray(state.get("kd", self.adapter.kd))[:6],
            )
            self.drop_active = False
            self.replay_start = float(state["sim_time"])
            return self.snapshot()
        if name == "step":
            if not self.paused or kwargs.get("n") != 1:
                raise ValueError("direct replay only accepts one paused physics tick")
            cmd = np.asarray(kwargs["cmd"], dtype=float)
            if cmd.shape != (7,) or not np.isfinite(cmd).all():
                raise ValueError("invalid native target")
            opening = cmd[6] - self.cmd[6] > 0.05
            self.cmd = cmd.copy()
            self.adapter.tick(self.cmd)
            self.ticks += 1
            if self.ticks % 500 == 0:
                self.adapter.check()
            state = self.snapshot()
            if self.contact_audit is not None and (self.drop_active or opening):
                self.drop_active = True
                contacts = self.adapter.contacts(robot_only=False)
                pairs = [
                    (a, b, distance, force)
                    for a, b, distance, force in contacts
                    if "/croissant/" in a or "/croissant/" in b
                ]
                fingers = [
                    p
                    for p in pairs
                    if "gripper0_right_" in p[0] or "gripper0_right_" in p[1]
                ]
                table = [
                    p
                    for p in pairs
                    if "table_collision" in p[0] or "table_collision" in p[1]
                ]
                self.drop_rows.append(
                    {
                        "tick_end_s": float(state["sim_time"] - self.replay_start),
                        "opening_command_at_tick_start": bool(opening),
                        "root_pose": state["scene"]["rigid_object"]["croissant"][
                            "root_pose"
                        ][0].tolist(),
                        "root_velocity": state["scene"]["rigid_object"]["croissant"][
                            "root_velocity"
                        ][0].tolist(),
                        "aperture": float(state["observations"]["gripper_pos"][0]),
                        "finger_contacts": fingers,
                        "table_contacts": table,
                    }
                )
            return {"state": state}
        raise ValueError(f"unsupported direct replay method {name}")

    def close(self):
        self.adapter.check()
        if self.contact_audit is not None:
            self.contact_audit.parent.mkdir(parents=True, exist_ok=True)
            self.contact_audit.write_text(
                json.dumps(
                    {
                        "schema": "mesa.drop_contact_audit.v1",
                        "timing": "forces from the completed native step; event is bounded by its 2ms step interval",
                        "instrumentation_changes_physics": False,
                        "rows": self.drop_rows,
                    },
                    indent=2,
                )
                + "\n"
            )


def main():
    from isaaclab.app import add_launcher_args, launch_simulation
    from mesa_isaac.teleop.door import (
        DEFAULT_DOOR_DAMPING_NM_S_PER_RAD,
        DEFAULT_DOOR_FRICTION_NM,
        DEFAULT_DOOR_SPRING_PRELOAD_NM,
        DEFAULT_DOOR_SPRING_STIFFNESS_NM_PER_RAD,
    )
    from mesa_isaac.teleop.gripper import DEFAULT_GRIPPER_PROFILE, GRIPPER_PROFILES
    from mesa_isaac.teleop.handle_geometry import (
        DEFAULT_OVEN_HANDLE_CONFIG,
        handle_config_argument,
    )

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("bundle", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--placement-overrides", type=Path)
    p.add_argument("--allow-unmeasured-start", action="store_true")
    p.add_argument("--allow-scene-change", action="store_true")
    p.add_argument("--state", default="parity_s1")
    p.add_argument("--door-friction", type=float, default=DEFAULT_DOOR_FRICTION_NM)
    p.add_argument(
        "--door-damping", type=float, default=DEFAULT_DOOR_DAMPING_NM_S_PER_RAD
    )
    p.add_argument(
        "--door-spring-stiffness",
        type=float,
        default=DEFAULT_DOOR_SPRING_STIFFNESS_NM_PER_RAD,
    )
    p.add_argument(
        "--door-spring-preload", type=float, default=DEFAULT_DOOR_SPRING_PRELOAD_NM
    )
    p.add_argument(
        "--gripper-profile", choices=GRIPPER_PROFILES, default=DEFAULT_GRIPPER_PROFILE
    )
    p.add_argument(
        "--croissant-contact-config",
        type=Path,
        help="Optional experimental rigid-mesh contact-compliance JSON; no volumetric deformation",
    )
    p.add_argument(
        "--oven-handle-config",
        type=handle_config_argument,
        default=DEFAULT_OVEN_HANDLE_CONFIG,
        help="Measured handle-spacing profile; explicit baseline restores previous geometry",
    )
    p.add_argument(
        "--contact-audit",
        type=Path,
        help="Record actual fingertip/object and table/object contact responses after opening",
    )
    add_launcher_args(p)
    args = p.parse_args()
    if args.repeat < 1:
        p.error("repeat must be positive")
    args.require_kit = True
    args.enable_cameras = False
    args.sim = "in-process IsaacAdapter; no cameras or hardware"
    arrays, metadata = load_bundle(args.bundle)

    import isaaclab.sim as sim_utils
    from isaaclab.utils import instantiate
    from mesa_isaac.teleop.scene import TeleopScene
    from mesa_isaac.twin.model import Twin
    from mesa_isaac.twin.scene import sim_cfg

    twin = Twin.load()
    cfg = sim_cfg(twin, "newton", args.device)
    with launch_simulation(cfg, args):
        from isaaclab_newton.physics import NewtonManager
        from mesa_isaac.teleop.adapter import IsaacAdapter

        visuals = TeleopScene(
            twin,
            state=args.state,
            door_friction=args.door_friction,
            door_damping=args.door_damping,
            door_spring_stiffness=args.door_spring_stiffness,
            door_spring_preload=args.door_spring_preload,
            gripper_profile=args.gripper_profile,
            croissant_contact_config=args.croissant_contact_config,
            oven_handle_config=args.oven_handle_config,
        )
        sim = sim_utils.SimulationContext(cfg)
        scene = instantiate(visuals.scene_cfg(backend="newton", cameras=False))
        NewtonManager.request_extended_contact_attribute("force")
        sim.reset()
        adapter = IsaacAdapter(sim, scene, twin, visuals)
        adapter.calibrate_gripper()
        adapter.metadata["offline_replay"] = {
            "mode": "native tick dynamics, no cameras/RPC",
            "renderer": "disabled",
            "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "schedule_sha256": hashlib.sha256(
                (
                    Path(__file__).parents[1] / "raiden/sim/command_replay.py"
                ).read_bytes()
            ).hexdigest(),
            "trace_writer_sha256": hashlib.sha256(
                Path(__file__).with_name("parity_replay.py").read_bytes()
            ).hexdigest(),
        }
        # Retain actual compiled support geometry for timing/height audits.
        solver = NewtonManager._solver
        native = solver.mj_model
        table_geoms = []
        for i in range(native.ngeom):
            if "table" not in native.geom(i).name.lower():
                continue
            position = solver.mjw_data.geom_xpos.numpy()[0, i]
            rotation = solver.mjw_data.geom_xmat.numpy()[0, i].reshape(3, 3)
            size = solver.mjw_model.geom_size.numpy()[0, i]
            row = {
                "index": i,
                "name": native.geom(i).name,
                "type": int(native.geom_type[i]),
                "position_world_m": position.tolist(),
                "rotation_world": rotation.tolist(),
                "size": size.tolist(),
                "margin_m": float(solver.mjw_model.geom_margin.numpy()[0, i]),
                "gap_m": float(solver.mjw_model.geom_gap.numpy()[0, i]),
            }
            if (
                int(native.geom_type[i]) == 6
            ):  # MuJoCo box: size is the local half-extent.
                row["top_world_z_m"] = float(position[2] + np.abs(rotation[2]) @ size)
            table_geoms.append(row)
        adapter.metadata["offline_replay"]["compiled_table_geometry"] = table_geoms
        replay(
            args, arrays, metadata, connection=DirectIsaac(adapter, args.contact_audit)
        )


if __name__ == "__main__":
    main()
