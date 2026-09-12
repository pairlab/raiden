"""The two candidate kinematic chains for the YAM arm, and the fingertip probe.

Two MuJoCo models describe the same physical arm:

* :func:`vendor_chain` — the i2rt model that ships with the driver. raiden uses it for the FK
  pose labels, teleop and replay IK, the reach guides and the hand-eye calibration.
* :func:`mesa_chain` — the MuJoCo Menagerie model the MESA twin simulates and renders, placed
  the way ``RaidenYam`` places it (its base body sits :data:`MENAGERIE_BASE_DZ` above the i2rt
  base origin).

They disagree. Joint 1 coincides, but joints 2-4 sit 4.6 mm apart perpendicular to their axes
and links 5-6 add a few mm more, so the same joint angles put the fingertips a median 7 mm
apart and the wrist camera 4.5 mm apart. Nothing on the rig has ever measured which one matches
the real arm: the 8-view ChArUco capture fits both to 2.6-2.9 px.

Every quantity here is in the i2rt base frame (z up, joint 1 on the z axis), so one touch-off
evaluated through both chains says which one describes the real arm. Flatness does not settle
it -- over poses that touch a horizontal table the two models differ by an almost constant
height -- but the tool tilt does: see :func:`compare_chains`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np

# The i2rt base origin sits this far below the Menagerie base body: joint 1 is at z = 0.067 in
# the i2rt model and at 0.0631 in the Menagerie one. ``RaidenYam.base_xpos_offset`` adds it so
# that joint 1 lands in the same place in both.
MENAGERIE_BASE_DZ = 0.067 - 0.0631


@dataclass
class Contact:
    """One touch-off, stored as joint angles only so any chain can be evaluated against it."""

    target: Tuple[float, float, float]  # commanded probe point (base frame)
    tilt_deg: float  # tool tilt off vertical; the axis that separates the chains
    azimuth_deg: float  # direction of that tilt
    cmd_joints: np.ndarray  # (n, 6) commanded arm joints through the descent
    meas_joints: np.ndarray  # (n, 6) measured arm joints, same grid
    torque: np.ndarray  # (n, 6) measured joint torque


class ArmChain:
    """A YAM model with its fingertips closed, posed by joint angles in the i2rt base frame.

    The probe is the lowest point of the two fingertip meshes. Both models carry the same
    vendor fingertips, so the probe cancels out of a chain comparison; and because the twin
    contacts the table with those same meshes, a table height measured this way reproduces
    real contact in sim even if the mesh itself is a millimetre off the moulded rubber.
    """

    def __init__(
        self,
        name: str,
        xml_path: str | Path,
        base_body: Optional[str] = None,
        arm_joints: Sequence[str] = (),
        finger_joints: Sequence[str] = (),
        tip_bodies: Sequence[str] = (),
    ):
        import mujoco

        self.name = name
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        self._mujoco = mujoco

        def jid(n: str) -> int:
            i = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
            if i < 0:
                raise ValueError(f"{name}: no joint '{n}' in {xml_path}")
            return i

        self._arm_adr = np.array([self.model.jnt_qposadr[jid(n)] for n in arm_joints])
        self._finger = [
            (self.model.jnt_qposadr[jid(n)], self.model.jnt_range[jid(n)])
            for n in finger_joints
        ]

        # The base body carries the whole arm; the i2rt base origin is MENAGERIE_BASE_DZ below
        # it in the Menagerie model and coincides with it in the vendor one.
        self.base_z = 0.0
        if base_body is not None:
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, base_body)
            self.base_z = float(self.model.body_pos[bid][2]) - MENAGERIE_BASE_DZ

        # Mesh geoms belonging to the arm itself. ``mesa_chain`` is handed a whole scene, so the
        # table and the task objects have to be excluded from "is anything below the fingertips".
        root = (
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, base_body)
            if base_body
            else 0
        )
        self._root = root
        self._arm_geoms = [
            g
            for g in range(self.model.ngeom)
            if self.model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
            and self._descends_from(int(self.model.geom_bodyid[g]), root)
        ]

        self._tip_geoms = []
        for body in tip_bodies:
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body)
            if bid < 0:
                raise ValueError(f"{name}: no body '{body}' in {xml_path}")
            for g in range(self.model.ngeom):
                if self.model.geom_bodyid[g] != bid:
                    continue
                if self.model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
                    continue
                mid = self.model.geom_dataid[g]
                va, vn = self.model.mesh_vertadr[mid], self.model.mesh_vertnum[mid]
                self._tip_geoms.append((g, self.model.mesh_vert[va : va + vn].copy()))
        if not self._tip_geoms:
            raise ValueError(f"{name}: no fingertip mesh geoms found in {tip_bodies}")
        self._closed = self._closed_finger_qpos()

    def _closed_finger_qpos(self) -> float:
        """Whichever end of the finger travel brings the two fingertips together."""
        spans = []
        for q in (float(self._finger[0][1][0]), float(self._finger[0][1][1])):
            self._pose(np.zeros(6), q)
            tips = [self._tip_points(g, v) for g, v in self._tip_geoms]
            spans.append((np.linalg.norm(tips[0].mean(0) - tips[-1].mean(0)), q))
        return min(spans)[1]

    def _pose(self, q6: np.ndarray, finger: float) -> None:
        self.data.qpos[:] = 0.0
        self.data.qpos[self._arm_adr] = np.asarray(q6, dtype=np.float64)[:6]
        for adr, _ in self._finger:
            self.data.qpos[adr] = finger
        self._mujoco.mj_forward(self.model, self.data)

    def _tip_points(self, geom: int, verts: np.ndarray) -> np.ndarray:
        R = self.data.geom_xmat[geom].reshape(3, 3)
        return verts @ R.T + self.data.geom_xpos[geom]

    def set_joints(self, q6: np.ndarray) -> None:
        """Pose the arm at ``q6`` with the gripper closed."""
        self._pose(q6, self._closed)

    def probe_point(self, q6: np.ndarray) -> np.ndarray:
        """Lowest fingertip point in the i2rt base frame (3,)."""
        self.set_joints(q6)
        pts = np.concatenate([self._tip_points(g, v) for g, v in self._tip_geoms])
        p = pts[np.argmin(pts[:, 2])].copy()
        p[2] -= self.base_z
        return p

    def probe_z(self, q6: np.ndarray) -> float:
        return float(self.probe_point(q6)[2])

    def _descends_from(self, body: int, root: int) -> bool:
        while body != 0:
            if body == root:
                return True
            body = int(self.model.body_parentid[body])
        return root == 0

    def fingertip_clearance(self, q6: np.ndarray) -> float:
        """Height of the lowest other moving arm part above the fingertips, in metres.

        Positive means the fingertips lead the descent and the probe is the part that lands.
        The static base mesh is excluded: it is bolted to the rail above the table and never
        approaches it, but it does hang below a hovering gripper.
        """
        self.set_joints(q6)
        tip_z = min(self._tip_points(g, v)[:, 2].min() for g, v in self._tip_geoms)
        tip_ids = {g for g, _ in self._tip_geoms}
        lowest = np.inf
        for g in self._arm_geoms:
            if g in tip_ids or int(self.model.geom_bodyid[g]) == self._root:
                continue
            mid = self.model.geom_dataid[g]
            va, vn = self.model.mesh_vertadr[mid], self.model.mesh_vertnum[mid]
            lowest = min(
                lowest,
                self._tip_points(g, self.model.mesh_vert[va : va + vn])[:, 2].min(),
            )
        return float(lowest - tip_z)


def vendor_chain() -> ArmChain:
    """The i2rt model the driver ships and raiden already uses everywhere else."""
    from raiden._xml_paths import get_yam_4310_linear_xml_path

    return ArmChain(
        "i2rt vendor",
        get_yam_4310_linear_xml_path(),
        arm_joints=[f"joint{i}" for i in range(1, 7)],
        finger_joints=["joint7", "joint8"],
        tip_bodies=["tip_left", "tip_right"],
    )


def mesa_chain(xml_path: str | Path) -> ArmChain:
    """The Menagerie model MESA renders, from any episode's ``sim_model.xml``."""
    return ArmChain(
        "MESA Menagerie",
        xml_path,
        base_body="robot0_base",
        arm_joints=[f"robot0_joint{i}" for i in range(1, 7)],
        finger_joints=["gripper0_right_left_finger", "gripper0_right_right_finger"],
        tip_bodies=["gripper0_right_tip_left", "gripper0_right_tip_right"],
    )


def contact_height(cmd_z: np.ndarray, meas_z: np.ndarray, min_points: int = 4) -> dict:
    """Table height from one descent, as the break in measured-vs-commanded probe height.

    Above the table the fingertips follow the command with a constant gravity sag (slope 1);
    once they land, the command keeps descending and they stop (slope ~0). Fitting both
    segments and taking the break is free of the millimetre of overshoot that a lag threshold
    would bake in, and the two slopes say whether a touch really happened.
    """
    cmd_z, meas_z = np.asarray(cmd_z, float), np.asarray(meas_z, float)
    n = len(cmd_z)
    if n < 2 * min_points:
        raise ValueError(f"need at least {2 * min_points} samples, got {n}")
    best = None
    for k in range(min_points, n - min_points + 1):
        sse, fits = 0.0, []
        for lo, hi in ((0, k), (k, n)):
            A = np.c_[cmd_z[lo:hi], np.ones(hi - lo)]
            coef, res, *_ = np.linalg.lstsq(A, meas_z[lo:hi], rcond=None)
            sse += (
                float(res[0])
                if len(res)
                else float(np.sum((A @ coef - meas_z[lo:hi]) ** 2))
            )
            fits.append(coef)
        if best is None or sse < best[0]:
            best = (sse, k, fits)
    sse, k, (free, pressed) = best
    z_break = float(cmd_z[k - 1])
    return {
        "z": float(
            free[0] * z_break + free[1]
        ),  # where the fingertips actually stopped
        "approach_slope": float(free[0]),
        "contact_slope": float(pressed[0]),
        "residual_mm": float(1000 * np.sqrt(sse / n)),
        "break_index": int(k),
    }


def fit_plane(points: np.ndarray) -> dict:
    """Least-squares z = ax + by + c over contact points; tilt and residual in rig units."""
    P = np.asarray(points, float)
    A = np.c_[P[:, 0], P[:, 1], np.ones(len(P))]
    coef, *_ = np.linalg.lstsq(A, P[:, 2], rcond=None)
    resid = P[:, 2] - A @ coef
    n = np.array([-coef[0], -coef[1], 1.0])
    n /= np.linalg.norm(n)
    return {
        "z_in_base_at_origin": float(coef[2]),
        "normal_in_base": n.tolist(),
        "tilt_deg": float(np.degrees(np.arctan(np.hypot(coef[0], coef[1])))),
        "residual_rms_mm": float(1000 * np.sqrt(np.mean(resid**2))),
        "residual_max_mm": float(1000 * np.abs(resid).max()),
        "n_points": int(len(P)),
    }


def compare_chains(contacts: Sequence[Contact], chains: Sequence[ArmChain]) -> dict:
    """Re-evaluate the same measured joint angles through each chain.

    The planner's chain cannot bias this: a touch-off records where the fingertips *were*, and
    each chain then says where that is.

    Flatness alone does not rank the chains. Over poses that touch a horizontal table the two
    models differ by an almost constant height at a given tool tilt (under 0.3 mm of spread
    across x, y and approach azimuth), so both fit an equally flat plane, one offset from the
    other. The tilt is what separates them: the offset sweeps about 5 mm between a vertical
    tool and one tilted 50 degrees. The table top is a single height, so the chain that
    describes the real arm reports the same height whichever way the wrist is turned, and
    ``tilt_spread_mm`` is the discriminator.
    """
    out = {}
    for chain in chains:
        rows, fits = [], []
        for c in contacts:
            fit = contact_height(
                np.array([chain.probe_z(q) for q in c.cmd_joints]),
                np.array([chain.probe_z(q) for q in c.meas_joints]),
            )
            p = chain.probe_point(c.meas_joints[fit["break_index"] - 1])
            rows.append((c.tilt_deg, [p[0], p[1], fit["z"]]))
            fits.append(
                {**fit, "tilt_deg": c.tilt_deg, "x": float(p[0]), "y": float(p[1])}
            )
        by_tilt = {}
        for tilt in sorted({t for t, _ in rows}):
            pts = np.array([p for t, p in rows if t == tilt])
            by_tilt[tilt] = (
                fit_plane(pts)
                if len(pts) >= 3
                else {
                    "z_in_base_at_origin": float(pts[:, 2].mean()),
                    "n_points": int(len(pts)),
                    "normal_in_base": [0.0, 0.0, 1.0],
                    "tilt_deg": 0.0,
                    "residual_rms_mm": float("nan"),
                    "residual_max_mm": float("nan"),
                }
            )
        heights = [f["z_in_base_at_origin"] for f in by_tilt.values()]
        out[chain.name] = {
            **fit_plane(np.array([p for _, p in rows])),
            "by_tilt": by_tilt,
            "tilt_spread_mm": float(1000 * (max(heights) - min(heights))),
            "contacts": fits,
        }
    return out
