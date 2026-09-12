"""Render recorded sim frames again from their logged states.

Sim cameras store no pixels.  Each recorded frame keeps the state it was rendered from
(``sim_state.npz``) and the episode keeps the model (``sim_model.xml``); rendering that
state again reproduces the frame (mean difference 0.002 grey levels on the 2026-09-11 demos).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import mujoco
import numpy as np

# MuJoCo cameras look down -z with +y up; dataset poses use the OpenCV convention
# (+z into the scene, +y down).
_MUJOCO_TO_OPENCV = np.diag([1.0, -1.0, -1.0, 1.0])


def camera_pose(data: mujoco.MjData, cam_id: int) -> np.ndarray:
    """The camera's OpenCV-convention pose in the model's world frame (4x4).

    Valid for the state ``data`` currently holds, so a moving camera must be read after
    ``mj_forward``.
    """
    T = np.eye(4)
    T[:3, :3] = data.cam_xmat[cam_id].reshape(3, 3)
    T[:3, 3] = data.cam_xpos[cam_id]
    return T @ _MUJOCO_TO_OPENCV


def camera_id(model: mujoco.MjModel, camera: str) -> int:
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
    if cam_id < 0:
        raise ValueError(f"the sim model has no camera named {camera!r}")
    return cam_id


def render_states(
    model_xml: Path,
    camera: str,
    states: np.ndarray,
    width: int,
    height: int,
    out_dir: Path,
) -> np.ndarray:
    """Write ``out_dir/<idx>.png`` (BGR, like RealSense) for each ``[time, qpos, qvel]`` row.

    ``camera`` is the MuJoCo camera name.  The scene options match the sim server's rig
    cameras: visual geoms (group 1) only, no sites.

    Returns the camera's world pose per frame, (N, 4, 4).  Each pose is read from the state
    that rendered its own frame, so a frame and its pose can never disagree.
    """
    model = mujoco.MjModel.from_xml_path(str(model_xml))
    if len(states) and states.shape[1] != 1 + model.nq + model.nv:
        raise ValueError(
            f"sim states have {states.shape[1]} values; {model_xml.name} needs 1 + nq + nv = "
            f"{1 + model.nq + model.nv}"
        )
    cam_id = camera_id(model, camera)
    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), width)
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), height)
    data = mujoco.MjData(model)
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = 0
    opt.geomgroup[1] = 1
    opt.sitegroup[:] = 0
    out_dir.mkdir(parents=True, exist_ok=True)
    poses = np.empty((len(states), 4, 4))
    renderer = mujoco.Renderer(model, height=height, width=width)
    try:
        for i, s in enumerate(states):
            data.time = s[0]
            data.qpos[:] = s[1 : 1 + model.nq]
            data.qvel[:] = s[1 + model.nq :]
            mujoco.mj_forward(model, data)
            poses[i] = camera_pose(data, cam_id)
            renderer.update_scene(data, camera=camera, scene_option=opt)
            cv2.imwrite(str(out_dir / f"{i:010d}.png"), renderer.render()[:, :, ::-1])
    finally:
        renderer.close()
    return poses
