"""Rig geometry in the converter's ``calibration_results.json`` form.

The sim cameras are exact (rig.json); the table pose is shared by sim and real recordings.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np

from raiden.sim.client import SimConnection
from raiden.sim.serialization import write_json


def _grasp_from_tcp(T_tcp_cam: np.ndarray) -> np.ndarray:
    """Re-express a camera pose given in the i2rt ``tcp_site`` frame in the ``grasp_site`` frame.

    The converter builds wrist extrinsics as ``FK(grasp_site) @ T_cam2ee``; the rig stores the
    wrist camera relative to ``tcp_site``. Both sites live on link 6, so the relative transform is
    configuration independent and can be read off the model at q = 0.
    """
    from i2rt.robots.kinematics import Kinematics

    from raiden._xml_paths import get_yam_4310_linear_xml_path

    kin = Kinematics(get_yam_4310_linear_xml_path(), "grasp_site")
    q = np.zeros(kin._configuration.model.nq)
    T_base_grasp = np.asarray(kin.fk(q, "grasp_site"), dtype=np.float64)
    T_base_tcp = np.asarray(kin.fk(q, "tcp_site"), dtype=np.float64)
    return np.linalg.inv(T_base_grasp) @ T_base_tcp @ T_tcp_cam


def table_pose(layout: dict) -> np.ndarray:
    """``T_base_table``: the table frame in the arm base frame (table-to-base, 4x4).

    Origin at the centre of the table top, axes parallel to the base axes. Built from layout.json
    (taped table extent, table height from the rig fit) the way the MESA arena places its table,
    so sim and real share it. Unlike the rig fit's own table frame, it does not move with the
    scene camera.
    """
    table = layout["table"]
    (x0, x1), (y0, y1) = table["atlas"]["x_range"], table["atlas"]["y_range"]
    T = np.eye(4)
    T[:3, 3] = [(x0 + x1) / 2, (y0 + y1) / 2, table["z_in_base"]]
    return T


def base_in_world(
    model_xml: Path, mujoco_cameras: Dict[str, str], calib: dict
) -> np.ndarray:
    """The arm base frame in a sim model's world frame (4x4).

    MESA builds the model's rig cameras from the same ``rig.json`` this module writes into the
    recording's calibration, so a static rig camera ties the two frames together: the camera
    MuJoCo renders from and the camera the calibration describes are the same camera.

    ``mujoco_cameras`` maps a recording's camera names to their MuJoCo camera names.
    """
    import mujoco

    from raiden.sim.render import camera_id, camera_pose

    for name, mujoco_name in mujoco_cameras.items():
        # A wrist camera moves with the arm, so only a static one fixes the frame.
        ext = calib.get("cameras", {}).get(name, {}).get("extrinsics")
        if ext and ext.get("success"):
            break
    else:
        raise ValueError(
            "no statically calibrated sim camera to place the model in the arm base frame"
        )

    T_base_cam = np.eye(4)
    T_base_cam[:3, :3] = np.asarray(ext["rotation_matrix"], dtype=np.float64)
    T_base_cam[:3, 3] = np.asarray(
        ext["translation_vector"], dtype=np.float64
    ).flatten()
    model = mujoco.MjModel.from_xml_path(str(model_xml))
    cam_id = camera_id(model, mujoco_name)
    if model.cam_bodyid[cam_id] != 0:
        raise ValueError(
            f"{mujoco_name} moves with the robot; it cannot fix the base frame"
        )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return camera_pose(data, cam_id) @ np.linalg.inv(T_base_cam)


def sim_calibration(rig: dict) -> dict:
    cams = rig["cameras"]
    out = {
        "source": rig.get("calibration_source", "mesa digital twin (rig.json)"),
        "backend": rig.get("backend", "mujoco"),
        "cameras": {},
    }
    if "table" in rig:
        out["table"] = rig["table"]
    elif "layout" in rig:
        out["table"] = {"T_base_table": table_pose(rig["layout"]).tolist()}
    if "bimanual_transform" in rig:
        out["bimanual_transform"] = rig["bimanual_transform"]
    for name, cam in cams.items():
        if cam.get("dynamic") and cam.get("pose_source") == "native per-frame camera pose":
            out["cameras"][name] = {"pose_source": "recorded_camera_poses", "reference_frame": "left_arm_base"}
            continue
        if "T_base_cam" in cam:
            T = np.asarray(cam["T_base_cam"], dtype=np.float64)
            method = "extrinsics"
        elif "T_grasp_cam" in cam:
            T = np.asarray(cam["T_grasp_cam"], dtype=np.float64)
            method = "hand_eye_calibration"
        elif "T_tool_cam" in cam:
            T = _grasp_from_tcp(np.asarray(cam["T_tool_cam"], dtype=np.float64))
            method = "hand_eye_calibration"
        else:
            continue
        out["cameras"][name] = {method: {
            "success": True,
            "rotation_matrix": T[:3, :3].tolist(),
            "translation_vector": T[:3, 3].tolist(),
        }}
    return out


def write_sim_calibration(dest_dir: Path, rig: dict) -> Path:
    path = Path(dest_dir) / "calibration_results.json"
    with open(path, "w") as f:
        json.dump(sim_calibration(rig), f, indent=2)
    return path


def write_sim_files(dest_dir: Path, address: str) -> None:
    """Copy episode rig/calibration and the backend's replay description.

    Isaac uses ``sim_metadata.json`` with immutable asset/calibration identifiers;
    MuJoCo uses its self-contained ``sim_model.xml``.
    """
    conn = SimConnection(address)
    try:
        rig = conn.call("get_rig")
        write_sim_calibration(dest_dir, rig)
        write_json(Path(dest_dir) / "sim_rig.json", rig)
        if rig.get("backend") == "isaac":
            write_json(Path(dest_dir) / "sim_metadata.json", conn.call("get_metadata"))
        else:
            (Path(dest_dir) / "sim_model.xml").write_text(conn.call("get_model_xml"))
    finally:
        conn.close()
