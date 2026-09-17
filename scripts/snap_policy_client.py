#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = [
#   "chiral",
#   "diffusers==0.35.2",
#   "easydict",
#   "einops",
#   "gymnasium",
#   "h5py",
#   "hydra-core==1.3.2",
#   "imageio",
#   "matplotlib",
#   "mujoco==3.3.5",
#   "natsort",
#   "numpy",
#   "open3d==0.19.0",
#   "opencv-python-headless",
#   "positional-encodings==6.0.3",
#   "timm",
#   "torch==2.10.0",
#   "torchvision==0.25.0",
# ]
#
# [tool.uv.sources]
# chiral = { git = "https://github.com/TRI-ML/chiral", rev = "f7293d7ddf2209da8d49922dddb2f1f1bc9b4454" }
# ///
"""Run an imitation frozen-token DiT checkpoint closed-loop on the left YAM arm.

These checkpoints read 256 px images through a frozen SNAP/DINOv3 tokenizer
instead of a trained ResNet, so the client builds a *live tokenizer* and
attaches it to the policy's ``FrozenTokenEncoder``.  Two arms are supported
and auto-detected from the checkpoint config:

``canon_v2_k1``
    the canonicalized arm (``rig_oven_canon``).  The SNAP decoder re-renders
    the scene view at one fixed canonical pose, so it needs the live
    scene-camera pose, camera-to-left-arm-base in OpenCV axes, before every
    inference.  A pose that drifts from the one the tokens were built around
    shifts every token silently, so the client checks it at start-up.
``dino``
    the raw DINOv3-base arm (``rig_oven_dino``).  No decoder, no pose.  Its
    featurizer weights are the ``featurizer.*`` keys of the canon backbone, so
    both arms load the same ``mesaft_v1_ep16_ema_full.pt``.

Unlike the ResNet client there is no temporal aggregation: one replan costs
~55 ms on a 3090 for canon (40 ms tokenizer + 13 ms DiT), more than the 33 ms
control period, so the policy always predicts a chunk, executes
``action_horizon`` steps and replans.  Actions are never sent late: planned
action k is due k control periods after its observation, and any action
already due when planning finishes is dropped.

The policy code comes from the imitation repo on ``PYTHONPATH``; ``snap_model``
and ``snap_decoder`` come from the checkpoint folder.  The script declares its
own dependencies, so ``uv run`` builds an isolated environment::

    PYTHONPATH=~/robot/imitation uv run scripts/snap_policy_client.py \\
        weights/3D_sim2real/checkpoints/rig_oven_canon/m54_real_croissant_canon_v1_ext24_epoch_0016.pth

The script homes the arm once at start-up and asks before starting each
episode.  Enter or Ctrl-C during an episode is the soft stop: the policy stops
sending, the server holds the last target for ``HOLD_BEFORE_HOME_S`` seconds,
then the arm returns to the rest pose over the server's usual 2 s ramp.  The
motors are never released.  Ctrl-C at a prompt quits.
"""

import argparse
import contextlib
import io
import json
import os
import select
import signal
import sys
import threading
import time
from pathlib import Path

import cv2
import imitation.utils.obs_utils as ObsUtils
import numpy as np
import torch
from chiral import PolicyClient
from chiral.types import CameraInfo, Observation
from hydra.utils import instantiate
from imitation.utils import utils
from omegaconf import OmegaConf

CAMERAS = ("scene_camera", "left_wrist_camera")
CAMERA_SHAPE = (480, 640, 3)
PROPRIO = "follower_l_joint_pos"
ACTION_DIM = 7
HOLD_BEFORE_HOME_S = 1.0

# The SNAP encoder+decoder pair. Both arms use it: canon through the decoder,
# dino through its (bit-identical) frozen featurizer.
BACKBONE_NAME = "mesaft_v1_ep16_ema_full.pt"
CANON_DIR = "rig_oven_canon"
SPEC_NAME = "canon_v2_k1_tokenizer_spec.json"
CALIBRATION = Path.home() / ".config/raiden/calibration_results.json"
# A shifted scene camera silently shifts every canonical token; no gate downstream
# catches it. Tolerances are tight because the pose is a calibration constant.
POSE_TOL_DEG = 0.5
POSE_TOL_M = 0.005
INTRINSIC_TOL_PX = 1.0


def enter_pressed() -> bool:
    """True if a line is waiting on stdin (the operator pressed Enter)."""
    ready, _, _ = select.select([sys.stdin], [], [], 0)
    if not ready:
        return False
    sys.stdin.readline()
    return True


def stop_and_home(client) -> None:
    """Soft stop: hold the last target, then return to the rest pose.

    The server keeps commanding the last action while nothing new arrives,
    so the arm stays put during the pause.  ``reset`` then ramps it home.
    """
    print(f"Policy stopped. Holding {HOLD_BEFORE_HOME_S:.0f} s, then moving home...")
    time.sleep(HOLD_BEFORE_HOME_S)
    client.reset()
    print("Arm at rest.")


def load_policy(checkpoint: str):
    sd = utils.load_checkpoint(checkpoint)
    cfg = sd["config"]
    shape_meta = OmegaConf.create(cfg["task"]["shape_meta"])
    ObsUtils.initialize_obs_utils_with_obs_specs(
        {
            "obs": {
                "rgb": list(shape_meta.observation.rgb),
                "depth": list(shape_meta.observation.depth),
                "low_dim": list(shape_meta.observation.lowdim),
            }
        }
    )
    with contextlib.redirect_stdout(io.StringIO()):
        model = instantiate(cfg["algo"]["policy"], shape_meta=shape_meta)
    model.to(model.device)
    model.load_state_dict(sd["model"], strict=True)
    model.normalizer.fit(sd["norm_stats"])
    model.eval()
    return model, shape_meta, cfg


def resolve_tokenizer(cfg, override: str | None) -> str:
    """Pick the live tokenizer for this checkpoint.

    The two arms share every config value except ``exp_name``, so that is what
    identifies them.  ``--tokenizer`` overrides the guess.
    """
    if override:
        return override
    exp = str(cfg.get("exp_name", ""))
    for name in ("canon", "dino"):
        if name in exp:
            return "canon_v2_k1" if name == "canon" else "dino"
    raise SystemExit(
        f"cannot tell the tokenizer from exp_name {exp!r}; pass --tokenizer"
    )


def find_file(checkpoint: str, name: str) -> Path:
    """Locate a tokenizer asset next to the checkpoint, or in the canon folder.

    The dino arm ships without the backbone and the spec; both live in the
    sibling ``rig_oven_canon`` folder, which is the certified source for it.
    """
    here = Path(checkpoint).resolve().parent
    for candidate in (here / name, here.parent / CANON_DIR / name):
        if candidate.exists():
            return candidate
    raise SystemExit(f"{name} not found next to {here} or in ../{CANON_DIR}")


def build_tokenizer(name: str, backbone: Path, spec: dict | None, device: str):
    """Instantiate a live tokenizer, importing ``snap_*`` from the backbone dir."""
    # live_tokenizers reads SNAP_PROBES_DIR at import time and falls back to a
    # Phoenix path; point both it and sys.path at the checkpoint folder.
    os.environ.setdefault("SNAP_PROBES_DIR", str(backbone.parent))
    if str(backbone.parent) not in sys.path:
        sys.path.insert(0, str(backbone.parent))
    from imitation.algos.encoders import live_tokenizers

    with contextlib.redirect_stdout(io.StringIO()):
        tokenizer = live_tokenizers.TOKENIZERS[name](
            checkpoint=str(backbone), device=device
        )
    if spec is not None:
        # The tokenizer's built-in constants must be the ones the token cache
        # was built with; the spec JSON is that cache's record.
        for attr, key in (
            ("T_canonical", "canonical_pose_4x4"),
            ("K_scene", "scene_camera_intrinsic_256"),
        ):
            got = getattr(tokenizer, attr).cpu().numpy()
            want = np.asarray(spec[key], dtype=np.float64)
            if not np.allclose(got, want, atol=1e-5):
                raise SystemExit(
                    f"tokenizer {attr} does not match {SPEC_NAME}:\n{got}\n{want}"
                )
    return tokenizer


class TimedTokenizer:
    """Pass-through wrapper that times each tokenizer call on the GPU."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.token_dim = tokenizer.token_dim
        self.requires_poses = getattr(tokenizer, "requires_poses", False)
        self.ms = 0.0

    def set_poses(self, *args, **kwargs) -> None:
        self.tokenizer.set_poses(*args, **kwargs)

    def __call__(self, rgb):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = self.tokenizer(rgb)
        torch.cuda.synchronize()
        self.ms = (time.perf_counter() - t0) * 1e3
        return out


def nominal_pose(spec: dict) -> np.ndarray:
    """``T_ref0``: the scene-camera pose the canonical view was orbited from.

    The builder made the canonical pose by rotating the nominal scene camera by
    ``canonical_yaw_deg`` about the base-frame z axis through ``canonical_pivot``
    (pitch 0, dolly 1), so undoing that rotation recovers the nominal pose —
    the pose every cached token was normalized by.
    """
    T = np.asarray(spec["canonical_pose_4x4"], dtype=np.float64)
    pivot = np.asarray(spec["canonical_pivot"], dtype=np.float64)
    yaw = -np.deg2rad(spec["canonical_yaw_deg"])
    c, s = np.cos(yaw), np.sin(yaw)
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    out = np.eye(4)
    out[:3, :3] = Rz @ T[:3, :3]
    out[:3, 3] = pivot + Rz @ (T[:3, 3] - pivot)
    return out


# Backbone fine-tune orbit bands relative to the training camera
# (scripts/teaser_capture.py).  The training camera itself sits at yaw 0.
ORBIT_BANDS = {
    "yaw_deg": (-156.2, -20.0),
    "pitch_deg": (0.0, 20.0),
    "dolly": (0.7, 1.2),
    "rot_off_deg": (0.0, 10.0),
}


def orbit(T: np.ndarray, T0: np.ndarray, pivot: np.ndarray) -> dict:
    """Yaw, pitch and dolly of camera pose T relative to T0 about the pivot,
    plus the residual rotation once that orbit is removed (teaser_capture.orbit)."""
    v, v0 = T[:3, 3] - pivot, T0[:3, 3] - pivot
    yaw = np.degrees(np.arctan2(v[1], v[0]) - np.arctan2(v0[1], v0[0]))
    yaw = (yaw + 180.0) % 360.0 - 180.0
    pitch = np.degrees(
        np.arcsin(v[2] / np.linalg.norm(v)) - np.arcsin(v0[2] / np.linalg.norm(v0))
    )
    dolly = np.linalg.norm(v) / np.linalg.norm(v0)
    # Rotation that carries v0 onto v; compare the orientations after removing it.
    a, b = v0 / np.linalg.norm(v0), v / np.linalg.norm(v)
    axis = np.cross(a, b)
    s, c = np.linalg.norm(axis), float(np.dot(a, b))
    if s < 1e-9:
        R = np.eye(3)
    else:
        k = axis / s
        K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        R = np.eye(3) + K * s + K @ K * (1 - c)
    rot_off, _ = pose_error(
        np.block([[R @ T0[:3, :3], np.zeros((3, 1))], [np.zeros((1, 3)), 1]]),
        np.block([[T[:3, :3], np.zeros((3, 1))], [np.zeros((1, 3)), 1]]),
    )
    return {
        "yaw_deg": float(yaw),
        "pitch_deg": float(pitch),
        "dolly": float(dolly),
        "rot_off_deg": float(rot_off),
    }


def calibration_pose() -> np.ndarray | None:
    """The scene-camera extrinsic the server itself publishes, read directly."""
    if not CALIBRATION.exists():
        return None
    ext = json.loads(CALIBRATION.read_text())["cameras"]["scene_camera"]["extrinsics"]
    if not ext.get("success"):
        return None
    T = np.eye(4)
    T[:3, :3] = np.asarray(ext["rotation_matrix"], dtype=np.float64)
    T[:3, 3] = np.asarray(ext["translation_vector"], dtype=np.float64).flatten()
    return T


def pose_error(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Rotation error in degrees and translation error in metres."""
    cos = (np.trace(a[:3, :3].T @ b[:3, :3]) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))), float(
        np.linalg.norm(a[:3, 3] - b[:3, 3])
    )


def check_scene_camera(obs, spec: dict, allow_drift: bool) -> None:
    """Fail loudly if the served scene camera is not the one behind the tokens.

    The canonical tokens are a function of the scene camera's pose *and* of the
    intrinsics baked into the tokenizer.  Both are constants of the calibration
    that built the training cache; a quiet change in either shifts every token
    while every downstream gate still looks healthy.
    """
    cam = obs["scene_camera"]
    served = np.asarray(cam.extrinsics, dtype=np.float64)
    problems = []

    # Blocking: the served pose must be the calibration file's pose.  The file
    # is the only record of where the camera really is, so it must be fresh
    # (scripts/write_scene_calib.py after a teaser_capture board shot).
    calib = calibration_pose()
    if calib is not None:
        deg, metres = pose_error(served, calib)
        ok = deg <= POSE_TOL_DEG and metres <= POSE_TOL_M
        print(
            f"  scene pose vs {CALIBRATION}: {deg:.3f} deg, {metres * 1e3:.2f} mm"
            f"  {'ok' if ok else 'MISMATCH'}"
        )
        if not ok:
            problems.append(str(CALIBRATION))

    # Informative: where this viewpoint sits relative to the training camera.
    # A novel viewpoint is a legitimate eval cell; the tokenizer normalises by
    # the served pose.  Outside the backbone's trained orbit bands, expect
    # degraded tokens.
    nominal = nominal_pose(spec)
    deg, metres = pose_error(served, nominal)
    orb = orbit(served, nominal, np.asarray(spec["canonical_pivot"], dtype=np.float64))
    in_band = all(lo <= orb[k] <= hi for k, (lo, hi) in ORBIT_BANDS.items())
    print(
        f"  scene pose vs training camera ({SPEC_NAME} T_ref0): {deg:.2f} deg, "
        f"{metres * 1e3:.0f} mm; orbit yaw {orb['yaw_deg']:+.1f} pitch "
        f"{orb['pitch_deg']:+.1f} dolly {orb['dolly']:.2f} rot_off "
        f"{orb['rot_off_deg']:.1f} deg  "
        f"{'(training pose)' if deg <= POSE_TOL_DEG and metres <= POSE_TOL_M else 'NOVEL VIEWPOINT'}"
        f"{'' if in_band else ', outside trained bands'}"
    )

    # The tokenizer's intrinsic is the 256 px frame of a 480x480 centre crop.
    scale = spec["input_size"] / CAMERA_SHAPE[0]
    K = np.asarray(cam.intrinsics, dtype=np.float64).copy()
    K[0, 2] -= (CAMERA_SHAPE[1] - CAMERA_SHAPE[0]) / 2.0
    K[:2] *= scale
    err = float(np.abs(K - np.asarray(spec["scene_camera_intrinsic_256"])).max())
    print(f"  scene intrinsics vs spec (256 px frame): {err:.3f} px")
    if err > INTRINSIC_TOL_PX:
        problems.append("scene_camera_intrinsic_256")

    if not problems:
        return
    print(
        "\n*** The served scene camera does not match: "
        + ", ".join(problems)
        + "\n*** The canonical tokens are built from this pose and intrinsic. A"
        "\n*** wrong pose shifts every token and NOTHING downstream will notice."
        "\n*** Refit with scripts/write_scene_calib.py, or pass"
        "\n*** --allow-pose-drift to serve anyway.\n"
    )
    if not allow_drift:
        raise SystemExit("refusing to serve on a drifted scene camera")
    print("--allow-pose-drift given; continuing.\n")


def preprocess_image(img: np.ndarray, size: int) -> np.ndarray:
    """640x480 RGB uint8 -> size x size RGB uint8: centre 480x480 crop, area resize."""
    h, w = img.shape[:2]
    x0 = (w - h) // 2
    return cv2.resize(img[:, x0 : x0 + h], (size, size), interpolation=cv2.INTER_AREA)


def to_model_input(obs, shape_meta) -> dict:
    """Map a chiral observation to ``ChunkPolicy.get_action`` kwargs.

    Images [B, T, H, W, C] float 0-255 RGB, lowdim [B, T, D], zero task embedding.
    """
    data = {}
    for cam in CAMERAS:
        size = shape_meta.observation.rgb[f"{cam}_image"][-1]
        img = preprocess_image(obs[cam].image, size)
        data[f"{cam}_image"] = img[None, None].astype(np.float32)
    q = obs.proprios[PROPRIO]
    data["robot0_joint_pos"] = q[None, None, :6]
    data["robot0_gripper_jaw_width"] = q[None, None, 6:7]
    return {"obs": data, "task_emb": torch.zeros(1, shape_meta.task.dim)}


class CudaGraphSampler:
    """Drop-in for ``DiffusionTransformerPolicy.sample_actions``.

    Observation encoding — including the live tokenizer — stays eager; the DiT
    encoder and the whole DDIM loop (eta=0, epsilon prediction, clipped x0)
    replay as one CUDA graph.  The conditioning is a fixed 514 tokens (2 x 256
    perception + proprio + language), exactly the shape the ResNet arm uses, so
    the graph captures unchanged.
    """

    def __init__(self, model):
        self.model = model
        sched = model.diffusion_schedule
        sched.set_timesteps(model._eval_diffusion_steps)
        stride = sched.config.num_train_timesteps // sched.num_inference_steps
        self.clip = sched.config.clip_sample_range
        self.steps = []
        for t in sched.timesteps.tolist():
            a_t = sched.alphas_cumprod[t]
            a_prev = (
                sched.alphas_cumprod[t - stride]
                if t >= stride
                else sched.final_alpha_cumprod
            )
            self.steps.append(
                (
                    torch.tensor([t], device=model.device),
                    float(a_t**0.5),
                    float((1 - a_t) ** 0.5),
                    float(a_prev**0.5),
                    float((1 - a_prev) ** 0.5),
                )
            )
        self.graph = None

    def _sample(self):
        enc_cache = self.model.noise_net.forward_enc(self.cond)
        x = self.noise
        for t, sa, sb, sa_prev, sb_prev in self.steps:
            eps = self.model.noise_net.forward_dec(x, t, enc_cache)
            x0 = ((x - sb * eps) / sa).clamp(-self.clip, self.clip)
            x = sa_prev * x0 + sb_prev * eps
        return x

    def _capture(self, cond):
        m = self.model
        self.cond = cond.clone()
        self.noise = torch.randn(1, m.chunk_size, m.network_action_dim, device=m.device)
        side = torch.cuda.Stream()
        with torch.cuda.stream(side):
            for _ in range(3):
                self._sample()
        torch.cuda.current_stream().wait_stream(side)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.out = self._sample()

    @torch.no_grad()
    def __call__(self, data):
        data = self.model.preprocess_input(data, train_mode=False)
        cond = self.model.get_cond(data)
        if self.graph is None:
            self._capture(cond)
        self.cond.copy_(cond)
        self.noise.normal_()
        self.graph.replay()
        return self.out.cpu().numpy()


def run_episode(client, model, shape_meta, tokenizer, hz: float) -> None:
    period = 1.0 / hz
    model.reset()
    plan: list[np.ndarray] = []
    obs_ms, tok_ms, dit_ms = [], [], []
    sent = dropped = sent_log = 0
    t_start = t_log = time.perf_counter()
    stop = threading.Event()
    previous_handler = signal.signal(signal.SIGINT, lambda *_: stop.set())
    try:
        while not stop.is_set():
            if enter_pressed():
                print("Enter pressed.")
                break
            if not plan:
                t0 = time.perf_counter()
                obs = client.get_obs()
                t1 = time.perf_counter()
                if tokenizer.requires_poses:
                    # Consumed by the next call, so it can never go stale.
                    tokenizer.set_poses(obs["scene_camera"].extrinsics)
                inputs = to_model_input(obs, shape_meta)
                chunk = model.get_action(**inputs, return_full_chunk=True)[0]
                plan = list(chunk[: model.action_horizon])
                t2 = time.perf_counter()
                obs_ms.append((t1 - t0) * 1e3)
                tok_ms.append(tokenizer.ms)
                dit_ms.append((t2 - t1) * 1e3 - tokenizer.ms)
                if not np.isfinite(plan).all():
                    print("Non-finite action; stopping.")
                    break
                # plan[k] is due k periods after the observation; drop the ones already due.
                late = min(int((t2 - t0) / period), len(plan))
                dropped += late
                del plan[:late]
                next_tick = t0 + late * period
                if not plan:
                    continue

            action = plan.pop(0)
            client.put_action(action)
            sent += 1

            now = time.perf_counter()
            if now - t_log >= 1.0:
                q = obs.proprios[PROPRIO]
                print(
                    f"t={now - t_start:6.1f}s  {(sent - sent_log) / (now - t_log):4.1f}Hz  "
                    f"obs {np.mean(obs_ms[-4:]):4.1f}ms  tok {np.mean(tok_ms[-4:]):4.1f}ms  "
                    f"dit {np.mean(dit_ms[-4:]):4.1f}ms  dropped {dropped}  "
                    f"|a-q| {np.abs(action[:6] - q[:6]).max():.3f}rad  jaw {action[6]:.2f}"
                )
                t_log, sent_log = now, sent

            next_tick += period
            delay = next_tick - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                next_tick = time.perf_counter()
    finally:
        signal.signal(signal.SIGINT, previous_handler)
    elapsed = time.perf_counter() - t_start
    if tok_ms:
        print(
            f"\nStopped: {sent} actions in {elapsed:.1f}s ({sent / elapsed:.1f} Hz), "
            f"{len(tok_ms)} replans, {dropped} late actions dropped; "
            f"tok p50 {np.median(tok_ms):.1f} p99 {np.percentile(tok_ms, 99):.1f}  "
            f"dit p50 {np.median(dit_ms):.1f} p99 {np.percentile(dit_ms, 99):.1f}ms"
        )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("checkpoint", help="imitation .pth checkpoint")
    ap.add_argument("--uri", default="ws://localhost:8765", help="rd serve address")
    ap.add_argument("--hz", type=float, default=30.0, help="control rate")
    ap.add_argument(
        "--tokenizer",
        choices=("canon_v2_k1", "dino"),
        help="live tokenizer (default: from the checkpoint's exp_name)",
    )
    ap.add_argument(
        "--backbone",
        help=f"SNAP checkpoint (default: {BACKBONE_NAME} beside the policy "
        f"or in ../{CANON_DIR})",
    )
    ap.add_argument(
        "--allow-pose-drift",
        action="store_true",
        help="serve even if the scene camera does not match the tokenizer spec",
    )
    args = ap.parse_args()

    print("Loading policy...")
    model, shape_meta, cfg = load_policy(args.checkpoint)
    encoder = model.encoder
    if not hasattr(encoder, "set_tokenizer"):
        raise SystemExit(
            f"{type(encoder).__name__} is not a FrozenTokenEncoder; this "
            "checkpoint wants scripts/dit_policy_client.py"
        )

    name = resolve_tokenizer(cfg, args.tokenizer)
    backbone = (
        Path(args.backbone)
        if args.backbone
        else find_file(args.checkpoint, BACKBONE_NAME)
    )
    spec = None
    if name == "canon_v2_k1":
        spec = json.loads(find_file(args.checkpoint, SPEC_NAME).read_text())
    print(f"Loading the {name} tokenizer from {backbone}...")
    tokenizer = TimedTokenizer(build_tokenizer(name, backbone, spec, model.device))
    if tokenizer.token_dim != encoder.token_dim:
        raise SystemExit(
            f"{name} emits dim {tokenizer.token_dim}, the policy wants "
            f"{encoder.token_dim}"
        )
    encoder.set_tokenizer(tokenizer)
    model.sample_actions = CudaGraphSampler(model)

    # Warm up and capture the CUDA graph before the arm is live.
    blank = Observation(
        cameras=[
            CameraInfo(c, np.eye(3), np.eye(4), np.zeros(CAMERA_SHAPE, np.uint8))
            for c in CAMERAS
        ],
        proprios={PROPRIO: np.zeros(ACTION_DIM, np.float32)},
    )
    for _ in range(3):
        if tokenizer.requires_poses:
            tokenizer.set_poses(nominal_pose(spec))
        model.get_action(**to_model_input(blank, shape_meta), return_full_chunk=True)
    size = shape_meta.observation.rgb[f"{CAMERAS[0]}_image"][-1]
    print(
        f"Ready: chunk {model.chunk_size}, horizon {model.action_horizon}, "
        f"{size} px, tokenizer {tokenizer.ms:.0f} ms"
    )

    with PolicyClient(args.uri) as client:
        meta = client.get_metadata()
        if meta.get("action_shape") != [ACTION_DIM]:
            raise SystemExit(
                f"Server action_shape {meta.get('action_shape')}, expected [{ACTION_DIM}]"
            )
        # One action is queued per tick; a fast dispatch sends it immediately.
        client.start_action_dispatch(hz=1000)
        try:
            input("\nEnter: move the arm home   (Ctrl-C: quit) ")
        except (KeyboardInterrupt, EOFError):
            print()
            return
        obs, _ = client.reset()
        for cam in CAMERAS:
            if obs[cam].image.shape != CAMERA_SHAPE:
                raise SystemExit(
                    f"{cam}: got {obs[cam].image.shape}, expected "
                    f"{CAMERA_SHAPE}; serve native frames without resizing"
                )
        if tokenizer.requires_poses:
            print("\nScene camera check:")
            check_scene_camera(obs, spec, args.allow_pose_drift)
        while True:
            try:
                input(
                    "\nEnter: start the policy   "
                    "(Enter or Ctrl-C during the episode: hold 1 s, then home) "
                )
            except (KeyboardInterrupt, EOFError):
                print()
                break
            run_episode(client, model, shape_meta, tokenizer, args.hz)
            stop_and_home(client)


if __name__ == "__main__":
    main()
