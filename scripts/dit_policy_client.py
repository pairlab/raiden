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
#   "torch==2.10.0",
#   "torchvision==0.25.0",
# ]
#
# [tool.uv.sources]
# chiral = { git = "https://github.com/TRI-ML/chiral", rev = "f7293d7ddf2209da8d49922dddb2f1f1bc9b4454" }
# ///
"""Run an imitation DiT checkpoint closed-loop on the left YAM arm through ``rd serve``.

Every control tick sends one absolute joint action ``[l_joint(6), l_gripper]``.
By default the policy predicts a full chunk, executes ``action_horizon`` steps
and then replans, so every action sent is one the model predicted, as in
training.  ``--temporal-agg`` instead queries it on the current observation
every tick and sends the uniform average of all overlapping chunks, a
smoothing the training labels never had.  Actions are never sent late: planned
action k is due k control periods after its observation, and any action
already due when planning finishes is dropped.

``rd serve`` reaches each action one period after it is sent, so every tick
sends the action for one period ahead: chunk element k + 1 at step k, or the
temporal-aggregation blend for the next step.

``--binary-gripper`` rounds the gripper to 1 (open) or 0 (closed) before it is
sent, as teleop recorded it since raiden c25c840; the server's clip then
floors 0 to the minimum opening, exactly as teleop did.  Use it for
checkpoints trained on binary gripper labels (e.g. nr_ditM).

The policy code comes from the imitation repo on ``PYTHONPATH``; the script
declares its own dependencies, so ``uv run`` builds an isolated environment::

    PYTHONPATH=~/robot/imitation uv run scripts/dit_policy_client.py \\
        weights/3D_sim2real/checkpoints/rig_oven/m57_ct_rgb_croissant_ext24_epoch_0008.pth

The script homes the arm once at start-up and asks before starting each
episode.  Enter or Ctrl-C during an episode is the soft stop: the policy stops
sending, the server holds the last target for ``HOLD_BEFORE_HOME_S`` seconds,
then the arm returns to the rest pose over the server's usual 2 s ramp.  The
motors are never released.  Ctrl-C at a prompt quits.
"""

import argparse
import contextlib
import io
import select
import signal
import sys
import threading
import time

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
    return model, shape_meta


def preprocess_image(img: np.ndarray, size: int) -> np.ndarray:
    """640x480 RGB uint8 -> size x size RGB uint8: centre 480x480 crop, area resize."""
    h, w = img.shape[:2]
    x0 = (w - h) // 2
    return cv2.resize(img[:, x0 : x0 + h], (size, size), interpolation=cv2.INTER_AREA)


def to_model_input(obs: Observation, shape_meta) -> dict:
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

    Observation encoding stays eager; the DiT encoder and the whole DDIM
    loop (eta=0, epsilon prediction, clipped x0) replay as one CUDA graph.
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


# rd serve interpolates every joint state to this camera's capture time (server.py
# _get_reference_camera), as the converter does for the training data.
REF_CAMERA = "scene_camera"


MAX_CAPTURE_AGE_S = 0.5  # older than this is not a live frame: fall back to the fetch time
_age_warned = set()


def capture_age_s(obs) -> float | None:
    """Seconds since the observation's reference frame was captured, or None if the
    observation carries no usable capture time. rd serve stamps frames with the
    RealSense global-time device timestamp in host-epoch ns; chiral's default (and
    the mock's) is monotonic seconds, which says nothing about the wall clock."""
    try:
        ts = float(obs[REF_CAMERA].timestamp or 0.0)
    except (KeyError, TypeError, ValueError):
        ts = 0.0
    age = time.time() - ts / 1e9 if ts > 1e17 else None  # host-epoch ns only
    if age is None or not 0.0 <= age <= MAX_CAPTURE_AGE_S:
        why = "no host-epoch capture timestamp" if age is None else f"capture age {age:.3f} s"
        if why.split(" ")[0] not in _age_warned:
            _age_warned.add(why.split(" ")[0])
            print(f"from_capture: {why}; timing this observation from its fetch instead")
        return None
    return age


def run_episode(
    client,
    model,
    shape_meta,
    hz: float,
    temporal_agg: bool,
    binary_gripper: bool,
    stop: threading.Event | None = None,
    max_steps: int | None = None,
    watch_stdin: bool = True,
    from_capture: bool = False,
) -> dict:
    """Run one episode; returns counts and the per-plan obs / inference times (ms).

    ``stop`` lets the caller end the episode (Ctrl-C still sets it), ``max_steps``
    ends it after that many actions, and ``watch_stdin=False`` leaves the keyboard
    to the caller instead of ending on Enter.

    ``from_capture``: time the plan from the observation's capture (the reference
    camera's frame timestamp, which the joint states are interpolated to, as in
    the training data), not from when the client fetched it. Plan step k is due k
    periods after the capture: without temporal aggregation the steps already due
    when the plan arrives are dropped; with it, the blend for the step due next is
    sent (``steps_ahead`` = 1 + the periods elapsed since the capture). Default off:
    the observation is taken as current when fetched.
    """
    period = 1.0 / hz
    model.reset()
    plan: list[np.ndarray] = []
    obs_ms, infer_ms, age_ms, ahead_log = [], [], [], []
    sent = dropped = sent_log = 0
    t_start = t_log = time.perf_counter()
    stop = stop if stop is not None else threading.Event()
    previous_handler = signal.signal(signal.SIGINT, lambda *_: stop.set())
    try:
        while not stop.is_set() and (max_steps is None or sent < max_steps):
            if watch_stdin and enter_pressed():
                print("Enter pressed.")
                break
            if not plan:
                t0 = time.perf_counter()
                obs = client.get_obs()
                t1 = time.perf_counter()
                # The observation shows the world at t_obs: its capture, or (default) its fetch.
                age = capture_age_s(obs) if from_capture else None
                t_obs = t0 if age is None else t1 - age
                inputs = to_model_input(obs, shape_meta)
                if temporal_agg:
                    ahead = 1
                    if from_capture:  # the step due next at the expected send time
                        infer_est = np.median(infer_ms[-8:]) / 1e3 if infer_ms else 0.0
                        ahead = min(max(1, 1 + int((t1 + infer_est - t_obs) / period)), model.chunk_size - 1)
                    plan = [model.get_action(**inputs, steps_ahead=ahead)[0]]
                else:
                    chunk = model.get_action(**inputs, return_full_chunk=True)[0]
                    plan = list(chunk[1 : model.action_horizon + 1])
                t2 = time.perf_counter()
                obs_ms.append((t1 - t0) * 1e3)
                infer_ms.append((t2 - t1) * 1e3)
                age_ms.append((t1 - t_obs) * 1e3)
                if not np.isfinite(plan).all():
                    print("Non-finite action; stopping.")
                    break
                if temporal_agg:
                    # The blend was chosen for its due time; drop it only if inference ran over a period.
                    ahead_log.append(ahead)
                    late = min(int((t2 - t0) / period), len(plan))
                    next_tick = t0 + late * period
                else:
                    # plan[k] (chunk step k + 1) goes out k periods after t_obs; drop the ones already past.
                    late = min(int((t2 - t_obs) / period), len(plan))
                    next_tick = t_obs + late * period
                dropped += late
                del plan[:late]
                if not plan:
                    continue

            action = plan.pop(0)
            if binary_gripper:
                action = action.copy()
                action[6] = 1.0 if action[6] >= 0.5 else 0.0
            client.put_action(action)
            sent += 1

            now = time.perf_counter()
            if now - t_log >= 1.0:
                q = obs.proprios[PROPRIO]
                print(
                    f"t={now - t_start:6.1f}s  {(sent - sent_log) / (now - t_log):4.1f}Hz  "
                    f"obs {np.mean(obs_ms[-8:]):4.1f}ms  infer {np.mean(infer_ms[-8:]):4.1f}ms  "
                    f"dropped {dropped}  |a-q| {np.abs(action[:6] - q[:6]).max():.3f}rad  "
                    f"jaw {action[6]:.2f}"
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
    if infer_ms:
        print(
            f"\nStopped: {sent} actions in {elapsed:.1f}s ({sent / elapsed:.1f} Hz), "
            f"{dropped} late actions dropped; infer p50 {np.median(infer_ms):.1f} "
            f"p99 {np.percentile(infer_ms, 99):.1f} max {max(infer_ms):.1f}ms"
        )
    return {
        "sent": sent,
        "dropped": dropped,
        "elapsed_s": elapsed,
        "obs_ms": obs_ms,
        "infer_ms": infer_ms,
        "age_ms": age_ms,  # capture (or fetch) to fetch done, per plan
        "steps_ahead": ahead_log,  # temporal aggregation: the blend step sent, per plan
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("checkpoint", help="imitation .pth checkpoint")
    ap.add_argument("--uri", default="ws://localhost:8765", help="rd serve address")
    ap.add_argument("--hz", type=float, default=30.0, help="control rate")
    ap.add_argument(
        "--temporal-agg",
        action="store_true",
        help="replan every tick and send the average of all overlapping chunks "
        "(default: execute action_horizon steps of each full chunk, then replan)",
    )
    ap.add_argument(
        "--binary-gripper",
        action="store_true",
        help="round the gripper action to 1 (open) or 0 (closed), as teleop recorded it",
    )
    args = ap.parse_args()

    print("Loading policy...")
    model, shape_meta = load_policy(args.checkpoint)
    model.sample_actions = CudaGraphSampler(model)
    blank = Observation(
        cameras=[
            CameraInfo(c, np.eye(3), np.eye(4), np.zeros(CAMERA_SHAPE, np.uint8))
            for c in CAMERAS
        ],
        proprios={PROPRIO: np.zeros(ACTION_DIM, np.float32)},
    )
    for _ in range(3):
        model.get_action(**to_model_input(blank, shape_meta))

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
        while True:
            try:
                input(
                    "\nEnter: start the policy   "
                    "(Enter or Ctrl-C during the episode: hold 1 s, then home) "
                )
            except (KeyboardInterrupt, EOFError):
                print()
                break
            run_episode(
                client,
                model,
                shape_meta,
                args.hz,
                args.temporal_agg,
                args.binary_gripper,
            )
            stop_and_home(client)


if __name__ == "__main__":
    main()
