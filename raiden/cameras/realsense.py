"""Intel RealSense camera implementation (e.g. D405).

Recording mode opens by serial number and records to .bag/.db3 via the SDK recorder.
Playback mode opens a native recording and reads color (+ depth, if recorded) frames;
use ``RealSenseCamera.from_bag()`` to create a playback instance.

.. warning:: **For dynamic tasks, prefer ZED cameras over RealSense.**
   RealSense cameras have known synchronization limitations that make them
   less suitable for capturing fast robot motion.  See *Synchronization issues*
   below for details.

Intrinsics note
---------------
RealSense cameras store per-device calibration on-board.  get_intrinsics()
reads the color stream's Brown-Conrady coefficients directly from the SDK,
which is the same model OpenCV uses, so the values can be passed straight
to cv2 functions without any conversion.

Synchronization issues
----------------------
RealSense cameras (especially the D405) have several properties that make
multi-camera synchronization harder than with ZED cameras:

**Timestamp reliability**
    The ``global_time_enabled`` option is supposed to stamp frames with the
    host wall-clock time, but support is inconsistent across D4xx firmware
    versions.  When it is not supported, the SDK reports hardware-relative
    timestamps (milliseconds since device boot) that must be converted to
    wall-clock time.  ``_measure_clock_offset()`` compensates for this by
    measuring the offset between the first frame timestamp and
    ``time.time_ns()`` at recording start, but this is only an approximation.

**Frame extraction drain-loop bug**
    The RealSense SDK pre-buffers frames during bag playback when
    ``set_real_time(False)`` is set.  A naive "drain the queue to get the
    latest frame" loop (appropriate for live preview) will consume every
    other buffered frame, halving the apparent FPS to ~15 fps even when the
    bag was recorded at 30 fps.  ``grab()`` therefore skips the drain loop
    in playback mode (``_is_playback = True``).

**Recommendation**
    Use ZED cameras for scene and wrist cameras in dynamic manipulation
    tasks.  ZED timestamps (``sl.TIME_REFERENCE.IMAGE``) are wall-clock
    Unix nanoseconds, consistent across cameras and with the robot
    controller clock, making multi-camera alignment straightforward.
    RealSense cameras are best suited for static or slow-motion scenes
    where their close-range depth quality is the primary requirement.
"""

import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pyrealsense2 as rs

from raiden.realsense_files import REALSENSE_RECORDING_FORMATS

from .base import Camera, CameraFrame
from .realsense_receipt import (
    color_frame_receipt,
    color_stream_sensor,
    device_color_receipt,
)


def _enable_global_time(device: rs.device) -> None:
    """Enable global timestamp on all sensors that support it."""
    for sensor in device.query_sensors():
        if sensor.supports(rs.option.global_time_enabled):
            sensor.set_option(rs.option.global_time_enabled, 1)


# Color-sensor options a camera can be locked to, in the order they are set.  D435 RGB
# units: white balance in K (2800-6500, steps of 10), exposure in 100 µs, gain 0-128.
COLOR_CONTROLS = ("white_balance", "exposure", "gain")


def _wait_for_device(serial: str, timeout_s: float = 20.0) -> bool:
    """Block until *serial* is enumerated again (after a hardware reset)."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        for dev in rs.context().query_devices():
            if dev.get_info(rs.camera_info.serial_number) == str(serial):
                return True
        time.sleep(0.5)
    return False


class RealSenseCamera(Camera):
    """Intel RealSense D4xx camera – records in the configured SDK format."""

    _COLOR_W, _COLOR_H = 640, 480
    _DEPTH_W, _DEPTH_H = 640, 480

    # A camera whose firmware is older than the build librealsense expects (the
    # wrist D435 runs 5.11.1.100 against a recommended 5.17.0.10) sometimes
    # starts its color stream while the depth stream delivers nothing at all.
    # It is silent and it costs the whole episode: the bag then holds color
    # only, and playback of a bag whose depth stream is empty returns no
    # framesets, so conversion extracts zero frames.  Every pipeline start
    # therefore waits for a real depth frame, restarts if none arrives, and
    # power-cycles the camera if restarting is not enough.
    _DEPTH_START_BUDGET_S = 3.0
    _DEPTH_START_ATTEMPTS = 6
    _DEPTH_RESET_EVERY = 2  # hardware-reset after this many failed attempts

    def __init__(
        self,
        camera_name: str,
        serial_number: str,
        fps: int = 30,
        resolution: Optional[Tuple[int, int]] = None,
        crop: Optional[Tuple[int, int, int, int]] = None,
        depth: bool = True,
        color_controls: Optional[dict] = None,
        recording_format: str = "bag",
    ):
        """
        Args:
            depth: stream and record depth too.  False records color only
                (smaller bags, less USB bandwidth); frames then carry no depth.
            resolution: (width, height) for the color and depth streams.
                Default 640×480.  848×480 uses the full sensor width of a
                D435 (69° instead of ~55° horizontal FOV).
            crop: (x, y, width, height) window, in stream pixels, applied to
                every frame returned by ``get_frame()``; intrinsics are
                shifted accordingly.  Use it to re-centre a camera whose
                mount is off-axis (e.g. a wrist camera between the fingers).
            color_controls: ``{"white_balance", "exposure", "gain"}`` to lock
                the color sensor to on every pipeline start (measured by
                ``scripts/lock_color.py``), or None for auto exposure and auto
                white balance.
            recording_format: ``"bag"`` (legacy SDK/builds, the default) or
                ``"db3"`` (SDK builds with ROS2 recording support). Explicit
                because the native writer depends on SDK build options.
        """
        if recording_format not in REALSENSE_RECORDING_FORMATS:
            raise ValueError("RealSense recording_format must be 'bag' or 'db3'")
        self._name = camera_name
        self._serial = serial_number
        self._fps = fps
        self._depth = depth
        self._recording_format = recording_format
        self._color_controls = dict(color_controls) if color_controls else None
        self._color_applied: Optional[dict] = None
        if resolution is not None:
            self._COLOR_W, self._COLOR_H = int(resolution[0]), int(resolution[1])
            self._DEPTH_W, self._DEPTH_H = self._COLOR_W, self._COLOR_H
        self._crop = tuple(int(v) for v in crop) if crop is not None else None
        self._pipeline: Optional[rs.pipeline] = None
        self._config: Optional[rs.config] = None
        self._profile: Optional[rs.pipeline_profile] = None
        self._latest_frames = None
        # Actual negotiated color resolution (set after pipeline.start()).
        self._color_w: int = self._COLOR_W
        self._color_h: int = self._COLOR_H
        # Offset (ns) to convert RS SDK timestamps to wall-clock:
        # wall_ns = frame.get_timestamp() * 1_000_000 + _clock_offset_ns
        # Measured once at start_recording() by comparing the first frame
        # timestamp to time.time_ns().  None until start_recording() is called.
        self._clock_offset_ns: Optional[int] = None
        self._clock_mapping_receipt: Optional[dict] = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return self._name

    @property
    def serial_number(self) -> str:
        return str(self._serial)

    @property
    def recording_extension(self) -> str:
        return self._recording_format

    @property
    def color_controls(self) -> Optional[dict]:
        """Color-sensor values read back at the last pipeline start; None on auto."""
        return self._color_applied

    # ------------------------------------------------------------------
    # Live recording
    # ------------------------------------------------------------------

    def _start_pipeline(self, cfg: "rs.config") -> None:
        """Start the pipeline and record the negotiated color resolution."""
        self._profile = self._pipeline.start(cfg)
        self._config = cfg
        color_stream = self._profile.get_stream(
            rs.stream.color
        ).as_video_stream_profile()
        self._color_w = color_stream.width()
        self._color_h = color_stream.height()

    def _apply_color_controls(self, device: rs.device) -> None:
        """Lock white balance, exposure and gain, or hand both back to auto.

        The options live on the device, not the pipeline: they outlast the
        process, and ``hardware_reset()`` puts them back to auto.  So they are
        set after every pipeline start, and a camera without controls is set
        to auto explicitly rather than left as another tool set it.
        """
        sensor = color_stream_sensor(device)
        lock = self._color_controls is not None
        if lock:
            unsupported = [
                key for key in COLOR_CONTROLS
                if not sensor.supports(getattr(rs.option, key))
            ]
            if unsupported:
                raise ValueError(
                    f"[{self._name}] color sensor does not support requested controls: "
                    + ", ".join(unsupported)
                )
            # D405 color shares the depth sensor's microsecond exposure control.
            # Separate D435 RGB sensors retain the historical 100 µs units.
            exposure_unit_us = 1 if sensor.is_depth_sensor() else 100
            if self._color_controls["exposure"] * exposure_unit_us > 1_000_000 / self._fps:
                raise ValueError(
                    f"[{self._name}] exposure {self._color_controls['exposure']} "
                    f"({exposure_unit_us} µs) is longer than a frame at {self._fps} fps"
                )
        for option in (rs.option.enable_auto_exposure, rs.option.enable_auto_white_balance):
            if sensor.supports(option):
                sensor.set_option(option, 0 if lock else 1)
        if not lock:
            self._color_applied = None
            return
        for key in COLOR_CONTROLS:
            sensor.set_option(getattr(rs.option, key), float(self._color_controls[key]))
        applied = {
            key: round(sensor.get_option(getattr(rs.option, key)))
            for key in COLOR_CONTROLS
        }
        if any(abs(applied[k] - self._color_controls[k]) > 0.5 for k in COLOR_CONTROLS):
            raise RuntimeError(
                f"[{self._name}] color controls {self._color_controls} read back as {applied}"
            )
        self._color_applied = applied

    def _stream_config(self, color_w: int, color_h: int) -> "rs.config":
        """Color, plus depth unless it is turned off."""
        cfg = rs.config()
        cfg.enable_device(self._serial)
        cfg.enable_stream(rs.stream.color, color_w, color_h, rs.format.bgr8, self._fps)
        if self._depth:
            cfg.enable_stream(
                rs.stream.depth, self._DEPTH_W, self._DEPTH_H, rs.format.z16, self._fps
            )
        return cfg

    def _depth_streams(self, budget_s: float) -> bool:
        """True once a frameset carrying depth arrives within *budget_s*."""
        deadline = time.time() + budget_s
        while time.time() < deadline:
            try:
                ok, frames = self._pipeline.try_wait_for_frames(timeout_ms=200)
            except RuntimeError:
                return False
            if ok and frames.get_depth_frame():
                return True
        return False

    def _start_verified(self, make_config) -> None:
        """Start the pipeline and do not return until depth really streams.

        *make_config* is called once per attempt rather than reused, because a
        config that records to a bag must drop the partial file it wrote on the
        failed attempt.  Raises RuntimeError once the attempts are spent, which
        is the point: a color-only bag recorded under ``"depth": true`` is worse
        than a session that refuses to start.
        """
        for attempt in range(1, self._DEPTH_START_ATTEMPTS + 1):
            self._start_pipeline(make_config())
            _enable_global_time(self._profile.get_device())
            self._apply_color_controls(self._profile.get_device())
            if not self._depth or self._depth_streams(self._DEPTH_START_BUDGET_S):
                if self._depth and attempt > 1:
                    print(f"  [{self._name}] depth streaming after {attempt} attempts")
                return

            device = self._profile.get_device()
            self._pipeline.stop()
            print(
                f"  [{self._name}] no depth frames "
                f"(attempt {attempt}/{self._DEPTH_START_ATTEMPTS})"
            )
            if attempt % self._DEPTH_RESET_EVERY == 0:
                print(f"  [{self._name}] power-cycling the camera ...")
                device.hardware_reset()
                if not _wait_for_device(self._serial):
                    raise RuntimeError(
                        f"[{self._name}] camera {self._serial} did not come back "
                        "after a hardware reset — replug it"
                    )
                self._pipeline = rs.pipeline()

        raise RuntimeError(
            f"[{self._name}] depth was requested but the camera streams none. "
            f"Check the firmware (see {self._serial} in `rd calibrate`/rs-fw-update) "
            'or set "depth": false for this camera to record color only.'
        )

    def open(self) -> None:
        self._pipeline = rs.pipeline()
        self._start_verified(lambda: self._stream_config(self._COLOR_W, self._COLOR_H))
        print(
            f"  [{self._name}] opened: "
            f"BGR8 {self._color_w}×{self._color_h} @ {self._fps}fps"
            + ("" if self._depth else " (no depth)")
        )

    def close(self) -> None:
        if self._pipeline:
            self._pipeline.stop()
            self._pipeline = None
            self._profile = None

    def start_recording(self, path: Path) -> None:
        """Restart pipeline with the configured native recorder attached."""
        path = Path(path)
        if path.suffix != f".{self._recording_format}":
            raise ValueError(
                f"[{self._name}] recording_format={self._recording_format!r} requires "
                f"a .{self._recording_format} output path, got {path.name!r}"
            )
        if self._pipeline:
            self._pipeline.stop()

        def bag_config() -> "rs.config":
            path.unlink(missing_ok=True)  # a retry must not append to a partial bag
            cfg = self._stream_config(self._COLOR_W, self._COLOR_H)
            cfg.enable_record_to_file(str(path))
            return cfg

        self._start_verified(bag_config)
        print(
            f"  [{self._name}] recording: "
            f"BGR8 {self._color_w}×{self._color_h} @ {self._fps}fps"
            + ("" if self._depth else " (no depth)")
        )
        self._measure_clock_offset()

    def _measure_clock_offset(self) -> None:
        """Grab the first available frame and compute RS-to-wall-clock offset.

        ``_clock_offset_ns`` is the additive correction such that:
            wall_ns = int(frame.get_timestamp() * 1_000_000) + _clock_offset_ns

        This works whether or not global_time_enabled is supported by the
        device: if it is supported the stored timestamps are already wall-clock
        and the offset contains host reception delay; if not, it approximates
        the difference between the RealSense hardware clock and the system
        clock at recording start. Transport/queue delay and clock drift remain
        unmeasured, so this is not an exposure synchronization guarantee.
        """
        # This is a host-receipt measurement, not an exposure-time measurement.
        # Keep its provenance in the existing camera receipt; native per-frame
        # timestamps, domains and counters are already retained by the bag.
        self._clock_offset_ns = None
        self._clock_mapping_receipt = {
            "method": "single_host_receipt_offset",
            "host_wait_start_ns": time.time_ns(),
            "interpretation": (
                "One received color frame mapped to host wall time; includes "
                "unknown transport/queue delay and does not measure clock error "
                "or synchronize exposures."
            ),
        }
        try:
            frames = self._pipeline.wait_for_frames(timeout_ms=2000)
            wall_ns = time.time_ns()
            color_frame = frames.get_color_frame()
            if color_frame:
                rs_ns = int(color_frame.get_timestamp() * 1_000_000)
                self._clock_offset_ns = wall_ns - rs_ns
                self._clock_mapping_receipt.update({
                    "host_receive_ns": wall_ns,
                    "sdk_timestamp_ms": float(color_frame.get_timestamp()),
                    "frame_timestamp_domain": str(color_frame.get_frame_timestamp_domain()),
                    "offset_ns": self._clock_offset_ns,
                })
            else:
                self._clock_mapping_receipt["error"] = "No color frame received"
        except Exception as error:
            self._clock_mapping_receipt["error"] = str(error)

    def stop_recording(self) -> None:
        """Restart pipeline without recorder to finalize the bag file."""
        if self._pipeline:
            self._pipeline.stop()
        # Rebuild config without enable_record_to_file — there is no
        # disable_record_and_playback() in the pyrealsense2 Python bindings.
        # Use the same color resolution that was negotiated in start_recording().
        cfg = self._stream_config(self._color_w, self._color_h)
        self._config = cfg
        self._profile = self._pipeline.start(cfg)
        _enable_global_time(self._profile.get_device())
        self._apply_color_controls(self._profile.get_device())

    def grab(self) -> bool:
        try:
            frames = self._pipeline.wait_for_frames(timeout_ms=500)
            if not getattr(self, "_is_playback", False):
                # Drain any additional buffered frames so we always use the most
                # recent one. Bag-file writing overhead can cause the queue to
                # accumulate, making wait_for_frames return increasingly stale frames.
                # Skip this in playback mode: set_real_time(False) pre-buffers the
                # next frame, so draining would skip every other frame (30fps → 15fps).
                while True:
                    ok, newer = self._pipeline.try_wait_for_frames(timeout_ms=0)
                    if not ok:
                        break
                    frames = newer
            self._latest_frames = frames
            return True
        except RuntimeError:
            return False

    def get_frame(self) -> CameraFrame:
        if self._latest_frames is None:
            raise RuntimeError("No frames available. Call grab() first.")

        frames = self._latest_frames
        align = getattr(self, "_align", None)
        if align is not None:
            frames = align.process(frames)

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()  # empty when depth is off

        color = np.asanyarray(color_frame.get_data())  # BGR uint8
        depth = None
        if depth_frame:
            depth_raw = np.asanyarray(depth_frame.get_data())  # uint16 raw units
            # Playback can report the D435's 1 mm sensor default for a D405
            # recording whose frames retain their correct 0.1 mm units.
            # Read the scale from the actual frame, including after alignment.
            depth_units = depth_frame.get_units()
            if not np.isfinite(depth_units) or depth_units <= 0:
                raise RuntimeError(f"[{self._name}] invalid depth frame units: {depth_units}")
            depth = depth_raw.astype(np.float32) * np.float32(depth_units)
        if self._crop is not None:
            x, y, w, h = self._crop
            color = color[y : y + h, x : x + w]
            if depth is not None:
                depth = depth[y : y + h, x : x + w]

        # RealSense timestamp is in milliseconds
        timestamp_ns = int(color_frame.get_timestamp() * 1_000_000)

        return CameraFrame(color=color, depth=depth, timestamp_ns=timestamp_ns)

    def get_intrinsics(self) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
        """Return on-device calibrated intrinsics for the color stream.

        RealSense stores per-unit calibration on-board.  The distortion model
        is Brown-Conrady (identical to OpenCV's), so coefficients are directly
        usable with cv2 functions.
        """
        if self._profile is None:
            raise RuntimeError("Camera is not open. Call open() first.")

        color_stream = self._profile.get_stream(
            rs.stream.color
        ).as_video_stream_profile()
        intr = color_stream.get_intrinsics()
        ppx, ppy, width, height = self._cropped_intrinsics(intr)

        camera_matrix = np.array(
            [[intr.fx, 0.0, ppx], [0.0, intr.fy, ppy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        # Brown-Conrady: [k1, k2, p1, p2, k3]
        dist_coeffs = np.array(intr.coeffs[:5], dtype=np.float64)
        image_size = (width, height)
        return camera_matrix, dist_coeffs, image_size

    def _cropped_intrinsics(self, intr) -> Tuple[float, float, int, int]:
        """Return (ppx, ppy, width, height) after applying the crop window."""
        if self._crop is None:
            return intr.ppx, intr.ppy, intr.width, intr.height
        x, y, w, h = self._crop
        if x < 0 or y < 0 or x + w > intr.width or y + h > intr.height:
            raise ValueError(
                f"[{self._name}] crop {self._crop} exceeds the "
                f"{intr.width}×{intr.height} stream"
            )
        return intr.ppx - x, intr.ppy - y, w, h

    # ------------------------------------------------------------------
    # Playback from .bag file
    # ------------------------------------------------------------------

    @classmethod
    def from_bag(
        cls,
        camera_name: str,
        bag_path: Path,
        crop: Optional[Tuple[int, int, int, int]] = None,
    ) -> "RealSenseCamera":
        """Open a native .bag or .db3 file for frame-by-frame extraction.

        The bag holds the full stream; *crop* (same meaning as in
        ``__init__``) is applied to the frames read back from it.
        """
        recording_format = Path(bag_path).suffix.removeprefix(".")
        if recording_format not in REALSENSE_RECORDING_FORMATS:
            raise ValueError("RealSense playback requires a .bag or .db3 file")
        cam = cls.__new__(cls)
        cam._name = camera_name
        cam._serial = ""
        cam._fps = 30
        cam._crop = tuple(int(v) for v in crop) if crop is not None else None
        cam._latest_frames = None
        cam._color_controls = None
        cam._color_applied = None
        cam._recording_format = recording_format

        # A file-only context also keeps offline conversion independent of
        # physical camera discovery and its udev monitor.
        cam._playback_context = rs.context({"device-mask": int(rs.product_line.sw_only)})
        cam._pipeline = rs.pipeline(cam._playback_context)
        cam._config = rs.config()
        rs.config.enable_device_from_file(
            cam._config, str(bag_path), repeat_playback=False
        )
        cam._profile = cam._pipeline.start(cam._config)

        # Non-realtime playback so we don't drop frames
        playback = cam._profile.get_device().as_playback()
        playback.set_real_time(False)

        # Align depth to color so they share the same pixel grid; a color-only bag has none.
        cam._depth = any(
            s.stream_type() == rs.stream.depth for s in cam._profile.get_streams()
        )
        cam._align = rs.align(rs.stream.color) if cam._depth else None
        cam._is_playback = True

        return cam

    def get_camera_info(self) -> dict:
        """Return camera intrinsics and metadata as a plain dict."""
        if not self._profile:
            return {}
        color_stream = self._profile.get_stream(
            rs.stream.color
        ).as_video_stream_profile()
        intr = color_stream.get_intrinsics()
        ppx, ppy, width, height = self._cropped_intrinsics(intr)
        receipt = device_color_receipt(self._profile.get_device())
        device_info = receipt["device"]
        latest = getattr(self, "_latest_frames", None)
        frame = latest.get_color_frame() if latest is not None else None
        return {
            "serial_number": device_info.get("serial_number") or self._serial,
            "model": device_info.get("name") or "RealSense",
            "recording_format": self._recording_format,
            "native_width": intr.width,
            "native_height": intr.height,
            "stream_format": str(color_stream.format()),
            "distortion_model": str(intr.model),
            "policy_crop": list(self._crop) if self._crop is not None else None,
            "color_receipt": receipt,
            "last_color_frame_receipt": color_frame_receipt(frame) if frame else None,
            "clock_mapping_receipt": getattr(self, "_clock_mapping_receipt", None),
            "fps": self._fps,
            "width": width,
            "height": height,
            "fx": intr.fx,
            "fy": intr.fy,
            "cx": ppx,
            "cy": ppy,
            "k1": intr.coeffs[0],
            "k2": intr.coeffs[1],
            "p1": intr.coeffs[2],
            "p2": intr.coeffs[3],
            "k3": intr.coeffs[4],
        }
