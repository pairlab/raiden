"""ZED camera implementation using the pyzed SDK.

Recording mode  : opens by serial number, or from a network stream sent by
                  ``scripts/zed_stream.py``, and records to SVO2 (H264; a stereo
                  stream is written as received).
                  Depth is NOT computed during recording (depth_mode=NONE)
                  for lower CPU usage; the raw stereo pair is stored instead.
Playback mode   : opens an SVO2 file, computes depth (NEURAL) for each frame.
                  Use ZedCamera.from_svo() to create a playback instance.

Mono cameras (ZED X One) use ``sl.CameraOne`` and have no depth.
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pyzed.sl as sl

from .base import Camera, CameraFrame

# scripts/zed_stream.py's default --bitrate (kbit/s).
_STREAM_KBPS = 20000


class ZedCamera(Camera):
    """ZED camera – records to .svo2

    ``stream`` is the sender's ``"host:port"``; a streamed camera comes at the
    sender's resolution.  ``resolution`` is an ``sl.RESOLUTION`` name for a local
    camera (GMSL cameras have no HD720).
    """

    def __init__(
        self,
        camera_name: str,
        serial_number: int,
        fps: int = 30,
        stream: Optional[str] = None,
        mono: bool = False,
        resolution: str = "HD720",
    ):
        self._name = camera_name
        self._serial = serial_number
        self._fps = fps
        self._stream = stream
        self._mono = mono
        self._resolution = resolution
        self._camera = sl.CameraOne() if mono else sl.Camera()
        self._is_open = False
        self._image = sl.Mat()
        self._depth = sl.Mat()
        self._has_depth = False  # True only in playback mode

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
        return "svo2"

    # ------------------------------------------------------------------
    # Live recording
    # ------------------------------------------------------------------

    def open(self) -> None:
        """Open live camera. Depth is disabled to reduce recording overhead."""
        init_params = sl.InitParametersOne() if self._mono else sl.InitParameters()
        if self._stream:
            host, port = self._stream.rsplit(":", 1)
            init_params.set_from_stream(host, int(port))
        else:
            init_params.set_from_serial_number(self._serial)
        init_params.camera_resolution = getattr(sl.RESOLUTION, self._resolution)
        init_params.camera_fps = self._fps
        if not self._mono:
            init_params.depth_mode = sl.DEPTH_MODE.NONE
        init_params.coordinate_units = sl.UNIT.METER

        status = self._camera.open(init_params)
        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(
                f"Failed to open ZED camera '{self._name}' "
                f"(serial: {self._serial}, stream: {self._stream}): {status}"
            )
        self._is_open = True
        # The port, not the serial, picks a streamed camera: check it is the right one.
        serial = self._camera.get_camera_information().serial_number
        if serial != self._serial:
            self.close()
            raise RuntimeError(
                f"ZED camera '{self._name}': {self._stream} streams camera {serial}, "
                f"not {self._serial}"
            )
        self._has_depth = False

    def close(self) -> None:
        if self._is_open:
            self._camera.close()
            self._is_open = False

    def start_recording(self, path: Path) -> None:
        """Enable SVO2 recording. Every subsequent grab() writes a frame."""
        params = sl.RecordingParameters()
        params.compression_mode = sl.SVO_COMPRESSION_MODE.H264
        params.video_filename = str(path)
        # Storing a mono stream as received segfaults the SDK when several streams record:
        # re-encode it at the sender's bitrate.
        params.transcode_streaming_input = self._mono
        if self._mono:
            params.bitrate = _STREAM_KBPS
        status = self._camera.enable_recording(params)
        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(
                f"Failed to start SVO2 recording for '{self._name}': {status}"
            )

    def stop_recording(self) -> None:
        self._camera.disable_recording()

    def grab(self) -> bool:
        """Grab next frame (blocks until ready at camera FPS)."""
        return self._camera.grab() == sl.ERROR_CODE.SUCCESS

    def get_frame(self) -> CameraFrame:
        """Retrieve color (and depth if in playback mode) from last grab."""
        self._camera.retrieve_image(self._image, sl.VIEW.LEFT)
        color = self._image.get_data()[:, :, :3].copy()  # drop alpha

        depth: Optional[np.ndarray] = None
        if self._has_depth:
            self._camera.retrieve_measure(self._depth, sl.MEASURE.DEPTH)
            raw = self._depth.get_data().copy()
            depth = np.where(np.isfinite(raw), raw, 0.0).astype(np.float32)

        timestamp_ns = self._camera.get_timestamp(
            sl.TIME_REFERENCE.IMAGE
        ).get_nanoseconds()

        return CameraFrame(color=color, depth=depth, timestamp_ns=timestamp_ns)

    # ------------------------------------------------------------------
    # Playback from SVO2 file
    # ------------------------------------------------------------------

    @classmethod
    def from_svo(
        cls,
        camera_name: str,
        svo_path: Path,
        compute_sdk_depth: bool = True,
        mono: bool = False,
    ) -> "ZedCamera":
        """Open an SVO2 file for frame-by-frame extraction.

        Parameters
        ----------
        compute_sdk_depth : bool
            If True (default) depth is computed by the ZED SDK (NEURAL_LIGHT).
            Set to False when depth is not needed; this skips the SDK depth
            pass for lower CPU/GPU load.
        mono : bool
            The file was recorded by a mono camera (ZED X One): no depth.
        """
        cam = cls.__new__(cls)
        cam._name = camera_name
        cam._serial = 0
        cam._fps = 30
        cam._mono = mono
        cam._camera = sl.CameraOne() if mono else sl.Camera()
        cam._is_open = False
        cam._image = sl.Mat()
        cam._depth = sl.Mat()
        cam._has_depth = compute_sdk_depth and not mono

        init_params = sl.InitParametersOne() if mono else sl.InitParameters()
        init_params.set_from_svo_file(str(svo_path))
        init_params.svo_real_time_mode = False  # process all frames
        init_params.coordinate_units = sl.UNIT.METER
        if not mono:
            init_params.depth_mode = (
                sl.DEPTH_MODE.NEURAL_LIGHT if compute_sdk_depth else sl.DEPTH_MODE.NONE
            )
            init_params.depth_minimum_distance = 0.01

        status = cam._camera.open(init_params)
        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Failed to open SVO2 '{svo_path}': {status}")

        cam._is_open = True
        return cam

    def get_intrinsics(self) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
        """Return ZED SDK factory-calibrated intrinsics for the left lens."""
        info = self._camera.get_camera_information()
        cal = info.camera_configuration.calibration_parameters
        cal = cal if self._mono else cal.left_cam
        res = info.camera_configuration.resolution

        camera_matrix = np.array(
            [[cal.fx, 0.0, cal.cx], [0.0, cal.fy, cal.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        # ZED disto order: k1, k2, p1, p2, k3
        dist_coeffs = np.array(
            [cal.disto[0], cal.disto[1], cal.disto[2], cal.disto[3], cal.disto[4]],
            dtype=np.float64,
        )
        image_size = (res.width, res.height)
        return camera_matrix, dist_coeffs, image_size

    def get_total_frames(self) -> int:
        """Total frame count (valid for SVO2 playback only)."""
        return self._camera.get_svo_number_of_frames()

    def get_frame_timestamp_ns(self) -> int:
        """Timestamp of the most recently grabbed frame in nanoseconds (ZED hardware clock)."""
        return self._camera.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds()

    def get_current_timestamp_ns(self) -> int:
        """Current host time in nanoseconds via the ZED SDK.

        On the same clock as ``get_frame_timestamp_ns()``, so robot observations
        recorded with this value can be directly interpolated against camera
        frame timestamps.  For a streamed camera it is this host's clock while
        the frame timestamps are the sender's: the two hosts' clocks must agree.
        """
        return self._camera.get_timestamp(sl.TIME_REFERENCE.CURRENT).get_nanoseconds()

    def get_camera_info(self) -> dict:
        """Return camera intrinsics and metadata as a plain dict."""
        info = self._camera.get_camera_information()
        cal = info.camera_configuration.calibration_parameters
        cal = cal if self._mono else cal.left_cam
        res = info.camera_configuration.resolution
        return {
            "serial_number": str(self._serial) or "unknown",
            "model": str(info.camera_model),
            "fps": self._fps,
            "width": res.width,
            "height": res.height,
            "fx": cal.fx,
            "fy": cal.fy,
            "cx": cal.cx,
            "cy": cal.cy,
            "k1": cal.disto[0],
            "k2": cal.disto[1],
            "p1": cal.disto[2],
            "p2": cal.disto[3],
            "k3": cal.disto[4],
        }
