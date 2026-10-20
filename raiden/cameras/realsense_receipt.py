"""Read-only RealSense provenance; native ISP controls are not color transforms."""

from __future__ import annotations

import importlib.metadata
import time

import pyrealsense2 as rs

COLOR_OPTIONS = (
    "enable_auto_exposure",
    "enable_auto_white_balance",
    "exposure",
    "gain",
    "white_balance",
    "gamma",
    "brightness",
    "contrast",
    "saturation",
    "hue",
    "sharpness",
    "backlight_compensation",
    "power_line_frequency",
    "auto_exposure_priority",
    "global_time_enabled",
)
COLOR_FRAME_FIELDS = (
    "actual_exposure",
    "gain_level",
    "auto_exposure",
    "manual_white_balance",
    "white_balance",
    "auto_white_balance_temperature",
    "gamma",
    "brightness",
    "contrast",
    "saturation",
    "hue",
    "sharpness",
    "backlight_compensation",
    "power_line_frequency",
    "low_light_compensation",
)
DEVICE_FIELDS = (
    "name",
    "serial_number",
    "asic_serial_number",
    "firmware_version",
    "recommended_firmware_version",
    "product_id",
    "product_line",
    "usb_type_descriptor",
    "physical_port",
)


def device_color_receipt(device) -> dict:
    """Snapshot identity/options using getters only, including missing readbacks.

    Playback ranges describe frozen recorded values, not hardware capabilities.
    No gain→ISO, WB→RGB, gamma→power-law, or frame-metadata unit mapping is made.
    """
    result = {
        "schema": "raiden.realsense_color_receipt/1",
        "host_readback_time_ns": time.time_ns(),
        "device": {},
        "color_options": {},
        "readback_errors": {},
        "interpretation": "Native SDK/UVC values; no radiometric transform inferred.",
    }
    try:
        result["pyrealsense2_version"] = importlib.metadata.version("pyrealsense2")
    except importlib.metadata.PackageNotFoundError:
        result["pyrealsense2_version"] = None
    try:
        result["is_playback"] = bool(device.is_playback())
    except Exception as error:
        result["is_playback"] = None
        result["readback_errors"]["is_playback"] = str(error)
    result["option_range_scope"] = (
        "frozen recorded snapshot; not live hardware range"
        if result["is_playback"]
        else "device readback at receipt time"
    )
    for name in DEVICE_FIELDS:
        key = getattr(rs.camera_info, name, None)
        try:
            result["device"][name] = (
                device.get_info(key)
                if key is not None and device.supports(key)
                else None
            )
        except Exception as error:
            result["device"][name] = None
            result["readback_errors"][name] = str(error)
    try:
        sensor = device.first_color_sensor()
    except Exception as error:
        result["readback_errors"]["color_sensor"] = str(error)
        return result
    for name in COLOR_OPTIONS:
        option = getattr(rs.option, name, None)
        item = {"supported": False}
        result["color_options"][name] = item
        try:
            item["supported"] = option is not None and bool(sensor.supports(option))
            if not item["supported"]:
                continue
            item["value"] = float(sensor.get_option(option))
            item["description"] = sensor.get_option_description(option)
            limits = sensor.get_option_range(option)
            item["range"] = {
                key: float(getattr(limits, key))
                for key in ("min", "max", "step", "default")
            }
        except Exception as error:
            item["error"] = str(error)
    # D435/D435i RGB exposure is the UVC absolute-exposure control (100 us).
    # This conversion is intentionally restricted to these verified USB models.
    model = (
        result["device"].get("name", "").upper() if result["device"].get("name") else ""
    )
    exposure = result["color_options"].get("exposure", {}).get("value")
    if (
        model in ("INTEL REALSENSE D435", "INTEL REALSENSE D435I")
        and exposure is not None
    ):
        result["rgb_exposure_seconds_from_uvc_control"] = exposure * 0.0001
        result["rgb_exposure_unit_basis"] = (
            "D435/D435i RGB UVC exposure absolute: 100 microseconds per unit; not depth exposure units."
        )
    return result


def color_frame_receipt(frame) -> dict:
    """Preserve native per-frame values and explicitly identify unavailable fields."""
    result = {
        "frame_number": int(frame.get_frame_number()),
        "sdk_timestamp_ms": float(frame.get_timestamp()),
        "timestamp_domain": str(frame.get_frame_timestamp_domain()),
        "native_metadata": {},
        "unavailable_fields": [],
    }
    for name in COLOR_FRAME_FIELDS:
        key = getattr(rs.frame_metadata_value, name, None)
        try:
            if key is not None and frame.supports_frame_metadata(key):
                result["native_metadata"][name] = int(frame.get_frame_metadata(key))
            else:
                result["unavailable_fields"].append(name)
        except Exception as error:
            result.setdefault("readback_errors", {})[name] = str(error)
    return result
