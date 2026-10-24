"""Resolve CAN channel names without importing robot drivers or touching hardware."""

import json
from collections.abc import Iterable, Mapping
from pathlib import Path

from raiden._config import ROBOT_CONFIG

DEFAULT_CAN_CHANNELS = {
    "follower_l": "can_follower_l",
    "follower_r": "can_follower_r",
    "leader_l": "can_leader_l",
    "leader_r": "can_leader_r",
}


def load_can_channels(
    enabled_arms: Iterable[str] = DEFAULT_CAN_CHANNELS,
    *,
    path: str | Path = ROBOT_CONFIG,
    overrides: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Merge a role-to-channel JSON object onto the historical channel names.

    A missing file preserves the defaults. An explicit override replaces the file
    input, which also makes tests independent of the machine's configuration.
    Only enabled arms must have distinct channels; every supplied key and value
    is validated before returning the resolved mapping.
    """
    if overrides is None:
        try:
            with Path(path).open() as stream:
                configured = json.load(stream)
        except FileNotFoundError:
            configured = {}
    else:
        configured = overrides
    if not isinstance(configured, Mapping):
        raise ValueError("robot CAN configuration must be a role-to-channel object")

    channels = DEFAULT_CAN_CHANNELS.copy()
    for arm, channel in configured.items():
        if arm not in channels:
            raise ValueError(f"Unknown robot CAN role: {arm!r}")
        if (
            not isinstance(channel, str)
            or not channel
            or channel.strip() != channel
            or any(character.isspace() for character in channel)
        ):
            raise ValueError(
                f"CAN channel for {arm} must be a nonempty string without whitespace"
            )
        channels[arm] = channel

    used = {}
    for arm in enabled_arms:
        if arm not in channels:
            raise ValueError(f"Unknown enabled robot CAN role: {arm!r}")
        channel = channels[arm]
        if channel in used:
            raise ValueError(
                "Enabled robot CAN channels must be distinct: "
                f"{used[channel]} and {arm} use {channel!r}"
            )
        used[channel] = arm
    return channels
