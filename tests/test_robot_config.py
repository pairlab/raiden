"""CAN role configuration is validated without loading robot drivers."""

import json

import pytest

from raiden.robot_config import DEFAULT_CAN_CHANNELS, load_can_channels


def test_missing_file_preserves_historical_channels(tmp_path):
    assert load_can_channels(path=tmp_path / "absent.json") == DEFAULT_CAN_CHANNELS


def test_rig_mapping_merges_defaults_and_override_bypasses_file(tmp_path):
    config = tmp_path / "robot.json"
    config.write_text(json.dumps({"follower_l": "can_left", "follower_r": "can_right"}))
    channels = load_can_channels(path=config)
    assert channels == {
        **DEFAULT_CAN_CHANNELS,
        "follower_l": "can_left",
        "follower_r": "can_right",
    }
    config.write_text("malformed JSON")
    assert load_can_channels(path=config, overrides={}) == DEFAULT_CAN_CHANNELS


@pytest.mark.parametrize(
    "configured",
    [
        [],
        None,
        {"left": "can_left"},
        {"follower_l": None},
        {"follower_l": 12},
        {"follower_l": ""},
        {"follower_l": " can_left"},
        {"follower_l": "can left"},
    ],
)
def test_invalid_file_fails_before_hardware_is_involved(tmp_path, configured):
    config = tmp_path / "robot.json"
    config.write_text(json.dumps(configured))
    with pytest.raises(ValueError):
        load_can_channels(path=config)


def test_duplicates_rejected_only_for_enabled_roles():
    configured = {"follower_l": "can_shared", "follower_r": "can_shared"}
    with pytest.raises(ValueError, match="must be distinct"):
        load_can_channels(["follower_l", "follower_r"], overrides=configured)
    assert (
        load_can_channels(["follower_l"], overrides=configured)["follower_l"]
        == "can_shared"
    )
    with pytest.raises(ValueError, match="Unknown enabled"):
        load_can_channels(["follower_middle"], overrides={})


def test_malformed_json_is_not_silently_treated_as_defaults(tmp_path):
    config = tmp_path / "robot.json"
    config.write_text("{")
    with pytest.raises(ValueError):
        load_can_channels(path=config)
