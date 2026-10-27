import json
import tarfile

import pytest

from raiden.sim.bundle import package_sim_episodes, read_episode_list, sha256


@pytest.fixture
def recording(tmp_path):
    assets = tmp_path / "assets"
    scene = assets / "example/runtime_scene.json"
    scene.parent.mkdir(parents=True)
    scene.write_text('{"robot":{"usd_path":"robot.usda"}}\n')
    robot = scene.parent / "robot.usda"
    robot.write_bytes(b"#usda 1.0\n")
    episode = tmp_path / "raw/task/0007"
    episode.mkdir(parents=True)
    (episode / "metadata.json").write_text(
        json.dumps({"task_name": "task", "complete": True, "duration_s": 20.0})
    )
    (episode / "robot_data.npz").write_bytes(b"recorded binary data")
    metadata = {
        "backend": "isaac",
        "physics_dt": 1 / 60,
        "physics_substeps": 4,
        "scene": {
            "descriptor": str(scene),
            "sha256": sha256(scene),
            "teleop": {
                "schema": "isaac-scene-teleop/1",
                "physics_dt": 1 / 60,
                "contact_profile": {
                    "content": {"solref": [0.01, 1]},
                    "file": "old/contact.json",
                },
            },
        },
        "asset_input_sha256": {str(robot): sha256(robot)},
    }
    (episode / "sim_metadata.json").write_text(json.dumps(metadata))
    return episode, assets, metadata


def test_recording_bytes_assets_portable_configuration_and_checksums(
    recording, tmp_path
):
    episode, assets, _ = recording
    out = tmp_path / "release"
    result = package_sim_episodes([episode], out, asset_root=assets)
    assert result["original_episode_ids"] == ["0007"]
    with tarfile.open(out / "sim/teleop/train/0007.tar") as archive:
        assert (
            archive.extractfile("task/0007/robot_data.npz").read()
            == (episode / "robot_data.npz").read_bytes()
        )
        assert (
            archive.extractfile("task/0007/sim_metadata.json").read()
            == (episode / "sim_metadata.json").read_bytes()
        )
    with tarfile.open(out / "sim/environment/scene_assets.tar") as archive:
        assert set(archive.getnames()) == {
            "example/runtime_scene.json",
            "example/robot.usda",
        }
    task = json.loads((out / "sim/environment/config/task.json").read_text())
    assert task["contact_profile"] == "contact.json"
    assert json.loads((out / "sim/environment/config/contact.json").read_text())[
        "solref"
    ] == [0.01, 1]
    for item in result["files"]:
        assert sha256(out / item["path"]) == item["sha256"]
    assert (
        json.loads((out / "sim/environment/entrypoints.json").read_text())["scene"]
        == "assets/example/runtime_scene.json"
    )


def test_refuses_changed_recorded_assets_before_creating_output(recording, tmp_path):
    episode, assets, _ = recording
    (assets / "example/robot.usda").write_text("changed")
    with pytest.raises(ValueError, match="differs from recorded"):
        package_sim_episodes([episode], tmp_path / "release", asset_root=assets)
    assert not (tmp_path / "release").exists()


def test_refuses_mixed_runtime_profiles(recording, tmp_path):
    import shutil

    episode, assets, metadata = recording
    second = episode.with_name("0008")
    shutil.copytree(episode, second)
    metadata["physics_dt"] = 1 / 120
    (second / "sim_metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="one recorded runtime profile"):
        package_sim_episodes([episode, second], tmp_path / "release", asset_root=assets)


def test_refuses_secret_attachment_even_at_nonhidden_destination(recording, tmp_path):
    episode, assets, _ = recording
    secret = tmp_path / ".env.huggingface"
    secret.write_text("example credential")
    with pytest.raises(ValueError, match="explicit data/configuration"):
        package_sim_episodes(
            [episode],
            tmp_path / "release",
            asset_root=assets,
            attachments={"metadata/credentials.txt": secret},
        )


@pytest.mark.parametrize(
    "bad", [".env.huggingface", "runtime.py", ".secrets/token.json"]
)
def test_refuses_credentials_and_source_in_recordings(recording, tmp_path, bad):
    episode, assets, _ = recording
    (episode / bad).parent.mkdir(parents=True, exist_ok=True)
    (episode / bad).write_text("not data")
    with pytest.raises(ValueError, match="unexpected recording file"):
        package_sim_episodes([episode], tmp_path / "release", asset_root=assets)


def test_rechecks_asset_bytes_after_archiving(recording, tmp_path, monkeypatch):
    from raiden.sim import bundle

    episode, assets, _ = recording
    original = bundle._archive

    def changed_asset(target, entries):
        original(target, entries)
        if target.name == "scene_assets.tar":
            (assets / "example/robot.usda").write_text("changed concurrently")

    monkeypatch.setattr(bundle, "_archive", changed_asset)
    with pytest.raises(RuntimeError, match="scene asset changed"):
        package_sim_episodes([episode], tmp_path / "release", asset_root=assets)


def test_refuses_nonportable_absolute_descriptor_paths(recording, tmp_path):
    episode, assets, metadata = recording
    scene = assets / "example/runtime_scene.json"
    scene.write_text(
        json.dumps({"robot": {"usd_path": str(assets / "example/robot.usda")}})
    )
    metadata["scene"]["sha256"] = sha256(scene)
    (episode / "sim_metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="must be relative"):
        package_sim_episodes([episode], tmp_path / "release", asset_root=assets)


def test_extracted_scene_resolves_inside_bundle(recording, tmp_path):
    pytest.importorskip("pxr")
    episode, assets, _ = recording
    out = tmp_path / "release"
    package_sim_episodes([episode], out, asset_root=assets, verify_assets=True)
    verification = json.loads((out / "sim/environment/verification.json").read_text())
    assert verification["status"] == "passed"
    assert verification["resolved_dependency_files"] == ["example/robot.usda"]


@pytest.mark.parametrize(
    "destination", ["../escape.json", "/absolute.json", ".hidden/file.json"]
)
def test_refuses_unsafe_attachment_destinations(recording, tmp_path, destination):
    episode, assets, _ = recording
    with pytest.raises(ValueError, match="unsafe attachment"):
        package_sim_episodes(
            [episode],
            tmp_path / "release",
            asset_root=assets,
            attachments={destination: episode / "metadata.json"},
        )


def test_refuses_incomplete_episode_and_existing_output(recording, tmp_path):
    episode, assets, _ = recording
    out = tmp_path / "release"
    out.mkdir()
    with pytest.raises(FileExistsError):
        package_sim_episodes([episode], out, asset_root=assets)
    (episode / "metadata.json").write_text(json.dumps({"complete": False}))
    with pytest.raises(ValueError, match="completed Isaac"):
        package_sim_episodes([episode], tmp_path / "new", asset_root=assets)


def test_selection_resolves_relative_paths_and_rejects_duplicates(recording, tmp_path):
    episode, _, _ = recording
    selection = tmp_path / "selection.txt"
    selection.write_text("# chosen episodes\nraw/task/0007\n")
    assert read_episode_list(selection) == [episode]
    selection.write_text("raw/task/0007\nraw/task/0007\n")
    with pytest.raises(ValueError, match="duplicates"):
        read_episode_list(selection)
