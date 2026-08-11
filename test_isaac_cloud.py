"""Unit tests for isaac_cloud: provisioning fail-fast + snapshot persistence.

These lock in behavior that was validated live against real Vast instances
(docs/VAST_TIMEOUT_EXPERIMENT_RESULTS.md and the snapshot integration run):
status_msg failure signatures, auth-denial recovery timing, append-only
snapshot semantics, and project resolution precedence.
"""

import sys
import time
from dataclasses import replace
from pathlib import Path

import pytest
import typer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import isaac_cloud as ic


@pytest.fixture()
def config(tmp_path):
    toml = tmp_path / "config.toml"
    toml.write_text(
        """
[defaults]
provider = "vast"

[persistence]
enabled = true
s3_uri = "s3://bkt/base"
project = "alpha"
keep_last = 3
"""
    )
    return ic.load_app_config(toml)


@pytest.fixture()
def clock(monkeypatch):
    """Controllable time: clock.now advances via clock.tick(); sleep advances it."""

    class Clock:
        now = 1_000_000.0

        def tick(self, seconds):
            Clock.now += seconds

    c = Clock()
    monkeypatch.setattr(ic.time, "time", lambda: c.now)
    monkeypatch.setattr(ic.time, "sleep", lambda s: c.tick(max(s, 13)))
    return c


def vast_info(status="loading", status_msg="", ssh=None, label="", instance_id="X"):
    return ic.InstanceInfo(
        provider="vast",
        instance_id=instance_id,
        status=status,
        label=label,
        ssh=ssh,
        raw={"status_msg": status_msg},
    )


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def test_config_persistence_fields(config):
    assert config.persistence_enabled is True
    assert config.persistence_s3_uri == "s3://bkt/base"
    assert config.persistence_project == "alpha"
    assert config.persistence_keep_last == 3


def test_config_defaults(tmp_path):
    cfg = ic.load_app_config(tmp_path / "missing.toml")
    assert cfg.persistence_project == ic.DEFAULT_PERSISTENCE_PROJECT
    assert cfg.persistence_keep_last == ic.DEFAULT_PERSISTENCE_KEEP_LAST
    assert cfg.persistence_enabled is False
    assert cfg.isaac_version == ic.DEFAULT_ISAAC_VERSION


def test_config_isaac_section(tmp_path):
    toml = tmp_path / "config.toml"
    toml.write_text("[isaac]\nversion = \"7.1.0\"\nagent = false\ncurobo = true\n")
    cfg = ic.load_app_config(toml)
    assert cfg.isaac_version == "7.1.0"
    assert cfg.agent_enabled is False
    assert cfg.curobo_enabled is True


def test_config_project_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("ISAAC_CLOUD_PROJECT", "from-env")
    cfg = ic.load_app_config(tmp_path / "missing.toml")
    assert cfg.persistence_project == "from-env"


# ---------------------------------------------------------------------------
# S3 layout helpers and project names
# ---------------------------------------------------------------------------


def test_parse_s3_uri():
    assert ic.parse_s3_uri("s3://bkt/a/b/") == ("bkt", "a/b/")
    assert ic.parse_s3_uri("s3://bkt") == ("bkt", "")


def test_snapshot_prefix_normalizes_slash(config):
    assert ic.snapshot_prefix_uri(config, "alpha") == "s3://bkt/base/projects/alpha/snapshots/"


def test_base_uri_validation(config):
    for bad in (None, "", "http://x", "s3://"):
        with pytest.raises(ic.IsaacCloudError):
            ic.build_persistence_base_uri(replace(config, persistence_s3_uri=bad))


@pytest.mark.parametrize("name", ["default", "arm-grasping", "user_a.projB", "9lives"])
def test_valid_project_names(name):
    assert ic.validate_project_name(name) == name


@pytest.mark.parametrize("name", ["", "../x", "a b", "-lead", ".lead", "a/b", None])
def test_invalid_project_names(name):
    with pytest.raises(ic.IsaacCloudError):
        ic.validate_project_name(name)


def test_resolve_project_precedence(config):
    labelled = vast_info(label="project=labelled")
    tagged = ic.InstanceInfo(
        "aws", "i-1", "running", "", None,
        {"Tags": [{"Key": ic.AWS_TAG_PROJECT, "Value": "tagged"}]},
    )
    assert ic.resolve_project(config, labelled, "explicit") == "explicit"
    assert ic.resolve_project(config, labelled) == "labelled"
    assert ic.resolve_project(config, tagged) == "tagged"
    assert ic.resolve_project(config, vast_info(label="unrelated")) == "alpha"
    assert ic.resolve_project(config) == "alpha"
    with pytest.raises(ic.IsaacCloudError):
        ic.resolve_project(config, vast_info(label="project=../etc"))


# ---------------------------------------------------------------------------
# Vast provisioning failure detection (constants measured live; see results doc)
# ---------------------------------------------------------------------------


class RecordingProvider(ic.Provider):
    name = "vast"

    def __init__(self, config):
        super().__init__(config)
        self.attached = 0

    def attach_ssh_key(self, instance_id):
        self.attached += 1


@pytest.fixture()
def monitor(config, clock):
    return ic.VastProvisionMonitor(RecordingProvider(config))


@pytest.mark.parametrize(
    "msg",
    [
        "docker login failed!",
        "Error response from daemon: manifest for x not found: manifest unknown",
        "pull access denied for nvcr.io/nvidia/isaac-sim",
        "unauthorized: authentication required",
        "tar: /x: No space left on device",
    ],
)
def test_fatal_status_msgs_doom_immediately(monitor, msg):
    with pytest.raises(ic.ProvisioningDoomed):
        monitor.check_status(vast_info(status_msg=msg))


def test_healthy_pull_msgs_are_fine(monitor):
    for msg in ["bfa54bd09267: Verifying Checksum", "#8 19.46 Setting up util-linux", ""]:
        monitor.check_status(vast_info(status_msg=msg))  # must not raise


def test_generic_daemon_error_requires_persistence(monitor, clock):
    flaky = vast_info(status_msg="Error response from daemon: toomanyrequests")
    monitor.check_status(flaky)  # first sighting: tolerated
    clock.tick(ic.VAST_DAEMON_ERROR_FATAL_S - 1)
    monitor.check_status(flaky)  # still within grace
    clock.tick(2)
    with pytest.raises(ic.ProvisioningDoomed):
        monitor.check_status(flaky)


def test_generic_daemon_error_resets_on_recovery(monitor, clock):
    flaky = vast_info(status_msg="Error response from daemon: hiccup")
    monitor.check_status(flaky)
    clock.tick(ic.VAST_DAEMON_ERROR_FATAL_S)
    monitor.check_status(vast_info(status_msg="layer: Download complete"))  # recovered
    monitor.check_status(flaky)  # a fresh sighting starts a fresh grace period


def test_auth_denial_attaches_then_dooms(monitor, clock):
    info = vast_info(status="running")
    monitor.auth_denied(info)
    assert monitor.provider.attached == 0
    clock.tick(ic.VAST_SSH_DENIED_ATTACH_S + 1)
    monitor.auth_denied(info)
    assert monitor.provider.attached == 1  # re-attach attempted exactly once
    clock.tick(ic.VAST_SSH_DENIED_GIVE_UP_S)
    with pytest.raises(ic.ProvisioningDoomed):
        monitor.auth_denied(info)
    assert monitor.provider.attached == 1


def test_auth_reset_clears_denial_window(monitor, clock):
    info = vast_info(status="running")
    monitor.auth_denied(info)
    clock.tick(ic.VAST_SSH_DENIED_GIVE_UP_S + 1)
    monitor.auth_reset()
    monitor.auth_denied(info)  # fresh window: must not raise


# ---------------------------------------------------------------------------
# wait_for_ssh loop
# ---------------------------------------------------------------------------


class ScriptedProvider(ic.Provider):
    name = "vast"

    def __init__(self, config, infos):
        super().__init__(config)
        self.infos = list(infos)
        self.polls = 0

    def get(self, instance_id):
        info = self.infos[min(self.polls, len(self.infos) - 1)]
        self.polls += 1
        return info

    def provision_monitor(self):
        return ic.VastProvisionMonitor(self)


def test_wait_for_ssh_success(config, clock, monkeypatch):
    target = ic.SshTarget(host="1.2.3.4", port=22, user="root")
    prov = ScriptedProvider(config, [vast_info(), vast_info(status="running", ssh=target)])
    monkeypatch.setattr(ic, "run_ssh", lambda *a, **k: "")
    info = ic.wait_for_ssh(config, prov, "X", timeout_seconds=120)
    assert info.ssh == target


def test_wait_for_ssh_bails_on_fatal_status(config, clock):
    prov = ScriptedProvider(config, [vast_info(status_msg="docker login failed!")])
    with pytest.raises(ic.ProvisioningDoomed):
        ic.wait_for_ssh(config, prov, "X", timeout_seconds=3600)
    assert prov.polls == 1  # detected on the very first poll


def test_wait_for_ssh_dooms_persistent_auth_denial(config, clock, monkeypatch):
    target = ic.SshTarget(host="1.2.3.4", port=22, user="root")
    prov = ScriptedProvider(config, [vast_info(status="running", ssh=target)])

    def deny(*a, **k):
        raise ic.IsaacCloudError("SSH command failed (255): Permission denied (publickey).")

    monkeypatch.setattr(ic, "run_ssh", deny)
    with pytest.raises(ic.ProvisioningDoomed):
        ic.wait_for_ssh(config, prov, "X", timeout_seconds=3600)


def test_wait_for_ssh_timeout_reports_status_msg(config, clock):
    prov = ScriptedProvider(config, [vast_info(status_msg="layer: Downloading")])
    with pytest.raises(ic.IsaacCloudError) as excinfo:
        ic.wait_for_ssh(config, prov, "X", timeout_seconds=60)
    assert not isinstance(excinfo.value, ic.ProvisioningDoomed)
    assert "layer: Downloading" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Snapshot store logic (S3 faked at the aws-CLI seam)
# ---------------------------------------------------------------------------


@pytest.fixture()
def fake_s3(monkeypatch):
    objects: dict[str, int] = {}  # key -> size

    def handler(config, args, *, timeout_seconds=3600):
        if args[0] == "s3api" and args[1] == "list-objects-v2":
            prefix = args[args.index("--prefix") + 1]
            delimiter = "--delimiter" in args
            contents, common = [], set()
            for key, size in sorted(objects.items()):
                if not key.startswith(prefix):
                    continue
                rest = key[len(prefix):]
                if delimiter and "/" in rest:
                    common.add(prefix + rest.split("/", 1)[0] + "/")
                    continue
                contents.append({"Key": key, "Size": size})
            out = {}
            if contents:
                out["Contents"] = contents
            if common:
                out["CommonPrefixes"] = [{"Prefix": p} for p in sorted(common)]
            import json

            return json.dumps(out)
        if args[0] == "s3" and args[1] == "rm":
            del objects[ic.parse_s3_uri(args[2])[1]]
            return ""
        raise AssertionError(f"unexpected aws call: {args}")

    monkeypatch.setattr(ic, "run_local_aws", handler)
    return objects


def put(objects, project, name, size=100):
    objects[f"base/projects/{project}/snapshots/{name}"] = size


def test_list_snapshots_sorted_and_filtered(config, fake_s3):
    put(fake_s3, "alpha", "2026-08-11T02-00-00.000000Z.tar.gz")
    put(fake_s3, "alpha", "2026-08-10T09-00-00.000000Z.tar.gz")
    put(fake_s3, "alpha", "notes.txt")  # non-snapshot object ignored
    put(fake_s3, "beta", "2026-08-11T03-00-00.000000Z.tar.gz")  # other project
    snaps = ic.list_snapshots(config, "alpha")
    assert [s["name"] for s in snaps] == [
        "2026-08-10T09-00-00.000000Z.tar.gz",
        "2026-08-11T02-00-00.000000Z.tar.gz",
    ]
    assert snaps[-1]["uri"] == (
        "s3://bkt/base/projects/alpha/snapshots/2026-08-11T02-00-00.000000Z.tar.gz"
    )


def test_list_projects(config, fake_s3):
    put(fake_s3, "alpha", "a.tar.gz")
    put(fake_s3, "beta", "b.tar.gz")
    assert ic.list_projects(config) == ["alpha", "beta"]
    fake_s3.clear()
    assert ic.list_projects(config) == []


def test_prune_keeps_newest(config, fake_s3):
    for hour in range(5):
        put(fake_s3, "alpha", f"2026-08-11T0{hour}-00-00.000000Z.tar.gz")
    removed = ic.prune_snapshots(config, "alpha", keep_last=3)
    assert removed == 2
    names = [s["name"] for s in ic.list_snapshots(config, "alpha")]
    assert names == [f"2026-08-11T0{hour}-00-00.000000Z.tar.gz" for hour in (2, 3, 4)]
    assert ic.prune_snapshots(config, "alpha", keep_last=0) == 0  # disabled


def reachable(monkeypatch):
    monkeypatch.setattr(ic, "check_tcp_connectivity", lambda *a, **k: True)


def test_pull_with_no_snapshots_starts_fresh(config, fake_s3, monkeypatch):
    reachable(monkeypatch)
    prov = ic.VastProvider(config)
    info = vast_info(status="running", ssh=ic.SshTarget("1.2.3.4", 22, "root"))
    msg = ic.snapshot_pull(config, prov, info, "alpha")
    assert "starting fresh" in msg


def test_pull_unknown_snapshot_name_lists_available(config, fake_s3, monkeypatch):
    reachable(monkeypatch)
    put(fake_s3, "alpha", "2026-08-11T02-00-00.000000Z.tar.gz")
    prov = ic.VastProvider(config)
    info = vast_info(status="running", ssh=ic.SshTarget("1.2.3.4", 22, "root"))
    with pytest.raises(ic.IsaacCloudError) as excinfo:
        ic.snapshot_pull(config, prov, info, "alpha", snapshot="2020-01-01T00-00-00.000000Z")
    assert "2026-08-11T02-00-00.000000Z.tar.gz" in str(excinfo.value)


def test_push_skips_empty_project_dir(config, monkeypatch):
    reachable(monkeypatch)
    monkeypatch.setattr(ic, "run_ssh", lambda *a, **k: "")  # find returns nothing
    prov = ic.VastProvider(config)
    info = vast_info(status="running", ssh=ic.SshTarget("1.2.3.4", 22, "root"))
    msg = ic.snapshot_push(config, prov, info, "alpha")
    assert "skipping snapshot" in msg


def test_persistence_requires_reachable_ssh(config, monkeypatch):
    monkeypatch.setattr(ic, "check_tcp_connectivity", lambda *a, **k: False)
    prov = ic.VastProvider(config)
    info = vast_info(status="running", ssh=ic.SshTarget("1.2.3.4", 22, "root"))
    with pytest.raises(ic.IsaacCloudError):
        ic.snapshot_push(config, prov, info, "alpha")
    with pytest.raises(ic.IsaacCloudError):
        ic.snapshot_pull(config, prov, info, "alpha")


# ---------------------------------------------------------------------------
# Vast row parsing and CLI error decorator
# ---------------------------------------------------------------------------


def test_vast_to_info_port_mapping(config):
    prov = ic.VastProvider(config)
    row = {
        "id": 1,
        "actual_status": "running",
        "label": "project=alpha",
        "public_ipaddr": "5.6.7.8",
        "ports": {"22/tcp": [{"HostIp": "0.0.0.0", "HostPort": "30720"}]},
    }
    info = prov._to_info(row)
    assert info.ssh == ic.SshTarget(host="5.6.7.8", port=30720, user="root")
    assert info.label == "project=alpha"
    assert prov._to_info({"id": 2, "actual_status": "loading"}).ssh is None


def test_vast_stop_start_surface_failures(config, monkeypatch):
    """stop/start tolerate vastai's plain-text output but must not swallow
    real failures — a silently-failed stop keeps billing."""
    prov = ic.VastProvider(config)

    def fail(cmd, **kwargs):
        raise ic.IsaacCloudError("vastai stop instance failed: host unreachable")

    monkeypatch.setattr(ic, "run_cli", fail)
    with pytest.raises(ic.IsaacCloudError):
        prov.stop("123")
    with pytest.raises(ic.IsaacCloudError):
        prov.start("123")


def test_cli_errors_decorator():
    @ic.cli_errors
    def boom():
        raise ic.IsaacCloudError("nope")

    with pytest.raises(typer.Exit) as excinfo:
        boom()
    assert excinfo.value.exit_code == 1
