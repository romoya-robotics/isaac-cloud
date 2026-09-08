"""Unit tests for isaac_cloud: provisioning fail-fast + snapshot persistence.

These lock in behavior that was validated live against real Vast instances
(docs/VAST_TIMEOUT_EXPERIMENT_RESULTS.md and the snapshot integration run):
status_msg failure signatures, auth-denial recovery timing, append-only
snapshot semantics, and project resolution precedence.
"""

import sys
import json
import base64
import time
import subprocess
import threading
from http.client import HTTPConnection
from dataclasses import replace
from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

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
    assert cfg.lab_enabled is False
    assert cfg.lab_ref == ic.DEFAULT_ISAAC_LAB_REF

    assert cfg.webrtc_enabled is False


def test_config_isaac_section(tmp_path):
    toml = tmp_path / "config.toml"
    toml.write_text(
        "[isaac]\nversion = \"7.1.0\"\nagent = false\ncurobo = true\n"
        "lab = true\nlab_ref = \"v4.0.0\"\n"
    )
    cfg = ic.load_app_config(toml)
    assert cfg.isaac_version == "7.1.0"
    assert cfg.agent_enabled is False
    assert cfg.curobo_enabled is True
    assert cfg.lab_enabled is True
    assert cfg.lab_ref == "v4.0.0"


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


# ---------------------------------------------------------------------------
# GUI stack: generated scripts, catalog ranking, tunnel port remapping
# ---------------------------------------------------------------------------


def _index(text, needle):
    i = text.find(needle)
    assert i >= 0, f"missing: {needle!r}"
    return i


def test_gui_stack_script_is_ordered_x_before_kit(config):
    """The kit must never start before the X display answers (a kit that raced
    Xvfb answers on 8226 but never maps its window)."""
    script = ic.build_gui_stack_script(config)
    xvfb = _index(script, "setsid Xvfb $X_DISPLAY")
    xwait = _index(script, "x_up && break")
    vulkan = _index(script, "# 2. Vulkan presentation preflight")
    websockify = _index(script, "setsid websockify")
    x11vnc = _index(script, "setsid nohup /root/x11vnc_loop.sh")
    kit = _index(script, "start_gui_kit\n    deadline")
    assert xvfb < xwait < vulkan < websockify < x11vnc < kit
    assert "GUI_STACK_VULKAN_PRESENT_FAILED" in script
    # readiness = mapped window AND agent port, never 8226 alone
    assert "Map State: IsViewable" in script
    assert 'w=$(gui_window)' in script
    assert "GUI_STACK_READY" in script


def test_gui_stack_script_process_guards(config):
    script = ic.build_gui_stack_script(config)
    # binary-name guards, not -f patterns that also appear in a start command
    assert "pgrep -x Xvfb" in script
    assert "pkill -x websockify" in script
    assert 'pgrep -f "Xvf[b]' not in script
    assert 'pgrep -f "websockif[y]' not in script
    # supervised x11vnc with -noxdamage
    assert "-noxdamage" in script
    assert "while true; do\n    x11vnc -display" in script
    # the headless kit is stopped before the GUI kit starts
    assert "stopping the headless streaming kit" in script
    # settings flow from config
    assert "GUI_RES=1920x1080" in script
    assert "isaacsim.code_editor.python_server" in script
    headless_only = ic.build_gui_stack_script(replace(config, agent_enabled=False))
    assert 'KIT_EXTRA_ARGS=""' in headless_only
    assert "AGENT_ENABLED=0" in headless_only


def test_gui_stack_install_wraps_in_quoted_heredoc(config):
    script = ic.build_gui_stack_install_script(config)
    assert script.startswith("#!/bin/bash\ncat > /root/gui_stack.sh <<'GUI_STACK_EOF'\n")
    assert script.rstrip().endswith("exec bash /root/gui_stack.sh up")
    assert script.count("GUI_STACK_EOF") == 2


def test_probe_script_includes_gui_checks_conditionally(config):
    probe = ic.build_container_probe_script(config)
    assert "gui_check()" in probe
    assert "if [ -x /root/gui_stack.sh ] || pgrep -x Xvfb" in probe
    assert probe.rstrip().endswith("exit 0")
    assert "port 8226" in probe.replace("{", "").replace("}", "") or "8226" in probe


def test_headless_script_side_loads_and_kills_previous_kit(config):
    script = ic.build_isaac_container_launch_script(config)
    assert "ensure_nvidia_userland()" in script
    assert 'pkill -f "[k]it/kit"' in script
    assert "runheadless.sh -v --enable isaacsim.code_editor.python_server" in script


def test_video_tools_installed_on_every_launch_path(config):
    """ffmpeg/ffprobe/libx264 (robot video capture) must be installed by the
    headless launch, the GUI stack, and reported by the status probe."""
    for script in (
        ic.build_isaac_container_launch_script(config),
        ic.build_gui_stack_script(config),
    ):
        assert "ensure_video_tools()" in script
        assert "apt-get install -y -qq ffmpeg libx264-dev" in script
        assert "ensure_video_tools || true" in script
        # readiness is the encoder actually being usable, not just the binary
        assert 'ffmpeg -hide_banner -encoders 2>/dev/null | grep -q "libx264"' in script
        assert "command -v ffprobe" in script
    headless = ic.build_isaac_container_launch_script(config)
    assert _index(headless, "ensure_video_tools || true") < _index(headless, "runheadless.sh")
    probe = ic.build_container_probe_script(config)
    assert "video_tools_ready()" in probe
    assert 'echo "video_tools: ready' in probe


def test_driver_major():
    assert ic.driver_major("580.95.05") == 580
    assert ic.driver_major("590.10") == 590
    assert ic.driver_major(None) == 0
    assert ic.driver_major("garbage") == 0


def test_catalog_prefers_new_drivers_for_gui(config, monkeypatch):
    offers = [
        {"id": 1, "driver_version": "580.95.05", "dph_total": 0.30, "reliability2": 0.999},
        {"id": 2, "driver_version": "590.10.01", "dph_total": 0.35, "reliability2": 0.999},
        {"id": 3, "driver_version": "575.64", "dph_total": 0.20, "reliability2": 0.5},
        {"id": 4, "driver_version": "595.00", "dph_total": 0.40, "reliability2": 0.999},
    ]
    monkeypatch.setattr(ic, "run_vastai_json", lambda args, **kw: list(offers))
    # headless: cheapest first (reliability filter still applies)
    ids = [o["id"] for o in ic.VastProvider(config).catalog()]
    assert ids == [1, 2, 4]
    # gui: driver >= 590 first, price order kept within each group
    gui = ic.VastProvider(replace(config, gui_enabled=True)).catalog()
    assert [o["id"] for o in gui] == [2, 4, 1]


def test_tunnel_forwards_remap_local_ports():
    default = ic.tunnel_forwards()
    assert (8226, 8226) in default and (6080, 6080) in default and (8554, 8554) in default
    remapped = dict(
        (remote, local) for local, remote in ic.tunnel_forwards({6080: 16080, 8226: 18226})
    )
    assert remapped == {8226: 18226, 8554: 8554, 6080: 16080}


def test_ssh_base_args_keepalive(config):
    args = ic.ssh_base_args(config, ic.SshTarget(host="h", port=22, user="root"))
    assert "ServerAliveInterval=15" in args


def test_remote_gui_stack_installed(config, monkeypatch):
    info = vast_info(status="running", ssh=ic.SshTarget(host="h", port=22, user="root"))
    monkeypatch.setattr(ic, "run_ssh", lambda *a, **k: "yes")
    assert ic.remote_gui_stack_installed(config, info) is True
    monkeypatch.setattr(ic, "run_ssh", lambda *a, **k: "no")
    assert ic.remote_gui_stack_installed(config, info) is False

# WebRTC: separate SSH signaling and mapped UDP media on Vast.

@pytest.fixture()
def webrtc_info():
    return ic.InstanceInfo(
        "vast", "123", "running", "", ic.SshTarget("203.0.113.42", 30022, "root"),
        {"public_ipaddr": "203.0.113.42", "ports": {
            "22/tcp": [{"HostPort": "30022"}],
            "47999/udp": [{"HostPort": "31234"}],
        }},
    )


def test_webrtc_config(tmp_path):
    path = tmp_path / "stream.toml"
    path.write_text("[webrtc]\nenabled = true\n")
    assert ic.load_app_config(path).webrtc_enabled


def test_webrtc_uses_mapped_media_port(webrtc_info):
    assert ic.webrtc_connection(webrtc_info) == {
        "signalingServer": "127.0.0.1", "signalingPort": 49100,
        "mediaServer": "203.0.113.42", "mediaPort": 31234,
    }


@pytest.mark.parametrize("raw", [
    {}, {"ports": {"47999/tcp": [{"HostPort": "31234"}]}},
    {"public_ipaddr": "$(touch /tmp/oops)", "ports": {"47999/udp": [{"HostPort": "31234"}]}},
    {"public_ipaddr": "203.0.113.42", "ports": {"47999/udp": [{"HostPort": "70000"}]}},
    {"public_ipaddr": "203.0.113.42", "ports": {"47999/udp": [{}]}},
])
def test_webrtc_rejects_missing_or_invalid_mapping(webrtc_info, raw):
    with pytest.raises(ic.IsaacCloudError):
        ic.webrtc_connection(replace(webrtc_info, raw=raw))


@pytest.mark.parametrize("enabled", [False, True])
def test_vast_requests_only_udp_for_webrtc(config, monkeypatch, enabled):
    calls = []
    cfg = replace(config, ngc_api_key="fake", webrtc_enabled=enabled)
    prov = ic.VastProvider(cfg)

    def vast(args, **kwargs):
        calls.append(args)
        return {"success": True, "new_contract": 123}

    monkeypatch.setattr(ic, "run_vastai_json", vast)
    monkeypatch.setattr(prov, "_ensure_account_ssh_key", lambda: None)
    prov.launch(offer_id="42")
    args = calls[0]
    if enabled:
        assert args[args.index("--env") + 1] == "-p 47999:47999/udp"
    else:
        assert "--env" not in args
    assert "49100" not in " ".join(args)
    assert "8226" not in " ".join(args)


@pytest.mark.parametrize("options,overrides,message", [
    (["--webrtc", "--gui"], {}, "Choose --webrtc --no-gui"),
    (["--webrtc"], {"vast_whole_machine": False}, "whole_machine"),
])
def test_webrtc_invalid_launch_fails_before_rental(config, monkeypatch, options, overrides, message):
    monkeypatch.setattr(ic, "_config", lambda: replace(config, **overrides))
    monkeypatch.setattr(ic.VastProvider, "launch", lambda *a, **k: pytest.fail("rented an instance"))
    monkeypatch.setattr(ic.AwsProvider, "launch", lambda *a, **k: pytest.fail("rented an instance"))
    result = CliRunner().invoke(ic.app, ["launch", *options])
    assert result.exit_code == 1, result.output
    assert message in result.output


def test_webrtc_launch_overrides_config_gui(config, monkeypatch, webrtc_info):
    cfg = replace(config, gui_enabled=True, persistence_enabled=False)
    monkeypatch.setattr(ic, "_config", lambda: cfg)
    monkeypatch.setattr(ic.VastProvider, "launch", lambda *a, **k: webrtc_info)
    monkeypatch.setattr(ic, "wait_for_ssh", lambda *a, **k: webrtc_info)
    monkeypatch.setattr(ic, "wait_for_container", lambda *a: None)
    setups = []
    monkeypatch.setattr(ic, "setup_isaac", lambda c, i: setups.append(c))
    result = CliRunner().invoke(ic.app, ["launch", "--webrtc"])
    assert result.exit_code == 0, result.output
    assert setups[0].webrtc_enabled and not setups[0].gui_enabled
    assert "isaac_cloud.py webrtc-view" in result.output
    assert "SSH is the only ingress" not in result.output


def test_webrtc_resume_remembers_launch_flag(config, monkeypatch, webrtc_info):
    monkeypatch.setattr(ic, "_config", lambda: config)
    monkeypatch.setattr(ic.VastProvider, "get", lambda *a: webrtc_info)
    monkeypatch.setattr(ic.VastProvider, "start", lambda *a: None)
    monkeypatch.setattr(ic, "wait_for_ssh", lambda *a: webrtc_info)
    monkeypatch.setattr(ic, "wait_for_container", lambda *a: None)
    monkeypatch.setattr(ic, "remote_gui_stack_installed", lambda *a: pytest.fail("WebRTC must skip GUI autodetection"))
    setups = []
    monkeypatch.setattr(ic, "setup_isaac", lambda c, i: setups.append(c))
    result = CliRunner().invoke(ic.app, ["resume", "--instance-id", "123"])
    assert result.exit_code == 0, result.output
    assert setups[0].webrtc_enabled
    assert not setups[0].gui_enabled


def test_webrtc_resume_rejects_gui_before_start(config, monkeypatch, webrtc_info):
    monkeypatch.setattr(ic, "_config", lambda: config)
    monkeypatch.setattr(ic.VastProvider, "get", lambda *a: replace(webrtc_info, status="stopped"))
    monkeypatch.setattr(ic.VastProvider, "start", lambda *a: pytest.fail("started an incompatible instance"))
    result = CliRunner().invoke(ic.app, ["resume", "--instance-id", "123", "--gui"])
    assert result.exit_code == 1, result.output
    assert "Choose --webrtc --no-gui" in result.output


def test_webrtc_relay_is_udp_only_and_restricted():
    script = ic.build_webrtc_relay_script("198.51.100.10")
    assert "range=198.51.100.10/32" in script
    assert "UDP4:127.0.0.1:47998" in script
    assert "UDP4-LISTEN:47999" in script
    assert "TCP" not in script
    assert "kill -0" in script
    with pytest.raises(ic.IsaacCloudError):
        ic.build_webrtc_relay_script("0.0.0.0/0")
    with pytest.raises(ic.IsaacCloudError):
        ic.build_webrtc_relay_script("1.2.3.4; echo injected")


def test_webrtc_shell_syntax(config):
    assert "--/exts/omni.services.livestream.session/quitOnSessionEnded=false" in (
        ic.build_isaac_container_launch_script(config)
    )
    for script in [ic.build_webrtc_check_script(), ic.build_webrtc_relay_script("198.51.100.10"),
                   ic.build_container_probe_script(config), ic.build_isaac_container_launch_script(config),
                   ic.build_lab_install_script(config.lab_ref)]:
        result = subprocess.run(["bash", "-n"], input=script, text=True, capture_output=True)
        assert result.returncode == 0, result.stderr


def test_webrtc_relay_uses_ssh_client_ip(config, webrtc_info, monkeypatch):
    scripts = []
    monkeypatch.setattr(ic, "run_ssh", lambda *a, **k: "198.51.100.10")
    monkeypatch.setattr(ic, "run_ssh_script", lambda c, t, s, **k: scripts.append(s) or "1234")
    assert ic.start_webrtc_relay(config, webrtc_info, None) == 1234
    assert "range=198.51.100.10/32" in scripts[0]


@pytest.mark.parametrize("local_signal_port", [49100, 49101])
def test_webrtc_tunnel_refreshes_mapping_and_keeps_local_bind(config, webrtc_info, monkeypatch, local_signal_port):
    calls, mapped = [], []
    second = replace(webrtc_info, raw={"public_ipaddr": "203.0.113.43", "ports": {
        "47999/udp": [{"HostPort": "32123"}],
    }})
    infos = iter([webrtc_info, second])
    prov = ic.VastProvider(config)
    monkeypatch.setattr(prov, "get", lambda _: next(infos))
    monkeypatch.setattr(ic.time, "sleep", lambda _: None)

    def run(args, **kwargs):
        calls.append(args)
        return subprocess.CompletedProcess(args, 1 if len(calls) == 1 else 130)

    monkeypatch.setattr(ic.subprocess, "run", run)
    ic.run_supervised_tunnel(config, prov, "123", local_ports={49100: local_signal_port},
                            service_ports=[(49100, "signal")],
                            on_connect=lambda i: mapped.append(ic.webrtc_connection(i)))
    assert [c["mediaPort"] for c in mapped] == [31234, 32123]
    assert f"127.0.0.1:{local_signal_port}:127.0.0.1:49100" in calls[0]
    assert "47999" not in " ".join(calls[0])


def test_webrtc_http_serves_assets_and_updated_config(tmp_path):
    (tmp_path / "index.html").write_text("viewer only")
    (tmp_path / "assets").mkdir()
    current = {"mediaPort": 31234}
    server = ic.make_webrtc_http_server(0, lambda: current, tmp_path)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    conn = HTTPConnection("127.0.0.1", server.server_port, timeout=2)
    try:
        conn.request("GET", "/")
        response = conn.getresponse()
        assert response.status == 200
        assert response.read() == b"viewer only"
        current = {"mediaPort": 32123}
        conn.request("GET", "/connection.json")
        response = conn.getresponse()
        assert response.getheader("Cache-Control") == "no-store"
        assert b"32123" in response.read()
        conn.request("GET", "/connection.json", headers={"Host": "untrusted.example"})
        response = conn.getresponse()
        assert response.status == 403
        response.read()
        conn.request("HEAD", "/connection.json", headers={"Host": "untrusted.example"})
        response = conn.getresponse()
        assert response.status == 403
        assert response.read() == b""
        conn.request("HEAD", "/connection.json")
        response = conn.getresponse()
        assert response.status == 200
        assert response.getheader("Content-Type") == "application/json"
        assert int(response.getheader("Content-Length")) > 0
        assert response.read() == b""
        conn.request("GET", "/assets/")
        response = conn.getresponse()
        assert response.status == 404
        response.read()
        with pytest.raises(ic.IsaacCloudError, match="Cannot start viewer"):
            ic.make_webrtc_http_server(server.server_port, lambda: current, tmp_path)
    finally:
        conn.close()
        server.shutdown()
        server.server_close()
        thread.join()


def test_webrtc_missing_viewer_build(tmp_path):
    with pytest.raises(ic.IsaacCloudError, match="Build the browser viewer first"):
        ic.make_webrtc_http_server(0, lambda: {}, tmp_path)


@pytest.mark.parametrize("failure", [KeyboardInterrupt(), OSError("tunnel failed"),
                                     ic.IsaacCloudError("relay failed")])
@pytest.mark.parametrize("command", ["webrtc-view", "view", "webrtc"])
def test_webrtc_command_cleans_up_relay(config, monkeypatch, webrtc_info, tmp_path, failure, command):
    (tmp_path / "index.html").write_text("viewer")
    monkeypatch.setattr(ic, "_config", lambda: config)
    monkeypatch.setattr(ic, "check_tcp_connectivity", lambda *a, **k: False)
    monkeypatch.setattr(ic.VastProvider, "get", lambda *a: webrtc_info)
    monkeypatch.setattr(ic, "start_webrtc_relay", lambda *a: 1234)
    stopped = []
    monkeypatch.setattr(ic, "stop_webrtc_relay", lambda *a: stopped.append(a))
    original = ic.make_webrtc_http_server
    monkeypatch.setattr(ic, "make_webrtc_http_server", lambda p, c: original(0, c, tmp_path))

    def tunnel(*a, **kw):
        kw["on_connect"](webrtc_info)
        assert (49100, "WebRTC signal") in kw["service_ports"]
        assert all(port != ic.DEFAULT_NOVNC_PORT for port, _ in kw["service_ports"])
        raise failure

    monkeypatch.setattr(ic, "run_supervised_tunnel", tunnel)
    result = CliRunner().invoke(ic.app, [command, "--instance-id", "123", "--client-ip", "198.51.100.10"])
    assert result.exit_code == (0 if isinstance(failure, KeyboardInterrupt) else 1), result.output
    assert stopped == [(config, webrtc_info.ssh, 1234)]


@pytest.fixture()
def aws_webrtc_info(config):
    return ic.AwsProvider(config)._to_info({
        "InstanceId": "i-test", "State": {"Name": "running"},
        "PublicIpAddress": "203.0.113.42",
        "Tags": [{"Key": ic.AWS_TAG_WEBRTC, "Value": "true"}],
        "SecurityGroups": [{"GroupId": "sg-test"}],
    })


def test_aws_webrtc_endpoint_and_validation(config, aws_webrtc_info):
    ic.validate_webrtc_config(replace(config, webrtc_enabled=True, vast_whole_machine=False),
                              "aws", selecting_offer=True)
    assert ic.webrtc_connection(aws_webrtc_info) == {
        "signalingServer": "127.0.0.1", "signalingPort": 49100,
        "mediaServer": "203.0.113.42", "mediaPort": 47999,
    }
    for changes in [{"PublicIpAddress": None}, {"Tags": []}]:
        with pytest.raises(ic.IsaacCloudError):
            ic.webrtc_connection(replace(aws_webrtc_info, raw={**aws_webrtc_info.raw, **changes}))


def test_aws_webrtc_ingress_is_restricted_and_cleanup_owned(config, aws_webrtc_info, monkeypatch):
    calls = []

    def aws(c, args, **kwargs):
        calls.append(args)
        return {"SecurityGroupRules": [{"SecurityGroupRuleId": "sgr-owned"}]}

    monkeypatch.setattr(ic, "run_aws_json", aws)
    close = ic.AwsProvider(config).open_webrtc_access(aws_webrtc_info, "198.51.100.10")
    permission = json.loads(calls[0][calls[0].index("--ip-permissions") + 1])[0]
    assert permission["IpProtocol"] == "udp"
    assert permission["FromPort"] == permission["ToPort"] == 47999
    assert permission["IpRanges"][0]["CidrIp"] == "198.51.100.10/32"
    close()
    assert calls[1] == ["ec2", "revoke-security-group-ingress", "--group-id", "sg-test",
                        "--security-group-rule-ids", "sgr-owned"]


def test_aws_webrtc_duplicate_rule_is_not_modified(config, aws_webrtc_info, monkeypatch):
    calls = []

    def aws(c, args, **kwargs):
        calls.append(args)
        raise ic.IsaacCloudError("InvalidPermission.Duplicate")

    monkeypatch.setattr(ic, "run_aws_json", aws)
    with pytest.raises(ic.IsaacCloudError, match="existing rules are not modified"):
        ic.AwsProvider(config).open_webrtc_access(aws_webrtc_info, "198.51.100.10")
    assert len(calls) == 1


def test_aws_webrtc_setup_and_relay_run_inside_container(config, aws_webrtc_info, monkeypatch):
    scripts = []
    monkeypatch.setattr(ic, "run_ssh_script", lambda c, t, s, **kw: scripts.append((s, kw)) or "1234")
    ic.setup_isaac(replace(config, webrtc_enabled=True), aws_webrtc_info)
    ic.start_webrtc_relay(config, aws_webrtc_info, "198.51.100.10")
    assert len(scripts) == 3
    assert all(kw["in_container"] for _, kw in scripts)
    calls = []
    monkeypatch.setattr(ic, "run_ssh", lambda *a, **kw: calls.append(kw))
    ic.stop_webrtc_relay(config, aws_webrtc_info.ssh, 1234)
    assert calls[0]["in_container"]
    assert ic.wrap_container_command(aws_webrtc_info.ssh, "bash -s") == (
        "sudo docker exec -i isaac-sim bash -c 'bash -s'"
    )


@pytest.mark.parametrize("enabled", [False, True])
def test_aws_launch_records_webrtc_mode(config, monkeypatch, enabled):
    provider = ic.AwsProvider(replace(config, ngc_api_key="fake", webrtc_enabled=enabled))
    monkeypatch.setattr(provider, "resolve_ami", lambda: "ami-test")
    monkeypatch.setattr(provider, "_ensure_key_pair", lambda: "key-test")
    monkeypatch.setattr(provider, "_ensure_security_group", lambda: "sg-test")
    calls = []
    monkeypatch.setattr(ic, "run_aws_json", lambda c, a, **kw: calls.append(a) or {
        "Instances": [{"InstanceId": "i-test"}],
    })
    provider.launch()
    tags = json.loads(calls[0][calls[0].index("--tag-specifications") + 1])[0]["Tags"]
    assert ({"Key": ic.AWS_TAG_WEBRTC, "Value": "true"} in tags) == enabled
    assert "--network=host" in provider._build_user_data()


def test_aws_resume_preserves_webrtc_mode(config, aws_webrtc_info, monkeypatch):
    monkeypatch.setattr(ic, "_config", lambda: config)
    monkeypatch.setattr(ic.AwsProvider, "get", lambda *a: aws_webrtc_info)
    monkeypatch.setattr(ic, "wait_for_ssh", lambda *a: aws_webrtc_info)
    monkeypatch.setattr(ic, "wait_for_container", lambda *a: None)
    setups = []
    monkeypatch.setattr(ic, "setup_isaac", lambda c, i: setups.append(c))
    result = CliRunner().invoke(ic.app, ["resume", "--provider", "aws", "--instance-id", "i-test"])
    assert result.exit_code == 0, result.output
    assert setups[0].webrtc_enabled and not setups[0].gui_enabled
    assert "--provider aws --instance-id i-test" in result.output


def test_aws_key_import_is_portable(config, tmp_path, monkeypatch):
    key = tmp_path / "key.pub"
    key.write_text("ssh-ed25519 test public key\n")
    calls = []
    monkeypatch.setattr(ic, "run_aws_json", lambda c, a, **kw: calls.append(a) or {})
    ic.AwsProvider(replace(config, ssh_public_key_path=str(key)))._ensure_key_pair()
    encoded = calls[1][calls[1].index("--public-key-material") + 1]
    assert base64.b64decode(encoded) == b"ssh-ed25519 test public key"


@pytest.mark.parametrize("stage", ["exit", "relay_start", "relay_stop", "reconnect"])
def test_aws_viewer_cleans_ingress_on_failures(config, aws_webrtc_info, tmp_path, monkeypatch, stage):
    (tmp_path / "index.html").write_text("viewer")
    monkeypatch.setattr(ic, "_config", lambda: config)
    monkeypatch.setattr(ic, "check_tcp_connectivity", lambda *a, **kw: False)
    monkeypatch.setattr(ic.AwsProvider, "get", lambda *a: aws_webrtc_info)
    original = ic.make_webrtc_http_server
    monkeypatch.setattr(ic, "make_webrtc_http_server", lambda p, c: original(0, c, tmp_path))
    events = []

    def access(self, info, ip):
        events.append(("open", ip))
        return lambda: events.append(("close", ip))

    def start(*a):
        if stage == "relay_start":
            raise ic.IsaacCloudError("relay startup failed")
        return 1234

    def stop(*a):
        if stage == "relay_stop":
            raise ic.IsaacCloudError("SSH unavailable during cleanup")

    def tunnel(*a, **kw):
        kw["on_connect"](aws_webrtc_info)
        if stage == "reconnect":
            kw["on_connect"](aws_webrtc_info)
        raise KeyboardInterrupt

    monkeypatch.setattr(ic.AwsProvider, "open_webrtc_access", access)
    monkeypatch.setattr(ic, "start_webrtc_relay", start)
    monkeypatch.setattr(ic, "stop_webrtc_relay", stop)
    monkeypatch.setattr(ic, "run_supervised_tunnel", tunnel)
    result = CliRunner().invoke(ic.app, ["webrtc-view", "--provider", "aws", "--instance-id", "i-test",
                                       "--client-ip", "198.51.100.10"])
    assert result.exit_code == (1 if stage == "relay_start" else 0), result.output
    expected = [("open", "198.51.100.10"), ("close", "198.51.100.10")]
    assert events == expected * (2 if stage == "reconnect" else 1)
