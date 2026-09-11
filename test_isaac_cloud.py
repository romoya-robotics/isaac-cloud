"""Unit tests for isaac_cloud: provisioning fail-fast + snapshot persistence.

These lock in behavior that was validated live against real Vast instances
(docs/VAST_TIMEOUT_EXPERIMENT_RESULTS.md and the snapshot integration run):
status_msg failure signatures, auth-denial recovery timing, append-only
snapshot semantics, and project resolution precedence.
"""

import sys
import json
import base64
import socket
import time
import subprocess
import urllib.error
import urllib.request
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

    assert cfg.gui_mode == "none"
    assert cfg.vnc_enabled is False and cfg.webrtc_enabled is False


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
    # A failed userland side-load must stop the bring-up before Xvfb, not
    # surface minutes later as a black window.
    side_load = _index(script, 'ensure_nvidia_userland || { log "FAIL: NVIDIA userland side-load failed')
    assert side_load < xvfb
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
    # The side-loaded userland must match the host driver exactly: keyed on the
    # full version, with NVIDIA's own installer as the fallback when Ubuntu's
    # archive has moved past it (seen live: host 595.71.05, archive 595.84).
    assert '[ ! -f "$LIBDIR/libGLX_nvidia.so.$DRIVER" ]' in script
    assert 'index($3, d) == 1' in script
    assert "NVIDIA-Linux-x86_64-$DRIVER.run" in script and "--extract-only" in script
    # The installer fallback needs curl (not guaranteed in the image) and must
    # also try NVIDIA's datacenter archive, where some driver builds live only.
    assert "command -v curl >/dev/null || DEBIAN_FRONTEND=noninteractive apt-get install -y -qq curl" in script
    assert "for base in XFree86/Linux-x86_64 tesla; do" in script
    assert "NVIDIA userland side-load failed" in script
    # A failed side-load must stop the launch (the script has no `set -e`);
    # otherwise the kit starts and hangs on "waiting for viewport handle".
    assert 'ensure_nvidia_userland || { echo "NVIDIA_USERLAND_FAILED' in script
    assert _index(script, "NVIDIA_USERLAND_FAILED") < _index(script, "runheadless.sh")
    assert subprocess.run(["bash", "-n"], input=script, text=True, capture_output=True, check=False).returncode == 0


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
    gui = ic.VastProvider(replace(config, gui_mode="vnc")).catalog()
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
        {"public_ipaddr": "203.0.113.42", "extra_env": VAST_WEBRTC_ENV, "ports": {
            "22/tcp": [{"HostPort": "30022"}],
            "47999/udp": [{"HostPort": "31234"}],
        }},
    )


# `extra_env` as the Vast API returns it for `launch --gui webrtc` (checked 2026-09-10).
VAST_WEBRTC_ENV = {"-p 47999:47999/udp": "1"}

# What webrtc_connection() resolves for the `webrtc_info` fixture.
CONNECTION = {"signalingServer": "127.0.0.1", "signalingPort": 49100,
              "mediaServer": "203.0.113.42", "mediaPort": 31234}


def stub_tools(directory, tools):
    """Write fake executables (name -> bash body) so container scripts run locally."""
    for name, body in tools.items():
        (directory / name).write_text(f"#!/bin/bash\n{body}\n")
        (directory / name).chmod(0o755)
    return {"PATH": f"{directory}:/usr/bin:/bin"}


def test_gui_mode_config(tmp_path):
    path = tmp_path / "stream.toml"
    path.write_text('[gui]\nmode = "webrtc"\n')
    cfg = ic.load_app_config(path)
    assert cfg.gui_mode == "webrtc" and cfg.webrtc_enabled and not cfg.vnc_enabled
    path.write_text('[gui]\nmode = "vnc"\n')
    cfg = ic.load_app_config(path)
    assert cfg.vnc_enabled and not cfg.webrtc_enabled
    path.write_text('[gui]\nmode = "both"\n')
    with pytest.raises(ic.IsaacCloudError, match="none, vnc, webrtc"):
        ic.load_app_config(path)


@pytest.mark.parametrize("stale, mode", [
    ("[gui]\nenabled = false\n[webrtc]\nenabled = false\n", "none"),
    ("[gui]\nenabled = true\n", "vnc"),
    ("[webrtc]\nenabled = true\n", "webrtc"),
])
def test_replaced_gui_keys_fail_with_the_exact_replacement(tmp_path, stale, mode):
    """No translation, but the error names the line to write: every command,
    including `stop`/`destroy` on a billing instance, is blocked until then."""
    path = tmp_path / "stale.toml"
    path.write_text(stale)
    with pytest.raises(ic.IsaacCloudError, match=rf'\[gui\].mode.*mode = "{mode}"'):
        ic.load_app_config(path)


def test_gui_option_rejects_unknown_modes():
    result = CliRunner().invoke(ic.app, ["launch", "--gui", "both"])
    assert result.exit_code == 2 and "Invalid value" in result.output


def test_webrtc_forces_whole_machine_offers(config):
    # NVENC needs host GPU 0; a whole machine guarantees it, whatever the config says.
    query = ic.VastProvider(replace(config, gui_mode="webrtc", vast_whole_machine=False))._query()
    assert query.endswith(" gpu_frac=1")
    assert not ic.VastProvider(replace(config, gui_mode="vnc", vast_whole_machine=False))._query().endswith("gpu_frac=1")


def test_webrtc_uses_mapped_media_port(webrtc_info):
    assert ic.webrtc_connection(webrtc_info) == {
        "signalingServer": "127.0.0.1", "signalingPort": 49100,
        "mediaServer": "203.0.113.42", "mediaPort": 31234,
    }


@pytest.mark.parametrize("raw,message", [
    ({"public_ipaddr": "203.0.113.42", "ports": {"47999/udp": [{"HostPort": "31234"}]}},
     "not launched with --gui webrtc"),
    ({"extra_env": VAST_WEBRTC_ENV}, "no UDP 47999 mapping"),
    ({"extra_env": VAST_WEBRTC_ENV, "ports": {"47999/tcp": [{"HostPort": "31234"}]}}, "no UDP 47999 mapping"),
    ({"extra_env": VAST_WEBRTC_ENV, "public_ipaddr": "$(touch /tmp/oops)",
      "ports": {"47999/udp": [{"HostPort": "31234"}]}}, "invalid"),
    ({"extra_env": VAST_WEBRTC_ENV, "public_ipaddr": "203.0.113.42",
      "ports": {"47999/udp": [{"HostPort": "70000"}]}}, "invalid"),
    ({"extra_env": VAST_WEBRTC_ENV, "public_ipaddr": "203.0.113.42", "ports": {"47999/udp": [{}]}}, "invalid"),
])
def test_webrtc_rejects_missing_or_invalid_mapping(webrtc_info, raw, message):
    with pytest.raises(ic.IsaacCloudError, match=message):
        ic.webrtc_connection(replace(webrtc_info, raw=raw))


def test_vast_webrtc_mode_survives_stop(webrtc_info):
    # A stopped Vast instance reports no runtime ports; extra_env persists.
    assert ic.uses_webrtc(replace(webrtc_info, status="stopped", raw={"extra_env": VAST_WEBRTC_ENV}))
    assert not ic.uses_webrtc(replace(webrtc_info, raw={**webrtc_info.raw, "extra_env": {}}))


@pytest.mark.parametrize("enabled", [False, True])
def test_vast_requests_only_udp_for_webrtc(config, monkeypatch, enabled):
    calls = []
    cfg = replace(config, ngc_api_key="fake", gui_mode="webrtc" if enabled else "none")
    prov = ic.VastProvider(cfg)

    def vast(args, **kwargs):
        calls.append(args)
        return {"success": True, "new_contract": 123}

    monkeypatch.setattr(ic, "run_vastai_json", vast)
    monkeypatch.setattr(prov, "_ensure_account_ssh_key", lambda: None)
    prov.launch(offer_id="42")
    args = calls[0]
    if enabled:
        assert args[args.index("--env") + 1] == ic.VAST_WEBRTC_PORT_OPTION
        assert ic.VAST_WEBRTC_PORT_OPTION in VAST_WEBRTC_ENV
    else:
        assert "--env" not in args
    assert "49100" not in " ".join(args)
    assert "8226" not in " ".join(args)


def test_webrtc_explicit_offer_is_checked_at_boot(config, monkeypatch):
    monkeypatch.setattr(ic, "_config", lambda: replace(config, vast_whole_machine=False))
    rented = []

    def launch(self, offer_id=None):
        rented.append(offer_id)
        raise ic.IsaacCloudError("stop after rental")

    monkeypatch.setattr(ic.VastProvider, "launch", launch)
    result = CliRunner().invoke(ic.app, ["launch", "--gui", "webrtc", "--offer-id", "42"])
    assert rented == ["42"], result.output
    assert "whole_machine" not in result.output


def test_webrtc_launch_overrides_config_gui(config, monkeypatch, webrtc_info):
    cfg = replace(config, gui_mode="vnc", persistence_enabled=False)
    monkeypatch.setattr(ic, "_config", lambda: cfg)
    monkeypatch.setattr(ic.VastProvider, "launch", lambda *a, **k: webrtc_info)
    monkeypatch.setattr(ic, "wait_for_ssh", lambda *a, **k: webrtc_info)
    monkeypatch.setattr(ic, "wait_for_container", lambda *a: None)
    setups = []
    monkeypatch.setattr(ic, "setup_isaac", lambda c, i: setups.append(c))
    result = CliRunner().invoke(ic.app, ["launch", "--gui", "webrtc"])
    assert result.exit_code == 0, result.output
    assert setups[0].webrtc_enabled and not setups[0].vnc_enabled
    assert "isaac_cloud.py tunnel --instance-id 123 --provider vast" in result.output
    assert "webrtc viewer  -> http://localhost:8210/" in result.output
    assert "Tunnel (raw ssh)" not in result.output  # raw ssh alone would not start the media relay
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
    assert not setups[0].vnc_enabled


def test_webrtc_resume_rejects_gui_before_start(config, monkeypatch, webrtc_info):
    stopped = replace(webrtc_info, status="stopped", ssh=None, raw={"extra_env": VAST_WEBRTC_ENV})
    monkeypatch.setattr(ic, "_config", lambda: config)
    monkeypatch.setattr(ic.VastProvider, "get", lambda *a: stopped)
    monkeypatch.setattr(ic.VastProvider, "start", lambda *a: pytest.fail("started an incompatible instance"))
    result = CliRunner().invoke(ic.app, ["resume", "--instance-id", "123", "--gui", "vnc"])
    assert result.exit_code == 1, result.output
    assert "resume it with --gui webrtc" in result.output


def test_resume_rejects_webrtc_on_ssh_only_instance(config, monkeypatch, webrtc_info):
    ssh_only = replace(webrtc_info, status="stopped", ssh=None, raw={"extra_env": {}})
    monkeypatch.setattr(ic, "_config", lambda: config)
    monkeypatch.setattr(ic.VastProvider, "get", lambda *a: ssh_only)
    monkeypatch.setattr(ic.VastProvider, "start", lambda *a: pytest.fail("started an instance that cannot stream"))
    result = CliRunner().invoke(ic.app, ["resume", "--instance-id", "123", "--gui", "webrtc"])
    assert result.exit_code == 1, result.output
    assert "not launched with --gui webrtc" in result.output


def test_webrtc_config_does_not_convert_ssh_only_resume(config, monkeypatch, webrtc_info):
    ssh_only = replace(webrtc_info, raw={**webrtc_info.raw, "extra_env": {}})
    monkeypatch.setattr(ic, "_config", lambda: replace(config, gui_mode="webrtc"))
    monkeypatch.setattr(ic.VastProvider, "get", lambda *a: ssh_only)
    monkeypatch.setattr(ic, "wait_for_ssh", lambda *a: ssh_only)
    monkeypatch.setattr(ic, "wait_for_container", lambda *a: None)
    monkeypatch.setattr(ic, "remote_gui_stack_installed", lambda *a: False)
    setups = []
    monkeypatch.setattr(ic, "setup_isaac", lambda c, i: setups.append(c))
    result = CliRunner().invoke(ic.app, ["resume", "--instance-id", "123"])
    assert result.exit_code == 0, result.output
    assert not setups[0].webrtc_enabled
    assert "isaac_cloud.py tunnel" in result.output


def test_webrtc_relay_is_udp_only_and_restricted():
    script = ic.build_webrtc_session_script("198.51.100.10", CONNECTION)
    assert "range=198.51.100.10/32" in script
    assert "UDP4:127.0.0.1:47998" in script
    assert "UDP4-LISTEN:47999" in script
    assert "TCP" not in script
    assert "kill -0" in script
    assert '"mediaPort": 31234' in script and "connection.json" in script
    with pytest.raises(ic.IsaacCloudError, match="--webrtc-client-ip"):
        ic.build_webrtc_session_script("0.0.0.0/0", CONNECTION)
    with pytest.raises(ic.IsaacCloudError):
        ic.build_webrtc_session_script("1.2.3.4; echo injected", CONNECTION)


@pytest.mark.parametrize("running", ["", "4321"])
def test_webrtc_session_script_keeps_live_relay(tmp_path, running):
    # Stub the process tools so the real script's branching runs locally.
    log = tmp_path / "calls"
    viewer = tmp_path / "webrtc-viewer"
    viewer.mkdir()
    (viewer / "serve.py").write_text("# installed\n")
    env = stub_tools(tmp_path, {
        # The viewer server is reported running; only the relay branch varies.
        "pgrep": f'echo "pgrep $*" >> {log}; case "$*" in *serve.py*) exit 0;; esac; '
                 f'[ -n "{running}" ] && echo {running}',
        "pkill": f'echo "pkill $*" >> {log}',
        "setsid": f'echo "setsid $*" >> {log}; exec sleep 5',
        "socat": "exit 0",
    })
    script = ic.build_webrtc_session_script("198.51.100.10", CONNECTION).replace("/root/", f"{tmp_path}/")
    result = subprocess.run(["bash", "-s"], input=script, text=True, capture_output=True, env=env, check=False)
    assert result.returncode == 0, result.stderr
    calls = log.read_text().splitlines()
    # Viewer server and endpoint first; the relay last, so nothing that can
    # fail runs after a socat the caller does not yet know the PID of.
    assert calls[0] == "pgrep -f webrtc-viewer/serve.py"
    assert calls[1].startswith("pgrep -o -x -f socat -T 60 UDP4-LISTEN:47999,")
    assert r"range=198\.51\.100\.10/32" in calls[1]
    if running:
        assert result.stdout.strip() == running
        assert calls[2:] == []  # no pkill, no restart
    else:
        subprocess.run(["kill", result.stdout.strip()], check=False)
        assert calls[2] == f"pkill -f ^{ic.WEBRTC_RELAY_PREFIX}"
        assert calls[3].startswith("setsid socat -T 60 UDP4-LISTEN:47999,") and "range=198.51.100.10/32" in calls[3]
    # The media endpoint is published to the container-hosted page on every connect.
    assert json.loads((viewer / "connection.json").read_text()) == CONNECTION


def test_webrtc_session_script_heredoc_survives_reindent():
    """The JSON is a one-line heredoc body, so the terminator stays at column 0
    however the f-string block is indented."""
    script = ic.build_webrtc_session_script("198.51.100.10", CONNECTION)
    body = script[script.index("connection.json"):]
    lines = body.splitlines()
    assert lines[1] == json.dumps(CONNECTION) and lines[2] == ic.VIEWER_HEREDOC_EOF
    assert _index(script, "connection.json") < _index(script, "setsid socat")


def test_webrtc_session_script_requires_installed_viewer(tmp_path):
    env = stub_tools(tmp_path, {"socat": "exit 0"})
    script = ic.build_webrtc_session_script("198.51.100.10", CONNECTION).replace("/root/", f"{tmp_path}/")
    result = subprocess.run(["bash", "-s"], input=script, text=True, capture_output=True, env=env, check=False)
    assert result.returncode == 1 and "viewer not installed" in result.stdout


def test_webrtc_install_script_embeds_viewer_and_pins_sdk(tmp_path):
    script = ic.build_webrtc_install_script()
    assert ic.WEBRTC_SDK_URL in script and ic.WEBRTC_SDK_SHA256 in script
    assert "sha256sum -c" in script
    assert f'"$VIEWER_DIR/serve.py" {ic.DEFAULT_WEBRTC_VIEWER_PORT}' in script
    assert f"http://127.0.0.1:{ic.DEFAULT_WEBRTC_VIEWER_PORT}/" in script and "kill -0 \"$viewer_pid\"" not in script
    assert "npm --prefix" not in script and "vite" not in script  # no local toolchain anywhere
    # Run it locally: GPU check, SDK already present (sha256sum stubbed), viewer not yet serving.
    log = tmp_path / "calls"
    env = stub_tools(tmp_path, {
        "nvidia-smi": 'echo "    Minor Number                          : 0"',
        "socat": "exit 0",
        # The viewer port answers on the second probe; any other curl (an SDK download) is a failure.
        "curl": f'case "$*" in *127.0.0.1:8210/) [ -f {tmp_path}/probed ] && exit 0; touch {tmp_path}/probed; exit 1;; esac; '
                f'echo "curl $*" >> {log}; exit 1',
        "sha256sum": "exit 0", "pgrep": "exit 1",
        "setsid": f'echo "setsid $*" >> {log}; exec sleep 5',
    })
    script = script.replace(f"VIEWER_DIR={ic.WEBRTC_VIEWER_DIR}", f"VIEWER_DIR={tmp_path}/viewer")
    result = subprocess.run(["bash", "-s"], input=script.replace("/root/", f"{tmp_path}/"),
                            text=True, capture_output=True, env=env, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip().endswith("WEBRTC_VIEWER_READY")
    calls = log.read_text().splitlines()
    assert len(calls) == 1 and calls[0].startswith(f"setsid /isaac-sim/python.sh {tmp_path}/viewer/serve.py 8210")
    # The repo's viewer files arrive byte-for-byte; the SDK module is the only download.
    for name in ic.WEBRTC_VIEWER_FILES:
        assert (tmp_path / "viewer" / name).read_text() == (ic.WEBRTC_VIEWER_SRC / name).read_text()
    assert "omniverse-webrtc-streaming-library.js" in (tmp_path / "viewer" / "viewer.js").read_text()


def test_webrtc_install_script_rejects_non_nvenc_host(tmp_path):
    env = stub_tools(tmp_path, {"nvidia-smi": 'echo "    Minor Number                          : 1"'})
    result = subprocess.run(["bash", "-s"], input=ic.build_webrtc_install_script(),
                            text=True, capture_output=True, env=env, check=False)
    assert result.returncode == 1 and "GPU minor 0" in result.stdout


def test_webrtc_viewer_server_serves_loopback_without_caching(tmp_path):
    for name in ic.WEBRTC_VIEWER_FILES:
        (tmp_path / name).write_text((ic.WEBRTC_VIEWER_SRC / name).read_text())
    (tmp_path / "connection.json").write_text(json.dumps(CONNECTION))
    (tmp_path / "assets").mkdir()
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    server = subprocess.Popen([sys.executable, str(tmp_path / "serve.py"), str(port)])
    try:
        for _ in range(50):
            try:
                page = urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=1)
                break
            except (urllib.error.URLError, ConnectionError):
                time.sleep(0.1)
        assert b"<title>Isaac Sim" in page.read()
        assert page.headers["Cache-Control"] == "no-store"
        endpoint = urllib.request.urlopen(f"http://127.0.0.1:{port}/connection.json", timeout=2)
        assert json.load(endpoint) == CONNECTION
        assert endpoint.headers["Cache-Control"] == "no-store"
        with pytest.raises(urllib.error.HTTPError) as listing:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/assets/", timeout=2)
        assert listing.value.code == 404
        # DNS rebinding guard: only a localhost Host (any port) is served.
        ok = urllib.request.Request(f"http://127.0.0.1:{port}/connection.json", headers={"Host": "localhost:18210"})
        assert json.load(urllib.request.urlopen(ok, timeout=2)) == CONNECTION
        for host in ("evil.example:80", ""):
            with pytest.raises(urllib.error.HTTPError) as refused:
                urllib.request.urlopen(urllib.request.Request(
                    f"http://127.0.0.1:{port}/connection.json", headers={"Host": host}), timeout=2)
            assert refused.value.code == 403
    finally:
        server.terminate()
        server.wait(timeout=5)


def test_webrtc_shell_syntax(config):
    assert "--/exts/omni.services.livestream.session/quitOnSessionEnded=false" in (
        ic.build_isaac_container_launch_script(config)
    )
    for script in [ic.build_webrtc_install_script(), ic.build_webrtc_session_script("198.51.100.10", CONNECTION),
                   ic.build_container_probe_script(config), ic.build_isaac_container_launch_script(config),
                   ic.build_lab_install_script(config.lab_ref)]:
        result = subprocess.run(["bash", "-n"], input=script, text=True, capture_output=True)
        assert result.returncode == 0, result.stderr


def test_webrtc_client_ip_is_address_seen_by_ssh(config, webrtc_info, monkeypatch):
    monkeypatch.setattr(ic, "run_ssh", lambda *a, **k: "198.51.100.10")
    assert ic.detect_client_ip(config, webrtc_info.ssh) == "198.51.100.10"
    monkeypatch.setattr(ic, "run_ssh", lambda *a, **k: "2001:db8::1")
    with pytest.raises(ic.IsaacCloudError, match="--webrtc-client-ip"):
        ic.detect_client_ip(config, webrtc_info.ssh)
    scripts = []
    monkeypatch.setattr(ic, "run_ssh_script", lambda c, t, s, **k: scripts.append(s) or "1234")
    assert ic.start_webrtc_relay(config, webrtc_info.ssh, "198.51.100.10", CONNECTION) == 1234
    assert "range=198.51.100.10/32" in scripts[0]


@pytest.mark.parametrize("local_signal_port", [49100, 49101])
def test_webrtc_tunnel_refreshes_mapping_on_reconnect(config, webrtc_info, monkeypatch, local_signal_port):
    calls, mapped = [], []
    second = replace(webrtc_info, raw={"public_ipaddr": "203.0.113.43", "extra_env": VAST_WEBRTC_ENV, "ports": {
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
    # Same unqualified bind as `tunnel` and format_tunnel_command: IPv4 and IPv6 loopback.
    assert f"{local_signal_port}:127.0.0.1:49100" in calls[0]
    assert "47999" not in " ".join(calls[0])


@pytest.mark.parametrize("error", [ic.IsaacCloudError("SSH command failed (255)"),
                                   subprocess.TimeoutExpired("ssh", 30)])
def test_tunnel_retries_failed_connection_setup(config, webrtc_info, monkeypatch, capsys, error):
    prov = ic.VastProvider(config)
    monkeypatch.setattr(prov, "get", lambda _: webrtc_info)
    monkeypatch.setattr(ic.time, "sleep", lambda _: None)
    attempts, ssh_runs = [], []

    def prepare(info):
        attempts.append(info)
        if len(attempts) == 1:
            raise error

    monkeypatch.setattr(ic.subprocess, "run", lambda args, **kw: ssh_runs.append(args)
                        or subprocess.CompletedProcess(args, 130))
    ic.run_supervised_tunnel(config, prov, "123", on_connect=prepare)
    assert len(attempts) == 2 and len(ssh_runs) == 1
    output = capsys.readouterr().out
    assert "Connection setup failed" in output and "reconnecting in 6s (drop #1)" in output
    assert "Tunnel to vast:123" in output


@pytest.fixture()
def free_local_ports(monkeypatch):
    """The tunnel command probes its local ports; tests never bind real ones."""
    monkeypatch.setattr(ic, "ensure_local_ports_free", lambda ports: None)


def test_tunnel_rejects_busy_local_port(monkeypatch):
    with socket.socket() as taken:
        taken.bind(("127.0.0.1", 0))
        taken.listen(1)
        port = taken.getsockname()[1]
        with pytest.raises(ic.IsaacCloudError, match=f"Local port {port} is already in use"):
            ic.ensure_local_ports_free([port])
    ic.ensure_local_ports_free([port])  # released


@pytest.mark.parametrize("failure", [
    None, KeyboardInterrupt(), OSError("tunnel failed"),
    subprocess.TimeoutExpired("ssh", 30), ic.IsaacCloudError("relay failed"),
])
def test_webrtc_tunnel_forwards_viewer_and_cleans_up_relay(config, monkeypatch, webrtc_info, free_local_ports, failure):
    monkeypatch.setattr(ic, "_config", lambda: config)
    monkeypatch.setattr(ic.VastProvider, "get", lambda *a: webrtc_info)
    started, stopped = [], []
    monkeypatch.setattr(ic, "start_webrtc_relay", lambda *a: started.append(a) or 1234)
    monkeypatch.setattr(ic, "stop_webrtc_relay", lambda *a: stopped.append(a))

    def tunnel(*a, **kw):
        kw["on_connect"](webrtc_info)
        ports = [port for port, _ in kw["service_ports"]]
        assert ports == [8226, 8554, 49100, 8210]  # viewer page and signaling replace noVNC
        if failure is not None:
            raise failure

    monkeypatch.setattr(ic, "run_supervised_tunnel", tunnel)
    result = CliRunner().invoke(ic.app, ["tunnel", "--instance-id", "123", "--webrtc-client-ip", "198.51.100.10"])
    interrupted = isinstance(failure, KeyboardInterrupt)
    assert result.exit_code == (0 if failure is None or interrupted else 1), result.output
    assert result.exception is None or isinstance(result.exception, SystemExit)  # never a traceback
    assert ("Error:" in result.output) == (result.exit_code == 1)
    assert ("Tunnel stopped." in result.output) == interrupted
    assert "http://localhost:8210/" in result.output
    assert started == [(config, webrtc_info.ssh, "198.51.100.10", CONNECTION)]
    assert stopped == [(config, webrtc_info.ssh, 1234)]


def test_webrtc_tunnel_remaps_viewer_and_signaling_ports(config, monkeypatch, webrtc_info, free_local_ports):
    """A second WebRTC tunnel needs its own local ports, and the browser dials
    the LOCAL signaling port, so connection.json must carry the remapped value."""
    monkeypatch.setattr(ic, "_config", lambda: config)
    monkeypatch.setattr(ic.VastProvider, "get", lambda *a: webrtc_info)
    started = []
    monkeypatch.setattr(ic, "start_webrtc_relay", lambda *a: started.append(a) or 1234)
    monkeypatch.setattr(ic, "stop_webrtc_relay", lambda *a: None)
    calls = []

    def tunnel(*a, **kw):
        calls.append((a, kw))
        kw["on_connect"](webrtc_info)

    monkeypatch.setattr(ic, "run_supervised_tunnel", tunnel)
    result = CliRunner().invoke(ic.app, ["tunnel", "--instance-id", "123", "--webrtc-client-ip", "198.51.100.10",
                                       "--gui-port", "18210", "--webrtc-signal-port", "59100", "--agent-port", "18226"])
    assert result.exit_code == 0, result.output
    assert "http://localhost:18210/" in result.output
    (_, _, _, local_ports), kw = calls[0]
    assert ic.tunnel_forwards(local_ports, service_ports=kw["service_ports"]) == [
        (18226, 8226), (8554, 8554), (59100, 49100), (18210, 8210)]
    assert started[0][3] == {**CONNECTION, "signalingPort": 59100}


@pytest.mark.parametrize("args, message", [
    (["--gui-port", "8210", "--webrtc-signal-port", "8210"], "must be distinct"),
    (["--gui-port", "8226"], "must be distinct"),
])
def test_webrtc_tunnel_rejects_colliding_ports(config, monkeypatch, webrtc_info, free_local_ports, args, message):
    monkeypatch.setattr(ic, "_config", lambda: config)
    monkeypatch.setattr(ic.VastProvider, "get", lambda *a: webrtc_info)
    monkeypatch.setattr(ic, "run_supervised_tunnel", lambda *a, **kw: pytest.fail("tunnel started"))
    result = CliRunner().invoke(ic.app, ["tunnel", "--instance-id", "123", *args])
    assert result.exit_code == 1 and "Error:" in result.output and message in result.output, result.output


@pytest.mark.parametrize("args", [["--webrtc-client-ip", "198.51.100.10"], ["--webrtc-signal-port", "59100"]])
def test_ssh_only_tunnel_rejects_webrtc_options(config, monkeypatch, webrtc_info, free_local_ports, args):
    ssh_only = replace(webrtc_info, raw={**webrtc_info.raw, "extra_env": {}})
    monkeypatch.setattr(ic, "_config", lambda: config)
    monkeypatch.setattr(ic.VastProvider, "get", lambda *a: ssh_only)
    monkeypatch.setattr(ic, "run_supervised_tunnel", lambda *a, **kw: pytest.fail("tunnel started"))
    result = CliRunner().invoke(ic.app, ["tunnel", "--instance-id", "123", *args])
    assert result.exit_code == 1 and "does not apply" in result.output and "not launched with --gui webrtc" in result.output


def test_tunnel_keeps_plain_forwards_for_ssh_only_instances(config, monkeypatch, webrtc_info, free_local_ports):
    ssh_only = replace(webrtc_info, raw={**webrtc_info.raw, "extra_env": {}})
    monkeypatch.setattr(ic, "_config", lambda: config)
    monkeypatch.setattr(ic.VastProvider, "get", lambda *a: ssh_only)
    monkeypatch.setattr(ic, "start_webrtc_relay", lambda *a: pytest.fail("relay started for an SSH-only box"))
    calls = []
    monkeypatch.setattr(ic, "run_supervised_tunnel", lambda *a, **kw: calls.append((a, kw)))
    result = CliRunner().invoke(ic.app, ["tunnel", "--instance-id", "123", "--gui-port", "16080"])
    assert result.exit_code == 0, result.output
    (_, _, instance_id, local_ports), kw = calls[0]
    assert instance_id == "123" and local_ports[ic.DEFAULT_NOVNC_PORT] == 16080
    assert kw == {}  # default SERVICE_PORTS (with noVNC), no on_connect hook


@pytest.mark.parametrize("command", ["view", "webrtc", "webrtc-view"])
def test_webrtc_view_command_removed(command):
    assert CliRunner().invoke(ic.app, [command, "--help"]).exit_code != 0


def test_webrtc_tunnel_reuses_relay_and_ingress_across_reconnects(config, monkeypatch, aws_webrtc_info, free_local_ports):
    monkeypatch.setattr(ic, "_config", lambda: config)
    monkeypatch.setattr(ic.AwsProvider, "get", lambda *a: aws_webrtc_info)
    addresses = iter(["198.51.100.10", "198.51.100.10", "198.51.100.20"])
    monkeypatch.setattr(ic, "detect_client_ip", lambda *a: next(addresses))
    events = []

    def access(self, info, ip):
        events.append(("open", ip))
        return lambda: events.append(("close", ip))

    monkeypatch.setattr(ic.AwsProvider, "open_webrtc_access", access)
    monkeypatch.setattr(ic, "start_webrtc_relay", lambda c, t, ip, conn: events.append(("relay", ip)) or 1234)
    monkeypatch.setattr(ic, "stop_webrtc_relay", lambda *a: events.append(("stop",)))

    def tunnel(*a, **kw):
        for _ in range(3):  # reconnect, then reconnect after a network change
            kw["on_connect"](aws_webrtc_info)

    monkeypatch.setattr(ic, "run_supervised_tunnel", tunnel)
    result = CliRunner().invoke(ic.app, ["tunnel", "--provider", "aws", "--instance-id", "i-test"])
    assert result.exit_code == 0, result.output
    # The relay script is idempotent; only a changed client IP replaces the ingress rule.
    assert events == [
        ("open", "198.51.100.10"), ("relay", "198.51.100.10"),
        ("relay", "198.51.100.10"),
        ("close", "198.51.100.10"), ("open", "198.51.100.20"), ("relay", "198.51.100.20"),
        ("stop",), ("close", "198.51.100.20"),
    ]


@pytest.fixture()
def aws_webrtc_info(config):
    return ic.AwsProvider(config)._to_info({
        "InstanceId": "i-test", "State": {"Name": "running"},
        "PublicIpAddress": "203.0.113.42",
        "Tags": [{"Key": ic.AWS_TAG_WEBRTC, "Value": "true"}],
        "SecurityGroups": [{"GroupId": "sg-test"}],
    })


def test_aws_webrtc_endpoint_and_validation(config, aws_webrtc_info):
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


def aws_duplicate_rule(calls, existing):
    """Fake AWS CLI whose authorize fails as a duplicate of `existing` rules."""

    def aws(c, args, **kwargs):
        calls.append(args)
        if args[1] == "authorize-security-group-ingress":
            raise ic.IsaacCloudError("aws ec2 authorize-security-group-ingress failed: InvalidPermission.Duplicate")
        if args[1] == "describe-security-group-rules":
            assert args[-1] == "Name=group-id,Values=sg-test"
            return {"SecurityGroupRules": existing}
        return {}

    return aws


def webrtc_rule(rule_id, description, **changes):
    return {"SecurityGroupRuleId": rule_id, "GroupId": "sg-test", "IsEgress": False, "IpProtocol": "udp",
            "FromPort": 47999, "ToPort": 47999, "CidrIpv4": "198.51.100.10/32",
            "Description": description, **changes}


def test_aws_webrtc_adopts_leftover_rule(config, aws_webrtc_info, monkeypatch):
    calls = []
    existing = [webrtc_rule("sgr-other-port", "isaac-cloud WebRTC i-old", FromPort=22, ToPort=22),
                webrtc_rule("sgr-egress", "isaac-cloud WebRTC i-old", IsEgress=True),
                webrtc_rule("sgr-leftover", "isaac-cloud WebRTC i-old")]
    monkeypatch.setattr(ic, "run_aws_json", aws_duplicate_rule(calls, existing))
    close = ic.AwsProvider(config).open_webrtc_access(aws_webrtc_info, "198.51.100.10")
    close()
    assert calls[-1] == ["ec2", "revoke-security-group-ingress", "--group-id", "sg-test",
                         "--security-group-rule-ids", "sgr-leftover"]


def test_aws_webrtc_leaves_foreign_rule_in_place(config, aws_webrtc_info, monkeypatch):
    calls = []
    monkeypatch.setattr(ic, "run_aws_json", aws_duplicate_rule(calls, [webrtc_rule("sgr-admin", "office VPN")]))
    close = ic.AwsProvider(config).open_webrtc_access(aws_webrtc_info, "198.51.100.10")
    close()
    assert not any("revoke-security-group-ingress" in call for call in calls)


def test_aws_webrtc_unexplained_duplicate_is_reported(config, aws_webrtc_info, monkeypatch):
    monkeypatch.setattr(ic, "run_aws_json", aws_duplicate_rule([], []))
    with pytest.raises(ic.IsaacCloudError, match="InvalidPermission.Duplicate"):
        ic.AwsProvider(config).open_webrtc_access(aws_webrtc_info, "198.51.100.10")


def test_aws_webrtc_setup_and_relay_run_inside_container(config, aws_webrtc_info, monkeypatch):
    scripts = []
    monkeypatch.setattr(ic, "run_ssh_script", lambda c, t, s, **kw: scripts.append((s, kw)) or "1234")
    ic.setup_isaac(replace(config, gui_mode="webrtc"), aws_webrtc_info)
    ic.start_webrtc_relay(config, aws_webrtc_info.ssh, "198.51.100.10", CONNECTION)
    assert len(scripts) == 3  # viewer install, Isaac launch, relay session
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
    provider = ic.AwsProvider(replace(config, ngc_api_key="fake", gui_mode="webrtc" if enabled else "none"))
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


def aws_instance(config, state, public_ip=None, webrtc=True):
    """AwsProvider view of describe-instances: tags persist across stop/start, the public IPv4 does not."""
    raw = {"InstanceId": "i-test", "State": {"Name": state},
           "Tags": [{"Key": ic.AWS_TAG_WEBRTC, "Value": "true"}] if webrtc else [],
           "SecurityGroups": [{"GroupId": "sg-test"}]}
    return ic.AwsProvider(config)._to_info({**raw, **({"PublicIpAddress": public_ip} if public_ip else {})})


def test_aws_webrtc_mode_survives_stop(config):
    stopped = aws_instance(config, "stopped")
    assert stopped.ssh is None and ic.uses_webrtc(stopped)
    with pytest.raises(ic.IsaacCloudError, match="invalid WebRTC public IP"):
        ic.webrtc_connection(stopped)
    # A stop/start assigns a new public IPv4; the media endpoint follows it.
    assert ic.webrtc_connection(aws_instance(config, "running", "203.0.113.77"))["mediaServer"] == "203.0.113.77"


@pytest.mark.parametrize("gui", [False, True])
def test_aws_resume_preserves_webrtc_mode(config, monkeypatch, gui):
    resumed = aws_instance(config, "running", "203.0.113.77")
    started = []
    monkeypatch.setattr(ic, "_config", lambda: config)
    monkeypatch.setattr(ic.AwsProvider, "get", lambda *a: aws_instance(config, "stopped"))
    monkeypatch.setattr(ic.AwsProvider, "start", lambda self, i: started.append(i))
    monkeypatch.setattr(ic, "wait_for_ssh", lambda *a: resumed)
    monkeypatch.setattr(ic, "wait_for_container", lambda *a: None)
    setups = []
    monkeypatch.setattr(ic, "setup_isaac", lambda c, i: setups.append((c, i)))
    result = CliRunner().invoke(ic.app, ["resume", "--provider", "aws", "--instance-id", "i-test",
                                         *(["--gui", "vnc"] if gui else [])])
    if gui:
        assert result.exit_code == 1, result.output
        assert "resume it with --gui webrtc" in result.output
        assert started == []
        return
    assert result.exit_code == 0, result.output
    assert started == ["i-test"]
    assert setups[0][0].webrtc_enabled and not setups[0][0].vnc_enabled
    assert setups[0][1].raw["PublicIpAddress"] == "203.0.113.77"
    assert "tunnel --instance-id i-test --provider aws" in result.output
    assert "http://localhost:8210/" in result.output


def test_aws_webrtc_view_follows_stop_start(config, monkeypatch, capsys):
    # A viewer left running across `stop`/`resume` waits, then re-prepares on the new address.
    prov = ic.AwsProvider(config)
    infos = iter([aws_instance(config, "running", "203.0.113.42"), aws_instance(config, "stopped"),
                  aws_instance(config, "running", "203.0.113.77"), aws_instance(config, "running", "203.0.113.77")])
    monkeypatch.setattr(prov, "get", lambda _: next(infos))
    monkeypatch.setattr(ic.time, "sleep", lambda _: None)
    attempts = []

    def prepare(info):
        attempts.append(info.raw["PublicIpAddress"])
        if len(attempts) == 2:
            raise ic.IsaacCloudError("SSH command failed (255)")  # sshd not up yet after start
        ic.webrtc_connection(info)

    results = iter([255, 130])
    monkeypatch.setattr(ic.subprocess, "run", lambda args, **_: subprocess.CompletedProcess(args, next(results)))
    ic.run_supervised_tunnel(config, prov, "i-test", service_ports=[(49100, "signal")], on_connect=prepare)
    assert attempts == ["203.0.113.42", "203.0.113.77", "203.0.113.77"]
    output = capsys.readouterr().out
    assert "Instance is stopped; waiting" in output and "Connection setup failed" in output


def test_aws_key_import_is_portable(config, tmp_path, monkeypatch):
    key = tmp_path / "key.pub"
    key.write_text("ssh-ed25519 test public key\n")
    calls = []
    monkeypatch.setattr(ic, "run_aws_json", lambda c, a, **kw: calls.append(a) or {})
    ic.AwsProvider(replace(config, ssh_public_key_path=str(key)))._ensure_key_pair()
    encoded = calls[1][calls[1].index("--public-key-material") + 1]
    assert base64.b64decode(encoded) == b"ssh-ed25519 test public key"


@pytest.mark.parametrize("stage", ["exit", "relay_start", "relay_stop", "reconnect"])
def test_aws_tunnel_cleans_ingress_on_failures(config, aws_webrtc_info, monkeypatch, free_local_ports, stage):
    monkeypatch.setattr(ic, "_config", lambda: config)
    monkeypatch.setattr(ic.AwsProvider, "get", lambda *a: aws_webrtc_info)
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

    monkeypatch.setattr(ic.AwsProvider, "open_webrtc_access", access)
    monkeypatch.setattr(ic, "start_webrtc_relay", start)
    monkeypatch.setattr(ic, "stop_webrtc_relay", stop)
    monkeypatch.setattr(ic, "run_supervised_tunnel", tunnel)
    result = CliRunner().invoke(ic.app, ["tunnel", "--provider", "aws", "--instance-id", "i-test",
                                       "--webrtc-client-ip", "198.51.100.10"])
    assert result.exit_code == (1 if stage == "relay_start" else 0), result.output
    assert events == [("open", "198.51.100.10"), ("close", "198.51.100.10")]
