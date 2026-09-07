"""isaac-cloud: launch and manage NVIDIA Isaac Sim on cloud GPUs.

Providers:
  - vast: Vast.ai marketplace containers (the Isaac image IS the instance).
  - aws:  EC2 GPU VMs (g6e/L40S) running the Isaac container under Docker.

Agent control, RTSP, and noVNC use SSH tunnels to remote loopback services.
Opt-in WebRTC keeps signaling on SSH and uses a source-IP-restricted
UDP relay for media; Isaac's own endpoints remain on remote loopback.
"""

from __future__ import annotations

import base64
import functools
import ipaddress
import json
import os
import re
import shlex
import shutil
import socket
import subprocess
import tempfile
import textwrap
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import typer

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]

APP_NAME = "isaac-cloud"
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.toml"

DEFAULT_PROVIDER = "vast"
DEFAULT_ISAAC_VERSION = "6.0.1"
DEFAULT_ISAAC_SIGNAL_PORT = 49100
DEFAULT_ISAAC_STREAM_PORT = 47998
DEFAULT_AGENT_CONTROL_PORT = 8226  # isaacsim.code_editor.python_server, fixed upstream
DEFAULT_RTSP_PORT = 8554
DEFAULT_NOVNC_PORT = 6080
DEFAULT_GUI_RESOLUTION = "1920x1080"
DEFAULT_WEBRTC_VIEWER_PORT = 8210
# A separate UDP ingress forwards to Isaac's loopback-only media socket.
# Mapping the socket itself would fail: Vast DNAT targets the container IP.
DEFAULT_WEBRTC_RELAY_PORT = 47999
WEBRTC_VIEWER_DIST = Path(__file__).resolve().parent / "web-viewer" / "dist"

DEFAULT_INSTANCE_NAME_PREFIX = "isaac-cloud"
DEFAULT_DISK_GB = 100

# Isaac Lab releases are paired to specific Isaac Sim versions; this git ref
# (tag or branch) is the release built for Isaac Sim 6.0.1. Bump it together
# with [isaac].version, or override per-config with [isaac].lab_ref.
DEFAULT_ISAAC_LAB_REF = "v3.0.0-beta2.patch1"

# Container-side paths (identical on both providers: same Isaac image).
CONTAINER_PERSISTENCE_DIR = "/isaac-sim/project"

# Persistence: append-only snapshots under <s3_uri>projects/<project>/snapshots/.
DEFAULT_PERSISTENCE_PROJECT = "default"
DEFAULT_PERSISTENCE_KEEP_LAST = 10
# Millisecond resolution keeps names unique (append-only) for back-to-back
# saves; zero-padded fields keep lexicographic order == chronological order.
SNAPSHOT_TIME_FMT = "%Y-%m-%dT%H-%M-%S.%fZ"
SNAPSHOT_SUFFIX = ".tar.gz"
PROJECT_LABEL_PREFIX = "project="
AWS_TAG_PROJECT = "IsaacCloudProject"

# Vast provider defaults. NVENC requires the rented GPU to be host GPU 0
# (see docs/VAST_EXPERIMENT_RESULTS.md), which whole-machine offers guarantee.
DEFAULT_VAST_QUERY = (
    'gpu_name in ["RTX_4090","L40S"] driver_version >= 580.95.05 '
    "verified=true rentable=true num_gpus=1 disk_space >= 80 inet_down >= 300"
)
DEFAULT_VAST_WHOLE_MACHINE = True
DEFAULT_VAST_MIN_RELIABILITY = 0.99

# Vast surfaces docker's own error text in `status_msg` while the instance sits
# in "loading" forever (measured: bad registry login visible at +30s, missing
# image tag at +50s; neither ever self-heals — see docs/VAST_TIMEOUT_EXPERIMENT_RESULTS.md).
# Matching one of these means the provision is doomed: bail instead of waiting
# out the timeout.
VAST_FATAL_STATUS_PATTERNS = (
    "docker login failed",       # bad registry credentials (e.g. NGC key)
    "manifest unknown",          # image tag does not exist
    "pull access denied",
    "unauthorized",
    "no space left on device",
)
# A healthy pull can go up to ~6 min without a status_msg change (large layers),
# so generic daemon errors only count as fatal once they persist across polls.
VAST_DAEMON_ERROR_MARKER = "error response from daemon"
VAST_DAEMON_ERROR_FATAL_S = 90
# Some hosts fail to inject the account-level SSH key: the instance reaches
# "running", TCP connects, but pubkey auth is denied indefinitely (measured 16
# min of steady denials). Re-attach the key once, then give up.
VAST_SSH_DENIED_ATTACH_S = 45
VAST_SSH_DENIED_GIVE_UP_S = 300

# AWS provider defaults. g6e = L40S (RT cores + NVENC, on Isaac's GPU list).
DEFAULT_AWS_REGION = "us-west-2"
DEFAULT_AWS_INSTANCE_TYPE = "g6e.xlarge"
DEFAULT_AWS_SECURITY_GROUP = "isaac-cloud-ssh"
DEFAULT_AWS_KEY_NAME = "isaac-cloud"
DEFAULT_AWS_SSH_USER = "ubuntu"
# Deep Learning Base OSS Nvidia Driver AMI: driver + docker + nvidia-container-toolkit preinstalled.
DEFAULT_AWS_AMI_SSM_PARAM = (
    "/aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-22.04/latest/ami-id"
)
AWS_TAG_MANAGED = "IsaacCloudManaged"
AWS_TAG_WEBRTC = "IsaacCloudWebRTC"

ISAAC_MINIMUM_GPU_CLASSES = {"rtx4080", "rtx4090", "l40", "l40s"}


class IsaacCloudError(Exception):
    """Raised for all fatal, user-reportable failures."""


class ProvisioningDoomed(IsaacCloudError):
    """The instance can never become usable (bad image, bad creds, broken host).

    Unlike a plain timeout — where the instance might still come up — a doomed
    instance only accrues cost, so callers should destroy it."""


def _raise(message: str) -> None:
    raise IsaacCloudError(message)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AppConfig:
    provider: str
    isaac_version: str
    instance_name_prefix: str
    disk_gb: int
    ngc_api_key: str | None
    ssh_private_key_path: str | None
    ssh_public_key_path: str | None
    agent_enabled: bool
    gui_enabled: bool
    gui_resolution: str
    curobo_enabled: bool
    lab_enabled: bool
    lab_ref: str
    # vast
    vast_query: str
    vast_whole_machine: bool
    vast_min_reliability: float
    # aws
    aws_region: str
    aws_instance_type: str
    aws_ami_ssm_param: str
    aws_security_group: str
    aws_key_name: str
    # persistence
    persistence_enabled: bool
    persistence_s3_uri: str | None
    persistence_aws_region: str | None
    persistence_project: str
    persistence_keep_last: int
    webrtc_enabled: bool = False


def load_toml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("rb") as handle:
            return tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        _raise(f"Failed to read config at {path}: {exc}")
    return {}


def nested_get(data: dict[str, Any], *keys: str) -> Any:
    node: Any = data
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


def _bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def load_app_config(config_path: Path | None = None) -> AppConfig:
    data = load_toml(config_path or DEFAULT_CONFIG_PATH)
    env = os.environ

    def get(env_name: str, *keys: str) -> Any:
        return env.get(env_name) or nested_get(data, *keys)

    return AppConfig(
        provider=(get("ISAAC_CLOUD_PROVIDER", "defaults", "provider") or DEFAULT_PROVIDER).lower(),
        isaac_version=get("ISAAC_CLOUD_ISAAC_VERSION", "isaac", "version")
        or DEFAULT_ISAAC_VERSION,
        instance_name_prefix=get(
            "ISAAC_CLOUD_INSTANCE_NAME_PREFIX", "defaults", "instance_name_prefix"
        )
        or DEFAULT_INSTANCE_NAME_PREFIX,
        disk_gb=int(get("ISAAC_CLOUD_DISK_GB", "defaults", "disk_gb") or DEFAULT_DISK_GB),
        ngc_api_key=get("NGC_API_KEY", "ngc", "api_key"),
        ssh_private_key_path=get("ISAAC_CLOUD_SSH_PRIVATE_KEY", "ssh", "private_key_path"),
        ssh_public_key_path=get("ISAAC_CLOUD_SSH_PUBLIC_KEY", "ssh", "public_key_path"),
        agent_enabled=_bool(nested_get(data, "isaac", "agent"), True),
        gui_enabled=_bool(nested_get(data, "gui", "enabled"), False),
        gui_resolution=nested_get(data, "gui", "resolution") or DEFAULT_GUI_RESOLUTION,
        curobo_enabled=_bool(nested_get(data, "isaac", "curobo"), False),
        lab_enabled=_bool(nested_get(data, "isaac", "lab"), False),
        lab_ref=nested_get(data, "isaac", "lab_ref") or DEFAULT_ISAAC_LAB_REF,
        webrtc_enabled=_bool(nested_get(data, "webrtc", "enabled"), False),
        vast_query=nested_get(data, "vast", "query") or DEFAULT_VAST_QUERY,
        vast_whole_machine=_bool(
            nested_get(data, "vast", "whole_machine"), DEFAULT_VAST_WHOLE_MACHINE
        ),
        vast_min_reliability=float(
            nested_get(data, "vast", "min_reliability") or DEFAULT_VAST_MIN_RELIABILITY
        ),
        aws_region=get("AWS_REGION", "aws", "region") or DEFAULT_AWS_REGION,
        aws_instance_type=nested_get(data, "aws", "instance_type") or DEFAULT_AWS_INSTANCE_TYPE,
        aws_ami_ssm_param=nested_get(data, "aws", "ami_ssm_param") or DEFAULT_AWS_AMI_SSM_PARAM,
        aws_security_group=nested_get(data, "aws", "security_group")
        or DEFAULT_AWS_SECURITY_GROUP,
        aws_key_name=nested_get(data, "aws", "key_name") or DEFAULT_AWS_KEY_NAME,
        persistence_enabled=_bool(nested_get(data, "persistence", "enabled"), False),
        persistence_s3_uri=nested_get(data, "persistence", "s3_uri")
        or nested_get(data, "aws", "s3_uri"),
        persistence_aws_region=nested_get(data, "persistence", "aws_region")
        or get("AWS_REGION", "aws", "region"),
        persistence_project=get("ISAAC_CLOUD_PROJECT", "persistence", "project")
        or DEFAULT_PERSISTENCE_PROJECT,
        persistence_keep_last=int(
            nested_get(data, "persistence", "keep_last") or DEFAULT_PERSISTENCE_KEEP_LAST
        ),
    )


# ---------------------------------------------------------------------------
# Instance model shared across providers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SshTarget:
    host: str
    port: int
    user: str
    # command prefix that places us inside the Isaac container ("" on vast,
    # "docker exec isaac-sim " semantics handled via wrap_container_command on aws)
    container_via_docker: bool = False


@dataclass(frozen=True)
class InstanceInfo:
    provider: str
    instance_id: str
    status: str
    label: str
    ssh: SshTarget | None
    raw: dict[str, Any] = field(default_factory=dict)


class ProvisionMonitor:
    """Per-launch provisioning failure detector used by wait_for_ssh.

    The default accepts anything and lets the timeout be the only backstop;
    providers with detectable failure modes override the hooks to raise
    ProvisioningDoomed early."""

    def check_status(self, info: InstanceInfo) -> None:
        """Called with every polled instance state before SSH is attempted."""

    def auth_denied(self, info: InstanceInfo) -> None:
        """Called each time SSH pubkey auth is denied on a running instance."""

    def auth_reset(self) -> None:
        """Called when an SSH failure was something other than an auth denial."""


class Provider:
    """Shared provider interface: lifecycle methods are required, hooks are no-ops."""

    name = "abstract"

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    # --- required lifecycle ---
    def launch(self, offer_id: str | None = None) -> InstanceInfo:
        raise NotImplementedError

    def list_instances(self) -> list[InstanceInfo]:
        raise NotImplementedError

    def get(self, instance_id: str) -> InstanceInfo:
        raise NotImplementedError

    def stop(self, instance_id: str) -> None:
        raise NotImplementedError

    def start(self, instance_id: str) -> None:
        raise NotImplementedError

    def destroy(self, instance_id: str) -> None:
        raise NotImplementedError

    def persistence_remote_path(self) -> str:
        raise NotImplementedError

    def remote_sudo(self) -> str:
        raise NotImplementedError

    # --- optional hooks ---
    def open_webrtc_access(self, info: InstanceInfo, client_ip: str) -> Callable[[], None]:
        """Open provider ingress for one viewer; return its cleanup function."""
        return lambda: None

    def set_project_label(self, instance_id: str, project: str) -> None:
        """Record the project an instance was launched for (best-effort)."""

    def attach_ssh_key(self, instance_id: str) -> None:
        """Re-attach the user's SSH key to a running instance (best-effort)."""

    def provision_monitor(self) -> ProvisionMonitor:
        return ProvisionMonitor()


def shell_quote(value: str) -> str:
    return shlex.quote(value)


def dedent_script(script: str) -> str:
    return textwrap.dedent(script)


def build_isaac_image_ref(version: str) -> str:
    return f"nvcr.io/nvidia/isaac-sim:{version}"


def check_tcp_connectivity(host: str, port: int, timeout_seconds: float = 5.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout_seconds):
            return True
    except OSError:
        return False


def run_cli(
    cmd: list[str],
    *,
    timeout_seconds: int,
    error_prefix: str,
    env: dict[str, str] | None = None,
    parse_json: bool = False,
) -> Any:
    """Run an external CLI, raising IsaacCloudError with its output on failure."""
    completed = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout_seconds,
        env=env,
    )
    if completed.returncode != 0:
        output = ((completed.stderr or "") + "\n" + (completed.stdout or "")).strip()
        _raise(f"{error_prefix}: {output[:600]}" if output else f"{error_prefix}.")
    if not parse_json:
        return (completed.stdout or "").strip()
    if not completed.stdout.strip():
        return {}
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError:
        _raise(f"{error_prefix}: non-JSON output: {completed.stdout[:300]}")


def run_cli_quiet(cmd: list[str], *, timeout_seconds: int) -> None:
    """Best-effort external command: failures are deliberately swallowed."""
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=timeout_seconds)
    except (OSError, subprocess.TimeoutExpired):
        pass


def read_public_key(config: AppConfig) -> str | None:
    if not config.ssh_public_key_path:
        return None
    try:
        return Path(config.ssh_public_key_path).expanduser().read_text().strip()
    except OSError:
        return None


# ---------------------------------------------------------------------------
# SSH plumbing
# ---------------------------------------------------------------------------


def ssh_base_args(config: AppConfig, target: SshTarget) -> list[str]:
    args = [
        "ssh",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        "-o",
        "LogLevel=ERROR",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=15",
        "-o",
        "ServerAliveInterval=15",
        "-p",
        str(target.port),
    ]
    if config.ssh_private_key_path:
        args += ["-i", config.ssh_private_key_path]
    args.append(f"{target.user}@{target.host}")
    return args


def wrap_container_command(target: SshTarget, command: str) -> str:
    """Return a shell command that runs `command` inside the Isaac container."""
    if not target.container_via_docker:
        return command
    return f"sudo docker exec -i isaac-sim bash -c {shell_quote(command)}"


def run_ssh(
    config: AppConfig,
    target: SshTarget,
    command: str,
    *,
    in_container: bool = False,
    timeout_seconds: int = 120,
    check: bool = True,
) -> str:
    if in_container:
        command = wrap_container_command(target, command)
    completed = subprocess.run(
        ssh_base_args(config, target) + [command],
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout_seconds,
    )
    output = ((completed.stdout or "") + "\n" + (completed.stderr or "")).strip()
    if check and completed.returncode != 0:
        _raise(f"SSH command failed ({completed.returncode}): {output[:800]}")
    return (completed.stdout or "").strip()


def run_ssh_script(
    config: AppConfig,
    target: SshTarget,
    script: str,
    *,
    in_container: bool = False,
    timeout_seconds: int = 600,
) -> str:
    command = "bash -s"
    if in_container:
        command = wrap_container_command(target, command)
    completed = subprocess.run(
        ssh_base_args(config, target) + [command],
        input=script,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout_seconds,
    )
    output = ((completed.stdout or "") + "\n" + (completed.stderr or "")).strip()
    if completed.returncode != 0:
        _raise(f"Remote script failed ({completed.returncode}): {output[-1200:]}")
    return (completed.stdout or "").strip()


SERVICE_PORTS: list[tuple[int, str]] = [
    (DEFAULT_AGENT_CONTROL_PORT, "agent control"),
    (DEFAULT_RTSP_PORT, "rtsp cameras"),
    (DEFAULT_NOVNC_PORT, "gui (noVNC)"),
]


def probe_local_tunnel(
    timeout_seconds: float = 5.0, local_port: int = DEFAULT_AGENT_CONTROL_PORT
) -> str:
    """End-to-end health of a local tunnel, zombie-aware.

    A dead ssh forward can still accept() locally, so port-open checks lie.
    The agent socket gives a true round trip: send a no-op, expect JSON back.
    """
    try:
        sock = socket.create_connection(("127.0.0.1", local_port), timeout=2.0)
    except OSError:
        return (
            f"tunnel: not running locally on {local_port} "
            "(start one with: isaac_cloud.py tunnel)"
        )
    try:
        sock.settimeout(timeout_seconds)
        sock.sendall(b"pass")
        sock.shutdown(socket.SHUT_WR)
        response = b""
        while chunk := sock.recv(1024):
            response += chunk
        if b'"status"' in response:
            return "tunnel: healthy (agent socket answered end-to-end)"
        return "tunnel: local port open but no agent response (service down?)"
    except socket.timeout:
        return "tunnel: ZOMBIE - local port accepts but nothing comes back; restart the tunnel"
    except OSError:
        return "tunnel: local port open but connection resets (remote service down or tunnel broken)"
    finally:
        sock.close()


def tunnel_forwards(
    local_ports: dict[int, int] | None = None,
    *,
    service_ports: list[tuple[int, str]] | None = None,
) -> list[tuple[int, int]]:
    """(local, remote) pairs for every service; `local_ports` remaps remote -> local
    so tunnels to two boxes can coexist (e.g. {6080: 16080, 8226: 18226})."""
    local_ports = local_ports or {}
    service_ports = SERVICE_PORTS if service_ports is None else service_ports
    return [(local_ports.get(port, port), port) for port, _ in service_ports]


def run_supervised_tunnel(
    config: AppConfig,
    provider: Provider,
    instance_id: str,
    local_ports: dict[int, int] | None = None,
    *,
    service_ports: list[tuple[int, str]] | None = None,
    on_connect: Callable[[InstanceInfo], None] | None = None,
) -> None:
    """Foreground self-healing tunnel: keepalives kill zombies within ~30s,
    then we reconnect with backoff, re-resolving the instance address in case
    it changed across a stop/start. Ctrl-C exits.

    `on_connect` prepares remote services before each tunnel attempt; it does
    not indicate that the tunnel is already listening or that Isaac is ready.
    """
    service_ports = SERVICE_PORTS if service_ports is None else service_ports
    forwards = tunnel_forwards(local_ports, service_ports=service_ports)
    local_of = dict((remote, local) for local, remote in forwards)
    drops = 0
    backoff = 3
    while True:
        info = provider.get(instance_id)
        if info.status != "running" or not info.ssh:
            typer.echo(f"Instance is {info.status}; waiting 15s for it to be reachable...")
            time.sleep(15)
            continue
        target = info.ssh
        if on_connect is not None:
            on_connect(info)
        if drops == 0:
            typer.echo(f"Tunnel to {info.provider}:{instance_id} ({target.host}:{target.port}):")
            for port, label in service_ports:
                suffix = "/vnc.html (browser)" if port == DEFAULT_NOVNC_PORT else ""
                typer.echo(f"  {label:14s} -> localhost:{local_of[port]}{suffix}")
            typer.echo("Ctrl-C to stop.")
        args = ssh_base_args(config, target)
        args[1:1] = [
            "-N",
            "-o",
            "ServerAliveInterval=10",
            "-o",
            "ServerAliveCountMax=3",
            "-o",
            "ExitOnForwardFailure=yes",
        ]
        for local_port, remote_port in forwards:
            args[1:1] = ["-L", f"127.0.0.1:{local_port}:127.0.0.1:{remote_port}"]
        started = time.time()
        try:
            completed = subprocess.run(args, check=False)
        except KeyboardInterrupt:
            typer.echo("\nTunnel stopped.")
            return
        if completed.returncode == 130:  # ssh took the SIGINT before we did
            typer.echo("Tunnel stopped.")
            return
        held = time.time() - started
        drops += 1
        backoff = 3 if held > 60 else min(backoff * 2, 30)
        typer.echo(
            f"Tunnel dropped (exit {completed.returncode}, held {held:.0f}s, drop #{drops}); "
            f"reconnecting in {backoff}s..."
        )
        try:
            time.sleep(backoff)
        except KeyboardInterrupt:
            typer.echo("\nTunnel stopped.")
            return


def format_tunnel_command(config: AppConfig, target: SshTarget, forwards: list[tuple[int, int]]) -> str:
    key_flag = f" -i {config.ssh_private_key_path}" if config.ssh_private_key_path else ""
    fw = " ".join(f"-L {lp}:127.0.0.1:{rp}" for lp, rp in forwards)
    return (
        f"ssh{key_flag} -o ServerAliveInterval=10 -o ServerAliveCountMax=3 -N "
        f"{fw} -p {target.port} {target.user}@{target.host}"
    )


def has_webrtc_mapping(info: InstanceInfo) -> bool:
    return bool((info.raw.get("ports") or {}).get(f"{DEFAULT_WEBRTC_RELAY_PORT}/udp"))


def uses_webrtc(info: InstanceInfo) -> bool:
    if info.provider == "aws":
        return any(tag.get("Key") == AWS_TAG_WEBRTC and tag.get("Value") == "true"
                   for tag in info.raw.get("Tags", []))
    return has_webrtc_mapping(info)


def validate_webrtc_config(config: AppConfig, provider: str, *, selecting_offer: bool = False) -> None:
    if not config.webrtc_enabled:
        return
    if config.gui_enabled:
        _raise("WebRTC and noVNC run different Isaac apps. Choose --webrtc --no-gui or --gui --no-webrtc.")
    if provider == "vast" and selecting_offer and not config.vast_whole_machine:
        _raise("WebRTC requires [vast].whole_machine = true for NVENC. Explicit offers are checked at boot.")


def webrtc_connection(info: InstanceInfo) -> dict[str, Any]:
    """Resolve the public media endpoint; signaling always stays on SSH."""
    mappings = (info.raw.get("ports") or {}).get(f"{DEFAULT_WEBRTC_RELAY_PORT}/udp")
    if info.provider == "vast" and not mappings:
        _raise("Instance has no WebRTC UDP mapping. Launch a new instance with --webrtc; "
               "existing SSH-only instances cannot add the required port in place.")
    try:
        if info.provider == "aws":
            if not uses_webrtc(info):
                _raise("AWS instance is not configured for WebRTC. Launch it with --webrtc.")
            host = str(ipaddress.IPv4Address(info.raw.get("PublicIpAddress")))
            port = DEFAULT_WEBRTC_RELAY_PORT
        elif info.provider == "vast":
            host = str(ipaddress.IPv4Address(info.raw.get("public_ipaddr")))
            port = int(mappings[0]["HostPort"])
        else:
            _raise(f"WebRTC is unsupported for provider {info.provider}.")
        if not 1 <= port <= 65535:
            raise ValueError("out of range")
    except (ValueError, TypeError, KeyError, IndexError) as exc:
        _raise(f"{info.provider} returned an invalid WebRTC public IP/UDP port: {exc}")
    return {
        "signalingServer": "127.0.0.1",
        "signalingPort": DEFAULT_ISAAC_SIGNAL_PORT,
        "mediaServer": host,
        "mediaPort": port,
    }


def build_webrtc_check_script() -> str:
    return dedent_script(
        """\
        #!/bin/bash
        set -e
        minors=$(nvidia-smi -q | awk '/Minor Number/ {print $NF}')
        if ! echo "$minors" | grep -qx 0; then
            echo "WebRTC requires host GPU minor 0 for NVENC; select a whole-machine offer."
            exit 1
        fi
        command -v socat >/dev/null || {
            apt-get update -qq
            DEBIAN_FRONTEND=noninteractive apt-get install -y -qq socat
        }
        echo WEBRTC_PREREQUISITES_OK
        """
    )


def build_webrtc_relay_script(client_ip: str) -> str:
    # The SDK overrides the advertised loopback ICE address/port in the client.
    # A UDP-to-UDP relay avoids binding Isaac to the unavailable host public IP
    # inside Vast's Docker namespace. No media is encapsulated in TCP or SSH.
    try:
        client_ip = str(ipaddress.IPv4Address(client_ip))
    except ipaddress.AddressValueError:
        _raise("WebRTC client IP must be an IPv4 address (the public IP used for UDP).")
    relay = f"socat -T 60 UDP4-LISTEN:{DEFAULT_WEBRTC_RELAY_PORT},"
    return dedent_script(
        f"""\
        #!/bin/bash
        set -e
        command -v socat >/dev/null || {{ echo 'socat missing; resume this WebRTC instance first.'; exit 1; }}
        pkill -f {shell_quote('^' + relay)} 2>/dev/null || true
        setsid socat -T 60 \\
            UDP4-LISTEN:{DEFAULT_WEBRTC_RELAY_PORT},bind=0.0.0.0,reuseaddr,fork,range={client_ip}/32 \\
            UDP4:127.0.0.1:{DEFAULT_ISAAC_STREAM_PORT} \\
            </dev/null >/root/isaac_webrtc_relay.log 2>&1 &
        relay_pid=$!
        sleep 1
        kill -0 "$relay_pid" 2>/dev/null || {{ cat /root/isaac_webrtc_relay.log; exit 1; }}
        echo "$relay_pid"
        """
    )


def start_webrtc_relay(config: AppConfig, info: InstanceInfo, client_ip: str | None) -> int:
    assert info.ssh
    if client_ip is None:
        # This is the address seen by SSH, avoiding a third-party IP lookup.
        client_ip = run_ssh(config, info.ssh, 'printf "%s" "${SSH_CONNECTION%% *}"')
    script = build_webrtc_relay_script(client_ip)
    output = run_ssh_script(config, info.ssh, script, in_container=True, timeout_seconds=30)
    try:
        pid = int(output.splitlines()[-1])
        if pid <= 1:
            raise ValueError("invalid PID")
    except (ValueError, IndexError):
        _raise("Could not determine WebRTC relay PID; check /root/isaac_webrtc_relay.log.")
    typer.echo(f"WebRTC UDP access restricted to {client_ip}; use --client-ip if your UDP egress differs.")
    return pid


def stop_webrtc_relay(config: AppConfig, target: SshTarget, pid: int) -> None:
    # Kill only this session's process group, including socat's forked peers.
    pattern = shell_quote(f"^socat -T 60 UDP4-LISTEN:{DEFAULT_WEBRTC_RELAY_PORT},")
    run_ssh(
        config, target,
        f"if ps -p {pid} -o args= | grep -q {pattern}; then kill -- -{pid}; fi",
        in_container=True, timeout_seconds=20,
    )


def make_webrtc_http_server(
    port: int, connection: Callable[[], dict[str, Any]], directory: Path = WEBRTC_VIEWER_DIST,
) -> ThreadingHTTPServer:
    """Serve only compiled viewer assets and current connection info on loopback."""
    if not (directory / "index.html").is_file():
        _raise("Build the browser viewer first: npm --prefix web-viewer ci --ignore-scripts "
               "&& npm --prefix web-viewer run build")

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(directory), **kwargs)

        def do_GET(self):
            self.serve_request(send_body=True)

        def do_HEAD(self):
            self.serve_request(send_body=False)

        def serve_request(self, *, send_body: bool):
            # Reject DNS rebinding to the local viewer/config endpoint.
            bound_port = self.server.server_port
            if self.headers.get("Host") not in {f"127.0.0.1:{bound_port}", f"localhost:{bound_port}"}:
                self.send_error(403)
                return
            if self.path == "/connection.json":
                payload = json.dumps(connection()).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                if send_body:
                    self.wfile.write(payload)
            else:
                if send_body:
                    super().do_GET()
                else:
                    super().do_HEAD()

        def list_directory(self, path):
            self.send_error(404)
            return None

        def end_headers(self):
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            super().end_headers()

        def log_message(self, format, *args):
            pass

    class ViewerServer(ThreadingHTTPServer):
        # One viewer owns its port, including on Python versions that enable
        # SO_REUSEPORT by default in ThreadingHTTPServer.
        allow_reuse_port = False

    try:
        return ViewerServer(("127.0.0.1", port), Handler)
    except OSError as exc:
        _raise(f"Cannot start viewer on 127.0.0.1:{port}: {exc}. Use --viewer-port to change it.")


# ---------------------------------------------------------------------------
# Container-side setup scripts (validated in docs/VAST_EXPERIMENT_RESULTS.md and
# docs/GUI_TUNNEL_EXPERIMENT_PLAN.md; identical inside the Isaac image on every
# provider).
#
# Shell text is kept in plain (non f-) strings; per-launch values are passed as
# shell variables in a small generated header (see build_*_script).
# ---------------------------------------------------------------------------

GUI_X_DISPLAY = ":1"
GUI_VNC_PORT = 5901
GUI_STACK_PATH = "/root/gui_stack.sh"
GUI_KIT_LOG = "/root/isaac_gui.log"
GUI_SCREENSHOT_PATH = "/root/gui_screen.png"
# GUI kit readiness = a mapped "Isaac Sim" window (+ agent port when enabled).
# Cold boots that compile shaders map the window well before 8226 opens; the
# overall budget is generous, the window budget is not (a kit that has run this
# long without mapping raced the X display and never will).
GUI_STACK_TIMEOUT_S = 600
GUI_WINDOW_TIMEOUT_S = 240
# Hosts on driver 580 could not present Vulkan on the X display (measured
# 2026-09-01: kit "vkCreateSwapchainKHR failed", black GUI, headless fine).
GUI_MIN_DRIVER_MAJOR = 590

NVIDIA_USERLAND_SH = """\
ensure_nvidia_userland() {
    # A minority of Vast hosts inject compute-only NVIDIA libraries (no
    # Vulkan/GLX/NVENC userland). Side-load the exact driver-matched libs from
    # the Ubuntu archive once, and re-export the env on every run.
    ls /usr/lib/x86_64-linux-gnu/libGLX_nvidia.so.0 >/dev/null 2>&1 && return 0
    local LIBDIR=/opt/nvgl/usr/lib/x86_64-linux-gnu
    if [ ! -f /opt/nvgl/icd.json ]; then
        local DRIVER MAJOR deb
        DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | tr -d " ")
        MAJOR=${DRIVER%%.*}
        apt-get update -qq >/dev/null 2>&1
        (cd /tmp && apt-get download -qq libnvidia-gl-$MAJOR libnvidia-encode-$MAJOR libnvidia-decode-$MAJOR 2>/dev/null)
        mkdir -p /opt/nvgl
        for deb in /tmp/libnvidia-*-$MAJOR*.deb; do dpkg -x "$deb" /opt/nvgl; done
        printf '{"file_format_version":"1.0.0","ICD":{"library_path":"%s/libGLX_nvidia.so.0","api_version":"1.3.194"}}' "$LIBDIR" > /opt/nvgl/icd.json
        echo "$LIBDIR" > /etc/ld.so.conf.d/zz-nvgl.conf && ldconfig
    fi
    export VK_DRIVER_FILES=/opt/nvgl/icd.json VK_ICD_FILENAMES=/opt/nvgl/icd.json LD_LIBRARY_PATH=$LIBDIR
}
"""

# ffmpeg (which ships ffprobe) with the libx264 encoder: projects capture video
# clips of the robot from the sim with it. Installed on every launch/resume
# path (headless and GUI); a no-op once present.
VIDEO_TOOLS_SH = """\
video_tools_ready() {
    command -v ffmpeg >/dev/null 2>&1 && command -v ffprobe >/dev/null 2>&1 \\
        && ffmpeg -hide_banner -encoders 2>/dev/null | grep -q "libx264"
}
ensure_video_tools() {
    video_tools_ready && return 0
    echo "installing video tools (ffmpeg, ffprobe, libx264)"
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq >/dev/null 2>&1
    apt-get install -y -qq ffmpeg libx264-dev >/dev/null 2>&1 \\
        || { echo "WARN: apt-get install of ffmpeg/libx264 failed; video capture unavailable"; return 1; }
    video_tools_ready || { echo "WARN: ffmpeg installed but libx264 encoder or ffprobe missing"; return 1; }
    echo "video tools ready: $(ffmpeg -version 2>/dev/null | head -1)"
}
"""

# Shared by /root/gui_stack.sh (bring-up) and the status probe (checks only).
# Expects the header variables X_DISPLAY, VNC_PORT, NOVNC_PORT, AGENT_PORT,
# AGENT_ENABLED, GUI_RES, KIT_LOG, SCREENSHOT.
#
# Process guards: match the binary (`pgrep -x`) or a bracketed `-f` pattern,
# and never run a guard and the start it protects from ONE `ssh host '...'`
# command line: `pgrep -f "Xvf[b] :1" || Xvfb :1 ...` matches the remote
# shell's own argv (it contains "Xvfb :1"), so the service is silently never
# started. Likewise a `pkill -f x11vnc` in the same command line as the new
# supervisor's start text kills the new supervisor. These functions run from a
# script file (`bash /root/gui_stack.sh`) or `bash -s` on stdin, whose argv
# contains none of the service names, so the guards and kills here are safe.
GUI_FUNCTIONS_SH = """\
log() { echo "[gui_stack $(date +%H:%M:%S)] $*"; }
port_open() { timeout 3 bash -c "echo > /dev/tcp/127.0.0.1/$1" 2>/dev/null; }
x_up() { DISPLAY=$X_DISPLAY xdpyinfo >/dev/null 2>&1; }
kit_pids() { pgrep -f "[k]it/kit"; }
headless_kit_pids() { pgrep -f "[k]it/kit.*exp\\.full\\.streaming|[r]unheadless\\.sh"; }
kit_age() { ps -o etimes= -p "$(kit_pids | head -1)" 2>/dev/null | tr -d ' '; }
gui_window() {
    # Prints the id of a MAPPED "Isaac Sim" window. A kit that raced the X
    # display keeps an IsUnMapped window (and answers on the agent port), so
    # port 8226 alone is not readiness.
    local w
    for w in $(DISPLAY=$X_DISPLAY xdotool search --name "Isaac Sim" 2>/dev/null); do
        if DISPLAY=$X_DISPLAY xwininfo -id "$w" 2>/dev/null | grep -q "Map State: IsViewable"; then
            echo "$w"; return 0
        fi
    done
    return 1
}
vulkan_present_check() {
    # 0 = can present on the X display, 1 = cannot (dud for GUI work), 2 = unknown
    local out
    out=$(DISPLAY=$X_DISPLAY timeout 60 vulkaninfo --summary 2>&1)
    echo "$out" | grep -q "vkGetPhysicalDeviceSurfacePresentModesKHR failed" && return 1
    echo "$out" | grep -qi "deviceName\\|GPU id" && return 0
    return 2
}
screen_stats() {
    # Prints "<mean gray 0..1> <unique gray levels>" of the root window.
    command -v import >/dev/null 2>&1 || return 1
    DISPLAY=$X_DISPLAY timeout 30 import -window root "$SCREENSHOT" 2>/dev/null || return 1
    convert "$SCREENSHOT" -colorspace Gray -format "%[fx:mean] %k" info: 2>/dev/null
}
screen_is_lit() { awk -v m="${1:-0}" 'BEGIN{exit !(m > 0.01)}'; }
gui_check() {
    # One line per check; returns non-zero when a hard check fails.
    local ok=0 w stats
    if x_up; then
        echo "gui_x: up ($X_DISPLAY $(DISPLAY=$X_DISPLAY xdpyinfo 2>/dev/null | awk '/dimensions/{print $2}'))"
    else
        echo "gui_x: DOWN"; ok=1
    fi
    if x_up; then
        vulkan_present_check
        case $? in
            0) echo "gui_vulkan: can present on $X_DISPLAY";;
            1) echo "gui_vulkan: CANNOT PRESENT on $X_DISPLAY (driver too old; relaunch on driver >= 590)"; ok=1;;
            *) echo "gui_vulkan: unknown (vulkaninfo enumerated no GPU)";;
        esac
    fi
    if port_open $VNC_PORT; then
        if pgrep -f "[x]11vnc_loop" >/dev/null; then
            echo "gui_vnc: port $VNC_PORT open (supervised)"
        else
            echo "gui_vnc: port $VNC_PORT open (UNSUPERVISED x11vnc: dies on the first XIO error)"
        fi
    else
        echo "gui_vnc: port $VNC_PORT CLOSED"; ok=1
    fi
    if port_open $NOVNC_PORT; then echo "gui_novnc: port $NOVNC_PORT open"; else echo "gui_novnc: port $NOVNC_PORT CLOSED"; ok=1; fi
    if [ -z "$(kit_pids)" ]; then
        echo "gui_kit: NOT RUNNING"; ok=1
    elif [ -n "$(headless_kit_pids)" ]; then
        echo "gui_kit: headless streaming kit is running (no GUI window)"; ok=1
    elif w=$(gui_window); then
        echo "gui_kit: window mapped (id $w)"
    else
        echo "gui_kit: running, window NOT MAPPED$(grep -aq 'backbuffers are not initialized' "$KIT_LOG" 2>/dev/null && echo ' (raced the X display: backbuffers not initialized)')"; ok=1
    fi
    if [ "$AGENT_ENABLED" = 1 ]; then
        if port_open $AGENT_PORT; then echo "gui_agent: port $AGENT_PORT open"; else echo "gui_agent: port $AGENT_PORT CLOSED"; ok=1; fi
    fi
    if x_up && stats=$(screen_stats) && [ -n "$stats" ]; then
        set -- $stats
        if screen_is_lit "$1"; then
            echo "gui_screen: non-black (mean $1, $2 gray levels) $SCREENSHOT"
        else
            echo "gui_screen: BLACK (mean $1) $SCREENSHOT"; ok=1
        fi
    else
        echo "gui_screen: unavailable (X down or imagemagick missing)"
    fi
    return $ok
}
"""

GUI_STACK_MAIN_SH = """\
start_gui_kit() {
    [ -f "$KIT_LOG" ] && mv -f "$KIT_LOG" "$KIT_LOG.prev"
    log "starting the GUI kit on $X_DISPLAY"
    (
        export ACCEPT_EULA=Y PRIVACY_CONSENT=Y OMNI_KIT_ALLOW_ROOT=1 DISPLAY=$X_DISPLAY
        setsid nohup /isaac-sim/isaac-sim.sh --allow-root $KIT_EXTRA_ARGS </dev/null >"$KIT_LOG" 2>&1 &
    )
}
stop_kits() {
    pkill -f "[k]it/kit" 2>/dev/null
    local i
    for i in $(seq 1 20); do [ -z "$(kit_pids)" ] && return 0; sleep 1; done
    pkill -9 -f "[k]it/kit" 2>/dev/null; sleep 1
}
gui_stack_up() {
    local i w stats missing="" reason="" restarted=0 deadline
    export DEBIAN_FRONTEND=noninteractive
    for i in Xvfb x11vnc websockify xdotool xdpyinfo xwininfo vulkaninfo import; do
        command -v $i >/dev/null 2>&1 || missing="$missing $i"
    done
    [ -d /usr/share/novnc ] || missing="$missing novnc"
    if [ -n "$missing" ]; then
        log "installing GUI packages (missing:$missing)"
        apt-get update -qq >/dev/null 2>&1
        apt-get install -y -qq xvfb x11vnc novnc websockify xdotool x11-utils x11-apps vulkan-tools imagemagick >/dev/null 2>&1 \\
            || { log "FAIL: apt-get install of the GUI packages failed"; return 1; }
    fi
    ensure_video_tools || true
    ensure_nvidia_userland

    # 1. X display FIRST. A kit started before it exists answers on the agent
    #    port but never maps its window ("backbuffers are not initialized",
    #    "Hotkeys cannot be setup without a default window"; black VNC view).
    if ! x_up; then
        if pgrep -x Xvfb >/dev/null; then
            log "Xvfb is running but $X_DISPLAY does not answer; restarting it"
            pkill -x Xvfb; sleep 1
        fi
        log "starting Xvfb $X_DISPLAY ${GUI_RES}x24"
        setsid Xvfb $X_DISPLAY -screen 0 ${GUI_RES}x24 </dev/null >/var/log/xvfb.log 2>&1 &
        for i in $(seq 1 30); do x_up && break; sleep 1; done
        x_up || { log "FAIL: Xvfb $X_DISPLAY did not come up (see /var/log/xvfb.log)"; return 1; }
    fi
    log "X display $X_DISPLAY up"

    # 2. Vulkan presentation preflight: fail fast instead of composing into a black window.
    vulkan_present_check
    case $? in
        1)
            log "FAIL: Vulkan cannot present on this host's X display (vulkaninfo: vkGetPhysicalDeviceSurfacePresentModesKHR failed)."
            log "Seen on driver 580 hosts: headless works, the GUI stays black. Relaunch on a driver >= 590 host."
            echo GUI_STACK_VULKAN_PRESENT_FAILED
            return 2;;
        2) log "warning: vulkaninfo enumerated no GPU; continuing";;
        *) log "Vulkan presentation preflight ok";;
    esac

    # 3. websockify serves noVNC on NOVNC_PORT and bridges to the VNC port.
    if ! port_open $NOVNC_PORT; then
        pkill -x websockify 2>/dev/null
        log "starting websockify $NOVNC_PORT -> $VNC_PORT"
        setsid websockify --web /usr/share/novnc 127.0.0.1:$NOVNC_PORT localhost:$VNC_PORT </dev/null >/var/log/websockify.log 2>&1 &
        for i in $(seq 1 15); do port_open $NOVNC_PORT && break; sleep 1; done
        port_open $NOVNC_PORT || { log "FAIL: websockify did not open $NOVNC_PORT (see /var/log/websockify.log)"; return 1; }
    fi
    log "noVNC on $NOVNC_PORT"

    # 4. x11vnc under supervision: a bare x11vnc dies on the first XIO error
    #    and freezes the viewer; -noxdamage matters for the kit's GL surface.
    if ! pgrep -f "[x]11vnc_loop" >/dev/null || ! port_open $VNC_PORT; then
        # Kills the bare launcher x11vnc and any older supervisor loop. Safe
        # here because this script's argv contains no "x11vnc" (see header).
        pkill -f "x11vnc" 2>/dev/null; sleep 1
        cat > /root/x11vnc_loop.sh <<EOS
#!/bin/bash
# supervised x11vnc (generated by isaac_cloud.py gui_stack.sh)
while true; do
    x11vnc -display $X_DISPLAY -localhost -forever -shared -nopw -noxdamage -rfbport $VNC_PORT </dev/null >>/var/log/x11vnc.log 2>&1
    sleep 2
done
EOS
        chmod +x /root/x11vnc_loop.sh
        log "starting supervised x11vnc on $VNC_PORT"
        setsid nohup /root/x11vnc_loop.sh </dev/null >/dev/null 2>&1 &
        for i in $(seq 1 15); do port_open $VNC_PORT && break; sleep 1; done
        port_open $VNC_PORT || { log "FAIL: x11vnc did not open $VNC_PORT (see /var/log/x11vnc.log)"; return 1; }
    fi
    log "VNC on $VNC_PORT (supervised)"

    # 5. The GUI kit. One GPU and one agent port: the headless streaming kit
    #    must not run alongside it.
    if [ -n "$(headless_kit_pids)" ]; then
        log "stopping the headless streaming kit (the GUI kit replaces it)"
        stop_kits
    fi
    [ -z "$(kit_pids)" ] && start_gui_kit
    deadline=$(( $(date +%s) + GUI_TIMEOUT ))
    while :; do
        if w=$(gui_window); then
            if [ "$AGENT_ENABLED" != 1 ] || port_open $AGENT_PORT; then break; fi
        elif [ -z "$(kit_pids)" ]; then
            reason="exited (tail $KIT_LOG)"
        elif grep -aq "backbuffers are not initialized" "$KIT_LOG" 2>/dev/null; then
            reason="never mapped its window (backbuffers not initialized: it raced the X display)"
        elif [ "$(kit_age)" -gt "$WINDOW_TIMEOUT" ] 2>/dev/null; then
            reason="has no mapped window after ${WINDOW_TIMEOUT}s"
        fi
        if [ -n "$reason" ]; then
            if [ "$restarted" = 1 ]; then
                log "FAIL: GUI kit $reason, even after a restart; see $KIT_LOG"; return 1
            fi
            log "GUI kit $reason; restarting it"
            restarted=1; reason=""
            stop_kits
            start_gui_kit
            continue
        fi
        if [ "$(date +%s)" -ge "$deadline" ]; then
            log "FAIL: GUI kit not ready within ${GUI_TIMEOUT}s (window $(gui_window >/dev/null && echo mapped || echo unmapped), agent port $(port_open $AGENT_PORT && echo open || echo closed)); see $KIT_LOG"
            return 1
        fi
        sleep 5
    done
    log "GUI kit ready: window $w mapped$([ "$AGENT_ENABLED" = 1 ] && echo ", agent port $AGENT_PORT open")"
    DISPLAY=$X_DISPLAY xdotool windowmove "$w" 0 0 windowsize "$w" ${GUI_RES%x*} ${GUI_RES#*x} 2>/dev/null
    for i in $(seq 1 12); do
        stats=$(screen_stats); set -- $stats
        screen_is_lit "${1:-0}" && break
        sleep 5
    done
    if gui_check; then echo GUI_STACK_READY; return 0; fi
    log "FAIL: post-start checks failed (see above)"
    return 1
}
case "${1:-up}" in
    up) gui_stack_up;;
    check) gui_check;;
    *) echo "usage: $0 [up|check]"; exit 64;;
esac
"""


def build_gui_header(config: AppConfig) -> str:
    """Shell variables the GUI functions read; one place for every per-launch value."""
    agent_flag = "--enable isaacsim.code_editor.python_server" if config.agent_enabled else ""
    return dedent_script(
        f"""\
        X_DISPLAY={GUI_X_DISPLAY}
        VNC_PORT={GUI_VNC_PORT}
        NOVNC_PORT={DEFAULT_NOVNC_PORT}
        AGENT_PORT={DEFAULT_AGENT_CONTROL_PORT}
        AGENT_ENABLED={1 if config.agent_enabled else 0}
        GUI_RES={config.gui_resolution}
        KIT_LOG={GUI_KIT_LOG}
        SCREENSHOT={GUI_SCREENSHOT_PATH}
        KIT_EXTRA_ARGS="{agent_flag}"
        GUI_TIMEOUT=${{GUI_STACK_TIMEOUT:-{GUI_STACK_TIMEOUT_S}}}
        WINDOW_TIMEOUT=${{GUI_STACK_WINDOW_TIMEOUT:-{GUI_WINDOW_TIMEOUT_S}}}
        """
    )


def build_gui_stack_script(config: AppConfig) -> str:
    """The contents of /root/gui_stack.sh: idempotent, ordered GUI bring-up.

    `gui_stack.sh [up]` brings the stack up (or verifies it if already up):
      apt deps (+ ffmpeg/libx264) -> Xvfb :1 (wait for xdpyinfo) -> Vulkan present preflight
      -> websockify/noVNC -> supervised x11vnc -> GUI kit (killing a headless
      kit first) -> wait for 8226 AND a mapped window -> checks.
    `gui_stack.sh check` runs only the checks (what `status` reports).

    Every step is guarded by a process/port check so re-running (resume, the
    project's own relaunch scripts) only starts what is missing. Each failure
    mode encoded here was observed live on Vast hosts, 2026-09-01..03.
    """
    return (
        "#!/bin/bash\n"
        "# Generated by isaac_cloud.py -- the noVNC GUI stack for the Isaac container.\n"
        "set -u\n"
        + build_gui_header(config)
        + NVIDIA_USERLAND_SH
        + VIDEO_TOOLS_SH
        + GUI_FUNCTIONS_SH
        + GUI_STACK_MAIN_SH
    )


def build_gui_stack_install_script(config: AppConfig) -> str:
    """Write /root/gui_stack.sh on the box and run it in the foreground."""
    body = build_gui_stack_script(config)
    assert "GUI_STACK_EOF" not in body
    return (
        "#!/bin/bash\n"
        f"cat > {GUI_STACK_PATH} <<'GUI_STACK_EOF'\n"
        f"{body}"
        "GUI_STACK_EOF\n"
        f"chmod +x {GUI_STACK_PATH}\n"
        f"exec bash {GUI_STACK_PATH} up\n"
    )


def build_isaac_container_launch_script(config: AppConfig) -> str:
    """Launch headless streaming Isaac inside the container.

    Env vars must be exported in-shell on Vast hosts, and compute-only hosts
    need the NVIDIA userland side-loaded (ensure_nvidia_userland).
    """
    agent_flag = " --enable isaacsim.code_editor.python_server" if config.agent_enabled else ""
    return (
        "#!/bin/bash\n"
        + NVIDIA_USERLAND_SH
        + VIDEO_TOOLS_SH
        + dedent_script(
            f"""\
            set -x
            export ACCEPT_EULA=Y PRIVACY_CONSENT=Y OMNI_KIT_ALLOW_ROOT=1
            export ISAACSIM_HOST=127.0.0.1
            export ISAACSIM_SIGNAL_PORT={DEFAULT_ISAAC_SIGNAL_PORT}
            export ISAACSIM_STREAM_PORT={DEFAULT_ISAAC_STREAM_PORT}
            ensure_video_tools || true
            ensure_nvidia_userland
            pkill -f "[k]it/kit" 2>/dev/null; sleep 2
            nohup /isaac-sim/runheadless.sh -v{agent_flag} > /root/isaac.log 2>&1 &
            echo LAUNCHED
            """
        )
    )


def build_curobo_install_script() -> str:
    """Install cuRobo into Isaac's bundled python, in the background.

    Validated 2026-08: current cuRobo runs its kernels on warp-lang via the
    cuda.core backend — no CUDA toolkit and no nvcc compile needed. Two traps:
    the editable install (`pip -e`) mis-maps the package root, and without
    `cuda-core[cu12]` collision kernels fail with "No curobo kernel backend".
    Writes /root/curobo_install.log; final line is CUROBO_INSTALL_OK.
    """
    return dedent_script(
        """\
        #!/bin/bash
        cat > /root/curobo_install.sh << 'EOS'
        #!/bin/bash
        set -e
        if /isaac-sim/python.sh -c "import curobo" >/dev/null 2>&1; then
            echo CUROBO_ALREADY_INSTALLED
            exit 0
        fi
        command -v git >/dev/null 2>&1 || { apt-get update -qq && apt-get install -y -qq git; }
        /isaac-sim/python.sh -m pip install --quiet ninja wheel
        /isaac-sim/python.sh -m pip install --quiet torch --index-url https://download.pytorch.org/whl/cu128
        rm -rf /root/curobo
        git clone --depth 1 https://github.com/NVlabs/curobo.git /root/curobo
        cd /root/curobo
        /isaac-sim/python.sh -m pip install . --no-build-isolation
        /isaac-sim/python.sh -m pip install --quiet 'cuda-core[cu12]'
        cd /root
        /isaac-sim/python.sh -c "from curobo.motion_planner import MotionPlanner; print('CUROBO_INSTALL_OK')"
        EOS
        chmod +x /root/curobo_install.sh
        nohup bash /root/curobo_install.sh > /root/curobo_install.log 2>&1 &
        echo CUROBO_INSTALL_LAUNCHED
        """
    )


def provision_curobo(config: AppConfig, info: InstanceInfo) -> None:
    """Kick off the background cuRobo install (returns immediately)."""
    assert info.ssh
    output = run_ssh_script(
        config, info.ssh, build_curobo_install_script(), in_container=True, timeout_seconds=60
    )
    typer.echo(output.splitlines()[-1] if output else "(no output)")
    typer.echo(
        "cuRobo installing in background (~5 min); check with: "
        "tail /root/curobo_install.log (expects CUROBO_INSTALL_OK)"
    )


def build_lab_install_script(lab_ref: str) -> str:
    """Install Isaac Lab into Isaac's bundled python, in the background.

    Clones IsaacLab at `lab_ref` (a git tag or branch, paired with the Isaac
    Sim version — see [isaac].lab_ref) and runs its own installer against
    /isaac-sim (found via the _isaac_sim symlink). Note the usage model: Lab scripts launch their own SimulationApp,
    so stop the streaming kit process (pkill -f kit/kit) before running Lab
    workloads, and keep outputs under /isaac-sim/project to be snapshotted.
    Writes /root/isaac_lab_install.log; final line is ISAAC_LAB_INSTALL_OK.
    """
    return dedent_script(
        f"""\
        #!/bin/bash
        cat > /root/isaac_lab_install.sh << 'EOS'
        #!/bin/bash
        set -e
        if /isaac-sim/python.sh -c "import isaaclab" >/dev/null 2>&1; then
            echo ISAAC_LAB_ALREADY_INSTALLED
            exit 0
        fi
        command -v git >/dev/null 2>&1 || {{ apt-get update -qq && apt-get install -y -qq git; }}
        rm -rf /root/IsaacLab
        git clone --depth 1 --branch {shell_quote(lab_ref)} \\
            https://github.com/isaac-sim/IsaacLab.git /root/IsaacLab
        ln -sfn /isaac-sim /root/IsaacLab/_isaac_sim
        cd /root/IsaacLab
        export ISAACSIM_PATH=/isaac-sim
        ./isaaclab.sh --install
        /isaac-sim/python.sh -c "import isaaclab; print('ISAAC_LAB_INSTALL_OK')"
        EOS
        chmod +x /root/isaac_lab_install.sh
        nohup bash /root/isaac_lab_install.sh > /root/isaac_lab_install.log 2>&1 &
        echo ISAAC_LAB_INSTALL_LAUNCHED
        """
    )


def provision_lab(config: AppConfig, info: InstanceInfo) -> None:
    """Kick off the background Isaac Lab install (returns immediately)."""
    assert info.ssh
    output = run_ssh_script(
        config,
        info.ssh,
        build_lab_install_script(config.lab_ref),
        in_container=True,
        timeout_seconds=60,
    )
    typer.echo(output.splitlines()[-1] if output else "(no output)")
    typer.echo(
        "Isaac Lab installing in background (~15 min); check with: "
        "tail /root/isaac_lab_install.log (expects ISAAC_LAB_INSTALL_OK)"
    )


def build_container_probe_script(config: AppConfig) -> str:
    """Report readiness of everything we care about inside the container.

    When the GUI stack was ever installed on the box (or an Xvfb is running),
    also runs the stack's own checks: X up, Vulkan can present, VNC and noVNC
    ports open, kit window mapped, agent port, non-black screenshot.
    """
    return (
        "#!/bin/bash\n"
        + build_gui_header(config)
        + VIDEO_TOOLS_SH
        + GUI_FUNCTIONS_SH
        + dedent_script(
            f"""\
            echo "gpu: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -1)"
            echo "gpu_minor: $(nvidia-smi -q 2>/dev/null | grep -i 'Minor Number' | awk '{{print $NF}}' | head -1)"
            echo "kit_procs: $(pgrep -c '[k]it' 2>/dev/null || echo 0)"
            for log in /root/isaac.log {GUI_KIT_LOG}; do
                if [ -f "$log" ]; then
                    grep -qm1 -E "Streaming App is loaded|app ready" "$log" && echo "$(basename $log): ready" || echo "$(basename $log): loading"
                fi
            done
            if [ -f /root/curobo_install.log ]; then
                grep -qm1 -E "CUROBO_INSTALL_OK|CUROBO_ALREADY_INSTALLED" /root/curobo_install.log && echo "curobo: ready" || echo "curobo: installing"
            fi
            if [ -f /root/isaac_lab_install.log ]; then
                grep -qm1 -E "ISAAC_LAB_INSTALL_OK|ISAAC_LAB_ALREADY_INSTALLED" /root/isaac_lab_install.log && echo "isaac_lab: ready" || echo "isaac_lab: installing"
            fi
            video_tools_ready && echo "video_tools: ready (ffmpeg/ffprobe with libx264)" || echo "video_tools: MISSING (ffmpeg/ffprobe/libx264)"
            if pgrep -f '^socat -T 60 UDP4-LISTEN:{DEFAULT_WEBRTC_RELAY_PORT},' >/dev/null; then
                echo "webrtc relay: running (UDP; confirm video in the browser)"
            fi
            for p in {DEFAULT_AGENT_CONTROL_PORT} {DEFAULT_ISAAC_SIGNAL_PORT} {DEFAULT_RTSP_PORT} {DEFAULT_NOVNC_PORT}; do
                (echo > /dev/tcp/127.0.0.1/$p) 2>/dev/null && echo "port $p: open" || echo "port $p: closed"
            done
            if [ -x {GUI_STACK_PATH} ] || pgrep -x Xvfb >/dev/null 2>&1; then
                echo "gui_stack: $([ -x {GUI_STACK_PATH} ] && echo installed || echo 'not installed (Xvfb running from an older launcher)')"
                gui_check
            fi
            exit 0
            """
        )
    )


def remote_gui_stack_installed(config: AppConfig, info: InstanceInfo) -> bool:
    """Was this box brought up with the GUI stack? (/root persists across stop/start.)"""
    assert info.ssh
    out = run_ssh(
        config,
        info.ssh,
        f"test -x {GUI_STACK_PATH} && echo yes || echo no",
        in_container=True,
        check=False,
    )
    return out.strip().endswith("yes")


# ---------------------------------------------------------------------------
# Vast.ai provider
# ---------------------------------------------------------------------------


def _vastai_bin() -> str:
    found = shutil.which("vastai") or shutil.which(
        str(Path.home() / ".local" / "bin" / "vastai")
    )
    if not found:
        candidate = Path.home() / ".local" / "bin" / "vastai"
        if candidate.exists():
            return str(candidate)
        _raise("vastai CLI not found. Install with: uv tool install vastai; then: vastai set api-key <KEY>")
    return found


def run_vastai_json(args: list[str], *, timeout_seconds: int = 60) -> Any:
    return run_cli(
        [_vastai_bin(), *args, "--raw"],
        timeout_seconds=timeout_seconds,
        error_prefix=f"vastai {' '.join(args[:2])} failed",
        parse_json=True,
    )


class VastProvisionMonitor(ProvisionMonitor):
    """Fail-fast detection for Vast (measured: docs/VAST_TIMEOUT_EXPERIMENT_RESULTS.md).

    Vast surfaces docker's own error text in status_msg while a doomed
    instance sits in "loading" forever; auth denial after "running" means the
    host missed the account-key injection and never self-heals."""

    def __init__(self, provider: "VastProvider") -> None:
        self.provider = provider
        self._daemon_error_since: float | None = None
        self._auth_denied_since: float | None = None
        self._key_reattached = False

    def check_status(self, info: InstanceInfo) -> None:
        msg = str(info.raw.get("status_msg") or "").strip()
        low = msg.lower()
        if any(pat in low for pat in VAST_FATAL_STATUS_PATTERNS):
            raise ProvisioningDoomed(
                f"Provisioning of {info.instance_id} failed (image pull cannot succeed): "
                f"{msg[:300]}. Retry with another offer (or fix credentials/image)."
            )
        if VAST_DAEMON_ERROR_MARKER in low:
            self._daemon_error_since = self._daemon_error_since or time.time()
            if time.time() - self._daemon_error_since >= VAST_DAEMON_ERROR_FATAL_S:
                raise ProvisioningDoomed(
                    f"Provisioning of {info.instance_id} stuck on a docker daemon error "
                    f"for {VAST_DAEMON_ERROR_FATAL_S}s: {msg[:300]}"
                )
        else:
            self._daemon_error_since = None

    def auth_denied(self, info: InstanceInfo) -> None:
        now = time.time()
        self._auth_denied_since = self._auth_denied_since or now
        denied_for = now - self._auth_denied_since
        if not self._key_reattached and denied_for >= VAST_SSH_DENIED_ATTACH_S:
            typer.echo(
                "SSH key auth denied after instance start; "
                "re-attaching the key to the instance..."
            )
            self.provider.attach_ssh_key(info.instance_id)
            self._key_reattached = True
        elif denied_for >= VAST_SSH_DENIED_GIVE_UP_S:
            raise ProvisioningDoomed(
                f"Instance {info.instance_id} is running but has refused pubkey "
                f"auth for {int(denied_for)}s (host failed to inject the SSH "
                f"key). This does not self-heal: rent another offer."
            )

    def auth_reset(self) -> None:
        self._auth_denied_since = None


def driver_major(version: Any) -> int:
    """'580.95.05' -> 580; unparseable -> 0."""
    match = re.match(r"\s*(\d+)", str(version or ""))
    return int(match.group(1)) if match else 0


class VastProvider(Provider):
    name = "vast"

    def _query(self) -> str:
        query = self.config.vast_query
        if self.config.vast_whole_machine:
            query += " gpu_frac=1"
        return query

    def catalog(self, limit: int = 15) -> list[dict[str, Any]]:
        offers = run_vastai_json(["search", "offers", self._query(), "-o", "dph"])
        offers = [
            o
            for o in offers
            if float(o.get("reliability2", 0)) >= self.config.vast_min_reliability
        ]
        if self.config.gui_enabled:
            # GUI work needs Vulkan presentation on the X display, which driver
            # 580 hosts could not do; rank driver >= 590 first (price order kept
            # within each group).
            offers.sort(key=lambda o: driver_major(o.get("driver_version")) < GUI_MIN_DRIVER_MAJOR)
        return offers[:limit]

    def launch(self, offer_id: str | None = None) -> InstanceInfo:
        if not self.config.ngc_api_key:
            _raise("Missing NGC API key ([ngc].api_key) — required to pull the Isaac image.")
        if offer_id:
            # User-picked offer (e.g. from `catalog` output): rent it as-is,
            # skipping the query/reliability filters. Vast rejects the create
            # if the offer is gone or taken.
            chosen = offer_id
            typer.echo(f"Renting offer {chosen} (explicitly requested).")
        else:
            offers = self.catalog(limit=1)
            if not offers:
                _raise("No Vast offers matched the query. Loosen [vast].query or min_reliability.")
            offer = offers[0]
            chosen = str(offer["id"])
            typer.echo(
                f"Renting offer {chosen}: {offer.get('gpu_name')} "
                f"driver {offer.get('driver_version')} at ${offer.get('dph_total', 0):.3f}/hr "
                f"({offer.get('geolocation')})"
            )
            if (
                self.config.gui_enabled
                and driver_major(offer.get("driver_version")) < GUI_MIN_DRIVER_MAJOR
            ):
                typer.echo(
                    f"Warning: no driver >= {GUI_MIN_DRIVER_MAJOR} offer matched; driver "
                    f"{offer.get('driver_version')} hosts may be unable to present the GUI "
                    "(the Vulkan preflight will abort the launch if so)."
                )
        image = build_isaac_image_ref(self.config.isaac_version)
        result = run_vastai_json(
            [
                "create",
                "instance",
                chosen,
                "--image",
                image,
                "--login",
                f"-u $oauthtoken -p {self.config.ngc_api_key} nvcr.io",
                "--disk",
                str(self.config.disk_gb),
                "--onstart-cmd",
                "sleep infinity",
                "--ssh",
                "--direct",
                *(["--env", f"-p {DEFAULT_WEBRTC_RELAY_PORT}:{DEFAULT_WEBRTC_RELAY_PORT}/udp"]
                  if self.config.webrtc_enabled else []),
            ],
            timeout_seconds=120,
        )
        if not result.get("success"):
            _raise(f"Vast create failed: {result}")
        instance_id = str(result["new_contract"])
        self._ensure_account_ssh_key()
        return InstanceInfo(
            provider=self.name, instance_id=instance_id, status="loading", label="", ssh=None
        )

    def _ensure_account_ssh_key(self) -> None:
        # Account-level keys are injected at container creation; per-instance
        # attach races the container start on some hosts.
        pub = read_public_key(self.config)
        if pub:
            run_cli_quiet([_vastai_bin(), "create", "ssh-key", pub], timeout_seconds=30)

    def list_instances(self) -> list[InstanceInfo]:
        rows = run_vastai_json(["show", "instances"])
        return [self._to_info(row) for row in rows]

    def get(self, instance_id: str) -> InstanceInfo:
        row = run_vastai_json(["show", "instance", instance_id])
        return self._to_info(row)

    def _to_info(self, row: dict[str, Any]) -> InstanceInfo:
        ssh = None
        ports = (row.get("ports") or {}).get("22/tcp") or []
        host_port = ports[0].get("HostPort") if ports else None
        ip = row.get("public_ipaddr")
        if ip and host_port:
            ssh = SshTarget(host=ip, port=int(host_port), user="root")
        return InstanceInfo(
            provider=self.name,
            instance_id=str(row.get("id")),
            status=str(row.get("actual_status") or "unknown"),
            label=str(row.get("label") or ""),
            ssh=ssh,
            raw=row,
        )

    def provision_monitor(self) -> ProvisionMonitor:
        return VastProvisionMonitor(self)

    def set_project_label(self, instance_id: str, project: str) -> None:
        # Best-effort: records which project this instance was launched for so
        # stop/destroy save back to the same project without --project.
        run_cli_quiet(
            [_vastai_bin(), "label", "instance", instance_id, f"{PROJECT_LABEL_PREFIX}{project}"],
            timeout_seconds=30,
        )

    def attach_ssh_key(self, instance_id: str) -> None:
        """Recovery for hosts that miss the account-key injection at create."""
        pub = read_public_key(self.config)
        if pub:
            run_cli_quiet([_vastai_bin(), "attach", "ssh", instance_id, pub], timeout_seconds=45)

    def stop(self, instance_id: str) -> None:
        # `vastai stop instance` prints plain text even with --raw; don't parse
        # JSON, but do surface failures — a silently-failed stop keeps billing.
        run_cli(
            [_vastai_bin(), "stop", "instance", instance_id],
            timeout_seconds=60,
            error_prefix=f"vastai stop instance {instance_id} failed",
        )

    def start(self, instance_id: str) -> None:
        run_cli(
            [_vastai_bin(), "start", "instance", instance_id],
            timeout_seconds=60,
            error_prefix=f"vastai start instance {instance_id} failed",
        )

    def destroy(self, instance_id: str) -> None:
        run_cli_quiet([_vastai_bin(), "destroy", "instance", instance_id, "-y"], timeout_seconds=60)

    def persistence_remote_path(self) -> str:
        # No volume mounts in the Vast container model; the project lives on
        # the container filesystem and durability comes from S3 sync.
        return CONTAINER_PERSISTENCE_DIR

    def remote_sudo(self) -> str:
        return ""  # container user is root; sudo is absent in the image


# ---------------------------------------------------------------------------
# AWS provider
# ---------------------------------------------------------------------------


def run_aws_json(config: AppConfig, args: list[str], *, timeout_seconds: int = 120) -> Any:
    return run_cli(
        ["aws", *args, "--region", config.aws_region, "--output", "json"],
        timeout_seconds=timeout_seconds,
        error_prefix=f"aws {' '.join(args[:3])} failed",
        parse_json=True,
    )


class AwsProvider(Provider):
    name = "aws"

    def open_webrtc_access(self, info: InstanceInfo, client_ip: str) -> Callable[[], None]:
        # Use an attached group, never change group membership or expose signaling.
        client_ip = str(ipaddress.IPv4Address(client_ip))
        groups = info.raw.get("SecurityGroups") or []
        if not groups:
            _raise("AWS instance has no attached security group for WebRTC UDP access.")
        group_id = groups[0]["GroupId"]
        permissions = json.dumps([{
            "IpProtocol": "udp",
            "FromPort": DEFAULT_WEBRTC_RELAY_PORT,
            "ToPort": DEFAULT_WEBRTC_RELAY_PORT,
            "IpRanges": [{"CidrIp": f"{client_ip}/32",
                          "Description": f"isaac-cloud WebRTC {info.instance_id}"}],
        }])
        try:
            result = run_aws_json(self.config, [
                "ec2", "authorize-security-group-ingress", "--group-id", group_id,
                "--ip-permissions", permissions,
            ])
        except IsaacCloudError as exc:
            if "InvalidPermission.Duplicate" in str(exc):
                raise IsaacCloudError(
                    f"UDP {DEFAULT_WEBRTC_RELAY_PORT} from {client_ip}/32 already exists in {group_id}. "
                    "Stop the other viewer or remove a stale rule before retrying; "
                    "existing rules are not modified."
                ) from exc
            raise
        rule_ids = [rule["SecurityGroupRuleId"] for rule in result.get("SecurityGroupRules", [])]
        typer.echo(f"AWS WebRTC ingress: {group_id}, UDP {DEFAULT_WEBRTC_RELAY_PORT} from {client_ip}/32.")

        def close() -> None:
            selector = (["--security-group-rule-ids", *rule_ids] if rule_ids
                        else ["--ip-permissions", permissions])
            run_aws_json(self.config, [
                "ec2", "revoke-security-group-ingress", "--group-id", group_id, *selector,
            ])

        return close

    def resolve_ami(self) -> str:
        result = run_aws_json(
            self.config,
            ["ssm", "get-parameter", "--name", self.config.aws_ami_ssm_param],
        )
        return result["Parameter"]["Value"]

    def _ensure_key_pair(self) -> str:
        name = self.config.aws_key_name
        existing = run_aws_json(
            self.config,
            ["ec2", "describe-key-pairs", "--filters", f"Name=key-name,Values={name}"],
        )
        if not existing.get("KeyPairs"):
            if not self.config.ssh_public_key_path:
                _raise("Set [ssh].public_key_path so the key can be imported to EC2.")
            pub = Path(self.config.ssh_public_key_path).expanduser().read_text().strip()
            run_aws_json(
                self.config,
                [
                    "ec2",
                    "import-key-pair",
                    "--key-name",
                    name,
                    "--public-key-material",
                    base64.b64encode(pub.encode()).decode("ascii"),
                ],
            )
        return name

    def _ensure_security_group(self) -> str:
        name = self.config.aws_security_group
        found = run_aws_json(
            self.config,
            ["ec2", "describe-security-groups", "--filters", f"Name=group-name,Values={name}"],
        )
        groups = found.get("SecurityGroups") or []
        if groups:
            return groups[0]["GroupId"]
        created = run_aws_json(
            self.config,
            [
                "ec2",
                "create-security-group",
                "--group-name",
                name,
                "--description",
                "isaac-cloud SSH-only access",
            ],
        )
        group_id = created["GroupId"]
        run_aws_json(
            self.config,
            [
                "ec2",
                "authorize-security-group-ingress",
                "--group-id",
                group_id,
                "--protocol",
                "tcp",
                "--port",
                "22",
                "--cidr",
                "0.0.0.0/0",
            ],
        )
        return group_id

    def _build_user_data(self) -> str:
        image = build_isaac_image_ref(self.config.isaac_version)
        persist_dir = "/home/ubuntu/isaac-cloud/project"
        return dedent_script(
            f"""\
            #!/bin/bash
            set -euxo pipefail
            exec > >(tee -a /var/log/isaac-cloud-bootstrap.log) 2>&1

            # DL Base AMI ships driver+docker+nvidia-container-toolkit. Isaac 6
            # needs driver >= 580; upgrade if the AMI is behind.
            DRIVER_MAJOR=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
            if [ "${{DRIVER_MAJOR:-0}}" -lt 580 ]; then
                export DEBIAN_FRONTEND=noninteractive
                apt-get update -qq && apt-get install -y nvidia-driver-580
                reboot
            fi

            mkdir -p {persist_dir}
            chmod -R a+rwX /home/ubuntu/isaac-cloud

            printf '%s\\n' {shell_quote(self.config.ngc_api_key or "")} | docker login nvcr.io --username '$oauthtoken' --password-stdin
            docker pull {shell_quote(image)}
            docker rm -f isaac-sim 2>/dev/null || true
            docker run -d --name isaac-sim --gpus all --network=host --restart unless-stopped \\
                --entrypoint bash \\
                -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y \\
                -v {persist_dir}:{CONTAINER_PERSISTENCE_DIR}:rw \\
                {shell_quote(image)} \\
                -c "sleep infinity"
            """
        )

    def launch(self, offer_id: str | None = None) -> InstanceInfo:
        if offer_id:
            _raise("--offer-id is a Vast.ai concept; on aws set [aws].instance_type instead.")
        if not self.config.ngc_api_key:
            _raise("Missing NGC API key ([ngc].api_key) — required to pull the Isaac image.")
        ami = self.resolve_ami()
        key_name = self._ensure_key_pair()
        group_id = self._ensure_security_group()
        name = f"{self.config.instance_name_prefix}-{int(time.time())}"
        typer.echo(
            f"Launching {self.config.aws_instance_type} in {self.config.aws_region} (AMI {ami})"
        )
        result = run_aws_json(
            self.config,
            [
                "ec2",
                "run-instances",
                "--image-id",
                ami,
                "--instance-type",
                self.config.aws_instance_type,
                "--key-name",
                key_name,
                "--security-group-ids",
                group_id,
                "--block-device-mappings",
                json.dumps(
                    [
                        {
                            "DeviceName": "/dev/sda1",
                            "Ebs": {"VolumeSize": self.config.disk_gb, "VolumeType": "gp3"},
                        }
                    ]
                ),
                "--tag-specifications",
                json.dumps(
                    [
                        {
                            "ResourceType": "instance",
                            "Tags": [
                                {"Key": "Name", "Value": name},
                                {"Key": AWS_TAG_MANAGED, "Value": "true"},
                                *([{"Key": AWS_TAG_WEBRTC, "Value": "true"}]
                                  if self.config.webrtc_enabled else []),
                            ],
                        }
                    ]
                ),
                "--user-data",
                self._build_user_data(),
            ],
            timeout_seconds=180,
        )
        instance_id = result["Instances"][0]["InstanceId"]
        return InstanceInfo(
            provider=self.name, instance_id=instance_id, status="pending", label=name, ssh=None
        )

    def list_instances(self) -> list[InstanceInfo]:
        result = run_aws_json(
            self.config,
            [
                "ec2",
                "describe-instances",
                "--filters",
                f"Name=tag:{AWS_TAG_MANAGED},Values=true",
                "Name=instance-state-name,Values=pending,running,stopping,stopped",
            ],
        )
        infos = []
        for reservation in result.get("Reservations", []):
            for inst in reservation.get("Instances", []):
                infos.append(self._to_info(inst))
        return infos

    def get(self, instance_id: str) -> InstanceInfo:
        result = run_aws_json(
            self.config, ["ec2", "describe-instances", "--instance-ids", instance_id]
        )
        inst = result["Reservations"][0]["Instances"][0]
        return self._to_info(inst)

    def _to_info(self, inst: dict[str, Any]) -> InstanceInfo:
        ip = inst.get("PublicIpAddress")
        ssh = (
            SshTarget(host=ip, port=22, user=DEFAULT_AWS_SSH_USER, container_via_docker=True)
            if ip
            else None
        )
        name = next(
            (t["Value"] for t in inst.get("Tags", []) if t["Key"] == "Name"), ""
        )
        return InstanceInfo(
            provider=self.name,
            instance_id=inst["InstanceId"],
            status=inst.get("State", {}).get("Name", "unknown"),
            label=name,
            ssh=ssh,
            raw=inst,
        )

    def set_project_label(self, instance_id: str, project: str) -> None:
        # Best-effort tag; read back by resolve_project via the raw Tags list.
        run_cli_quiet(
            ["aws", "ec2", "create-tags", "--resources", instance_id,
             "--tags", f"Key={AWS_TAG_PROJECT},Value={project}",
             "--region", self.config.aws_region],
            timeout_seconds=60,
        )

    def stop(self, instance_id: str) -> None:
        run_aws_json(self.config, ["ec2", "stop-instances", "--instance-ids", instance_id])

    def start(self, instance_id: str) -> None:
        run_aws_json(self.config, ["ec2", "start-instances", "--instance-ids", instance_id])

    def destroy(self, instance_id: str) -> None:
        run_aws_json(self.config, ["ec2", "terminate-instances", "--instance-ids", instance_id])

    def persistence_remote_path(self) -> str:
        return "/home/ubuntu/isaac-cloud/project"

    def remote_sudo(self) -> str:
        return "sudo "


PROVIDERS = {"vast": VastProvider, "aws": AwsProvider}


def get_provider(config: AppConfig, override: str | None = None) -> Provider:
    name = (override or config.provider).lower()
    if name not in PROVIDERS:
        _raise(f"Unknown provider '{name}'. Choose from: {', '.join(PROVIDERS)}")
    return PROVIDERS[name](config)


# ---------------------------------------------------------------------------
# Post-boot orchestration (shared)
# ---------------------------------------------------------------------------


def wait_for_ssh(
    config: AppConfig, provider: Provider, instance_id: str, timeout_seconds: int = 900
) -> InstanceInfo:
    """Wait for SSH, bailing early on provisioning failures the provider can detect.

    Measured on Vast (docs/VAST_TIMEOUT_EXPERIMENT_RESULTS.md): healthy launches are
    SSH-ready in 1.5-10 min; every observed failure mode is detectable in under
    2 min via status_msg, or via persistent pubkey denial after "running".
    The provider's ProvisionMonitor owns that detection and raises
    ProvisioningDoomed; the timeout here is only the backstop for slow-but-alive
    provisions.
    """
    deadline = time.time() + timeout_seconds
    monitor = provider.provision_monitor()
    info: InstanceInfo | None = None
    while time.time() < deadline:
        info = provider.get(instance_id)
        monitor.check_status(info)
        if info.status in {"running"} and info.ssh:
            try:
                run_ssh(config, info.ssh, "true", timeout_seconds=20)
                return info
            except (IsaacCloudError, subprocess.TimeoutExpired) as exc:
                if "permission denied" in str(exc).lower():
                    monitor.auth_denied(info)
                else:
                    monitor.auth_reset()
        time.sleep(12)
    last_msg = str(info.raw.get("status_msg") or "").strip()[:200] if info else ""
    _raise(f"Instance {instance_id} did not become SSH-reachable within {timeout_seconds}s "
           f"(last status: {info.status if info else 'unknown'}"
           + (f", status_msg: {last_msg!r}" if last_msg else "") + ").")
    raise AssertionError  # unreachable


def wait_for_container(config: AppConfig, info: InstanceInfo, timeout_seconds: int = 600) -> None:
    """AWS only: wait for user-data to finish pulling and starting the container."""
    if not info.ssh or not info.ssh.container_via_docker:
        return
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        out = run_ssh(
            config,
            info.ssh,
            "sudo docker inspect -f '{{.State.Running}}' isaac-sim 2>/dev/null || echo absent",
            check=False,
        )
        if out.strip() == "true":
            return
        time.sleep(15)
    _raise("Isaac container did not start on the VM (check /var/log/isaac-cloud-bootstrap.log).")


def setup_isaac(config: AppConfig, info: InstanceInfo) -> None:
    """Start Isaac on the box: the GUI stack (foreground, verified) or the
    headless streaming kit (background)."""
    assert info.ssh
    if config.webrtc_enabled:
        validate_webrtc_config(config, info.provider)
        webrtc_connection(info)  # Fail before restarting Isaac if the mapping is absent.
        run_ssh_script(config, info.ssh, build_webrtc_check_script(), in_container=True, timeout_seconds=180)
    if not config.gui_enabled:
        output = run_ssh_script(
            config, info.ssh, build_isaac_container_launch_script(config),
            in_container=True, timeout_seconds=900,
        )
        typer.echo(output.splitlines()[-1] if output else "(no output)")
        return
    typer.echo(
        f"Bringing up the GUI stack via {GUI_STACK_PATH} (Xvfb -> Vulkan preflight -> "
        "noVNC -> supervised x11vnc -> GUI kit; waits for a mapped window)..."
    )
    try:
        output = run_ssh_script(
            config, info.ssh, build_gui_stack_install_script(config),
            in_container=True, timeout_seconds=GUI_STACK_TIMEOUT_S + 600,
        )
    except IsaacCloudError as exc:
        if "GUI_STACK_VULKAN_PRESENT_FAILED" in str(exc):
            raise IsaacCloudError(
                "This host cannot present Vulkan on an X display (vulkaninfo: "
                "vkGetPhysicalDeviceSurfacePresentModesKHR failed) -- a dud for GUI work; "
                f"headless would still run. Destroy it and relaunch on a driver >= "
                f"{GUI_MIN_DRIVER_MAJOR} host (`catalog --gui` ranks those first)."
            ) from exc
        raise IsaacCloudError(
            f"GUI stack failed to come up: {exc}\n"
            f"Re-run it on the box with `{GUI_STACK_PATH}` (idempotent) or `resume`; "
            f"kit log: {GUI_KIT_LOG}"
        ) from exc
    typer.echo(output)


def print_access(config: AppConfig, info: InstanceInfo) -> None:
    assert info.ssh
    if uses_webrtc(info):
        config = replace(config, gui_enabled=False)
    t = info.ssh
    typer.echo("")
    typer.echo(f"Instance: {info.provider}:{info.instance_id}  status={info.status}")
    key_flag = f" -i {config.ssh_private_key_path}" if config.ssh_private_key_path else ""
    typer.echo(f"SSH: ssh{key_flag} -p {t.port} {t.user}@{t.host}")
    forwards: list[tuple[int, int]] = []
    if config.agent_enabled:
        forwards.append((DEFAULT_AGENT_CONTROL_PORT, DEFAULT_AGENT_CONTROL_PORT))
    forwards.append((DEFAULT_RTSP_PORT, DEFAULT_RTSP_PORT))
    if config.gui_enabled:
        forwards.append((DEFAULT_NOVNC_PORT, DEFAULT_NOVNC_PORT))
    typer.echo(
        f"Tunnel ({'agent/RTSP only' if uses_webrtc(info) else 'recommended'}): "
        f"uv run python isaac_cloud.py tunnel "
        f"--instance-id {info.instance_id} --provider {info.provider}"
    )
    typer.echo(f"Tunnel (raw ssh):     {format_tunnel_command(config, t, forwards)}")
    if config.agent_enabled:
        typer.echo(
            f"  agent control  -> localhost:{DEFAULT_AGENT_CONTROL_PORT} "
            "(isaac-sim-remote skill / raw python over TCP)"
        )
    typer.echo(f"  rtsp cameras   -> rtsp://127.0.0.1:{DEFAULT_RTSP_PORT}/stream (once a writer is attached)")
    if config.gui_enabled:
        typer.echo(f"  gui (noVNC)    -> http://localhost:{DEFAULT_NOVNC_PORT}/vnc.html")
        typer.echo(
            f"  GUI stack: {GUI_STACK_PATH} on the box (re-run to repair; `check` for the "
            "probes); `status` reports the gui_* checks."
        )
    if uses_webrtc(info):
        typer.echo(
            f"WebRTC browser: uv run python isaac_cloud.py webrtc "
            f"--provider {info.provider} --instance-id {info.instance_id}"
        )
        typer.echo("WebRTC uses SSH signaling + direct, source-IP-restricted UDP media.")
        typer.echo("Use the WebRTC command instead of running a separate tunnel.")
    else:
        typer.echo("All ports are localhost-only on the remote side; SSH is the only ingress.")


# ---------------------------------------------------------------------------
# Persistence (append-only project snapshots in S3, streamed over SSH from the
# local machine; instances never hold cloud credentials).
#
# Layout: <s3_uri>projects/<project>/snapshots/<utc-timestamp>.tar.gz
# Push uploads a new snapshot (never deletes or overwrites saved work); pull
# restores the newest — or a named — snapshot with an atomic remote directory
# swap. An empty S3 prefix therefore means "no saved work yet", not "delete
# everything", which the previous mirror sync (`aws s3 sync --delete`) got
# fatally wrong in both directions.
# ---------------------------------------------------------------------------


def build_persistence_base_uri(config: AppConfig) -> str:
    if not config.persistence_s3_uri:
        _raise("Missing [persistence].s3_uri.")
    uri = config.persistence_s3_uri.strip()
    if not uri.startswith("s3://") or uri == "s3://":
        _raise("[persistence].s3_uri must look like s3://bucket/path/")
    return uri if uri.endswith("/") else uri + "/"


def validate_project_name(project: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", project or ""):
        _raise(
            f"Invalid project name {project!r}: use letters, digits, '.', '_' or '-' "
            f"(must start with a letter or digit)."
        )
    return project


def snapshot_prefix_uri(config: AppConfig, project: str) -> str:
    return f"{build_persistence_base_uri(config)}projects/{validate_project_name(project)}/snapshots/"


def parse_s3_uri(uri: str) -> tuple[str, str]:
    rest = uri.removeprefix("s3://")
    bucket, _, key = rest.partition("/")
    return bucket, key


def build_local_aws_env(config: AppConfig) -> dict[str, str]:
    env = os.environ.copy()
    if config.persistence_aws_region:
        env["AWS_REGION"] = config.persistence_aws_region
        env["AWS_DEFAULT_REGION"] = config.persistence_aws_region
    env.setdefault("AWS_PAGER", "")
    return env


def run_local_aws(config: AppConfig, args: list[str], *, timeout_seconds: int = 3600) -> str:
    return run_cli(
        ["aws", *args],
        timeout_seconds=timeout_seconds,
        error_prefix=f"aws {' '.join(args[:3])} failed",
        env=build_local_aws_env(config),
    )


def list_snapshots(config: AppConfig, project: str) -> list[dict[str, Any]]:
    """Snapshots for a project, oldest first (timestamped names sort correctly)."""
    bucket, key_prefix = parse_s3_uri(snapshot_prefix_uri(config, project))
    out = run_local_aws(
        config,
        ["s3api", "list-objects-v2", "--bucket", bucket, "--prefix", key_prefix,
         "--output", "json"],
        timeout_seconds=120,
    )
    contents = (json.loads(out).get("Contents") if out else None) or []
    snaps = [
        {
            "name": c["Key"].rsplit("/", 1)[-1],
            "key": c["Key"],
            "size": c.get("Size", 0),
            "uri": f"s3://{bucket}/{c['Key']}",
        }
        for c in contents
        if c["Key"].endswith(SNAPSHOT_SUFFIX)
    ]
    return sorted(snaps, key=lambda s: s["name"])


def list_projects(config: AppConfig) -> list[str]:
    bucket, key_prefix = parse_s3_uri(build_persistence_base_uri(config) + "projects/")
    out = run_local_aws(
        config,
        ["s3api", "list-objects-v2", "--bucket", bucket, "--prefix", key_prefix,
         "--delimiter", "/", "--output", "json"],
        timeout_seconds=120,
    )
    prefixes = (json.loads(out).get("CommonPrefixes") if out else None) or []
    return [p["Prefix"].removeprefix(key_prefix).rstrip("/") for p in prefixes]


def _require_reachable_ssh(info: InstanceInfo) -> SshTarget:
    if not info.ssh:
        _raise("Instance does not expose SSH, so persistence is unavailable.")
    if not check_tcp_connectivity(info.ssh.host, info.ssh.port):
        _raise("SSH is unreachable, so persistence is unavailable.")
    return info.ssh


def snapshot_push(
    config: AppConfig,
    provider: Provider,
    info: InstanceInfo,
    project: str,
    *,
    allow_empty: bool = False,
    timeout_seconds: int = 3600,
) -> str:
    """Save the remote project directory as a new snapshot. Never deletes saved work."""
    target = _require_reachable_ssh(info)
    remote_path = provider.persistence_remote_path()
    sudo = provider.remote_sudo()
    if not allow_empty:
        probe = run_ssh(
            config, target,
            f"{sudo}find {shell_quote(remote_path)} -mindepth 1 -print -quit 2>/dev/null",
            check=False,
        )
        if not probe.strip():
            return (
                f"Project directory {remote_path} is empty; skipping snapshot "
                f"(use --allow-empty to save an empty checkpoint)."
            )
    name = datetime.now(timezone.utc).strftime(SNAPSHOT_TIME_FMT) + SNAPSHOT_SUFFIX
    uri = snapshot_prefix_uri(config, project) + name
    remote_command = (
        f"{sudo}mkdir -p {shell_quote(remote_path)} && "
        f"{sudo}tar -C {shell_quote(remote_path)} -czf - ."
    )
    with tempfile.TemporaryDirectory(prefix="isaac-cloud-snap-") as staging_dir:
        staging_file = os.path.join(staging_dir, name)
        with open(staging_file, "wb") as handle:
            with subprocess.Popen(
                ssh_base_args(config, target) + [remote_command],
                stdout=handle,
                stderr=subprocess.PIPE,
            ) as ssh_proc:
                stderr_bytes = ssh_proc.stderr.read() if ssh_proc.stderr is not None else b""
                if ssh_proc.wait(timeout=timeout_seconds) != 0:
                    _raise(
                        "Snapshot archive creation failed: "
                        + (stderr_bytes.decode("utf-8", errors="replace").strip() or "(no stderr)")
                    )
        size_mb = os.path.getsize(staging_file) / 1e6
        run_local_aws(config, ["s3", "cp", staging_file, uri], timeout_seconds=timeout_seconds)
    pruned = prune_snapshots(config, project, keep_last=config.persistence_keep_last)
    message = f"Saved {uri} ({size_mb:.1f} MB)."
    if pruned:
        message += f" Pruned {pruned} old snapshot(s) (keep_last={config.persistence_keep_last})."
    return message


def snapshot_pull(
    config: AppConfig,
    provider: Provider,
    info: InstanceInfo,
    project: str,
    *,
    snapshot: str | None = None,
    timeout_seconds: int = 3600,
) -> str:
    """Restore a snapshot into the remote project directory, mount-safely.

    The archive fully extracts into a sibling directory before anything is
    deleted, so a dropped connection leaves the previous contents intact.
    The live directory's *children* are then replaced rather than the
    directory itself: on AWS it is a docker bind mount, and bind mounts track
    the inode — swapping the directory would leave the container looking at
    the orphaned old one (verified live; a `mv`-swap restore succeeded on the
    VM while the container still saw stale contents)."""
    target = _require_reachable_ssh(info)
    snaps = list_snapshots(config, project)
    if snapshot:
        wanted = snapshot if snapshot.endswith(SNAPSHOT_SUFFIX) else snapshot + SNAPSHOT_SUFFIX
        matches = [s for s in snaps if s["name"] == wanted]
        if not matches:
            available = ", ".join(s["name"] for s in snaps[-5:]) or "(none)"
            _raise(f"Snapshot {snapshot!r} not found for project '{project}'. Newest: {available}")
        chosen = matches[0]
    elif snaps:
        chosen = snaps[-1]
    else:
        return f"No snapshots for project '{project}'; starting fresh."
    remote_path = provider.persistence_remote_path()
    sudo = provider.remote_sudo()
    incoming = remote_path + ".incoming"
    script = (
        f"set -e; rm -rf {shell_quote(incoming)}; mkdir -p {shell_quote(incoming)}; "
        f"tar -C {shell_quote(incoming)} -xzf -; "
        f"mkdir -p {shell_quote(remote_path)}; "
        f"find {shell_quote(remote_path)} -mindepth 1 -maxdepth 1 -exec rm -rf -- {{}} +; "
        f"find {shell_quote(incoming)} -mindepth 1 -maxdepth 1 -exec mv -t {shell_quote(remote_path)} -- {{}} +; "
        f"rmdir {shell_quote(incoming)}; "
        f"chmod -R a+rwX {shell_quote(remote_path)} || true"
    )
    remote_command = f"{sudo}bash -c {shell_quote(script)}"
    with tempfile.TemporaryDirectory(prefix="isaac-cloud-snap-") as staging_dir:
        staging_file = os.path.join(staging_dir, chosen["name"])
        run_local_aws(config, ["s3", "cp", chosen["uri"], staging_file], timeout_seconds=timeout_seconds)
        with open(staging_file, "rb") as handle:
            completed = subprocess.run(
                ssh_base_args(config, target) + [remote_command],
                stdin=handle,
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout_seconds,
            )
    if completed.returncode != 0:
        output = ((completed.stdout or "") + "\n" + (completed.stderr or "")).strip()
        _raise(f"Snapshot restore failed (previous contents kept): {output[:800]}")
    return f"Restored {chosen['uri']} into {remote_path}."


def prune_snapshots(config: AppConfig, project: str, *, keep_last: int) -> int:
    if keep_last <= 0:
        return 0
    snaps = list_snapshots(config, project)
    excess = snaps[:-keep_last] if len(snaps) > keep_last else []
    for snap in excess:
        run_local_aws(config, ["s3", "rm", snap["uri"]], timeout_seconds=120)
    return len(excess)


def resolve_project(config: AppConfig, info: InstanceInfo | None = None, override: str | None = None) -> str:
    """Explicit --project beats the label the instance was launched with, beats config."""
    if override:
        return validate_project_name(override)
    if info is not None:
        labeled = getattr(info, "label", "") or ""
        if labeled.startswith(PROJECT_LABEL_PREFIX):
            return validate_project_name(labeled.removeprefix(PROJECT_LABEL_PREFIX))
        tags = info.raw.get("Tags") if isinstance(info.raw, dict) else None
        if isinstance(tags, list):
            for tag in tags:
                if tag.get("Key") == AWS_TAG_PROJECT and tag.get("Value"):
                    return validate_project_name(tag["Value"])
    return validate_project_name(config.persistence_project)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

app = typer.Typer(
    name=APP_NAME,
    help="Launch and manage NVIDIA Isaac Sim on Vast.ai or AWS EC2, with SSH access and optional WebRTC.",
    no_args_is_help=True,
)
sync_app = typer.Typer(help="Save/restore project snapshots (append-only) between the instance and S3.")
app.add_typer(sync_app, name="sync")

PROVIDER_OPTION = typer.Option(None, "--provider", "-p", help="vast or aws (default from config).")
INSTANCE_ID_OPTION = typer.Option(..., "--instance-id")
PROJECT_OPTION = typer.Option(
    None,
    "--project",
    help="Project namespace (default: the project the instance was launched with, else config).",
)
AGENT_PORT_OPTION = typer.Option(
    DEFAULT_AGENT_CONTROL_PORT,
    "--agent-port",
    help="Local port the agent control socket is tunnelled to (remote side is always 8226).",
)


def _config() -> AppConfig:
    return load_app_config()


def cli_errors(func):
    """Print IsaacCloudError as `Error: ...` and exit 1 (shared by every command).

    Commands needing extra failure-path output (launch's leftover-instance
    hint, stop/destroy's not-stopped warnings) print it before re-raising."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except IsaacCloudError as exc:
            typer.echo(f"Error: {exc}")
            raise typer.Exit(1)

    return wrapper


@app.command()
@cli_errors
def catalog(
    provider: str = PROVIDER_OPTION,
    gui: bool = typer.Option(
        None,
        "--gui/--no-gui",
        help=f"Rank driver >= {GUI_MIN_DRIVER_MAJOR} hosts first (GUI presentation needs them).",
    ),
) -> None:
    """List available GPU offers/instance options for the provider."""
    config = _config()
    if gui is not None:
        config = replace(config, gui_enabled=gui)
    prov = get_provider(config, provider)
    if prov.name == "vast":
        offers = prov.catalog()
        if not offers:
            typer.echo("No matching offers.")
            raise typer.Exit(1)
        if config.gui_enabled:
            typer.echo(f"(--gui: driver >= {GUI_MIN_DRIVER_MAJOR} hosts ranked first)")
        for o in offers:
            typer.echo(
                f"offer={o['id']} machine={o.get('machine_id')} {o.get('gpu_name')} "
                f"driver={o.get('driver_version')} ${o.get('dph_total', 0):.3f}/hr "
                f"rel={o.get('reliability2', 0):.3f} inet={o.get('inet_down', 0):.0f}Mbps "
                f"{o.get('geolocation')}"
            )
    else:
        ami = prov.resolve_ami()
        typer.echo(
            f"aws {config.aws_region}: instance_type={config.aws_instance_type} ami={ami}"
        )


@app.command()
@cli_errors
def instances(provider: str = PROVIDER_OPTION) -> None:
    """List managed instances."""
    config = _config()
    prov = get_provider(config, provider)
    rows = prov.list_instances()
    if not rows:
        typer.echo("No instances found.")
        return
    for info in rows:
        ssh = f"{info.ssh.user}@{info.ssh.host}:{info.ssh.port}" if info.ssh else "-"
        typer.echo(f"{info.provider}:{info.instance_id} {info.status:10s} {ssh} {info.label}")


@app.command()
def launch(
    provider: str = PROVIDER_OPTION,
    offer_id: str = typer.Option(
        None,
        "--offer-id",
        help="Rent a specific Vast offer (from `catalog` output) instead of the top-ranked one.",
    ),
    gui: bool = typer.Option(None, "--gui/--no-gui", help="Also start the noVNC GUI stack."),
    webrtc: bool = typer.Option(None, "--webrtc/--no-webrtc", help="Enable native WebRTC with direct UDP media on Vast or AWS."),
    agent: bool = typer.Option(None, "--agent/--no-agent", help="Enable the agent control socket."),
    curobo: bool = typer.Option(
        None,
        "--curobo/--no-curobo",
        help="Install cuRobo into Isaac's python in the background after launch.",
    ),
    lab: bool = typer.Option(
        None,
        "--lab/--no-lab",
        help="Install Isaac Lab into Isaac's python in the background after launch "
        "(run Lab workloads manually over SSH; they need the streaming Isaac stopped).",
    ),
    project: str = PROJECT_OPTION,
    timeout_seconds: int = typer.Option(1200, help="How long to wait for SSH readiness."),
) -> None:
    """Rent/launch an instance, start Isaac, and print SSH tunnel commands."""
    config = _config()
    if gui is not None:
        config = replace(config, gui_enabled=gui)
    if webrtc is not None:
        config = replace(config, webrtc_enabled=webrtc)
        if webrtc and gui is None:
            config = replace(config, gui_enabled=False)
    if agent is not None:
        config = replace(config, agent_enabled=agent)
    if curobo is not None:
        config = replace(config, curobo_enabled=curobo)
    if lab is not None:
        config = replace(config, lab_enabled=lab)
    prov = get_provider(config, provider)
    live_instance_id: str | None = None
    try:
        validate_webrtc_config(config, prov.name, selecting_offer=True)
        launch_project = resolve_project(config, override=project)
        info = prov.launch(offer_id=offer_id)
        live_instance_id = info.instance_id
        if config.persistence_enabled:
            prov.set_project_label(info.instance_id, launch_project)
        typer.echo(f"Created {info.provider}:{info.instance_id}; waiting for SSH...")
        try:
            info = wait_for_ssh(config, prov, info.instance_id, timeout_seconds=timeout_seconds)
        except ProvisioningDoomed as exc:
            # The instance can never become usable and only accrues cost.
            typer.echo(f"Destroying doomed instance {info.instance_id}...")
            prov.destroy(info.instance_id)
            live_instance_id = None
            raise IsaacCloudError(f"{exc} (instance destroyed)")
        typer.echo("SSH reachable.")
        wait_for_container(config, info)
        if config.persistence_enabled:
            typer.echo(f"Rehydrating project '{launch_project}' from S3...")
            typer.echo(snapshot_pull(config, prov, info, launch_project))
        typer.echo("Starting Isaac (first boot compiles shaders; allow several minutes)...")
        setup_isaac(config, info)
        if config.curobo_enabled:
            provision_curobo(config, info)
        if config.lab_enabled:
            provision_lab(config, info)
        print_access(config, info)
    except IsaacCloudError as exc:
        typer.echo(f"Error: {exc}")
        if live_instance_id:
            typer.echo(
                f"Instance {live_instance_id} is still running (and billing). Inspect it, or "
                f"destroy with: uv run python isaac_cloud.py destroy "
                f"--provider {provider} --instance-id {live_instance_id}"
            )
        raise typer.Exit(1)


@app.command()
@cli_errors
def status(
    instance_id: str = INSTANCE_ID_OPTION,
    provider: str = PROVIDER_OPTION,
    agent_port: int = AGENT_PORT_OPTION,
) -> None:
    """Show instance state plus in-container readiness probes.

    On a box brought up with --gui the probe also runs the GUI stack's checks
    (gui_x, gui_vulkan, gui_vnc, gui_novnc, gui_kit, gui_agent, gui_screen)."""
    config = _config()
    prov = get_provider(config, provider)
    info = prov.get(instance_id)
    ssh = f"{info.ssh.user}@{info.ssh.host}:{info.ssh.port}" if info.ssh else "-"
    typer.echo(f"{info.provider}:{info.instance_id} {info.status} {ssh}")
    if info.ssh and check_tcp_connectivity(info.ssh.host, info.ssh.port):
        probe = run_ssh_script(
            config, info.ssh, build_container_probe_script(config),
            in_container=True, timeout_seconds=180,
        )
        typer.echo(probe)
        if "gui_stack:" in probe:
            config = replace(config, gui_enabled=True)
        typer.echo(probe_local_tunnel(local_port=agent_port))
        print_access(config, info)
    else:
        typer.echo("SSH not reachable yet.")


@app.command()
@cli_errors
def stop(
    instance_id: str = INSTANCE_ID_OPTION,
    provider: str = PROVIDER_OPTION,
    project: str = PROJECT_OPTION,
    skip_push: bool = typer.Option(False, "--skip-push", help="Skip the snapshot save before stopping."),
) -> None:
    """Stop an instance (saves a project snapshot to S3 first if enabled)."""
    config = _config()
    prov = get_provider(config, provider)
    if config.persistence_enabled and not skip_push:
        info = prov.get(instance_id)
        target_project = resolve_project(config, info, project)
        typer.echo(f"Saving snapshot of project '{target_project}' to S3...")
        try:
            typer.echo(snapshot_push(config, prov, info, target_project))
        except IsaacCloudError as exc:
            typer.echo(f"Error: snapshot save failed: {exc}")
            typer.echo(
                f"Instance {instance_id} was NOT stopped (it is still running and billing). "
                f"Retry, or stop without saving: ... stop --instance-id {instance_id} --skip-push"
            )
            raise typer.Exit(1)
    prov.stop(instance_id)
    typer.echo(f"Stopped {instance_id}.")


@app.command()
@cli_errors
def resume(
    instance_id: str = INSTANCE_ID_OPTION,
    provider: str = PROVIDER_OPTION,
    gui: bool = typer.Option(
        None,
        "--gui/--no-gui",
        help="Bring up the noVNC GUI stack (default: whatever the box was launched with).",
    ),
    agent: bool = typer.Option(None, "--agent/--no-agent", help="Enable the agent control socket."),
) -> None:
    """Start a stopped instance and relaunch Isaac (the GUI stack if the box had one)."""
    config = _config()
    if agent is not None:
        config = replace(config, agent_enabled=agent)
    prov = get_provider(config, provider)
    info = prov.get(instance_id)
    if config.webrtc_enabled or uses_webrtc(info):
        config = replace(config, webrtc_enabled=True, gui_enabled=bool(gui))
        validate_webrtc_config(config, prov.name)
    if info.status != "running":
        prov.start(instance_id)
    info = wait_for_ssh(config, prov, instance_id)
    # Provider metadata preserves WebRTC mode across stop/start, even when
    # --webrtc was only supplied at launch.
    if uses_webrtc(info):
        config = replace(config, webrtc_enabled=True, gui_enabled=False)
    wait_for_container(config, info)
    if gui is None:
        if config.webrtc_enabled:
            gui = False
            typer.echo("Resuming with WebRTC headless streaming.")
        else:
            gui = remote_gui_stack_installed(config, info)
            typer.echo(
                f"GUI stack {'found' if gui else 'not found'} on the box "
                f"({GUI_STACK_PATH}); resuming {'with the GUI' if gui else 'headless'}."
            )
    config = replace(config, gui_enabled=gui)
    setup_isaac(config, info)
    print_access(config, info)


@app.command()
@cli_errors
def destroy(
    instance_id: str = typer.Option(None, "--instance-id"),
    provider: str = PROVIDER_OPTION,
    all_instances: bool = typer.Option(False, "--all"),
    yes: bool = typer.Option(False, "--yes", "-y"),
    project: str = PROJECT_OPTION,
    skip_push: bool = typer.Option(False, "--skip-push", help="Destroy without saving a snapshot first."),
) -> None:
    """Destroy instance(s), saving a project snapshot to S3 first if enabled.

    If the snapshot cannot be taken (SSH unreachable, save fails), the instance
    is NOT destroyed — destroying would silently discard the work. Use
    --skip-push to destroy anyway."""
    config = _config()
    prov = get_provider(config, provider)
    targets = (
        [i.instance_id for i in prov.list_instances()] if all_instances else [instance_id]
    )
    targets = [t for t in targets if t]
    if not targets:
        typer.echo("Nothing to destroy.")
        return
    if not yes:
        typer.confirm(f"Destroy {len(targets)} instance(s) on {prov.name}?", abort=True)
    kept: list[str] = []
    for tid in targets:
        if config.persistence_enabled and not skip_push:
            try:
                info = prov.get(tid)
                target_project = resolve_project(config, info, project)
                typer.echo(f"Saving snapshot of project '{target_project}' from {tid}...")
                typer.echo(snapshot_push(config, prov, info, target_project))
            except IsaacCloudError as exc:
                typer.echo(f"Not destroying {tid}: snapshot save failed: {exc}")
                kept.append(tid)
                continue
        prov.destroy(tid)
        typer.echo(f"Destroyed {tid}.")
    if kept:
        typer.echo(
            f"{len(kept)} instance(s) kept (still running and billing): {', '.join(kept)}. "
            f"Retry, or destroy without saving: ... destroy --skip-push"
        )
        raise typer.Exit(1)


@app.command()
@cli_errors
def tunnel(
    instance_id: str = INSTANCE_ID_OPTION,
    provider: str = PROVIDER_OPTION,
    agent_port: int = AGENT_PORT_OPTION,
    rtsp_port: int = typer.Option(
        DEFAULT_RTSP_PORT, "--rtsp-port", help="Local port for the RTSP cameras."
    ),
    novnc_port: int = typer.Option(
        DEFAULT_NOVNC_PORT, "--novnc-port", help="Local port for the noVNC GUI."
    ),
) -> None:
    """Run a supervised SSH tunnel to the instance (auto-reconnects; Ctrl-C to stop).

    Forwards agent control (8226), RTSP (8554), and noVNC (6080) to localhost;
    the local ports are configurable so tunnels to two boxes can coexist.
    SSH keepalives detect dead/zombie connections within ~30s and the tunnel
    re-establishes itself with backoff, so client sessions (noVNC, agent
    scripts, RTSP players) see a brief blip instead of needing manual repair.
    """
    config = _config()
    prov = get_provider(config, provider)
    local_ports = {
        DEFAULT_AGENT_CONTROL_PORT: agent_port,
        DEFAULT_RTSP_PORT: rtsp_port,
        DEFAULT_NOVNC_PORT: novnc_port,
    }
    run_supervised_tunnel(config, prov, instance_id, local_ports)


@app.command()
@cli_errors
def webrtc(
    instance_id: str = INSTANCE_ID_OPTION,
    provider: str = PROVIDER_OPTION,
    viewer_port: int = typer.Option(DEFAULT_WEBRTC_VIEWER_PORT, min=1024, max=65535),
    client_ip: str = typer.Option(None, help="Public IPv4 allowed for UDP (default: address seen by SSH)."),
) -> None:
    """Serve the local WebRTC browser viewer and tunnel signaling; Ctrl-C to stop.

    Requires an instance launched with --webrtc. Video travels directly over
    UDP, not through SSH. Close any existing tunnel before running this command.
    """
    config = _config()
    prov = get_provider(config, provider)
    ports = [(p, label) for p, label in SERVICE_PORTS if p != DEFAULT_NOVNC_PORT]
    ports.append((DEFAULT_ISAAC_SIGNAL_PORT, "WebRTC signal"))
    if viewer_port in {p for p, _ in ports}:
        _raise("--viewer-port conflicts with an SSH forwarded service port.")
    if client_ip is not None:
        build_webrtc_relay_script(client_ip)  # Validate before any remote mutations.
    for port, _ in ports:
        if check_tcp_connectivity("127.0.0.1", port, timeout_seconds=0.2):
            _raise(f"Local port {port} is already in use. Stop the existing tunnel/viewer before connecting.")
    info = prov.get(instance_id)
    current_connection = webrtc_connection(info)
    if info.status != "running" or not info.ssh:
        _raise("Instance must be running and SSH reachable; resume it first.")
    relay: tuple[SshTarget, int] | None = None
    close_access: Callable[[], None] | None = None

    def prepare(current: InstanceInfo) -> None:
        nonlocal current_connection, relay, close_access
        current_connection = webrtc_connection(current)
        assert current.ssh
        address = client_ip or run_ssh(config, current.ssh, 'printf "%s" "${SSH_CONNECTION%% *}"')
        build_webrtc_relay_script(address)  # Validate before changing provider ingress.
        if close_access is not None:
            close_access()
            close_access = None
        close_access = prov.open_webrtc_access(current, address)
        pid = start_webrtc_relay(config, current, address)
        relay = (current.ssh, pid)

    server = make_webrtc_http_server(viewer_port, lambda: current_connection)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    typer.echo(f"Open http://127.0.0.1:{viewer_port} in Chrome or Edge after signaling is ready.")
    typer.echo("First boot may take 5–10 minutes. Click Connect once Isaac has loaded.")
    typer.echo("After an instance restart/address change, reload the browser page.")
    try:
        run_supervised_tunnel(
            config, prov, instance_id,
            service_ports=ports,
            on_connect=prepare,
        )
    except KeyboardInterrupt:
        typer.echo("WebRTC viewer stopped.")
    except (OSError, subprocess.TimeoutExpired) as exc:
        _raise(f"WebRTC connection failed: {exc}")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        if relay is not None:
            try:
                stop_webrtc_relay(config, *relay)
            except (IsaacCloudError, OSError, subprocess.TimeoutExpired) as exc:
                typer.echo(f"Could not stop the remote UDP relay: {exc}. It remains source-IP-restricted.")
        if close_access is not None:
            try:
                close_access()
            except (IsaacCloudError, OSError, subprocess.TimeoutExpired) as exc:
                typer.echo(f"Could not remove WebRTC ingress: {exc}. Remove the reported UDP rule manually.")


@sync_app.command("pull")
@cli_errors
def sync_pull(
    instance_id: str = INSTANCE_ID_OPTION,
    provider: str = PROVIDER_OPTION,
    project: str = PROJECT_OPTION,
    snapshot: str = typer.Option(
        None, "--snapshot", help="Snapshot name/timestamp to restore (default: newest)."
    ),
) -> None:
    """Restore a project snapshot from S3 into the instance."""
    config = _config()
    prov = get_provider(config, provider)
    info = prov.get(instance_id)
    target_project = resolve_project(config, info, project)
    typer.echo(snapshot_pull(config, prov, info, target_project, snapshot=snapshot))


@sync_app.command("push")
@cli_errors
def sync_push(
    instance_id: str = INSTANCE_ID_OPTION,
    provider: str = PROVIDER_OPTION,
    project: str = PROJECT_OPTION,
    allow_empty: bool = typer.Option(
        False, "--allow-empty", help="Save a snapshot even if the project directory is empty."
    ),
) -> None:
    """Save the instance's project directory as a new snapshot in S3."""
    config = _config()
    prov = get_provider(config, provider)
    info = prov.get(instance_id)
    target_project = resolve_project(config, info, project)
    typer.echo(snapshot_push(config, prov, info, target_project, allow_empty=allow_empty))


@sync_app.command("list")
@cli_errors
def sync_list(
    project: str = typer.Option(None, "--project", help="List snapshots of this project only."),
) -> None:
    """List saved projects and their snapshots."""
    config = _config()
    projects = [validate_project_name(project)] if project else list_projects(config)
    if not projects:
        typer.echo("No saved projects.")
        return
    for name in projects:
        snaps = list_snapshots(config, name)
        typer.echo(f"{name}: {len(snaps)} snapshot(s)")
        for snap in snaps:
            typer.echo(f"  {snap['name']:32s} {snap['size'] / 1e6:8.1f} MB")


if __name__ == "__main__":
    app()
