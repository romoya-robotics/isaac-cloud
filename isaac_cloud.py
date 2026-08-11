"""isaac-cloud: launch and manage NVIDIA Isaac Sim on cloud GPUs.

Providers:
  - vast: Vast.ai marketplace containers (the Isaac image IS the instance).
  - aws:  EC2 GPU VMs (g6e/L40S) running the Isaac container under Docker.

Access model is SSH-only on every provider: the agent control socket (8226),
RTSP camera streams (8554), and the noVNC GUI (6080) are all bound to
localhost on the remote side and reached through SSH tunnels printed by the
CLI. Nothing Isaac-related is ever exposed to the public internet.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import socket
import subprocess
import tempfile
import textwrap
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
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

DEFAULT_INSTANCE_NAME_PREFIX = "isaac-cloud"
DEFAULT_DISK_GB = 100

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
# (see VAST_EXPERIMENT_RESULTS.md), which whole-machine offers guarantee.
DEFAULT_VAST_QUERY = (
    'gpu_name in ["RTX_4090","L40S"] driver_version >= 580.95.05 '
    "verified=true rentable=true num_gpus=1 disk_space >= 80 inet_down >= 300"
)
DEFAULT_VAST_WHOLE_MACHINE = True
DEFAULT_VAST_MIN_RELIABILITY = 0.99

# Vast surfaces docker's own error text in `status_msg` while the instance sits
# in "loading" forever (measured: bad registry login visible at +30s, missing
# image tag at +50s; neither ever self-heals — see VAST_TIMEOUT_EXPERIMENT_RESULTS.md).
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
        isaac_version=get("ISAAC_CLOUD_ISAAC_VERSION", "defaults", "isaac_version")
        or DEFAULT_ISAAC_VERSION,
        instance_name_prefix=get(
            "ISAAC_CLOUD_INSTANCE_NAME_PREFIX", "defaults", "instance_name_prefix"
        )
        or DEFAULT_INSTANCE_NAME_PREFIX,
        disk_gb=int(get("ISAAC_CLOUD_DISK_GB", "defaults", "disk_gb") or DEFAULT_DISK_GB),
        ngc_api_key=get("NGC_API_KEY", "ngc", "api_key"),
        ssh_private_key_path=get("ISAAC_CLOUD_SSH_PRIVATE_KEY", "ssh", "private_key_path"),
        ssh_public_key_path=get("ISAAC_CLOUD_SSH_PUBLIC_KEY", "ssh", "public_key_path"),
        agent_enabled=_bool(nested_get(data, "agent", "enabled"), True),
        gui_enabled=_bool(nested_get(data, "gui", "enabled"), False),
        gui_resolution=nested_get(data, "gui", "resolution") or DEFAULT_GUI_RESOLUTION,
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
    return f"sudo docker exec isaac-sim bash -c {shell_quote(command)}"


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
        _raise(f"Remote script failed ({completed.returncode}): {output[:1200]}")
    return (completed.stdout or "").strip()


SERVICE_PORTS: list[tuple[int, str]] = [
    (DEFAULT_AGENT_CONTROL_PORT, "agent control"),
    (DEFAULT_RTSP_PORT, "rtsp cameras"),
    (DEFAULT_NOVNC_PORT, "gui (noVNC)"),
]


def probe_local_tunnel(timeout_seconds: float = 5.0) -> str:
    """End-to-end health of a local tunnel, zombie-aware.

    A dead ssh forward can still accept() locally, so port-open checks lie.
    The agent socket gives a true round trip: send a no-op, expect JSON back.
    """
    try:
        sock = socket.create_connection(("127.0.0.1", DEFAULT_AGENT_CONTROL_PORT), timeout=2.0)
    except OSError:
        return "tunnel: not running locally (start one with: isaac_cloud.py tunnel)"
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


def run_supervised_tunnel(config: AppConfig, provider, instance_id: str) -> None:
    """Foreground self-healing tunnel: keepalives kill zombies within ~30s,
    then we reconnect with backoff, re-resolving the instance address in case
    it changed across a stop/start. Ctrl-C exits."""
    forwards = [(port, port) for port, _ in SERVICE_PORTS]
    drops = 0
    backoff = 3
    while True:
        info = provider.get(instance_id)
        if info.status != "running" or not info.ssh:
            typer.echo(f"Instance is {info.status}; waiting 15s for it to be reachable...")
            time.sleep(15)
            continue
        target = info.ssh
        if drops == 0:
            typer.echo(f"Tunnel to {info.provider}:{instance_id} ({target.host}:{target.port}):")
            for port, label in SERVICE_PORTS:
                suffix = "/vnc.html (browser)" if port == DEFAULT_NOVNC_PORT else ""
                typer.echo(f"  {label:14s} -> localhost:{port}{suffix}")
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
            args[1:1] = ["-L", f"{local_port}:127.0.0.1:{remote_port}"]
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


# ---------------------------------------------------------------------------
# Container-side setup scripts (validated in VAST_EXPERIMENT_RESULTS.md and
# GUI_TUNNEL_EXPERIMENT_PLAN.md; identical inside the Isaac image on every
# provider).
# ---------------------------------------------------------------------------


def build_isaac_container_launch_script(config: AppConfig) -> str:
    """Launch headless streaming Isaac inside the container.

    Handles the two container quirks found on Vast hosts: env vars must be
    exported in-shell, and a minority of hosts inject compute-only NVIDIA
    libraries (no Vulkan/GLX/NVENC), which we fix by side-loading the exact
    driver-matched userland libs from the Ubuntu archive.
    """
    agent_flag = " --enable isaacsim.code_editor.python_server" if config.agent_enabled else ""
    return dedent_script(
        f"""\
        #!/bin/bash
        set -x
        export ACCEPT_EULA=Y PRIVACY_CONSENT=Y OMNI_KIT_ALLOW_ROOT=1
        export ISAACSIM_HOST=127.0.0.1
        export ISAACSIM_SIGNAL_PORT={DEFAULT_ISAAC_SIGNAL_PORT}
        export ISAACSIM_STREAM_PORT={DEFAULT_ISAAC_STREAM_PORT}

        # Side-load Vulkan/GLX/NVENC userland libs when the host runtime only
        # injected compute libraries. Version must match the host kernel driver.
        if ! ls /usr/lib/x86_64-linux-gnu/libGLX_nvidia.so.0 >/dev/null 2>&1; then
            DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | tr -d " ")
            MAJOR=${{DRIVER%%.*}}
            apt-get update -qq >/dev/null 2>&1
            cd /tmp && apt-get download -qq \\
                libnvidia-gl-$MAJOR libnvidia-encode-$MAJOR libnvidia-decode-$MAJOR 2>/dev/null
            mkdir -p /opt/nvgl
            for deb in libnvidia-*-$MAJOR*.deb; do dpkg -x "$deb" /opt/nvgl; done
            LIBDIR=/opt/nvgl/usr/lib/x86_64-linux-gnu
            printf '{{"file_format_version":"1.0.0","ICD":{{"library_path":"%s/libGLX_nvidia.so.0","api_version":"1.3.194"}}}}' "$LIBDIR" > /opt/nvgl/icd.json
            export VK_DRIVER_FILES=/opt/nvgl/icd.json VK_ICD_FILENAMES=/opt/nvgl/icd.json
            export LD_LIBRARY_PATH=$LIBDIR
            echo "$LIBDIR" > /etc/ld.so.conf.d/zz-nvgl.conf && ldconfig
        fi

        pkill -f "[k]it/kit" 2>/dev/null; sleep 2
        nohup /isaac-sim/runheadless.sh -v{agent_flag} > /root/isaac.log 2>&1 &
        echo LAUNCHED
        """
    )


def build_gui_setup_script(config: AppConfig) -> str:
    """noVNC GUI stack: Isaac GUI -> Xvfb -> x11vnc -> websockify, one TCP port.

    Community-validated approach for TCP-only access (works regardless of the
    NVENC lottery on fractional hosts).
    """
    agent_flag = " --enable isaacsim.code_editor.python_server" if config.agent_enabled else ""
    return dedent_script(
        f"""\
        #!/bin/bash
        set -x
        export DEBIAN_FRONTEND=noninteractive
        apt-get update -qq >/dev/null 2>&1
        apt-get install -y -qq xvfb x11vnc novnc websockify >/dev/null 2>&1

        pkill -f "Xvf[b] :1" 2>/dev/null; pkill -f "x11vn[c]" 2>/dev/null
        pkill -f "[k]it/kit" 2>/dev/null; sleep 2
        setsid Xvfb :1 -screen 0 {config.gui_resolution}x24 </dev/null >/var/log/xvfb.log 2>&1 &
        sleep 2
        setsid x11vnc -display :1 -localhost -forever -shared -nopw -rfbport 5901 </dev/null >/var/log/x11vnc.log 2>&1 &
        sleep 1
        pgrep -f "websockif[y]" >/dev/null || setsid websockify --web /usr/share/novnc 127.0.0.1:{DEFAULT_NOVNC_PORT} localhost:5901 </dev/null >/var/log/websockify.log 2>&1 &
        sleep 1

        export ACCEPT_EULA=Y PRIVACY_CONSENT=Y OMNI_KIT_ALLOW_ROOT=1 DISPLAY=:1
        nohup /isaac-sim/isaac-sim.sh --allow-root{agent_flag} > /root/isaac_gui.log 2>&1 &
        echo GUI_LAUNCHED
        for p in 5901 {DEFAULT_NOVNC_PORT}; do
            (echo > /dev/tcp/127.0.0.1/$p) 2>/dev/null && echo "port $p: open" || echo "port $p: closed"
        done
        """
    )


def build_container_probe_script() -> str:
    """Report readiness of everything we care about inside the container."""
    return dedent_script(
        f"""\
        #!/bin/bash
        echo "gpu: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -1)"
        echo "gpu_minor: $(nvidia-smi -q 2>/dev/null | grep -i 'Minor Number' | awk '{{print $NF}}' | head -1)"
        echo "kit_procs: $(pgrep -c '[k]it' 2>/dev/null || echo 0)"
        for log in /root/isaac.log /root/isaac_gui.log; do
            if [ -f "$log" ]; then
                grep -qm1 -E "Streaming App is loaded|app ready" "$log" && echo "$(basename $log): ready" || echo "$(basename $log): loading"
            fi
        done
        for p in {DEFAULT_AGENT_CONTROL_PORT} {DEFAULT_ISAAC_SIGNAL_PORT} {DEFAULT_RTSP_PORT} {DEFAULT_NOVNC_PORT}; do
            (echo > /dev/tcp/127.0.0.1/$p) 2>/dev/null && echo "port $p: open" || echo "port $p: closed"
        done
        """
    )


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
    completed = subprocess.run(
        [_vastai_bin(), *args, "--raw"],
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout_seconds,
    )
    if completed.returncode != 0:
        _raise(f"vastai {' '.join(args[:2])} failed: {(completed.stderr or completed.stdout)[:500]}")
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError:
        _raise(f"vastai returned non-JSON output: {completed.stdout[:300]}")


class VastProvider:
    name = "vast"

    def __init__(self, config: AppConfig) -> None:
        self.config = config

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
        pub_path = self.config.ssh_public_key_path
        if not pub_path:
            return
        try:
            pub = Path(pub_path).expanduser().read_text().strip()
            subprocess.run(
                [_vastai_bin(), "create", "ssh-key", pub],
                capture_output=True,
                text=True,
                check=False,
                timeout=30,
            )
        except OSError:
            pass

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

    def provisioning_error(self, info: InstanceInfo) -> str | None:
        """Return docker's error text when the provision is doomed (else None)."""
        msg = str(info.raw.get("status_msg") or "").strip()
        low = msg.lower()
        if any(pat in low for pat in VAST_FATAL_STATUS_PATTERNS):
            return msg
        return None

    def transient_daemon_error(self, info: InstanceInfo) -> str | None:
        """Generic docker daemon error: fatal only if it persists across polls."""
        msg = str(info.raw.get("status_msg") or "").strip()
        if VAST_DAEMON_ERROR_MARKER in msg.lower():
            return msg
        return None

    def set_project_label(self, instance_id: str, project: str) -> None:
        # Best-effort: records which project this instance was launched for so
        # stop/destroy save back to the same project without --project.
        subprocess.run(
            [_vastai_bin(), "label", "instance", instance_id, f"{PROJECT_LABEL_PREFIX}{project}"],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )

    def attach_ssh_key(self, instance_id: str) -> None:
        """Recovery for hosts that miss the account-key injection at create."""
        pub_path = self.config.ssh_public_key_path
        if not pub_path:
            return
        try:
            pub = Path(pub_path).expanduser().read_text().strip()
            subprocess.run(
                [_vastai_bin(), "attach", "ssh", instance_id, pub],
                capture_output=True,
                text=True,
                check=False,
                timeout=45,
            )
        except OSError:
            pass

    def stop(self, instance_id: str) -> None:
        run_vastai_json(["stop", "instance", instance_id])

    def start(self, instance_id: str) -> None:
        run_vastai_json(["start", "instance", instance_id])

    def destroy(self, instance_id: str) -> None:
        subprocess.run(
            [_vastai_bin(), "destroy", "instance", instance_id, "-y"],
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )

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
    completed = subprocess.run(
        ["aws", *args, "--region", config.aws_region, "--output", "json"],
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout_seconds,
    )
    if completed.returncode != 0:
        _raise(f"aws {' '.join(args[:3])} failed: {(completed.stderr or completed.stdout)[:600]}")
    return json.loads(completed.stdout) if completed.stdout.strip() else {}


class AwsProvider:
    name = "aws"

    def __init__(self, config: AppConfig) -> None:
        self.config = config

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
                    subprocess.run(  # base64 without newlines
                        ["base64", "-w", "0"], input=pub, capture_output=True, text=True
                    ).stdout,
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
        subprocess.run(
            ["aws", "ec2", "create-tags", "--resources", instance_id,
             "--tags", f"Key={AWS_TAG_PROJECT},Value={project}",
             "--region", self.config.aws_region],
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
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


def get_provider(config: AppConfig, override: str | None = None):
    name = (override or config.provider).lower()
    if name not in PROVIDERS:
        _raise(f"Unknown provider '{name}'. Choose from: {', '.join(PROVIDERS)}")
    return PROVIDERS[name](config)


# ---------------------------------------------------------------------------
# Post-boot orchestration (shared)
# ---------------------------------------------------------------------------


def wait_for_ssh(config: AppConfig, provider, instance_id: str, timeout_seconds: int = 900) -> InstanceInfo:
    """Wait for SSH, bailing early on provisioning failures the provider can detect.

    Measured on Vast (VAST_TIMEOUT_EXPERIMENT_RESULTS.md): healthy launches are
    SSH-ready in 1.5-10 min; every observed failure mode is detectable in under
    2 min via status_msg, or via persistent pubkey denial after "running".
    """
    deadline = time.time() + timeout_seconds
    info: InstanceInfo | None = None
    daemon_error_since: float | None = None
    auth_denied_since: float | None = None
    key_reattached = False
    while time.time() < deadline:
        info = provider.get(instance_id)
        fatal = getattr(provider, "provisioning_error", lambda _info: None)(info)
        if fatal:
            raise ProvisioningDoomed(
                f"Provisioning of {instance_id} failed (image pull cannot succeed): "
                f"{fatal[:300]}. Retry with another offer (or fix credentials/image)."
            )
        transient = getattr(provider, "transient_daemon_error", lambda _info: None)(info)
        if transient:
            daemon_error_since = daemon_error_since or time.time()
            if time.time() - daemon_error_since >= VAST_DAEMON_ERROR_FATAL_S:
                raise ProvisioningDoomed(
                    f"Provisioning of {instance_id} stuck on a docker daemon error "
                    f"for {VAST_DAEMON_ERROR_FATAL_S}s: {transient[:300]}"
                )
        else:
            daemon_error_since = None
        if info.status in {"running"} and info.ssh:
            try:
                run_ssh(config, info.ssh, "true", timeout_seconds=20)
                return info
            except (IsaacCloudError, subprocess.TimeoutExpired) as exc:
                if "permission denied" in str(exc).lower():
                    auth_denied_since = auth_denied_since or time.time()
                    denied_for = time.time() - auth_denied_since
                    if (
                        not key_reattached
                        and denied_for >= VAST_SSH_DENIED_ATTACH_S
                        and hasattr(provider, "attach_ssh_key")
                    ):
                        typer.echo(
                            "SSH key auth denied after instance start; "
                            "re-attaching the key to the instance..."
                        )
                        provider.attach_ssh_key(instance_id)
                        key_reattached = True
                    elif denied_for >= VAST_SSH_DENIED_GIVE_UP_S:
                        raise ProvisioningDoomed(
                            f"Instance {instance_id} is running but has refused pubkey "
                            f"auth for {int(denied_for)}s (host failed to inject the SSH "
                            f"key). This does not self-heal: rent another offer."
                        )
                else:
                    auth_denied_since = None
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
    assert info.ssh
    if config.gui_enabled:
        script = build_gui_setup_script(config)
    else:
        script = build_isaac_container_launch_script(config)
    output = run_ssh_script(config, info.ssh, script, in_container=True, timeout_seconds=900)
    typer.echo(output.splitlines()[-1] if output else "(no output)")


def print_access(config: AppConfig, info: InstanceInfo) -> None:
    assert info.ssh
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
        f"Tunnel (recommended): uv run python isaac_cloud.py tunnel "
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
    completed = subprocess.run(
        ["aws", *args],
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout_seconds,
        env=build_local_aws_env(config),
    )
    if completed.returncode != 0:
        output = ((completed.stdout or "") + "\n" + (completed.stderr or "")).strip()
        _raise(output or f"aws {' '.join(args[:3])} failed.")
    return (completed.stdout or "").strip()


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
    provider,
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
    provider,
    info: InstanceInfo,
    project: str,
    *,
    snapshot: str | None = None,
    timeout_seconds: int = 3600,
) -> str:
    """Restore a snapshot into the remote project directory via an atomic swap.

    The live directory is only replaced after the archive has fully extracted,
    so a dropped connection leaves the previous contents intact."""
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
    old = remote_path + ".old"
    script = (
        f"set -e; rm -rf {shell_quote(incoming)}; mkdir -p {shell_quote(incoming)}; "
        f"tar -C {shell_quote(incoming)} -xzf -; "
        f"rm -rf {shell_quote(old)}; mv {shell_quote(remote_path)} {shell_quote(old)} 2>/dev/null || true; "
        f"mv {shell_quote(incoming)} {shell_quote(remote_path)}; rm -rf {shell_quote(old)}; "
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
    help="Launch and manage NVIDIA Isaac Sim on Vast.ai or AWS EC2, accessed over SSH only.",
    no_args_is_help=True,
)
sync_app = typer.Typer(help="Save/restore project snapshots (append-only) between the instance and S3.")
app.add_typer(sync_app, name="sync")

PROVIDER_OPTION = typer.Option(None, "--provider", "-p", help="vast or aws (default from config).")


def _config() -> AppConfig:
    return load_app_config()


@app.command()
def catalog(provider: str = PROVIDER_OPTION) -> None:
    """List available GPU offers/instance options for the provider."""
    config = _config()
    prov = get_provider(config, provider)
    try:
        if prov.name == "vast":
            offers = prov.catalog()
            if not offers:
                typer.echo("No matching offers.")
                raise typer.Exit(1)
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
    except IsaacCloudError as exc:
        typer.echo(f"Error: {exc}")
        raise typer.Exit(1)


@app.command()
def instances(provider: str = PROVIDER_OPTION) -> None:
    """List managed instances."""
    config = _config()
    prov = get_provider(config, provider)
    try:
        rows = prov.list_instances()
    except IsaacCloudError as exc:
        typer.echo(f"Error: {exc}")
        raise typer.Exit(1)
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
    agent: bool = typer.Option(None, "--agent/--no-agent", help="Enable the agent control socket."),
    project: str = typer.Option(
        None, "--project", help="Project namespace to rehydrate from / save to (default from config)."
    ),
    timeout_seconds: int = typer.Option(1200, help="How long to wait for SSH readiness."),
) -> None:
    """Rent/launch an instance, start Isaac, and print SSH tunnel commands."""
    config = _config()
    if gui is not None:
        config = replace(config, gui_enabled=gui)
    if agent is not None:
        config = replace(config, agent_enabled=agent)
    prov = get_provider(config, provider)
    live_instance_id: str | None = None
    try:
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
def status(
    instance_id: str = typer.Option(..., "--instance-id"),
    provider: str = PROVIDER_OPTION,
) -> None:
    """Show instance state plus in-container readiness probes."""
    config = _config()
    prov = get_provider(config, provider)
    try:
        info = prov.get(instance_id)
        ssh = f"{info.ssh.user}@{info.ssh.host}:{info.ssh.port}" if info.ssh else "-"
        typer.echo(f"{info.provider}:{info.instance_id} {info.status} {ssh}")
        if info.ssh and check_tcp_connectivity(info.ssh.host, info.ssh.port):
            probe = run_ssh_script(
                config, info.ssh, build_container_probe_script(),
                in_container=True, timeout_seconds=120,
            )
            typer.echo(probe)
            typer.echo(probe_local_tunnel())
            print_access(config, info)
        else:
            typer.echo("SSH not reachable yet.")
    except IsaacCloudError as exc:
        typer.echo(f"Error: {exc}")
        raise typer.Exit(1)


@app.command()
def stop(
    instance_id: str = typer.Option(..., "--instance-id"),
    provider: str = PROVIDER_OPTION,
    project: str = typer.Option(None, "--project", help="Project to save into (default: launch label, then config)."),
    skip_push: bool = typer.Option(False, "--skip-push", help="Skip the snapshot save before stopping."),
) -> None:
    """Stop an instance (saves a project snapshot to S3 first if enabled)."""
    config = _config()
    prov = get_provider(config, provider)
    try:
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
    except IsaacCloudError as exc:
        typer.echo(f"Error: {exc}")
        raise typer.Exit(1)


@app.command()
def resume(
    instance_id: str = typer.Option(..., "--instance-id"),
    provider: str = PROVIDER_OPTION,
) -> None:
    """Start a stopped instance and relaunch Isaac."""
    config = _config()
    prov = get_provider(config, provider)
    try:
        prov.start(instance_id)
        info = wait_for_ssh(config, prov, instance_id)
        wait_for_container(config, info)
        setup_isaac(config, info)
        print_access(config, info)
    except IsaacCloudError as exc:
        typer.echo(f"Error: {exc}")
        raise typer.Exit(1)


@app.command()
def destroy(
    instance_id: str = typer.Option(None, "--instance-id"),
    provider: str = PROVIDER_OPTION,
    all_instances: bool = typer.Option(False, "--all"),
    yes: bool = typer.Option(False, "--yes", "-y"),
    project: str = typer.Option(None, "--project", help="Project to save into (default: launch label, then config)."),
    skip_push: bool = typer.Option(False, "--skip-push", help="Destroy without saving a snapshot first."),
) -> None:
    """Destroy instance(s), saving a project snapshot to S3 first if enabled.

    If the snapshot cannot be taken (SSH unreachable, save fails), the instance
    is NOT destroyed — destroying would silently discard the work. Use
    --skip-push to destroy anyway."""
    config = _config()
    prov = get_provider(config, provider)
    try:
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
    except IsaacCloudError as exc:
        typer.echo(f"Error: {exc}")
        raise typer.Exit(1)


@app.command()
def tunnel(
    instance_id: str = typer.Option(..., "--instance-id"),
    provider: str = PROVIDER_OPTION,
) -> None:
    """Run a supervised SSH tunnel to the instance (auto-reconnects; Ctrl-C to stop).

    Forwards agent control (8226), RTSP (8554), and noVNC (6080) to localhost.
    SSH keepalives detect dead/zombie connections within ~30s and the tunnel
    re-establishes itself with backoff, so client sessions (noVNC, agent
    scripts, RTSP players) see a brief blip instead of needing manual repair.
    """
    config = _config()
    prov = get_provider(config, provider)
    try:
        run_supervised_tunnel(config, prov, instance_id)
    except IsaacCloudError as exc:
        typer.echo(f"Error: {exc}")
        raise typer.Exit(1)


@sync_app.command("pull")
def sync_pull(
    instance_id: str = typer.Option(..., "--instance-id"),
    provider: str = PROVIDER_OPTION,
    project: str = typer.Option(None, "--project", help="Project to restore from (default: launch label, then config)."),
    snapshot: str = typer.Option(
        None, "--snapshot", help="Snapshot name/timestamp to restore (default: newest)."
    ),
) -> None:
    """Restore a project snapshot from S3 into the instance (atomic swap)."""
    config = _config()
    prov = get_provider(config, provider)
    try:
        info = prov.get(instance_id)
        target_project = resolve_project(config, info, project)
        typer.echo(snapshot_pull(config, prov, info, target_project, snapshot=snapshot))
    except IsaacCloudError as exc:
        typer.echo(f"Error: {exc}")
        raise typer.Exit(1)


@sync_app.command("push")
def sync_push(
    instance_id: str = typer.Option(..., "--instance-id"),
    provider: str = PROVIDER_OPTION,
    project: str = typer.Option(None, "--project", help="Project to save into (default: launch label, then config)."),
    allow_empty: bool = typer.Option(
        False, "--allow-empty", help="Save a snapshot even if the project directory is empty."
    ),
) -> None:
    """Save the instance's project directory as a new snapshot in S3."""
    config = _config()
    prov = get_provider(config, provider)
    try:
        info = prov.get(instance_id)
        target_project = resolve_project(config, info, project)
        typer.echo(snapshot_push(config, prov, info, target_project, allow_empty=allow_empty))
    except IsaacCloudError as exc:
        typer.echo(f"Error: {exc}")
        raise typer.Exit(1)


@sync_app.command("list")
def sync_list(
    project: str = typer.Option(None, "--project", help="List snapshots of this project only."),
) -> None:
    """List saved projects and their snapshots."""
    config = _config()
    try:
        projects = [validate_project_name(project)] if project else list_projects(config)
        if not projects:
            typer.echo("No saved projects.")
            return
        for name in projects:
            snaps = list_snapshots(config, name)
            typer.echo(f"{name}: {len(snaps)} snapshot(s)")
            for snap in snaps:
                typer.echo(f"  {snap['name']:32s} {snap['size'] / 1e6:8.1f} MB")
    except IsaacCloudError as exc:
        typer.echo(f"Error: {exc}")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
