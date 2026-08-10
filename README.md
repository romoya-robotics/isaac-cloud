# isaac-cloud

Launch and manage NVIDIA Isaac Sim 6 on cloud GPUs, accessed **over SSH only**.

Two providers:

| | `vast` (Vast.ai) | `aws` (EC2) |
| --- | --- | --- |
| Model | the Isaac container **is** the instance | VM (Deep Learning Base AMI) running the Isaac container |
| Cost | ~$0.32/hr (RTX 4090) | ~$1.9/hr (g6e.xlarge, L40S) |
| Startup | instant marketplace, hosts vary | slower, but consistent |
| Best for | day-to-day dev, experiments | must-not-flake runs, AWS-native persistence |

Every access path binds to localhost on the remote side and is reached through
an SSH tunnel the CLI prints for you — nothing Isaac-related is exposed to the
internet, and your SSH key is the only authentication that exists:

- **Agent control** (`127.0.0.1:8226`) — Isaac 6's built-in
  `isaacsim.code_editor.python_server`. Drive the live sim with Python from
  Claude Code via the official `isaac-sim-remote` skill
  (from [isaac-sim/IsaacSim](https://github.com/isaac-sim/IsaacSim) `skills/`).
- **RTSP cameras** (`rtsp://127.0.0.1:8554/stream`) — TCP camera feeds via
  `isaacsim.streaming.rtsp`; watch in VLC/ffplay. Requires NVENC (see GPU
  notes below).
- **Full GUI** (`http://localhost:6080/vnc.html`, `--gui`) — the native Isaac
  Sim application rendered into a virtual display and served with noVNC over a
  single TCP port. Works on any host; NVENC not required.

## Setup

```bash
uv sync
# Vast provider:
uv tool install vastai
vastai set api-key <YOUR_VAST_KEY>
# AWS provider:
aws login   # and G-instance vCPU quota > 0 (Service Quotas code L-DB2E81BA)

cp config.example.toml config.toml   # then fill in [ngc] and [ssh]
```

## Usage

```bash
uv run python isaac_cloud.py catalog                     # browse offers
uv run python isaac_cloud.py launch                      # headless + agent socket
uv run python isaac_cloud.py launch --gui                # + noVNC GUI
uv run python isaac_cloud.py launch --provider aws
uv run python isaac_cloud.py instances
uv run python isaac_cloud.py status  --instance-id <ID>
uv run python isaac_cloud.py tunnel  --instance-id <ID>   # supervised, auto-reconnecting
uv run python isaac_cloud.py sync pull --instance-id <ID>
uv run python isaac_cloud.py sync push --instance-id <ID>
uv run python isaac_cloud.py stop    --instance-id <ID>
uv run python isaac_cloud.py resume  --instance-id <ID>
uv run python isaac_cloud.py destroy --instance-id <ID> --yes
```

`launch` prints an SSH command plus a ready-made tunnel command; run the
tunnel in a spare terminal and every service above is on `localhost`.

First boot compiles RTX shaders — allow 5–10 minutes before the sim is
responsive. Warm restarts take under a minute.

## GPU notes (important for video)

**NVENC (hardware H.264) only works when the rented GPU is host GPU 0.**
This is an NVIDIA driver limitation
([k8s-device-plugin#1282](https://github.com/NVIDIA/k8s-device-plugin/issues/1282)),
not a provider quirk. Consequences:

- The default Vast query rents **whole machines** (`gpu_frac=1`), which
  guarantees GPU 0. Set `[vast].whole_machine = false` to allow cheaper
  fractional hosts — agent control and the noVNC GUI still work there, but
  RTSP/WebRTC video will fail if you draw the wrong GPU slot.
- EC2 instances always see their GPU as device 0; NVENC always works there.

A minority of Vast hosts inject compute-only NVIDIA libraries (no
Vulkan/GLX/NVENC userland). The launch script detects this and side-loads the
exact driver-matched libraries automatically.

## Persistence

When `[persistence].enabled = true`, the project directory
(`/isaac-sim/project` in the container) is synced with S3:

- restored from S3 automatically on `launch`
- pushed to S3 automatically on `stop` / `destroy`
- manual `sync pull` / `sync push` anytime

Sync runs **through your local machine** (SSH + tar + `aws s3 sync`): cloud
instances never receive AWS credentials.

On AWS the project directory also lives on the VM at
`/home/ubuntu/isaac-cloud/project` (bind-mounted into the container), so it
additionally survives container restarts.

## Config reference

See `config.example.toml`. Highlights:

- `[defaults].provider` — `vast` or `aws`; `--provider` overrides per command.
- `[agent].enabled` — agent control socket (default true).
- `[gui].enabled` / `resolution` — noVNC GUI stack (default off; `--gui` per launch).
- `[vast].whole_machine` / `min_reliability` / `query` — offer selection.
- `[aws].region` / `instance_type` — defaults `us-west-2` / `g6e.xlarge`.
- `[persistence].s3_uri` — `s3://bucket/path/` workspace location.

## Background

This tool previously targeted TensorDock, whose marketplace emptied out after
the Voltage Park acquisition (2025–2026). The experiment logs from the
migration — including why WebRTC browser streaming over SSH tunnels was
abandoned in favor of noVNC, and the NVENC device-index discovery — live in
`VAST_EXPERIMENT_RESULTS.md` and `GUI_TUNNEL_EXPERIMENT_PLAN.md`.
