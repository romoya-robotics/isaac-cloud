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
uv run python isaac_cloud.py launch --curobo             # + cuRobo motion planning (bg install)
uv run python isaac_cloud.py launch --provider aws
uv run python isaac_cloud.py instances
uv run python isaac_cloud.py status  --instance-id <ID>
uv run python isaac_cloud.py tunnel  --instance-id <ID>   # supervised, auto-reconnecting
uv run python isaac_cloud.py sync list                   # saved projects + snapshots
uv run python isaac_cloud.py sync pull --instance-id <ID> [--project P] [--snapshot TS]
uv run python isaac_cloud.py sync push --instance-id <ID> [--project P]
uv run python isaac_cloud.py stop    --instance-id <ID>
uv run python isaac_cloud.py resume  --instance-id <ID>
uv run python isaac_cloud.py destroy --instance-id <ID> --yes
```

`launch` prints an SSH command plus a ready-made tunnel command; run the
tunnel in a spare terminal and every service above is on `localhost`.

First boot compiles RTX shaders — allow 5–10 minutes before the sim is
responsive. Warm restarts take under a minute.

## Driving Isaac from Claude Code

Two complementary integrations. Both assume an instance is up and the tunnel
is running (`uv run python isaac_cloud.py tunnel --instance-id <ID>`), which
puts Isaac's agent control socket at `localhost:8226`. The socket requires
`[isaac].agent = true` (the default); `status` reports `port 8226: open`
once Isaac has finished loading.

### Live control — the `isaac-sim-remote` skill

NVIDIA ships an official Claude Code skill in the Isaac Sim repo
([`skills/isaac-sim-remote`](https://github.com/isaac-sim/IsaacSim/tree/main/skills/isaac-sim-remote))
that executes Python inside the running sim over that socket: USD stage
manipulation, play/pause/step, screenshots, annotator data (depth,
segmentation), with named execution contexts that persist state between
calls. Install it by checking out the skill directory and linking it into
your skills folder:

```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/isaac-sim/IsaacSim.git
git -C IsaacSim sparse-checkout set skills/isaac-sim-remote

ln -s "$PWD/IsaacSim/skills/isaac-sim-remote" ~/.claude/skills/isaac-sim-remote  # personal
# or project-scoped, shared with the repo: cp -r IsaacSim/skills/isaac-sim-remote .claude/skills/
```

Claude Code follows symlinks and picks up skill changes without a restart.
Invoke it with `/isaac-sim-remote` (or just describe what you want in the
sim — the skill self-selects when relevant). On a cold instance, wait for
the shader compile to finish (`status` probe shows the app ready) before
driving it.

### Reference lookup — the official Isaac Sim MCP server

NVIDIA also publishes an [Isaac Sim MCP server](https://docs.isaacsim.omniverse.nvidia.com/latest/development_tools/isaac_sim_mcp.html)
([source](https://github.com/NVIDIA-Omniverse/kit-usd-agents/tree/main/source/mcp/isaacsim_mcp)).
Know what it is: a **documentation/knowledge server** — semantic search over
Isaac extensions, code examples, and settings — not a control channel to
your instance. It runs locally in Docker (needs an NVIDIA API key from
build.nvidia.com):

```bash
git clone https://github.com/NVIDIA-Omniverse/kit-usd-agents.git
cd kit-usd-agents/source/mcp/isaacsim_mcp && ./build-docker.sh
docker run --rm -p 9904:9904 --env-file ../.env isaacsim-mcp:latest

claude mcp add --transport http isaac-sim-mcp http://localhost:9904
```

Check it with `/mcp` inside a session, or `claude mcp list`. Pairing the two
works well: the MCP server answers "how do I do X in Isaac", the skill then
does X in your live sim.

No official MCP server wraps the live `8226` socket — live control goes
through the skill.

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
(`/isaac-sim/project` in the container) is saved to S3 as **append-only
snapshots**, namespaced by project:

```
s3://<bucket>/<base>/projects/<project>/snapshots/<utc-timestamp>.tar.gz
```

- `launch` rehydrates the newest snapshot of the chosen project (a project
  with no snapshots just starts fresh)
- `stop` / `destroy` save a new snapshot first — and **refuse to proceed if
  the save fails** (`--skip-push` overrides)
- `sync push` / `sync pull` anytime; `sync pull --snapshot <name>` rolls
  back to an older save; `sync list` shows what's stored
- choose the namespace per run with `--project` (different users or
  workstreams use different names); instances remember the project they were
  launched with, so `stop`/`destroy` save back to the right one
- the last `[persistence].keep_last` snapshots per project are retained
  (default 10); older ones are pruned after each successful save

Saves never overwrite or delete existing snapshots, restores fully extract
before deleting anything (a dropped connection can't leave the project
half-restored, and the container bind mount on AWS is preserved), and
pushing an empty project directory is skipped rather than saved. Transfers
run **through your local machine** (SSH + tar + `aws s3 cp`): cloud instances
never receive AWS credentials.

On AWS the project directory also lives on the VM at
`/home/ubuntu/isaac-cloud/project` (bind-mounted into the container), so it
additionally survives container restarts.

## Config reference

See `config.example.toml`. Highlights:

- `[defaults].provider` — `vast` or `aws`; `--provider` overrides per command.
- `[isaac].version` — Isaac Sim image tag (default `6.0.1`).
- `[isaac].agent` — agent control socket (default true).
- `[gui].enabled` / `resolution` — noVNC GUI stack (default off; `--gui` per launch).
- `[isaac].curobo` — install cuRobo into Isaac's python after launch (default off;
  `--curobo` per launch). Background, ~5 min; `status` probe reports `curobo: ready`.
- `[vast].whole_machine` / `min_reliability` / `query` — offer selection.
- `[aws].region` / `instance_type` — defaults `us-west-2` / `g6e.xlarge`.
- `[persistence].s3_uri` — `s3://bucket/path/` base for snapshot storage.
- `[persistence].project` / `keep_last` — default namespace, snapshots kept.

## Background

This tool previously targeted TensorDock, whose marketplace emptied out after
the Voltage Park acquisition (2025–2026). The experiment logs from the
migration — including why WebRTC browser streaming over SSH tunnels was
abandoned in favor of noVNC, and the NVENC device-index discovery — live in
`docs/VAST_EXPERIMENT_RESULTS.md` and `docs/GUI_TUNNEL_EXPERIMENT_PLAN.md`.
