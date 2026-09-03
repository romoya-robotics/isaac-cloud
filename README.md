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
  single TCP port. Works on any host; NVENC not required. Needs a host whose
  driver can present Vulkan on an X display (driver >= 590 in practice; see
  [The GUI stack](#the-gui-stack)).

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
uv run python isaac_cloud.py catalog [--gui]             # browse offers (--gui ranks driver >= 590 first)
uv run python isaac_cloud.py launch                      # headless + agent socket
uv run python isaac_cloud.py launch --gui                # + noVNC GUI
uv run python isaac_cloud.py launch --curobo             # + cuRobo motion planning (bg install)
uv run python isaac_cloud.py launch --lab                # + Isaac Lab (bg install)
uv run python isaac_cloud.py launch --provider aws
uv run python isaac_cloud.py instances
uv run python isaac_cloud.py status  --instance-id <ID>
uv run python isaac_cloud.py tunnel  --instance-id <ID>   # supervised, auto-reconnecting
uv run python isaac_cloud.py tunnel  --instance-id <ID2> --novnc-port 16080 --agent-port 18226   # second box
uv run python isaac_cloud.py sync list                   # saved projects + snapshots
uv run python isaac_cloud.py sync pull --instance-id <ID> [--project P] [--snapshot TS]
uv run python isaac_cloud.py sync push --instance-id <ID> [--project P]
uv run python isaac_cloud.py stop    --instance-id <ID>
uv run python isaac_cloud.py resume  --instance-id <ID>  # relaunches the GUI stack if the box had one
uv run python isaac_cloud.py destroy --instance-id <ID> --yes
```

`launch` prints an SSH command plus a ready-made tunnel command; run the
tunnel in a spare terminal and every service above is on `localhost`.

First boot compiles RTX shaders — allow 5–10 minutes before the sim is
responsive. Warm restarts take under a minute.

## The GUI stack

`launch --gui` (and `resume` on a box that had it) writes `/root/gui_stack.sh`
into the container and runs it in the foreground. The script is idempotent
and strictly ordered — each step is guarded by its own process or port check,
so re-running it (a repair, `resume`, a project's own relaunch hook) only
starts what is missing:

1. apt deps: `xvfb x11vnc novnc websockify xdotool x11-utils x11-apps vulkan-tools imagemagick`
2. `Xvfb :1` at `[gui].resolution`, then **wait until `DISPLAY=:1 xdpyinfo` answers**
3. Vulkan presentation preflight (`DISPLAY=:1 vulkaninfo --summary`); aborts
   the launch with a clear message if the host cannot present
4. `websockify --web /usr/share/novnc 127.0.0.1:6080 localhost:5901`
5. `x11vnc -display :1 -localhost -forever -shared -nopw -noxdamage -rfbport 5901`
   inside a supervision loop (`/root/x11vnc_loop.sh`) that restarts it on exit
6. the GUI kit (`isaac-sim.sh --allow-root`, plus the agent extension when
   `[isaac].agent`), after stopping any headless streaming kit — one GPU, one
   agent port
7. **wait for a mapped "Isaac Sim" window AND port 8226** (8226 alone is not
   readiness), then `xdotool windowmove 0 0 windowsize <res>`
8. the checks below; the last line is `GUI_STACK_READY`

`status` (and `/root/gui_stack.sh check` on the box) reports the same checks:

```
gui_x: up (:1 1920x1080)
gui_vulkan: can present on :1
gui_vnc: port 5901 open (supervised)
gui_novnc: port 6080 open
gui_kit: window mapped (id 41943041)
gui_agent: port 8226 open
gui_screen: non-black (mean 0.137, 212 gray levels) /root/gui_screen.png
```

Failure modes the script encodes (all observed on Vast hosts, 2026-09-01..03):

- **Kit started before the X display existed.** It answers on 8226 and looks
  ready, but its window never maps; `/root/isaac_gui.log` says
  "backbuffers are not initialized" and the VNC view is black. The stack waits
  for `xdpyinfo` before starting the kit, and restarts a kit whose window is
  unmapped with that signature in its log.
- **Bare `x11vnc` dies on the first XIO error** and the viewer freezes. It runs
  under a supervisor, with `-noxdamage`.
- **`pgrep -f "Xvf[b] :1" || Xvfb :1 ...` in one `ssh box '...'` command
  matches the remote shell's own argv**, so the guard is always true and the
  service is silently never started; likewise a `pkill -f x11vnc` in the same
  command line as the new supervisor's start kills the new supervisor. The
  stack runs from a script file whose argv contains none of the service
  names, and guards with `pgrep -x`. Keep it that way if you edit it.
- **Driver 580 hosts cannot present Vulkan on the X display** (kit logs
  "vkCreateSwapchainKHR failed", GUI black, headless fine). The preflight
  aborts with `GUI_STACK_VULKAN_PRESENT_FAILED`; `catalog --gui` and
  `launch --gui` rank driver >= 590 offers first and warn otherwise.
- **Headless and GUI kits cannot coexist**; the stack stops the headless kit
  before starting the GUI one.

Set `GUI_STACK_TIMEOUT` (default 600 s) on the box to change how long the
script waits for the kit. Two boxes at once: give the second tunnel its own
local ports (`tunnel --novnc-port 16080 --agent-port 18226`; `status
--agent-port 18226` probes that tunnel).

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
- `[gui].enabled` / `resolution` — noVNC GUI stack (default off; `--gui` per launch;
  see [The GUI stack](#the-gui-stack)).
- `[isaac].curobo` — install cuRobo into Isaac's python after launch (default off;
  `--curobo` per launch). Background, ~5 min; `status` probe reports `curobo: ready`.
- `[isaac].lab` — install Isaac Lab into Isaac's python after launch (default off;
  `--lab` per launch). Background, ~15 min; `status` probe reports `isaac_lab: ready`.
- `[isaac].lab_ref` — IsaacLab git ref (tag or branch) to install (default
  `v3.0.0-beta2.patch1`, the release built for Isaac Sim 6.0.1). Lab releases are
  paired with Isaac Sim versions — bump together with `[isaac].version`.
  Lab scripts launch their own SimulationApp: stop the streaming Isaac first
  (`pkill -f kit/kit` in the container), and keep outputs under `/isaac-sim/project`
  so stop/destroy snapshots capture them.
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
