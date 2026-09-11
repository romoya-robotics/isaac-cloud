# isaac-cloud

Launch and manage NVIDIA Isaac Sim 6 on cloud GPUs, with **SSH access** and
optional **WebRTC viewing on Vast.ai and AWS EC2**.

Two providers:

| | `vast` (Vast.ai) | `aws` (EC2) |
| --- | --- | --- |
| Model | the Isaac container **is** the instance | VM (Deep Learning Base AMI) running the Isaac container |
| Cost | ~$0.32/hr (RTX 4090) | ~$1.9/hr (g6e.xlarge, L40S) |
| Startup | instant marketplace, hosts vary | slower, but consistent |
| Best for | day-to-day dev, experiments | must-not-flake runs, AWS-native persistence |

By default, access paths bind to localhost on the remote side and are reached
through an SSH tunnel the CLI prints for you:

- **Agent control** (`127.0.0.1:8226`) — Isaac 6's built-in
  `isaacsim.code_editor.python_server`. Drive the live sim with Python from
  Claude Code via the official `isaac-sim-remote` skill
  (from [isaac-sim/IsaacSim](https://github.com/isaac-sim/IsaacSim) `skills/`).
- **RTSP cameras** (`rtsp://127.0.0.1:8554/stream`) — TCP camera feeds via
  `isaacsim.streaming.rtsp`; watch in VLC/ffplay. Requires NVENC (see GPU
  notes below).
- **GUI over VNC** (`http://localhost:6080/vnc.html`, `--gui vnc`) — the native
  Isaac Sim application rendered into a virtual display and served with noVNC
  over a single TCP port. Works on any host; NVENC not required. Needs a host
  whose driver can present Vulkan on an X display (driver >= 590 in practice;
  see [The GUI stack](#the-gui-stack)).
- **GUI over WebRTC** (`http://localhost:8210/`, `--gui webrtc`, Vast or AWS) —
  the same Isaac UI via its built-in streaming: encoded on the remote GPU,
  displayed and controlled in a local Chromium browser. The viewer page is
  served from the container on loopback like noVNC; signaling uses SSH; media
  uses a public UDP relay restricted to your client IP. Needs host GPU 0 for
  NVENC (whole-machine Vast offers). See the experimental workflow below.

`--gui` takes one of `none` (headless, the default), `vnc`, or `webrtc`; see
[GUI modes](#gui-modes) for how to choose.

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
uv run python isaac_cloud.py catalog [--gui vnc|webrtc]  # browse offers (vnc ranks driver >= 590 first; webrtc = whole machines)
uv run python isaac_cloud.py launch                      # headless + agent socket
uv run python isaac_cloud.py launch --gui vnc            # + noVNC GUI
uv run python isaac_cloud.py launch --gui webrtc         # + native WebRTC streaming (UDP mapping reserved at launch)
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

## GUI modes

Both GUI modes show the same Isaac Sim editor in your browser through the SSH
tunnel; they differ in how the picture gets to you and what the host must
provide. An instance runs exactly one of them (they are different Isaac
kits), chosen with `--gui` at `launch` or `[gui].mode` in `config.toml`.

| Mode | Open | How the video travels | Host needs | What it feels like |
| --- | --- | --- | --- | --- |
| `none` (default) | nothing | none: agent control and RTSP only | any | headless |
| `vnc` | `http://localhost:6080/vnc.html` | Isaac draws on a virtual X display; x11vnc + noVNC send screen updates over TCP inside the SSH tunnel | driver >= 590 (must present Vulkan on X); **any GPU slot** | noticeably lower framerate and higher latency, since each update is a lossless screen diff squeezed through SSH; fine for editing scenes and inspecting state, not for watching motion smoothly |
| `webrtc` | `http://localhost:8210/` | Isaac encodes H.264 on the GPU (NVENC) and streams it over direct UDP to your browser; only the page and signaling go through SSH | **host GPU 0** (whole-machine Vast offers; any AWS `g6e`), UDP reachable from your network; experimental | smooth video at up to 60 fps with low latency |

Rules of thumb:

- `vnc` is the proven path and works on cheap fractional Vast hosts. Start
  there if you just need to look at and edit a scene.
- `webrtc` is for watching the robot move. It is fixed at launch: Vast must
  reserve the UDP port when the instance is created, so `resume` keeps the mode
  the box was launched with and `--gui webrtc` cannot be added to an existing
  instance. `catalog --gui webrtc` and `launch --gui webrtc` search
  whole-machine offers only.
- `resume` without `--gui` picks the mode the box already has: `webrtc` if it
  was launched that way, else `vnc` if the GUI stack is installed, else `none`.

**Why WebRTC needs host GPU 0.** The stream is encoded by NVENC, and NVIDIA's
encoder library assumes that `/dev/nvidiaN` is the GPU with index N. On a
fractional Vast rental the container's only GPU is index 0 but its device
node keeps the host's slot number, so the encoder opens the wrong device and
fails with `OpenEncodeSessionEx failed: unsupported device`. Rendering, CUDA,
agent control, and noVNC are unaffected. A fractional offer that happens to
sit in slot 0 works, which is why an explicit `--offer-id` is allowed and the
GPU minor number is checked at boot before Isaac starts. Details and the
upstream issue are in [GPU notes](#gpu-notes-important-for-video).

## The GUI stack

`launch --gui vnc` (and `resume` on a box that had it) writes `/root/gui_stack.sh`
into the container and runs it in the foreground. The script is idempotent
and strictly ordered — each step is guarded by its own process or port check,
so re-running it (a repair, `resume`, a project's own relaunch hook) only
starts what is missing:

1. apt deps: `xvfb x11vnc novnc websockify xdotool x11-utils x11-apps vulkan-tools imagemagick`
   plus the video tools every launch path installs: `ffmpeg` (with `ffprobe`) and
   `libx264`, used to capture clips of the robot from the sim
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
  aborts with `GUI_STACK_VULKAN_PRESENT_FAILED`; `catalog --gui vnc` and
  `launch --gui vnc` rank driver >= 590 offers first and warn otherwise.
- **Headless and GUI kits cannot coexist**; the stack stops the headless kit
  before starting the GUI one.

Set `GUI_STACK_TIMEOUT` (default 600 s) on the box to change how long the
script waits for the kit. Two boxes at once: give the second tunnel its own
local ports (`tunnel --novnc-port 16080 --agent-port 18226`; `status
--agent-port 18226` probes that tunnel).

## WebRTC browser viewing (experimental)

Simulation, rendering, and video encoding run in Isaac Sim on the remote GPU.
Your local computer decodes the video and sends input using NVIDIA's
Omniverse WebRTC SDK. This does not require a local Isaac Sim installation.

Nothing is built or installed locally. Launch a **new** instance with
`--gui webrtc`, then run the usual tunnel (use `--provider aws` in both
commands for AWS):

```bash
uv run python isaac_cloud.py launch --provider vast --gui webrtc
uv run python isaac_cloud.py tunnel --provider vast --instance-id <ID>
```

Open **http://localhost:8210/** in Chrome or Edge and click **Connect** once
Isaac has loaded. `status --provider vast --instance-id <ID>` reports the
Isaac log readiness and the signaling and viewer ports. Only one streaming
client should be connected at a time. The viewer reports **Connected** when
video starts playing, rather than treating a signaling handshake as working
video.

The viewer page lives in the container, exactly like noVNC: `launch`/`resume`
install `webrtc-viewer/` from this repo into `/root/webrtc-viewer/` on the box,
download NVIDIA's streaming SDK module next to it (pinned by SHA-256), and
serve the directory on `127.0.0.1:8210` with the container's own Python. For a
WebRTC instance, `tunnel` forwards that page and the signaling port instead of
noVNC (agent control and RTSP as usual), starts the remote UDP media relay for
your IP, and writes the media endpoint into the page's `connection.json` on
every (re)connect. Ctrl-C attempts to stop the relay; the page keeps being
served and the GPU instance continues running and billing. Stop or destroy it
with the existing lifecycle commands when finished.

The topology adapts native Isaac streaming to Vast's container networking:

```text
Local browser --TCP signaling via SSH--> Isaac 127.0.0.1:49100
Local browser <--UDP--> Vast public IP:mapped UDP port
                       -> container UDP relay :47999
                       <-> Isaac 127.0.0.1:47998
```

`launch --gui webrtc` requests only the relay's UDP mapping. Isaac stays bound to
loopback, avoiding the NVIDIA SDK's attempt to bind a host public IP that is
absent inside the Vast container. The browser SDK's `mediaServer` and
`mediaPort` overrides target the mapped UDP endpoint. The relay carries UDP
directly; it does not encapsulate video in TCP or SSH.

The relay allows only the public IPv4 seen by SSH. If a VPN or different UDP
route changes that address, pass `tunnel --client-ip <YOUR_PUBLIC_IPV4>`.
The relay is started when you connect. Media does not travel over SSH, so an
SSH reconnect keeps the running relay and ingress rule, and replaces them only
if your public IPv4 changed; failed reconnect steps are retried with backoff.
After stopping/resuming the instance or changing networks, reload the browser
page.

WebRTC mode is fixed at launch: `resume` reads it from the instance (Vast's
recorded UDP port option, AWS's instance tag), not from `[gui].mode`, which
only sets the default for `launch`. `resume --gui vnc` on such a box is
rejected, and `resume --gui webrtc` on a box launched without it is too.

On AWS, the existing Docker container uses host networking. The viewer sends
UDP to the instance's public IPv4 on port `47999`; the same container relay
forwards it to Isaac on `127.0.0.1:47998`. Signaling still uses SSH.

On AWS the `tunnel` command temporarily adds UDP `47999` ingress for your public
IPv4 (`/32`) to the first attached security group (instances launched by this
tool have exactly one: `[aws].security_group`). It removes the rule on exit,
including if relay startup fails. Your local AWS credentials need
`ec2:AuthorizeSecurityGroupIngress`, `ec2:RevokeSecurityGroupIngress`, and
`ec2:DescribeSecurityGroupRules`, in addition to the existing instance
permissions. No AWS credentials are copied to the instance.

Security group rules apply to every instance sharing that group. Use a
separate `[aws].security_group` for isolated deployments. If the viewer is
killed before cleanup, its rule (description `isaac-cloud WebRTC <instance>`)
stays until the next `tunnel` from the same IP adopts it and removes it on
exit; remove it manually if you will not reconnect from that IP. An identical
rule with any other description is used as-is and never modified. AWS treats
identical rules as one, so two viewers from one IP through one group share it,
and the first to exit removes it. The relay also restricts traffic to the same
client IP. Network ACLs and host firewalls must allow the traffic; this command
does not modify them.

AWS records WebRTC mode in the `IsaacCloudWebRTC=true` instance tag. A public IPv4 and
an NVIDIA GPU with working video encoding are required (the default AWS
instance type is `g6e.xlarge`).

Requirements and limits:

- On Vast, `--gui webrtc` searches whole-machine offers only (`gpu_frac=1`,
  regardless of `[vast].whole_machine`) so the GPU is host GPU 0 for NVENC; an
  explicit `--offer-id` is allowed, and setup checks every host for GPU minor 0.
  This check does not guarantee that every host has working hardware encoding.
- `vnc` and `webrtc` run different Isaac kits, so `--gui` picks exactly one.
- Existing Vast SSH-only instances lack the UDP mapping and need a new instance.
  AWS instances must be launched with `--gui webrtc` so setup and resume use streaming mode.
- The browser's network must permit UDP to the mapped port. There is no TURN
  fallback. A black screen can mean blocked UDP, an incorrect client IP,
  incomplete shader compilation, or an NVENC failure. Inspect
  `/root/isaac.log`, `/root/isaac_webrtc_relay.log`,
  `/root/isaac_webrtc_viewer.log`, and Chrome's `chrome://webrtc-internals`
  for diagnostics.
- **Live video verified on Vast:** 2026-09-08 with the original locally built
  viewer (Isaac 6.0.1, RTX 5080: 1920×1080, 962 frames in about 18 s, some
  drops and brief freezes), and 2026-09-11 with the container-served viewer
  described above (Isaac 6.0.1, whole-machine RTX 4090, driver 595.71.05, a
  compute-only host that needed the exact-version library side-load). These
  establish video delivery, not sustained performance. AWS live video remains
  unverified.
- Isaac is launched with `quitOnSessionEnded=false`, so closing or reloading
  the viewer leaves the simulation running. The upstream streaming app defaults
  to quitting when its viewer session ends.

NVIDIA recommends the [WebRTC browser viewer for cloud deployments](https://docs.isaacsim.omniverse.nvidia.com/6.0.1/installation/manual_livestream_clients.html).
Its standard Docker Compose deployment uses host networking. This repo uses
the same [NVIDIA WebRTC SDK](https://github.com/isaac-sim/IsaacSim/blob/main/tools/docker/web-viewer/Dockerfile)
in a container-served viewer to accommodate
[Vast's assigned port mappings](https://docs.vast.ai/guides/instances/docker-environment).
The SDK is a single self-contained ES module, so the viewer needs no bundler:
`isaac_cloud.py` downloads the pinned package from NVIDIA's npm registry onto
the box at setup and verifies its SHA-256. It is distributed under NVIDIA's
license and is never vendored here.

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
NVIDIA's encoder library assumes `/dev/nvidiaN` is GPU index N, so a container
whose single GPU is really the host's slot 3 opens the wrong device and NVENC
fails (`OpenEncodeSessionEx failed: unsupported device`). This is an NVIDIA
driver limitation
([k8s-device-plugin#1282](https://github.com/NVIDIA/k8s-device-plugin/issues/1282)),
not a provider quirk, and there is no container-side workaround; the repo's
probe reproduced it (`docs/VAST_EXPERIMENT_RESULTS.md`). Consequences:

- The default Vast query rents **whole machines** (`gpu_frac=1`), which
  guarantees GPU 0. Set `[vast].whole_machine = false` to allow cheaper
  fractional hosts — agent control and the noVNC GUI still work there, but
  RTSP/WebRTC video will fail if you draw the wrong GPU slot. `--gui webrtc`
  therefore searches whole machines regardless of that setting, and every
  WebRTC setup checks `nvidia-smi`'s minor number at boot.
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
- `[gui].mode` — `none` (default), `vnc` (noVNC GUI stack; see
  [The GUI stack](#the-gui-stack)), or `webrtc` (experimental native streaming
  on Vast and AWS); `--gui` overrides per launch. `[gui].resolution` applies to
  `vnc`.
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
migration — including why TCP-tunneled WebRTC video was abandoned in favor
of noVNC, and the NVENC device-index discovery — live in
`docs/VAST_EXPERIMENT_RESULTS.md` and `docs/GUI_TUNNEL_EXPERIMENT_PLAN.md`.
