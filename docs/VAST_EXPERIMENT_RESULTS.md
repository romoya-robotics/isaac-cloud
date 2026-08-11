# Vast.ai Experiment Results

Date: 2026-08-10
Plan: `VAST_EXPERIMENT_PLAN.md`
Total cost: $0.83 across three sessions (initial Isaac run $0.33, NVENC
probes $0.04, whole-machine validation run ~$0.21 + probe overhead)

## FINAL OUTCOME (validation run on whole-machine host)

All goals validated on a `gpu_frac=1` host (machine 24077, France, RTX 4090,
driver 580.173.02, $0.32/hr):

- NVENC gate-check passed (ffmpeg n7.1 `h264_nvenc`, GPU minor 0)
- full NVIDIA library injection out of the box - no side-load needed
- Isaac 6.0.1 loaded in 285 s cold, agent socket AND WebRTC signaling
  (49100) both up with no workarounds beyond env exports + allow-root
- RTSP end-to-end: camera + render product + `RTSPStreamWriter` built
  entirely through the agent socket; server on 8554 served
  h264 1280x720@60; 30 frames decoded cleanly; zero NVENC errors
- RTSP DESCRIBE through a local SSH tunnel answered `200 OK` with an H264
  SDP - a local VLC pointed at the tunneled port would play the feed

Host selection is the entire game: rent whole-machine offers (or NVENC-probe
at boot). Everything below documents how we got here.

## Result Summary

| Phase | Outcome |
| --- | --- |
| 1. Boot + GPU sanity | PASS (with workarounds on the first, fractional host) |
| 2. Zero-ingress agent control | PASS |
| 3. RTSP camera view | PASS on whole-machine host (failed on fractional host: NVENC index mismatch) |
| 4. Browser GUI probe | WebRTC signaling up; 5.1-era `8011` creds routes are gone (TURN recipe needs rework) |

The headline: **the agent-driven workflow works on Vast.ai today.** Remote
Python execution against a live Isaac Sim 6.0.1, through an SSH tunnel, with
zero public ports. Video paths (RTSP, WebRTC media) are blocked on hardware
encoding, which failed on this host.

## Infrastructure Used

- Provider: Vast.ai
- Offer: 45996846, machine 58187, United Kingdom
- Instance (contract): 47323707
- GPU: RTX 4090, host driver 580.173.02
- Image: `nvcr.io/nvidia/isaac-sim:6.0.1` run directly as the instance
  container (no VM, no bootstrap)
- Access: direct SSH `root@195.224.35.250:20120`

## What Worked

### Phase 1: container boots, GPU visible

- NGC-authenticated image pull and instance start took ~4 minutes
- `nvidia-smi` inside the container saw the RTX 4090 immediately
- `runheadless.sh` forwards extra kit args (verified: `--enable
  isaacsim.code_editor.python_server` reached the kit command line), and the
  6.0.1 container still ships and uses `/isaac-sim/config/open_endpoint.toml`

### Phase 2: zero-ingress agent control (the primary goal)

With `ssh -N -L 18226:127.0.0.1:8226`:

- sent raw Python to the socket, received JSON `{"status":"ok","output":...}`
- created prims on the live USD stage, traversed the stage, toggled
  extensions, started the timeline and Replicator orchestrator
- the exec namespace persists across socket connections
- no public port was exposed at any point

### Streaming stack loads (after library workarounds)

- "Isaac Sim Full Streaming App is loaded" in 32 s on a warm shader cache
  (~8 min cold)
- WebRTC signaling bound on `49100/tcp` once NVENC libraries were present
- Replicator frame pipeline confirmed working: a `BasicWriter` canary wrote
  584 PNGs in 15 s from a camera render product

## Required Workarounds (must go in any Vast onstart script)

1. **Env vars do not propagate.** Vast's `--env` values were not present in
   onstart/SSH shells; the onstart command died silently on the EULA check.
   Export explicitly: `ACCEPT_EULA=Y PRIVACY_CONSENT=Y`.
2. **Containers run as root.** Kit refuses root without
   `OMNI_KIT_ALLOW_ROOT=1` (exits with segfault after the error).
3. **The host injected compute-only NVIDIA libraries.** No
   `libGLX_nvidia.so.0`, no Vulkan driver lib, no `libnvidia-encode` —
   despite `NVIDIA_DRIVER_CAPABILITIES=all`. The RTX renderer failed with
   `VkResult: ERROR_INCOMPATIBLE_DRIVER` and kit looped on
   "waiting for viewport handle" forever (while python/extensions kept
   working, masking the failure).

   Fix that worked, generalizable because Ubuntu's archive carries the exact
   driver version (`580.173.02-0ubuntu0.24.04.1` matched the host):

   ```bash
   apt-get download libnvidia-gl-580 libnvidia-encode-580 libnvidia-decode-580
   dpkg -x libnvidia-gl-580_*.deb /opt/nvgl        # dpkg -i fails: files are
   dpkg -x libnvidia-encode-580_*.deb /opt/nvgl    # bind-mounted (EXDEV)
   dpkg -x libnvidia-decode-580_*.deb /opt/nvgl
   cat > /opt/nvgl/icd.json <<'EOF'
   {"file_format_version":"1.0.0","ICD":{"library_path":"/opt/nvgl/usr/lib/x86_64-linux-gnu/libGLX_nvidia.so.0","api_version":"1.3.194"}}
   EOF
   export VK_DRIVER_FILES=/opt/nvgl/icd.json
   export LD_LIBRARY_PATH=/opt/nvgl/usr/lib/x86_64-linux-gnu
   echo /opt/nvgl/usr/lib/x86_64-linux-gnu > /etc/ld.so.conf.d/zz-nvgl.conf && ldconfig
   ```

   The driver version must match the host exactly; derive the package pin
   from `nvidia-smi --query-gpu=driver_version`.

## What Failed

### NVENC hardware encoding (blocks RTSP h264 and likely WebRTC media)

- `NvEncSession: nvEncOpenEncodeSessionEx failed (NVENCSTATUS=2)`
  (NV_ENC_ERR_UNSUPPORTED_DEVICE), repeating for every frame
- the side-loaded `libnvidia-encode.so.1` was on the kit process's
  `LD_LIBRARY_PATH` (verified via `/proc/<pid>/environ`) and registered via
  `ldconfig`, so the library loads; the driver call itself fails
- frames render fine (CUDA, Vulkan, disk writers all work), so this is
  narrowly an encode-session failure
- RTSP raw mode also never opened its port; inconclusive whether it shares
  the NVENC dependency (its server starts lazily on first frame)

**RESOLVED by the follow-up NVENC probe (same day, 2 additional hosts,
$0.04):** the failure is the documented NVENC device-index-mismatch bug
([NVIDIA/k8s-device-plugin#1282](https://github.com/NVIDIA/k8s-device-plugin/issues/1282)):
NVENC requires the rented GPU to be host GPU 0 (`nvidia-smi -q` "Minor
Number" 0). Probe results with ffmpeg 7.1 `h264_nvenc`:

| Host | GPU minor | NVENC |
| --- | --- | --- |
| machine 143223, whole-machine 1x4090 (`gpu_frac=1`) | 0 | **PASS** |
| machine 57509, fractional 1-of-4 4090 | 3 | FAIL: `OpenEncodeSessionEx failed: unsupported device (2)` — identical to the Isaac failure |

No container-side workaround is known (the upstream issue lists none; the
fractional host even exposed `/dev/nvidia0`, so node presence is not the
issue). Two probe notes for the future: both probe hosts injected the FULL
library set (GLX/encode/cuvid) unlike the UK Isaac host, so the library
side-load is only needed on a minority of hosts; and ffmpeg *master* static
builds require NVENC API 13.1 (driver 610+) — use the pinned n7.1 build to
probe driver-580 hosts or you get a false failure.

**Host selection rule: rent `gpu_frac=1` (whole-machine) offers, or probe
NVENC at boot and discard mismatched hosts.** Whole-machine 1x4090s were
plentiful at ~$0.33/hr.

### 5.1-era streaming control plane routes are gone

Port `8011` is alive in 6.0.1 but serves only Kit services core routes
(CAD/asset conversion, health). `/v1/streaming/ready` and
`/v1/streaming/creds` return 404. The EC2 TURN recipe's server-side
credential injection **does not port as-is** — the TURN experiment's method
of steering Isaac's ICE candidates needs rediscovering in 6.0 (or the
browser GUI path needs the integrated web viewer / Docker Compose stack
instead).

## Conclusions

1. Vast.ai works today for the agent-first workflow: rent, side-load driver
   libs, tunnel 8226, drive the sim from Claude Code. This is the cheapest
   and simplest Isaac topology the repo has had (no VM bootstrap at all).
2. Video encoding works on Vast when the rented GPU is host GPU 0 — rent
   whole-machine (`gpu_frac=1`) offers. The original Isaac host failed NVENC
   because its GPU was (very likely) not minor 0. RTSP/WebRTC video should be
   re-validated on a whole-machine host; agent control is unaffected either way.
3. The 6.0 TURN story regressed: the undocumented creds endpoint is gone.
   Browser GUI on Vast should probably wait for either (a) an NVENC-good
   host plus research into 6.0 ICE configuration, or (b) trying NVIDIA's
   official Docker Compose web-viewer stack, which may manage its own
   signaling/media pathing.
4. Automation-worthy sequence for a `vast` provider mode in `isaac_cloud.py`:
   search offers (driver >= 580, verified, direct ports) -> create instance
   from the NGC image -> onstart script with the env exports + driver lib
   side-load + `runheadless.sh --enable isaacsim.code_editor.python_server`
   -> print SSH tunnel command for 8226.
