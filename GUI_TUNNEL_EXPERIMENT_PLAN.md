# GUI-over-SSH Experiment Plan

## Goal

Verify the full interactive Isaac Sim GUI is viewable from the local machine
with **all traffic over SSH forwarding** — no public ports, no TURN. Two
client candidates:

1. Isaac 6.0's web-based streaming client (in the container)
2. the native Isaac Sim WebRTC Streaming Client desktop app

## Why this should work in 6.0 (and didn't in 5.1)

Isaac 6.0's container defaults to `ISAACSIM_HOST=127.0.0.1` — NVIDIA's
supported baseline is browser and sim on the same machine over loopback.
Tunneling recreates exactly that topology: both ends see matching
`127.0.0.1` ICE candidates. The 5.1 WireGuard failure was mismatched
advertised candidates (`172.17.0.1`), which this sidesteps.

## The UDP wrinkle

WebRTC media is UDP on `ISAACSIM_STREAM_PORT` (47998). SSH forwards only
TCP, so the media path needs a socat bridge:

```
browser --UDP 47998--> local socat --TCP 47999 via ssh -L--> remote socat --UDP--> Isaac 47998
```

TCP-encapsulated media adds jitter but is fine over a good link. This is
the unvalidated piece.

## Steps

1. Rent a whole-machine (gpu_frac=1) RTX 4090, driver >= 580 (NVENC rule
   from `VAST_EXPERIMENT_RESULTS.md`).
2. Launch `runheadless.sh` with `ISAACSIM_HOST=127.0.0.1` explicit.
3. Enumerate listening TCP ports after "Streaming App is loaded" to find the
   web client's HTTP port (expected: signaling 49100 + a web UI port).
4. Local setup:
   - `ssh -L` for the web UI port and 49100
   - socat UDP bridge for 47998 (remote: TCP-LISTEN 47999 -> UDP 47998;
     local: UDP-LISTEN 47998 -> TCP 47999 through a third `ssh -L`)
5. Human check: open the tunneled web client URL in a browser and confirm
   live interactive viewport.
6. Optional second check: point the native Isaac Sim WebRTC Streaming
   Client at 127.0.0.1. (WSL2 note: Windows->WSL2 localhost forwarding is
   TCP-only historically; if UDP doesn't cross the WSL boundary, run the
   browser inside WSL via WSLg instead, or the socat UDP leg on Windows.)

## Success criteria

- Web client loads over the tunnel and shows a live, mouse-interactive
  viewport, OR the native client connects and streams.
- `ss`/log evidence that media actually flows through the bridge.

## Failure interpretation

- Signaling connects but no video: media not crossing the UDP bridge —
  capture browser webrtc-internals and the socat byte counters to see
  which leg dropped.
- If only TCP-media clients work someday, revisit TURN-with-auth as the
  public-sharing option; SSH-only remains the default posture either way.
