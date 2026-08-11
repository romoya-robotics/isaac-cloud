# GUI-over-SSH Experiment Results

Date: 2026-08-10
Plan: `GUI_TUNNEL_EXPERIMENT_PLAN.md`

## Outcome

**The winning approach is noVNC, not tunneled WebRTC.** The full native Isaac
Sim GUI, served over a single TCP port through an SSH tunnel, confirmed
working interactively by the user (Taiwan RTX 4090 host, viewed from Kyoto,
up to 2560x1440):

```
Isaac GUI (isaac-sim.sh) -> Xvfb -> x11vnc -> websockify/noVNC -> :6080 -> ssh -L
```

This matches independent community findings (NVIDIA forum thread on TCP-only
clouds; a Vast.ai user in isaac-sim/IsaacSim#597 confirming "noVNC worked")
and NVIDIA's own IsaacAutomator, which ships noVNC access.

Why it wins:

- pure TCP, one port: tunnels over SSH with zero NAT/ICE machinery
- no NVENC dependency: works on fractional hosts that lose the GPU-0 lottery
- no public ports; SSH key is the only auth
- survives everything that killed the WebRTC attempts

## The WebRTC-over-bridge attempts (5 tries, 3 hosts)

The alternative — tunneling Isaac's WebRTC livestream by encapsulating the
UDP media port in TCP with a socat pair — was *architecturally* validated but
never delivered video:

- Austria host: signaling, ICE (through the bridge!), DTLS, and data channels
  all connected; video suffered 98.5% packet loss (903 ms STUN RTT, 8 Mbps
  stream vs a lossy long-RTT path; kernel drops at the bridge's UDP socket
  when the TCP leg stalls).
- Server floors video at 4000 kbps minimum even at 720p30 (nvst
  `vqos.bw.minimumBitrateKbps`), so bitrate cannot be tuned below what a bad
  path can carry.
- The bridge itself proved operationally fragile: socat 1.8 requires explicit
  `UDP4-`/`TCP4-` address families (silently dies otherwise), duplicate
  listeners black-hole datagrams, and long-lived SSH tunnels wedge without
  keepalives + an external restart loop.

Verdict: possible in principle on a low-RTT path, but not worth the moving
parts when noVNC exists. Not productized.

## Incidental findings

- Vast `--env` values do not propagate into onstart/SSH shells; export
  ACCEPT_EULA/PRIVACY_CONSENT/OMNI_KIT_ALLOW_ROOT in-script.
- Some Vast machines never inject attached SSH keys (two hosts observed);
  account-level keys (`vastai create ssh-key`) applied at creation are
  reliable.
- `runheadless.sh` forwards extra kit args (`--enable <ext>` works).
- The 6.0.1 container still contains and uses `open_endpoint.toml`.
- Kit services port 8011 in 6.0.1 serves only CAD/asset-conversion routes;
  the 5.1-era `/v1/streaming/creds` TURN-injection endpoint is gone.

## Productization

`isaac_cloud.py launch --gui` now provisions this stack automatically and
prints the tunnel command; open `http://localhost:6080/vnc.html`. Resolution
is `[gui].resolution` in config.toml.
