# Vast.ai Experiment Plan

## Goal

Determine whether Isaac Sim 6.0.1 can run usefully on Vast.ai despite NAT-only
networking, using the all-TCP paths added in Isaac 6.0 and the built-in agent
control socket, with the browser GUI as a stretch goal.

TensorDock's marketplace has been empty since roughly mid-2026, so this is also
the candidate replacement provider for the repo's launch tooling.

## Background

Prior findings that shape this design (see `WIREGUARD_EXPERIMENT_RESULTS.md`
and `EC2_TURN_EXPERIMENT_RESULTS.md`):

- Isaac's WebRTC media path (UDP 47998) does not establish through tunnels,
  even when the tunnel itself works. Signaling gets through; media never
  starts.
- The only working browser stream came from split-host TURN: coturn on a
  separate public EC2 host, viewer forced to `iceTransportPolicy='relay'`,
  and TURN credentials injected into Isaac via
  `POST 127.0.0.1:8011/v1/streaming/creds`.
- Vast.ai instances are NAT'd containers with randomized external port
  mappings, so direct-UDP streaming is off the table from the start.

Isaac 6.0 additions that make this worth trying anyway:

- the container is the deployable unit (`runheadless.sh`, env-var port config),
  which matches Vast's container-only model
- built-in agent control socket: `--enable isaacsim.code_editor.python_server`,
  TCP `127.0.0.1:8226`, SSH-tunnelable
- RTSP camera streaming (`isaacsim.streaming.rtsp`): pure TCP, viewable in
  VLC/ffplay through an SSH tunnel

## Phases

### Phase 1 - boot and GPU sanity

- rent a verified 1x RTX 4090 offer with `driver_version >= 580.95.05`
- run `nvcr.io/nvidia/isaac-sim:6.0.1` directly as the instance image with
  NGC registry login
- verify the container starts, `nvidia-smi` sees the GPU, and the Isaac kit
  process reaches "app ready" in headless streaming mode

### Phase 2 - zero-ingress agent control (primary goal)

- launch with `--enable isaacsim.code_editor.python_server`
- SSH tunnel `8226` and send Python to the socket (stage query, create a prim,
  step simulation)
- success criterion: remote code execution against the live sim with no
  public ports exposed at all

### Phase 3 - RTSP camera view (secondary goal)

- enable `isaacsim.streaming.rtsp`, attach a writer to a camera render product
- SSH tunnel the RTSP port and open the feed in VLC/ffplay locally
- success criterion: watchable live camera feed over TCP only

### Phase 4 - browser GUI probe (stretch)

- check whether the streaming control plane survived into 6.0.1:
  `curl 127.0.0.1:8011/v1/streaming/ready` inside the instance
- if yes, the proven EC2 TURN recipe is expected to carry over
  (media is outbound from the sim host, so NAT is not fatal); actually
  standing up TURN is a follow-up experiment, not part of this one

## Prerequisites

- Vast.ai account with billing at https://cloud.vast.ai
- API key stored locally: `vastai set api-key <KEY>`
- NGC API key (already in `config.toml`) for the `nvcr.io` image pull

## Launch Commands

Search (46 matching offers as of 2026-08-10, ~$0.31-0.34/hr):

```bash
vastai search offers 'gpu_name in ["RTX_4090","L40S"] driver_version >= 580.95.05 verified=true rentable=true num_gpus=1 disk_space >= 80 inet_down >= 500 direct_port_count >= 3' -o 'dph'
```

Create (substitute OFFER_ID and the NGC key):

```bash
vastai create instance OFFER_ID \
  --image nvcr.io/nvidia/isaac-sim:6.0.1 \
  --login '-u $oauthtoken -p <NGC_API_KEY> nvcr.io' \
  --disk 100 \
  --env '-e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y' \
  --onstart-cmd '/isaac-sim/runheadless.sh -v --enable isaacsim.code_editor.python_server' \
  --ssh --direct
```

Notes:

- disk 100 GB: the image is ~15 GB and shader cache needs headroom
- first boot includes RTX shader compilation; expect several minutes before
  the app reports ready, same as on TensorDock
- if `runheadless.sh` does not forward the `--enable` flag to kit, fall back
  to `--onstart-cmd 'bash -c "/isaac-sim/isaac-sim.streaming.sh --allow-root -v --enable isaacsim.code_editor.python_server"'`

## Verification

```bash
vastai show instances
vastai ssh-url INSTANCE_ID          # gives ssh -p PORT root@HOST
ssh -p PORT root@HOST nvidia-smi
ssh -p PORT root@HOST 'tail -n 50 /isaac-sim/kit/logs/Kit/*/kit_*.log 2>/dev/null || ls /root/.nvidia-omniverse/logs'
ssh -p PORT root@HOST 'ss -tlnp | grep 8226'
# then from a second terminal with the tunnel open:
ssh -p PORT -N -L 8226:127.0.0.1:8226 root@HOST
# send python to the socket (Phase 2)
```

## Teardown

```bash
vastai destroy instance INSTANCE_ID
```

## Success / Failure Interpretation

- Phase 1 fails on driver/GPU: tighten the search filter, try another host;
  Vast hosts are heterogeneous and one bad host proves nothing
- Phase 2 works: Vast is viable for the agent-driven workflow today, and the
  repo should grow a Vast provider mode
- Phase 4 shows `8011` alive: browser GUI via split-host TURN is expected to
  be portable; schedule the TURN follow-up
- Phase 4 shows `8011` gone: browser GUI on Vast needs new research; agent
  control and RTSP remain unaffected
