# EC2 TURN Experiment Results

## Result

The split-host TURN experiment succeeded.

Browser streaming worked when:

- Isaac and the web viewer stayed on the TensorDock GPU VM
- TURN moved to a separate EC2 instance
- the viewer forced `iceTransportPolicy = 'relay'`
- Isaac received matching TURN credentials through `POST /v1/streaming/creds`

This is the first experiment in this repo that produced a working TURN-assisted browser stream.

## Conclusion

The important conclusion is:

- TURN is a viable path for Isaac browser streaming
- the earlier same-VM TURN failure should not be treated as proof that TURN does not work
- same-host TURN was likely failing because local networking and candidate generation polluted the ICE path

The strongest evidence for that interpretation is:

- same-VM TURN reached the point of real relay candidate creation
- but ICE never selected a usable pair
- browser stats showed bad remote candidates such as `172.17.0.1:47998`
- moving TURN to a separate public EC2 host removed that ambiguity
- after that change, the stream worked

## Infrastructure Used

### Isaac / Viewer Host

- Provider: TensorDock
- Instance ID: `4e72fd72-0400-4bcd-9c47-b6877007b08a`
- Public IP: `38.224.253.238`

### TURN Host

- Provider: AWS EC2
- Region: `us-west-2`
- Instance ID: `i-08b6e4618c7270b56`
- Instance type: `t3.micro`
- Ubuntu AMI: `ami-09222573bc99a7788`
- Public IP: `34.221.249.165`
- Private IP: `172.31.9.46`

## TURN Host Configuration

The EC2 instance was configured with `coturn`.

Relevant settings:

- `listening-port=3478`
- `listening-ip=0.0.0.0`
- `external-ip=34.221.249.165/172.31.9.46`
- `realm=isaac-turn.local`
- `lt-cred-mech`
- `user=turntest2026:turntest-credential-2026-valid-abcdef`
- `min-port=49152`
- `max-port=49232`
- `no-tls`
- `no-dtls`

The service was installed directly on Ubuntu and started under `systemd` as `coturn.service`.

## AWS Security Group

The EC2 security group allowed only the current client IP `97.113.140.128/32` for:

- `22/tcp`
- `3478/tcp`
- `3478/udp`
- `49152-49232/udp`

This was intentionally narrow for the experiment.

## Isaac VM Changes

The live viewer patch on the TensorDock VM was updated to force relay through the EC2 TURN host:

- `turn:34.221.249.165:3478?transport=udp`
- `turn:34.221.249.165:3478?transport=tcp`
- `iceTransportPolicy: 'relay'`

The viewer was then rebuilt and `isaac-cloud-viewer.service` was restarted.

Isaac was updated via:

- `POST http://127.0.0.1:8011/v1/streaming/creds`

with:

- `stunIp=34.221.249.165`
- `stunPort=3478`
- `username=turntest2026`
- `password=turntest-credential-2026-valid-abcdef`

Isaac accepted the update successfully.

## Why This Was Necessary

Earlier experiments established:

- WireGuard worked as a tunnel but did not produce a working browser stream
- removing `open_endpoint.toml` did not fix that
- same-VM TURN initially failed because the TURN credentials were not ICE-valid
- once the credentials were fixed, same-VM TURN still failed because ICE never completed

So this EC2 test was designed specifically to isolate TURN from the Isaac host network topology.

## What Changed Relative to Same-VM TURN

The key difference was placement:

- before: TURN and Isaac shared the same VM
- now: TURN is on its own public host

That removed Docker and local host-network ambiguity from the relay path.

## Operational Takeaway

For secure hosting, a separate TURN service should be treated as the promising path forward.

The working direction is:

- keep Isaac on the GPU host
- keep the viewer on the GPU host or another web host
- place TURN on a separate, public, tightly scoped service
- restrict TURN ingress to expected client IPs when possible
- do not rely on same-host TURN behavior as representative

## Recommended Next Steps

1. Capture a reproducible runbook for the split-host TURN setup.
2. Decide whether TURN should remain manual for now or be automated in repo tooling.
3. If automating:
   - add config for external TURN host/IP and credentials
   - add a controlled viewer-side TURN/relay mode
   - add a documented security posture for TURN port exposure
4. Tighten the remaining baseline security issues in `isaac_cloud.py`, especially:
   - secret handling in bootstrap
   - plain HTTP viewer exposure
   - firewall automation
   - file permissions
