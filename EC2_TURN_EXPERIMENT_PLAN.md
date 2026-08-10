# EC2 TURN Experiment Plan

## Goal

Prove whether a separate public TURN host fixes Isaac browser streaming where same-VM TURN did not.

This experiment keeps:

- Isaac on the current TensorDock GPU VM
- the browser viewer on the current TensorDock GPU VM

and moves only TURN onto a separate EC2 instance.

## Why EC2 Is Fine

TURN does not need a GPU.

The TURN host only needs:

- a stable public IP
- open TURN listener ports
- open relay UDP ports
- enough CPU and network for relay traffic

So EC2 is a good fit and is cleaner than trying to co-locate TURN with Isaac.

## Why Separate TURN Matters

The same-VM TURN experiment showed:

- TURN credentials could be made ICE-valid
- the browser created real relay candidates
- but ICE still did not complete
- browser stats showed remote candidates like `172.17.0.1:47998`

That is strong evidence that same-host networking is contaminating the candidate path.

A separate TURN host removes that ambiguity.

## Target Architecture

- Browser loads viewer from the Isaac VM public IP
- Viewer forces `iceTransportPolicy = relay`
- Viewer points at the EC2 TURN server
- Isaac receives TURN credentials via `/v1/streaming/creds`
- Isaac continues signaling on `49100/tcp`
- Media should relay through EC2 TURN instead of trying to reach Isaac directly

## Minimal AWS Shape

- Instance type: small general-purpose instance is enough
  - for example `t3.small` or `t3.micro`
- OS: Ubuntu 24.04 LTS or similar
- Public IPv4 address required
- No EBS tuning needed for the proof-of-concept

## Security Group

Open inbound:

- `22/tcp` from your IP only
- `3478/tcp` from the client IPs you want to test from
- `3478/udp` from the client IPs you want to test from
- relay UDP range, for example `49152-49232/udp`, from the client IPs you want to test from

If source-IP restriction is impractical for the first pass, open to `0.0.0.0/0` only temporarily.

## TURN Configuration

Use `coturn`.

Required settings:

- `realm=isaac-turn.local`
- `lt-cred-mech`
- static username/password for the test
- `listening-port=3478`
- `external-ip=<ec2-public-ip>`
- `listening-ip=0.0.0.0`
- `min-port=49152`
- `max-port=49232`

Do not enable TLS for the first pass unless there is a specific need.

## Credential Requirements

The previous failure showed that TURN credentials must also satisfy ICE validity rules.

So use credentials that are safe for both:

- username: short and plain
- password: at least 22 characters

Example:

- username: `turntest2026`
- password: `turntest-credential-2026-valid-abcdef`

## Isaac VM Changes

Keep the current forced-relay viewer patch concept, but point it at the EC2 TURN IP:

- `turn:<ec2-public-ip>:3478?transport=udp`
- `turn:<ec2-public-ip>:3478?transport=tcp`

Keep:

- `iceTransportPolicy: 'relay'`

Post matching credentials into Isaac:

- `POST http://127.0.0.1:8011/v1/streaming/creds`

with:

- `stunIp=<ec2-public-ip>`
- `stunPort=3478`
- `username=<turn-username>`
- `password=<turn-password>`

## Experiment Steps

1. Launch a small EC2 instance with a public IPv4 address.
2. Configure its security group for `3478/tcp`, `3478/udp`, and the relay UDP range.
3. Install or run `coturn`.
4. Verify `coturn` is listening.
5. Repoint the Isaac VM viewer patch to the EC2 TURN IP.
6. Repost TURN credentials into Isaac using the EC2 TURN IP.
7. Hard refresh the viewer and retry the stream.
8. Capture browser dumps and TURN logs.

## Success Criteria

- browser accepts the SDP offer
- relay candidates are created against the EC2 TURN IP
- TURN logs show allocations and relay traffic
- a selected candidate pair appears
- stream media actually flows

## Failure Criteria

- no TURN allocations on EC2
- ICE remains `checking` or `failed`
- no selected candidate pair
- bytes sent/received remain zero

## What This Experiment Should Resolve

If EC2 TURN works:

- same-VM TURN failure was likely due to local candidate/path contamination
- TURN remains a viable security and connectivity path

If EC2 TURN still fails:

- the issue is probably deeper in Isaac’s livestream candidate model or browser integration
- further work should focus on signaling and candidate generation, not just TURN placement
