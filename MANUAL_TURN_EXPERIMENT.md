# Manual TURN Experiment

## Goal

Test whether a TURN relay fixes the Isaac browser streaming path when direct private-network approaches did not.

This experiment keeps the current VM and current working public viewer path, then adds TURN on the same VM as the fastest proof-of-concept.

## Why Same VM First

- the VM is already provisioned and known-good on the baseline public path
- we can avoid another 10-15 minute provisioning cycle
- we can validate TURN behavior quickly before designing a cleaner deployment
- if TURN fails here, we learn fast
- if TURN works here, we can later move it to a separate hardened service

## Scope

This is a proof-of-concept only.

It is not the final production design.

The experiment is limited to:

- installing and running `coturn` on the current VM
- exposing a minimal TURN endpoint and relay port range
- patching the browser viewer to force TURN relay
- observing whether the stream works when media is relayed

## Baseline Assumption

Before starting:

- the VM should already work on the public baseline path
- the browser should already be able to connect to the current public viewer

If the baseline public path is broken, stop and fix that first.

## First Experiment Shape

- keep Isaac on the current working public configuration
- run `coturn` on the same VM
- publish:
  - `3478/tcp`
  - `3478/udp`
  - a small relay UDP range, for example `49152-49232/udp`
- use static test credentials
- configure the web viewer to force TURN relay

## Inputs

Choose test values for:

- TURN realm
- TURN username
- TURN password
- relay UDP range

Example:

- realm: `isaac-turn.local`
- username: `turntest`
- password: `turntest-secret`
- relay range: `49152-49232`

## Step 1: Confirm Baseline Still Works

Before changing anything:

- open the current viewer URL on the VM public IP
- verify the current baseline streaming path still works

This protects the experiment from false attribution.

## Step 2: Install And Run `coturn`

Options:

- install the package directly on the VM, or
- run `coturn/coturn` in Docker

For this experiment, Docker is acceptable if it is faster.

Required listeners:

- `3478/tcp`
- `3478/udp`

Required relay range:

- `49152-49232/udp`

Required config:

- `realm`
- `listening-ip=<public-ip>`
- `external-ip=<public-ip>`
- static username/password
- relay port range

## Step 3: Confirm TURN Is Reachable

Before touching the viewer:

- confirm `coturn` is listening on the VM
- confirm the public TURN ports are reachable
- check logs for successful startup

Useful VM-side checks:

```bash
ss -ltnup
docker logs <turn-container>
```

## Step 4: Configure Isaac With TURN Credentials

Before patching the viewer, use Isaac's existing livestream service endpoint on `8011`:

- `POST /v1/streaming/creds`

This endpoint maps to `set_stun_credentials(...)` in the livestream stack and is the cleanest first attempt.

Post:

- TURN host set to the VM public IP
- TURN port `3478`
- username/password set to the static test credentials

Example payload:

```json
{
  "stunIp": "<public-ip>",
  "stunPort": 3478,
  "username": "turntest",
  "password": "turntest-secret"
}
```

If this is not enough to make the browser use relay candidates, then fall back to a viewer-side override patch.

## Step 5: Retry The Browser Stream

Using the normal public viewer path:

- load the viewer page
- start the stream
- watch for whether the session now establishes successfully

## Step 6: Tighten The Proof

If the stream still works after TURN credentials are applied, make the proof stronger:

- temporarily block direct public UDP to Isaac media port `47998`
- retry the stream
- if the session still works, media must be using the relay path

Only do this after baseline-with-TURN is tested once.

## Step 7: Capture Evidence

On the VM:

- `coturn` logs
- `ss -ltnup`
- TURN relay port counters
- Isaac logs

In the browser:

- `webrtc-internals`
- selected ICE candidate pair
- confirmation that the candidate type is `relay`

## Success Criteria

The experiment is successful only if:

- the stream establishes successfully
- browser diagnostics show a `relay` candidate pair in use
- TURN logs show allocations or relay activity
- and ideally the stream still works with direct public UDP to `47998` blocked

## Failure Criteria

Treat the experiment as failed if:

- the session still never establishes
- no TURN allocations appear
- the browser continues trying direct candidates only
- forcing relay mode still does not produce a usable stream

## If It Works

If this same-VM TURN test works:

- write up the findings
- plan a cleaner deployment
- likely move TURN to a separate hardened service or VM
- automate config and viewer overrides in the repo

## If It Fails

If TURN fails here:

- preserve logs and browser evidence
- stop assuming TURN is the answer without a more specific hypothesis
- reassess whether the problem is in the browser viewer integration itself
