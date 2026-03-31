# Persistence Plan

## Goal

Make it practical to destroy a GPU VM without losing the user-authored Isaac work that matters.

For this repo, persistence means:

- durable project files survive VM destruction
- restore onto a fresh VM is predictable
- disposable caches remain disposable
- destroy does not silently discard unsaved work

## Current Design

Use S3 as the persistence backend, but keep all AWS access local to the machine running this CLI.

That means:

- the VM stores the durable workspace at `/home/<ssh-user>/isaac-cloud/project`
- the VM never receives AWS credentials
- the local CLI talks to AWS with the local `aws` command
- the local CLI moves files to and from the VM over SSH
- the VM keeps only a persistence manifest so `status --verbose` can report the last pull/push result

This fits SSO/SAML-based AWS setups well because the tool can rely on the user’s normal local AWS login flow.

## Scope

Implemented:

- `[persistence]` policy config
- `[aws]` backend config for S3 URI and region
- durable workspace layout on the VM
- `sync pull`
- `sync push`
- `auto_pull_on_launch`
- `auto_push_on_destroy`
- destroy aborts if save fails
- remote manifest reporting through `status --verbose`

Not in scope:

- background sync
- merge/conflict resolution
- multi-project workspaces on one VM
- filesystem-style object storage mounts

## Configuration Model

```toml
[persistence]
enabled = true
provider = "s3"
auto_pull_on_launch = true
auto_push_on_destroy = true

[aws]
s3_uri = "s3://your-bucket/your/path/"
region = "us-west-2"
```

Environment overrides:

- `ISAAC_CLOUD_S3_URI`
- `AWS_REGION`

Local requirements when persistence is enabled with `provider = "s3"`:

- `aws` CLI installed locally
- local AWS auth already active
- `ssh` installed locally

The CLI verifies this before launch, sync, or destroy-save by running:

- `aws sts get-caller-identity`
- an S3 access check against the configured URI

## Filesystem Layout

```text
/opt/isaac-cloud/
  bootstrap/
  systemd/

/var/lib/isaac-cloud/
  state/
    persistence-manifest.json

/home/<ssh-user>/isaac-cloud/
  project/
```

Only `/home/<ssh-user>/isaac-cloud/project` is persisted.

Do not persist:

- Isaac caches
- compute caches
- container layers
- viewer build artifacts
- cloned MCP repo contents

## Sync Model

### Pull

`sync pull` does this:

1. verify local AWS auth
2. download the configured S3 prefix to a local staging directory with `aws s3 sync`
3. replace the remote durable workspace over SSH
4. record pull status in the VM manifest

### Push

`sync push` does this:

1. copy the remote durable workspace to a local staging directory over SSH
2. upload that staging directory to S3 with `aws s3 sync --delete`
3. record push status in the VM manifest

## Lifecycle Behavior

### Launch

If `auto_pull_on_launch = true`:

1. launch and bootstrap the VM normally
2. wait for SSH reachability
3. pull the configured S3 workspace locally
4. copy it onto the VM over SSH
5. restart Isaac

### Stop

`stop` does not sync automatically.

### Destroy

If `auto_push_on_destroy = true`:

1. copy the remote durable workspace down over SSH
2. push it to S3 locally
3. if push fails, abort destroy
4. only destroy the VM after save succeeds

`destroy --all` stays blocked while persistence is enabled so save checks cannot be skipped silently.

## Notes

- Viewer and Isaac streaming still use the VM public IP.
- MCP remains on the SSH-forwarded path.
- Persistence is independent of viewer and MCP transport.
- The local machine is now the trust boundary for AWS auth; the VM is not.
