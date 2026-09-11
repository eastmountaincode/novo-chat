# Deployment checklist

These are reviewable templates, not an installer. They intentionally contain
placeholders and make no account, hostname, image, model, Apache-layout, or
OpenSSH-version assumptions. Render them into a separate staging configuration,
review the result, and preserve the previous configuration before changing a
live service.

The role-based directory names and template content use `gateway host` and
`compute host` so another single-lab Novo installation can reuse them without
adopting site-specific names.

## Shipped environment profile

| Resource | Staging | Production |
| --- | --- | --- |
| Gateway listener | `127.0.0.1:3181` | `127.0.0.1:3180` |
| One-origin browser mux | `127.0.0.1:3182` | existing HTTPS virtual host |
| Novo loopback API | `127.0.0.1:3155` | `127.0.0.1:3148` |
| Reverse listener on gateway host | `127.0.0.1:8196` | `127.0.0.1:8195` |
| Worker listener on compute host | `127.0.0.1:8096` | `127.0.0.1:8095` |
| Gateway state | `/var/lib/novo-chat-gateway/staging` | `/var/lib/novo-chat-gateway/production` |
| Worker state | `/var/lib/novo-chat-worker/staging` | `/var/lib/novo-chat-worker/production` |
| Model backend document | `/etc/novo-chat/model-backends-staging.json` | `/etc/novo-chat/model-backends-production.json` |
| Browser route | optional `/chat-staging/` | `/chat/` |

Do not share state databases, indexes, HMAC keys, SSH target keys, tunnel
accounts, model-controller sockets, or model allowlists between the two rows.
The templates retain gateway ownership records, terminal worker jobs, active
index pointers, normalized documents, and index artifacts for seven days
(`604800` seconds). Successful finalization removes duplicate staging batches
immediately; idle maintenance removes unreferenced derived state after a
one-hour grace period (`3600` seconds). The worker also enforces separate 50 GiB
document and index quotas. This keeps gateway mappings and worker results
aligned while bounding retained notebook-derived data. Change these settings
only after reviewing the invariant, available storage, and privacy policy
together.
The legacy monolith also used worker port `8095`; stop it only during an
approved production cutover and never try to bind both services there.

## 1. Build and identify one image

From a clean reviewed checkout, run the test suite, build once, and record an
immutable image digest:

```bash
python -m pytest
git diff --check
git status --short
task_git_sha=$(git rev-parse --verify HEAD)
docker build --pull --tag "novo-chat:${task_git_sha}" .
docker image inspect "novo-chat:${task_git_sha}" --format '{{json .RepoDigests}}'
```

The rendered systemd units must replace `@NOVO_CHAT_IMAGE_DIGEST@` with the
same content-addressed digest in staging and production. Do not put `latest` in
a unit and do not rebuild between promotion steps.

## 2. Create independent file secrets

Environment files contain paths, IDs, and URLs only. Generate at least 32 random
bytes per secret and store them as files. The expected mount names are:

| Gateway file | Matching peer / purpose |
| --- | --- |
| `novo-integration-token` | Same token configured in the local Novo companion API |
| `csrf-key` | Gateway-only CSRF derivation key |
| `gateway-request-key` | Same bytes as worker `gateway-request-key` |
| `worker-response-key` | Same bytes as worker `worker-response-key` |

Use different values for staging and production. Transfer the two protocol keys
through an approved secret channel; do not store them in Git, shell history,
Docker image layers, an environment variable value, or a systemd command line.
The SSH private keys described later stay only on the compute host.

On each host, make the role secret directory `root:10001` mode `0750` and its
files `root:10001` mode `0440`. The container process is UID/GID `10001`; no
human service account needs to read the contents. Before startup, confirm that
the rendered files contain paths rather than secret values:

```bash
if grep -RE '(_SECRET|_KEY)=' /etc/novo-chat/*.env; then
  echo 'literal secret setting found in environment file' >&2
  exit 1
fi
grep -RE '(_SECRET_FILE|_KEY_FILE)=' /etc/novo-chat/*.env
find /etc/novo-chat/secrets -type f -exec stat -c '%a %U:%G %n' {} \;
```

Review the second grep output manually; key IDs are not secrets, while each setting
ending in `_FILE` must be an absolute container path under `/run/secrets`.

## 3. Install the staging gateway container

1. Copy and render `gateway-host/gateway-staging.env.template` as
   `/etc/novo-chat/gateway-staging.env`. For the loopback tunnel described in
   step 7, replace `@STAGING_PUBLIC_ORIGIN@` with
   `http://127.0.0.1:3182`. The value must exactly equal the browser-visible
   origin, including its port. Reject any remaining `@...@` token.
2. Create `/var/lib/novo-chat-gateway/staging` as `10001:10001` mode `0750`.
3. Create `/etc/novo-chat/secrets/gateway-staging` and the four files above with
   the ownership and modes from step 2.
4. Render `gateway-host/novo-chat-gateway@.service.template` into the systemd unit
   directory with the immutable image digest, then run `systemd-analyze verify`
   against the rendered unit.
5. Run the image's configuration check with the same env, state, and secret
   mounts as the unit. It must exit zero without opening port `3181`.
6. Reload systemd, enable `novo-chat-gateway@staging.service`, and verify its
   journal and loopback listener before adding an Apache route.

The configuration-only container check from step 5 is:

```bash
docker run --rm --network host --read-only --cap-drop=ALL \
  --security-opt=no-new-privileges --user=10001:10001 \
  --env-file=/etc/novo-chat/gateway-staging.env \
  --mount=type=bind,src=/var/lib/novo-chat-gateway/staging,dst=/var/lib/novo-chat-gateway/staging \
  --mount=type=bind,src=/etc/novo-chat/secrets/gateway-staging,dst=/run/secrets,readonly \
  @NOVO_CHAT_IMAGE_DIGEST@ check gateway
```

The gateway uses host networking only so that its literal loopback Novo and
reverse-forward URLs refer to the host namespaces. Its application listener is
still required to bind `127.0.0.1`; startup rejects any other address or port.

## 4. Install the restricted model controller

On the compute host:

1. Create a system group named `novo-chat-modelctl` with no users added by
   default. Record its numeric GID with `getent group novo-chat-modelctl`.
2. Install `compute-host/novo-chat.tmpfiles.template` as a reviewed tmpfiles entry and
   run `systemd-tmpfiles --create`. Verify `/run/novo-chat` is
   `root:novo-chat-modelctl` mode `0750`; otherwise the container's supplemental
   group cannot traverse to the `0660` socket.
3. Install `compute-host/modelctl_server.py` under a root-owned release path that is
   not writable by the worker or tunnel account.
4. Render `compute-host/modelctl-exclusive-containers.json.template` once as
   `/etc/novo-chat/modelctl-exclusive-containers.json`, root-owned and not
   group/world-writable. It must list every mutually exclusive model container
   managed on this host, including staging, production, and any retained legacy
   container. Both environment controllers must use this exact file; update it
   before adding a managed container.
5. Render `compute-host/models-staging.json.template` as
   `/etc/novo-chat/models-staging.json`. `@STAGING_APPROVED_MODEL@` must be a
   model ID configured in the model backend document and must map to one
   existing, pre-reviewed container name that also appears in the shared
   inventory. Do not invent a controller-only alias: a model can be started by
   Docker but can never pass backend readiness or generation unless the
   identical ID is in both configurations. The JSON accepts no images,
   commands, arguments, ports, or volumes.
6. Install and review `novo-chat-modelctl@.socket` and the rendered
   `novo-chat-modelctl@.service`. Confirm the controller script and JSON are
   root-owned and not group/world-writable.
7. Run `systemd-analyze verify`, reload systemd, and enable only
   `novo-chat-modelctl@staging.socket`. A status request should socket-activate
   the service; no TCP listener is created.

The controller is the only process that talks to Docker. `PrivateNetwork=true`
and `RestrictAddressFamilies=AF_UNIX` prevent it from opening a network path.
During model startup, controller status may include only the latest validated
checkpoint percentage from the current container run. Raw Docker and vLLM logs
remain inside the privileged boundary. The worker persists that percentage
during its readiness wait so the gateway's existing one-second job polling can
render live progress without a separate streaming channel.
Every controller instance uses `/run/novo-chat/modelctl-gpu.lock` plus the one
shared inventory while performing start and stop transitions. A start reloads
that inventory only after taking the lock. This makes the GPU exclusion
decision atomic across staging and production processes, not just within one
environment's allowlist, and lets a root-managed inventory addition take effect
without leaving older controller processes on a stale snapshot. Do not remove
an active controller's container from the inventory, or start or stop an
inventoried model container through another service or a manual Docker command
while Novo Chat is managing it.

Before promotion, start the staging and production models simultaneously
through their separate controller sockets. Exactly one request must succeed;
the other must return HTTP-equivalent status `409` with `MODEL_CONFLICT`.
Confirm only one inventoried container is running. Keep the shared inventory
installed for as long as either environment controller is enabled.

## 5. Install the staging worker container

1. Render `compute-host/worker-staging.env.template` as
   `/etc/novo-chat/worker-staging.env`. `@STAGING_APPROVED_MODEL@` must exactly
   match the public ID in the controller allowlist.
2. Render `compute-host/model-backends-staging.json.template` as
   `/etc/novo-chat/model-backends-staging.json`, root-owned and not
   group/world-writable. Its embedding and generation origins must be literal
   compute-host loopback HTTP origins. The ordered model keys must exactly equal
   `NOVO_CHAT_APPROVED_MODELS` and the controller allowlist IDs. This document
   contains endpoints and model names, not credentials. The `modelSize`,
   `maxModelLen`, `thinking`, and `totalVramGb` members are optional,
   display-only metadata for the browser. Their template defaults are `null`;
   replace those values only with facts about the reviewed backend, or remove
   the members when a value is unknown. They do not change model execution.
3. Create `/var/lib/novo-chat-worker/staging` as `10001:10001` mode `0750`.
4. Create `/etc/novo-chat/secrets/worker-staging` with the two matching protocol
   key files, owned `root:10001` mode `0440`.
5. Render `compute-host/novo-chat-worker@.service.template`, replacing both the image
   digest and `@MODELCTL_SOCKET_GID@` with the numeric host GID recorded during
   the model-controller installation.
6. Verify, reload, enable, and start `novo-chat-worker@staging.service`.

Before enabling it, run the same dispatcher in validation-only mode:

```bash
docker run --rm --network host --read-only --cap-drop=ALL \
  --security-opt=no-new-privileges --user=10001:10001 \
  --group-add=@MODELCTL_SOCKET_GID@ \
  --env-file=/etc/novo-chat/worker-staging.env \
  --mount=type=bind,src=/var/lib/novo-chat-worker/staging,dst=/var/lib/novo-chat-worker/staging \
  --mount=type=bind,src=/etc/novo-chat/secrets/worker-staging,dst=/run/secrets,readonly \
  --mount=type=bind,src=/etc/novo-chat/model-backends-staging.json,dst=/run/config/model-backends.json,readonly \
  --mount=type=bind,src=/run/novo-chat/staging-modelctl.sock,dst=/run/novo-chat/staging-modelctl.sock \
  @NOVO_CHAT_IMAGE_DIGEST@ check worker
```

The numeric `--group-add` is necessary because the container's UID namespace
does not resolve the host group name. Do not instead add the process to the
Docker group, mount `/var/run/docker.sock`, run `--privileged`, or run the
worker as root.

Verify the privilege boundary:

```bash
docker inspect novo-chat-worker-staging --format '{{.Config.User}} {{.HostConfig.Privileged}} {{json .HostConfig.CapDrop}}'
docker inspect novo-chat-worker-staging --format '{{range .Mounts}}{{println .Source "->" .Destination}}{{end}}'
docker exec novo-chat-worker-staging sh -c 'id; test ! -e /var/run/docker.sock'
docker exec novo-chat-worker-staging python -c 'from novo_chat.model_controller import UnixSocketModelController as C; print(C("/run/novo-chat/staging-modelctl.sock").status("@STAGING_APPROVED_MODEL@", request_id="install-check", actor_user_id="install-check"))'
```

The mounts should be limited to worker state, worker secrets, the read-only
model-backend document, and the one model-controller socket. The final command
is a read-only controller status request; replace its model placeholder before
running it.

## 6. Install the compute-initiated reverse tunnel

Prefer direct SSH when policy and routing allow it. If a jump host is required,
the supplied config uses a distinct jump key and target key. In either case:

1. Create dedicated, noninteractive staging and production target accounts on
   the gateway host. If needed, create a separate jump account.
2. Generate Ed25519 private keys on the compute host. Keep the directory
   `root:novo-chat-tunnel` mode `0750` and each private key root-owned, group
   `novo-chat-tunnel`, mode `0640`. The service can read but cannot replace or
   modify them. Confirm the installed OpenSSH accepts this root-owned mode before
   enabling the unit; otherwise use systemd credentials rather than making the
   key service-user-writable.
3. Install only the public keys on the SSH hosts. Render the relevant lines from
   `gateway-host/authorized_keys.template` and the supported directives from
   `gateway-host/sshd-match.template`.
4. Obtain pinned host keys through a trusted administrative channel and put them
   in `/etc/novo-chat/tunnel/known_hosts`. Do not bootstrap trust with
   `StrictHostKeyChecking=accept-new`.
5. Validate the server configuration with `sshd -t` before reload. Test that a
   shell, PTY, agent forwarding, arbitrary local forwarding, and an unapproved
   remote listen port are denied.
6. Render `compute-host/tunnel-staging.conf.template` as
   `/etc/novo-chat/tunnel/staging.conf`; use `ssh -G novo-chat-gateway-staging`
   to inspect its resolved destination and forwarding policy.
7. Install and verify `compute-host/novo-chat-tunnel@.service.template`, then enable
   `novo-chat-tunnel@staging.service`.

Confirm `127.0.0.1:8196` exists on the gateway host and is not bound to `0.0.0.0`
or `::`. Stop the tunnel and confirm the listener disappears, then restart it.
The gateway should degrade cleanly while the listener is absent.

## 7. Expose staging through one browser origin

The simplest private staging test uses one SSH local forward and the supplied
`gateway-host/Caddyfile-staging-mux.template`. The Caddy listener binds only to
gateway-host loopback port `3182`. It sends `/chat-staging` and its descendants
to the Chat gateway on `3181`, sends every other browser path to Novo staging on
`3155`, and returns `404` for the private integration API before either proxy.
This is important: tunneling `3155` and `3181` to different local ports would
give Novo and Chat different browser origins, so the Novo session cookie and
same-origin request checks would not compose correctly.

Configure the Novo staging instance with `NOVO_CHAT_URL=/chat-staging/` and the
matching integration-token file. If that staging instance runs a production
Next.js build, it must also use its existing staging-only
`ELN_ALLOW_INSECURE_COOKIES=true` option so the browser can return the Novo
session cookie over this HTTP loopback origin. Never enable that option in
production. Restart Novo staging after changing these settings, then sign in
again through the mux origin.

Install the reviewed Caddyfile as a dedicated, always-on Caddy 2.10 or newer
instance using the gateway host's service manager. Version 2.10 is required for
the streaming `request_body` size limit. Do not merge it into a public
listener. Validate it with the installed Caddy version, start it, and confirm
that only IPv4 loopback owns `3182`:

```bash
caddy validate --config /etc/novo-chat/Caddyfile-staging-mux --adapter caddyfile
ss -ltnp | grep ':3182[[:space:]]'
curl -fsS http://127.0.0.1:3182/
test "$(curl -sS -o /dev/null -w '%{http_code}' http://127.0.0.1:3182/api/integrations/v1/context)" = 404
```

From the tester's workstation, forward that one mux port and leave the SSH
process running:

```bash
ssh -N -L 127.0.0.1:3182:127.0.0.1:3182 gateway-host-alias
```

Then open `http://127.0.0.1:3182/`, sign in to Novo staging normally, and follow
the Chat link at `http://127.0.0.1:3182/chat-staging/`. Use exactly
`127.0.0.1`, not `localhost`, because the staging HTTP exception is restricted
to literal loopback origins. If local port `3182` is unavailable, choose one
other local port, set `NOVO_CHAT_PUBLIC_ORIGIN` to that exact loopback origin,
and forward it to gateway-host port `3182`; restart the gateway after changing
the origin.

Plain HTTP is accepted only for this staging loopback workflow. Production
continues to require a bare HTTPS origin.

### Optional staging Apache route

Back up the active virtual-host configuration first. Confirm that `mod_proxy`,
`mod_proxy_http`, `mod_headers`, and `mod_alias` are enabled. Inside the existing
TLS virtual host, before its catch-all proxy:

1. Install `gateway-host/apache-novo-chat-common.conf.template` once. It makes the
   Novo companion integration path non-public even if a later catch-all would
   otherwise proxy it.
2. Optionally render `apache-novo-chat-staging.conf.template`, replacing
   `@STAGING_ALLOWED_CIDR@`, or leave staging unadvertised and test gateway port
   `3181` through the one-origin tunnel above.
3. Run the platform's Apache configuration test and inspect the full rendered
   virtual host before a graceful reload.

Verify that the existing Novo root route and login still work, the integration
path is denied from the public origin, the Chat route has no mixed-content or
certificate errors, unauthenticated users are returned to Novo login, and an
authenticated user sees only currently permitted notebooks.

## 8. Staging acceptance checks

Record command output and timestamps for all of the following:

```bash
ss -ltnp | grep -E ':(3181|3182|8196)[[:space:]]'
curl -fsS http://127.0.0.1:3182/chat-staging/healthz
systemctl is-active novo-chat-gateway@staging.service
systemctl is-active novo-chat-worker@staging.service
systemctl is-active novo-chat-tunnel@staging.service
systemctl is-active novo-chat-modelctl@staging.socket
journalctl -u novo-chat-gateway@staging.service -u novo-chat-worker@staging.service --since '-10 min' --no-pager
```

On the compute host, separately confirm `127.0.0.1:8096`. Exercise the browser
flow with at least two Novo users who have different notebook grants. For each
user, verify list, query, result retrieval, index rebuild, model status, model
start, and model stop. Revoke one grant in Novo and verify Chat reflects it
without copying a database. Restart the worker and tunnel and verify durable job
recovery and a clear degraded state. Check that cookies, integration tokens,
HMAC keys, normalized page bodies, and model output are absent from ordinary
logs.

### Rolling application-image upgrades

The first-installation startup order below does not apply to an in-place image
upgrade. New response fields are additive and optional for the new gateway, but
an older gateway can reject fields emitted by a newer worker. For an in-place
application-image upgrade, restart the gateway with the new image before the
worker. Verify gateway health against the existing worker, then install and
restart the worker with the exact same image digest and repeat query acceptance.

For an application-image rollback, roll back the worker before the gateway.
The current gateway accepts the older worker response shape; after worker
rollback is verified, restore the older gateway. Change or restart the tunnel
and controller only when their own configuration changed.

## 9. Promote the exact image to production

Production is a second installation, not a rename of staging:

1. Use the production env, state, secret, model-backend document, controller
   allowlist, SSH account/key, tunnel config, and systemd instances.
2. Use the exact image digest accepted in staging.
3. Confirm compute port `8095` is free. Preserve and stop the legacy prototype
   only after an approved cutover window.
4. For the first installation, bring up controller socket, worker, tunnel, and
   gateway in that order. Use the gateway-first sequence above for later upgrades.
5. Install the production Apache `/chat/` directives, test the complete vhost,
   and gracefully reload.
6. Repeat the acceptance checks on ports `3180`, `8195`, and `8095`, including a
   real rendered browser check through the existing HTTPS origin.

Do not make Novo startup depend on Chat. A gateway, worker, model, or tunnel
failure must leave the ELN usable.

## Rollback

Prepare rollback before the staging change and again before production:

1. Save the previous Apache virtual-host file and note the exact current image
   digest and systemd unit contents.
2. To remove user traffic, restore the previous Apache configuration (or remove
   only the Chat include), run the Apache config test, and gracefully reload.
   Verify Novo root/login before continuing.
3. Stop and disable the gateway instance, then the tunnel and worker instances.
   Stop the model-controller socket only if no retained worker uses it. These
   actions do not stop or delete pre-existing model containers.
4. Retain state directories, logs, rendered configuration, and old key material
   under access-controlled rollback names until the review window closes. Do
   not delete or overwrite them during rollback.
5. For an application-image rollback, render the units with the previously
   recorded immutable digest, run configuration checks, and use the worker-first
   then gateway sequence described above.
6. Revoke the new SSH public keys and protocol/integration tokens only after the
   rollback target is confirmed not to use them.
7. Verify that ports `3180/3181`, `8195/8196`, and `8095/8096` match the intended
   post-rollback state and that no listener became non-loopback.

Rollback is complete only after the existing Novo UI is rendered and its normal
login/notebook path works independently of Chat.
