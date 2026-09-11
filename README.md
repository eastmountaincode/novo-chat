# Novo Chat

Novo Chat is an optional companion service for a single Novo installation. It
keeps Novo responsible for identity and notebook authorization while sending
approved, normalized notebook content to a separately hosted compute worker for
indexing, retrieval, and local-model generation.

The distributed service has two always-on roles:

- **Gateway:** runs beside Novo, serves `/chat/`, reuses the browser's existing
  Novo session, asks Novo for current notebook permissions, and owns public job
  handles.
- **Worker:** runs on a private compute host, stores derived documents and
  indexes, executes queries and rebuilds, and controls only explicitly approved
  model containers through a narrow Unix socket.

The gateway and worker communicate through a signed internal protocol over a
loopback-only reverse SSH forward. The compute host initiates that tunnel. The
gateway host therefore stores no password or private key capable of logging in
to the compute host.

This repository is separate from Novo on purpose. A Novo installation can omit
Chat entirely by not installing the companion integration and by leaving its
Chat route/navigation flag disabled. There is no multi-tenant abstraction in
this version: each lab runs its own Novo and, if desired, its own Novo Chat.

## Request path

```text
browser -> existing HTTPS Novo origin /chat/
        -> Apache -> gateway on 127.0.0.1
        -> Novo integration API on 127.0.0.1 (session and permissions)
        -> reverse-forward listener on 127.0.0.1
        -> worker on compute-host 127.0.0.1
        -> approved model endpoint / model-controller Unix socket
```

The Novo session cookie is sent only from the gateway to Novo's loopback API.
It is never forwarded to the worker. The worker receives explicit notebook and
revision scopes plus normalized content; it receives no Novo password, cookie,
database, session secret, or general-purpose command.

## Repository map

```text
novo_chat/gateway/       same-origin web gateway and Novo/worker clients
novo_chat/protocol.py    signed gateway-worker protocol
novo_chat/worker.py      private worker API
novo_chat/jobs.py        durable jobs and active-index catalog
novo_chat/model_controller.py
                         unprivileged Unix-socket client
novo_chat/model_backend.py
                         loopback-only embedding/generation configuration
deploy/gateway-host/         generic gateway, Apache, and SSH templates
deploy/compute-host/         generic worker, tunnel, and controller templates
tests/                   protocol, authorization, gateway, worker, and runtime tests
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for trust boundaries and
[deploy/README.md](deploy/README.md) for the staged installation and rollback
checklist.

## Deployment contract

The supplied packaging intentionally fails closed:

- `novo-chat-runtime` accepts only the explicit `staging` and `production`
  environments.
- Both roles must bind to literal `127.0.0.1` and to their audited role port.
- Secrets are read from root-managed files, never literal environment values or
  command-line arguments.
- Gateway URLs for Novo and the worker must use explicit loopback hosts.
- Worker embedding and generation endpoints come from one read-only backend
  document and must be explicit loopback origins; its ordered model IDs must
  exactly match the worker and controller allowlists.
- The worker container runs as UID/GID `10001`, has no Docker socket or Docker
  group, and receives only the numeric supplemental group for the controller
  socket.
- Staging and production have different ports, state directories, signing keys,
  tunnel keys, and model allowlists.
- All compute-host controllers share one root-managed container inventory and
  one host lock, so staging and production model transitions are GPU-exclusive.
- Container templates require an immutable image digest placeholder; they do not
  pull `latest` at service start.

Audited defaults:

| Role | Staging | Production |
| --- | ---: | ---: |
| Gateway listener | `127.0.0.1:3181` | `127.0.0.1:3180` |
| Private browser mux | `127.0.0.1:3182` | existing HTTPS virtual host |
| Compute worker | `127.0.0.1:8096` | `127.0.0.1:8095` |
| Reverse-forward listener | `127.0.0.1:8196` | `127.0.0.1:8195` |

These ports and base paths are an opinionated two-environment deployment
profile, not lab identity or tenant logic. Hostnames, model IDs, keys, and
allowed networks remain rendered configuration. A deployment that needs a
different port matrix should change the runtime contract, templates, and tests
together so staging/production isolation remains reviewable.

The `/chat/` production route uses the Novo site's existing DNS name and TLS
certificate. It does not require a new DNS record or a certificate for the
compute host.

Staging can instead use a single SSH-forwarded loopback origin. A loopback-only
Caddy mux on gateway-host port `3182` presents Novo staging and
`/chat-staging/` on the same browser origin. Plain HTTP public origins are
accepted only for `staging` with the literal host `127.0.0.1` or `::1`;
production remains HTTPS-only. See the deployment checklist for the exact
one-port tunnel.

## Local development

Install the locked development environment and run the tests:

```bash
uv sync --locked --extra dev
uv run --frozen python -m pytest
```

The deployment entrypoint can validate a populated environment without opening
a listener:

```bash
novo-chat-runtime check gateway
novo-chat-runtime check worker
```

`uv.lock` is the reproducible dependency source used by the container build.
Use the role-specific templates under `deploy/` as the complete list of
required settings. Do not put secret bytes into an `.env` file. Development
tests create temporary secret files with production-equivalent permissions.

## Authorization behavior

Novo remains authoritative on every browser operation. The gateway resolves the
live session and notebook grants before listing notebooks, exporting content,
submitting a query or rebuild, and releasing a result. Model start/stop and
index rebuild are available to any authenticated user, while a user
can only name notebooks currently returned in that user's Novo authorization
context.

Workers key active indexes by environment, notebook ID, content revision, and
index schema version. A database refresh is not part of this design: Novo
exports the normalized content needed for an exact authorized revision, and
only derived worker indexes need to be built or rebuilt.

## Derived-data retention

The worker never receives a Novo database snapshot. It stores only normalized
exports and indexes for exact notebook revisions. A successful export finalize
deletes its duplicate ingest batches immediately. Terminal jobs, gateway job
mappings, active index pointers, finalized documents, and index artifacts use
the same seven-day maximum active-retention window. Idle worker maintenance
removes unreferenced storage after a one-hour grace period; separate 50 GiB
document and index quotas stop short-term accumulation. Re-exporting and
rebuilding a missing, stale, or expired index requires an explicit **Rebuild
index** request. Asking a question only checks index readiness; it never exports
notebooks or rebuilds indexes. If any selected notebook is not ready, the gateway
returns `409 INDEX_REBUILD_REQUIRED` with instructions to rebuild or select an
indexed notebook. It does not silently search only the ready subset.

Novo authorization is still checked before every operation and again before a
result is released. Retention is therefore an at-rest privacy and capacity
bound, never an authorization mechanism.

Snapshot-password authentication and direct Novo SQLite access are
intentionally absent from this repository and from the packaged runtime.

## Current state

This checkout contains implementation and generic deployment templates. It does
not imply that a service, tunnel, Apache route, account, key, or model container
has been installed on any host. Deployment starts in the isolated staging
ports, is verified end to end, and only then promotes the exact tested image
digest to production.
