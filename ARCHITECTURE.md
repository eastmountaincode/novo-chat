# Architecture

Novo Chat is a companion to one Novo instance, not a subsystem that every Novo
installation must run. Novo owns users, sessions, notebook grants, and
normalized notebook exports. Novo Chat owns only the optional conversational
surface and derived compute state.

## Components and trust boundaries

| Component | May receive | Must not receive or do |
| --- | --- | --- |
| Novo | Its own session cookie; companion bearer token on loopback | Run models or depend on Chat for normal ELN use |
| Gateway | Browser cookie; live permitted notebook descriptors/content; file-backed signing keys | Parse Novo SQLite; store user passwords; send a cookie to the worker; control Docker |
| Reverse SSH tunnel | Signed worker HTTP bytes | Offer a shell, arbitrary forwarding, or a non-loopback listener |
| Worker | Signed typed jobs; explicit notebook/revision scopes; normalized documents | Receive Novo credentials; infer grants; accept arbitrary commands; mount Docker control |
| Model backends | Explicit loopback embedding and generation requests for configured public model IDs | Receive Novo identity/session data; use remote or credentialed URLs |
| Model controller | Approved model ID plus `status`, `start`, or `stop` | Accept images, container names, shell strings, network requests, or unapproved models |

The worker loads one root-managed model-backend document. It accepts only plain
HTTP loopback origins and requires its ordered public model IDs to exactly match
the worker allowlist. The same backend object handles embeddings, generation,
and readiness, so a controller-only alias cannot appear healthy.

The only privileged component is the small, root-owned, socket-activated model
controller. Its configuration maps public model IDs to pre-existing container
names. The unprivileged worker has the controller socket's supplemental group
but does not have `/var/run/docker.sock`, a Docker group membership, or a
privileged container.

All environment-specific controller processes on one compute host use the same
root-owned container inventory and the same host-wide advisory file lock. A
start transition holds that lock while reloading the inventory, inspecting
every listed container, and changing Docker state; stop transitions use the
same lock. Staging and production therefore cannot race to occupy the mutually
exclusive GPU. Every managed staging, production, or
retained legacy model container must be in the shared inventory; each
environment's public-model mapping must be a subset. Starting those containers
outside the controller is outside this single-authority contract.

## Browser authorization flow

1. Apache proxies the same-origin Chat route to the gateway loopback listener.
2. The gateway forwards the Novo cookie only to Novo's loopback companion API,
   with an independent file-backed integration token.
3. Novo validates its session and returns the current user plus current notebook
   grants.
4. The gateway intersects the requested notebook IDs with those grants.
5. Before compute, the gateway obtains normalized content for an exact revision
   and synchronizes that revision to the worker if necessary.
6. The gateway submits a signed job whose opaque public handle is bound to the
   Novo user. It rechecks authorization before returning job state or results.

Changing a user's Novo access therefore affects new Chat requests without a
password or database snapshot refresh. Worker artifacts are cacheable derived
data, not an authorization source.

## Internal protocol

Gateway-to-worker requests and worker responses are HMAC authenticated with
separate file-backed keys and key IDs. Signatures bind the protocol version,
audience, environment, request ID, timestamp, method/path/status, and body.
Requests have a bounded clock skew and a durable nonce replay guard. Staging and
production signatures cannot be replayed across environments.

The worker API is not routed by Apache. It listens on compute-host loopback and
is reached only through a compute-host-initiated remote forward whose gateway
listener is also loopback-only.

## Durable and derived state

The gateway persists only job ownership/mapping data. The worker persists job
state, normalized document revisions, and index artifacts under a role- and
environment-specific state directory. Staging and production never share
SQLite files, indexes, secrets, ports, or tunnel listeners.

The default retention window is seven days across gateway mappings, terminal
worker jobs, active index pointers, normalized documents, and index artifacts.
Successful ingest finalization removes staging batches immediately. A periodic
idle pass expires old pointers and deletes unreferenced document and index
directories after a one-hour grace period; it skips deletion while an operation
is queued or running. Documents and indexes also have independent 50 GiB
application quotas. An expired notebook revision is exported and rebuilt on its
next authorized use.

Index identity includes notebook ID, Novo content revision, and index schema
version. Ingestion finalization is atomic: an interrupted or checksum-mismatched
export never becomes the active index. `novo:all`, if exposed by a backend, is a
derived aggregate scope and never a grant or a real notebook.

## Availability

The gateway does not require the worker or tunnel to start. If compute is down,
the Chat shell can report a sanitized retryable unavailable state while Novo
continues normally. Restarted workers recover interrupted durable operations;
model state is queried from the controller rather than inferred from stale UI
state.

## Optional installation

Novo's integration endpoint and Chat navigation should be guarded by an
installation-level feature flag. When disabled, the integration route is not
mounted and no Chat link is rendered. This keeps the reusable Novo project free
of compute-host assumptions and avoids introducing a tenant model prematurely.
