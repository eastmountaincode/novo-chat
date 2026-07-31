# Deployment

The supported distributed deployment is documented in
[deploy/README.md](deploy/README.md). It uses the gateway host for the existing
HTTPS `/chat/` route and a private compute host for the worker. The old
single-host snapshot deployment is intentionally unsupported.

No template in this repository is ready to install verbatim. Replace and review
every `@PLACEHOLDER@`, use file-backed secrets, pin the exact tested image
digest, and complete staging verification before production.
