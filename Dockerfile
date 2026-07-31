FROM python:3.12.11-slim-bookworm

ARG VCS_REF=unknown
ARG SOURCE_URL=""
LABEL org.opencontainers.image.title="Novo Chat" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.source="${SOURCE_URL}"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_NO_PROGRESS=1 \
    PATH=/app/.venv/bin:$PATH

ARG UV_VERSION=0.8.6

RUN groupadd --system --gid 10001 novo-chat \
    && useradd --system --uid 10001 --gid 10001 --home-dir /nonexistent --shell /usr/sbin/nologin novo-chat

WORKDIR /app
RUN python -m pip install --no-cache-dir "uv==${UV_VERSION}" \
    && mkdir -p /run/config /run/novo-chat /run/secrets /var/lib/novo-chat-gateway /var/lib/novo-chat-worker \
    && chown -R 10001:10001 /run/config /run/novo-chat /run/secrets /var/lib/novo-chat-gateway /var/lib/novo-chat-worker

COPY pyproject.toml uv.lock README.md MANIFEST.in ./
COPY novo_chat ./novo_chat
RUN uv sync --locked --no-dev --no-editable \
    && .venv/bin/novo-chat-runtime --help >/dev/null

USER 10001:10001

ENTRYPOINT ["novo-chat-runtime"]
CMD ["gateway"]
