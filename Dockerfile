# ====================================================================
# Stage 1: build — get deps
# ====================================================================
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS build

WORKDIR /app

# Pin exact deps from lockfile
COPY pyproject.toml uv.lock /app/
RUN uv sync --frozen --no-dev --no-install-project

# Copy source
COPY neuroflow_core/ /app/neuroflow_core/

# Now install the project itself
RUN uv sync --frozen --no-dev

# ====================================================================
# Stage 2: runtime — minimal image
# ====================================================================
FROM python:3.13-slim-bookworm AS runtime

WORKDIR /app

# Runtime deps only
RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends \
    tini \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /app /app
COPY --from=build /app/.venv /app/.venv
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    NEUROFLOW_DB_PATH=/data/neuroflow.db

EXPOSE 8888

ENTRYPOINT ["tini", "--", "/entrypoint.sh"]
CMD ["neuroflow"]
