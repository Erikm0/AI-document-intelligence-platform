#!/bin/sh
# Apply database migrations, then hand over to the CMD (uvicorn by default).
set -e

# Models are baked into the image: never contact the Hugging Face Hub at runtime
# (faster start-up, works air-gapped). Set HF_HUB_OFFLINE=0 to allow downloads,
# e.g. after switching EMBEDDING_MODEL without rebuilding the image.
if [ "${PRELOAD_MODELS:-false}" = "true" ] && [ -z "${HF_HUB_OFFLINE:-}" ]; then
  export HF_HUB_OFFLINE=1
fi

if [ "${RUN_MIGRATIONS:-true}" = "true" ]; then
  echo "Applying database migrations..."
  attempt=1
  # PostgreSQL may still be initialising on first boot - retry briefly.
  until alembic upgrade head; do
    if [ "$attempt" -ge 10 ]; then
      echo "Database migrations failed after $attempt attempts" >&2
      exit 1
    fi
    attempt=$((attempt + 1))
    echo "Database not ready, retrying in 3s ($attempt/10)..."
    sleep 3
  done
fi

exec "$@"
