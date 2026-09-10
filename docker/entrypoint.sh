#!/bin/sh
# Applies pending migrations before handing off to the server, so a fresh
# database (or a newly deployed migration) needs no manual step.
set -e

echo "Running database migrations..."
alembic upgrade head

echo "Starting: $*"
exec "$@"
