# Two stages so the CUDA-enabled torch wheels never reach the final image.
# Installing CPU-only torch first brings the built image to ~2GB; the CUDA
# wheels alone are larger than that.
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .

RUN pip install --upgrade pip \
    && pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt


FROM python:3.12-slim

# libpq for psycopg, curl for the container healthcheck.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libpq5 curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/models

WORKDIR /app

# Bake the embedding and reranker weights into the image. Without this the
# first knowledge search in a fresh container blocks on a ~120MB download,
# and an air-gapped deployment fails outright.
ARG EMBEDDING_MODEL=all-MiniLM-L6-v2
ARG RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
RUN python -c "\
from sentence_transformers import CrossEncoder, SentenceTransformer; \
SentenceTransformer('${EMBEDDING_MODEL}'); \
CrossEncoder('${RERANKER_MODEL}')" \
    && chmod -R a+rX /models

COPY alembic.ini ./
COPY alembic ./alembic
COPY app ./app
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
# Strip CR as well as setting the exec bit: a checkout on Windows can carry
# CRLF, and the kernel then reads the shebang as "/bin/sh\r" and the
# container exits with "bad interpreter". .gitattributes guards the repo
# side; this guards the build against any checkout.
RUN sed -i 's/\r$//' /usr/local/bin/entrypoint.sh \
    && chmod +x /usr/local/bin/entrypoint.sh

# Non-root: the app writes only to the Chroma volume, mounted separately.
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /data/chroma \
    && chown -R appuser:appuser /data
USER appuser

ENV CHROMA_PATH=/data/chroma

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

ENTRYPOINT ["entrypoint.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
