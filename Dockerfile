# Heart disease prediction API - production image.
#
#   docker build -t heart-disease-api:1.0.0 .
#   docker run -d -p 8000:8000 --name heart-api heart-disease-api:1.0.0
#
# Two stages: dependencies are installed in a "builder" stage, and only the finished
# virtual environment, the code and the model are copied into the final image.

ARG PYTHON_VERSION=3.13

# ---------------------------------------------------------------- builder
FROM python:${PYTHON_VERSION}-slim AS builder
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
# Copied on its own first, so this slow layer is cached until the requirements change.
COPY requirements-serve.txt .
RUN pip install -r requirements-serve.txt

# ---------------------------------------------------------------- runtime
FROM python:${PYTHON_VERSION}-slim AS runtime

LABEL org.opencontainers.image.title="heart-disease-api" \
      org.opencontainers.image.description="FastAPI service predicting heart disease (UCI Cleveland)" \
      org.opencontainers.image.source="https://github.com/SnehanjaliGandham/heart-disease-prediction-mlops" \
      org.opencontainers.image.version="1.0.0"

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH=/app/models/heart_disease_model.joblib \
    PREDICTION_THRESHOLD=0.5 \
    LOG_LEVEL=INFO

# Unprivileged user with a fixed numeric id (Kubernetes can verify runAsNonRoot).
RUN groupadd --system --gid 10001 app \
 && useradd --system --uid 10001 --gid app --no-create-home --shell /usr/sbin/nologin app

WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
# Code and model stay owned by root: the app user can read them but not change them.
COPY src/ src/
COPY models/heart_disease_model.joblib models/model_metadata.json models/

USER 10001:10001
EXPOSE 8000

# Docker marks the container "healthy" only once the model is loaded and serving.
HEALTHCHECK --interval=15s --timeout=3s --start-period=20s --retries=3 \
  CMD ["python", "-c", "import urllib.request, sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/ready', timeout=2).status == 200 else 1)"]

# --no-access-log: every request is already logged by the app as one JSON line.
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
