# CUDA userspace libraries and ptxas arrive in the pinned torch/Triton wheels.
# Host NVIDIA driver is provided by RunPod; no host driver is installed here.
FROM python:3.12-slim-bookworm
ARG SOURCE_REVISION=unknown
LABEL org.opencontainers.image.source="https://github.com/jasonjiang8866/flashattentionanycard" \
      org.opencontainers.image.revision=$SOURCE_REVISION
ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    BO_PYTHON=/opt/venv/bin/python \
    BO_SUITE=smoke \
    BO_WORKSPACE=/workspace/bo-kernel-tuner \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential git ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && python -m venv /opt/venv
WORKDIR /opt/bo-kernel-tuner
COPY . .
RUN bash runpod/install.sh \
    && printf '%s\n' "$SOURCE_REVISION" > SOURCE_REVISION \
    && /opt/venv/bin/python runpod/launch.py smoke --dry-run
CMD ["/bin/bash", "/opt/bo-kernel-tuner/runpod/entrypoint.sh"]
