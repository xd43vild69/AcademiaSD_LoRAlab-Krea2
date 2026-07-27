# Multi-stage build: pequeño, seguro, listo para GPU
# Stage 1: builder (temporal, ~8 GB)
FROM nvidia/cuda:12.4-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential python3.13 python3.13-dev python3.13-venv \
    git wget curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copiar installer
COPY install_LoRAlab-Krea2.sh .
COPY scripts/ scripts/

# Crear venv e instalar
RUN python3.13 -m venv /opt/loralab/venv && \
    . /opt/loralab/venv/bin/activate && \
    pip install --upgrade pip setuptools wheel && \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 && \
    pip install \
      diffusers==0.39.0 \
      transformers==5.14.1 \
      peft==0.19.1 \
      accelerate==1.14.0 \
      safetensors==0.8.0 \
      bitsandbytes==0.49.2 \
      huggingface-hub==1.24.0 \
      Flask==3.1.3 \
      Flask-Limiter==3.5.0 \
      Flask-Login==0.6.3 \
      Werkzeug==3.1.8 \
      psutil==7.2.2 \
      Pillow==12.2.0 \
      numpy==2.4.4 \
      scipy==1.18.0 \
      insightface==1.0.1 \
      onnxruntime==1.28.0 \
      opencv-python==5.0.0.93 \
      && rm -rf /root/.cache/pip

# Stage 2: runtime (final, ~5.5 GB)
FROM nvidia/cuda:12.4-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 \
    PYTHONPATH=/opt/loralab:$PYTHONPATH \
    PATH=/opt/loralab/venv/bin:$PATH \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.13 openssh-server openssh-client ca-certificates \
    curl wget git openssl bash nano \
    && rm -rf /var/lib/apt/lists/*

# Usuario no-privilegiado
RUN useradd -m -s /bin/bash loralab && \
    mkdir -p /run/sshd /data/{datasets,output,cache,logs,models,ssh,certs,auth} && \
    chown -R loralab:loralab /data

# Copiar venv desde builder
COPY --from=builder --chown=loralab:loralab /opt/loralab/venv /opt/loralab/venv

# Copiar código
COPY --chown=loralab:loralab . /opt/loralab/

# SSH config (hardened)
COPY scripts/config/sshd_config /etc/ssh/sshd_config
RUN chmod 600 /etc/ssh/sshd_config

# Entrypoint
COPY entrypoint.sh /opt/loralab/entrypoint.sh
RUN chmod +x /opt/loralab/entrypoint.sh /opt/loralab/scripts/shell/generate_certs.sh && \
    chown loralab:loralab /opt/loralab/entrypoint.sh

WORKDIR /opt/loralab
USER loralab

EXPOSE 22 5000

ENTRYPOINT ["/opt/loralab/entrypoint.sh"]
