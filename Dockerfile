
FROM nvidia/cuda:12.3.2-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive     PYTHONDONTWRITEBYTECODE=1     PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv git build-essential curl && \
    rm -rf /var/lib/apt/lists/*

# Install JAX CUDA wheels
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install "jax[cuda12_local]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html && \
    python3 -m pip install optax equinox numpy scipy matplotlib tqdm einops pyyaml chex pytest coverage pylint

WORKDIR /workspace
COPY . /workspace

RUN pip install -e .

# Default command shows help
CMD ["ppo-train", "--help"]
