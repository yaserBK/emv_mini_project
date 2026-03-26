# ──────────────────────────────────────────────────────────────────────────────
# emv_mini_project · Jetson Nano anomaly-detection container
#
# Target: Jetson Nano 4 GB / 2 GB running JetPack 4.6.1 (L4T r32.7.1)
#   • Ubuntu 18.04  •  Python 3.6.9  •  CUDA 10.2  •  PyTorch 1.10
#
# Build:
#   docker build -t emv-anomaly .
#
# Prerequisites on the Nano (run once after flashing JetPack):
#   See compose.yaml for the full Docker + NVIDIA runtime setup commands.
#
# ── Upgrading to a newer Jetson ──
# For JetPack 5.x / 6.x the base image already ships PyTorch + torchvision;
# only change the FROM tag — no other lines need updating:
# JetPack 5.1.x (Orin)  →  nvcr.io/nvidia/l4t-pytorch:r35.4.1-pth2.1-py3
# JetPack 6.x  (Orin)   →  nvcr.io/nvidia/l4t-pytorch:r36.2.0-pth2.2-py3
# ──────────────────────────────────────────────────────────────────────────────
FROM nvcr.io/nvidia/l4t-pytorch:r32.7.1-pth1.10-py3

WORKDIR /app

# ── System libraries ──────────────────────────────────────────────────────────
# X11 client libraries needed for cv2.imshow to render via the host display.
# These may not be present in the base image and are safe to add.
#
# Do NOT install python3-opencv here — the base image already includes OpenCV
# compiled with CUDA 10.2 support and its Python 3.6 bindings.  The Ubuntu
# 18.04 apt package (OpenCV 3.2.0, no CUDA) would shadow those bindings and
# remove GPU acceleration.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# ── Python packages ───────────────────────────────────────────────────────────
# The base image (r32.7.1-pth1.10-py3) already ships:
#   • PyTorch 1.10.0      — CUDA 10.2, Python 3.6
#   • torchvision 0.11.0  — CUDA 10.2, Python 3.6
#   • numpy               — Python 3.6-compatible build
#   • OpenCV              — CUDA-enabled, Python 3.6 bindings
#
# Do NOT reinstall torch or torchvision — PyPI has no cp36-aarch64-CUDA wheel
# and any reinstall would overwrite the CUDA build with a CPU-only one.
#
# Do NOT upgrade numpy to >=1.20 — numpy dropped Python 3.6 in 1.20.0.
# The pre-installed version (1.19.x) is fully compatible with this codebase.
#
# Pillow is a torchvision dependency but may not be present; installing it is
# safe and idempotent.
RUN pip3 install --no-cache-dir "pillow>2.1.1"

# ── Project source ────────────────────────────────────────────────────────────
COPY anomaly/      ./anomaly/
COPY preprocess.py build_distribution.py inference.py video_inference.py ./

# ── Runtime mount points ──────────────────────────────────────────────────────
# /models → place your .pkl calibration file here (or mount at docker run time)
# /data   → mount image directories for preprocessing / training / inference
VOLUME ["/models", "/data"]

# ── Default: live webcam scoring ──────────────────────────────────────────────
# Override CMD when running other scripts, e.g.:
#   docker compose run inference
#   docker compose run build
ENTRYPOINT ["python3"]
CMD ["video_inference.py", "--calibration", "/models/distribution.pkl"]
