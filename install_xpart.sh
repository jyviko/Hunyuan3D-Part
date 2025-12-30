#!/usr/bin/env bash
set -euo pipefail

# ---------------- user-configurable ----------------
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"

# Torch baseline that worked well with wheels ecosystem
TORCH_VER="${TORCH_VER:-2.5.1+cu121}"
TV_VER="${TV_VER:-0.20.1+cu121}"
TA_VER="${TA_VER:-2.5.1+cu121}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"

# SpConv wheel package name (adjust for other CUDA builds)
SPCONV_PKG="${SPCONV_PKG:-spconv-cu124}"

# CUDA arch list for flash-attn (A100 SM80, H100 SM90)
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0;9.0}"
export MAX_JOBS="${MAX_JOBS:-1}"
export NINJA_FLAGS="${NINJA_FLAGS:--j1}"

# PyG wheel page matching torch+cuda
PYG_WHL_PAGE="${PYG_WHL_PAGE:-https://data.pyg.org/whl/torch-2.5.1+cu121.html}"

# Hugging Face writable locations
export HF_HOME="${HF_HOME:-$HOME/hf_cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
SONATA_ROOT="${SONATA_ROOT:-$HOME/sonata}"

# ---------------- helpers ----------------
log() { echo -e "\n[install] $*\n"; }

# ---------------- sanity checks ----------------
if [[ ! -f "XPart/gradio_demo.py" ]]; then
  echo "[install] ERROR: Run this script from the Hunyuan3D-Part repo root (expected XPart/gradio_demo.py)."
  exit 1
fi

# ---------------- venv ----------------
log "Creating fresh virtual environment: ${VENV_DIR}"
rm -rf "${VENV_DIR}"
"${PYTHON_BIN}" -m venv "${VENV_DIR}"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

log "Upgrading installer toolchain"
python -m pip install -U pip setuptools wheel ninja psutil packaging

# ---------------- system deps (optional) ----------------
log "Checking system OpenGL libraries for pymeshlab"
if [[ "$(uname -s)" == "Linux" ]] && command -v apt-get >/dev/null 2>&1; then
  if command -v sudo >/dev/null 2>&1; then
    sudo apt-get update -y || true
    sudo apt-get install -y libopengl0 libgl1 libglu1-mesa || true
  else
    apt-get update -y || true
    apt-get install -y libopengl0 libgl1 libglu1-mesa || true
  fi
else
  log "Skipping OpenGL libs install (non-Linux or apt-get unavailable)"
fi

# ---------------- torch ----------------
log "Installing PyTorch: torch==${TORCH_VER} torchvision==${TV_VER} torchaudio==${TA_VER}"
python -m pip install "torch==${TORCH_VER}" "torchvision==${TV_VER}" "torchaudio==${TA_VER}" \
  --index-url "${TORCH_INDEX_URL}"

log "Verifying torch"
python - <<'EOF'
import torch
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
EOF

# ---------------- core python deps ----------------
log "Installing Hunyuan3D-Part package (Sonata core deps included)"
python -m pip install -e ".[sonata]" --find-links "${PYG_WHL_PAGE}"

# ---------------- flash-attn ----------------
log "Installing flash-attn (source build, constrained for A100)"
log "TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} MAX_JOBS=${MAX_JOBS} NINJA_FLAGS=${NINJA_FLAGS}"
python -m pip install --no-build-isolation --no-cache-dir flash-attn

log "Verifying flash-attn import"
python - <<'EOF'
import flash_attn
print("flash_attn import: OK")
print("flash_attn version:", getattr(flash_attn, "__version__", "unknown"))
EOF

# ---------------- spconv ----------------
log "Installing spconv (Sonata dependency)"
python -m pip install "${SPCONV_PKG}"

# ---------------- torch_scatter (+ other PyG extensions) ----------------
log "Ensuring PyG extensions are installed (Sonata dependency)"
python -m pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
  -f "${PYG_WHL_PAGE}" || true

# ---------------- sonata editable install if present ----------------
if [[ -d "sonata" ]]; then
  log "Installing sonata in editable mode"
  (cd sonata && python -m pip install -e .)
fi

# ---------------- create cache dirs + runner ----------------
log "Creating Hugging Face cache dirs and sonata dir"
mkdir -p "${HF_HOME}" "${HUGGINGFACE_HUB_CACHE}" "${TRANSFORMERS_CACHE}" "${SONATA_ROOT}"

log "Creating runner script: run_xpart_demo.sh"
cat > run_xpart_demo.sh <<EOF
#!/usr/bin/env bash
set -euo pipefail
source "${VENV_DIR}/bin/activate"

# Use writable Hugging Face cache locations
export HF_HOME="${HF_HOME}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE}"
export SONATA_ROOT="${SONATA_ROOT}"

mkdir -p "${HF_HOME}" "${HUGGINGFACE_HUB_CACHE}" "${TRANSFORMERS_CACHE}" "${SONATA_ROOT}"

xpart-gradio
EOF
chmod +x run_xpart_demo.sh

log "Installation complete."
echo "[install] Next:"
echo "  ./run_xpart_demo.sh"
echo
echo "[install] If remote, use SSH port-forwarding from your laptop:"
echo "  ssh -L 7860:localhost:7860 user@REMOTE_HOST"
echo "  then open http://localhost:7860"
