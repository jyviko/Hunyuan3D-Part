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

# A100 (SM80) build constraints for flash-attn
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
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
log "Installing Python dependencies (runtime + utilities)"
python -m pip install \
  gradio \
  huggingface_hub \
  omegaconf \
  pyyaml \
  einops \
  safetensors \
  tqdm \
  numpy \
  scipy \
  trimesh \
  scikit-image \
  pytorch_lightning \
  pymeshlab \
  timm \
  numba \
  diffusers \
  accelerate \
  transformers \
  httpx \
  joblib \
  threadpoolctl \
  pybind11 \
  pillow \
  opencv-python

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

# ---------------- torch_scatter (+ other PyG extensions) ----------------
log "Installing torch-scatter from PyG wheels"
python -m pip install torch-scatter -f "${PYG_WHL_PAGE}"

# Optional: common additional PyG extensions (safe if unused; skip if you want minimal)
log "Installing additional PyG extensions (torch-sparse/cluster/spline-conv) from PyG wheels"
python -m pip install torch-sparse torch-cluster torch-spline-conv -f "${PYG_WHL_PAGE}" || true

# ---------------- install repo requirements if present ----------------
if [[ -f "XPart/requirements.txt" ]]; then
  log "Installing XPart requirements.txt"
  python -m pip install -r XPart/requirements.txt
fi

# ---------------- sonata editable install if present ----------------
if [[ -d "sonata" ]]; then
  log "Installing sonata in editable mode"
  (cd sonata && python -m pip install -e .)
fi

# ---------------- apply Fix B: avoid /root/sonata permission issue ----------------
log "Patching P3-SAM/model.py to use a writable sonata download_root (Fix B)"
python - <<'EOF'
from pathlib import Path
import re

p = Path("P3-SAM/model.py")
if not p.exists():
    raise SystemExit("P3-SAM/model.py not found; cannot apply sonata download_root patch.")

txt = p.read_text()

# Replace hardcoded /root/sonata with ~/sonata (using pathlib)
repls = [
    ("download_root='/root/sonata'", "download_root=str(pathlib.Path.home() / 'sonata')"),
    ('download_root="/root/sonata"', "download_root=str(pathlib.Path.home() / 'sonata')"),
]
changed = False
for a,b in repls:
    if a in txt:
        txt = txt.replace(a,b)
        changed = True

if not changed:
    # Don't fail; just inform
    print("NOTE: Did not find hardcoded download_root='/root/sonata' in P3-SAM/model.py; skipping patch.")
else:
    if "import pathlib" not in txt:
        # Insert import after the first block of imports
        lines = txt.splitlines()
        insert_at = 0
        for i,l in enumerate(lines):
            if l.startswith("import ") or l.startswith("from "):
                insert_at = i+1
        lines.insert(insert_at, "import pathlib")
        txt = "\n".join(lines) + "\n"
    p.write_text(txt)
    print("Patched P3-SAM/model.py successfully.")
EOF

# ---------------- create cache dirs + runner ----------------
log "Creating Hugging Face cache dirs and sonata dir"
mkdir -p "${HF_HOME}" "${HUGGINGFACE_HUB_CACHE}" "${TRANSFORMERS_CACHE}" "${SONATA_ROOT}"

log "Creating runner script: run_xpart_demo.sh"
cat > run_xpart_demo.sh <<EOF
#!/usr/bin/env bash
set -euo pipefail
source "${VENV_DIR}/bin/activate"

# Make repo modules discoverable (P3-SAM uses a non-package folder name)
export PYTHONPATH="\$PWD/P3-SAM:\$PWD/XPart:\${PYTHONPATH:-}"

# Use writable Hugging Face cache locations
export HF_HOME="${HF_HOME}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE}"

mkdir -p "${HF_HOME}" "${HUGGINGFACE_HUB_CACHE}" "${TRANSFORMERS_CACHE}" "${SONATA_ROOT}"

python XPart/gradio_demo.py
EOF
chmod +x run_xpart_demo.sh

log "Installation complete."
echo "[install] Next:"
echo "  ./run_xpart_demo.sh"
echo
echo "[install] If remote, use SSH port-forwarding from your laptop:"
echo "  ssh -L 7860:localhost:7860 user@REMOTE_HOST"
echo "  then open http://localhost:7860"

