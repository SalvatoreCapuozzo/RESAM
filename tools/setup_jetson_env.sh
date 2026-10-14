#!/bin/bash
# Reproducible inference/training env for RESAM on Jetson Orin Nano (JetPack 5.1.1, L4T 35.3.1,
# TensorRT 8.5.2, CUDA 11.4). Layers a venv on the existing conda env that already has
# NVIDIA's torch 2.0.0+nv23.05 wheel, so nothing shared is modified.
#
#   bash tools/setup_jetson_env.sh            # creates .venv in the repo root
set -e
cd "$(dirname "$0")/.."
BASE_PY=${BASE_PY:-$HOME/anaconda3/envs/dl_env/bin/python}   # must provide torch (CUDA) + torchvision

"$BASE_PY" -m venv --system-site-packages .venv
SP=$(.venv/bin/python -c "import site;print(site.getsitepackages()[0])")

# TensorRT python bindings live in the system python; expose only that package
ln -sf /usr/lib/python3.8/dist-packages/tensorrt "$SP/tensorrt"
ln -sf /usr/lib/python3.8/dist-packages/tensorrt-8.5.2.2.dist-info "$SP/"

cat > /tmp/resam_constraints.txt <<EOF
torch==2.0.0+nv23.05
torchvision==0.15
numpy==1.24.4
EOF
# --no-deps for ultralytics: its requirements would pull a CPU-only torch from PyPI
.venv/bin/pip install --no-deps "ultralytics==8.4.46" ultralytics-thop
.venv/bin/pip install -c /tmp/resam_constraints.txt "transformers==4.46.3" "onnxslim==0.1.71" "polars==1.8.2"

.venv/bin/python - <<'EOF'
import torch, tensorrt, ultralytics, transformers
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("tensorrt", tensorrt.__version__, "| ultralytics", ultralytics.__version__, "| transformers", transformers.__version__)
EOF
echo "Optional for stable benchmarks (needs sudo): sudo nvpmodel -m 0 && sudo jetson_clocks"
