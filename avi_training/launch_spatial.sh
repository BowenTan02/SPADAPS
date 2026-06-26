#!/usr/bin/env bash
# Train the 2D SPATIAL AVI Gamma net on the cluster.
# Edit the paths below (or pass via env), then:  bash launch_spatial.sh
# Multi-GPU:  NGPU=4 bash launch_spatial.sh
set -euo pipefail

# Repo + data (override via environment)
export GUIDED_DIFFUSION_PATH="${GUIDED_DIFFUSION_PATH:-/homes/tan583/scratch/guided-diffusion}"
DATA_DIR="${DATA_DIR:-$GUIDED_DIFFUSION_PATH/data/}"  # SAME frames/normalization as the 2D prior
LOG_DIR="${LOG_DIR:-avi_spatial_256}"
NGPU="${NGPU:-1}"

ARGS=(
  --data_dir "$DATA_DIR"
  --image_size 256
  --batch_size 8
  --lr 1e-4
  --lr_anneal_steps 50000
  --save_interval 5000
  --log_interval 100
  --log_flux_max 9.210340371976182  # = log(10000); MUST match the notebook LOG_FLUX_MAX
  --rate_link normflux
  --log_dir "$LOG_DIR"
)

cd "$(dirname "$0")"
if [ "$NGPU" -gt 1 ]; then
  mpiexec -n "$NGPU" python train_avi_spatial.py "${ARGS[@]}"
else
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python train_avi_spatial.py "${ARGS[@]}"
fi
