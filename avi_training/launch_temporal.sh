#!/usr/bin/env bash
# Train the 1D TEMPORAL AVI Gamma net on the cluster.
# Edit the paths below (or pass via env), then:  bash launch_temporal.sh
# Multi-GPU:  NGPU=4 bash launch_temporal.sh
set -euo pipefail

# Repo + data (override via environment)
export IMPROVED_DIFFUSION_PATH="${IMPROVED_DIFFUSION_PATH:-/u/scratch1/tan583/improved-diffusion-main}"
DATA_PATH="${DATA_PATH:-$IMPROVED_DIFFUSION_PATH/data/normalized_sim_flux.pt}"   # SAME .pt as the 1D prior
LOG_DIR="${LOG_DIR:-avi_temporal_1024}"
NGPU="${NGPU:-1}"

ARGS=(
  --data_path "$DATA_PATH"
  --sequence_length 1024
  --num_channels 64
  --channel_mult "1,2,3,4"
  --num_res_blocks 2
  --attention_resolutions "256,128"
  --diffusion_steps 1000
  --noise_schedule linear
  --normalize False                 # data is already in [-1, 1]
  --batch_size 64
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
  mpiexec -n "$NGPU" python train_avi_temporal.py "${ARGS[@]}"
else
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python train_avi_temporal.py "${ARGS[@]}"
fi
