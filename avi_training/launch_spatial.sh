#!/usr/bin/env bash
# Train the 2D SPATIAL AVI Gamma net on the cluster.
# Edit the paths below (or pass via env), then:  bash launch_spatial.sh
# Multi-GPU:  NGPU=4 bash launch_spatial.sh
set -euo pipefail

# Reduce allocator fragmentation (you were ~512 MiB short on a 32GB card).
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Repo + data (override via environment)
export GUIDED_DIFFUSION_PATH="${GUIDED_DIFFUSION_PATH:-/homes/tan583/scratch/guided-diffusion}"
DATA_DIR="${DATA_DIR:-$GUIDED_DIFFUSION_PATH/data/}"  # SAME frames/normalization as the 2D prior
LOG_DIR="${LOG_DIR:-avi_spatial_256}"
NGPU="${NGPU:-1}"

# ---- resume + speed knobs (all overridable via env) ----
#   Resume (same fp32 config):  RESUME_CHECKPOINT=avi_spatial_256/model020000.pt bash launch_spatial.sh
#   Faster (fp16, see notes):   USE_FP16=True USE_CHECKPOINT=False MICROBATCH=8 \
#                                 RESUME_CHECKPOINT=avi_spatial_256/model020000.pt bash launch_spatial.sh
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"   # path to modelNNNNNN.pt to resume from ("" = fresh start)
USE_FP16="${USE_FP16:-False}"                # True ~= 2x on V100 tensor cores + halves activations
USE_CHECKPOINT="${USE_CHECKPOINT:-True}"     # gradient checkpointing; False is ~30% faster if it fits
MICROBATCH="${MICROBATCH:-2}"                # grads accumulate to effective batch 8; 8 = single pass

ARGS=(
  --data_dir "$DATA_DIR"
  --image_size 256
  --batch_size 8
  --microbatch "$MICROBATCH"
  --use_fp16 "$USE_FP16"
  --use_checkpoint "$USE_CHECKPOINT"
  --lr 1e-4
  --lr_anneal_steps 50000
  --save_interval 5000
  --log_interval 100
  --log_flux_max 9.210340371976182  # = log(10000); MUST match the notebook LOG_FLUX_MAX
  --rate_link normflux
  --log_dir "$LOG_DIR"
)
if [ -n "$RESUME_CHECKPOINT" ]; then
  ARGS+=(--resume_checkpoint "$RESUME_CHECKPOINT")
fi

cd "$(dirname "$0")"
if [ "$NGPU" -gt 1 ]; then
  mpiexec -n "$NGPU" python train_avi_spatial.py "${ARGS[@]}"
else
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python train_avi_spatial.py "${ARGS[@]}"
fi
