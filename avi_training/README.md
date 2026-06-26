# Tier-B AVI Gamma net — training

Trains the amortized-VI inference network `ζ_ρ(x_t, t)` of the **evidence trick**
(Micheli, Monod & Bhatt 2025) for the SPAD Poisson problem. The net predicts a
per-voxel **Gamma** belief over the Poisson intensity; at inference, Gamma is
conjugate to the Poisson likelihood, so the DAPS inner Langevin loop can be
replaced by a **closed-form conjugate posterior draw** (Tier B).

## Files
| file | what it is |
|---|---|
| `avi_gamma.py` | shared link/loss/inference module — **single source of truth** for train & inference consistency |
| `train_avi_temporal.py` | 1D temporal AVI net (reuses `improved-diffusion-main` TrainLoop) |
| `train_avi_spatial.py` | 2D spatial AVI net (reuses `guided-diffusion` TrainLoop) |
| `launch_temporal.sh`, `launch_spatial.sh` | ready-to-run cluster launch commands |

Both training scripts reuse the **entire** existing TrainLoop (EMA, fp16, MPI/DDP,
checkpointing, resume). The only changes vs. the score-prior training are: build
the UNet with `learn_sigma=True` (→ 2 output channels = `α_raw, β_raw`) and swap
`diffusion.training_losses` for the Poisson–Gamma AVI objective (Eq. 13).

## Objective (no observations needed)
Per voxel, with `α = softplus(out₀)+ε`, `β = softplus(out₁)+ε`, `θ = g⁻¹(x₀)`:
```
L = lgamma(α) − α·log(β) − (α−1)·log θ + β·θ
```
`(α, β)` come from the net on the **noised** `x_t`; `θ` from the **clean** `x₀`.
Training therefore uses only the clean normalized-log-flux corpus — **the same
data the score prior was trained on** — and never sees photon counts `y`.

## Data
- **Temporal:** the `.pt` of shape `[N, T]` already in `[-1,1]` used for the 1D
  prior (e.g. `data/normalized_sim_flux.pt`). Pass `--normalize False`.
- **Spatial:** frames of normalized log-flux in `[-1,1]`, **same normalization as
  the 2D prior**. ⚠️ See the *DATA CONTRACT* note in `train_avi_spatial.py`: the
  default loader uses `guided_diffusion.load_data` + grayscale reduction; if your
  2D prior was trained from a custom source, edit `load_spatial_data`.

## Run
```bash
# 1D temporal, single GPU
bash launch_temporal.sh
# 1D temporal, 4 GPUs (MPI, same as the prior)
NGPU=4 bash launch_temporal.sh

# 2D spatial (set DATA_DIR first)
DATA_DIR=/path/to/frames bash launch_spatial.sh
```
**Smoke test** (runs a couple of steps then exits, via the repo's built-in flag):
```bash
DIFFUSION_TRAINING_TEST=1 CUDA_VISIBLE_DEVICES=0 python train_avi_temporal.py \
    --data_path <data.pt> --sequence_length 1024 --normalize False \
    --save_interval 1 --log_dir /tmp/avi_smoke
```

## Outputs
In `--log_dir`: `model_best.pt` (temporal only), `modelNNNNNN.pt`, `ema_<rate>_NNNNNN.pt`,
`optNNNNNN.pt`. Watch `loss` in the log; ~50k steps matches the paper.

## `rate_link` (keep train == inference)
- `normflux` *(default)*: `θ = φ/φ_max ∈ (0,1]`, `log θ = (x₀−1)·L/2`. Numerically
  stable. Inference exposure `E = B_k·ppp_scale·φ_max` (`φ_max = e^L`).
- `flux`: `θ = φ ∈ [1,φ_max]`, `log θ = (x₀+1)·L/2` (matches the notebook's `phi`).
  Inference exposure `E = B_k·ppp_scale`.

`--log_flux_max` **must equal** the notebook's `LOG_FLUX_MAX = log(10000) ≈ 9.2103`.

## Consuming the checkpoint at inference (Tier-B inner loop, wired separately)
Rebuild the **same** UNet with `learn_sigma=True`, load `model_best.pt`, then per
annealing step replace `langevin_mcmc` with:
```python
from avi_gamma import gamma_params_from_output, spad_exposure, conjugate_posterior_sample
# out = zeta_net(x_ddpm, t)            # [B,2,...] applied like eps_from_*_prior
# E   = spad_exposure(bin_sizes, ppp_scale, LOG_FLUX_MAX, rate_link)   # per bin
# (pool counts Y and exposure E over the coarse->fine pool, then:)
# x0  = conjugate_posterior_sample(out, Y_pooled, E_pooled, LOG_FLUX_MAX, rate_link)
```
Requires `DARK_COUNT = 0` for exact conjugacy. Two nets (temporal + spatial) can
be blended exactly like the score priors (`blend_weight_high`) — or start with one
and keep analytic-Poisson Langevin for the other phase.
