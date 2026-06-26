"""
Train the 1D TEMPORAL AVI Gamma net (SPADAPS Tier-B / evidence trick).

Mirrors improved-diffusion's scripts/temporal_train.py, with two changes:
  * the temporal UNet is built with learn_sigma=True  -> 2 output channels (alpha, beta);
  * diffusion.training_losses is swapped for the Poisson--Gamma AVI objective.
Everything else (data loader, EMA, fp16, MPI/DDP, checkpointing, best-checkpoint
tracking, resume) is reused unchanged from improved_diffusion.train_util.TrainLoop.

The training data is the SAME normalized-log-flux trajectories used for the 1D
score prior: a .pt tensor of shape [N, T] already in [-1, 1] (pass
--normalize False, exactly as the prior was trained).

Single GPU:
  CUDA_VISIBLE_DEVICES=0 python train_avi_temporal.py \
      --data_path /u/scratch1/tan583/improved-diffusion-main/data/normalized_sim_flux.pt \
      --sequence_length 1024 --num_channels 64 --channel_mult "1,2,3,4" \
      --num_res_blocks 2 --attention_resolutions "256,128" \
      --batch_size 64 --lr 1e-4 --diffusion_steps 1000 --noise_schedule linear \
      --normalize False --save_interval 5000 --lr_anneal_steps 50000 \
      --log_flux_max 9.210340371976182 --rate_link normflux \
      --log_dir avi_temporal_1024

Multi-GPU (MPI, like the prior):
  mpiexec -n 4 python train_avi_temporal.py  <same flags>

Outputs (in --log_dir): model_best.pt, modelNNNNNN.pt, ema_<rate>_*.pt, optNNNNNN.pt.
Load model_best.pt at inference into a learn_sigma=True temporal UNet of the same
config, then use avi_gamma.conjugate_posterior_sample for the Tier-B inner step.
"""
import argparse
import os
import sys
import types

# --- make the improved-diffusion repo importable (override with env var if needed) ---
IMPROVED_DIFFUSION_PATH = os.environ.get(
    "IMPROVED_DIFFUSION_PATH", "/u/scratch1/tan583/improved-diffusion-main"
)
if IMPROVED_DIFFUSION_PATH not in sys.path:
    sys.path.insert(0, IMPROVED_DIFFUSION_PATH)
# avi_gamma.py lives next to this script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from improved_diffusion import dist_util, logger
from improved_diffusion.temporal_datasets import load_temporal_data
from improved_diffusion.resample import create_named_schedule_sampler
from improved_diffusion.temporal_script_util import (
    temporal_model_and_diffusion_defaults,
    create_temporal_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from improved_diffusion.train_util import TrainLoop

from avi_gamma import make_avi_training_losses


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure(dir=args.log_dir)

    logger.log("creating 1D temporal AVI Gamma net (learn_sigma=True -> 2 channels = alpha,beta)...")
    model_diff_args = args_to_dict(args, temporal_model_and_diffusion_defaults().keys())
    model_diff_args["learn_sigma"] = True       # force 2 output channels (alpha_raw, beta_raw)
    model, diffusion = create_temporal_model_and_diffusion(**model_diff_args)
    model.to(dist_util.dev())

    # ---- swap the standard DSM loss for the Poisson--Gamma AVI objective ----
    diffusion.training_losses = types.MethodType(
        make_avi_training_losses(
            log_flux_max=args.log_flux_max,
            rate_link=args.rate_link,
            eps=args.gamma_eps,
        ),
        diffusion,
    )

    total_params = sum(p.numel() for p in model.parameters())
    logger.log(f"AVI net parameters: {total_params:,}")
    logger.log(f"  link={args.rate_link}  log_flux_max={args.log_flux_max:.6f}  gamma_eps={args.gamma_eps}")
    logger.log(f"  sequence_length={args.sequence_length}  num_channels={args.num_channels}  "
               f"channel_mult={args.channel_mult}  attn={args.attention_resolutions}")

    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader (clean normalized log-flux trajectories)...")
    data = load_temporal_data(
        data_path=args.data_path,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        deterministic=False,
        normalize=args.normalize,
    )

    logger.log("training AVI Gamma net...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        track_best_checkpoint=args.track_best_checkpoint,
        best_checkpoint_metric=args.best_checkpoint_metric,
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_path="",
        log_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=50000,          # paper trains the inference net ~50k steps
        batch_size=64,
        microbatch=-1,
        ema_rate="0.9999",
        log_interval=100,
        save_interval=5000,
        resume_checkpoint="",
        use_fp16=False,                 # AVI loss is not MSE; keep fp32 for stable loss scaling
        fp16_scale_growth=1e-3,
        normalize=False,                # data file is already normalized to [-1, 1]
        track_best_checkpoint=True,
        best_checkpoint_metric="loss",
        # ---------------- AVI-specific ----------------
        log_flux_max=9.210340371976182,  # = log(10000); MUST equal the notebook's LOG_FLUX_MAX
        rate_link="normflux",            # "normflux" (theta in (0,1], stable) or "flux"
        gamma_eps=1e-4,
    )
    defaults.update(temporal_model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
