"""
Train the 2D SPATIAL AVI Gamma net (SPADAPS Tier-B / evidence trick).

Mirrors guided-diffusion's scripts/image_train.py, with two changes:
  * the spatial UNet is built with in_channels=1, learn_sigma=True -> 2 output
    channels (alpha, beta), matching the 2D score prior's architecture;
  * diffusion.training_losses is swapped for the Poisson--Gamma AVI objective.
Everything else (EMA, fp16, MPI/DDP, checkpointing, resume) is reused unchanged
from guided_diffusion.train_util.TrainLoop.

>>> DATA CONTRACT (read this) <<<
x_start fed to the loss MUST be normalized log-flux in [-1, 1], using the SAME
normalization as the 2D score prior's training frames. The default loader below
uses guided_diffusion.load_data (a directory of frames) and reduces to a single
channel -- correct IF your 2D prior was trained that way. If you trained the 2D
prior from a custom source (e.g. an .npy/.pt of normalized log-flux frames),
replace `load_spatial_data` so it yields ([B,1,H,W] in [-1,1], {}) pairs with
that same normalization. Otherwise the link theta=g^{-1}(x0) will be inconsistent
with inference and the conjugate posterior will be wrong.

Single GPU:
  CUDA_VISIBLE_DEVICES=0 python train_avi_spatial.py \
      --data_dir /path/to/normalized_logflux_frames \
      --image_size 256 --batch_size 8 --lr 1e-4 \
      --log_flux_max 9.210340371976182 --rate_link normflux \
      --lr_anneal_steps 50000 --save_interval 5000 \
      --log_dir avi_spatial_256

Multi-GPU (MPI):
  mpiexec -n 4 python train_avi_spatial.py  <same flags>

Outputs go to $OPENAI_LOGDIR (set here from --log_dir): modelNNNNNN.pt, ema_*.pt,
optNNNNNN.pt. (guided-diffusion's TrainLoop has no best-checkpoint tracking, so
pick the lowest-loss step from the logs.)
"""
import argparse
import os
import sys
import types

# --- make the guided-diffusion repo importable (override with env var if needed) ---
GUIDED_DIFFUSION_PATH = os.environ.get(
    "GUIDED_DIFFUSION_PATH", "/homes/tan583/scratch/guided-diffusion"
)
if GUIDED_DIFFUSION_PATH not in sys.path:
    sys.path.insert(0, GUIDED_DIFFUSION_PATH)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch as th

from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop

from avi_gamma import make_avi_training_losses

# Architecture of the 2D score prior (see the notebook's SPATIAL_2D_OVERRIDES).
# The AVI net reuses this exact backbone; learn_sigma=True yields out_channels=2.
SPATIAL_OVERRIDES = dict(
    image_size=256,
    in_channels=1,
    num_channels=256,
    num_res_blocks=2,
    num_head_channels=64,
    attention_resolutions="32,16,8",
    resblock_updown=True,
    use_scale_shift_norm=True,
    class_cond=False,
    use_fp16=False,
)


def load_spatial_data(data_dir, batch_size, image_size):
    """Yield ([B,1,H,W] in [-1,1], {}) pairs of normalized log-flux frames.

    Default: guided_diffusion.load_data on a frame directory, reduced to 1 channel.
    REPLACE THIS if your 2D prior was trained from a different source / normalization
    (see the DATA CONTRACT note at the top of the file).
    """
    gen = load_data(
        data_dir=data_dir,
        batch_size=batch_size,
        image_size=image_size,
        class_cond=False,
    )
    for batch, cond in gen:
        if batch.shape[1] != 1:                 # collapse RGB -> single channel
            batch = batch.mean(dim=1, keepdim=True)
        yield batch, {}


def model_defaults():
    d = model_and_diffusion_defaults()
    d.update(SPATIAL_OVERRIDES)
    return d


def main():
    args = create_argparser().parse_args()

    if args.log_dir:
        os.environ.setdefault("OPENAI_LOGDIR", args.log_dir)
    dist_util.setup_dist()
    logger.configure()

    logger.log("creating 2D spatial AVI Gamma net (in_channels=1, learn_sigma=True -> 2 channels)...")
    model_diff_args = args_to_dict(args, model_defaults().keys())
    model_diff_args["learn_sigma"] = True       # force out_channels = 2*in_channels = 2 (alpha,beta)
    model, diffusion = create_model_and_diffusion(**model_diff_args)
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

    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader (normalized log-flux frames)...")
    data = load_spatial_data(args.data_dir, args.batch_size, args.image_size)

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
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_dir="",
        log_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=50000,
        batch_size=8,
        microbatch=-1,
        ema_rate="0.9999",
        log_interval=100,
        save_interval=5000,
        resume_checkpoint="",
        use_fp16=False,                 # keep fp32: AVI loss is not MSE
        fp16_scale_growth=1e-3,
        # ---------------- AVI-specific ----------------
        log_flux_max=9.210340371976182,  # = log(10000); MUST equal the notebook's LOG_FLUX_MAX
        rate_link="normflux",            # "normflux" (theta in (0,1], stable) or "flux"
        gamma_eps=1e-4,
    )
    defaults.update(model_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
