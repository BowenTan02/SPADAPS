"""
Shared Poisson--Gamma AVI ("evidence trick") utilities for SPADAPS Tier-B.

Single source of truth for the link convention used by BOTH
  * training   (train_avi_temporal.py / train_avi_spatial.py), and
  * inference  (the conjugate posterior draw in the DAPS inner loop).
Keeping them here guarantees train/inference consistency.

------------------------------------------------------------------------------
What the AVI net predicts
------------------------------------------------------------------------------
The amortized-VI inference network zeta_rho(x_t, t) predicts, per voxel, the two
natural hyperparameters of a Gamma belief over the (normalized) Poisson
intensity theta, from a 2-channel UNet output:

    alpha = softplus(out[:, 0]) + eps      # Gamma shape  = nu + 1   (> 0)
    beta  = softplus(out[:, 1]) + eps      # Gamma rate   = tau      (> 0)
    theta = g^{-1}(x0)                      # link, see log_theta_from_xnorm

------------------------------------------------------------------------------
Training objective (no observations needed)
------------------------------------------------------------------------------
Micheli, Monod & Bhatt (2025), "Diffusion Models for Inverse Problems in the
Exponential Family", Theorem 3.5 / Eq. (13). For the Poisson--Gamma case the
simulation-free per-voxel loss is

    L = lgamma(alpha) - alpha*log(beta) - (alpha - 1)*log_theta + beta*theta

where (alpha, beta) come from the net applied to the NOISED x_t, and theta is
evaluated on the CLEAN x0. No data y is used, so the training set is exactly the
clean normalized-log-flux corpus the score prior was trained on. (Sanity check:
the per-sample minimizer gives posterior mean alpha/beta = theta, with finite
spread because the diffusion noising mixes many x0 into each x_t.)

------------------------------------------------------------------------------
Inference (Tier-B inner loop; reference, wired separately)
------------------------------------------------------------------------------
Gamma is conjugate to Poisson, so with pooled photon counts Y = sum N and pooled
exposure E the posterior is closed form (Prop. A.8):

    theta | x_t, y  ~  Gamma(alpha + Y, beta + E)

and x0 is recovered by inverting the link. Requires dark_count = 0 for exact
conjugacy (a non-zero additive dark term breaks Gamma closure).
"""
import math

import torch
import torch.nn.functional as F


def mean_flat(x):
    """Mean over all non-batch dims -> shape [B]."""
    return x.mean(dim=list(range(1, x.dim())))


def gamma_params_from_output(out, eps=1e-4):
    """Raw 2-channel net output -> strictly-positive (alpha, beta).

    out : [B, 2, ...]  channel 0 = shape logit, channel 1 = rate logit.
    Returns (alpha, beta), each [B, 1, ...].
    """
    assert out.shape[1] >= 2, f"expected >=2 output channels (alpha,beta), got {out.shape[1]}"
    alpha = F.softplus(out[:, 0:1]) + eps      # Gamma shape = nu + 1
    beta = F.softplus(out[:, 1:2]) + eps       # Gamma rate  = tau
    return alpha, beta


def log_theta_from_xnorm(x0_norm, log_flux_max, rate_link="normflux"):
    """Link g^{-1}: normalized log-flux x0 in [-1, 1] -> log Poisson intensity.

    "normflux" (default): theta = phi / phi_max in (0, 1]   -- numerically stable
                          log_theta = (x0 - 1) * L/2
    "flux"             : theta = phi in [1, phi_max]         -- matches notebook `phi`
                          log_theta = (x0 + 1) * L/2
    """
    half = log_flux_max / 2.0
    if rate_link == "normflux":
        return (x0_norm - 1.0) * half
    if rate_link == "flux":
        return (x0_norm + 1.0) * half
    raise ValueError(f"unknown rate_link: {rate_link!r}")


def xnorm_from_theta(theta, log_flux_max, rate_link="normflux", clamp_eps=1e-12):
    """Inverse link: Poisson intensity theta -> normalized log-flux x0 in [-1, 1]."""
    half = log_flux_max / 2.0
    log_theta = torch.log(theta.clamp_min(clamp_eps))
    if rate_link == "normflux":
        x0 = log_theta / half + 1.0
    elif rate_link == "flux":
        x0 = log_theta / half - 1.0
    else:
        raise ValueError(f"unknown rate_link: {rate_link!r}")
    return x0.clamp_(-1.0, 1.0)


def avi_gamma_loss_elementwise(out, x0_norm, log_flux_max, rate_link="normflux", eps=1e-4):
    """Per-voxel AVI Poisson--Gamma loss (Eq. 13).  out:[B,2,...], x0_norm:[B,1,...]."""
    alpha, beta = gamma_params_from_output(out, eps=eps)
    log_theta = log_theta_from_xnorm(x0_norm, log_flux_max, rate_link)
    theta = torch.exp(log_theta)
    return (
        torch.lgamma(alpha)
        - alpha * torch.log(beta)
        - (alpha - 1.0) * log_theta
        + beta * theta
    )


def make_avi_training_losses(log_flux_max, rate_link="normflux", eps=1e-4):
    """Build a drop-in replacement for ``GaussianDiffusion.training_losses``.

    Bind onto a diffusion instance with ``types.MethodType`` so the standard
    TrainLoop trains the AVI net instead of the DSM score. It reuses the
    diffusion's own ``q_sample`` and ``_scale_timesteps`` so the noising and the
    timestep convention match the score prior exactly. Returns ``{"loss": [B]}``,
    which is what TrainLoop / LossAwareSampler / best-checkpoint tracking expect.
    """
    def training_losses(self, model, x_start, t, model_kwargs=None, noise=None):
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = torch.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise=noise)
        out = model(x_t, self._scale_timesteps(t), **model_kwargs)      # [B, 2, ...]
        loss_elem = avi_gamma_loss_elementwise(out, x_start, log_flux_max, rate_link, eps)
        return {"loss": mean_flat(loss_elem)}

    return training_losses


# ----------------------------- inference helpers -----------------------------
# (Reference implementations for the Tier-B DAPS inner loop; not used by training.)
def spad_exposure(bin_sizes, ppp_scale, log_flux_max, rate_link="normflux"):
    """Per-bin Poisson exposure E such that  N ~ Poisson(E * theta)  is consistent
    with the chosen link.  bin_sizes = frames-per-bin B_k.

    "flux"     : E = B_k * ppp_scale
    "normflux" : E = B_k * ppp_scale * phi_max,  phi_max = exp(L)
    Then  E * theta = B_k * ppp_scale * phi = mu  (matches the notebook, dark=0).
    """
    E = bin_sizes * ppp_scale
    if rate_link == "normflux":
        E = E * math.exp(log_flux_max)
    elif rate_link != "flux":
        raise ValueError(f"unknown rate_link: {rate_link!r}")
    return E


@torch.no_grad()
def conjugate_posterior_sample(out, counts, exposure, log_flux_max,
                               rate_link="normflux", eps=1e-4):
    """Tier-B inner step: draw x0 from the closed-form Poisson--Gamma posterior.

    out      : [B, 2, ...] AVI net output (the Gamma PRIOR over theta given x_t).
    counts   : pooled photon counts Y (broadcastable to [B, 1, ...]).
    exposure : pooled exposure E       (broadcastable to [B, 1, ...]).
    Returns x0_norm in [-1, 1].  Requires dark_count = 0 for exact conjugacy.
    """
    alpha, beta = gamma_params_from_output(out, eps=eps)
    a_post = alpha + counts            # posterior shape  = alpha + Y
    b_post = beta + exposure           # posterior rate   = beta  + E
    theta = torch.distributions.Gamma(a_post, b_post).rsample()
    return xnorm_from_theta(theta, log_flux_max, rate_link)
