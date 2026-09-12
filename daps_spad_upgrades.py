#!/usr/bin/env python
"""DAPS inner-sampler upgrades for the low-photon SPAD problem.

Drop-in replacements for three pieces of ``DAPS_3D_Poisson.ipynb``:

  1. ``fused_eps``            -- averaged spatio-temporal priors (matches
                                 run_proxdiffpir_schedule.py's `--fuse-mode`)
  2. ``langevin_prox_newton`` -- Langevin whose data half-step is the CERTIFIED
                                 Newton prox instead of an explicit gradient
  3. ``langevin_tree``        -- Langevin driven by the multiscale tree score,
                                 the module's own documented DAPS entry point

plus ``travel_report`` -- the diagnostic that motivated all three.

WHY (measured 2026-08-28)
-------------------------
The notebook's inner Langevin never mixes. Its per-annealing-step travel budget
is ``tau = num_mcmc * lr``; the prior anchor ``(x0_hat - x)/sigma_t^2`` relaxes
at rate ``tau/sigma_t^2``. With the shipped lr schedule (1e-3 -> 1e-5, M=40):

    step   sigma     tau      sigma^2    tau/sigma^2
       0  157.407  4.0e-02   2.478e+04     1.6e-06
      50    3.424  2.0e-02   1.172e+01     1.7e-03
      99    0.044  4.0e-04   1.898e-03     2.1e-01

``tau/sigma^2 << 1`` at EVERY step, so ``x0_y ~= x0_hat`` throughout and the
likelihood barely enters. At the final step one photon moves a voxel by
~1.8e-3 out of a [-1,1] range, while the injected Brownian noise over the same
40 steps is 2.8e-2 -- a noise-to-signal ratio of 15x. THAT is the observed
smoothing: not a property of DAPS, an under-relaxed chain.

The effective data weight in DAPS is ``lambda_data * sigma_t^2``, algebraically
identical to ProxDiffPIR's ``gamma * sigma_t^2``. So lambda_data=1.0 is
nominally the DIRTIEST ProxDiffPIR setting (gamma=1, candidate_rate 1.3e-2) --
it only looks clean because the travel limit throttles it by 3-6 decades.

THE WARNING THAT FOLLOWS
------------------------
Fixing the mixing ALONE walks DAPS toward its true inner target
``p(y|x) * N(x0_hat, sigma_t^2 I)``, which ``DAPS_PnP_ULA_Repair.tex`` shows is
fully voxel-separable -- so no amount of sampling can move a photon's evidence
to a neighbour, and salt-and-pepper is the stationary law. Raise the step size
without adding coupling and you should EXPECT the artifact to appear. That is a
prediction, and a cheap one to test.

Coupling can be restored on either side:
  * prior side     -- annealed PnP-ULA (the .tex), coupled score inside the loop
  * likelihood side-- the tree factorization (this module's ``langevin_tree``)
The Newton prox does NOT restore it: the certified solver is separable by
construction. It fixes stiffness, not coupling. Use it for step size, not as
the answer to the artifact.
"""
from __future__ import annotations

import math
import numpy as np
import torch

__all__ = [
    "spatial_weight", "fused_eps",
    "sigma_scaled_lr", "travel_report", "make_rng_streams",
    "langevin_explicit", "langevin_prox_newton", "langevin_tree",
    "langevin_prox_tree",
    "binomial_loglik", "exact_inner_sample",
    "sample_daps_3d_v2", "sample_daps_v2",
]


# ---------------------------------------------------------------------------
# 1. Prior fusion -- averaging instead of hard switching
# ---------------------------------------------------------------------------
def spatial_weight(sigma, mode="average", fuse_w=0.5, switch_sigma=0.5,
                   sigma_c=1.0, sharpness=4.0, temporal_first=False):
    """Weight on the SPATIAL (2D) prior at this sigma.

    Semantics copied from ``run_proxdiffpir_schedule.spatial_weight`` so a DAPS
    arm and a ProxDiffPIR arm with the same flags fuse identically. The runner's
    default is mode="average", fuse_w=0.5 -- a constant 50/50 blend, NOT a
    switch. The notebook currently ships the "hard" branch.
    """
    if mode == "average":
        return float(fuse_w)
    if mode == "hard":
        high = "1D" if temporal_first else "2D"
        low = "2D" if temporal_first else "1D"
        phase = high if sigma >= switch_sigma else low
        return 1.0 if phase == "2D" else 0.0
    if mode == "soft":
        s = 1.0 / (1.0 + np.exp(-sharpness * (np.log(max(sigma, 1e-8))
                                              - np.log(max(sigma_c, 1e-8)))))
        return float((1.0 - s) if temporal_first else s)
    raise ValueError(f"unknown fuse mode {mode!r}")


@torch.no_grad()
def fused_eps(x_ddpm, t_idx, sigma_here, eps_2d_fn, eps_1d_fn, *,
              mode="average", fuse_w=0.5, switch_sigma=0.5,
              sigma_c=1.0, sharpness=4.0, temporal_first=False):
    """eps = w * eps_2D + (1 - w) * eps_1D.

    ``eps_2d_fn(x_ddpm, t_idx)`` / ``eps_1d_fn(x_ddpm, t_idx)`` are the
    notebook's existing wrappers. At w in {0, 1} exactly one prior is evaluated,
    so "hard" costs what it always did; "average" evaluates both.

    Cost note: the second prior is NOT 2x. The 1D prior is 11.1M params against
    the 2D prior's 552.8M, and run_proxdiffpir_schedule's measured cost model
    charges 137 s/step for both priors against 105 s/step for one -- about
    +30%, not +100%.
    """
    w = spatial_weight(sigma_here, mode=mode, fuse_w=fuse_w,
                       switch_sigma=switch_sigma, sigma_c=sigma_c,
                       sharpness=sharpness, temporal_first=temporal_first)
    if w >= 1.0 - 1e-12:
        return eps_2d_fn(x_ddpm, t_idx), "2D", w
    if w <= 1e-12:
        return eps_1d_fn(x_ddpm, t_idx), "1D", w
    eps = w * eps_2d_fn(x_ddpm, t_idx) + (1.0 - w) * eps_1d_fn(x_ddpm, t_idx)
    return eps, f"avg(w={w:g})", w


# ---------------------------------------------------------------------------
# 1b. Reproducibility helper
# ---------------------------------------------------------------------------
def _randn_like(x, generator=None):
    """randn_like that honours an explicit torch.Generator.

    torch.randn_like takes no generator, so every kernel that used it drew from
    the global RNG: a run could not be reproduced from a recorded seed, and two
    arms differing only in inner step count consumed different amounts of global
    entropy, silently changing every later outer-noise draw. Pass a generator to
    pin a stream; omit it and the behaviour is byte-identical to before.
    """
    if generator is None:
        return torch.randn_like(x)
    return torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)


def make_rng_streams(seed, device):
    """Three independent generators: (init, outer renoising, inner sampling).

    Separate streams so that changing the inner kernel or its iteration count
    cannot change the initial state or the outer noise sequence -- the property
    that makes two arms a paired comparison. Returns (g_init, g_outer, g_inner);
    seed=None returns (None, None, None), i.e. the legacy global-RNG behaviour.
    """
    if seed is None:
        return None, None, None
    dev = torch.device(device)
    out = []
    for k in range(3):
        g = torch.Generator(device=dev)
        g.manual_seed((int(seed) << 2) + k)
        out.append(g)
    return tuple(out)


# ---------------------------------------------------------------------------
# 2. Step-size law and the diagnostic that justifies it
# ---------------------------------------------------------------------------
def sigma_scaled_lr(sigma_t, c=0.05, lr_floor=0.0, lr_cap=None):
    """lr = c * sigma_t^2 -- the step size the anchor's curvature actually asks for.

    The inner target's Gaussian factor has precision 1/sigma_t^2, so an explicit
    Langevin step is stable for lr < 2*sigma_t^2 and relaxes at rate lr/sigma^2.
    Setting lr = c*sigma^2 makes the relaxation rate a CONSTANT c per step, so
    ``num_mcmc * c`` (not the shipped 1.6e-06) controls mixing, uniformly in
    sigma. c=0.05 with num_mcmc=40 gives 2.0 -- mixed.

    CAVEAT: this law is only safe with an IMPLICIT data step. The Poisson term
    contributes curvature (L/2)^2 * mu ~ 8 at bright voxels, so an explicit
    scheme needs lr < ~0.25 regardless of sigma; at sigma=157, c*sigma^2 = 1237
    would diverge instantly. Pair this with ``langevin_prox_newton`` (or cap it
    hard via lr_cap) -- that is the real reason to want the Newton solver here.
    """
    lr = float(c) * float(sigma_t) ** 2
    if lr_cap is not None:
        lr = min(lr, float(lr_cap))
    return max(lr, float(lr_floor))


def travel_report(sigma_arr, t_indices, num_mcmc, lr_of_step, tag=""):
    """Print tau/sigma^2 per annealing step. tau/sigma^2 << 1 => chain frozen."""
    rows = []
    K = len(t_indices) - 1
    for s in range(K):
        sg = float(sigma_arr[int(t_indices[s])])
        lr = float(lr_of_step(s, sg))
        tau = num_mcmc * lr
        rows.append((s, sg, lr, tau, sg ** 2, tau / max(sg ** 2, 1e-30)))
    print(f"\nLangevin travel budget {tag}")
    print(f"{'step':>5}{'sigma':>10}{'lr':>10}{'tau=M*lr':>11}"
          f"{'sigma^2':>11}{'tau/sigma^2':>13}  verdict")
    for s, sg, lr, tau, s2, r in rows[:: max(1, K // 10)]:
        print(f"{s:>5}{sg:>10.3f}{lr:>10.2e}{tau:>11.2e}{s2:>11.3e}{r:>13.2e}"
              f"  {'FROZEN' if r < 0.3 else 'mixing'}")
    # Report WHERE it mixes, not just the minimum: a capped explicit schedule
    # is frozen at high sigma and mixed at low sigma, and calling that "never
    # mixes" on the min alone is misleading.
    mixed = [r for r in rows if r[5] >= 0.3]
    worst = min(r[5] for r in rows)
    if not mixed:
        verdict = "chain NEVER mixes -- lambda cannot act anywhere"
    elif len(mixed) == len(rows):
        verdict = "mixes at EVERY sigma"
    else:
        verdict = (f"mixes over the last {len(mixed)}/{len(rows)} steps "
                   f"(sigma <= {max(r[1] for r in mixed):.2f}); frozen above")
    print(f"  min tau/sigma^2 = {worst:.2e}  ->  {verdict}")
    return rows


# ---------------------------------------------------------------------------
# 3. Three inner samplers, one signature
# ---------------------------------------------------------------------------
@torch.no_grad()
def langevin_explicit(x0_hat, sigma_t, data_score_fn, lr, num_steps,
                      lambda_data=1.0, clip_x=True, generator=None):
    """The notebook's current inner sampler, kept verbatim for A/B."""
    sigma_sq = float(sigma_t) ** 2 + 1e-12
    sqrt_2lr = math.sqrt(2.0 * float(lr))
    x = x0_hat.clone()
    for _ in range(int(num_steps)):
        score = lambda_data * data_score_fn(x) + (x0_hat - x) / sigma_sq
        score = torch.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
        x = x + lr * score + sqrt_2lr * _randn_like(x, generator)
        if clip_x:
            x = x.clamp_(-1.0, 1.0)
    return x


@torch.no_grad()
def langevin_prox_newton(x0_hat, sigma_t, prox_fn, lr, num_steps,
                         lambda_data=1.0, clip_x=True, generator=None):
    """Split (prox-)Langevin: explicit anchor + Brownian, IMPLICIT Poisson step.

        z = x + lr*(x0_hat - x)/sigma^2 + sqrt(2*lr)*xi
        x = prox_{lr*lambda*D}(z)       <- certified Newton solver

    ``prox_fn(z, rho)`` must solve  argmin_u D(u;y) + (rho/2)||u - z||^2.
    ``run_proxdiffpir_schedule.newton_data_step`` has exactly this shape, and
    the tempering identity rho_eff = rho/gamma from that module gives

        rho = 1 / (lr * lambda_data).

    Handling D implicitly removes the Poisson curvature from the stability
    bound, which is what lets ``sigma_scaled_lr`` be used at all. Because each
    step is far larger, use FEWER steps: num_steps ~ 5-10 here does more work
    than 40 explicit ones, and each prox is a full Newton solve over the volume.

    NOTE this operator is separable -- it fixes stiffness, not the missing
    inter-voxel coupling. See the module docstring.
    """
    sigma_sq = float(sigma_t) ** 2 + 1e-12
    sqrt_2lr = math.sqrt(2.0 * float(lr))
    rho = 1.0 / max(float(lr) * float(lambda_data), 1e-30)
    x = x0_hat.clone()
    for _ in range(int(num_steps)):
        z = x + lr * (x0_hat - x) / sigma_sq + sqrt_2lr * _randn_like(x, generator)
        z = torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
        if clip_x:
            # Only when the caller asked for it. Clamping z unconditionally
            # truncates the proposal asymmetrically about a non-zero mean and
            # biases the stationary law -- measured at -0.036 on a Gaussian
            # target whose analytic mean is known.
            z = z.clamp_(-1.0, 1.0)
        x = prox_fn(z, rho)
        if clip_x:
            x = x.clamp_(-1.0, 1.0)
    return x


@torch.no_grad()
def langevin_tree(x0_hat, sigma_t, tree, lr, num_steps,
                  lambda_data=1.0, clip_x=True, use_natural=False, weights=None,
                  generator=None):
    """Langevin driven by the multiscale tree score -- the coupled likelihood.

    ``tree`` is a ``TreePoissonLikelihood`` from the 4.3b module of
    ProxDiffPIR_3D_Poisson_Tree.ipynb. That module's own header names this
    integration explicitly:

        "Prox-DiffPIR consumes prox(); DAPS would consume nll_grad() inside its
         inner Langevin loop (and optionally natural_direction() as a
         frozen-metric preconditioner - flag daps_use_natural, off by default)."

    Two reasons this is the interesting arm:
      * the exact Poisson-thinning factorization is NOT voxel-separable -- the
        per-node multinomial factors couple voxels, which is the coupling
        DAPS_PnP_ULA_Repair.tex identifies as missing, supplied on the
        likelihood side instead of the prior side;
      * in log-intensity coordinates the node score is BOUNDED by the parent
        count, where the per-voxel Poisson score is not, so a larger explicit
        step stays stable.

    ``use_natural`` stays OFF by default, matching the module: Fisher
    preconditioning is scale-inverted on this problem (dark voxels carry the
    least information and get amplified the most).
    """
    sigma_sq = float(sigma_t) ** 2 + 1e-12
    sqrt_2lr = math.sqrt(2.0 * float(lr))
    w = weights if weights is not None else tree.weights(float(sigma_t))
    x = x0_hat.clone()
    for _ in range(int(num_steps)):
        if use_natural:
            data_score = tree.natural_direction(x, w=w)
        else:
            data_score = -tree.nll_grad(x, w=w)      # score = -grad(NLL)
        score = lambda_data * data_score + (x0_hat - x) / sigma_sq
        score = torch.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
        x = x + lr * score + sqrt_2lr * _randn_like(x, generator)
        if clip_x:
            x = x.clamp_(-1.0, 1.0)
    return x


# ---------------------------------------------------------------------------
# 4. Cost model -- why the tree route is affordable and PnP-ULA is not
# ---------------------------------------------------------------------------
# Where the coupling comes from decides where the cost lands:
#
#   PnP-ULA : coupling from the PRIOR. s_theta,eps(x^(k)) is a DENOISER call,
#             and it sits INSIDE the inner loop -> network calls scale with
#             num_mcmc. That is the 20+ h.
#   tree    : coupling from the LIKELIHOOD. nll_grad is pooling pyramids and
#             elementwise ops -- no UNet. The inner loop stays cheap and the
#             network is still only called at the outer/PF-ODE steps.
#
# So the tree buys the coupling PnP-ULA was after at vanilla-DAPS cost.
#
# COST CONSTANTS. `s_prior_both` / `s_prior_one` are MEASURED (the cost model in
# run_proxdiffpir_schedule.py: 137 s/step both priors, 105 s/step one, at
# 1024x256x256 on an A100-class card; wave-1 measured ~3.5 h for 100 steps on a
# V100, so these hold there too). `s_prox` and `s_tree_grad` are ESTIMATES --
# measure them once with `time_inner_ops` before trusting any total.
COST = dict(s_prior_both=130.0, s_prior_one=105.0, s_prox=7.0, s_tree_grad=0.5)


def estimate_runtime(num_annealing=100, num_inner_pfode=5, num_mcmc=40,
                     inner="explicit", fuse="average", cost=None):
    """Hours for one DAPS run. inner in {explicit, prox, tree, pnp_ula}."""
    c = dict(COST); c.update(cost or {})
    s_prior = c["s_prior_both"] if fuse == "average" else c["s_prior_one"]
    prior_calls = num_annealing * max(1, num_inner_pfode)
    per_inner = {"explicit": 0.0, "prox": c["s_prox"],
                 "tree": c["s_tree_grad"], "pnp_ula": s_prior}[inner]
    inner_s = num_annealing * num_mcmc * per_inner
    total = prior_calls * s_prior + inner_s
    return dict(hours=total / 3600.0, prior_calls=prior_calls,
                prior_h=prior_calls * s_prior / 3600.0, inner_h=inner_s / 3600.0)


def time_inner_ops(fn, x, repeats=5, warmup=2, label=""):
    """Measure one inner-loop op (tree nll_grad, prox, ...) on the real volume."""
    import time
    for _ in range(warmup):
        fn(x)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn(x)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    per = (time.perf_counter() - t0) / repeats
    print(f"  {label or 'op'}: {per*1000:.1f} ms/call")
    return per


# ---------------------------------------------------------------------------
# 5. Loading the tree module from its notebook -- one source of truth
# ---------------------------------------------------------------------------
DEFAULT_TREE_NB = ("/Users/tan583/Documents/Diffusion/improved-diffusion-main/"
                   "ProxDiffPIR_3D_Poisson_Tree.ipynb")


def load_tree_module(notebook_path=None, log_flux_max=None, b_offset=1e-7,
                     marker="make_tree_data_step"):
    """exec the 4.3b tree-likelihood cell and return its namespace.

    The module lives in a notebook cell, not a .py. Copying it here would fork
    it; instead we exec the cell straight out of the notebook JSON, so the Tree
    notebook stays the single source of truth and an edit there is picked up on
    the next import. The cell is self-contained -- it imports math, dataclasses,
    numpy and torch itself, and reaches for only two notebook globals
    (LOG_FLUX_MAX, PP_B_OFFSET), both via globals().get(..., fallback).

    b_offset = 1e-7, and that number is a THREE-WAY CONTRACT CONFLICT resolved
    by measurement, not taste:

      * the tree REQUIRES lam_dc = bin_sizes*dark_count + b_offset > 0 and
        raises otherwise (spec 7.1);
      * the certified Newton prox REQUIRES dark_count == 0 AND b_offset == 0
        and raises otherwise (frozen matched contract);
      * the strategy note kills the notebook's legacy b_offset=1e-3 because b is
        not in the simulator and at phi=1 it is ~25x the signal.

    All three cannot hold. Measured on the tree score (|nll_grad|_rms, one tile,
    ppp=0.001), b_offset=1e-3 perturbs it by 40% against the small-b limit,
    while 1e-7 / 1e-8 / 1e-9 agree to five significant figures. So 1e-7 is a
    pure positivity floor: the tree at 1e-7 and the Newton prox at 0 are solving
    objectives that differ below measurement resolution, and a tree-vs-voxel
    comparison at these settings is not confounded by b. Do NOT raise it toward
    1e-3 to "match the notebook" -- that reintroduces a 40% perturbation.

    Returns a namespace dict with TreePoissonLikelihood, TreeLikelihoodConfig,
    make_tree_data_step, LogFluxObservationMap, ...
    """
    import json
    path = notebook_path or DEFAULT_TREE_NB
    with open(path) as fh:
        nb = json.load(fh)
    src = None
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        text = "".join(cell["source"])
        if marker in text and "class TreePoissonLikelihood" in text:
            src = text
            break
    if src is None:
        raise RuntimeError(f"no cell defining {marker} found in {path}")
    # A real module registered in sys.modules, not a bare dict: @dataclass
    # resolves sys.modules[cls.__module__].__dict__ for its type checks and
    # dies with AttributeError on a namespace that is not a registered module.
    import sys, types
    mod_name = "tree_likelihood_module"
    mod = types.ModuleType(mod_name)
    sys.modules[mod_name] = mod
    ns = mod.__dict__
    if log_flux_max is None:
        log_flux_max = math.log(10000.0)
    ns["LOG_FLUX_MAX"] = float(log_flux_max)
    ns["PP_B_OFFSET"] = float(b_offset)
    exec(compile(src, f"{path}::4.3b", "exec"), ns)
    missing = [n for n in ("TreePoissonLikelihood", "make_tree_data_step",
                           "TreeLikelihoodConfig") if n not in ns]
    if missing:
        raise RuntimeError(f"tree cell exec'd but is missing {missing}")
    return ns


# ---------------------------------------------------------------------------
# 6. Composed sampler -- fused priors + sigma-scaled lr + choice of inner kernel
# ---------------------------------------------------------------------------
@torch.no_grad()
def sample_daps_v2(bin_counts, bin_sizes, ppp_scale, *,
                   eps_2d_fn, eps_1d_fn, sigma_arr, alphas_cumprod_t,
                   vol_shape, device,
                   num_annealing=100, num_inner_pfode=1, num_mcmc=40,
                   inner="tree", tree=None, prox_fn=None, data_score_fn=None,
                   lambda_data=0.1, lr_c=0.05, lr_cap=None,
                   fuse="average", fuse_w=0.5, switch_sigma=0.5,
                   temporal_first=False, clip_x=True,
                   snapshot_every=10, report=True, verbose=True):
    """DAPS with the three repairs wired in. Returns (x0, trajectory).

    Differences from the notebook's ``sample_daps_3d``:
      * priors are FUSED (default average w=0.5), not hard-switched;
      * lr = lr_c * sigma_t^2 instead of a linear anneal, so the chain's
        relaxation rate is num_mcmc*lr_c at EVERY sigma rather than 1.6e-06;
      * the inner kernel is selectable; "tree" uses the coupled multiscale
        score, which is the only option that is both coupled AND cheap.

    lambda_data defaults to 0.1, NOT 1.0. lambda*sigma^2 is algebraically
    ProxDiffPIR's gamma*sigma^2, and the 12-arm sweep put gamma=1 at
    candidate_rate 1.3e-2 (worst measured) against gamma=0.1 at 1.8e-4.
    """
    B = vol_shape[0]
    t_indices = np.linspace(len(sigma_arr) - 1, 0, num_annealing + 1).astype(np.int64)
    lr_of = lambda s, sg: sigma_scaled_lr(sg, c=lr_c, lr_cap=lr_cap)
    if report:
        travel_report(sigma_arr, t_indices, num_mcmc, lr_of,
                      tag=f"(inner={inner}, lr=c*sigma^2, c={lr_c})")
        est = estimate_runtime(num_annealing, num_inner_pfode, num_mcmc,
                               inner=inner, fuse=fuse)
        print(f"  estimated runtime ~{est['hours']:.1f} h "
              f"({est['prior_h']:.1f} h priors + {est['inner_h']:.1f} h inner)")

    x_t = torch.randn(vol_shape, device=device) * float(sigma_arr[-1])
    trajectory, x0_y = [], None

    for step in range(num_annealing):
        t_cur = int(t_indices[step])
        sigma_t = float(sigma_arr[t_cur])

        # --- inner PF-ODE (fused priors at every substep) -> x0_hat ----------
        x = x_t
        seq = np.unique(np.clip(np.linspace(t_cur, 0, max(2, num_inner_pfode + 1))
                                .astype(np.int64), 0, len(sigma_arr) - 1))[::-1]
        for i in range(len(seq) - 1):
            tc, tn = int(seq[i]), int(seq[i + 1])
            sc = float(sigma_arr[tc])
            x_ddpm = torch.sqrt(alphas_cumprod_t[tc]) * x
            eps, _, _ = fused_eps(x_ddpm, tc, sc, eps_2d_fn, eps_1d_fn,
                                  mode=fuse, fuse_w=fuse_w,
                                  switch_sigma=switch_sigma,
                                  temporal_first=temporal_first)
            eps = torch.nan_to_num(eps, nan=0.0, posinf=0.0, neginf=0.0)
            x = x + (float(sigma_arr[tn]) - sc) * eps
        x0_hat = x.clamp(-1.0, 1.0)

        # --- inner sampler ---------------------------------------------------
        lr = lr_of(step, sigma_t)
        if inner == "tree":
            x0_y = langevin_tree(x0_hat, sigma_t, tree, lr, num_mcmc,
                                 lambda_data=lambda_data, clip_x=clip_x)
        elif inner == "prox":
            x0_y = langevin_prox_newton(x0_hat, sigma_t, prox_fn, lr, num_mcmc,
                                        lambda_data=lambda_data, clip_x=clip_x)
        elif inner == "explicit":
            x0_y = langevin_explicit(x0_hat, sigma_t, data_score_fn, lr, num_mcmc,
                                     lambda_data=lambda_data, clip_x=clip_x)
        else:
            raise ValueError(f"unknown inner {inner!r}")

        if step < num_annealing - 1:
            x_t = x0_y + float(sigma_arr[int(t_indices[step + 1])]) * torch.randn_like(x0_y)
        else:
            x_t = x0_y

        if (step % snapshot_every == 0) or (step == num_annealing - 1):
            trajectory.append((step, t_cur, inner, x0_y.detach().cpu().clone()))
        if verbose and (step % max(1, num_annealing // 10) == 0
                        or step == num_annealing - 1):
            print(f"  step {step:3d} t={t_cur:4d} sigma={sigma_t:8.3f} lr={lr:.2e} "
                  f"x0_y=[{x0_y.min():+.3f},{x0_y.max():+.3f}]", flush=True)
    return x0_y, trajectory


@torch.no_grad()
def langevin_prox_tree(x0_hat, sigma_t, tree, lr, num_steps,
                       lambda_data=1.0, clip_x=True, weights=None, generator=None):
    """Prox-Langevin whose implicit data step is the TREE prox.

    Structurally identical to ``langevin_prox_newton`` -- same explicit anchor,
    same Brownian term, same ``rho = 1/(lr*lambda)`` -- so a voxel arm and a
    tree arm differ ONLY in which likelihood the prox solves. That is the point:
    with both arms proximal, "coupled vs per-voxel" is no longer confounded with
    "implicit vs explicit".

    ``tree.prox(x0_bar, rho_t, w)`` shares the certified solver's contract:
    argmin_x nll(x) + (rho_t/2)||x - x0_bar||^2, rho_t on the quadratic.

    TWO DIFFERENCES from the certified solver, worth stating in any writeup:
      * it is APPROXIMATE -- K_prox damped preconditioned iterations with
        monotone-decrease backtracking, not a bounded safeguarded exact solve;
      * its default direction is ``natural_direction`` (Fisher-preconditioned,
        zeta_inner=0.5), so Fisher preconditioning is ON inside the tree prox
        even when ``daps_use_natural`` is off. Given that Fisher is
        scale-inverted on this problem, ``cfg.prox_use_exact_grad=True`` is the
        ablation that turns it off.
    """
    sigma_sq = float(sigma_t) ** 2 + 1e-12
    sqrt_2lr = math.sqrt(2.0 * float(lr))
    rho = 1.0 / max(float(lr) * float(lambda_data), 1e-30)
    w = weights if weights is not None else tree.weights(float(sigma_t))
    x = x0_hat.clone()
    for _ in range(int(num_steps)):
        z = x + lr * (x0_hat - x) / sigma_sq + sqrt_2lr * _randn_like(x, generator)
        z = torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
        if clip_x:
            z = z.clamp_(-1.0, 1.0)
        x = tree.prox(z, rho, w=w)
        if clip_x:
            x = x.clamp_(-1.0, 1.0)
    return x


# ---------------------------------------------------------------------------
# 4. Drop-in sampler for DAPS_3D_Poisson.ipynb
# ---------------------------------------------------------------------------
# COST (measured from the notebook's own tqdm: 482.67 s/annealing step at
# NUM_INNER_PFODE=5, i.e. 13.3 h/run):
#
#     inner   fuse       s/step   hours
#         1   hard           97     2.7
#         1   average       126     3.5     <- recommended
#         3   average       378    10.5
#         5   average       630    17.5     <- worse than the rejected PnP-ULA
#
# 5 * 96.5 s accounts for the entire measured step, so NUM_INNER_PFODE is ~100%
# of the cost and the 40 Langevin iterations are free. Turning prior averaging
# on WITHOUT dropping inner takes you from 13.3 h to 17.5 h -- the two requests
# are in tension unless inner comes down first.
#
# STABILITY, and the quantitative case for the prox:
#   explicit Langevin needs lr < 2/((L/2)^2 * mu_max) ~ 0.24, so with a safe
#   lr_cap=0.1 the law lr=c*sigma^2 is capped whenever sigma > sqrt(cap/c).
#   At c=0.05 that is sigma > 1.41 -- above it the chain is STILL frozen
#   (tau/sigma^2 = 4/sigma^2 = 1.6e-4 at sigma=157). An implicit data step has
#   no such cap, which is the concrete reason to want the Newton solver here.

@torch.no_grad()
def sample_daps_3d_v2(
    *, eps_2d_fn, eps_1d_fn, sigma_arr, alphas_cumprod_t, vol_shape, device,
    inner_kind="explicit", data_score_fn=None, prox_fn=None, tree=None,
    obs_counts=None, obs_sizes=None, ppp_scale=None,
    num_annealing=100, num_inner_pfode=1, num_mcmc=40,
    lr_c=0.05, lr_cap=0.1, lambda_data=0.1,
    fuse_mode="average", fuse_w=0.5, switch_sigma=0.5, temporal_first=False,
    clip_x=True, num_diffusion_steps=1000, snapshot_every=10, verbose=True,
    lr_fn=None, seed=None, exact_final_mean=True, step_log=None,
    exact_frame_chunk=64,
):
    """DAPS with fused priors, a sigma-scaled step size, and a choice of inner kernel.

    inner_kind: "condmap" -- the conditional MAP of the SAME inner target:
                           x0_y = prox_{lambda sigma_t^2 D}(x0_hat), deterministic,
                           no chain and no draw (needs prox_fn). The mechanism
                           control for "is it the draw or the target?": identical
                           outer loop, photons and priors, with the draw replaced
                           by the mode. rho = 1/(lambda sigma_t^2) is the same
                           tempering ProxDiffPIR calls gamma. 2026-09-12.
                "exact" -- EXACT inverse-CDF sample of the inner target
                           p(y|x)^lam N(x; x0_hat, sigma_t^2) on [-1,1]; no
                           chain, no step size (needs obs_counts/obs_sizes/
                           ppp_scale). The honest DAPS arm -- see
                           exact_inner_sample. 2026-09-12.
                "prox"  -- certified Newton prox on the per-voxel Poisson
                "tree"  -- the TREE prox (same contract, coupled likelihood)
                "explicit"   -- legacy per-voxel score, CONTROL ONLY
                "tree_score" -- explicit tree score, kept for an ablation
    The first two are the experimental arms: both proximal, so the only thing
    that varies is WHICH likelihood the prox solves.
    Defaults are the recommended config: inner=1, averaged priors, lambda=0.1
    (the ProxDiffPIR ladder's best point, since lambda*sigma^2 == gamma*sigma^2).

    Execution provenance (2026-09-12):
      * ``lr_fn(step, sigma_t)`` overrides the built-in ``lr = lr_c*sigma^2``.
        Callers with their own schedule MUST pass it here: there was previously
        no way to do so, so a runner could print one schedule and execute
        another.
      * ``seed`` seeds three independent streams (initial state, outer
        renoising, inner sampling) via ``make_rng_streams``, so a run is
        reproducible and two arms differing only in the inner kernel share the
        initial state and the outer noise sequence. seed=None keeps the legacy
        global-RNG behaviour.
      * ``step_log``, if a list, receives one dict per annealing step recording
        the settings ACTUALLY used (t, sigma, lr, inner kind, mean-vs-draw).
        Write it next to the metrics so a tag can be checked against execution.
      * ``exact_final_mean`` takes the exact posterior MEAN on the last level of
        the ``exact`` arm instead of a draw (see ``exact_inner_sample``).
      * ``exact_frame_chunk`` is the inner step's memory knob; it does not move
        the sampled law. Measured peak added over the inputs at 1024x256x256:
        unchunked 4.1 GiB, 256 -> 2.7, 64 -> 1.0, 16 -> 0.8.
    """
    g_init, g_outer, g_inner = make_rng_streams(seed, device)
    t_indices = np.linspace(num_diffusion_steps - 1, 0, num_annealing + 1).astype(np.int64)
    x_t = torch.randn(vol_shape, device=device, generator=g_init) * float(sigma_arr[-1])
    fuse = dict(mode=fuse_mode, fuse_w=fuse_w, switch_sigma=switch_sigma,
                temporal_first=temporal_first)
    traj, x0_y = [], None

    for step in range(num_annealing):
        t_cur = int(t_indices[step]); sigma_t = float(sigma_arr[t_cur])

        # --- inner reverse PF-ODE (EDM DDIM), fused prior at every substep ---
        x = x_t
        if num_inner_pfode <= 1:
            seq = [t_cur, t_cur]
        else:
            seq = np.unique(np.clip(np.linspace(t_cur, 0, num_inner_pfode + 1)
                                    .astype(np.int64), 0, len(sigma_arr) - 1))[::-1]
        if num_inner_pfode <= 1:
            sg = float(sigma_arr[t_cur])
            xd = torch.sqrt(alphas_cumprod_t[t_cur]) * x
            eps, _, _ = fused_eps(xd, t_cur, sg, eps_2d_fn, eps_1d_fn, **fuse)
            x0_hat = x - sg * torch.nan_to_num(eps)
        else:
            for i in range(len(seq) - 1):
                tc, tn = int(seq[i]), int(seq[i + 1])
                sc, sn = float(sigma_arr[tc]), float(sigma_arr[tn])
                xd = torch.sqrt(alphas_cumprod_t[tc]) * x
                eps, _, _ = fused_eps(xd, tc, sc, eps_2d_fn, eps_1d_fn, **fuse)
                x = x + (sn - sc) * torch.nan_to_num(eps)
            x0_hat = x
        x0_hat = x0_hat.clamp_(-1.0, 1.0)

        # --- inner Langevin, step size from the anchor's own curvature ---
        # lr_fn wins when given: the caller's schedule must be the EXECUTED one.
        lr = (float(lr_fn(step, sigma_t)) if lr_fn is not None
              else sigma_scaled_lr(sigma_t, c=lr_c, lr_cap=lr_cap))
        take_mean = bool(exact_final_mean) and step == num_annealing - 1
        if inner_kind == "condmap":
            # Conditional MAP of the inner target: no Brownian term, no chain.
            # A prox arm with num_mcmc=1 is NOT this -- its first half-step kicks
            # the iterate by sqrt(2 lr) ~ sqrt(2) sigma_t before the solve.
            if prox_fn is None:
                raise ValueError("inner_kind='condmap' needs prox_fn(z, rho)")
            rho = 1.0 / max(float(lambda_data) * sigma_t ** 2, 1e-30)
            x0_y = prox_fn(x0_hat, rho)
            if clip_x:
                x0_y = x0_y.clamp(-1.0, 1.0)
        elif inner_kind == "prox":
            if prox_fn is None:
                raise ValueError("inner_kind='prox' needs prox_fn(z, rho)")
            x0_y = langevin_prox_newton(x0_hat, sigma_t, prox_fn, lr, num_mcmc,
                                        lambda_data=lambda_data, clip_x=clip_x,
                                        generator=g_inner)
        elif inner_kind == "exact":
            if obs_counts is None or obs_sizes is None or ppp_scale is None:
                raise ValueError("inner_kind='exact' needs obs_counts, obs_sizes "
                                 "and ppp_scale")
            x0_y = exact_inner_sample(x0_hat, sigma_t, obs_counts, obs_sizes,
                                      ppp_scale, lambda_data=lambda_data,
                                      generator=g_inner, return_mean=take_mean,
                                      frame_chunk=exact_frame_chunk)
        elif inner_kind == "tree":
            # PROXIMAL by default, to match the voxel arm's Newton data step.
            if tree is None:
                raise ValueError("inner_kind='tree' needs a TreePoissonLikelihood")
            x0_y = langevin_prox_tree(x0_hat, sigma_t, tree, lr, num_mcmc,
                                      lambda_data=lambda_data, clip_x=clip_x,
                                      generator=g_inner)
        elif inner_kind == "tree_score":
            if tree is None:
                raise ValueError("inner_kind='tree_score' needs a TreePoissonLikelihood")
            x0_y = langevin_tree(x0_hat, sigma_t, tree, lr, num_mcmc,
                                 lambda_data=lambda_data, clip_x=clip_x,
                                 generator=g_inner)
        else:
            if data_score_fn is None:
                raise ValueError("inner_kind='explicit' needs data_score_fn(x)")
            x0_y = langevin_explicit(x0_hat, sigma_t, data_score_fn, lr, num_mcmc,
                                     lambda_data=lambda_data, clip_x=clip_x,
                                     generator=g_inner)

        if isinstance(step_log, list):
            step_log.append(dict(step=step, t=t_cur, sigma=sigma_t, lr=lr,
                                 inner_kind=inner_kind, num_mcmc=int(num_mcmc),
                                 lambda_data=float(lambda_data),
                                 readout="mean" if (take_mean and inner_kind == "exact")
                                         else "draw"))
        if step < num_annealing - 1:
            x_t = x0_y + float(sigma_arr[int(t_indices[step + 1])]) * _randn_like(x0_y, g_outer)
        else:
            x_t = x0_y
        if (step % snapshot_every == 0) or (step == num_annealing - 1):
            traj.append((step, t_cur, sigma_t, x0_y.detach().cpu().clone()))
        if verbose and (step % max(1, num_annealing // 10) == 0):
            chain = ("inner=exact (no chain)" if inner_kind == "exact" else
                     f"lr={lr:.2e} tau/s^2={num_mcmc*lr/sigma_t**2:.2e}")
            print(f"  step {step:3d} t={t_cur:4d} sigma={sigma_t:8.3f} {chain} "
                  f"x0_y=[{x0_y.min():+.3f},{x0_y.max():+.3f}]", flush=True)
    return x0_y, traj


# ---------------------------------------------------------------------------
# 7. EXACT inner sampler -- the inner target is separable, so do not chain it
# ---------------------------------------------------------------------------
# The 2026-09-12 audit (scratchpad daps_inner_mixing_audit.py) ran every kernel
# above against the exact 1-D law of the inner target. Legacy explicit never
# left the anchor; uncapped prox (round 1) put every voxel on a rail; capped
# lam=0.1 (round 2) was ~faithful; and NO feasible Langevin setting samples a
# HIT voxel at sigma >= 2: the Poisson curvature at a hit is kappa^2*mu ~ 21 in
# box units, the split scheme's bias is O(lr*L), so lr <= 0.005 and M >= 300
# (sigma 0.7) to ~2000 (sigma 2) -- tens of GPU-hours of full-volume proxes.
#
# But the target is a PRODUCT of 1-D laws indexed by (count, trials, anchor).
# A 1-D law is sampled exactly by inverse CDF. So: one quantile table per
# distinct (count, trials) pair over an anchor grid, a bilinear lookup per
# voxel. No step size, no mixing, exact truncation to the training box.
# Likelihood is the certified arms' contract: y ~ Binomial(M, 1-exp(-N)),
# N = ppp_scale*exp(kappa(x+1)), dark 0, b_offset 0 (gradient == sploc's
# spad_separable_gradient to 1.6e-16, tests/test_daps_exact_inner.py).
def binomial_loglik(x, counts, trials, ppp_scale, log_flux_max=math.log(1e4)):
    """log p(y | x) up to x-independent constants; broadcasts."""
    kappa = log_flux_max / 2.0
    n = ppp_scale * torch.exp(kappa * (x + 1.0))           # expected photons / frame
    p = -torch.expm1(-n)                                    # detection prob / frame
    return counts * torch.log(p.clamp_min(1e-300)) + (trials - counts) * (-n)


@torch.no_grad()
def exact_inner_sample(x0_hat, sigma_t, counts, trials, ppp_scale, *, lambda_data=1.0,
                       log_flux_max=math.log(1e4), n_a=513, n_x=4097, n_u=2049,
                       z_max=8.0, generator=None, table_dtype=torch.float64,
                       return_mean=False, frame_chunk=64):
    """x ~ p(y|x)^lam N(x; x0_hat, sigma_t^2) 1[-1,1], per voxel, by inverse CDF.

    x0_hat: [B,T,H,W] anchor. counts: [B,T,H,W] or [T,H,W]. trials: [T] or
    broadcastable to x0_hat. Returns a tensor shaped and typed like x0_hat.

    ``return_mean=True`` returns the per-voxel posterior MEAN of the same law
    instead of a draw. Use it on the FINAL annealing level, where a draw leaves
    N(0, sigma_final^2) white speckle in the returned estimate (sd 0.044 in x =
    +-22% flux at t=9) with no further renoise-and-denoise to remove it. A
    single level's conditional mean is NOT an ensemble mean over runs; report
    the two separately.

    TAIL FIX (2026-09-12). The quantile table is tabulated on a GAUSSIAN-spaced
    u-grid, u = Phi(z) for z uniform on [-z_max, z_max] plus the exact endpoints
    0 and 1. With the previous uniform u-grid (spacing 1/1024) the first and last
    cells interpolated linearly between the box edge (u=0 -> x=-1, u=1 -> x=+1)
    and the +-3.1 sd quantile, so 2/1024 of all voxels per level were placed on a
    uniform ramp to the rails: ~1.2e5 voxels per 1024x256x256 volume per level,
    displaced by up to 1.7 x-units (2400x in flux) at sigma_t = 0.044 --
    sampler-made salt and pepper. Measured at sigma_t=0.044, anchor 0.69, y=0:
    sd 0.054 against the target's 0.044 and P(|x-mean| > 5 sd) = 9e-4 against
    3e-7. The suite's W1 < 0.005 assertion passes either way (box width 2), so
    the tails are asserted directly in tests/test_daps_exact_inner_tails.py.
    With the Gaussian grid the worst moment error is 0.7% of the target sd over
    sigma_t in [0.044, 3.4], anchors {0.3, 0.69, 0.95} and counts {0, 1, 2}.

    MEMORY. The tables are small but the per-voxel work is not: an f64 array over
    a 1024x256x256 volume is 0.5 GiB and this needs three of them (anchor, u,
    key). Work is therefore chunked over ``frame_chunk`` frames of the time axis
    (64 frames = 4.2M voxels = 34 MiB per f64 array), with the per-(count,trials)
    quantile tables cached across chunks. Lower frame_chunk if the sampler is
    sharing a card with the 552M spatial prior.
    """
    dev = x0_hat.device
    counts = torch.as_tensor(counts, device=dev)
    if counts.ndim == x0_hat.ndim - 1:
        counts = counts.unsqueeze(0)
    counts = torch.broadcast_to(counts, x0_hat.shape)
    trials = torch.as_tensor(trials, device=dev, dtype=table_dtype)
    if trials.ndim == 1:
        trials = trials.view(1, -1, *([1] * (x0_hat.ndim - 2)))
    trials = torch.broadcast_to(trials, x0_hat.shape)
    if float(trials.max()) >= 1e6:
        raise ValueError("exposure >= 1e6 frames/bin collides with the (count, trials) "
                         "table key; raise the key scale before using it")

    a_grid = torch.linspace(-1.0, 1.0, n_a, device=dev, dtype=table_dtype)
    x_grid = torch.linspace(-1.0, 1.0, n_x, device=dev, dtype=table_dtype)
    # Gaussian-spaced probabilities: dense where the quantile function is steep.
    z_grid = torch.linspace(-float(z_max), float(z_max), n_u - 2, device=dev,
                            dtype=table_dtype)
    u_grid = torch.cat([torch.zeros(1, device=dev, dtype=table_dtype),
                        0.5 * (1.0 + torch.erf(z_grid / math.sqrt(2.0))),
                        torch.ones(1, device=dev, dtype=table_dtype)])
    tether = -0.5 * (x_grid[None, :] - a_grid[:, None]) ** 2 / float(sigma_t) ** 2

    out = torch.empty_like(x0_hat)
    table = {}                                   # key -> (n_a, n_u) quantiles or (n_a,) means

    def _table(k):
        if k in table:
            return table[k]
        yv = math.floor(k / 1e6); mv = k - yv * 1e6          # exact decode of the key
        ll = float(lambda_data) * binomial_loglik(x_grid, yv, mv, ppp_scale, log_flux_max)
        lp = ll[None, :] + tether
        w = torch.exp(lp - lp.max(dim=1, keepdim=True).values)
        if return_mean:
            table[k] = (w * x_grid[None, :]).sum(dim=1) / w.sum(dim=1)
            return table[k]
        cdf = torch.cumsum(w, dim=1); cdf = cdf / cdf[:, -1:]
        idx = torch.searchsorted(cdf.contiguous(),
                                 u_grid[None, :].expand(n_a, -1).contiguous()).clamp(1, n_x - 1)
        c1, c0 = cdf.gather(1, idx), cdf.gather(1, idx - 1)
        frac = ((u_grid[None, :] - c0) / (c1 - c0).clamp_min(1e-30)).clamp(0, 1)
        table[k] = x_grid[idx - 1] + frac * (x_grid[idx] - x_grid[idx - 1])
        return table[k]

    T = x0_hat.shape[1] if x0_hat.ndim >= 2 else 1
    step = max(1, int(frame_chunk)) if x0_hat.ndim >= 2 else T
    for t0 in range(0, T, step):
        t1 = min(t0 + step, T)
        sl = (slice(None), slice(t0, t1)) if x0_hat.ndim >= 2 else (slice(None),)
        anchor = x0_hat[sl].to(table_dtype).clamp(-1.0, 1.0)
        cnt_c, tri_c = counts[sl], trials[sl]
        key = cnt_c.to(table_dtype) * 1e6 + tri_c
        buf = torch.empty_like(x0_hat[sl])
        ai = (anchor + 1.0) * 0.5 * (n_a - 1)
        a0_all = ai.floor().long().clamp(0, n_a - 2)
        fa_all = ai - a0_all
        u = (None if return_mean else
             torch.rand(anchor.shape, device=dev, generator=generator, dtype=table_dtype))
        for k in torch.unique(key).tolist():
            sel = key == k
            tab = _table(k)
            a0, fa = a0_all[sel], fa_all[sel]
            if return_mean:
                val = tab[a0] * (1 - fa) + tab[a0 + 1] * fa
            else:
                uv = u[sel]
                # locate u in the (non-uniform) Gaussian-spaced grid
                u0 = (torch.searchsorted(u_grid.contiguous(), uv.contiguous()) - 1
                      ).clamp(0, n_u - 2)
                fu = ((uv - u_grid[u0]) / (u_grid[u0 + 1] - u_grid[u0]).clamp_min(1e-300)
                      ).clamp(0, 1)
                val = (tab[a0, u0] * (1 - fa) * (1 - fu) + tab[a0, u0 + 1] * (1 - fa) * fu
                       + tab[a0 + 1, u0] * fa * (1 - fu) + tab[a0 + 1, u0 + 1] * fa * fu)
            buf[sel] = val.to(buf.dtype)
        out[sl] = buf
    return out
