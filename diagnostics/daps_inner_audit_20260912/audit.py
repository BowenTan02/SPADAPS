"""Reproduce the DAPS kernel audit without checkpoints or GPU reconstruction.

Run with the Python environment used by improved-diffusion-main:
    python diagnostics/daps_inner_audit_20260912/audit.py

Production files are only imported. The independent reference uses SciPy
trapezoidal quadrature, not the sampler's rectangular-grid CDF. Synthetic
counts are valid integers; the reference exposure is 98 SPAD trials, with
0.098 incident photons/bin at normalized log flux x=0.5.
"""
import argparse
import contextlib
import csv
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import sys
import tempfile
import time
from unittest.mock import patch

import numpy as np
from scipy.integrate import cumulative_trapezoid, trapezoid
import torch


HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--repo', type=Path, default=HERE.parents[2] / 'improved-diffusion-main')
parser.add_argument('--samples', type=int, default=8192)
parser.add_argument('--output', type=Path, default=HERE / 'results')
args = parser.parse_args()
sys.path.insert(0, str(args.repo))
import daps_spad_upgrades as U
from sp_localization_suite import spad_data_step_poisson_3d_newton as newton
from sp_localization_suite import spad_separable_gradient as gradient

torch.set_num_threads(1)
KAPPA = math.log(1e4) / 2
TRIALS = 98
PPP = 0.098 / (TRIALS * math.exp(KAPPA * 1.5))
N = args.samples
shape = (1, 1, 1, N)
args.output.mkdir(parents=True, exist_ok=True)
ab = np.cumprod(1 - np.linspace(1e-4, 0.02, 1000))
sigmas = np.sqrt((1 - ab) / ab)
tidx = np.linspace(999, 0, 101).astype(int)


def reference(anchor, sigma, count, lam=1., n_grid=100001):
    x = np.linspace(-1, 1, n_grid)
    photons = PPP * np.exp(KAPPA * (x + 1))
    loglik = count * np.log(-np.expm1(-photons)) - (TRIALS - count) * photons
    logp = lam * loglik - (x - anchor)**2 / (2 * sigma**2)
    p = np.exp(logp - logp.max())
    p /= trapezoid(p, x)
    cdf = cumulative_trapezoid(p, x, initial=0)
    cdf /= cdf[-1]
    mean = trapezoid(x * p, x)
    std = np.sqrt(trapezoid((x - mean)**2 * p, x))
    rail = np.interp(-.99, x, cdf) + 1 - np.interp(.99, x, cdf)
    return dict(x=x, cdf=cdf, mean=mean, std=std, rail=rail)


def compare(s, ref):
    s = np.sort(s.flatten().double().numpy())
    ecdf = np.searchsorted(s, ref['x'], side='right') / len(s)
    return dict(mean=float(s.mean()), std=float(s.std()),
                rail=float((np.abs(s) > .99).mean()),
                boundary_atoms=float((np.abs(s) == 1).mean()),
                w1=float(trapezoid(np.abs(ecdf - ref['cdf']), ref['x'])),
                ks=float(np.max(np.abs(ecdf - ref['cdf']))))


def kernel(anchor, sigma, count, lam, kind, lr, steps, clip=True, seed=1729):
    a = torch.full(shape, anchor, dtype=torch.float64)
    y = torch.full(shape, count, dtype=torch.float64)
    m = torch.tensor([float(TRIALS)], dtype=torch.float64)
    torch.manual_seed(seed)
    if kind == 'cdf':
        return U.exact_inner_sample(a, sigma, y, m, PPP, lambda_data=lam)
    if kind == 'explicit':
        score = lambda x: -gradient(x, x, y, m.view(1,1,1,1), 0.,
                                    ppp_scale=PPP, dark_count=0.)
        return U.langevin_explicit(a, sigma, score, lr, steps, lambda_data=lam, clip_x=clip)
    prox = lambda z, rho: newton(z, y, m, rho, PPP, dark_count=0., n_iter=32)
    return U.langevin_prox_newton(a, sigma, prox, lr, steps, lambda_data=lam, clip_x=clip)


rows = []
def add_case(name, anchor, step, count, lam, kind, lr, steps, clip=True):
    sigma = float(sigmas[tidx[step]])
    ref = reference(anchor, sigma, count, lam)
    start = time.perf_counter()
    s = kernel(anchor, sigma, count, lam, kind, lr, steps, clip)
    row = dict(name=name, anchor=anchor, step=step, sigma=sigma, count=count,
               lam=lam, kind=kind, lr=lr, steps=steps, clip=clip,
               target_mean=ref['mean'], target_std=ref['std'], target_rail=ref['rail'],
               seconds=time.perf_counter()-start, **compare(s, ref))
    rows.append(row)
    print(f"{name:20} step={step:2} y={count} lam={lam:g} "
          f"mean={row['mean']:+.4f}/{ref['mean']:+.4f} "
          f"sd={row['std']:.4f}/{ref['std']:.4f} W1={row['w1']:.4f} "
          f"atoms={row['boundary_atoms']:.3f}", flush=True)


for step in [0, 60, 80, 99]:
    sg = float(sigmas[tidx[step]])
    for y in [0, 1]:
        add_case('legacy_scheduled', .5, step, y, 1., 'explicit', .001 + step/99*(.00001-.001), 40)
        add_case('uncapped_prox', .5, step, y, 1., 'prox', .2*sg**2, 10)
        add_case('capped_prox', .5, step, y, 1., 'prox', min(.2*sg**2, .1), 10)
        add_case('cdf', .5, step, y, 1., 'cdf', 0., 0)
        if step in [0, 80]:
            add_case('capped_tempered', .5, step, y, .1, 'prox', min(.2*sg**2, .1), 40)
            add_case('cdf_tempered', .5, step, y, .1, 'cdf', 0., 0)

# Isolate stationary bias from burn-in; also test whether pre-prox clipping
# really is redundant with the bounded Newton solver.
for y in [0, 1]:
    for m in [100, 1000]:
        add_case(f'long_cap_M{m}', .5, 60, y, 1., 'prox', .1, m)
    add_case('no_preclip', .5, 60, y, 1., 'prox', .1, 1000, False)
    add_case('small_step', .5, 60, y, 1., 'prox', .005, 1000)

with (args.output / 'kernels.csv').open('w') as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)

# Quantile interpolation error separately from finite Monte Carlo error.
qrows = []
for sigma in [157.407, .719, .043587, .01]:
    for anchor in [-1., -.993, .503, .997, 1.]:
        for y in [0, 1, 3]:
            ref = reference(anchor, sigma, y)
            a = torch.full(shape, anchor, dtype=torch.float64)
            cnt = torch.full(shape, float(y), dtype=torch.float64)
            gen = torch.Generator().manual_seed(912)
            uniforms = torch.rand(shape, dtype=torch.float64, generator=gen).numpy().flatten()
            gen.manual_seed(912)
            sample = U.exact_inner_sample(a, sigma, cnt, TRIALS, PPP, generator=gen)
            qref = np.interp(uniforms, ref['cdf'], ref['x'])
            error = np.abs(sample.numpy().flatten() - qref)
            qrows.append(dict(sigma=sigma, anchor=anchor, count=y,
                              mean_quantile_error=float(error.mean()),
                              p99_quantile_error=float(np.quantile(error,.99)),
                              max_quantile_error=float(error.max()), **compare(sample, ref)))
with (args.output / 'quantiles.csv').open('w') as f:
    writer = csv.DictWriter(f, fieldnames=qrows[0].keys()); writer.writeheader(); writer.writerows(qrows)

# Run the real runner with tiny synthetic inputs and zero-cost prior wrappers.
# Only the expensive prior and final scoring are stubbed. The actual outer
# loop and explicit inner sampler execute. The spy records executed lr.
import run_daps_sweep as runner
def runner_probe(seed, external_seed, legacy=True):
    recorded = []
    outputs = []
    original = U.langevin_explicit
    def spy(a, sigma, score, lr, n, **kw):
        recorded.append(float(lr))
        return original(a, sigma, score, lr, n, **kw)
    argv = ['run_daps_sweep.py', '--inner-kind', 'explicit', '--num-annealing', '3',
            '--vol-t','1','--vol-h','4','--vol-w','4','--num-mcmc','1',
            '--seed',str(seed)] + (['--legacy-lr'] if legacy else [])
    torch.manual_seed(external_seed)
    with tempfile.TemporaryDirectory() as tmp:
        argv += ['--out-root', tmp]
        with patch.object(sys, 'argv', argv), \
             patch.object(runner.rp, 'load_ground_truth', return_value=np.full((1,4,4), 1000., np.float32)), \
             patch.object(runner.rp, 'load_or_simulate_observation', return_value=(np.zeros((1,4,4),np.float32), np.array([98.],np.float32), PPP, 'audit')), \
             patch.object(runner.rp, 'Models', return_value=None), \
             patch.object(runner.rp, 'eps_from_2d_prior', side_effect=lambda _, x, t: torch.zeros_like(x)), \
             patch.object(runner.rp, 'eps_from_1d_prior', side_effect=lambda _, x, t: torch.zeros_like(x)), \
             patch.object(runner, '_score_and_write', side_effect=lambda a,t,o,x,*rest,**kw: outputs.append(x.clone())), \
             patch.object(U, 'langevin_explicit', side_effect=spy), \
             contextlib.redirect_stdout(io.StringIO()):
            runner.main()
    return recorded, outputs[0]

lr1, out1 = runner_probe(10001, 111)
_, out2 = runner_probe(10001, 222)
_, out3 = runner_probe(10002, 111)
checks = dict(executed_legacy_lrs=lr1, requested_legacy_lrs=[.001,.000505,.00001],
              same_cli_seed_different_global_rng_equal=bool(torch.equal(out1,out2)),
              different_cli_seed_same_global_rng_equal=bool(torch.equal(out1,out3)),
              same_cli_seed_max_difference=float((out1-out2).abs().max()),
              imported_sampler=str(Path(U.__file__).resolve()))
checks['source_sha256'] = {
    str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
    [Path(U.__file__), Path(runner.__file__), args.repo / 'sp_localization_suite.py']}
checks['config'] = dict(samples=N, trials=TRIALS, ppp_scale=PPP, anchor_reference=.5,
                        expected_incident_photons_at_reference=.098, torch=torch.__version__)
(args.output / 'runner_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
print(json.dumps(checks,indent=2), flush=True)
print('Worst quantile errors:', sorted(qrows,key=lambda r:r['mean_quantile_error'])[-3:], flush=True)
