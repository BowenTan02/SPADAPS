"""Deterministic tail test and an isolated prototype for a resolved CDF table.

This does not replace production code. It evaluates 200,000 evenly spaced
probabilities, including the low-probability events missed by average W1 tests.
The prototype uses trapezoidal CDF integration and extra probability knots in
the tails. It still needs general-input and production-scale validation.
"""
import csv
import json
import math
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
from scipy.integrate import cumulative_trapezoid, trapezoid
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'improved-diffusion-main'))
import daps_spad_upgrades as U

torch.set_num_threads(1)
K = math.log(1e4)/2
PPP = 1e-6
M = 98
N = 200000
uniforms = (np.arange(N) + .5) / N


def cdf_for(a, sigma, y, x):
    n = PPP * np.exp(K*(x+1))
    lp = y*np.log(-np.expm1(-n)) - (M-y)*n - (x-a)**2/(2*sigma**2)
    p = np.exp(lp-lp.max())
    cdf = cumulative_trapezoid(p, x, initial=0)
    return cdf/cdf[-1]


def tail_resolved_quantiles(a, sigma, y, u, n_a=257, n_x=4097):
    ag = np.linspace(-1, 1, n_a)
    xg = np.linspace(-1, 1, n_x)
    tails = np.geomspace(1e-12, .01, 256)
    ug = np.unique(np.r_[0., tails, np.linspace(.01,.99,513), 1-tails[::-1], 1.])
    ai = (a+1)*.5*(n_a-1)
    i = min(int(math.floor(ai)), n_a-2)
    f = ai-i
    q = []
    for av in ag[i:i+2]:
        cdf = cdf_for(av, sigma, y, xg)
        qt = np.interp(ug, cdf, xg)
        qt[0], qt[-1] = -1., 1.
        q.append(np.interp(u, ug, qt))
    return (1-f)*q[0]+f*q[1]


rows = []
ab = np.cumprod(1-np.linspace(1e-4,.02,1000))
schedule = np.sqrt((1-ab)/ab)
last_sigma = float(schedule[np.linspace(999,0,101).astype(int)[99]])
for sigma, a, y in [(last_sigma,.5,0), (last_sigma,.5,1),
                    (last_sigma,.997,1), (.01,.997,1), (.719,.503,1),
                    (157.407,.503,0)]:
    xg = np.linspace(-1,1,200001)
    cdf = cdf_for(a,sigma,y,xg)
    qref = np.interp(uniforms,cdf,xg)
    anchor = torch.full((1,1,1,N),a,dtype=torch.float64)
    counts = torch.full_like(anchor,float(y))
    with patch.object(U.torch,'rand',return_value=torch.from_numpy(uniforms.copy()).reshape(anchor.shape)):
        old = U.exact_inner_sample(anchor,sigma,counts,M,PPP).numpy().flatten()
    fixed = tail_resolved_quantiles(a,sigma,y,uniforms)
    refined = tail_resolved_quantiles(a,sigma,y,uniforms,n_a=513,n_x=8193)
    for name,q in [('reference',qref),('production',old),('prototype',fixed),('refined_prototype',refined)]:
        rows.append(dict(sigma=sigma,anchor=a,count=y,name=name,
                         mean=float(q.mean()),std=float(q.std()),
                         w1=float(np.abs(q-qref).mean()),
                         max_quantile_error=float(np.abs(q-qref).max()),
                         p_below_anchor_minus_point3=float((q<a-.3).mean()),
                         analytic_p_below=float(np.interp(a-.3,xg,cdf))))
    print(json.dumps(rows[-4:],indent=2))

# A Metropolis-adjusted per-coordinate Langevin control: proposals outside
# the box are rejected, not clipped. This is valid only because q factorizes.
# This diagnostic deliberately leaves production kernels unchanged.
def mala(y, sigma, steps=300, h=.05, seed=20260912):
    rng=np.random.default_rng(seed)
    a=.5
    x=np.full(16384,a)
    accepted=0
    def lp(z):
        n=PPP*np.exp(K*(z+1))
        return y*np.log(-np.expm1(-n))-(M-y)*n-(z-a)**2/(2*sigma**2)
    def score(z):
        n=PPP*np.exp(K*(z+1))
        return K*n*(y/np.expm1(n)-(M-y))-(z-a)/sigma**2
    for _ in range(steps):
        g=score(x)
        z=x+h*g+np.sqrt(2*h)*rng.normal(size=x.shape)
        legal=np.abs(z)<=1
        # Evaluate likelihood only in its declared support. The placeholder
        # for illegal proposals is ignored in the accept decision.
        safe=np.where(legal,z,x)
        gz=score(safe)
        logalpha=lp(safe)-lp(x)-((x-safe-h*gz)**2-(safe-x-h*g)**2)/(4*h)
        accept=legal & (np.log(rng.uniform(size=x.shape))<logalpha)
        x=np.where(accept,safe,x)
        accepted+=accept.sum()
    xg=np.linspace(-1,1,100001)
    cdf=cdf_for(a,sigma,y,xg)
    ecdf=np.searchsorted(np.sort(x),xg,side='right')/len(x)
    return dict(count=y,sigma=sigma,steps=steps,lr=h,mean=float(x.mean()),std=float(x.std()),
                acceptance=float(accepted/(steps*len(x))),
                w1=float(trapezoid(np.abs(cdf-ecdf),xg)),boundary_atoms=float((np.abs(x)==1).mean()))

checks=[mala(y,float(schedule[np.linspace(999,0,101).astype(int)[60]])) for y in (0,1)]
(HERE/'results'/'mala_control.json').write_text(json.dumps(checks,indent=2)+'\n')
with (HERE/'results'/'tails.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
print('MALA controls',json.dumps(checks,indent=2))
