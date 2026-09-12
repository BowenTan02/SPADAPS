# DAPS inner sampling audit — 2026-09-12

The existing results do not establish that a correctly sampled DAPS formulation fails on this SPAD problem. The old small-step chain under-explores, while the large-step proximal chain is biased toward bounded data-fitting solutions. Increasing the length of a biased chain does not repair its stationary distribution. The current inverse-CDF alternative also has a rare-tail interpolation defect that can directly generate impulse artifacts.

This audit reads the SPADAPS notebooks and both copies of `daps_spad_upgrades.py`, and tests the production sweep/kernel in the sibling `improved-diffusion-main` checkout at commit `b41b70f5253c3b37565c3df0c391f80229fea1be`. Both module copies were byte-identical when inspected. Source hashes are saved in `results/runner_checks.json`. Production files were not edited. The new files in this directory are diagnostics and an isolated numerical prototype.

## Reproduce

From `/Users/tan583/Documents/Diffusion/SPADAPS`, using the environment with Torch and SciPy:

```sh
python diagnostics/daps_inner_audit_20260912/audit.py
python diagnostics/daps_inner_audit_20260912/tail_probe.py
```

The first script uses the actual `langevin_explicit`, `langevin_prox_newton`, and certified `spad_data_step_poisson_3d_newton`. It independently integrates the target with SciPy trapezoidal quadrature. It runs 8,192 independent chain endpoints per case. The second evaluates 200,000 deterministic, evenly spaced quantiles and a separate coordinatewise MALA implementation. These are CPU conditional-distribution tests, not video reconstructions or measurements of final artifact rates.

The existing test suite was also rerun:

```sh
cd /Users/tan583/Documents/Diffusion/improved-diffusion-main
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=1 python -m pytest -q tests/test_daps_exact_inner.py
```

Result: **10 passed in 4.77 s**. Passing those tests is insufficient for tail accuracy, as demonstrated below.

## The target and the theoretical distinction

The current flat inner target, interpreted on the declared training-domain box, is

\[
q_t(x\mid a,y)\propto\prod_i p(y_i\mid x_i)^\lambda
\exp\left[-\frac{(x_i-a_i)^2}{2\sigma_t^2}\right]\mathbf1_{[-1,1]}(x_i),
\qquad a=\hat x_0(x_t).
\]

For the certified sweep the likelihood is the exact binned SPAD law, despite functions being named “Poisson”:

\[
y_i\sim\mathrm{Binomial}(m_i,1-e^{-n_i}),\quad
n_i=\alpha\exp\{\kappa(x_i+1)\},\quad \kappa=\log(10^4)/2.
\]

The notebook `DAPS_3D_Poisson.ipynb` instead uses the Poisson approximation in its explicit score. Its configured 40-step linear learning-rate anneal really is executed inside that notebook. The “Exact” notebook has additional Gamma-belief/pooled modes; those are distinct targets and were not used as references here.

DAPS Proposition 1 requires a draw from the true conditional `p(x0 | xt,y)`. Its practical Gaussian approximation replaces `p(x0 | xt)` with `N(x0_hat,r_t² I)`; `r_t` is a design choice. The present implementation sets `r_t=sigma_t`. Correctly sampling this surrogate removes inner numerical bias but does not establish exact sampling of the original diffusion posterior. See [DAPS paper, Sections 3.1–3.2](https://arxiv.org/html/2407.01521v3#S3).

Avoiding optimization collapse also does not guarantee that individual posterior draws have fewer visible impulses than a point estimate. Samples contain conditional uncertainty. Report individual draws and a separately labeled ensemble-mean estimator; neither should be mistaken for the other.

## 1. The proximal update is stable without being distributionally correct

Production update, `daps_spad_upgrades.py:195–231`:

\[
z=x+h(a-x)/\sigma^2+\sqrt{2h}\,\xi,
\quad z\leftarrow\mathrm{clip}(z),
\quad x^+=\operatorname{prox}_{h\lambda D}(z).
\]

The certified Newton solver solves that proximal optimization accurately. It does not certify a sampling distribution. At high sigma, the uncapped schedule `h=.2 sigma²` makes the proposal noise enormous and the proximal precision `rho=1/(h lambda)` tiny. After clipping, the prox effectively solves the bounded data-only MLE. Injecting randomness before this solve does not prevent collapse: the solve removes it.

A simple derivation already exposes finite-step bias without clipping. Let anchor precision be `A=1/sigma²`, and let the quadratic data energy have weighted curvature `B`. For the split update, the stationary variance is

\[
\operatorname{Var}_{\mathrm{split}}
=\frac{1}{(A+B)[1+\tfrac h2(B-A)]},
\qquad
\operatorname{Var}_{q}=\frac{1}{A+B},
\]

when the autoregression is stable. With data curvature dominating, implicit treatment suppresses variance. With only the explicit Gaussian anchor, it inflates variance. The bounded SPAD case adds projection bias and boundary atoms.

Fresh production-kernel measurements below use anchor `a=.5`, integer exposure `m=98`, `alpha=1e-6`, one observed count, and `lambda=1`. The expected incident count at the anchor is .098/bin. This is a controlled sparsity-matched example, not a scene-average measurement.

| Inner kernel, at sigma=2.03085 | Mean | Standard deviation | W1 error | Exactly at a boundary |
|---|---:|---:|---:|---:|
| Continuous reference | .7276 | .2372 | 0 | 0% |
| Notebook-style scheduled explicit, M=40 | .5628 | .1726 | .1798 | .24% |
| Uncapped prox, h=.8249, M=10 | .9666 | .0464 | .2390 | 48.8% |
| Capped prox, h=.1, M=10 | .8709 | .1920 | .1433 | 40.1% |
| Capped prox, h=.1, M=1000 | .8720 | .1961 | .1444 | 41.5% |
| Independent coordinatewise MALA control, h=.05, M=300 | .7256 | .2386 | .0026 | 0% |

`W1` is the integrated absolute CDF error in normalized log-flux units; the domain width is 2. The MALA row uses 16,384 independent endpoints and had 60.3% mean acceptance. Its zero-count companion had W1=.0047 and 90.5% acceptance. These are encouraging correctness controls for those strata, not a production-ready universal step schedule.

At sigma=157.4, uncapped prox returned exactly -1 for every zero-count coordinate and +1 for every one-count coordinate in the tested population. The correct continuous reference instead has standard deviations .5487 and .2392. This directly separates MLE collapse from genuine posterior dispersion.

At the last executed inner level, sigma=.04357, the legacy schedule gave standard deviation .0255 instead of .0435 for a one-count coordinate. Even when absolute W1 becomes small, relative dispersion can be substantially wrong.

## 2. Earlier diagnostics were overinterpreted

`travel_report` in `daps_spad_upgrades.py:144` declares mixing from `M h/sigma²`. That is an unconstrained anchor-relaxation heuristic, not a convergence test for the bounded SPAD conditional.

In particular:

- As sigma tends to infinity on a fixed box, a no-data conditional approaches a uniform law with finite variance, not a sigma-wide Gaussian. A properly sampled continuous law has no boundary atoms. A broad Gaussian factor does not imply that every correctly mixed sample must saturate the clamp.
- Low truth correlation after one high-noise inner block need not indicate bad sampling. A reference posterior draw can legitimately differ from the anchor; the outer prior must be evaluated as part of the full algorithm.
- The old capped-versus-uncapped null result demonstrates that that particular cap did not restore reconstruction. It does not demonstrate that either chain sampled its target, nor rule out sampler defects.
- Removing `clip_x` does not make the certified prox unbounded, but pre-prox clipping can still change the transition. The fresh zero-count long-chain comparison changes its mean from -.0963 to -.0697 when that clipping is removed. Neither result passes the reference test. “No-clip is always a no-op” is too strong.

The old “no feasible Langevin setting” statement is also too broad. A corrected coordinatewise kernel can work on this factorized target, as the MALA control demonstrates. Its full-volume runtime and all-strata accuracy remain to be measured.

## 3. The inverse-CDF sampler has a tail defect

`exact_inner_sample`, `daps_spad_upgrades.py:702–730`, builds quantiles on 1,025 uniformly spaced probabilities including 0 and 1, then linearly interpolates between them. The first probability interval has width `1/1024`. When the target is narrow, interpolating between `Q(0)=-1` and an ordinary lower-tail quantile spreads almost .1% of draws across an implausibly long range. Rectangular CDF accumulation also adds a smaller grid offset.

At the actual last inner sigma=.0435705, anchor=.5, and zero counts:

| Quantity | Continuous reference | Current CDF sampler | Isolated prototype |
|---|---:|---:|---:|
| Standard deviation | .043483 | .052076 | .043486 |
| Probability x < anchor-.3 | 3.11e-12 | 8.60e-4 | <5e-6 resolution of deterministic probe |
| W1 against reference quantiles | 0 | .000990 | .00000184 |

A change of -.3 in normalized log flux corresponds to approximately **four times lower flux**. This sampler can therefore introduce rare dark impulses exactly where the final output has no subsequent denoising. The existing W1<.005 assertion passes despite this defect. The .00086 rate is for this homogeneous test stratum, not a prediction of the whole video's final artifact rate.

`tail_probe.py` includes an isolated prototype with trapezoidal integration and geometrically spaced tail probabilities down to 1e-12. A doubled anchor/state grid further checks interpolation error. This resolves the tested tail mechanism, but it is not integrated into the production sampler. A production fix needs counts/exposure/anchor coverage, lambda=.1 and 1, all executed sigma levels, boundary anchors, dtype checks, and explicit rare-event tests. “Exact” should mean numerically validated to a declared tolerance; the table is not algebraically exact.

## 4. Two runner bugs invalidate nominal controls

**Learning-rate selection.** `run_daps_sweep.py:233` creates the requested legacy schedule, and line 258 prints it. The actual call at line 350 never passes this schedule. `sample_daps_3d_v2` always computes `sigma_scaled_lr` at line 613. A three-step invocation with `--legacy-lr` requested `[.001, .000505, .00001]` and actually executed `[.1, .1, .1]`. Wire one schedule function into execution and diagnostics, and test the steps observed by the kernel. The original notebook does not have this wiring defect.

**Sampling seed.** `--seed` is recorded in tags/configs but never passed to the sampler or used to seed its draws. With identical external Torch RNG state, changing CLI seed 10001 to 10002 produced identical output. With CLI seed held fixed and external RNG state changed, output changed. The observation seed has a separate working NumPy path; this finding concerns sampler randomness. Use an explicit generator, and separate initialization, outer re-noising, and inner sampling streams so changing inner step count does not silently change all later outer-noise draws.

There is also an import provenance risk: the runner prepends absolute SPADAPS paths to `sys.path`. On this machine both module copies currently agree. Log the actually imported module path and source hash to prevent future drift between the audited and executed copies.

## What to address before an honest artifact evaluation

1. **Fix execution provenance:** wire the requested schedule, apply sampler seeds, isolate RNG streams, and save actual per-step settings plus code/checkpoint/observation fingerprints.
2. **Establish an accurate flat conditional sampler:** repair CDF tails and verify convergence of the numerical tables. Because the target factorizes, direct one-dimensional sampling is a practical reference. Coordinatewise MALA with out-of-domain rejection is a useful independent check. Do not apply independent accept/reject decisions to a coupled tree target.
3. **Use distributional checks:** conditional means, variances, CDF/PIT, boundary atoms, log-flux and flux tails, and dependence on chain initialization. Test no-data, zero-count, hit, multi-hit, and saturation strata at the actual exposures and sigma levels. For chain methods compare step-size refinement, chain-length refinement, and multiple dispersed starts; acceptable movement size or truth correlation is insufficient.
4. **Measure production cost and memory before the full run:** the present CDF implementation creates whole-volume float64 arrays, long gather indices, and masks, and repeats table/group processing. “A small table” does not imply small peak memory. At 67 million voxels a single float64 array is .5 GiB. Chunk application over voxels and profile the full intended geometry without loading the diffusion models before combining memory budgets. Smaller `num_annealing` alone changes the sigma grid; timing should replay selected production levels.
5. **Run the controlled reconstruction:** first retain the exact physical likelihood, lambda=1, prior fusion, outer schedule, observations, and common outer randomness. Compare corrected conditional draws against the current kernel and against the conditional MAP using the same outer algorithm. The latter is a mechanism control, not a claim that the entire pipeline is conventional DiffPIR. For the flat MAP use the same certified data solver with `rho=1/(lambda sigma²)` and anchor `x0_hat`. Then compare full DAPS and DiffPIR under matched experimental inputs. Treat lambda=.1 as a tempered-target ablation.
6. **Keep sampling validity separate from reconstruction quality:** report samples individually, plus an explicitly labeled ensemble mean if desired; measure artificial impulse tails, contrast/flux bias, motion beyond a static baseline, held-out photon likelihood, and uncertainty across clips and photon draws. More randomness alone is not evidence of valid posterior exploration.

Only after the inner sampler passes should a remaining failure be attributed to the surrogate or prior. One remaining hypothesis is that `r_t=sigma_t` supplies too weak a local anchor at high noise, while the isotropic conditional discards spatial/temporal covariance. Outer denoising still couples voxels, so the factorized inner target does not by itself prove inevitable final shot noise. Calibrate any revised covariance on held-out denoising pairs, or separately evaluate a richer conditional model; changing `r_t`, lambda, pooling, or tree weights changes the target being tested.

Also keep inner PF-ODE accuracy as a separate ablation. The one-step arm uses a Tweedie estimate; multiple steps change the anchor. The multi-step implementation ends at the smallest nonzero training sigma (~.01), and the 100-step outer loop's last inner sigma is ~.04357. These finite-sigma choices should be recorded and checked for sensitivity, not silently treated as exact zero-noise endpoints.

The next scientifically useful GPU run is a validated conditional-sampling baseline with fixed controls. Current chain output, and the uncorrected CDF output, should not decide whether DAPS can reduce shot-noise artifacts.
