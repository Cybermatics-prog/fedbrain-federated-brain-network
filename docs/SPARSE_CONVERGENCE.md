<!-- SPDX-License-Identifier: MIT -->
# Eq. 11 sparse optimization: equivalent speedup and training-set convergence diagnosis

## Proposition of this round and pre-registration

Only two implementation-level propositions are examined; no validation/test metric is read or evaluated:
1. The time-segment-grouped float64 BLAS implementation and the original `einsum + add.at` implementation agree to within absolute error `1e-10` on the summed logistic loss, the gradients, the proximal updates, and the 40-step trajectory, and the support set must be exactly identical on the deterministic test cases.
2. With the original main experiment's training subjects, training mean/std, 8 equally spaced relative time segments per subject, `lambda_aug=.01`, and `coupling=1` fixed, the 400-step objective value must not exceed the 40-step objective value (tolerance `1e-8`). Exact zero coefficients, support-set Jaccard, Jaccard after 512 truncation, and the original stopping criterion are also reported; convergence at 400 steps is not presupposed.

As source data, only the `train` list of the `../runs/full-fedbrain-s42/split.json`, the saved training preprocessing arrays, and the validated FC cache of these training subjects may be read; no predictions, result metrics, or checkpoints are read. All probes cap the CPU thread pool at 4.

## The reference formulation and the semantics that stay unchanged

- The Eq. 11 formulation and the Algorithm 1 semantics used here are described in [RELATED_WORK.md](RELATED_WORK.md). They use the sum of classification losses over each time segment, an L1 penalty on the time-segment coefficients, and an L1 coupling between the shared and time-segment coefficients. This speedup preserves:

- **sum BCE, not mean BCE**; defaults `lambda_aug=.01`, `coupling=1`, learning-rate cap `.1`;
- the initial 8-step independent L1 warm start, the exact shared-coordinate median, and the exact dual-L1 proximal map;
- the original Frobenius Lipschitz-upper-bound step size, the parameter-change `<=1e-6` stopping criterion, and the default 40 steps;
- no feature cap in the module by default; cross-site nonzero support sets are unioned, and the 512 cap is separately and explicitly labeled as an approximation.

Each segment's training matrix is laid out as a contiguous array only once, and each gradient uses `X_t @ w_t` and `X_t.T @ residual_t`. The `[n,d]` `weights[encoded]` and `X*residual` are no longer materialized on every iteration. Grouping changes the floating-point reduction order but not the mathematical objective.

The original diagnostic's `converged` is a parameter-change stopping rule, not a certificate of global optimality; the newly added relative final-step objective change is likewise not an optimality gap. After unioning all sites, the support set may still be close to or equal to 6670, so the artificial 512 truncation must not be described as sparsity that L1 obtains automatically.

## Numerical-equivalence verification

[The new offline tests](<../tests/test_sparse_convergence.py>) pass 5/5 (0.206 seconds):
- the old/new gradients, sum loss, and dual-L1 proximal single step agree;
- over the full default 40 steps, the shared/segmented coefficients, bias, and objective trajectory agree within error `1e-10`, and the support sets are exactly identical;
- duplicating training rows doubles the loss and the gradient, preventing an accidental change to a mean loss;
- large-magnitude logits remain finite, the 400-step trajectory does not increase, and true zero coefficients are preserved.

## Repeatable training-set probe

Run the [probe script](<../scripts/sparse_probe.py>):

```powershell
.\.venv\Scripts\python.exe scripts/sparse_probe.py --threads 4
```

It always reads the original full training split and the training standardization; it compares 40/400 steps only on the same 8-segment grid. The script verifies that the FC cache corresponds to the original time series by SHA256, and records input-file hashes and training-ID hashes; it does not recompute/rewrite the FC cache. The completed artifacts are the `../artifacts/sparse_probe.json` and the `../artifacts/sparse_probe_summary.json`.

## Real training-set results: faster, but 400 steps still do not converge

292 original training subjects; all 12 site fits took **19.72 seconds** in total (CPU threads 4, excluding cache preparation). On the real NYU matrix, the gradient+loss kernel is **10.63x** faster at the median of 5 timings; this is not the speedup of a whole training run. The maximum absolute error between the old/new gradients is `1.78e-14`, and the loss error is 0.

|Site|Training subjects|40-step objective|400-step objective|Objective decrease|Nonzero shared coefficients 40→400|Relative objective change at end of 400 steps|
|---|---:|---:|---:|---:|---:|---:|
|NYU|109|423.9138|227.4079|46.36%|5646→5608|5.61e-4|
|UM|56|200.6508|127.4505|36.48%|5450→5449|3.08e-4|
|USM|42|141.6808|97.2015|31.39%|5480→5478|2.68e-4|
|UCLA|40|140.2327|98.1331|30.02%|5755→5755|2.65e-4|
|KKI|20|64.7937|52.2901|19.30%|5681→5681|1.96e-4|
|Yale|25|80.1727|62.3458|22.24%|5579→5579|2.12e-4|

- Original parameter-change convergence flag: **0/6** at 40 steps, still **0/6** at 400 steps; no 400-step objective trajectory increases.
- Cross-site nonzero union: **6670→6670**; no global dimensionality reduction occurs. The 512 dimensions still depend entirely on an explicit external truncation.
- The cross-site 512-mask Jaccard is only **0.4402**, indicating that the ranking changes markedly with the optimization budget; this number is not an accuracy.
- The 5 new numerical unit tests pass locally; the Lead confirmed that the expanded full suite passes **37/37** unit tests (2.037 seconds).

Conclusion: what this round fixes is computational cost and diagnostic visibility; **it does not fix the mathematical convergence, and it still less implies any accuracy improvement**. The current 400-step candidate serves only as a control with a bounded optimization budget.

## Analytic counterexample: blockwise proximal stagnation is not joint optimality

`../artifacts/sparse_probe_counterexample.json` is also available, and it uses no ABIDE or held-out data at all:

- Two exactly identical time segments, 20 samples each; the one-dimensional input alternates between `-1,+1`, `y=1[x>0]`, the two classes are balanced, and the initial bias is 0.
- With `lambda=.01, coupling=1, max_steps=400` fixed, along the direction in which all segments and the shared parameters jointly take `w>0`, the coupling penalty is always 0.
- The joint objective reduces to `F(w)=40*log(1+exp(-w)) + .02*w`; the analytic minimizer is `w*=log(20/.01-1)=7.6004023345`, with `F(w*)=.1720130484`.
- The current blockwise algorithm reports parameter-change convergence already at step 94, yet `w=2.9339528878` and `F=2.1314878055`, and the derivative along the jointly consistent movement direction is still **−2.0000178492**. A clear descent direction therefore still exists.

Cause: when the shared parameters are fixed while the time segments are updated, the kink of `|theta_t-theta|` stops the segments from moving together; then fixing the segments while updating the shared median likewise cannot perform a jointly consistent movement. An exact single-block proximal map does not automatically resolve cross-block nonsmooth coupling. This is a limitation of the existing blockwise stopping criterion, not an error introduced by the BLAS speedup.

The candidate search frozen for this round changes no further algorithm and extends no budget. If work continues, a joint proximal, primal-dual, or ADMM solver should be pre-registered separately, and it must first pass this analytic counterexample and the KKT/duality-residual checks; raising the budget to 400 steps must not be claimed as having solved Eq. 11.
