<!-- SPDX-License-Identifier: MIT -->
# Validation-only optimization diagnostics

Six completed full-* runs. Held-out metrics and predictions were not deserialized or used; existing test-derived generalization_gap excluded. No training/source edits.

## Curves and matched train-validation gaps

Accuracy values below are site-macro percentages; gaps are percentage points. Best rounds are retrospective diagnostics, not established early-stopping improvements.

| Run | Best val % | Best rounds | Final val % | Best-final pp | Train-val gap pp | CE first → final |
|---|---:|---|---:|---:|---:|---:|
| full-fedavg-s42 | 69.01 | 4,5 | 63.92 | 5.09 | 35.24 | 0.4941 → 0.0345 |
| full-fedavg-s43 | 73.18 | 2,3 | 71.07 | 2.11 | 28.93 | 0.4682 → 0.0181 |
| full-fedavg-s44 | 77.94 | 10,11,13 | 74.96 | 2.98 | 25.04 | 0.5662 → 0.0513 |
| full-fedbrain-s42 | 68.69 | 13 | 67.82 | 0.87 | 27.18 | 0.6862 → 0.0976 |
| full-fedbrain-s43 | 71.07 | 50 | 71.07 | 0.00 | 27.04 | 0.6781 → 0.0945 |
| full-fedbrain-s44 | 76.15 | 33,36,40 | 70.42 | 5.74 | 29.58 | 0.6901 → 0.0877 |

## Method means
{
  "fedbrain": {
    "val_best": 0.7196969696969696,
    "val_final": 0.6976911976911977,
    "best_minus_final_pp": 2.2005772005771953,
    "macro_train_val_gap_pp": 27.93193843193843,
    "first10_mean": 0.5774110149110149,
    "last10_mean": 0.6788119288119288,
    "val_final_sd": 0.017180203318601203
  },
  "fedavg": {
    "val_best": 0.7337662337662337,
    "val_final": 0.6998556998556998,
    "best_minus_final_pp": 3.3910533910533966,
    "macro_train_val_gap_pp": 29.736652236652244,
    "first10_mean": 0.676400913900914,
    "last10_mean": 0.6954365079365079,
    "val_final_sd": 0.055984922754881226
  }
}

## Alignment scale
Weighted alignment / CE first, median, final, maximum (loss scale, not gradient scale):
- full-fedbrain-s42: 0.377, 0.853, 2.555, 2.555. Raw first/final 2.5841, 2.4938
- full-fedbrain-s43: 0.417, 0.796, 2.399, 2.501. Raw first/final 2.8298, 2.2670
- full-fedbrain-s44: 0.434, 0.930, 2.841, 2.841. Raw first/final 2.9963, 2.4906

## Selection
0/18 unique site fits converged; all ran 40 steps. All objectives monotone: True. FedAvg reuses same-seed FedBrain training-only features; these are 18 unique fits, not 36 independent fits.
{
  "unique_site_fits": 18,
  "converged": 0,
  "all_monotone": true,
  "uncapped_nonzero_range": [
    5395,
    5828
  ],
  "decrease_pct_range": [
    22.477307639031462,
    29.72841702809072
  ],
  "last_step_pct_range": [
    0.28082332595977577,
    0.5468197958837986
  ],
  "last10_pct_range": [
    3.1539100673593117,
    5.770366237807172
  ]
}

Final selection is always 512/6670 = 7.68%; uncapped union is 6670. Thus the explicit cap, not convergence to a sparse 512-dimensional solution, determines feature count.

## Noise/clipping
Clipping applies to the full local shared parameter vector, not the update. Expected RMS noise L2 assumes independent Laplace draws and equal six-client averaging.
- full-fedavg-s42: n=4113; final postaggregation norm=7.1766, clip=10.0; per-client/aggregate noise RMS L2=0.9070/0.3703; aggregate RMS/final norm=5.16%.
- full-fedavg-s43: n=4113; final postaggregation norm=7.2812, clip=10.0; per-client/aggregate noise RMS L2=0.9070/0.3703; aggregate RMS/final norm=5.09%.
- full-fedavg-s44: n=4113; final postaggregation norm=6.6448, clip=10.0; per-client/aggregate noise RMS L2=0.9070/0.3703; aggregate RMS/final norm=5.57%.
- full-fedbrain-s42: n=4113; final postaggregation norm=4.6656, clip=10.0; per-client/aggregate noise RMS L2=0.9070/0.3703; aggregate RMS/final norm=7.94%.
- full-fedbrain-s43: n=4113; final postaggregation norm=4.9536, clip=10.0; per-client/aggregate noise RMS L2=0.9070/0.3703; aggregate RMS/final norm=7.47%.
- full-fedbrain-s44: n=4113; final postaggregation norm=4.7487, clip=10.0; per-client/aggregate noise RMS L2=0.9070/0.3703; aggregate RMS/final norm=7.80%.

Final postaggregation norms cannot establish historical preclip norms or clipping frequency. No benefit from changing noise is inferred.

## Bounded single-variable experiments (not run)

A fixed shorter horizon is not recommended from these mixed curves: one FedBrain seed is best at round 50.
1. **alignment_weight: 0.1 → 0.03**. Reduce weighted alignment pressure relative to CE; loss magnitude does not establish gradient dominance.
2. **sparse_max_iter: 40 → 120**. Test whether additional train-only selector optimization changes capped feature ranking and validation; retain feature cap 512.

Each candidate individually versus existing baseline with all other variables fixed and seeds 42,43,44. WIN: paired mean final validation site-macro accuracy >= baseline +0.01 and >=2/3 seeds nonnegative. KILL: no mean validation gain; stop this candidate. Intermediate result otherwise inconclusive. No held-out evaluation during tuning; no improvement claimed.

The sparse-iteration candidate additionally requires lower selector objective at every site; objective improvement alone is not accuracy improvement.
