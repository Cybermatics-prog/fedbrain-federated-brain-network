# FedBrain Accuracy Optimization Based on the Original Paper

No stable accuracy improvement was obtained in this round: the preregistered >=2 percentage point minimum-effect gate was not met, so the original default model is not replaced.

## Validation Locked First, Test Evaluated Afterwards

Selection time: 2026-11-13T08:34:14.699919+00:00; variant **E** was selected by seed42 validation macro, not by test performance.

|Candidate|Change|Validation macro|Selected round|Test during search|
|---|---|---:|---:|---|
|A|Validation-set round selection|68.69%|13|Not evaluated|
|B|Round selection + paper-inspired batch loss scale|71.07%|11|Not evaluated|
|C|B + refresh alignment target after aggregation|71.12%|25|Not evaluated|
|D|C + L2 regularization 0.01|73.23%|27|Not evaluated|
|E|C + 400-step sparse optimization|73.50%|31|Not evaluated|

## Three-Seed Results for the Locked Recipe

|seed|Round-selection baseline A macro|Candidate macro|Change (percentage points)|Original fixed 50-round macro|
|---|---:|---:|---:|---:|
|42|56.77%|60.62%|+3.85|56.89%|
|43|60.41%|61.31%|+0.90|60.41%|
|44|59.81%|58.14%|-1.67|64.97%|

- Original fixed 50-round macro mean: 60.76%.
- Same validation round-selection baseline A: 59.00%; optimized: 60.02% ± 1.67 percentage points.
- Optimized pooled subject accuracy mean: 63.54%.
- Paired mean change vs A: +1.03 percentage points; 2/3 seeds improved; passes preregistered >=2pp with at least 2/3 improved: False.
- Matched-mask/noise FedAvg with the same best-val round selection: 60.23% macro.

## Paper Basis and Interpretation

- §5.5.1 explicitly introduces early stopping: this fixes the problem of saving only the final round and discarding a model that validated better.
- The paper-inspired minibatch convention (sumCE_batch+DA_batch)/actual batch adjusts the relative scale of classification/alignment; it is not a strict equivalence to the full-data Eq16 or a restoration of the authors' source code.
- After aggregation/broadcast, the train hidden means are recomputed with the current shared encoder, avoiding alignment to the coordinates of the previous round's local encoder.
- L2 regularization 0.01 is an anti-overfitting extension; sparse 40→400 is a convergence diagnostic and does not claim to be a hyperparameter fully specified in the original paper.

## Conclusion Boundaries

- Candidates are selected only on the seed42 validation set; one evaluation after locking, and no candidate is changed based on the test set.
- The original test set was made public in the previous round; this round is exploratory follow-up validation, not a fresh external validation.
- Rounds are selected on validation within 50 rounds, and all private heads/gates are restored along with the model; the comparison baseline uses the same round-selection rule.
- All candidates and failures are retained; the best seed is not cherry-picked. The paper's 79% remains an independent reference.
- E uses the 400-step mask; FedAvg is additionally matched with the same 400-step mask/standardization/split/round selection, and the masks are identical across all three seeds.

The paper's 79% reference is retained; whether it is reached must be read from real test values, and validation accuracy is not a substitute for test results.

## The Original Baseline Must Not Be Hidden

|Mean over the same three seeds|Site macro|Pooled subjects|
|---|---:|---:|
|Original fixed 50 rounds|60.76%|62.85%|
|A: validation round selection only|59.00%|62.15%|
|E: validation-selected candidate|60.02%|63.54%|

Candidate vs original macro change **-0.73 percentage points**; pooled change +0.69 percentage points — do not report only the more flattering metric.

## Failure Autopsy and Stop Decision

1. The failure is in generalization: seed42 validation +4.82pp, but the three-seed test average vs A is only +1.03pp, below the 2pp gate set before the run; seed44 is negative, and the original macro is higher too.
2. Training being unlearnable and obvious pipeline errors have been ruled out: the existing real train-only 8-subject power gate passes; full private/gated restoration, loss gradients and the current broadcast-representation timing all have unit tests; an independent audit separately checked original predictions/subject exclusivity/checkpoint replay.
3. Remaining mechanism: 400 steps still do not solve the joint nonsmooth optimization of Eq11, and 77 validators with the smallest site at 6 subjects make round/candidate selection visibly unstable; a single KKI validation classification flip changes macro by about 2.78pp.

This round of bounded five candidates is complete; B/C/D test sets will not be opened, and D will not be reselected based on E's test failure. Keep the original interface default model and do not declare the accuracy optimization successful.

## Next-Round Priorities (Not Executed; Not Claimed Effective)

1. First solve Eq11 with a joint proximal/primal-dual/ADMM method; it must pass analytic counterexamples and KKT/dual residuals before measuring generalization.
2. Select lambda_aug and the feature count in nested validation instead of fixing top512 with unconverged coefficients; the outer fixed repeated cross-validation is used only for final comparison.
3. If the first two items still give no gain, separately preregister multi-scale dynamic FC + static FC fusion; do not first enlarge the network or the noise.

## Matched-Control Audit

400-step same-mask/noise/round-selection FedAvg mean 60.23% macro; the candidate is -0.20pp relative to it, which does not support superiority over FedAvg.

|seed|Identical indices|Identical training scaler|Identical subject split|
|---|---|---|---|
|42|Yes|Yes|Yes|
|43|Yes|Yes|Yes|
|44|Yes|Yes|Yes|

The original 40-step mask + best-val FedAvg is 61.93% macro, stored separately as an auxiliary control with different preprocessing; it is not claimed to be the same mask.
