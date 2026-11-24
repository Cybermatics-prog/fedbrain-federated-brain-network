<!-- SPDX-License-Identifier: MIT -->
# FedBrain Results from Real Data

This report is generated automatically by `scripts/report.py` from real `result.json` files; experiment numbers are never entered by hand.

## Data and Evaluation Scope

- Public subjects actually loaded: 465; download failures: 0.
- CPAC / filt_noglobal / AAL116; window 32, stride 1, Pearson→Fisher-z, 6670-dimensional upper triangle.
- Subjects are split into train/val/test first; sparsification and standardization use training data only; each subject's window predicted probabilities are averaged.
- macro-site is the equal-weight mean of the six site accuracies; pooled is the accuracy over all test subjects combined. The two must not be conflated.

|Site|Actual subjects|Reference counts|
|---|---:|---:|
|KKI|33|48|
|NYU|172|167|
|UCLA|64|63|
|UM|88|88|
|USM|67|52|
|Yale|41|56|

## Full 50-Round Experiments

Experiments are grouped by recipe; only FedAvg runs with the matching configuration/seed/feature mask are matched-mask controls, so 40-step and 400-step masks must not be conflated.

|Run|Method|seed|macro-site accuracy|pooled accuracy|AUC|Test subjects|Training accuracy|Elapsed seconds|
|---|---|---:|---:|---:|---:|---:|---:|---:|
|`full-fedbrain-s42`|fedbrain|42|56.89%|59.38%|0.6318|96|97.26%|133.2|
|`full-fedavg-s42`|fedavg|42|60.73%|62.50%|0.6574|96|99.32%|55.6|
|`full-fedbrain-s43`|fedbrain|43|60.41%|61.46%|0.6565|96|98.63%|94.4|
|`full-fedbrain-s44`|fedbrain|44|64.97%|67.71%|0.7086|96|100.00%|92.8|
|`full-fedavg-s43`|fedavg|43|67.07%|66.67%|0.6939|96|100.00%|41.9|
|`full-fedavg-s44`|fedavg|44|63.26%|64.58%|0.7139|96|100.00%|36.7|
|`opt-A-s42`|fedbrain|42|56.77%|60.42%|0.7143|96|82.53%|84.3|
|`opt-E-s42`|fedbrain|42|60.62%|63.54%|0.6956|96|99.66%|100.7|
|`opt-A-s43`|fedbrain|43|60.41%|61.46%|0.6565|96|98.63%|81.8|
|`opt-E-s43`|fedbrain|43|61.31%|65.62%|0.7069|96|96.23%|103.0|
|`opt-A-s44`|fedbrain|44|59.81%|64.58%|0.7269|96|96.23%|93.3|
|`opt-matched-FedAvg-s42`|fedavg|42|63.09%|67.71%|0.7490|96|99.66%|74.7|
|`opt-matched-FedAvg-s43`|fedavg|43|63.02%|62.50%|0.6830|96|100.00%|63.5|
|`opt-E-s44`|fedbrain|44|58.14%|61.46%|0.6539|96|100.00%|109.9|
|`opt-matched-FedAvg-s44`|fedavg|44|54.57%|57.29%|0.6548|96|99.66%|68.9|
|`opt-FedAvg-s42`|fedavg|42|64.40%|64.58%|0.6969|96|88.01%|39.5|
|`opt-FedAvg-s43`|fedavg|43|56.66%|60.42%|0.7082|96|84.59%|39.9|
|`opt-FedAvg-s44`|fedavg|44|64.73%|68.75%|0.7260|96|91.10%|38.1|

### Summary by Experiment Recipe (candidates, budgets and round-selection rules not mixed)

- fedbrain / original-fixed-final: n=3, macro-site mean=60.76%, sample std=4.05 percentage points; pooled mean=62.85%.
- fedavg / original-fixed-final: n=3, macro-site mean=63.69%, sample std=3.19 percentage points; pooled mean=64.58%.
- fedbrain / optimization-A: n=3, macro-site mean=59.00%, sample std=1.95 percentage points; pooled mean=62.15%.
- fedbrain / optimization-E: n=3, macro-site mean=60.02%, sample std=1.67 percentage points; pooled mean=63.54%.
- fedavg / optimized-matched-FedAvg: n=3, macro-site mean=60.23%, sample std=4.90 percentage points; pooled mean=62.50%.
- fedavg / optimization-A: n=3, macro-site mean=61.93%, sample std=4.57 percentage points; pooled mean=64.58%.

## Site results for main run full-fedbrain-s42

|Site|Test subjects|Accuracy|AUC|Sensitivity|Specificity|
|---|---:|---:|---:|---:|---:|
|KKI|7|71.43%|0.7500|33.33%|100.00%|
|NYU|35|71.43%|0.7333|60.00%|80.00%|
|UCLA|13|23.08%|0.2500|12.50%|40.00%|
|UM|18|55.56%|0.5714|42.86%|63.64%|
|USM|14|64.29%|0.7556|77.78%|40.00%|
|Yale|9|55.56%|0.4500|20.00%|100.00%|

### Interpretable Conclusion

Original failures and low-scoring results are kept; no tuning on the test set or selection of rounds is performed.

## Independent Validity Checks

- Real train-only 8-subject overfitting power control: passed=True, accuracy=100.00%.
- Split/raw-probability/recomputed-score consistency across completed experiments: 20 checks, all_passed=True.
- Maximum predicted-probability error between Paillier 2048-bit and plaintext smoke runs: 9.685754776000977e-08.
- Real browser checks passed on a 1440 desktop and a 390 mobile viewport; all displayed values come from real API responses.
- Real interface START/STOP: win=True, HTTP=200/200, persisted state=stopped.

## Acceleration Approximations and Privacy Settings

- In the main configuration, the sparse stage uses 8 relative time segments per training subject, 40 iterations, and at most 512 connections; the classification stage uses all windows.
- Optimization/merging/batching details that the group's formulation does not specify are explicitly fixed in the configuration; the original subject list is unknown.
- The main configuration clips shared parameters and then adds Laplace noise with noise_scale=0.01; this is not epsilon-DP with a budget-accounting proof.
- Homomorphic aggregation is optional real Paillier 2048-bit arithmetic; holding the key in the same process on one machine is not equivalent to a deployment across independent hospitals.

## Smoke and Homomorphic Runs

|Run|Status|Rounds|Homomorphic aggregation|Test accuracy|
|---|---|---:|---|---:|
|`smoke-six-sites`|failed|0|False|—|
|`smoke-six-sites-v2`|completed|2|False|54.17%|
|[encrypted-smoke-2048](../runs/encrypted-smoke-2048/result.json)|completed|2|True|54.17%|
|`20261113-161021-672878-fedbrain`|stopped|0|False|—|
|`opt-smoke-validation-only`|completed|2|False|—|
|`opt-B-s42`|completed|50|False|—|
|`opt-C-s42`|completed|50|False|—|
|`opt-D-s42`|completed|50|False|—|

## Evidence Entry Points

- Test-unlocked runs contain result.json / split.json / predictions.json / checkpoint.pt / preprocessing.npz; validation candidates have only validation_predictions.json and do not generate test predictions.
- Test command: `.\.venv\Scripts\python.exe -m unittest discover -s tests -v`.
- Full 5-repeat × 5-fold script: `scripts/benchmark.py`; do not claim the 25-run protocol is complete unless it has been executed to completion.
- The original failing smoke-six-sites recorded insufficient minority-class samples caused by ID truncation; this was fixed with fixed-seed stratified sampling within each site plus a regression test.
