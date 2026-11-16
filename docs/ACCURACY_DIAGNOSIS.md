# Why current accuracy is low: a read-only diagnosis of historical records

This round only reads existing predictions and results; it does not train, does not choose new thresholds, and does not re-select models based on test. Original recomputable details: [diagnosis JSON](<../artifacts/accuracy_diagnosis.json>); script: [diagnose_accuracy.py](<../scripts/diagnose_accuracy.py>).

## The current default 8 rounds and 50 rounds must not be conflated

- The default federation run is `federation-demo-s42`; its 8 rounds are used to demonstrate client update/upload/aggregate/broadcast and are not the 50-round main performance experiment.
- This model has train pooled 69.18%, test pooled 55.21%, and test macro 53.41%. Of the 96 subjects, 47 have ASD, yet only 8 are judged ASD, of which 6 are correctly identified; ASD recall is 6/47=12.77% and HC specificity is 95.92%, a clear bias toward HC.
- AUC 0.7377 shows that the ranking contains signal, but the default 0.5 decision is not ideal; the mean predicted probabilities for HC/ASD are 0.4121/0.4500 respectively, both below 0.5. No independent validation design has yet shown that calibration can improve consistently, so the threshold is not changed according to test.
- With train/val/test subjects exactly identical to the original seed42 and the same main learning configuration, the validation macro differs by exactly 0.0 round by round over the first 8 rounds. From 8→50 rounds, test pooled goes 55.21%→59.38%, only +4.17pp; this cannot be attributed to the newly added visualization recorder, nor can one expect more rounds to close the entire gap.

## The more fundamental bottleneck: generalization is still insufficient after training to 50 rounds

- Across the original three seeds, train pooled averages 98.63%, test pooled 62.85%, and site macro 60.76%. High on training and low on unseen subjects is a clear generalization gap, not a model that simply cannot learn.
- There are only 292 truly independent training subjects; the sliding windows add correlated windows, not independent subjects. High-dimensional connectivity, site distribution differences, and validation-based selection at small sites all make overfitting/estimation variance worth investigating first.
- The matched-mask/noise FedAvg with the same three seeds has macro 63.69%, higher than FedBrain's 60.76%; the extra alignment/personalization modules in the current implementation do not show a stable additional benefit. This does not justify declaring the paper invalid.
- The existing five-candidate validation changes did not bring a stable test improvement; the Eq. 11 sparse solve has still not converged at 400 steps, and the current top512 is an approximation of a public implementation, not the authors' exact source code. Its differences from the data queue/repeated-evaluation protocol have not been fully closed, so it cannot be claimed that the paper's 79% has been reproduced.

## What to target next

For the unconverged feature selection, first do joint proximal/ADMM and KKT certification; for generalization/decision shift, use nested train-validation to choose the feature strength and calibration. The outer level is used only for final evaluation; do not keep trying parameters one by one against an already-seen test, and do not package the 8-round process demo as a performance improvement.

The interface already labels the small evaluation section explicitly as `Process demo · 8 rounds · Not the 50-round main experiment`, while keeping the original 50-round run available for switching; no displayed number was raised and no highest-test-seed was cherry-picked.
