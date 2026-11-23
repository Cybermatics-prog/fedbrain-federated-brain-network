<!-- SPDX-License-Identifier: MIT -->
# FedBrain Reproduction Pre-registration (2026-11-13)

## Testable Goals
On the six sites of the public ABIDE-I / PCP / CPAC `filt_noglobal` / AAL116 data, establish FedBrain training, subject-level evaluation, and a real inference demo without subject leakage. Paper DOI: https://doi.org/10.1016/j.neucom.2023.126791.

## Order and Budget
1. Evaluator gold-label and random-label self-checks; unit tests for dynamic FC, grouped splits, MoE, sparsity, and encrypted aggregation.
2. Real six-site small-sample smoke test with 2 communication rounds. WIN: end-to-end completion, finite loss and predictions, independent train/test subjects at all six sites. KILL: any cross-set overlap or non-finite value; full training is forbidden until fixed.
3. All six-site subjects that downloaded successfully and are valid; first run a fixed-seed-42 50-round held-out experiment and save the full config, split, probabilities, model, and per-round metrics. Split by subject first, sliding windows afterward; average the predicted probabilities of a subject's windows.
4. A FedAvg control on the same split; run repeated stratified 5-fold if the budget allows (the paper uses 5 repeats×5 folds). Call a difference stable only with at least 3 seeds; otherwise call it an observation under the current configuration.

## Validity Gates (not adjustable after the fact)
- When the scorer is given gold labels, accuracy=1.0; with 10000 fixed balanced random labels, the control accuracy is within [0.47,0.53].
- The train/validation/test subject ID sets are pairwise disjoint; windows must not be assigned across sets; sparse screening and standardization may use only the training set.
- The positive control for real small-sample training can reach accuracy>=0.95 on the training set; if not, only debug the implementation and do not use that to reject the paper.
- Main-experiment performance WIN: six-site subject-level mean accuracy>=0.79 and the above gates passed; if not reached, report only the measured value and do not claim the paper's 79% has been reproduced.
- Comparing FedBrain with FedAvg must use the same subject split, windows, hidden width, sparse mask, and number of communication rounds; FedAvg does not enable domain alignment/private experts. Test sets must not be used to select hyperparameters or rounds.

## Verified Paper Settings and Implementation Gaps
- Six sites: NYU, UM_1, USM, UCLA_1, KKI, YALE; the paper's counts are 167/88/52/63/48/56, and the currently valid public download counts will be recorded separately.
- AAL116, Pearson→Fisher-z, upper triangle 6670 dimensions, window 32, stride 1, MLP hidden width 8, 50 communication rounds.
- Eq11 sparse global/time-segment coefficients use alternating proximal/gradient optimization; masks are merged across sites. The paper does not give complete optimization hyperparameters, site-merging rules, training batch size, or DP budget accounting.
- Eq12–14 alignment uses linear-kernel MMD marginal and per-class hidden-representation means; it exchanges aggregate statistics from the training set and does not pool raw windows across sites.
- Eq15 uses a logit mixture of the shared encoder/global head and each site's private head/gate.
- The paper text is inconsistent between λ_M=0.01 and 0.1: the default explicitly selects 0.01, it is configurable, and it is recorded in every run; parameter noise addition does not claim epsilon-DP without accounting proof.
- Encrypted aggregation provides actual Paillier homomorphic arithmetic validation and can optionally be used for shared-model aggregation; a single-machine simulation is not equivalent to a security system deployed across isolated hospitals.

## Record Keeping
All runs are saved to `runs/<run_id>/`, including config, subject split, per-round history, real subject predictions, checkpoint, and encryption/communication settings. The literature's 79% and the measured metrics are displayed in separate UI columns, with no simulated curves filled in.
