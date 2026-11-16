# Federation-Process Main Visual Rebuild and Authenticity Acceptance

User correction: the interface should visualize the process and effect of federated collaboration, with accuracy only as a small auxiliary panel.

## What the Main Visual Must Answer
1. What data does each of the six sites train on? Which data never leaves its site?
2. How much did the local model update this round? How are shared parameters distinguished from private experts/gates?
3. What did the server actually receive, how is it averaged and aggregated, and to whom is it broadcast?
4. After broadcast, are the shared parameters really identical, and were the private parameters really not overwritten by aggregation?
5. In the same training-diagnostic projection coordinates, what happens to the representation after local training versus after aggregate broadcast? Do not pass this comparison off as an isolated causal MMD gain.

## Data and Presentation
- Record real local_train/upload/aggregate/broadcast events, timestamps, and sites each round; the animation is a controllable replay of these real records and does not reconstruct events that do not exist in old experiments.
- The parameter plot uses 64 fixed shared-model parameter coordinates, not a brain-connectivity/ROI map; it shows the actual local delta, upload perturbation, aggregate value, and broadcast consistency.
- The representation plot uses only site means/covariances/class means from fixed training-window probes; PCA coordinates are fit at training start and fixed throughout; no individual windows/raw FC are exported.
- The single-machine diagnostic record can observe both the local clean parameters and the uploaded parameters; this does not mean a real server could see private weights that were never uploaded.
- Historical runs without records are explicitly shown as not replayable, preserving the original measured metrics and inference; the new federation demo is selected by trace existence by default, not by accuracy.
- Accuracy/ablation/optimization reports are all placed in the default-collapsed auxiliary results area and do not occupy the first screen.

## Pre-Run Acceptance Gates
- First run two rounds with the same configuration as the original smoke; the new observation feature toggle must not change the original model output: same test subjects as smoke-six-sites-v2, max absolute probability difference <=1e-6; higher accuracy is not the goal.
- Unit tests guarantee that model parameters, module train/eval flags, and NumPy/PyTorch random-number state are unchanged before and after observation.
- Every complete round must have six client_started, six client_uploaded, one aggregate, and one broadcast operation (two events: pending at start and verified on completion), with timestamps and sequence numbers genuinely ordered.
- The max error between the equal-weight mean of real upload statuses and the actual aggregate <=1e-6 (plaintext); shared-parameter max difference after broadcast = 0; each site's private parameter hash is exactly identical before and after that round's aggregation.
- All recorded values are finite, the 64 coordinates are fixed per round, the fixed PCA reads only training data, and incomplete/stopped runs honestly show partial status.
- After the gates pass, run exactly one 8-round process demo on the full 465-subject cohort, ID=federation-demo-s42, reusing the original seed42 train-only512mask, hidden8/noise0.01; select the final round to avoid confusing replay's last round with the inference-selected round. Do not treat the 8-round demo as the main performance experiment.
- Actual checks at desktop 1440 and mobile 390: six nodes + server, upload/broadcast direction, play/pause/step/round selection, site clicks, before/after representation/parameter plots, missing-record notice for old runs, default-collapsed accuracy area, 0 JS errors/no horizontal overflow.

## Smoke Audit Autopsy
Two rounds of real training had completed, but the first audit version wrongly expected a single event for broadcast start/verified completion instead of the two that occur, tripping the assertion; training and recording themselves did not fail. Aggregation/training errors were ruled out: the failing line was only the event count. The audit was fixed to one pending plus one verified in correct order, and the existing records were re-audited directly, without retraining or changing the original events. The 8-round process demo was started only after the audit and the non-intrusive output gates passed.
