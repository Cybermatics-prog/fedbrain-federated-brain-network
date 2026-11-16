# FedBrain accuracy-optimization pre-registration (2026-11-13)

## Testable propositions
The current generalization gap may come from three localizable biases: the best epoch is not retained according to the validation set; the Eq. 16 classification loss takes the mean while the Eq. 14 alignment takes the sum, which makes alignment relatively too strong; and the alignment target comes from the local encoder before aggregation, so its coordinates differ from those of the shared encoder after broadcast. Another shared ceiling is that the sparse optimization has not converged at 40 steps and the union is still 6670-dimensional.

The original test metrics were already disclosed in the previous round. This round uses strictly only training/validation data to select changes, with a single evaluation locked afterward; it is an exploratory follow-up experiment and does not pose as a brand-new independent external validation.

## Settings left unchanged
Same six sites and same 465 subjects, same train/val/test split, AAL116/window 32/stride 1, hidden8, max50round, batch256, Laplace scale0.01, same mask512 (the sparse control is described separately), same learning rate 0.001. Model epoch selection looks only at subject-level six-site macro validation accuracy, with ties keeping the earliest epoch; every main branch runs the full 50 rounds, to isolate epoch selection from the training budget. Parameters, models, or seeds must not be selected by test results.

## Bounded candidate set (seed42, validation-set screening)
- A `best_val`: only adds validation-based saving/restoring of all six client models (shared, private heads, gates); everything else matches the original implementation.
- B `paper_objective`: on top of A, changes the objective to BCE_mean + DA_sum / actual batch size; it adopts the paper-inspired minibatch convention (sumCE_batch+DA_batch)/batch with a DA coefficient of 1; this is not equivalent to proving a reproduction of the full-data Eq. 16 sumCE_full+DA_full (the nonlinear mean distance and the sample count differ), nor does it pose as the authors' source code. The site-level sum that the paper makes explicit is not mistakenly changed to a mean.
- C `fresh_alignment`: on top of B, recomputes the training hidden means after each round's aggregation/broadcast completes, so the alignment target no longer comes from the stale local encoder.
- D `regularized`: on top of C, only adds Adam weight_decay=0.01, as a clearly labeled anti-overfitting extension (not a recipe the paper provides).
- E `sparse400`: on top of C, only raises the original Eq. 11 proximal iterations from 40 to 400, using the same lambda0.01/same number of training segments/same 512 cap. Check objective and mask stability first, then run the empirical validation; if there are no valid nonzero features, fail honestly and do not patch it according to test results.

A wider MLP, higher noise, a changed QC queue, or new data splits are not enabled automatically; noise and shuffle currently share the RNG, so this round runs no noise ablation.

## WIN / KILL / validity conditions
- Gate: all existing offline tests and the new checkpoint-isolation/DA-scaling/synchronized-timing tests must pass; run a 2-round real smoke first.
- Data validity: subject-wise splitting comes first, and sparsification/standardization use train only; search runs use `defer_test=true` and must not produce test predictions or test-metric files.
- seed42 screening: candidates are ordered by best macro validation accuracy descending, with ties broken by the fixed A→B→C→D→E order; only an improvement of >=2 percentage points over A counts as a minimally useful optimization, otherwise report no decisive improvement.
- Training validity: all losses/gradients are finite and the budget is actually completed; low-scoring/failed runs are retained.
- Confirmation: after one candidate recipe is locked, compare three seeds seed42/43/44 against the same best-val baseline A; only if at least 2/3 seeds improve in macro test accuracy with a mean improvement of >=2 percentage points may this be called a reproducible gain in this configuration. All seeds are reported, and falling short is not called success.
- The same-mask FedAvg uses the same best-val epoch selection, so the new method does not get an extra epoch-selection advantage. The final comparison does not tune parameters further on this basis.
- Control-audit addendum during execution (the candidate has already been selected as E on validation, and the three-seed test comparison has not yet been read): E uses the new 400-step mask while FedAvg(A) in the original script still uses the 40-step mask, so it cannot be called a same-mask control. Add an independent control: the same 400-step mask/noise/epoch-selection FedAvg with three seeds; this only repairs control matching, takes no part in selection, and does not change the E recipe or the WIN/KILL criteria. During verification the per-seed indices must be checked to be exactly identical; if unmatched, that FedAvg comparison is invalid. The old 40-step FedAvg results are retained and labeled separately.

## Record-keeping and budget
Each item runs at most 50 rounds, and an existing GPU 50-round run takes about 1–2 minutes; the empirical validation covers at most 5 candidates plus the three-seed confirmation, with no unbounded search. Save the config, source hashes, training/validation history, best epoch, whether test was executed, and the model. Confirmation tests may be opened only after the candidate-screening evidence has been independently written to JSON. The sparse 40→400 diagnostic separately records convergence and mask Jaccard, and does not substitute "more sparse" for accuracy evidence.
