<!-- SPDX-License-Identifier: MIT -->
# Federated data transfer diagram

The page no longer substitutes three explanation boxes for the visualization. The diagram shows separately:

- individual FC and BOLD examples inside the six sites;
- the uploaded shared-parameter sketch and the alignment-statistics category;
- the central aggregator's shared-parameter result and the model return path.

`Play Transfer` only demonstrates the direction of messages; it starts no training, nor does it mean that actual network packets were captured. FC and BOLD always stay inside the site frames, and moving markers represent only shared parameters or alignment statistics.

## Data sources

`GET /api/health-data/sites` is independent of the model. For each site, the first HC is chosen by anonymous ID (the first subject if there is no HC); the FC is that subject's complete 116×116 Pearson matrix. For BOLD, the ROI with the largest temporal standard deviation is chosen, the complete series is z-scored first, then at most 128 raw frames are chosen for the thumbnail, keeping the first and last frames; no interpolation, no waveform generation. The examples come from the public cohort and do not indicate participation in the training of the selected run.

The `transfer` field of `GET /api/data-protection` comes from the latest recorded round:

- `uploaded_shared_sketch`: the shared weights after clipping and noising, at 64 fixed coordinates;
- `clean_shared_sketch`: the same-coordinate values before noising;
- `aggregate_shared_sketch`: the same-coordinate values after actual aggregation;
- `private_preserved`: the private-module preservation check in the record.

The parameter sketch is not ROI, FC, or PCA. Unrecorded alignment means and counts are not fabricated, and historical runs without a trace generate no parameter arrays or ciphertext.

## Display rules

The transferred-parameter sketch in the diagram is only used to explain the message type. The full default model has 4113 shared parameters, of which the sketch shows 64; 16.1KB is the pre-encryption size of a single client's shared tensor, not the total communication volume. The alignment mean, class means, and class-window counts are exchanged independently and currently do not pass through the noising or Paillier path of the shared parameters.

Noising, homomorphic encryption, subject count, and round count always follow the selected record: the default 8-round record has 465 subjects/train 292 and Paillier not enabled; the historical encryption validation has 120 subjects/train 72, only 2 rounds, and no parameter trace, so the full-cohort diagram must not be passed off as 465 subjects having been encrypted.

This is currently a single-machine simulation on public data. Switching data/upload only changes the display and does not constitute access control or a hospital isolation deployment.
