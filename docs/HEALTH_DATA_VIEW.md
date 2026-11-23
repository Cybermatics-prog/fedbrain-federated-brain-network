<!-- SPDX-License-Identifier: MIT -->
# Brain-health data visualization: data definitions and usage

This page displays real ABIDE-I resting-state fMRI data and research labels; it is not a federated-training animation or a model-accuracy chart. It defaults to the HC control, with switching to ASD and to the descriptive difference between the two groups; viewing the data requires no prior training.

## Data scope
- 465 real subjects: HC241, ASD224; six sites NYU/UM/USM/UCLA/KKI/Yale.
- Ages from 6.47 to 50.2231 years, and all current 465 subjects have a valid age; this page filters and computes statistics on the actual values.
- CPAC / filt_noglobal / AAL116; every subject has a validated public time series with at least 32 frames and 116 columns.
- HC/ASD are original research labels, and HC does not equal a clinical health certification; the chart contains no fabricated disease risk or health score.

## What to look at
1. **Group FC**: the 116×116 Pearson connectivity of HC or ASD; Fisher-z per subject first, then equally weighted averaging, and finally tanh back to r. Every subject carries the same weight, instead of a longer scan duration carrying more weight.
2. **Difference FC**: the ASD group mean minus the HC group mean, in units of Δr; age/site and other confounders are not corrected for, and it is not to be interpreted as a disease marker or a causal difference.
3. **Strong-connectivity network**: takes the connections with the largest absolute value in the upper triangle of the current real matrix, keeping positive/negative signs. The circle is only a layout and AAL-### is only a file column index; it does not pose as an anatomical localization.
4. **Individual time series**: after a real subject is selected, shows the complete BOLD series of the three ROIs with the largest temporal standard deviation, z-scored per ROI. The horizontal axis is the frame index, and because no TR is provided it is not labeled in seconds.
5. **HC reference**: the individual FC minus the descriptive HC mean over the whole cohort; if the current subject is also HC, the reference mean excludes that subject (241→240). This is not a clinical normal range or a risk probability.
6. **Distributions and filters**: site, age lower/upper bounds (endpoints included), HC/ASD research label; an empty group shows no data instead of an all-zero matrix.

Showing the FC diagonal as 1 is an explicit plotting convention; both the connection strength and the ranking exclude the diagonal, and the difference matrix has 0 on the diagonal.

## Interface and recomputation

- `GET /api/health-data`: cohort, age/site distributions, group matrices, difference, and subject list; `site=KKI&age_min=10&age_max=20` may be appended.
- `GET /api/health-data/subjects/51036`: a real HC example, 176 frames, 3 actual ROI curves, 30 strong connections, and a 240-subject leave-one-out HC reference.
- `python scripts/build_health_data.py`: builds and audits the cache; reads no checkpoint, trains nothing, and calls no classification model.
- `python -m unittest discover -s tests -v`: includes regression tests for the analysis formulas, per-subject aggregation, source invalidation, empty groups, the leave-one-out method, and the model-free API.

The original file paths/sizes/mtime, cohort metadata, and manifest digest together form the provenance fingerprint; unchanged subjects may reuse their FC, while new/changed files are recomputed. No synthetic data or model outputs are used to fill in missing data.
