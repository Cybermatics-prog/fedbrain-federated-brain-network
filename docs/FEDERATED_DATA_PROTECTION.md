# Health data and federated protection: one view expressing two things

## Goal of this round
Keep health data as the protagonist, while letting readers understand where the raw data, the uploaded messages, and the aggregation results each live; training replay or accuracy curves are no longer treated as the primary security visualization.

## Visualization semantics
- Researcher view: real brain connectivity, age, and BOLD signals from the public cohort; annotated as research-browsing data, not federated-training uploaded messages.
- Site data domain: the training protocol does not upload raw BOLD, individual FC, or per-sample diagnostic labels; the personalized private head and gate do not enter the shared parameter dictionary.
- Upload envelope: shared encoder/global head parameters, plus the marginal and class-mean statistics exchanged when alignment is enabled. It cannot be written vaguely as "only parameters"; class statistics also carry information.
- Aggregation side: processes these uploaded messages according to the protocol and returns the global shared model; the researcher's 116×116 individual data map is not treated as parameters that the server receives.
- Optional aggregation-side protocol view: a UI teaching illustration only, not a permission mechanism; it does not claim that the API has access control after switching.

## State that must come from an actual run
|Item|Default 8-round process run|Meaning|
|---|---|---|
|Shared parameters|4113, raw per-client tensor 16452 bytes|Excludes alignment statistics or the communication volume after homomorphic-encryption expansion|
|Private parameters|18 per site|The shared state and broadcast do not overwrite private modules; the record is verifiable|
|Parameter perturbation|Laplace b=0.01, clip norm10|Not equivalent to certified ε differential privacy|
|Homomorphic encryption|Not enabled in the default run|A configurable capability is not passed off as currently in use|
|Historical encryption example|encrypted-smoke-2048 actually completed|A Paillier2048 algorithm demo; without per-client process records, evidence must not be fabricated|
|Alignment statistics|Exchanged independently, not covered by the current parameter noising/Paillier|Encrypted parameters do not mean that all exchanged information is encrypted|

The current environment is a single-machine six-client simulation that reads public data centrally, with the coordinator and the Paillier private key in the same process; the health-data interface allows research browsing. This expresses the intended federated communication boundary; it is not a completed hospital isolation deployment or an absolute security guarantee.

## Acceptance
1. Health FC/BOLD and the existing filters remain usable, and the page shows no fabricated ciphertext, privacy budget, or security score.
2. HE is shown as not enabled by default; the Paillier2048 algorithm status appears only after switching to the real historical HE run.
3. With no process records, the private-preservation/shared-consistency evidence must be unknown and must not default to pass; round 0 shows configuration only and is not described as already executed.
4. Switching the protocol view does not automatically train, modify parameters, or change existing predictions/accuracy.
5. Both desktop and mobile have readable descriptions of the data domains, the uploaded content, and the scope of the single-machine illustration.
