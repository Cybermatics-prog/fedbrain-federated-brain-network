# SPDX-License-Identifier: MIT
# Related federated work

This page collects the federated-learning literature that this system builds on or is
compared against, and it is the place to record the group's own federated publications.
Bibliographic entries live in [references.bib](references.bib); this page explains how each
entry relates to the code.

## How this page is used

- One row per publication, grouped by the role it plays for this system.
- `Role` is one of: *method reference* (a formulation the code follows), *evaluation
  reference* (a number kept for orientation only), *privacy or aggregation reference*, or
  *group publication* (work produced by this group).
- Add the BibTeX entry to [references.bib](references.bib) first, then add the row here so
  the key and the table stay in sync.

## Methodological and evaluation references

| Key | Publication | Role | Where it appears |
|---|---|---|---|
| `zhang2023fedbrain` | FedBrain: a robust multi-site brain-network analysis framework based on federated learning for brain disease diagnosis (full record in `references.bib`) | Method reference: the formulation followed by the sparse selection, alignment, mixture-of-experts and perturbation modules; the reported mean accuracy is kept only as an orientation figure | `fedbrain/sparse.py`, `fedbrain/model.py`, `fedbrain/privacy.py`, `docs/RESULTS.md` |

The system is an independent implementation maintained by this group. It is not a
redistribution of another code base, and measured results are reported as this system's own
numbers; see [RESULTS.md](RESULTS.md).

## Data references

| Key | Publication | Role | Where it appears |
|---|---|---|---|
| `dimartino2014abide` | The autism brain imaging data exchange: towards a large-scale evaluation of the intrinsic brain architecture in autism (full record in `references.bib`) | Dataset description for the public ABIDE-I derivatives | `fedbrain/data.py`, README |

## Group publications

Add the group's federated-learning publications here. Remove the example row when the first real
entry is added.

| Key | Publication | Role | Where it appears |
|---|---|---|---|
| _example-key_ | _Author list. Title. Venue, year._ | Group publication | _module or document that implements or cites it_ |

## Adding an entry

1. Append the BibTeX record to [references.bib](references.bib), keeping the key short and
   unique (`firstauthorYYYYtopic`).
2. Add a row to the matching table above with the same key.
3. If the publication changes how a module behaves, mention it in that module's docstring so
   the provenance stays next to the code.
