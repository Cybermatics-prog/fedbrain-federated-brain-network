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

## Methodological formulation

The formulation used by this system is documented in the group's own publication. That entry
is intentionally not listed yet: it will be added here, together with its BibTeX record in
`references.bib`, once the group uploads and organises the publication.

| Key | Publication | Role | Where it appears |
|---|---|---|---|
| _to be added_ | _Group publication describing the formulation used here_ | Method reference: sparse selection, alignment, mixture-of-experts and perturbation modules | `fedbrain/sparse.py`, `fedbrain/model.py`, `fedbrain/privacy.py` |

The system is an independent implementation maintained by this group. Measured results are
reported as this system's own numbers; see [RESULTS.md](RESULTS.md).

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
