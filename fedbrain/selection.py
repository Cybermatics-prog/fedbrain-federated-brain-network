# SPDX-License-Identifier: MIT
"""Validation-only checkpoint selection; includes all private expert/gate tensors."""
from __future__ import annotations

import math


class ValidationCheckpoint:
    def __init__(self):
        self.score = -math.inf
        self.round = 0
        self.states = None

    def consider(self, round_index, score, models):
        if not math.isfinite(float(score)):
            raise ValueError('Validation score must be finite')
        if float(score) <= self.score + 1e-12:
            return False  # Earliest tied best, no test or later tie-break tuning.
        self.score, self.round = float(score), int(round_index)
        self.states = {site:{name:value.detach().cpu().clone() for name,value in model.state_dict().items()}
                       for site,model in models.items()}
        return True

    def restore(self, models):
        if self.states is None:
            raise RuntimeError('No validation checkpoint recorded')
        if set(models) != set(self.states):
            raise ValueError('Client sets differ')
        for site,model in models.items():
            model.load_state_dict(self.states[site])


def scaled_alignment_loss(ce_mean, alignment_sum, batch_size, weight=1., normalization='legacy'):
    """Eq16(sumCE+DA)/N is CEmean+DA/N, NOT CEmean+DA."""
    if batch_size <= 0:
        raise ValueError('Positive minibatch size required')
    if normalization not in ('legacy','batch'):
        raise ValueError('Unknown loss normalization')
    factor = float(weight) / batch_size if normalization=='batch' else float(weight)
    weighted = alignment_sum * factor
    return ce_mean + weighted, weighted
