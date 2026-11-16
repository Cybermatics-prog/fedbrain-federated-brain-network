"""Small shared MLP, private expert/gate, and moment alignment (Eqs.12–16)."""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping

import torch
from torch import Tensor, nn


class FedBrainModel(nn.Module):
    """One hidden layer (paper default8) with globally shared and private heads.

    The sigmoid gate mixes *logits*, exactly as the binary formulation in Eq.15;
    sigmoid is applied only for prediction, not before BCEWithLogitsLoss.
    Only ``encoder`` and ``global_head`` belong to the federated shared state.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 8):
        super().__init__()
        if input_dim < 1 or hidden_dim < 1:
            raise ValueError("input_dim and hidden_dim must be positive")
        self.input_dim, self.hidden_dim = int(input_dim), int(hidden_dim)
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.global_head = nn.Linear(hidden_dim, 1)
        self.private_head = nn.Linear(hidden_dim, 1)
        self.gate = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())

    def forward(self, inputs: Tensor) -> dict[str, Tensor]:
        hidden = self.encoder(inputs)
        global_logit = self.global_head(hidden).squeeze(-1)
        private_logit = self.private_head(hidden).squeeze(-1)
        gate = self.gate(hidden).squeeze(-1)
        mixed = gate * global_logit + (1 - gate) * private_logit
        return {"logit": mixed, "mixed_logit": mixed,
                "global_logit": global_logit, "private_logit": private_logit,
                "hidden": hidden, "gate": gate}

    def shared_state_dict(self) -> OrderedDict[str, Tensor]:
        """A detached clone: caller aggregation cannot mutate local parameters."""
        return OrderedDict((name, tensor.detach().clone()) for name, tensor in self.state_dict().items()
                           if name.startswith(("encoder.", "global_head.")))

    def load_shared_state_dict(self, state: Mapping[str, Tensor], strict: bool = True):
        expected = set(self.shared_state_dict())
        supplied = set(state)
        if strict and supplied != expected:
            raise ValueError(f"Shared state mismatch: missing={sorted(expected-supplied)}, unexpected={sorted(supplied-expected)}")
        unexpected = supplied - expected
        if unexpected:
            raise ValueError(f"Refusing non-shared parameters: {sorted(unexpected)}")
        return self.load_state_dict(dict(state), strict=False)

    def private_state_dict(self) -> OrderedDict[str, Tensor]:
        return OrderedDict((name, tensor.detach().clone()) for name, tensor in self.state_dict().items()
                           if name.startswith(("private_head.", "gate.")))


def domain_moments(hidden: Tensor, labels: Tensor, detach: bool = True) -> dict:
    """First moments only; no subject features leave the local client.

    Absent classes have no conditional mean, and are skipped during alignment.
    These unnoised moments themselves are not a differential-privacy guarantee.
    """
    if hidden.ndim != 2 or len(hidden) == 0:
        raise ValueError("hidden must have shape [nonempty_batch,hidden_dim]")
    labels = torch.as_tensor(labels, device=hidden.device).reshape(-1)
    if len(labels) != len(hidden):
        raise ValueError("labels and hidden must have the same batch size")
    if not torch.all((labels == 0) | (labels == 1)):
        raise ValueError("Binary labels must be 0/1")
    values = hidden.detach() if detach else hidden
    conditional, counts = {}, {}
    for label in (0, 1):
        mask = labels == label
        counts[label] = int(mask.sum().item())
        if counts[label]:
            conditional[label] = values[mask].mean(dim=0)
    return {"marginal": values.mean(dim=0), "conditional": conditional,
            "counts": counts, "count": len(hidden)}


def domain_alignment_loss(hidden: Tensor, labels: Tensor, reference_moments=None) -> Tensor:
    """Sum marginal + class-conditional L2 mean distances to other clients.

    Implements the (unsquared) norms printed in Eqs.12–14. References can be one
    ``domain_moments`` dictionary, a list of them, or a site->moments mapping;
    references are detached while local hidden representations remain trainable.
    """
    local = domain_moments(hidden, labels, detach=False)
    loss = hidden.sum() * 0
    if reference_moments is None:
        return loss
    if isinstance(reference_moments, Mapping):
        references = [reference_moments] if "marginal" in reference_moments else list(reference_moments.values())
    else:
        references = list(reference_moments)
    for reference in references:
        if reference is None or not reference.get("count", 1):
            continue
        marginal = torch.as_tensor(reference["marginal"], device=hidden.device, dtype=hidden.dtype).detach()
        if marginal.shape != local["marginal"].shape:
            raise ValueError("Reference hidden dimension differs from local model")
        loss = loss + torch.linalg.vector_norm(local["marginal"] - marginal, ord=2)
        for label, mean in local["conditional"].items():
            targets = reference.get("conditional", {})
            target = targets.get(label, targets.get(str(label)))
            if target is not None:
                target = torch.as_tensor(target, device=hidden.device, dtype=hidden.dtype).detach()
                loss = loss + torch.linalg.vector_norm(mean - target, ord=2)
    return loss


# Short alias for runners that call the operation rather than its loss name.
domain_alignment = domain_alignment_loss
