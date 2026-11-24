# SPDX-License-Identifier: MIT
"""Parameter perturbation and real packed Paillier aggregation.

This is a single-process research simulator: the coordinator also holds the key.
It does NOT claim a distributed threat model or a formal epsilon-DP guarantee.
"""
from __future__ import annotations

import math
import secrets
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch


def perturb_shared_state(state: dict[str, torch.Tensor], noise_scale: float = 0.01,
                         clip_norm: float = 10.0, generator=None):
    """Clip the shared parameter vector then add Laplace noise (Eq. 17 formulation).

    noise_scale is the direct noise scale, not an accounted epsilon budget.
    """
    if noise_scale < 0 or clip_norm <= 0:
        raise ValueError("noise_scale>=0 and clip_norm>0 required")
    norm = math.sqrt(sum(float(v.detach().double().square().sum()) for v in state.values()))
    factor = min(1.0, clip_norm / max(norm, 1e-12))
    result = {}
    rng = generator if generator is not None else np.random.default_rng()
    for name, value in state.items():
        clean = value.detach().cpu().clone() * factor
        if noise_scale:
            noise = rng.laplace(0.0, noise_scale, size=tuple(clean.shape))
            clean += torch.as_tensor(noise, dtype=clean.dtype)
        result[name] = clean
    return result


def _probable_prime(n: int, rounds: int = 32) -> bool:
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0:
            return n == p
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for _ in range(rounds):
        a = secrets.randbelow(n - 3) + 2
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True


def _prime(bits: int) -> int:
    while True:
        candidate = secrets.randbits(bits) | (1 << (bits - 1)) | 1
        if _probable_prime(candidate):
            return candidate


@dataclass
class PaillierKey:
    n: int
    lam: int
    mu: int

    @classmethod
    def generate(cls, bits: int = 2048):
        if bits < 256:
            raise ValueError("Use >=256 bits for tests; >=2048 for demonstrations")
        p, q = _prime(bits // 2), _prime(bits - bits // 2)
        while p == q:
            q = _prime(bits - bits // 2)
        n = p * q
        lam = math.lcm(p - 1, q - 1)
        return cls(n=n, lam=lam, mu=pow(lam, -1, n))

    def encrypt(self, message: int) -> int:
        if not 0 <= message < self.n:
            raise ValueError("Plaintext outside Paillier message space")
        r = secrets.randbelow(self.n - 1) + 1
        while math.gcd(r, self.n) != 1:
            r = secrets.randbelow(self.n - 1) + 1
        return ((1 + self.n * message) * pow(r, self.n, self.n * self.n)) % (self.n * self.n)

    def decrypt(self, ciphertext: int) -> int:
        return (((pow(ciphertext, self.lam, self.n * self.n) - 1) // self.n) * self.mu) % self.n

    def add(self, a: int, b: int) -> int:
        return (a * b) % (self.n * self.n)


class PackedPaillierAggregator:
    """Uniform client mean, fixed point, integer packing without carry.

    Quantization error <=0.5/scale per scalar. Ciphertext multiplication performs
    addition without individual plaintext decoding. Keys never leave this process.
    """
    def __init__(self, bits: int = 2048, scale: int = 1_000_000):
        self.key = PaillierKey.generate(bits)
        self.scale = scale
        self.slot_bits = 40
        self.offset = 1 << 30
        self.base = 1 << self.slot_bits
        self.slots = (self.key.n.bit_length() - 2) // self.slot_bits
        self.last_wire_bytes = 0

    def aggregate(self, vectors: Sequence[np.ndarray]) -> np.ndarray:
        if not vectors:
            raise ValueError("No client vectors")
        n_clients = len(vectors)
        length = len(vectors[0])
        if any(len(v) != length for v in vectors):
            raise ValueError("Different shared parameter shapes")
        if n_clients * 2 * self.offset >= self.base:
            raise ValueError("Too many clients for carry-free slots")
        quantized = [np.rint(np.asarray(v, dtype=np.float64) * self.scale).astype(np.int64) for v in vectors]
        if any(np.any(np.abs(v) >= self.offset) for v in quantized):
            raise ValueError("Model values exceed encrypted packing range")
        result = np.empty(length, dtype=np.float64)
        cipher_bytes = (2 * self.key.n.bit_length() + 7) // 8
        self.last_wire_bytes = 0
        for start in range(0, length, self.slots):
            end = min(start + self.slots, length)
            aggregate_cipher = 1
            for v in quantized:
                message = sum((int(v[j]) + self.offset) << ((j - start) * self.slot_bits)
                              for j in range(start, end))
                aggregate_cipher = self.key.add(aggregate_cipher, self.key.encrypt(message))
                self.last_wire_bytes += cipher_bytes
            plaintext = self.key.decrypt(aggregate_cipher)
            for j in range(start, end):
                digit = plaintext & (self.base - 1)
                result[j] = (digit - n_clients * self.offset) / (n_clients * self.scale)
                plaintext >>= self.slot_bits
        return result.astype(np.float32)


def average_shared_states(states: Sequence[dict[str, torch.Tensor]], encrypted=None):
    """Uniform FedAvg as defined by Eq. 2; never includes private/gate params."""
    if not states:
        raise ValueError("No states to aggregate")
    names = list(states[0])
    if any(list(s) != names for s in states):
        raise ValueError("Inconsistent shared state keys")
    if encrypted is None:
        return {name: torch.stack([s[name].cpu() for s in states]).mean(0) for name in names}
    arrays = [np.concatenate([s[n].cpu().numpy().ravel() for n in names]) for s in states]
    mean = encrypted.aggregate(arrays)
    result, offset = {}, 0
    for name in names:
        template = states[0][name]
        size = template.numel()
        result[name] = torch.from_numpy(mean[offset:offset + size].copy()).reshape(template.shape).to(template.dtype)
        offset += size
    return result
