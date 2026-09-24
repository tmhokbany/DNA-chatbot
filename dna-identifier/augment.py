"""
Real identification works on partial reads (e.g. Sanger or short-amplicon data), so training samples are built the same way: random sliding-window fragments of the reference gene, with light point mutations and random strand orientation to simulate sequencing noise.
"""
from __future__ import annotations

import random

from .sequence_utils import reverse_complement

_BASES = "ACGT"

def mutate(seq: str, mutation_rate: float, rng: random.Random) -> str:
    if mutation_rate <= 0:
        return seq
    chars = list(seq)
    for i, base in enumerate(chars):
        if rng.random() < mutation_rate:
            chars[i] = rng.choice([b for b in _BASES if b != base])
    return "".join(chars)

def fragment(seq: str, min_len: int, max_len: int, rng: random.Random) -> str:
    frag_len = min(len(seq), rng.randint(min_len, max_len))
    if frag_len >= len(seq):
        return seq
    start = rng.randint(0, len(seq) - frag_len)
    return seq[start : start + frag_len]

def build_training_set(
    reference_sequences: dict[str, str],
    fragments_per_class: int = 60,
    min_len: int = 250,
    max_len: int = 500,
    mutation_rate: float = 0.01,
    reverse_complement_fraction: float = 0.5,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    """Returns (sequences, labels) built from full-length references."""
    rng = random.Random(seed)
    sequences: list[str] = []
    labels: list[str] = []

    for label, full_seq in reference_sequences.items():
        for _ in range(fragments_per_class):
            frag = fragment(full_seq, min_len, max_len, rng)
            frag = mutate(frag, mutation_rate, rng)
            if rng.random() < reverse_complement_fraction:
                frag = reverse_complement(frag)
            sequences.append(frag)
            labels.append(label)

    return sequences, labels
