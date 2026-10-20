"""
Real identification works on partial reads (e.g. Sanger or short-amplicon data), so training samples are built the same way:
random sliding-window fragments of the reference gene, with point mutations, small indels and random strand orientation
to simulate sequencing noise and strain-level variation.

The number of fragments per organism scales with reference length, so a 30kb viral genome gets the same sampling depth
as a 1.5kb 16S gene. With a flat count per class the long viral genomes were barely covered, and reads from the unseen
regions were misclassified.
"""
from __future__ import annotations

import math
import random

from .sequence_utils import reverse_complement

_BASES = "ACGT"

def mutate(seq: str, mutation_rate: float, rng: random.Random, indel_rate: float = 0.0) -> str:
    if mutation_rate <= 0 and indel_rate <= 0:
        return seq
    out = []
    for base in seq:
        r = rng.random()
        if r < indel_rate / 2:
            continue  
        if r < indel_rate:
            out.append(rng.choice(_BASES))  
        if rng.random() < mutation_rate:
            out.append(rng.choice([b for b in _BASES if b != base]))
        else:
            out.append(base)
    return "".join(out)

def fragment(seq: str, min_len: int, max_len: int, rng: random.Random) -> str:
    # log-uniform lengths: plenty of short reads, but long amplicons too
    frag_len = int(math.exp(rng.uniform(math.log(min_len), math.log(max_len))))
    frag_len = min(len(seq), frag_len)
    if frag_len >= len(seq):
        return seq
    start = rng.randint(0, len(seq) - frag_len)
    return seq[start : start + frag_len]

def build_training_set(
    reference_sequences: dict[str, str],
    fragments_per_class: int | None = None,
    coverage: float = 30.0,
    min_fragments: int = 300,
    min_len: int = 80,
    max_len: int = 1500,
    mutation_rate: float = 0.03,
    indel_rate: float = 0.01,
    reverse_complement_fraction: float = 0.5,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    """Returns (sequences, labels) built from full-length references.

    Each fragment gets its own substitution and indel rates drawn uniformly
    from [0, mutation_rate] and [0, indel_rate]. If `fragments_per_class` is
    None, each reference is sampled to roughly `coverage`x depth (at least
    `min_fragments` fragments).
    """
    rng = random.Random(seed)
    sequences: list[str] = []
    labels: list[str] = []
    mean_len = (min_len + max_len) / 2

    for label, full_seq in reference_sequences.items():
        if fragments_per_class is None:
            n_fragments = max(min_fragments, int(coverage * len(full_seq) / mean_len))
        else:
            n_fragments = fragments_per_class
        for _ in range(n_fragments):
            frag = fragment(full_seq, min_len, max_len, rng)
            frag = mutate(
                frag,
                rng.uniform(0, mutation_rate),
                rng,
                indel_rate=rng.uniform(0, indel_rate),
            )
            if rng.random() < reverse_complement_fraction:
                frag = reverse_complement(frag)
            sequences.append(frag)
            labels.append(label)

    return sequences, labels
