"""Helpers for parsing, cleaning, and describing raw DNA sequence text."""
from __future__ import annotations

import re
from dataclasses import dataclass

_VALID_BASES = set("ACGT")
_IUPAC_AMBIGUITY = set("RYSWKMBDHVN")

def clean_sequence(raw: str) -> str:
    """Strip whitespace/digits/FASTA headers and uppercase a raw sequence string."""
    lines = [ln for ln in raw.strip().splitlines() if not ln.startswith(">")]
    text = "".join(lines)
    text = re.sub(r"[^A-Za-z]", "", text)
    return text.upper()

def is_valid_dna(seq: str, max_ambiguous_fraction: float = 0.05) -> bool:
    """A sequence is usable if it's non-trivial length and mostly A/C/G/T."""
    if len(seq) < 20:
        return False
    bad = sum(1 for base in seq if base not in _VALID_BASES and base not in _IUPAC_AMBIGUITY)
    ambiguous = sum(1 for base in seq if base in _IUPAC_AMBIGUITY)
    if bad > 0:
        return False
    return (ambiguous / len(seq)) <= max_ambiguous_fraction

def strip_ambiguous(seq: str) -> str:
    """Drop any IUPAC ambiguity codes, keeping only definite A/C/G/T calls."""
    return "".join(base for base in seq if base in _VALID_BASES)


_COMPLEMENT = str.maketrans("ACGT", "TGCA")

def reverse_complement(seq: str) -> str:
    return seq.translate(_COMPLEMENT)[::-1]

@dataclass
class SequenceStats:
    length: int
    gc_content: float
    a_count: int
    c_count: int
    g_count: int
    t_count: int
    ambiguous_bases: int

def sequence_stats(seq: str) -> SequenceStats:
    length = len(seq)
    a, c, g, t = seq.count("A"), seq.count("C"), seq.count("G"), seq.count("T")
    gc = (g + c) / length * 100 if length else 0.0
    ambiguous = length - (a + c + g + t)
    return SequenceStats(
        length=length,
        gc_content=round(gc, 2),
        a_count=a,
        c_count=c,
        g_count=g,
        t_count=t,
        ambiguous_bases=ambiguous,
    )


def parse_fasta(path: str):
    """Yield (header, sequence) pairs from a FASTA file."""
    header = None
    chunks: list[str] = []
    with open(path, "r") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(chunks).upper()
                header = line[1:].strip()
                chunks = []
            else:
                chunks.append(line.strip())
        if header is not None:
            yield header, "".join(chunks).upper()
