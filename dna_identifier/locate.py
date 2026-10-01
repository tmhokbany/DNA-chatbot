"""
Finds where a query sequence sits within its predicted reference genome.
The classifier only says which organism a fragment most resembles by k-mer
composition; it doesn't align anything. This module uses local (Smith-Waterman-style) 
alignment of the query against that organism's reference sequence, in both
orientations, and reports the matching coordinates, percent identity, and any 
annotated gene(s) the match falls inside.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from Bio.Align import PairwiseAligner

from .sequence_utils import reverse_complement, strip_ambiguous

@dataclass
class LocationMatch:
    ref_start: int  # 1-based, inclusive
    ref_end: int  # 1-based, inclusive
    ref_length: int
    strand: str  # "+" or "-"
    percent_identity: float
    genes: list[str] = field(default_factory=list)

def _make_aligner() -> PairwiseAligner:
    aligner = PairwiseAligner()
    aligner.mode = "local"
    aligner.match_score = 2
    aligner.mismatch_score = -1
    aligner.open_gap_score = -6
    aligner.extend_gap_score = -1
    return aligner

_ALIGNER = _make_aligner()

def _align_one(query: str, reference: str):
    alignments = _ALIGNER.align(reference, query)
    alignment = alignments[0]
    ref_blocks, query_blocks = alignment.aligned

    ref_start = int(ref_blocks[0][0]) + 1
    ref_end = int(ref_blocks[-1][1])

    aligned_len = 0
    matches = 0
    for (rs, re_), (qs, qe) in zip(ref_blocks, query_blocks):
        rs, re_, qs, qe = int(rs), int(re_), int(qs), int(qe)
        aligned_len += re_ - rs
        r_chunk = reference[rs:re_]
        q_chunk = query[qs:qe]
        matches += sum(1 for a, b in zip(r_chunk, q_chunk) if a == b)

    pct_identity = (matches / aligned_len * 100) if aligned_len else 0.0
    return alignment.score, ref_start, ref_end, pct_identity


def locate(query: str, reference: str, genes: list[dict] | None = None) -> LocationMatch:
    """Aligns `query` (either strand) against `reference` and returns the
    best-matching coordinates, percent identity, and overlapping gene names."""
    query = strip_ambiguous(query.upper())

    fwd_score, fwd_start, fwd_end, fwd_pid = _align_one(query, reference)
    rc_query = reverse_complement(query)
    rev_score, rev_start, rev_end, rev_pid = _align_one(rc_query, reference)

    if rev_score > fwd_score:
        start, end, pct_identity, strand = rev_start, rev_end, rev_pid, "-"
    else:
        start, end, pct_identity, strand = fwd_start, fwd_end, fwd_pid, "+"

    overlapping_genes = []
    for gene in genes or []:
        if gene["start"] <= end and gene["end"] >= start:
            overlapping_genes.append(gene["product"])

    return LocationMatch(
        ref_start=start,
        ref_end=end,
        ref_length=len(reference),
        strand=strand,
        percent_identity=round(pct_identity, 1),
        genes=overlapping_genes,
    )
