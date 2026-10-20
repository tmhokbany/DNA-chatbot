"""
Finds where a query sequence sits within a reference genome.
The classifier only says which organism a fragment most resembles by k-mer
composition; it doesn't align anything. This module uses local (Smith-Waterman-style)
alignment of the query against a reference sequence, in both orientations, and
reports the matching coordinates, percent identity, query coverage, and any
annotated gene(s) the match falls inside.

Aligning a read against a whole 30kb genome with full dynamic programming is
slow (seconds per read), so this works like BLAST's seed-and-extend: shared
exact 11-mers locate the strand and diagonal the read sits on, and only that
window of the reference is aligned. Long queries are aligned in chunks so the
cost stays linear in query length.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache

from Bio.Align import PairwiseAligner

from .sequence_utils import reverse_complement

SEED_K = 11
CHUNK_LEN = 500
FULL_ALIGN_MAX_CELLS = 2_500_000
STRAND_MIN_SEEDS = 20
_MAX_SEED_OCCURRENCES = 20  

@dataclass
class LocationMatch:
    ref_start: int  
    ref_end: int  
    ref_length: int
    strand: str  
    percent_identity: float  
    genes: list[str] = field(default_factory=list)
    query_coverage: float = 100.0  
    score: float = 0.0
    aligned_columns: int = 0

def _make_aligner() -> PairwiseAligner:
    aligner = PairwiseAligner()
    aligner.mode = "local"
    aligner.match_score = 2
    aligner.mismatch_score = -1
    aligner.open_gap_score = -6
    aligner.extend_gap_score = -1
    return aligner

_ALIGNER = _make_aligner()

@dataclass
class _Hit:
    score: float
    ref_start: int  
    ref_end: int  
    query_span: int
    matches: int
    columns: int

def _strip_non_acgt(seq: str) -> str:
    return "".join(b for b in seq if b in "ACGT")

def _align_one(query: str, reference: str, offset: int = 0) -> _Hit | None:
    if not query or not reference:
        return None
    alignments = _ALIGNER.align(reference, query)
    try:
        alignment = alignments[0]
    except (IndexError, ValueError):
        return None
    if alignment.score <= 0:
        return None
    ref_blocks, query_blocks = alignment.aligned
    if len(ref_blocks) == 0:
        return None

    matches = 0
    columns = 0
    prev_re = prev_qe = None
    for (rs, re_), (qs, qe) in zip(ref_blocks, query_blocks):
        rs, re_, qs, qe = int(rs), int(re_), int(qs), int(qe)
        if prev_re is not None:
            columns += (rs - prev_re) + (qs - prev_qe) 
        columns += re_ - rs
        matches += sum(1 for a, b in zip(reference[rs:re_], query[qs:qe]) if a == b)
        prev_re, prev_qe = re_, qe

    return _Hit(
        score=float(alignment.score),
        ref_start=int(ref_blocks[0][0]) + offset,
        ref_end=int(ref_blocks[-1][1]) + offset,
        query_span=int(query_blocks[-1][1]) - int(query_blocks[0][0]),
        matches=matches,
        columns=columns,
    )

@lru_cache(maxsize=64)
def _seed_index(reference: str) -> dict[str, list[int]]:
    index: dict[str, list[int]] = {}
    for i in range(len(reference) - SEED_K + 1):
        index.setdefault(reference[i : i + SEED_K], []).append(i)
    return index

def _seed_diagonals(query: str, index: dict[str, list[int]]) -> list[int]:
    diagonals = []
    for i in range(len(query) - SEED_K + 1):
        positions = index.get(query[i : i + SEED_K])
        if positions and len(positions) <= _MAX_SEED_OCCURRENCES:
            diagonals.extend(p - i for p in positions)
    return diagonals

def seed_hits(query: str, reference: str) -> int:
    """Number of query 11-mers (either strand) found in `reference`: a cheap
    pre-alignment estimate of how related two sequences are."""
    query = query.upper()
    index = _seed_index(reference)
    hits = 0
    for q in (query, reverse_complement(query)):
        hits = max(hits, sum(1 for i in range(len(q) - SEED_K + 1) if q[i : i + SEED_K] in index))
    return hits

def _best_diagonal(diagonals: list[int], bin_size: int = 64) -> int:
    bins = Counter(d // bin_size for d in diagonals)
    best_bin = max(bins, key=lambda b: bins[b] + bins.get(b - 1, 0) + bins.get(b + 1, 0))
    nearby = sorted(d for d in diagonals if abs(d // bin_size - best_bin) <= 1)
    return nearby[len(nearby) // 2]

def _align_windowed(query: str, reference: str, index: dict[str, list[int]]) -> _Hit | None:
    """Aligns the query in chunks, each against the reference window its seeds point to."""
    total = None
    n_chunks = -(-len(query) // CHUNK_LEN)
    chunk_len = -(-len(query) // n_chunks) 
    for start in range(0, len(query), chunk_len):
        chunk = query[start : start + chunk_len]
        if len(chunk) < SEED_K:
            continue
        diagonals = _seed_diagonals(chunk, index)
        if not diagonals:
            continue
        diagonal = _best_diagonal(diagonals)
        margin = 50 + len(chunk) // 10
        lo = max(0, diagonal - margin)
        hi = min(len(reference), diagonal + len(chunk) + margin)
        hit = _align_one(chunk, reference[lo:hi], offset=lo)
        if hit is None:
            continue
        if total is None:
            total = hit
        else:
            total = _Hit(
                score=total.score + hit.score,
                ref_start=min(total.ref_start, hit.ref_start),
                ref_end=max(total.ref_end, hit.ref_end),
                query_span=total.query_span + hit.query_span,
                matches=total.matches + hit.matches,
                columns=total.columns + hit.columns,
            )
    return total

def locate(query: str, reference: str, genes: list[dict] | None = None) -> LocationMatch | None:
    """Aligns `query` (either strand) against `reference` and returns the
    best-matching coordinates, percent identity, query coverage and
    overlapping gene names, or None if nothing aligns at all."""
    query = _strip_non_acgt(query.upper())
    if not query:
        return None

    index = _seed_index(reference)
    rc_query = reverse_complement(query)
    fwd_seeds = len(_seed_diagonals(query, index))
    rev_seeds = len(_seed_diagonals(rc_query, index))

    if fwd_seeds or rev_seeds:
        # only spend alignment time on strands with real seed support
        strong = max(fwd_seeds, rev_seeds)
        floor = 0.1 * strong if strong >= STRAND_MIN_SEEDS else 0
        fwd = _align_windowed(query, reference, index) if fwd_seeds > floor else None
        rev = _align_windowed(rc_query, reference, index) if rev_seeds > floor else None
    elif len(query) * len(reference) <= FULL_ALIGN_MAX_CELLS:
        fwd = _align_one(query, reference)
        rev = _align_one(rc_query, reference)
    else:
        return None

    candidates = [(h, s) for h, s in ((fwd, "+"), (rev, "-")) if h is not None]
    if not candidates:
        return None
    if len(candidates) == 2 and rev.score == fwd.score:
        hit, strand = (rev, "-") if rev_seeds > fwd_seeds else (fwd, "+")
    else:
        hit, strand = max(candidates, key=lambda c: c[0].score)

    start, end = hit.ref_start + 1, hit.ref_end
    overlapping_genes = []
    for gene in genes or []:
        if gene["start"] <= end and gene["end"] >= start:
            overlapping_genes.append(gene["product"])

    return LocationMatch(
        ref_start=start,
        ref_end=end,
        ref_length=len(reference),
        strand=strand,
        percent_identity=round(hit.matches / hit.columns * 100, 1) if hit.columns else 0.0,
        genes=overlapping_genes,
        query_coverage=round(min(100.0, hit.query_span / len(query) * 100), 1),
        score=hit.score,
        aligned_columns=hit.columns,
    )
