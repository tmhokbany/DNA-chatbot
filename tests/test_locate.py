import random

from dna_identifier.locate import locate
from dna_identifier.sequence_utils import reverse_complement

_rng = random.Random(7)
REFERENCE = "".join(_rng.choice("ACGT") for _ in range(200))

def test_locate_forward_strand_exact_match():
    query = REFERENCE[20:60]
    match = locate(query, REFERENCE)
    assert match.strand == "+"
    assert match.percent_identity == 100.0
    assert match.ref_start == 21  # 1-based
    assert match.ref_end == 60

def test_locate_reverse_strand_match():
    query = reverse_complement(REFERENCE[10:50])
    match = locate(query, REFERENCE)
    assert match.strand == "-"
    assert match.percent_identity == 100.0

def test_locate_reports_overlapping_genes():
    genes = [
        {"start": 1, "end": 30, "product": "gene A"},
        {"start": 31, "end": 100, "product": "gene B"},
    ]
    query = REFERENCE[25:35]
    match = locate(query, REFERENCE, genes=genes)
    assert "gene A" in match.genes
    assert "gene B" in match.genes
