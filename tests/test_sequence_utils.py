from dna_identifier.sequence_utils import (
    clean_sequence,
    is_valid_dna,
    reverse_complement,
    sequence_stats,
)

def test_clean_sequence_strips_header_and_whitespace():
    raw = ">my seq\nACGT acgt\nNNAC\n"
    assert clean_sequence(raw) == "ACGTACGTNNAC"

def test_is_valid_dna_rejects_short_and_garbage():
    assert not is_valid_dna("ACGT")
    assert not is_valid_dna("ACGTXYZ" * 10)
    assert is_valid_dna("ACGT" * 10)

def test_reverse_complement():
    assert reverse_complement("ACGT") == "ACGT"
    assert reverse_complement("AAGGCC") == "GGCCTT"

def test_sequence_stats_gc_content():
    stats = sequence_stats("GGCC")
    assert stats.length == 4
    assert stats.gc_content == 100.0
