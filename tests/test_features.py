import numpy as np

from dna_identifier.features import KmerFeaturizer

def test_kmer_vector_shape_and_normalization():
    feat = KmerFeaturizer(k=2)
    feat.fit([])
    X = feat.transform(["ACGTACGT", "AAAAAAAA"])
    assert X.shape == (2, 16)
    assert np.allclose(X.sum(axis=1), 1.0)

def test_kmer_counts_aaaa_sequence():
    feat = KmerFeaturizer(k=2)
    feat.fit([])
    X = feat.transform(["AAAA"])
    aa_index = feat.index_["AA"]
    assert X[0, aa_index] == 1.0

def test_short_sequence_returns_zero_vector():
    feat = KmerFeaturizer(k=4)
    feat.fit([])
    X = feat.transform(["AC"])
    assert X.shape == (1, 256)
    assert X.sum() == 0.0
