"""K-mer composition features: the standard way to turn a variable-length DNA
sequence into a fixed-length numeric vector for a classifier, since raw
sequence length varies but k-mer frequency profiles are comparable across
sequences of any length."""
from __future__ import annotations

import itertools

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from .sequence_utils import strip_ambiguous

_BASES = "ACGT"
class KmerFeaturizer(BaseEstimator, TransformerMixin):
    """Converts DNA sequences into normalized k-mer frequency vectors."""

    def __init__(self, k: int = 4):
        self.k = k

    def _vocabulary(self) -> list[str]:
        return ["".join(p) for p in itertools.product(_BASES, repeat=self.k)]

    def fit(self, X, y=None):
        self.vocabulary_ = self._vocabulary()
        self.index_ = {kmer: i for i, kmer in enumerate(self.vocabulary_)}
        return self

    def transform(self, X) -> np.ndarray:
        vectors = np.zeros((len(X), len(self.vocabulary_)), dtype=np.float64)
        for row, seq in enumerate(X):
            seq = strip_ambiguous(seq.upper())
            n_kmers = len(seq) - self.k + 1
            if n_kmers <= 0:
                continue
            for i in range(n_kmers):
                kmer = seq[i : i + self.k]
                idx = self.index_.get(kmer)
                if idx is not None:
                    vectors[row, idx] += 1
            vectors[row] /= n_kmers
        return vectors

    def get_feature_names_out(self, input_features=None):
        return np.array(self.vocabulary_)
