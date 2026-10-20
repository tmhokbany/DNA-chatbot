"""
K-mer composition features: the standard way to turn a variable-length DNA
sequence into a fixed-length numeric vector for a classifier, since raw
sequence length varies but k-mer frequency profiles are comparable across
sequences of any length.

With `canonical=True` each k-mer is merged with its reverse complement, so a
read and its reverse complement produce the same vector. Sequencing reads come
off either strand, so this halves what the classifier has to learn.
"""
from __future__ import annotations

import itertools

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

_BASES = "ACGT"

# byte value -> 0..3 for A/C/G/T, -1 for anything else (N, IUPAC codes, ...)
_CODE = np.full(256, -1, dtype=np.int64)
for _i, _b in enumerate(_BASES):
    _CODE[ord(_b)] = _i


def _reverse_complement_code(code: int, k: int) -> int:
    rc = 0
    for _ in range(k):
        rc = (rc << 2) | (3 - (code & 3))
        code >>= 2
    return rc


class KmerFeaturizer(BaseEstimator, TransformerMixin):

    def __init__(self, k: int = 4, canonical: bool = False):
        self.k = k
        self.canonical = canonical

    def fit(self, X, y=None):
        all_kmers = ["".join(p) for p in itertools.product(_BASES, repeat=self.k)]
        codes = np.arange(4 ** self.k)
        if self.canonical:
            rc = np.array([_reverse_complement_code(int(c), self.k) for c in codes])
            codes = np.minimum(codes, rc)
        kept, column = np.unique(codes, return_inverse=True)
        self.vocabulary_ = [all_kmers[c] for c in kept]
        # maps every k-mer (including non-canonical ones) to its feature column
        self.index_ = {kmer: int(column[i]) for i, kmer in enumerate(all_kmers)}
        self._column = column
        return self

    def transform(self, X) -> np.ndarray:
        n_features = len(self.vocabulary_)
        vectors = np.zeros((len(X), n_features), dtype=np.float64)
        weights = 4 ** np.arange(self.k - 1, -1, -1)
        for row, seq in enumerate(X):
            encoded = _CODE[np.frombuffer(seq.upper().encode("ascii", "replace"), dtype=np.uint8)]
            if len(encoded) < self.k:
                continue
            windows = np.lib.stride_tricks.sliding_window_view(encoded, self.k)
        
            windows = windows[(windows >= 0).all(axis=1)]
            if len(windows) == 0:
                continue
            counts = np.bincount(self._column[windows @ weights], minlength=n_features)
            vectors[row] = counts / len(windows)
        return vectors

    def get_feature_names_out(self, input_features=None):
        return np.array(self.vocabulary_)
