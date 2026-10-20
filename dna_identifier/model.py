"""The trained classifier: canonical k-mer features -> logistic regression ->
species label, wrapped with save/load and a friendly top-N prediction helper.

A multinomial logistic regression on square-root k-mer frequencies beat the
original RandomForest on every held-out read length (most clearly on short
reads and on the closely related Dengue serotypes), gives far better calibrated
confidences, and is ~100x smaller on disk.
"""
from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from .features import KmerFeaturizer

@dataclass
class Prediction:
    label: str
    confidence: float
    top_matches: list[tuple[str, float]]

class SequenceIdentifier:
    def __init__(
        self,
        k: int = 6,
        canonical: bool = True,
        C: float = 50.0,
        max_iter: int = 3000,
        random_state: int = 42,
    ):
        self.pipeline = Pipeline(
            [
                ("kmers", KmerFeaturizer(k=k, canonical=canonical)),
                # sqrt damps the dominance of very common k-mers so rare,
                # discriminative ones carry more weight in a linear model
                ("sqrt", FunctionTransformer(np.sqrt)),
                (
                    "clf",
                    LogisticRegression(C=C, max_iter=max_iter, random_state=random_state),
                ),
            ]
        )
        self.classes_: list[str] | None = None

    def fit(self, sequences: list[str], labels: list[str]) -> "SequenceIdentifier":
        self.pipeline.fit(sequences, labels)
        self.classes_ = list(self.pipeline.named_steps["clf"].classes_)
        return self

    def predict(self, sequence: str, top_n: int = 3) -> Prediction:
        return self.predict_many([sequence], top_n=top_n)[0]

    def predict_many(self, sequences: list[str], top_n: int = 3) -> list[Prediction]:
        probas = self.pipeline.predict_proba(sequences)
        classes = self.pipeline.named_steps["clf"].classes_
        predictions = []
        for proba in probas:
            order = np.argsort(proba)[::-1]
            top_matches = [(str(classes[i]), float(proba[i])) for i in order[:top_n]]
            best_label, best_conf = top_matches[0]
            predictions.append(
                Prediction(label=best_label, confidence=best_conf, top_matches=top_matches)
            )
        return predictions

    def save(self, path: str) -> None:
        joblib.dump(self.pipeline, path)

    @classmethod
    def load(cls, path: str) -> "SequenceIdentifier":
        obj = cls()
        obj.pipeline = joblib.load(path)
        obj.classes_ = list(obj.pipeline.named_steps["clf"].classes_)
        return obj
