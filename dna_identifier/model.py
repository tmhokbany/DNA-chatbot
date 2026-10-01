"""The trained classifier: k-mer features -> RandomForest -> species label,
wrapped with save/load and a friendly top-N prediction helper."""
from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from .features import KmerFeaturizer

@dataclass
class Prediction:
    label: str
    confidence: float
    top_matches: list[tuple[str, float]]

class SequenceIdentifier:
    def __init__(
        self,
        k: int = 5,
        n_estimators: int = 150,
        max_depth: int = 20,
        random_state: int = 42,
    ):
        self.pipeline = Pipeline(
            [
                ("kmers", KmerFeaturizer(k=k)),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
        self.classes_: list[str] | None = None

    def fit(self, sequences: list[str], labels: list[str]) -> "SequenceIdentifier":
        self.pipeline.fit(sequences, labels)
        self.classes_ = list(self.pipeline.named_steps["clf"].classes_)
        return self

    def predict(self, sequence: str, top_n: int = 3) -> Prediction:
        proba = self.pipeline.predict_proba([sequence])[0]
        order = np.argsort(proba)[::-1]
        classes = self.pipeline.named_steps["clf"].classes_
        top_matches = [(classes[i], float(proba[i])) for i in order[:top_n]]
        best_label, best_conf = top_matches[0]
        return Prediction(label=best_label, confidence=best_conf, top_matches=top_matches)

    def save(self, path: str) -> None:
        joblib.dump(self.pipeline, path)

    @classmethod
    def load(cls, path: str) -> "SequenceIdentifier":
        obj = cls()
        obj.pipeline = joblib.load(path)
        obj.classes_ = list(obj.pipeline.named_steps["clf"].classes_)
        return obj
