#!/usr/bin/env python3
"""Train the k-mer + RandomForest classifier from data/reference_sequences.fasta
and save the fitted pipeline to models/species_identifier.joblib.

Usage:
    python train_model.py [--fasta data/reference_sequences.fasta] [--out models/species_identifier.joblib]
"""
from __future__ import annotations

import argparse

from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from dna_identifier.augment import build_training_set
from dna_identifier.model import SequenceIdentifier
from dna_identifier.reference_data import load_reference_sequences


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fasta", default="data/reference_sequences.fasta")
    parser.add_argument("--out", default="models/species_identifier.joblib")
    parser.add_argument("--fragments-per-class", type=int, default=80)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    references = load_reference_sequences(args.fasta)
    print(f"Loaded {len(references)} reference species from {args.fasta}:")
    for label, seq in references.items():
        print(f"  - {label}: {len(seq)} bp")

    sequences, labels = build_training_set(
        references, fragments_per_class=args.fragments_per_class
    )
    print(f"\nBuilt {len(sequences)} training fragments via sliding-window augmentation.")

    X_train, X_test, y_train, y_test = train_test_split(
        sequences, labels, test_size=0.25, random_state=42, stratify=labels
    )

    model = SequenceIdentifier(k=args.k)
    model.fit(X_train, y_train)

    y_pred = [model.predict(seq).label for seq in X_test]
    acc = accuracy_score(y_test, y_pred)
    print(f"\nHeld-out fragment accuracy: {acc:.3f}\n")
    print(classification_report(y_test, y_pred, zero_division=0))

    model.save(args.out)
    print(f"Saved trained model to {args.out}")


if __name__ == "__main__":
    main()
