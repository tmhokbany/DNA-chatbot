#!/usr/bin/env python3
"""Interactive command-line chatbot for DNA sequence identification.

Paste a raw sequence, a FASTA block, or point it at a .fasta file, and the
bot replies with basic sequence stats plus its top species predictions from
the trained k-mer/RandomForest classifier.

Usage:
    python chatbot.py [--model models/species_identifier.joblib]
"""
from __future__ import annotations

import argparse
import os
import sys

from dna_identifier.locate import locate
from dna_identifier.model import SequenceIdentifier
from dna_identifier.reference_data import load_gene_annotations, load_reference_sequences
from dna_identifier.sequence_utils import clean_sequence, is_valid_dna, parse_fasta, sequence_stats

BOT_NAME = "SeqBot"

HELP_TEXT = """\
Commands:
  help          show this message
  stats         show basic info about the last sequence you gave me
  quit / exit   leave the chat

Otherwise, just paste a DNA sequence (raw bases, a FASTA record, or a path
to a .fasta file) and I'll try to identify the species it's from.
"""

def load_model(path: str) -> SequenceIdentifier:
    if not os.path.exists(path):
        print(
            f"{BOT_NAME}: I can't find a trained model at '{path}'.\n"
            "Run `python train_model.py` first to build one from data/reference_sequences.fasta."
        )
        sys.exit(1)
    return SequenceIdentifier.load(path)


def read_sequences_from_input(text: str) -> list[tuple[str, str]]:
    """Returns a list of (label, sequence). Handles raw text, FASTA text, or a file path."""
    stripped = text.strip()
    if os.path.isfile(stripped):
        return [(header or "sequence", seq) for header, seq in parse_fasta(stripped)]
    if stripped.startswith(">"):
        records = []
        header, chunks = None, []
        for line in stripped.splitlines():
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(chunks)))
                header = line[1:].strip()
                chunks = []
            else:
                chunks.append(line.strip())
        if header is not None:
            records.append((header, "".join(chunks)))
        return records
    return [("pasted sequence", clean_sequence(stripped))]

def format_prediction(
    label_name: str,
    seq: str,
    model: SequenceIdentifier,
    references: dict[str, str] | None = None,
    gene_annotations: dict[str, list[dict]] | None = None,
) -> str:
    seq = clean_sequence(seq)
    if not is_valid_dna(seq):
        return (
            f"'{label_name}' ({len(seq)} bp) doesn't look like a clean DNA sequence "
            "to me (too short, or too many non-ACGT characters). I need at least "
            "20bp of mostly A/C/G/T bases."
        )

    stats = sequence_stats(seq)
    pred = model.predict(seq, top_n=3)

    lines = [
        f"'{label_name}': {stats.length} bp, GC content {stats.gc_content}%",
        f"Best match: {pred.label}  ({pred.confidence * 100:.1f}% confidence)",
        "Top matches:",
    ]
    for name, conf in pred.top_matches:
        lines.append(f"    {name:<28} {conf * 100:5.1f}%")

    if pred.confidence < 0.4:
        lines.append(
            "(Low confidence — this may be a species outside my training set, "
            "or too short/noisy a fragment. Consider confirming with BLAST.)"
        )

    reference_seq = (references or {}).get(pred.label)
    if reference_seq:
        genes = (gene_annotations or {}).get(pred.label)
        loc = locate(seq, reference_seq, genes=genes)
        lines.append(
            f"Location: reference positions {loc.ref_start}-{loc.ref_end} of "
            f"{loc.ref_length} bp ({loc.strand} strand), {loc.percent_identity}% identity"
        )
        if loc.genes:
            lines.append(f"Falls within: {', '.join(loc.genes)}")

    return "\n".join(lines)

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/species_identifier.joblib")
    parser.add_argument("--references", default="data/reference_sequences.fasta")
    parser.add_argument("--gene-annotations", default="data/gene_annotations.json")
    args = parser.parse_args()

    model = load_model(args.model)
    references = load_reference_sequences(args.references)
    gene_annotations = load_gene_annotations(args.gene_annotations)
    species_list = ", ".join(sorted(model.classes_))

    print(f"{BOT_NAME}: Hi! I identify bacterial and viral species from a DNA sequence fragment,")
    print(f"{BOT_NAME}: and tell you where in the reference genome it lines up.")
    print(f"{BOT_NAME}: I currently know: {species_list}")
    print(f"{BOT_NAME}: Paste a sequence (or type 'help'). Type 'quit' to leave.\n")

    last_seq = None
    while True:
        try:
            user_input = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{BOT_NAME}: Bye!")
            break

        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit"}:
            print(f"{BOT_NAME}: Bye!")
            break
        if user_input.lower() == "help":
            print(HELP_TEXT)
            continue
        if user_input.lower() == "stats":
            if last_seq is None:
                print(f"{BOT_NAME}: I haven't seen a sequence yet.")
            else:
                stats = sequence_stats(clean_sequence(last_seq))
                print(f"{BOT_NAME}: {stats}")
            continue

        records = read_sequences_from_input(user_input)
        for label, seq in records:
            last_seq = seq
            reply = format_prediction(label, seq, model, references, gene_annotations)
            print(f"{BOT_NAME}: {reply}\n")

if __name__ == "__main__":
    main()
