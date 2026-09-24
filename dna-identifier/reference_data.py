"""Loads the bundled reference genomes/genes and their gene-coordinate
annotations, shared by training and by the chat interfaces at inference time."""
from __future__ import annotations

import json
import os

from .sequence_utils import parse_fasta

def load_reference_sequences(fasta_path: str) -> dict[str, str]:
    """Returns {species/virus label: full reference sequence}."""
    references: dict[str, str] = {}
    for header, seq in parse_fasta(fasta_path):
        label = header.split("|")[0].replace("_", " ").strip()
        references[label] = seq
    return references

def load_gene_annotations(json_path: str) -> dict[str, list[dict]]:
    """Returns {label: [{"start": int, "end": int, "product": str}, ...]}.

    Only organisms with sub-genome gene annotations (currently the viral
    genomes) have entries; a bacterial 16S rRNA reference has none because
    the whole reference *is* the gene.
    """
    if not os.path.exists(json_path):
        return {}
    with open(json_path) as fh:
        return json.load(fh)
