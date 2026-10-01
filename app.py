"""Streamlit web chat UI for the DNA sequence identifier.

Usage:
    streamlit run app.py
"""
from __future__ import annotations

import os

import streamlit as st

from chatbot import format_prediction, read_sequences_from_input
from dna_identifier.model import SequenceIdentifier
from dna_identifier.reference_data import load_gene_annotations, load_reference_sequences

MODEL_PATH = "models/species_identifier.joblib"
REFERENCES_PATH = "data/reference_sequences.fasta"
GENE_ANNOTATIONS_PATH = "data/gene_annotations.json"

st.set_page_config(page_title="SeqBot: A ML DNA Species Identifier")
st.title("SeqBot")
st.caption(
    "Chat with a k-mer + RandomForest classifier trained on real NCBI reference "
    "genomes, with alignment-based genome-location lookup."
)

@st.cache_resource
def load_model() -> SequenceIdentifier | None:
    if not os.path.exists(MODEL_PATH):
        return None
    return SequenceIdentifier.load(MODEL_PATH)

@st.cache_resource
def load_references() -> dict[str, str]:
    return load_reference_sequences(REFERENCES_PATH)


@st.cache_resource
def load_genes() -> dict[str, list[dict]]:
    return load_gene_annotations(GENE_ANNOTATIONS_PATH)

model = load_model()

if model is None:
    st.error(
        f"No trained model found at `{MODEL_PATH}`. Run `python train_model.py` "
        "from the project root first."
    )
    st.stop()

references = load_references()
gene_annotations = load_genes()

with st.sidebar:
    st.subheader("Organisms this model knows")
    for name in sorted(model.classes_):
        st.write(f"- {name}")
    st.divider()
    st.caption(
        "Demo model trained on one NCBI RefSeq reference per organism (a 16S "
        "rRNA gene for bacteria, a complete genome for the Dengue serotypes), "
        "augmented into many short training fragments. Location matches use a "
        "real Biopython local alignment against that reference, not a guess. "
        "Not a substitute for a real BLAST/NCBI search — see README for details."
    )

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hi! Paste a DNA sequence (raw bases or a FASTA record) and I'll "
                "predict which organism it's most likely from and where in its "
                "reference genome the match falls."
            ),
        }
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Paste a DNA sequence or FASTA record..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    records = read_sequences_from_input(prompt)
    reply_parts = [
        format_prediction(label, seq, model, references, gene_annotations)
        for label, seq in records
    ]
    reply = "\n\n---\n\n".join(reply_parts)

    with st.chat_message("assistant"):
        st.markdown(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})
