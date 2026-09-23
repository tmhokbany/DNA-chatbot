#!/usr/bin/env python3
"""Streamlit web chat UI for the DNA sequence identifier.

Usage:
    streamlit run app.py
"""
from __future__ import annotations

import os

import streamlit as st

from chatbot import format_prediction, read_sequences_from_input
from dna_identifier.model import SequenceIdentifier

MODEL_PATH = "models/species_identifier.joblib"

st.set_page_config(page_title="SeqBot: A DNA Species Identifier", page_icon="🧬🦠")
st.title("🧬🦠 SeqBot")
st.caption("Chat with a k-mer + RandomForest classifier trained on NCBI 16S rRNA reference sequences.")

@st.cache_resource
def load_model() -> SequenceIdentifier | None:
    if not os.path.exists(MODEL_PATH):
        return None
    return SequenceIdentifier.load(MODEL_PATH)

model = load_model()

if model is None:
    st.error(
        f"No trained model found at `{MODEL_PATH}`. Run `python train_model.py` "
        "from the project root first."
    )
    st.stop()

with st.sidebar:
    st.subheader("Species this model knows")
    for name in sorted(model.classes_):
        st.write(f"- {name}")
    st.divider()
    st.caption(
        "Demo model trained on one NCBI RefSeq 16S rRNA record per species, "
        "augmented into many short training fragments. Not a substitute for "
        "a real BLAST/NCBI search — see README for details."
    )

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hi! Paste a DNA sequence (raw bases or a FASTA record) and I'll "
                "predict which bacterial species it's most likely from."
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
    reply_parts = [format_prediction(label, seq, model) for label, seq in records]
    reply = "\n\n---\n\n".join(reply_parts)

    with st.chat_message("assistant"):
        st.markdown(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})
