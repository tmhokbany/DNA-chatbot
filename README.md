# ML DNA Sequence Chatbot Identifier AKA SeqBot

A chatbot that reads a raw DNA sequence (or FASTA record) and predicts which
bacterial species it most likely came from, using a scikit-learn classifier
trained on k-mer composition features. 
Available as both a terminal chatbot and a Streamlit web chat UI!

# Preview
```
you> ATTGAACGCTGGCGGCAGGCCTAACACATGCAAGTCGAACGGTAACAGG...
SeqBot: 'pasted sequence': 1467 bp, GC content 54.3%
SeqBot: Best match: Escherichia coli  (91.0% confidence)
SeqBot: Top matches:
    Escherichia coli             91.0%
    Salmonella enterica           6.0%
    Klebsiella pneumoniae         2.0%
```

## How does it work?

1. **Reference data** (`data/reference_16s.fasta`) one real, full-length
   16S rRNA RefSeq sequence per species, pulled from NCBI for 14 common
   bacterial species. See [`data/SOURCES.md`](data/SOURCES.md) for the exact
   accessions.
2. **Augmentation** (`DNA identifier/augment.py`) each reference genome per species is cut into hundreds of randomized
   250-500bp fragments (random start position, ~1% point-mutation rate,
   random strand) to simulate real partial sequencing reads and give the
   classifier enough examples per class to generalize instead of memorize.
3. **Features** (`DNA identifier/features.py`) every fragment is converted
   into a normalized k-mer (default k=4, 256 features) frequency vector.
   K-mer composition is a standard, alignment-free way to fingerprint a DNA
   sequence regardless of length or read direction.
4. **Model** (`DNA identifier/model.py`) a `RandomForestClassifier` maps
   k-mer vectors to species labels, wrapped in a scikit-learn `Pipeline` so
   featurization and prediction travel together in one saved artifact.
5. **Chat interfaces**`chatbot.py` (terminal) and `app.py` (Streamlit)
   both call the same `DNA identifier` package, so predictions are identical
   in either UI.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate       
pip install -r requirements.txt
```

## Train the model

```bash
python train_model.py
```

This builds the augmented training set, fits the classifier, prints a held-out accuracy report, and saves the pipeline to `models/species_identifier.joblib`. Re-run it any time after editing `data/reference_16s.fasta` or the augmentation parameters.

## Chatbot terminal

```bash
python chatbot.py
```

Paste raw bases, a full FASTA record, or a path to a `.fasta` file. Type
`help` for commands, `quit` to exit.

## Chatbot web UI

```bash
streamlit run app.py
```

Opens a browser-based chat interface backed by the same trained model.

## Run the tests

```bash
pytest
```

## Species this model knows

Escherichia coli, Bacillus subtilis, Staphylococcus aureus, Pseudomonas
aeruginosa, Salmonella enterica, Streptococcus pyogenes, Lactobacillus
acidophilus, Mycobacterium tuberculosis, Clostridioides difficile,
Enterococcus faecalis, Vibrio cholerae, Klebsiella pneumoniae, Helicobacter
pylori, Listeria monocytogenes.

Anything else will get force-matched to whichever of these it resembles
most. 
Note the confidence score, and treat low-confidence predictions as
"unknown," not as a real identification.

## Limitations

This is a learning/demo project, not a diagnostic or research tool :octocat: :

- Only 14 species, one reference strain each, real-world diversity within a
  species (or misidentified NCBI records) isn't represented.
- The training fragments are all derived from a single sequence per species,
  so accuracy numbers reflect augmented-fragment classification, not
  independent biological replicates.
- For real species identification from Sanger/16S data, BLAST the sequence
  against NCBI's curated 16S ribosomal RNA database.
