# SeqBot: a ML DNA Sequence Chatbot Identifier

A chatbot that reads a raw DNA sequence (or FASTA record) and predicts which
organism it most likely came from, using a scikit-learn classifier trained
on k-mer composition features — then aligns the sequence against that
organism's reference genome to report *where* it matches. Available as both
a terminal chatbot and a Streamlit web chat UI.

```
you> ATTGAACGCTGGCGGCAGGCCTAACACATGCAAGTCGAACGGTAACAGG...
SeqBot: 'pasted sequence': 1467 bp, GC content 54.3%
SeqBot: Best match: Escherichia coli  (91.0% confidence)
SeqBot: Top matches:
    Escherichia coli             91.0%
    Salmonella enterica           6.0%
    Klebsiella pneumoniae         2.0%
SeqBot: Location: reference positions 1-1467 of 1467 bp (+ strand), 100.0% identity
```

## How does it work

1. **Reference data**: (`data/reference_sequences.fasta`) — one real NCBI
   RefSeq reference per organism: the 16S rRNA gene for 15 bacterial
   species (including *Brucella melitensis*), the complete ~10.7kb genome
   for all four Dengue virus serotypes, and the complete ~30.1kb genome for
   MERS-CoV (the Saudi Arabia isolate, HCoV-EMC/2012). See
   [`data/SOURCES.md`](data/SOURCES.md) for exact accessions and why
   viruses are represented differently (no 16S gene in an RNA virus).
2. **Augmentation**: (`dna_identifier/augment.py`) — since there's only one
   reference sequence per organism, each one is cut into hundreds of
   randomized 250–500bp fragments (random start position, ~1%
   point-mutation rate, random strand) to simulate real partial sequencing
   reads and give the classifier enough examples per class to generalize
   instead of memorize.
3. **Features**: (`dna_identifier/features.py`) — every fragment is converted
   into a normalized k-mer (default k=5, 1024 features) frequency vector.
   K-mer composition is a standard, alignment-free way to fingerprint a DNA
   sequence regardless of its length or read direction.
4. **Model**: (`dna_identifier/model.py`) — a `RandomForestClassifier` maps
   k-mer vectors to organism labels, wrapped in a scikit-learn `Pipeline` so
   featurization and prediction travel together in one saved artifact.
5. **Genome location**: (`dna_identifier/locate.py`) — once the classifier
   picks an organism, the query is locally aligned (Biopython
   `PairwiseAligner`, both strands) against *that organism's actual
   reference sequence*. This is a real Smith-Waterman-style alignment, not
   another model guess — it reports the matched start/end coordinates,
   percent identity, and (for the viral genomes, via
   `data/gene_annotations.json`) which annotated gene/protein the match
   falls inside, e.g. "envelope protein E" or "spike glycoprotein S".
6. **Chat interfaces**: `chatbot.py` (terminal) and `app.py` (Streamlit)
   both call the same `dna_identifier` package, so predictions and location
   output are identical in either UI.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python train_model.py
```

The trained model isn't committed to the repo (it's a generated binary that
doesn't belong in version control), so that last step is required before the
chatbot will run. It builds the augmented training set, fits the classifier,
prints a held-out accuracy report, and saves the pipeline to
`models/species_identifier.joblib`. Re-run it any time after editing
`data/reference_sequences.fasta` or the augmentation parameters.

## Setup in VS Code

VS Code works great for this project — here's the exact sequence:

1. **Open the folder.** `File > Open Folder...` and pick the `DNA-chatbot`
   project folder (not a parent folder). VS Code needs to see `chatbot.py`,
   `requirements.txt`, etc. directly inside the opened folder.
2. **Install the Python extension**, if you haven't already (Microsoft's
   official one — search "Python" in the Extensions panel, `Cmd+Shift+X`).
   This gives you the interpreter picker, the Testing panel, and inline
   run/debug buttons.
3. **Create the virtual environment from VS Code's integrated terminal**
   (`` Ctrl+` `` or `Terminal > New Terminal`):
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
4. **Point VS Code at that venv.** Open the Command Palette
   (`Cmd+Shift+P`), run `Python: Select Interpreter`, and choose the one
   listed under `./venv/bin/python` (VS Code usually suggests it
   automatically once `venv/` exists — look for a popup in the bottom
   right). You'll know it worked when the Python version in the bottom
   status bar shows the venv path, not a system Python.
5. **Train the model** — same command, now in VS Code's terminal (which is
   already using the venv you selected):
   ```bash
   python train_model.py
   ```
6. **Run it.** Either keep using the integrated terminal (`python
   chatbot.py`, `streamlit run app.py`), or open `chatbot.py` and click the
   ▷ "Run Python File" button in the top-right of the editor — same thing,
   one click. Streamlit specifically needs the terminal (`streamlit run
   app.py`), since the ▷ button runs a file with plain `python`, not
   through Streamlit's launcher.
7. **(Optional) Debugging.** Set a breakpoint by clicking left of a line
   number in `chatbot.py`, then `Run > Start Debugging` (`F5`) instead of
   the ▷ button — useful if you want to step through what happens when a
   sequence comes in, e.g. for a Q&A after a demo.

If `python3` isn't found at all, install Python 3.10+ from
[python.org](https://www.python.org/downloads/) first (VS Code doesn't
bundle a Python interpreter, it just drives whichever one is installed).

## Chat — terminal

```bash
python chatbot.py
```

Paste raw bases, a full FASTA record, or a path to a `.fasta` file. Type
`help` for commands, `quit` to exit.

## Chat — web UI

```bash
streamlit run app.py
```

Opens a browser-based chat interface backed by the same trained model —
this is the better option for a live demo, since it looks like a normal
chat app rather than a terminal.

## Run the tests

```bash
pytest
```

12 tests covering feature extraction, sequence utilities, model
training/save/load, and alignment correctness.

## Organisms this model knows

**Bacteria** (16S rRNA gene): Escherichia coli, Bacillus subtilis,
Staphylococcus aureus, Pseudomonas aeruginosa, Salmonella enterica,
Streptococcus pyogenes, Lactobacillus acidophilus, Mycobacterium
tuberculosis, Clostridioides difficile, Enterococcus faecalis, Vibrio
cholerae, Klebsiella pneumoniae, Helicobacter pylori, Listeria
monocytogenes, Brucella melitensis.

**Viruses** (complete genome): Dengue virus 1, Dengue virus 2, Dengue virus
3, Dengue virus 4, MERS-CoV (Saudi Arabia).

Anything else will get force-matched to whichever of these it resembles
most. Note the confidence score, and treat low-confidence predictions as
"unknown," not as a real identification.

## Limitations

This is a learning/demo project, not a diagnostic or research tool:

- Only 20 organisms, one reference sequence each — real-world diversity
  within a species/serotype (or misidentified NCBI records) isn't
  represented.
- The training fragments are all derived from a single sequence per
  organism, so accuracy numbers reflect augmented-fragment classification,
  not independent biological replicates.
- The genome-location alignment is a real pairwise alignment, but it's only
  run against the *predicted* organism's reference — if the classifier picks
  the wrong organism, the reported location is meaningless for the actual
  source sequence.
- For real species/strain identification from sequencing data, BLAST the
  sequence against NCBI's curated databases (see the companion
  [16S-rRNA-pipeline](https://github.com/tmhokbany/16S-rRNA-pipeline) repo
  for a full trim/assemble/cluster pipeline that hands off to BLAST).
