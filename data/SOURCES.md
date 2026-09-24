# Reference sequence provenance

`reference_sequences.fasta` contains one real reference sequence per
organism, fetched from NCBI's `nuccore` database via the E-utilities API
(`esearch` + `efetch`.
Bacteria are represented by their 16S ribosomal RNA gene (the standard bacterial barcode); the RNA viruses (Dengue, MERS-CoV) don't have a 16S gene, so they're represented by their complete genome instead. NCBI serves RNA virus genomes as standard A/C/G/T FASTA (T representing U), which is what's stored here too.

| Organism | Reference type | Accession |
|---|---|---|
| Escherichia coli | 16S rRNA gene | NR_114042.1 |
| Bacillus subtilis | 16S rRNA gene | NR_102783.2 |
| Staphylococcus aureus | 16S rRNA gene | NR_037007.2 |
| Pseudomonas aeruginosa | 16S rRNA gene | NR_117678.1 |
| Salmonella enterica | 16S rRNA gene | NR_074910.1 |
| Streptococcus pyogenes | 16S rRNA gene | NR_112088.1 |
| Lactobacillus acidophilus | 16S rRNA gene | NR_117062.1 |
| Mycobacterium tuberculosis | 16S rRNA gene | NR_044826.2 |
| Clostridioides difficile | 16S rRNA gene | NR_113132.1 |
| Enterococcus faecalis | 16S rRNA gene | NR_115765.1 |
| Vibrio cholerae | 16S rRNA gene | NR_117894.1 |
| Klebsiella pneumoniae | 16S rRNA gene | NR_119278.1 |
| Helicobacter pylori | 16S rRNA gene | NR_044761.1 |
| Listeria monocytogenes | 16S rRNA gene | NR_044823.1 |
| Brucella melitensis | 16S rRNA gene | NR_118257.1 |
| Dengue virus 1 | complete genome | NC_001477.1 |
| Dengue virus 2 | complete genome | NC_001474.2 |
| Dengue virus 3 | complete genome | NC_001475.2 |
| Dengue virus 4 | complete genome | NC_002640.1 |
| MERS-CoV (Saudi Arabia) | complete genome | NC_019843.3 |

Each record can be viewed directly at
`https://www.ncbi.nlm.nih.gov/nuccore/<accession>`.

**MERS-CoV (Saudi Arabia)** is RefSeq NC_019843.3, isolate HCoV-EMC/2012 —
the first characterized Middle East respiratory syndrome coronavirus genome. It's the canonical MERS-CoV reference used throughout the literature and NCBI's own RefSeq annotation.

## Gene annotations (`gene_annotations.json`)

For the viral genomes, `gene_annotations.json` holds protein-coding
coordinates parsed straight out of each RefSeq GenBank record's feature
table, 1-based, matching that genome's own numbering:

- **Dengue (4 serotypes):** the mature peptide (`mat_peptide`) cleavage
  products of the single Dengue polyprotein — capsid C, prM/M, envelope E,
  NS1, NS2A, NS2B, NS3, NS4A, 2K, NS4B, NS5.
- **MERS-CoV:** the `nsp1`–`nsp16` replicase cleavage products inside
  `orf1ab` (the ~21kb non-structural-protein region making up most of the
  genome) plus the structural and accessory genes downstream — spike
  glycoprotein S, ORF3/4a/4b/5, envelope E, membrane M, nucleocapsid N, and
  ORF8b.

This is what lets the chatbot report something like "falls within: spike
glycoprotein S" instead of just a bare coordinate range. 

Bacterial 16S references have no entry here, the whole reference sequence *is* the one
gene, so there's nothing to subdivide.

## Why augmentation instead of training on 20 raw sequences

There's only one reference sequence per organism, so `train_model.py`
doesn't train directly on these 20 records. Instead
`dna_identifier/augment.py` cuts each one into many randomized,
mutated, strand-randomized read-length fragments (see README for why). The
model is therefore only as good as this 20-organism, single-reference-each
set, it will confidently misclassify anything outside it. 

It's a learning/demo project, not a diagnostic tool.
