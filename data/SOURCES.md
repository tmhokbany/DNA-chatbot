# Reference sequence provenance

`reference_16s.fasta` contains one real 16S ribosomal RNA RefSeq record per
species, fetched from NCBI's `nuccore` database via the E-utilities API
(`esearch` + `efetch`, `srcdb_refseq[PROP]`).

| Species | Accession |
|---|---|
| Escherichia coli | NR_114042.1 |
| Bacillus subtilis | NR_102783.2 |
| Staphylococcus aureus | NR_037007.2 |
| Pseudomonas aeruginosa | NR_117678.1 |
| Salmonella enterica | NR_074910.1 |
| Streptococcus pyogenes | NR_112088.1 |
| Lactobacillus acidophilus | NR_117062.1 |
| Mycobacterium tuberculosis | NR_044826.2 |
| Clostridioides difficile | NR_113132.1 |
| Enterococcus faecalis | NR_115765.1 |
| Vibrio cholerae | NR_117894.1 |
| Klebsiella pneumoniae | NR_119278.1 |
| Helicobacter pylori | NR_044761.1 |
| Listeria monocytogenes | NR_044823.1 |

Each record can be viewed directly at
`https://www.ncbi.nlm.nih.gov/nuccore/<accession>`.

Because there's only one reference genome per species, `train_model.py` does
not train directly on these 14 full-length sequences. Instead it uses
`dna_identifier/augment.py` to cut each one into many randomized,
mutated, strand-randomized read-length fragments. The
model is therefore only as good as this 14-species, single-strain-per-species
reference set, it will confidently misclassify anything outside it. It's a
learning/demo project.
