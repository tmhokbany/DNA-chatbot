"""Combines the k-mer classifier with alignment against the real reference
sequences, so the final answer is checked against actual sequence data
instead of resting on a composition-based guess.

1. The classifier ranks every organism by k-mer composition.
2. The query is aligned to the classifier's top candidates plus whichever
   references share the most exact 11-mers with it (a cheap BLAST-like
   prefilter that catches cases the classifier ranks too low).
3. The best-scoring alignment decides the answer, graded by percent identity
   and query coverage:
     - >= 98.7% identity over >= 80% of the query: confirmed match. 98.7% is
       the modern 16S rRNA species-level threshold (Kim et al. 2014); the
       older 97% let ~17% of reads from a missing species "confirm" as a
       relative (e.g. Salmonella reads as Klebsiella).
     - same, but another reference matches about as well: ambiguous. Short
       reads from conserved 16S regions can be identical across species.
     - >= 90% identity over >= 50% of the query: related. The closest organism
       in the database, but probably a different species/strain/genotype.
     - anything weaker: no reliable match. Most likely an organism outside
       the database, so no species name is claimed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .locate import LocationMatch, locate, seed_hits
from .model import Prediction, SequenceIdentifier

CONFIRMED_IDENTITY = 98.7
CONFIRMED_COVERAGE = 80.0
RELATED_IDENTITY = 90.0
RELATED_COVERAGE = 50.0
AMBIGUITY_MARGIN = 1.0

MIN_SEED_FRACTION = 0.1

@dataclass
class Identification:
    status: str 
    label: str | None  
    location: LocationMatch | None
    classifier: Prediction
    alignments: list[tuple[str, LocationMatch]] = field(default_factory=list)  
    close_alternatives: list[str] = field(default_factory=list)

    @property
    def agrees_with_classifier(self) -> bool:
        return self.label == self.classifier.label

def _grade(loc: LocationMatch) -> str:
    if loc.percent_identity >= CONFIRMED_IDENTITY and loc.query_coverage >= CONFIRMED_COVERAGE:
        return "confirmed"
    if loc.percent_identity >= RELATED_IDENTITY and loc.query_coverage >= RELATED_COVERAGE:
        return "related"
    return "no_match"

def identify(
    seq: str,
    model: SequenceIdentifier,
    references: dict[str, str],
    gene_annotations: dict[str, list[dict]] | None = None,
    n_classifier_candidates: int = 3,
    n_seed_candidates: int = 3,
) -> Identification:
    pred = model.predict(seq, top_n=max(3, n_classifier_candidates))

    seeds = {name: seed_hits(seq, ref) for name, ref in references.items()}
    by_seeds = sorted(seeds, key=seeds.get, reverse=True)
    min_seeds = MIN_SEED_FRACTION * seeds[by_seeds[0]] if by_seeds else 0

    candidates = [name for name, _ in pred.top_matches[:n_classifier_candidates]]
    for name in by_seeds[:n_seed_candidates]:
        if name not in candidates:
            candidates.append(name)
    candidates = [
        name for name in candidates
        if name in references and (seeds[name] >= min_seeds or name == by_seeds[0])
    ]

    alignments = []
    for name in candidates:
        loc = locate(seq, references[name], genes=(gene_annotations or {}).get(name))
        if loc is not None:
            alignments.append((name, loc))
    alignments.sort(key=lambda item: item[1].score, reverse=True)

    if not alignments:
        return Identification("no_match", None, None, pred)

    best_name, best_loc = alignments[0]
    status = _grade(best_loc)
    close = [
        name
        for name, loc in alignments[1:]
        if loc.percent_identity >= best_loc.percent_identity - AMBIGUITY_MARGIN
        and loc.query_coverage >= best_loc.query_coverage - 10
    ]
    if status == "confirmed" and close:
        status = "ambiguous"
    return Identification(
        status=status,
        label=best_name if status != "no_match" else None,
        location=best_loc,
        classifier=pred,
        alignments=alignments,
        close_alternatives=close,
    )
