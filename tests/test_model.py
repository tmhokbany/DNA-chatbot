from dna_identifier.augment import build_training_set
from dna_identifier.model import SequenceIdentifier

REFERENCES = {
    "Species A": "ACGT" * 100 + "AAAAA" * 20,
    "Species B": "TTTT" * 100 + "CCCCC" * 20,
}

def test_model_trains_and_predicts_distinguishable_classes():
    sequences, labels = build_training_set(
        REFERENCES, fragments_per_class=20, min_len=100, max_len=200, mutation_rate=0.0
    )
    model = SequenceIdentifier(k=3, n_estimators=50)
    model.fit(sequences, labels)

    assert set(model.classes_) == {"Species A", "Species B"}

    pred = model.predict("ACGT" * 30)
    assert pred.label == "Species A"
    assert 0.0 <= pred.confidence <= 1.0
    assert len(pred.top_matches) == 2

def test_save_and_load_roundtrip(tmp_path):
    sequences, labels = build_training_set(
        REFERENCES, fragments_per_class=10, min_len=100, max_len=150, mutation_rate=0.0
    )
    model = SequenceIdentifier(k=3, n_estimators=20)
    model.fit(sequences, labels)

    path = tmp_path / "model.joblib"
    model.save(str(path))

    loaded = SequenceIdentifier.load(str(path))
    pred = loaded.predict("TTTT" * 30)
    assert pred.label == "Species B"
