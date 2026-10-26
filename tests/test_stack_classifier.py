import numpy as np

from ids.models.stack_classifier import StackedClassifier


def test_stacking_supports_string_attack_labels_and_bounded_training():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(36, 4))
    y = np.array(["DoS", "PortScan", "Bot"] * 12)

    model = StackedClassifier.fit(
        X,
        y,
        n_splits=2,
        random_forest_estimators=2,
        xgboost_estimators=2,
        max_rows=18,
    )

    predictions, confidence = model.predict_with_confidence(X[:5])
    assert set(model.classes_) == {"DoS", "PortScan", "Bot"}
    assert predictions.shape == (5,)
    assert confidence.shape == (5,)
