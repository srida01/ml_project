import numpy as np
import pytest

from ids.models.svm_filter import SVMFilter, _stratified_sample_indices


def test_stratified_sample_is_bounded_and_keeps_classes():
    labels = np.array([0] * 90 + [1] * 10)
    indices = _stratified_sample_indices(labels, 20, seed=42)
    assert len(indices) == 20
    assert set(labels[indices]) == {0, 1}


def test_approximate_svm_scores_large_input_in_batches():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(40, 4)).astype(np.float32)
    y = np.array([0, 1] * 20)
    model = SVMFilter.fit(X, y, max_exact_rows=10, approx_components=32, transform_batch_size=5)
    scores = model.score(X)
    assert model.model_name == "rbf_sampler_linear_svm"
    assert scores.shape == (40,)


def test_approximate_svm_does_not_materialize_full_projection(monkeypatch):
    from sklearn.kernel_approximation import RBFSampler

    def fail_fit_transform(self, X, y=None):
        pytest.fail("approximate SVM must stream projected batches")

    monkeypatch.setattr(RBFSampler, "fit_transform", fail_fit_transform)
    rng = np.random.default_rng(7)
    X = rng.normal(size=(24, 6)).astype(np.float32)
    y = np.array([0, 1] * 12)

    model = SVMFilter.fit(X, y, max_exact_rows=8, approx_components=16, transform_batch_size=4)

    assert model.score(X).shape == (24,)


def test_high_recall_threshold_matches_highest_valid_candidate():
    class FixedScores:
        def decision_function(self, X):
            return np.asarray(X[:, 0], dtype=float)

    scores = np.array([-0.4, 0.1, 0.2, 0.2, 0.8, 1.1])
    labels = np.array([0, 1, 0, 1, 1, 0])
    model = SVMFilter(FixedScores())

    threshold = model.select_high_recall_threshold(scores[:, None], labels, target_recall=2 / 3)

    assert threshold == 0.2


def test_high_recall_threshold_rejects_validation_without_attacks():
    class FixedScores:
        def decision_function(self, X):
            return np.asarray(X[:, 0], dtype=float)

    scores = np.array([-0.4, 0.1, 0.2])
    with pytest.raises(ValueError, match="no attack rows"):
        SVMFilter(FixedScores()).select_high_recall_threshold(scores[:, None], np.zeros(3, dtype=int))
