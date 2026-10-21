import numpy as np

from fraudwatch.metrics import at_threshold, best_threshold, precision_at_k, recall_at_precision


def test_at_threshold_counts():
    y = np.array([1, 1, 0, 0, 0])
    p = np.array([0.9, 0.2, 0.8, 0.1, 0.05])
    amt = np.array([100, 50, 10, 10, 10])
    r = at_threshold(y, p, amt, 0.5, review_cost=5)
    assert (r["frauds_caught"], r["frauds_missed"], r["false_alerts"]) == (1, 1, 1)
    assert r["fraud_amount_missed"] == 50
    assert r["total_cost"] == 50 + 2 * 5


def test_best_threshold_beats_extremes():
    rng = np.random.default_rng(1)
    y = (rng.random(5000) < 0.02).astype(int)
    p = np.clip(0.6 * y + rng.random(5000) * 0.5, 0, 1)
    amt = np.full(5000, 100.0)
    t, curve = best_threshold(y, p, amt, review_cost=5)
    best = curve["total_cost"].min()
    assert best <= at_threshold(y, p, amt, 0.999, 5)["total_cost"]  # alert on nothing
    assert best <= at_threshold(y, p, amt, 0.0, 5)["total_cost"]    # alert on everything


def test_precision_at_k():
    y = np.array([1, 0, 1, 0])
    p = np.array([0.9, 0.8, 0.7, 0.1])
    assert precision_at_k(y, p, 2) == 0.5
    assert precision_at_k(y, p, 1) == 1.0


def test_recall_at_precision():
    y = np.array([1, 1, 0, 1])
    p = np.array([0.9, 0.8, 0.7, 0.1])
    assert recall_at_precision(y, p, 1.0) == round(2 / 3, 4)
