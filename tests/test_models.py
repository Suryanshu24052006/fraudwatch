from sklearn.metrics import average_precision_score

from fraudwatch.models import fit_logistic, top_reasons
from fraudwatch.policy import decide


def test_lightgbm_learns_the_pattern(model, data):
    X, y, _ = data
    n = int(len(X) * 0.75)
    p = model.predict_proba(X.iloc[n:])[:, 1]
    assert average_precision_score(y.iloc[n:], p) > 0.3  # base rate is ~1%


def test_logistic_baseline(data):
    X, y, _ = data
    m = fit_logistic(X, y)
    assert average_precision_score(y, m.predict_proba(X)[:, 1]) > 0.3


def test_reasons_point_at_v14_and_v4(model, data):
    X, _, _ = data
    risky = X.head(1).copy()
    risky[["V14", "V4"]] = [-6.0, 4.0]
    reasons = top_reasons(model, risky)[0]
    assert reasons and {r["feature"] for r in reasons} & {"V14", "V4"}
    assert all(r["contribution"] > 0 for r in reasons)


def test_decide():
    assert decide(0.9, 0.3, 0.8) == "block"
    assert decide(0.5, 0.3, 0.8) == "review"
    assert decide(0.1, 0.3, 0.8) == "allow"
