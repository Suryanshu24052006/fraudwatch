import numpy as np
import pandas as pd
import pytest

from fraudwatch.data import FEATURES
from fraudwatch.models import fit_lightgbm


# fake transactions: ~1% fraud, fraud shows up as low V14 and high V4 (like the real data)
def make_transactions(n=20_000, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 28)), columns=[f"V{i}" for i in range(1, 29)])
    X["log_amount"] = np.log1p(rng.gamma(1.5, 60, n))
    X["hour"] = rng.integers(0, 24, n)
    logit = -8 - 3.0 * X["V14"] + 2.0 * X["V4"]
    y = pd.Series((rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int), name="Class")
    amount = np.expm1(X["log_amount"])
    return X[FEATURES], y, amount


@pytest.fixture(scope="session")
def data():
    return make_transactions()


@pytest.fixture(scope="session")
def model(data):
    X, y, _ = data
    n = int(len(X) * 0.75)
    return fit_lightgbm(X.iloc[:n], y.iloc[:n], X.iloc[n:], y.iloc[n:])
