from __future__ import annotations

import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def fit_logistic(X, y):
    # balanced class weights, otherwise it just predicts "not fraud" for everything
    model = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000))
    model.fit(X, y)
    return model


def fit_isolation_forest(X, seed=42):
    # unsupervised baseline, never sees the labels
    return IsolationForest(n_estimators=300, contamination="auto", random_state=seed, n_jobs=-1).fit(X)


def isolation_scores(model, X):
    # higher = more anomalous, squashed to 0-1 so it can be compared with the others
    s = -model.score_samples(X)
    return (s - s.min()) / (s.max() - s.min() + 1e-12)


LGB_PARAMS = dict(
    n_estimators=3000,
    learning_rate=0.02,
    num_leaves=15,
    min_child_samples=50,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.7,
    reg_lambda=5.0,
    verbose=-1,
)


def fit_lightgbm(X, y, X_valid, y_valid, seed=42):
    # no class weights here on purpose: keeps the probabilities closer to real fraud rates.
    # the imbalance is handled later by choosing the threshold on cost instead of using 0.5
    model = lgb.LGBMClassifier(**LGB_PARAMS, random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", message=".*eval_set.*")
        model.fit(X, y, eval_set=[(X_valid, y_valid)], eval_metric="average_precision",
                  callbacks=[lgb.early_stopping(200, verbose=False)])
    return model


def top_reasons(model, X, top_n=3):
    """SHAP values from lightgbm (pred_contrib). Features pushing the score up the most."""
    contrib = model.predict(X, pred_contrib=True)[:, :-1]
    cols = np.asarray(X.columns)
    out = []
    for row, values in zip(contrib, X.to_numpy()):
        idx = np.argsort(-row)[:top_n]
        out.append([{"feature": str(cols[i]), "value": round(float(values[i]), 4),
                     "contribution": round(float(row[i]), 4)} for i in idx if row[i] > 0])
    return out


def shap_importance(model, X):
    contrib = model.predict(X, pred_contrib=True)[:, :-1]
    return pd.Series(np.abs(contrib).mean(axis=0), index=X.columns).sort_values(ascending=False)
