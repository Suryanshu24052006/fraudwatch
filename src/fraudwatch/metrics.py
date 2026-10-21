"""Metrics for fraud. Accuracy is useless here (99.8% of transactions are fine),
so everything is about how many frauds we catch and how many alerts it costs."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

# cost assumptions (in the dataset's currency, euros)
REVIEW_COST = 5.0  # analyst time to check one alert


def summary(y, p):
    return {
        "pr_auc": round(float(average_precision_score(y, p)), 4),
        "roc_auc": round(float(roc_auc_score(y, p)), 4),
        "frauds": int(np.sum(y)),
        "transactions": int(len(y)),
    }


def at_threshold(y, p, amount, threshold, review_cost=REVIEW_COST):
    """What happens if every transaction with score >= threshold becomes an alert."""
    y = np.asarray(y)
    amount = np.asarray(amount)
    alert = np.asarray(p) >= threshold
    tp = int((alert & (y == 1)).sum())
    fp = int((alert & (y == 0)).sum())
    fn = int((~alert & (y == 1)).sum())
    caught = float(amount[alert & (y == 1)].sum())
    missed = float(amount[~alert & (y == 1)].sum())
    return {
        "threshold": float(threshold),
        "alerts": int(alert.sum()),
        "frauds_caught": tp,
        "frauds_missed": fn,
        "false_alerts": fp,
        "precision": round(tp / max(tp + fp, 1), 4),
        "recall": round(tp / max(tp + fn, 1), 4),
        "fraud_amount_caught": round(caught, 2),
        "fraud_amount_missed": round(missed, 2),
        "review_cost": round(review_cost * int(alert.sum()), 2),
        "total_cost": round(missed + review_cost * int(alert.sum()), 2),
    }


def cost_curve(y, p, amount, thresholds, review_cost=REVIEW_COST):
    return pd.DataFrame([at_threshold(y, p, amount, t, review_cost) for t in thresholds])


def best_threshold(y, p, amount, review_cost=REVIEW_COST):
    """Threshold with the lowest total cost = missed fraud amount + review cost of all alerts."""
    grid = np.unique(np.concatenate([np.linspace(0.001, 0.999, 999), np.quantile(p, np.linspace(0.99, 1, 200))]))
    curve = cost_curve(y, p, amount, grid, review_cost)
    return float(curve.loc[curve["total_cost"].idxmin(), "threshold"]), curve


def precision_at_k(y, p, k):
    """If the fraud team can only check k transactions, how many of the top k are fraud."""
    top = np.argsort(-np.asarray(p))[:k]
    return round(float(np.asarray(y)[top].mean()), 4)


def recall_at_precision(y, p, min_precision):
    prec, rec, _ = precision_recall_curve(y, p)
    ok = prec >= min_precision
    return round(float(rec[ok].max()) if ok.any() else 0.0, 4)
