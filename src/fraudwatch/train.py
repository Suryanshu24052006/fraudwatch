# python -m fraudwatch.train
from __future__ import annotations

import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve

from .data import FEATURES, PROJECT_ROOT, TARGET, load_splits
from .metrics import (REVIEW_COST, at_threshold, best_threshold, precision_at_k,
                      recall_at_precision, summary)
from .models import (fit_isolation_forest, fit_lightgbm, fit_logistic, isolation_scores,
                     shap_importance)
from .policy import BLOCK_THRESHOLD, decide

MODELS = PROJECT_ROOT / "models"
REPORTS = PROJECT_ROOT / "reports"
FIGS = REPORTS / "figures"


def main():
    for d in (MODELS, REPORTS, FIGS):
        d.mkdir(parents=True, exist_ok=True)

    train, valid, test = load_splits()
    Xtr, ytr = train[FEATURES], train[TARGET]
    Xva, yva = valid[FEATURES], valid[TARGET]
    Xte, yte = test[FEATURES], test[TARGET]
    for name, d in (("train", train), ("valid", valid), ("test", test)):
        print(f"{name}: {len(d):,} transactions, {int(d[TARGET].sum())} frauds "
              f"(hours {d.Time.min() / 3600:.1f}-{d.Time.max() / 3600:.1f})")

    logit = fit_logistic(Xtr, ytr)
    iso = fit_isolation_forest(Xtr)
    gbm = fit_lightgbm(Xtr, ytr, Xva, yva)

    scores_test = {
        "isolation_forest": isolation_scores(iso, Xte),
        "logistic_regression": logit.predict_proba(Xte)[:, 1],
        "lightgbm": gbm.predict_proba(Xte)[:, 1],
    }
    results = {"split": {"train": len(train), "valid": len(valid), "test": len(test),
                         "frauds_test": int(yte.sum())},
               "review_cost_per_alert": REVIEW_COST,
               "lightgbm_trees": int(gbm.best_iteration_),
               "models_test": {}}
    for name, p in scores_test.items():
        results["models_test"][name] = {
            **summary(yte, p),
            "precision_at_100": precision_at_k(yte, p, 100),
            "recall_at_80pct_precision": recall_at_precision(yte, p, 0.8),
        }

    # thresholds are picked on validation only, then checked on test
    p_valid = gbm.predict_proba(Xva)[:, 1]
    review_t, valid_curve = best_threshold(yva, p_valid, valid["Amount"])
    block_t = BLOCK_THRESHOLD

    p_test = scores_test["lightgbm"]
    results["thresholds"] = {"review": round(review_t, 4), "block": round(block_t, 4)}
    results["lightgbm_test_at_0.5"] = at_threshold(yte, p_test, test["Amount"], 0.5)
    results["lightgbm_test_at_review_threshold"] = at_threshold(yte, p_test, test["Amount"], review_t)
    results["lightgbm_test_at_block_threshold"] = at_threshold(yte, p_test, test["Amount"], block_t)
    results["test_no_model"] = {"fraud_amount_missed": round(float(test.loc[yte == 1, "Amount"].sum()), 2)}

    # how the allow / review / block policy splits the test transactions
    decisions = np.array([decide(p, review_t, block_t) for p in p_test])
    results["policy_test"] = {
        d: {"transactions": int((decisions == d).sum()), "frauds": int(yte[decisions == d].sum())}
        for d in ("allow", "review", "block")
    }

    # a few real test transactions for the demo page (the V columns mean nothing to a human,
    # so it's easier to try the API with real rows)
    rng = np.random.default_rng(0)
    frauds = test.index[yte == 1].to_numpy()
    normals = test.index[yte == 0].to_numpy()
    picks = list(rng.choice(frauds, 4, replace=False)) + list(rng.choice(normals, 4, replace=False))
    demo = []
    for i in picks:
        row = test.loc[i]
        demo.append({"label": "fraud" if row[TARGET] == 1 else "normal",
                     "transaction": {"amount": round(float(row["Amount"]), 2), "hour": int(row["hour"]),
                                     **{c.lower(): round(float(row[c]), 6) for c in FEATURES if c.startswith("V")}}})
    (REPORTS / "demo_transactions.json").write_text(json.dumps(demo, indent=1))

    (REPORTS / "results.json").write_text(json.dumps(results, indent=2))
    (MODELS / "thresholds.json").write_text(json.dumps(results["thresholds"], indent=2))
    valid_curve.round(4).to_csv(REPORTS / "cost_curve_valid.csv", index=False)
    shap_importance(gbm, Xte).round(4).rename("mean_abs_shap").to_csv(REPORTS / "feature_importance.csv")
    make_plots(yte, scores_test, valid_curve, review_t)

    joblib.dump(gbm, MODELS / "lightgbm.joblib")
    joblib.dump(logit, MODELS / "logistic.joblib")

    print(json.dumps(results["models_test"], indent=2))
    print("thresholds:", results["thresholds"])
    for k in ("lightgbm_test_at_0.5", "lightgbm_test_at_review_threshold", "lightgbm_test_at_block_threshold"):
        print(k, results[k])
    return results


def make_plots(y, scores, valid_curve, review_t):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for name, p in scores.items():
        prec, rec, _ = precision_recall_curve(y, p)
        ax.plot(rec, prec, label=name)
    ax.set(xlabel="Recall (share of frauds caught)", ylabel="Precision (share of alerts that are fraud)",
           title="Precision-recall, test set")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGS / "pr_curve.png", dpi=130)
    plt.close(fig)

    c = valid_curve[valid_curve["threshold"] <= 0.95]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(c["threshold"], c["fraud_amount_missed"], label="missed fraud amount")
    ax.plot(c["threshold"], c["review_cost"], label="review cost")
    ax.plot(c["threshold"], c["total_cost"], label="total", lw=2)
    ax.axvline(review_t, ls="--", c="grey", lw=0.8)
    ax.set(xscale="log", xlabel="Alert threshold (log scale)", ylabel="Cost (EUR)", title="Picking the threshold, validation set")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGS / "cost_vs_threshold.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
