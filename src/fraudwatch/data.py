from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "creditcard.csv"
# same file as the kaggle "Credit Card Fraud Detection" dataset (ULB), mirrored on github
MIRROR_URL = ("https://raw.githubusercontent.com/nsethi31/Kaggle-Data-Credit-Card-Fraud-Detection/"
              "master/creditcard.csv")

TARGET = "Class"
PCA_COLS = [f"V{i}" for i in range(1, 29)]
FEATURES = PCA_COLS + ["log_amount", "hour"]


def load_raw():
    if not DATA_PATH.exists():
        print(f"downloading dataset to {DATA_PATH} (~100 MB)...")
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(MIRROR_URL, DATA_PATH)
    return pd.read_csv(DATA_PATH)


def add_features(df):
    df = df.copy()
    df["log_amount"] = np.log1p(df["Amount"])
    # Time is seconds since the first transaction, the data covers 2 days
    df["hour"] = (df["Time"] // 3600) % 24
    return df


def load_splits():
    """Split by time, not randomly: train on the past, test on the future.
    first 60% of transactions = train, next 20% = validation, last 20% = test"""
    df = add_features(load_raw()).sort_values("Time", kind="stable").reset_index(drop=True)
    n = len(df)
    a, b = int(n * 0.6), int(n * 0.8)
    return df.iloc[:a].reset_index(drop=True), df.iloc[a:b].reset_index(drop=True), df.iloc[b:].reset_index(drop=True)
