"""
IPL match winner prediction using Logistic Regression.

Usage:
    python3 main.py                     # uses data/matches.csv
    python3 main.py path/to/file.csv    # custom path
"""
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")  # saves plots to files, no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else "data/matches.csv"

# Team renames so the same franchise has one name
TEAM_FIX = {
    "Delhi Daredevils": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings",
    "Deccan Chargers": "Sunrisers Hyderabad",
    "Rising Pune Supergiants": "Rising Pune Supergiant",
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
}

# Possible column names -> we pick whichever exists in your CSV
CANDIDATES = {
    "date": ["date", "match_date"],
    "team1": ["team1", "team_1", "batting_team", "home_team"],
    "team2": ["team2", "team_2", "bowling_team", "away_team"],
    "winner": ["winner", "match_winner", "winning_team"],
    "venue": ["venue", "stadium", "ground"],
    "city": ["city"],
    "toss_winner": ["toss_winner", "tosswinner"],
    "toss_decision": ["toss_decision", "toss_decison", "toss_choice"],
}


def find_columns(df):
    lower = {c.lower().strip(): c for c in df.columns}
    found = {}
    for key, options in CANDIDATES.items():
        for opt in options:
            if opt in lower:
                found[key] = lower[opt]
                break
    missing = [k for k in ("team1", "team2", "winner") if k not in found]
    if missing:
        print("Columns in your file:", list(df.columns))
        raise SystemExit(f"Could not find required columns: {missing}. "
                         "Edit CANDIDATES at the top of main.py.")
    return found


def load_and_clean(path):
    df = pd.read_csv(path)
    print("Shape:", df.shape)
    print("Columns:", list(df.columns))
    cols = find_columns(df)
    df = df.rename(columns={v: k for k, v in cols.items()})

    # Fix team names in all team columns
    for c in ("team1", "team2", "winner", "toss_winner"):
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().replace(TEAM_FIX)

    # Drop no-result / abandoned matches (winner missing or not a playing team)
    df = df[df["winner"].notna()]
    df = df[(df["winner"] == df["team1"]) | (df["winner"] == df["team2"])]

    # Sort in time order
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values("date")
    df = df.reset_index(drop=True)
    return df


def build_features(df):
    """Create features using ONLY information known before each match."""
    results = defaultdict(list)                       # team -> list of 1/0
    h2h = defaultdict(lambda: defaultdict(int))       # pair -> team -> wins

    form1, form2, win1, win2, h2h_rate = [], [], [], [], []

    for _, r in df.iterrows():
        t1, t2 = r["team1"], r["team2"]
        pair = frozenset([t1, t2])

        def form(t):
            last = results[t][-5:]
            return np.mean(last) if last else 0.5

        def overall(t):
            return np.mean(results[t]) if results[t] else 0.5

        form1.append(form(t1))
        form2.append(form(t2))
        win1.append(overall(t1))
        win2.append(overall(t2))

        total = h2h[pair][t1] + h2h[pair][t2]
        h2h_rate.append(h2h[pair][t1] / total if total else 0.5)

        # update AFTER recording features (prevents leakage)
        results[t1].append(1 if r["winner"] == t1 else 0)
        results[t2].append(1 if r["winner"] == t2 else 0)
        h2h[pair][r["winner"]] += 1

    df["form_diff"] = np.array(form1) - np.array(form2)
    df["overall_diff"] = np.array(win1) - np.array(win2)
    df["h2h_rate"] = h2h_rate
    if "toss_winner" in df.columns:
        df["toss_winner_is_team1"] = (df["toss_winner"] == df["team1"]).astype(int)
    df["team1_won"] = (df["winner"] == df["team1"]).astype(int)  # TARGET
    return df


def main():
    df = build_features(load_and_clean(CSV_PATH))
    print(f"\nMatches used: {len(df)}")

    numeric = ["form_diff", "overall_diff", "h2h_rate"]
    if "toss_winner_is_team1" in df.columns:
        numeric.append("toss_winner_is_team1")
    categorical = [c for c in ("team1", "team2", "venue", "toss_decision")
                   if c in df.columns]
    features = numeric + categorical

    X, y = df[features], df["team1_won"]

    # Time-based split: oldest 80% train, newest 20% test
    split = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    print(f"Train: {len(X_train)} matches | Test: {len(X_test)} matches")

    prep = ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])
    model = Pipeline([
        ("prep", prep),
        ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
    ])
    model.fit(X_train, y_train)

    # Baseline: always predict team1 wins (or the majority class)
    majority = int(y_train.mean() >= 0.5)
    baseline_acc = accuracy_score(y_test, np.full(len(y_test), majority))

    pred = model.predict(X_test)
    print("\n=== RESULTS ===")
    print(f"Baseline accuracy (majority class): {baseline_acc:.3f}")
    print(f"Logistic Regression accuracy:       {accuracy_score(y_test, pred):.3f}")
    print("\nClassification report:")
    print(classification_report(y_test, pred, digits=3))

    # Confusion matrix plot
    cm = confusion_matrix(y_test, pred)
    ConfusionMatrixDisplay(cm, display_labels=["Team2 won", "Team1 won"]).plot()
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Top coefficients plot (which features matter most)
    names = model.named_steps["prep"].get_feature_names_out()
    coefs = model.named_steps["clf"].coef_[0]
    top = pd.Series(coefs, index=names).sort_values(key=abs, ascending=False).head(15)
    top.sort_values().plot(kind="barh", figsize=(8, 6))
    plt.title("Top 15 Logistic Regression coefficients")
    plt.savefig("feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: confusion_matrix.png, feature_importance.png")


if __name__ == "__main__":
    main()