"""
Step 3 of the pipeline: LMG / Shapley R² decomposition by surface.

For each surface (Hard, Clay, Grass) and for all matches pooled ("All"):
  1. Fit a linear probability model (OLS): a_won ~ 7 difference features. Report R².
  2. Decompose that R² across the features with LMG (= Shapley value of R²):
     a feature's share is its marginal R² gain when added to a subset of the
     other features, averaged over every possible order of adding features.
     With 7 features there are only 2^7 = 128 subsets, so we compute it exactly.
  3. Fit a logistic regression on an 80/20 split and report test accuracy as a
     sanity check that the features genuinely separate winners from losers.

Output: data/processed/lmg_results.csv
Usage:
    python src/analysis.py
"""

from itertools import combinations
from math import factorial
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IN_PATH = PROJECT_ROOT / "data" / "processed" / "matches.csv"
OUT_PATH = PROJECT_ROOT / "data" / "processed" / "lmg_results.csv"

FEATURES = [
    "first_serve_in",
    "first_serve_won",
    "second_serve_won",
    "bp_saved",
    "ace_rate",
    "df_rate",
    "return_pts_won",
]
SURFACES = ["All", "Hard", "Clay", "Grass"]
SEED = 42

# Human-readable labels for printing.
LABELS = {
    "first_serve_in": "1st serve in %",
    "first_serve_won": "1st serve points won %",
    "second_serve_won": "2nd serve points won %",
    "bp_saved": "Break points saved %",
    "ace_rate": "Ace rate",
    "df_rate": "Double fault rate",
    "return_pts_won": "Return points won %",
}


def ols_r2(X: pd.DataFrame, y: pd.Series, cols: tuple) -> float:
    """R² of an OLS fit of y on an intercept plus `cols` (empty subset -> 0)."""
    if not cols:
        return 0.0
    return sm.OLS(y, sm.add_constant(X[list(cols)])).fit().rsquared


def lmg_decomposition(X: pd.DataFrame, y: pd.Series) -> tuple[dict, float]:
    """Exact LMG (Shapley) decomposition of the OLS R².

    Shapley value of feature j:
        phi_j = sum over subsets S of the other features of
                |S|! (p - |S| - 1)! / p!  *  [R²(S ∪ {j}) - R²(S)]
    The weight is the fraction of all p! orderings in which exactly the
    features in S come before j, so this equals "average marginal gain over
    all orderings" -- the definition of LMG.
    """
    p = len(FEATURES)

    # Fit every one of the 2^p subsets once and cache its R² (keyed by frozenset).
    r2 = {}
    for k in range(p + 1):
        for subset in combinations(FEATURES, k):
            r2[frozenset(subset)] = ols_r2(X, y, subset)

    shares = {}
    for j in FEATURES:
        others = [f for f in FEATURES if f != j]
        phi = 0.0
        for k in range(p):  # size of the subset S that precedes j
            weight = factorial(k) * factorial(p - k - 1) / factorial(p)
            for S in combinations(others, k):
                S = frozenset(S)
                phi += weight * (r2[S | {j}] - r2[S])
        shares[j] = phi

    full_r2 = r2[frozenset(FEATURES)]
    return shares, full_r2


def logit_accuracy(X: pd.DataFrame, y: pd.Series) -> float:
    """Hold-out accuracy of a logistic regression (80/20 stratified split)."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(X_train, y_train)
    return model.score(X_test, y_test)


def analyse_surface(df: pd.DataFrame, surface: str) -> pd.DataFrame:
    """Run OLS + LMG + logit for one surface and return tidy result rows."""
    sub = df if surface == "All" else df[df["surface"] == surface]
    X, y = sub[FEATURES], sub["a_won"]

    shares, full_r2 = lmg_decomposition(X, y)
    acc = logit_accuracy(X, y)

    # Key verification: Shapley efficiency -> shares must add up to the full R².
    total = sum(shares.values())
    assert np.isclose(total, full_r2, atol=1e-10), f"{surface}: shares {total} != R² {full_r2}"
    print(f"{surface:>5}: n={len(sub):>6,}  R²={full_r2:.4f}  sum(LMG)={total:.4f}  "
          f"logit acc={acc:.3f}  [shares sum to R² ✓]")

    return pd.DataFrame(
        {
            "surface": surface,
            "feature": FEATURES,
            "r2_share": [shares[f] for f in FEATURES],
            "pct_of_r2": [100 * shares[f] / full_r2 for f in FEATURES],
            "model_r2": full_r2,
            "logit_accuracy": acc,
            "n_matches": len(sub),
        }
    )


def print_summary(results: pd.DataFrame) -> None:
    """Short text summary: top 3 factors per surface."""
    print("\nTop factors (share of explained R²):")
    for surface in SURFACES:
        top = results[results["surface"] == surface].nlargest(3, "pct_of_r2")
        parts = [f"{LABELS[r.feature]} {r.pct_of_r2:.1f}%" for r in top.itertuples()]
        print(f"  {surface:>5}: " + " | ".join(parts))


def main() -> None:
    df = pd.read_csv(IN_PATH)
    print(f"Loaded {len(df):,} matches\n")

    results = pd.concat([analyse_surface(df, s) for s in SURFACES], ignore_index=True)
    results.to_csv(OUT_PATH, index=False)
    print_summary(results)
    print(f"\nSaved {OUT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
