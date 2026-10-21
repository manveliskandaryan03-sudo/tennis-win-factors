"""
Step 2 of the pipeline: clean raw ATP matches and build a modelling table.

Output: data/processed/matches.csv with ONE row per match:
  - match metadata (date, tournament, surface, ...)
  - 7 feature columns, each = (player A's rate) - (player B's rate)
  - target `a_won` (1 if player A won)

Players are randomly assigned to slot A/B (seed 42) so the target is ~50/50.
Otherwise "winner minus loser" would make every row a win and nothing could
be learned.

Usage:
    python src/prepare.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUT_PATH = PROJECT_ROOT / "data" / "processed" / "matches.csv"

SURFACES = ["Hard", "Clay", "Grass"]
SEED = 42

# The raw serve-stat columns (prefixed w_ for winner, l_ for loser).
SERVE_STATS = ["ace", "df", "svpt", "1stIn", "1stWon", "2ndWon", "SvGms", "bpSaved", "bpFaced"]

# The 7 rate features we compute for each player (order = display order).
FEATURES = [
    "first_serve_in",
    "first_serve_won",
    "second_serve_won",
    "bp_saved",
    "ace_rate",
    "df_rate",
    "return_pts_won",
]


def load_raw() -> pd.DataFrame:
    """Read and concatenate every downloaded season file."""
    files = sorted(RAW_DIR.glob("atp_matches_*.csv"))
    if not files:
        raise FileNotFoundError(f"No raw files in {RAW_DIR}. Run src/download_data.py first.")
    df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    print(f"Loaded {len(df):,} raw matches from {len(files)} files")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to completed matches on the three main surfaces with full serve stats."""
    n0 = len(df)

    # 1) Main surfaces only (drops Carpet and missing surfaces).
    df = df[df["surface"].isin(SURFACES)]
    print(f"  surface in {SURFACES}: {len(df):,} (-{n0 - len(df):,})")

    # 2) Drop retirements, walkovers and defaults: their stats describe a partial match.
    n = len(df)
    score = df["score"].fillna("").str.upper()
    df = df[~score.str.contains("RET|W/O|DEF", regex=True)]
    print(f"  completed matches (no RET / W/O / DEF): {len(df):,} (-{n - len(df):,})")

    # 3) Require every serve stat for both players.
    n = len(df)
    stat_cols = [f"{p}_{s}" for p in ("w", "l") for s in SERVE_STATS]
    df = df.dropna(subset=stat_cols)
    print(f"  non-null serve stats: {len(df):,} (-{n - len(df):,})")

    # 4) Guard against zero denominators (bad/empty stat lines would give inf/NaN).
    n = len(df)
    for p in ("w", "l"):
        second_serve_pts = df[f"{p}_svpt"] - df[f"{p}_1stIn"] - df[f"{p}_df"]
        df = df[(df[f"{p}_svpt"] > 0) & (df[f"{p}_1stIn"] > 0) & (second_serve_pts > 0)]
    print(f"  valid denominators: {len(df):,} (-{n - len(df):,})")

    return df.reset_index(drop=True)


def player_rates(df: pd.DataFrame, me: str, opp: str) -> pd.DataFrame:
    """Compute the 7 rate stats for one side of every match.

    `me` and `opp` are column prefixes: ("w", "l") for the winner, ("l", "w") for the loser.
    """
    svpt = df[f"{me}_svpt"]
    first_in = df[f"{me}_1stIn"]
    second_pts = svpt - first_in - df[f"{me}_df"]  # second serves actually put in play

    rates = pd.DataFrame(index=df.index)
    rates["first_serve_in"] = first_in / svpt
    rates["first_serve_won"] = df[f"{me}_1stWon"] / first_in
    rates["second_serve_won"] = df[f"{me}_2ndWon"] / second_pts
    # 0 break points faced -> 0/0; treat as "saved everything" (1.0).
    rates["bp_saved"] = (df[f"{me}_bpSaved"] / df[f"{me}_bpFaced"]).where(df[f"{me}_bpFaced"] > 0, 1.0)
    rates["ace_rate"] = df[f"{me}_ace"] / svpt
    rates["df_rate"] = df[f"{me}_df"] / svpt
    # Return points won = share of the opponent's service points that the opponent lost.
    rates["return_pts_won"] = 1 - (df[f"{opp}_1stWon"] + df[f"{opp}_2ndWon"]) / df[f"{opp}_svpt"]
    return rates


def build_match_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per match: random A/B assignment, features = A minus B."""
    winner = player_rates(df, "w", "l")
    loser = player_rates(df, "l", "w")

    # Flip a fair coin per match: 1 -> winner is player A, 0 -> loser is player A.
    rng = np.random.default_rng(SEED)
    a_is_winner = rng.integers(0, 2, size=len(df)).astype(bool)

    # Winner-minus-loser differences; flip the sign when the loser sits in slot A.
    diff = winner - loser
    sign = np.where(a_is_winner, 1.0, -1.0)
    features = diff.mul(sign, axis=0)

    meta = pd.DataFrame(
        {
            "tourney_id": df["tourney_id"],
            "tourney_name": df["tourney_name"],
            "tourney_date": pd.to_datetime(df["tourney_date"].astype(int).astype(str), format="%Y%m%d"),
            "tourney_level": df["tourney_level"],
            "surface": df["surface"],
            "best_of": df["best_of"],
            "round": df["round"],
            "player_a": np.where(a_is_winner, df["winner_name"], df["loser_name"]),
            "player_b": np.where(a_is_winner, df["loser_name"], df["winner_name"]),
        }
    )
    meta["year"] = meta["tourney_date"].dt.year

    out = pd.concat([meta, features[FEATURES]], axis=1)
    out["a_won"] = a_is_winner.astype(int)
    return out


def main() -> None:
    raw = load_raw()
    print("Cleaning:")
    df = clean(raw)
    matches = build_match_table(df)

    # Sanity checks: no missing/infinite features, balanced target.
    assert np.isfinite(matches[FEATURES].to_numpy()).all(), "non-finite feature values"
    print(f"\nFinal table: {len(matches):,} matches, a_won mean = {matches['a_won'].mean():.3f}")
    print(matches["surface"].value_counts().to_string())

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    matches.to_csv(OUT_PATH, index=False)
    print(f"\nSaved {OUT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
