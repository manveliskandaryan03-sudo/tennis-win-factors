"""
Streamlit dashboard: "What Wins Tennis Matches?"

Reads the precomputed results in data/processed/lmg_results.csv (no model
fitting at runtime), so it loads instantly.

Usage:
    streamlit run app.py
"""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

RESULTS_PATH = Path(__file__).resolve().parent / "data" / "processed" / "lmg_results.csv"

# Display labels for the 7 features (same order as the analysis).
LABELS = {
    "first_serve_in": "1st serve in %",
    "first_serve_won": "1st serve points won %",
    "second_serve_won": "2nd serve points won %",
    "bp_saved": "Break points saved %",
    "ace_rate": "Ace rate",
    "df_rate": "Double fault rate",
    "return_pts_won": "Return points won %",
}
SURFACES = ["All", "Hard", "Clay", "Grass"]

# One fixed color per surface (colorblind-validated categorical slots 1-3).
# Color follows the surface everywhere, so Clay is always orange, etc.
SURFACE_COLORS = {"All": "#6b6a66", "Hard": "#2a78d6", "Clay": "#eb6834", "Grass": "#1baf7a"}


# --- Data ----------------------------------------------------------------------
@st.cache_data
def load_results() -> pd.DataFrame:
    df = pd.read_csv(RESULTS_PATH)
    df["label"] = df["feature"].map(LABELS)
    return df


def pct(df: pd.DataFrame, surface: str, feature: str) -> float:
    """% of explained R² for one feature on one surface."""
    row = df[(df["surface"] == surface) & (df["feature"] == feature)]
    return float(row["pct_of_r2"].iloc[0])


# --- Charts --------------------------------------------------------------------
def surface_bar_chart(df: pd.DataFrame, surface: str) -> go.Figure:
    """Horizontal bars: % of explained R² per feature for one surface, largest on top."""
    sub = df[df["surface"] == surface].sort_values("pct_of_r2")
    fig = go.Figure(
        go.Bar(
            x=sub["pct_of_r2"],
            y=sub["label"],
            orientation="h",
            marker=dict(color=SURFACE_COLORS[surface], cornerradius=4),
            text=[f"{v:.1f}%" for v in sub["pct_of_r2"]],
            textposition="outside",
            cliponaxis=False,
            customdata=sub["r2_share"],
            hovertemplate="<b>%{y}</b><br>%{x:.1f}% of explained R²"
            "<br>R² share: %{customdata:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        height=380,
        margin=dict(l=10, r=50, t=10, b=40),
        xaxis=dict(title="% of explained R²", ticksuffix="%", showgrid=True, zeroline=False),
        yaxis=dict(title=None),
        bargap=0.35,
    )
    return fig


def comparison_chart(df: pd.DataFrame) -> go.Figure:
    """Grouped horizontal bars: each feature compared across Hard / Clay / Grass."""
    # Order features by their pooled ("All") importance so the biggest is on top.
    order = df[df["surface"] == "All"].sort_values("pct_of_r2")["feature"].tolist()
    fig = go.Figure()
    # Add traces in reverse so the legend and bar order read Hard, Clay, Grass top-down.
    for surface in ["Grass", "Clay", "Hard"]:
        sub = df[df["surface"] == surface].set_index("feature").loc[order]
        fig.add_trace(
            go.Bar(
                name=surface,
                x=sub["pct_of_r2"],
                y=sub["label"],
                orientation="h",
                marker=dict(color=SURFACE_COLORS[surface], cornerradius=4,
                            line=dict(width=0)),
                hovertemplate=f"<b>{surface}</b> · %{{y}}<br>%{{x:.1f}}% of explained R²<extra></extra>",
            )
        )
    fig.update_layout(
        barmode="group",
        bargap=0.3,
        bargroupgap=0.12,  # small gap between adjacent bars in a group
        height=520,
        margin=dict(l=10, r=20, t=10, b=40),
        xaxis=dict(title="% of explained R²", ticksuffix="%", zeroline=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, traceorder="reversed"),
    )
    return fig


# --- Insights --------------------------------------------------------------------
def key_insights(df: pd.DataFrame) -> list[str]:
    """Three data-driven sentences, recomputed from whatever the CSV contains."""
    surfaces = ["Hard", "Clay", "Grass"]
    insights = []

    # 1) The single biggest factor overall, and how it compares to serve stats.
    allsurf = df[df["surface"] == "All"].set_index("feature")["pct_of_r2"]
    top = allsurf.idxmax()
    serve_total = allsurf[["first_serve_won", "second_serve_won"]].sum()
    insights.append(
        f"Across all surfaces, **{LABELS[top].lower()}** is the single biggest factor, "
        f"accounting for **{allsurf[top]:.0f}%** of the model's explained R² on its own "
        f"(1st + 2nd serve points won combined: {serve_total:.0f}%)."
    )

    # 2) Where return points matter most vs least.
    ret = {s: pct(df, s, "return_pts_won") for s in surfaces}
    hi, lo = max(ret, key=ret.get), min(ret, key=ret.get)
    insights.append(
        f"On **{hi.lower()}**, return points won explains **{ret[hi]:.1f}%** — "
        f"more than on {lo.lower()} ({ret[lo]:.1f}%). Returning is the most valuable skill on every surface."
    )

    # 3) The feature whose importance varies most across surfaces.
    wide = df[df["surface"].isin(surfaces)].pivot(index="feature", columns="surface", values="pct_of_r2")
    # Relative spread so small-but-swingy features (aces, break points) can surface.
    spread = (wide.max(axis=1) - wide.min(axis=1)) / wide.mean(axis=1)
    feat = spread.drop(["df_rate", "first_serve_in"]).idxmax()  # skip tiny-share features
    row = wide.loc[feat]
    insights.append(
        f"**{LABELS[feat]}** swings most by surface: it carries **{row.max():.1f}%** on "
        f"{row.idxmax().lower()} vs {row.min():.1f}% on {row.idxmin().lower()} — "
        f"surface changes how much the 'clutch' and 'free point' stats matter."
    )
    return insights


# --- Page ------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="What Wins Tennis Matches?", page_icon="🎾", layout="wide")
    df = load_results()

    st.title("🎾 What Wins Tennis Matches?")
    st.caption(
        "How much each in-match stat explains winning, by court surface. "
        "ATP tour-level matches 2015–2024 · data: Jeff Sackmann (CC BY-NC-SA 4.0)."
    )

    # Key insights box
    with st.container(border=True):
        st.subheader("Key insights")
        for sentence in key_insights(df):
            st.markdown(f"- {sentence}")

    tab_one, tab_compare, tab_method = st.tabs(["By surface", "Compare surfaces", "Method"])

    # --- Single-surface view ---
    with tab_one:
        surface = st.selectbox("Surface", SURFACES, index=0)
        sub = df[df["surface"] == surface]
        c1, c2, c3 = st.columns(3)
        c1.metric("Matches", f"{int(sub['n_matches'].iloc[0]):,}")
        c2.metric("Model R² (linear prob.)", f"{sub['model_r2'].iloc[0]:.3f}")
        c3.metric("Logit test accuracy", f"{sub['logit_accuracy'].iloc[0]:.1%}")
        st.plotly_chart(surface_bar_chart(df, surface), width="stretch")
        st.caption("Bars sum to 100% of the variance the model explains. Hover for the raw R² share.")

    # --- Comparison view ---
    with tab_compare:
        st.plotly_chart(comparison_chart(df), width="stretch")
        with st.expander("Table view"):
            table = (
                df[df["surface"] != "All"]
                .pivot(index="label", columns="surface", values="pct_of_r2")[["Hard", "Clay", "Grass"]]
                .sort_values("Hard", ascending=False)
                .round(1)
            )
            table.index.name = "Feature (% of explained R²)"
            st.dataframe(table, width="stretch")

    # --- Method ---
    with tab_method:
        st.markdown(
            """
**Data.** Every completed ATP tour-level match 2015–2024 on hard, clay or grass
(retirements and walkovers removed). For each player we compute 7 serve/return rates,
then take the difference between two randomly labelled players A and B.

**Model.** A linear regression predicts whether player A won from those 7 differences.
Its **R²** is the share of the variation in match outcomes the stats explain (~0.6).

**LMG / Shapley decomposition — in plain language.** The stats overlap: a big server
wins more 1st-serve points *and* hits more aces. So "how much does each one matter?"
depends on the order you add them to the model. LMG solves this fairly: it tries
**every possible order** of adding the 7 stats, records how much R² each stat adds
at the moment it joins, and averages. This is the Shapley value from game theory —
each stat gets credit for its average contribution, and the credits add up exactly to
the model's total R².

**Sanity check.** A logistic regression trained on 80% of matches classifies the winner
of the held-out 20% with ~93% accuracy.

**Limitation.** These are *in-match* stats, so this describes what winning looks like
on each surface — it is not a pre-match prediction model.
            """
        )


if __name__ == "__main__":
    main()
