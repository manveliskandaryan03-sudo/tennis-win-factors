# 🎾 What Wins Tennis Matches?

**AQX Sports Analytics Data Bowl 4.0**: which in-match stats explain who wins, and how
does that change from hard courts to clay to grass?

We fit a model that explains about **62% of the variance in ATP match outcomes** from
seven serve and return stats. Then we use **LMG / Shapley R² decomposition** to split
that 62% fairly among the stats, separately for each surface.

**▶ Live dashboard: https://tennis-win-factors-cf2nhibwx4dq2jkuvenk63.streamlit.app/**

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://tennis-win-factors-cf2nhibwx4dq2jkuvenk63.streamlit.app/)

---

## Problem

Coaches and players get a long list of match stats: aces, first-serve %, break points saved,
return points won, and more. These stats overlap. A big server hits more aces *and* wins more
first-serve points. So "which stat matters most?" has no simple answer from regression
coefficients or correlations alone. We want an **importance ranking that adds up**, and we
want to see how it changes with the court surface.

## Data

- **Source:** Jeff Sackmann's [`tennis_atp`](https://github.com/JeffSackmann/tennis_atp)
  match files, 2015–2024. Licensed
  [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). All credit for
  collecting the data goes to Jeff Sackmann / Tennis Abstract.
  - *Note:* as of October 2026 the original GitHub repo returns 404.
    `src/download_data.py` tries it first, then falls back to byte-identical archival
    mirrors. We checked this by matching SHA-1 hashes across two independent mirrors.
- **Cleaning (`src/prepare.py`):**
  27,672 raw matches → 25,836 used. We keep Hard/Clay/Grass only, drop retirements,
  walkovers and defaults, and drop matches with missing serve stats.
  - Hard: 15,138 · Clay: 7,870 · Grass: 2,828
- **Features (7 per player):** first-serve in %, first-serve points won %, second-serve
  points won %, break points saved % (0/0 → 1.0), ace rate, double-fault rate, and return
  points won %.
- **One row per match.** A coin flip (seed 42) decides which player is "A". Each
  feature is **A's rate minus B's rate**, and the target is `a_won` (49.8% ones).

## Method

1. **Linear probability model (OLS):** `a_won ~ 7 difference features`, fit for each
   surface and for all matches pooled.
2. **LMG (Shapley) decomposition of R²:** we fit OLS on all 2⁷ = 128 subsets of
   features. A feature's share is its **average gain in R² when it joins the model**, taken
   over every possible order of adding the features. These shares add up exactly to the
   full model's R². We check this with an assertion for every surface.
3. **Sanity check:** a logistic regression trained on an 80/20 stratified split.

| Surface | Matches | Model R² | Σ LMG shares | Logit test accuracy |
|---|---:|---:|---:|---:|
| All   | 25,836 | 0.616 | 0.616 ✓ | 92.9% |
| Hard  | 15,138 | 0.612 | 0.612 ✓ | 92.9% |
| Clay  |  7,870 | 0.628 | 0.628 ✓ | 94.3% |
| Grass |  2,828 | 0.615 | 0.615 ✓ | 92.2% |

## Key findings

Each stat's share of the explained R², in percent (each column sums to 100):

| Stat | All | Hard | Clay | Grass |
|---|---:|---:|---:|---:|
| **Return points won %** | **36.5** | **36.1** | **37.6** | **35.2** |
| 1st serve points won %  | 25.9 | 25.0 | 27.6 | 25.3 |
| 2nd serve points won %  | 18.1 | 18.5 | 18.0 | 16.1 |
| Break points saved %    | 10.6 | 11.1 |  9.0 | **13.0** |
| Ace rate                |  5.0 |  5.2 |  4.2 |  **6.6** |
| Double fault rate       |  2.3 |  2.4 |  2.4 |  2.0 |
| 1st serve in %          |  1.5 |  1.7 |  1.3 |  1.8 |

1. **Returning is the single most important skill on every surface.** Return points won
   accounts for 35–38% of the explained variance, more than any single serve stat.
   This holds even on grass, the most serve-friendly surface.
2. **Clay rewards rally and return play.** Clay has the highest share for return points won
   (37.6%) and for first-serve points won (27.6%). It has the lowest share for aces (4.2%)
   and for break points saved (9.0%).
3. **Grass rewards free points and clutch serving.** On grass, break points saved
   (13.0% vs 9.0% on clay) and aces (6.6% vs 4.2%) carry clearly more weight. Holding serve
   under pressure decides grass matches more often, because breaks are rare.
4. **First-serve %** (1.5%) and **double faults** (2.3%) explain almost nothing *once you
   know whether the points were won*. Getting the first serve in matters only through the
   points it produces.

*Robustness:* we bootstrapped the LMG shares 200 times per surface. The clay vs grass
gaps in break points saved, ace rate, return points won, and first and second serve points
won all have 95% intervals that exclude zero. The differences in double faults and
first-serve % are within noise.

## Takeaways for coaches & players

- **Train the return as hard as the serve.** On every surface it explains the most variance
  in who wins.
- **On clay**, focus on return depth and building points. Aces are worth the least there,
  so don't over-invest in first-serve pace at the cost of first-serve *points won*.
- **On grass**, practise serving under break-point pressure and aim for free points
  (aces, unreturned serves). These have the biggest surface-specific gains.
- **Don't chase first-serve %** for its own sake. What matters is the share of points won
  behind each serve. A safer, slower first serve that lands but gets attacked does not help.

## Limitations

- **Descriptive, not predictive.** These are *in-match* stats: they describe what a win
  looks like on each surface. They can't predict a match before it's played, because you
  only know the stats after the match ends.
- The linear probability model is used because LMG decomposes R², and it has a clean
  meaning for OLS. The logistic model is only a sanity check.
- Shares are relative to explained variance. They don't show how much one extra percentage
  point of a stat is worth.
- Men's ATP tour-level matches only. Grass has the smallest sample (2,828 matches).

## How to run

```bash
pip install -r requirements.txt
python src/download_data.py   # downloads 2015–2024 ATP match files into data/raw/
python src/prepare.py         # cleans → data/processed/matches.csv
python src/analysis.py        # LMG decomposition → data/processed/lmg_results.csv
streamlit run app.py          # dashboard
```

The dashboard reads only `data/processed/lmg_results.csv`, which is committed, so it
starts instantly. You don't need to download anything to view it.

## Project structure

```
tennis-win-factors/
├── app.py                 # Streamlit dashboard
├── src/
│   ├── download_data.py   # fetch raw CSVs (with mirror fallback)
│   ├── prepare.py         # cleaning + feature engineering
│   └── analysis.py        # OLS, LMG/Shapley decomposition, logit check
├── data/
│   ├── raw/               # (git-ignored) downloaded match files
│   └── processed/         # lmg_results.csv (committed), matches.csv (git-ignored)
├── requirements.txt
└── README.md
```

## License & credit

Data: © Jeff Sackmann / Tennis Abstract, licensed
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). Derived results in
`data/processed/` are shared under the same license.
