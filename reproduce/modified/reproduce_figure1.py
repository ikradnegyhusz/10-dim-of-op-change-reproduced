#!/usr/bin/env python3
"""Modified reproduction of Figure 1 of Monti et al. (2022), Sci. Rep. 12:17920.

Backbone: ``reproduce/from_original/reproduce_figure1.py`` (itself a literal port
of ``original_repo/notebook/3_odds-ratios.ipynb``).  The modifications are:

Panel (a) - comments
    Same estimator as the notebook (mean length-discounted score as p, Eq. 9
    count-based CI).  The 95% interval is now drawn asymmetrically (lower and
    upper log-odds bounds) instead of the notebook's ``yerr = CI_upper - OR``.

Panel (b) - posts
    The notebook kept one row per comment (and ``comment_id`` is actually the
    parent id in the released data, so sibling comments collapsed too).  The
    panel is now computed on **one row per post**: a post is a delta post if at
    least one of its comments received a delta, and each post contributes once.

Panel (c) - post x comment matrix
    The notebook restricted the no-delta baseline to comments that received an
    answer by the OP and used a single unseeded random permutation as the null.
    The modified version uses **all non-delta comments** as the baseline (which
    is what the paper describes: "probability of achieving a delta"), averages
    the null over several post permutations, and keeps the notebook's statistic
    (conditional OR of delta vs no-delta, difference of the two) and its
    one-sided p < 0.01 mask.  This reproduces 9 of the paper's 14 values (6 of
    them significant at p<0.01) and the five positive diagonal cells; the four
    negative "Support" cells cannot be reproduced from the released data (see
    ``JOURNAL.md``).

Table 2 is untouched: it is already exact (``reproduce_table2.py``).

Usage (from the repository root)::

    env/Scripts/python.exe reproduce/modified/reproduce_figure1.py
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA_PATH = ROOT / "data" / "changemyview-sociopol-processed.pickle.gz"
OUT_DIR = HERE / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_PERMUTATIONS = 300
ALPHA = 0.01

DIMENSION_COLUMNS = [
    "support", "knowledge", "conflict", "power", "similarity",
    "fun", "status", "trust", "identity",
]
DIMENSIONS_EXTENDED = DIMENSION_COLUMNS + ["nodimensions"]
DISPLAY = {d: d.capitalize() for d in DIMENSION_COLUMNS}
DISPLAY["nodimensions"] = "No dim"

# Published Figure 1c, read off the paper (post row, comment column), used only
# for the comparison CSV/printout.
PAPER_C = {
    ("support", "support"): 0.10, ("support", "nodimensions"): -0.10,
    ("conflict", "conflict"): 0.13, ("conflict", "identity"): -0.12,
    ("power", "knowledge"): -0.12, ("power", "power"): 0.22,
    ("power", "status"): -0.14, ("power", "fun"): -0.13,
    ("similarity", "knowledge"): -0.11, ("fun", "knowledge"): -0.09,
    ("fun", "fun"): 0.21, ("status", "knowledge"): -0.07,
    ("identity", "identity"): 0.21, ("nodimensions", "knowledge"): 0.07,
}


# --------------------------------------------------------------------------- helpers
def standardize_length(x: float, mu: float, std: float) -> float:
    """Length-discounting factor of paper Eq. 2."""
    z = (x - mu) / std
    if z >= 0:
        return 1 / (1 + z)
    z = -z
    return 2 - (1 / (1 + z))


def odds_ratio_ci(mean_delta: float, mean_no_delta: float,
                  n_delta: int, n_no_delta: int) -> tuple[float, float, float]:
    """Odds ratio and asymmetric 95% CI (Eq. 5 and Eq. 9)."""
    odds = (mean_delta / (1 - mean_delta)) / (mean_no_delta / (1 - mean_no_delta))
    # Cardinalities of the four cells, as in the notebook (int truncation).
    a = int(mean_delta * n_delta)
    b = n_delta - a
    c = int(mean_no_delta * n_no_delta)
    d = n_no_delta - c
    se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    low = float(np.exp(np.log(odds) - 1.96 * se))
    high = float(np.exp(np.log(odds) + 1.96 * se))
    return float(odds), low, high


def load_dataset() -> pd.DataFrame:
    t = time.time()
    data = pd.read_pickle(DATA_PATH)
    print(f"[data] {data.shape[0]:,} rows in {time.time() - t:.1f}s", flush=True)
    return data


# --------------------------------------------------------------------------- panel a
def panel_a(data: pd.DataFrame) -> pd.DataFrame:
    comment_columns = (
        ["is_op", "has_delta", "log_number_of_words"]
        + [f"{d}_comment" for d in DIMENSION_COLUMNS]
        + [f"{d}_weighted_comment" for d in DIMENSION_COLUMNS]
    )
    df = data[comment_columns].copy()
    df.columns = (["is_op", "has_delta", "log_number_of_words"]
                  + DIMENSION_COLUMNS
                  + [f"{d}_weighted" for d in DIMENSION_COLUMNS])

    # No-dimension indicator and length discount (Eq. 2).
    has_any = df[DIMENSION_COLUMNS].sum(axis=1) > 0
    df["nodimensions"] = (~has_any).astype(int)
    logs = df.loc[df["nodimensions"] == 1, "log_number_of_words"]
    mu, sd = float(logs.mean()), float(logs.std())
    df["nodimensions_weighted"] = df["log_number_of_words"].apply(
        lambda x: standardize_length(x, mu, sd)) * df["nodimensions"]

    delta = df[df["has_delta"] & ~df["is_op"]].dropna()
    no_delta = df[~df["has_delta"] & ~df["is_op"]].dropna()
    print(f"[figure 1a] delta comments {len(delta):,}; no delta {len(no_delta):,}",
          flush=True)

    rows = []
    for dim in DIMENSIONS_EXTENDED:
        column = f"{dim}_weighted"
        mean_d = float(delta[column].mean())
        mean_nd = float(no_delta[column].mean())
        or_, low, high = odds_ratio_ci(mean_d, mean_nd, len(delta), len(no_delta))
        rows.append({
            "dimension": dim,
            "mean_delta": mean_d,
            "mean_no_delta": mean_nd,
            "odds_ratio": or_,
            "ci_low": low,
            "ci_high": high,
            "n_delta": len(delta),
            "n_no_delta": len(no_delta),
        })
    return pd.DataFrame(rows).sort_values("odds_ratio", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- panel b
def panel_b(data: pd.DataFrame) -> pd.DataFrame:
    post_columns = (
        ["post_id", "log_number_of_words_post"]
        + [f"{d}_post" for d in DIMENSION_COLUMNS]
        + [f"{d}_weighted_post" for d in DIMENSION_COLUMNS]
    )
    posts = data[post_columns].drop_duplicates(subset=["post_id"]).dropna().copy()

    has_any = posts[[f"{d}_post" for d in DIMENSION_COLUMNS]].gt(0).any(axis=1)
    posts["nodimensions_post"] = (~has_any).astype(int)
    logs = posts.loc[posts["nodimensions_post"] == 1, "log_number_of_words_post"]
    mu, sd = float(logs.mean()), float(logs.std())
    posts["nodimensions_weighted_post"] = posts["log_number_of_words_post"].apply(
        lambda x: standardize_length(x, mu, sd)) * posts["nodimensions_post"]

    delta_post_ids = set(data.loc[data["has_delta"], "post_id"].unique())
    posts["is_delta"] = posts["post_id"].isin(delta_post_ids)
    delta = posts[posts["is_delta"]]
    no_delta = posts[~posts["is_delta"]]
    print(f"[figure 1b] delta posts {len(delta):,}; no-delta posts {len(no_delta):,} "
          f"(one row per post)", flush=True)

    rows = []
    for dim in DIMENSIONS_EXTENDED:
        column = f"{dim}_weighted_post"
        mean_d = float(delta[column].mean())
        mean_nd = float(no_delta[column].mean())
        or_, low, high = odds_ratio_ci(mean_d, mean_nd, len(delta), len(no_delta))
        rows.append({
            "dimension": dim,
            "mean_delta": mean_d,
            "mean_no_delta": mean_nd,
            "odds_ratio": or_,
            "ci_low": low,
            "ci_high": high,
            "n_delta": len(delta),
            "n_no_delta": len(no_delta),
        })
    return pd.DataFrame(rows).sort_values("odds_ratio", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- panel c
def _binary_matrices(frame: pd.DataFrame):
    post = frame[[f"{d}_post" for d in DIMENSION_COLUMNS]].gt(0).to_numpy(np.float32)
    post_nodim = (~post.astype(bool).any(axis=1)).astype(np.float32)
    post = np.column_stack([post, post_nodim])

    comment = frame[[f"{d}_comment" for d in DIMENSION_COLUMNS]].gt(0).to_numpy(np.float32)
    comment_nodim = (~comment.astype(bool).any(axis=1)).astype(np.float32)
    comment = np.column_stack([comment, comment_nodim])

    weight = frame[[f"{d}_weighted_comment" for d in DIMENSION_COLUMNS]].to_numpy(
        np.float32)
    weight = np.column_stack([weight, comment_nodim])
    return post, comment, weight


def _conditional(post: np.ndarray, weight: np.ndarray, mask: np.ndarray):
    """p(comment dim | post dim) and the weighted joint counts (Eq. 6-7)."""
    p = post[mask]
    joint = p.T @ weight[mask]
    n_post = p.sum(axis=0)
    return joint / np.maximum(n_post, 1)[:, None], joint


def panel_c(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    comment_columns = (["post_id", "has_delta"]
                       + [f"{d}_comment" for d in DIMENSION_COLUMNS]
                       + [f"{d}_weighted_comment" for d in DIMENSION_COLUMNS])
    post_columns = (["post_id"] + [f"{d}_post" for d in DIMENSION_COLUMNS])
    comments = data[comment_columns].copy()
    posts = data[post_columns].drop_duplicates(subset=["post_id"]).copy()
    frame = comments.merge(posts, on="post_id", how="inner", validate="many_to_one")
    del comments, posts

    post, comment, weight = _binary_matrices(frame)
    weighted_comment = weight * comment
    delta = frame["has_delta"].to_numpy()
    del frame
    print(f"[figure 1c] comments {len(delta):,}; delta {int(delta.sum()):,}; "
          f"null permutations {N_PERMUTATIONS}", flush=True)

    p_delta, joint_delta = _conditional(post, weighted_comment, delta)
    p_no_delta, joint_no_delta = _conditional(post, weighted_comment, ~delta)

    # Null model: randomly permute the post of each comment, average the
    # conditional probabilities and the joint counts over permutations.
    q_delta = np.zeros_like(p_delta)
    q_no_delta = np.zeros_like(p_no_delta)
    joint_q_delta = np.zeros_like(joint_delta)
    joint_q_no_delta = np.zeros_like(joint_no_delta)
    rng = np.random.default_rng(2024)
    for _ in range(N_PERMUTATIONS):
        shuffled = post[rng.permutation(len(post))]
        q, jq = _conditional(shuffled, weighted_comment, delta)
        q_delta += q
        joint_q_delta += jq
        q, jq = _conditional(shuffled, weighted_comment, ~delta)
        q_no_delta += q
        joint_q_no_delta += jq
    q_delta /= N_PERMUTATIONS
    q_no_delta /= N_PERMUTATIONS
    joint_q_delta /= N_PERMUTATIONS
    joint_q_no_delta /= N_PERMUTATIONS

    odds_delta = (p_delta / (1 - p_delta)) / (q_delta / (1 - q_delta))
    odds_no_delta = (p_no_delta / (1 - p_no_delta)) / (q_no_delta / (1 - q_no_delta))
    difference = odds_delta - odds_no_delta

    n_delta, n_no_delta = int(delta.sum()), int((~delta).sum())
    se_delta = np.sqrt(1 / joint_delta + 1 / n_delta
                       + 1 / joint_q_delta + 1 / n_delta)
    se_no_delta = np.sqrt(1 / joint_no_delta + 1 / n_no_delta
                          + 1 / joint_q_no_delta + 1 / n_no_delta)
    z = difference / np.sqrt(se_delta ** 2 + se_no_delta ** 2)
    pvalue = norm.sf(np.abs(z))  # one-sided, as in the notebook
    shown = np.where(pvalue <= ALPHA, difference, 0.0)

    def frame_of(matrix):
        return pd.DataFrame(matrix, index=DIMENSIONS_EXTENDED,
                            columns=DIMENSIONS_EXTENDED).rename_axis(
            index="post", columns="comment").reset_index()

    return frame_of(difference), frame_of(pvalue), frame_of(shown)


# --------------------------------------------------------------------------- plotting
def plot_odds_ratio(ax, table: pd.DataFrame, title: str, xlim: tuple[float, float]):
    ordered = table.sort_values("odds_ratio", ascending=False).reset_index(drop=True)
    y = np.arange(len(ordered))
    ax.errorbar(
        ordered["odds_ratio"], y,
        xerr=[ordered["odds_ratio"] - ordered["ci_low"],
              ordered["ci_high"] - ordered["odds_ratio"]],
        fmt="o", color="#3b6ea5", ecolor="#3b6ea5", elinewidth=1.2,
        capsize=3, markersize=5,
    )
    ax.axvline(1.0, linestyle="--", linewidth=1, color="gray")
    ax.set_yticks(y)
    ax.set_yticklabels([DISPLAY[d] for d in ordered["dimension"]])
    ax.invert_yaxis()
    ax.set_xlabel("Odds ratio")
    ax.set_title(title, fontsize=11)
    ax.set_xlim(*xlim)
    for yi, value in zip(y, ordered["odds_ratio"]):
        ax.annotate(f"{value:.2f}", (value, yi), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8)


def plot_figure(a, b, values, pvalues, path):
    fig = plt.figure(figsize=(16, 6.5))
    grid = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.6], wspace=0.35)
    plot_odds_ratio(fig.add_subplot(grid[0, 0]), a, "(a) Odds ratio (comments)", (0, 2.4))
    plot_odds_ratio(fig.add_subplot(grid[0, 1]), b, "(b) Odds ratio (posts)", (0.5, 1.6))

    ax = fig.add_subplot(grid[0, 2])
    matrix = values.set_index("post").to_numpy()
    pval = pvalues.set_index("post").to_numpy()
    displayed = np.where(pval <= ALPHA, matrix, np.nan)
    cmap = plt.get_cmap("coolwarm_r").copy()
    cmap.set_bad("white")
    im = ax.matshow(displayed, cmap=cmap,
                    norm=mcolors.TwoSlopeNorm(vcenter=0.0, vmin=-0.2, vmax=0.2))
    labels = [DISPLAY[d] for d in DIMENSIONS_EXTENDED]
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="left")
    ax.set_yticklabels(labels)
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    ax.set_xlabel("Comment")
    ax.set_ylabel("Post")
    ax.set_title("(c) Post x comment interaction", fontsize=11, pad=30)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if pval[i, j] <= ALPHA:
                ax.text(j, i, f"{matrix[i, j]:.2f}", va="center", ha="center",
                        fontsize=7)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- main
def main():
    data = load_dataset()

    print("\n=== Panel (a): comments ===", flush=True)
    a = panel_a(data)
    a.to_csv(OUT_DIR / "figure1a_comments.csv", index=False)
    print(a.to_string(index=False, float_format=lambda x: f"{x:.4f}"), flush=True)

    print("\n=== Panel (b): posts (one row per post) ===", flush=True)
    b = panel_b(data)
    b.to_csv(OUT_DIR / "figure1b_posts.csv", index=False)
    print(b.to_string(index=False, float_format=lambda x: f"{x:.4f}"), flush=True)

    print("\n=== Panel (c): post x comment interaction ===", flush=True)
    values, pvalues, shown = panel_c(data)
    values.to_csv(OUT_DIR / "figure1c_values.csv", index=False)
    pvalues.to_csv(OUT_DIR / "figure1c_pvalues.csv", index=False)
    shown.to_csv(OUT_DIR / "figure1c_shown.csv", index=False)

    val = values.set_index("post")
    pv = pvalues.set_index("post")
    comparison = pd.DataFrame([
        {"post": post, "comment": comment, "paper_value": paper_value,
         "reproduced_value": float(val.loc[post, comment]),
         "difference": float(val.loc[post, comment]) - paper_value,
         "pvalue": float(pv.loc[post, comment]),
         "significant": bool(pv.loc[post, comment] <= ALPHA)}
        for (post, comment), paper_value in PAPER_C.items()
    ])
    comparison.to_csv(OUT_DIR / "figure1c_vs_paper.csv", index=False)
    n_close = int((np.abs(comparison["difference"]) <= 0.05).sum())
    print(f"[figure 1c] {n_close}/{len(comparison)} paper cells within 0.05; "
          f"{int(comparison['significant'].sum())} of them significant at p<0.01",
          flush=True)
    print(comparison.to_string(index=False, float_format=lambda x: f"{x:.3f}"),
          flush=True)

    path = OUT_DIR / "figure1_reproduced.png"
    plot_figure(a, b, values, pvalues, path)
    print(f"\n[saved] {path}", flush=True)


if __name__ == "__main__":
    main()
