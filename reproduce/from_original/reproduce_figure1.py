#!/usr/bin/env python3
"""Reproduce Figure 1 of Monti et al. (2022), Sci. Rep. 12:17920.

Port of ``original_repo/notebook/3_odds-ratios.ipynb``.  The notebook cells are
quoted in the comments; the code follows them as closely as possible, including
their known quirks (see ``JOURNAL.md``).  Only compatibility fixes and dead code
removal were applied:

* a random seed is set before the null-model permutation (the notebook has none;
  without a seed the figure is not reproducible at all);
* the ``_weighted-comment_discounted`` matrix pickles loaded by the notebook's
  cell 27 are not shipped in the repository; the matrices computed by cell 26
  are used directly instead;
* ``matplotlib.cm.get_cmap`` (removed in matplotlib >= 3.9) is replaced by
  ``plt.get_cmap``;
* the full ``D.copy()`` of cell 16 is replaced by a column subset (memory only);
* diagnostic prints and quantities that never reach the figure (t-intervals of
  the means, ratio columns, the joint-probability matrix plot, etc.) are dropped;
* the notebook also computes the two sentiment scores in panels (a) and (b);
  they are kept in the CSV outputs but not drawn, because the published figure
  only shows the nine dimensions plus "No dim".

Usage (from the repository root)::

    env/Scripts/python.exe reproduce/from_original/reproduce_figure1.py
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

SEED = 1234
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- cell 1
dimensions_names = [
    "support", "knowledge", "conflict", "power", "similarity",
    "fun", "status", "trust", "identity", "romance",
]
dimensions_names_redux = [
    "support", "knowledge", "conflict", "power", "similarity",
    "fun", "status", "trust", "identity",
]
additional_dims_names_binarized = ["sentiment_pos", "sentiment_neg"]


def standardized_dimension_score(x, mu, std):
    """Length-discounting factor (cell 1, paper Eq. 2)."""
    zscore = (x - mu) / std
    if zscore >= 0:
        return 1 / (1 + zscore)
    else:
        zscore = -zscore
        return 2 - (1 / (1 + zscore))


# --------------------------------------------------------------------------- load
t = time.time()
D = pd.read_pickle(DATA_PATH)
print(f"[data] {DATA_PATH.name}: {D.shape[0]:,} rows in {time.time() - t:.1f}s",
      flush=True)

# --------------------------------------------------------------------------- cell 5
df_comments = D[[
    "post_id", "comment_id", "vader_sentiment_comment", "number_of_words",
    "score_comment", "is_op", "has_delta", "has_answer_by_op",
    "log_number_of_words", "support_comment", "knowledge_comment",
    "conflict_comment", "power_comment", "similarity_comment", "fun_comment",
    "status_comment", "trust_comment", "identity_comment", "romance_comment",
    "sentiment_pos_comment", "sentiment_neg_comment", "support_weighted_comment",
    "knowledge_weighted_comment", "conflict_weighted_comment",
    "power_weighted_comment", "similarity_weighted_comment", "fun_weighted_comment",
    "status_weighted_comment", "trust_weighted_comment", "identity_weighted_comment",
    "romance_weighted_comment", "sentiment_pos_weighted_comment",
    "sentiment_neg_weighted_comment",
]].copy()
df_comments.columns = [
    "post_id", "comment_id", "vader_sentiment", "number_of_words", "score_comment",
    "is_op", "has_delta", "has_answer_by_op", "log_number_of_words",
    "support", "knowledge", "conflict", "power", "similarity", "fun", "status",
    "trust", "identity", "romance", "sentiment_pos", "sentiment_neg",
    "support_weighted", "knowledge_weighted", "conflict_weighted",
    "power_weighted", "similarity_weighted", "fun_weighted", "status_weighted",
    "trust_weighted", "identity_weighted", "romance_weighted",
    "sentiment_pos_weighted", "sentiment_neg_weighted",
]

# --------------------------------------------------------------------------- cell 13
df_posts = D[[
    "post_id", "vader_sentiment_post", "score_post", "number_of_words_post",
    "log_number_of_words_post", "support_post", "knowledge_post", "conflict_post",
    "power_post", "similarity_post", "fun_post", "status_post", "trust_post",
    "identity_post", "romance_post", "sentiment_pos_post", "sentiment_neg_post",
    "support_weighted_post", "knowledge_weighted_post", "conflict_weighted_post",
    "power_weighted_post", "similarity_weighted_post", "fun_weighted_post",
    "status_weighted_post", "trust_weighted_post", "identity_weighted_post",
    "romance_weighted_post", "sentiment_pos_weighted_post",
    "sentiment_neg_weighted_post",
]].copy()
df_posts.columns = [
    "post_id", "vader_sentiment", "score_post", "number_of_words_post",
    "log_number_of_words_post", "support", "knowledge", "conflict", "power",
    "similarity", "fun", "status", "trust", "identity", "romance",
    "sentiment_pos", "sentiment_neg", "support_weighted", "knowledge_weighted",
    "conflict_weighted", "power_weighted", "similarity_weighted", "fun_weighted",
    "status_weighted", "trust_weighted", "identity_weighted", "romance_weighted",
    "sentiment_pos_weighted", "sentiment_neg_weighted",
]
df_posts = df_posts.drop_duplicates(subset=["post_id"])

# --------------------------------------------------------------------------- cell 16
# The notebook does ``df_paired = D.copy()`` and then filters.  D has 88 columns
# and copying it costs ~2.7 GB, so only the columns used downstream are kept.
paired_columns = (
    ["post_id", "comment_id", "has_delta", "has_answer_by_op", "number_of_words",
     "number_of_words_post", "log_number_of_words_post"]
    + [f"{d}_{t}" for d in dimensions_names for t in ("comment", "post")]
    + [f"{d}_weighted_{t}" for d in dimensions_names for t in ("comment", "post")]
    + ["sentiment_pos_weighted_post", "sentiment_neg_weighted_post"]
)
df_paired = D[paired_columns].copy()
df_paired = df_paired[
    (df_paired["has_delta"] == True) | (df_paired["has_answer_by_op"] == True)
]
del D

# --------------------------------------------------------------------------- cell 8
df_comments["nodimensions"] = df_comments[
    [f"{d}" for d in dimensions_names_redux]
].sum(axis=1)
df_comments["nodimensions"] = df_comments["nodimensions"].apply(
    lambda x: 0 if x > 0 else 1)
log_lengths = df_comments[df_comments["nodimensions"] == 1]["log_number_of_words"]
avg = np.mean(log_lengths)
stdev = np.std(log_lengths)
df_comments["nodimensions_weighted"] = df_comments["log_number_of_words"].apply(
    lambda x: standardized_dimension_score(x, avg, stdev)
) * df_comments["nodimensions"]

# --------------------------------------------------------------------------- cell 9
df_no_delta = df_comments[df_comments["has_delta"] == False].dropna()
df_delta = df_comments[df_comments["has_delta"] == True].dropna()
df_delta = df_delta[df_delta["is_op"] == False]
df_no_delta = df_no_delta[df_no_delta["is_op"] == False]
print(f"[figure 1a] comments with delta: {len(df_delta):,}; "
      f"without: {len(df_no_delta):,}", flush=True)


# --------------------------------------------------------------------------- cell 10 / 23
def odds_ratio_and_ci(scores_delta, scores_no_delta):
    """Odds ratio and the upper error bar used by the notebook."""
    avg_delta = float(scores_delta.mean())
    avg_no_delta = float(scores_no_delta.mean())
    odds_ratio = (avg_delta / (1 - avg_delta)) / (avg_no_delta / (1 - avg_no_delta))
    log_odds_ratio = np.log(odds_ratio)

    _dim_delta = int(avg_delta * len(scores_delta))
    _no_dim_delta = len(scores_delta) - _dim_delta
    _dim_no_delta = int(avg_no_delta * len(scores_no_delta))
    _no_dim_no_delta = len(scores_no_delta) - _dim_no_delta

    standard_error_log_or = np.sqrt(
        1 / _dim_delta + 1 / _no_dim_delta
        + 1 / _dim_no_delta + 1 / _no_dim_no_delta)
    ci_log_or = 1.96 * standard_error_log_or
    ci_upper = np.power(np.e, log_odds_ratio + ci_log_or)
    ci_lower = np.power(np.e, log_odds_ratio - ci_log_or)
    return {
        "avg_delta": avg_delta,
        "avg_no_delta": avg_no_delta,
        "odds_ratio": float(odds_ratio),
        "ci_log_upper": float(ci_upper),
        "ci_log_lower": float(ci_lower),
        # The notebook plots ``yerr = ci_OR_upper - OR`` (symmetric in matplotlib).
        "yerr": float(ci_upper - odds_ratio),
    }


panel_a_rows = []
all_dims_a = dimensions_names_redux + additional_dims_names_binarized + ["nodimensions"]
for dim in all_dims_a:
    y = odds_ratio_and_ci(df_delta[f"{dim}_weighted"],
                          df_no_delta[f"{dim}_weighted"])
    y["dimension"] = dim
    panel_a_rows.append(y)
panel_a = pd.DataFrame(panel_a_rows).sort_values(
    "odds_ratio", ascending=False).reset_index(drop=True)
panel_a.to_csv(OUT_DIR / "figure1a_comments.csv", index=False)
print("[figure 1a] all computed dimensions (the notebook's loop):", flush=True)
print(panel_a[["dimension", "odds_ratio", "ci_log_lower", "ci_log_upper"]]
      .to_string(index=False, float_format=lambda x: f"{x:.4f}"), flush=True)

# --------------------------------------------------------------------------- cell 14
df_comments_shuffled = df_comments.copy(deep=True)
np.random.seed(SEED)  # not in the notebook; needed to make the null model reproducible
df_comments_shuffled["post_id"] = np.random.permutation(
    df_comments_shuffled["post_id"].values)
# create shuffled versions of comments and join it with posts
df_paired_shuffled = pd.merge(
    df_comments_shuffled, df_posts,
    left_on="post_id", right_on="post_id", how="inner",
    suffixes=("_comment", "_post"))
del df_comments_shuffled

# --------------------------------------------------------------------------- cell 16
df_paired_shuffled = df_paired_shuffled[
    (df_paired_shuffled["has_delta"] == True)
    | (df_paired_shuffled["has_answer_by_op"] == True)
]

# --------------------------------------------------------------------------- cell 18
for t_ in ["post", "comment"]:
    df_paired[f"nodimensions_{t_}"] = sum(
        [df_paired[f"{d}_{t_}"] for d in dimensions_names_redux])
    df_paired[f"nodimensions_{t_}"] = df_paired[f"nodimensions_{t_}"].apply(
        lambda x: 0 if x > 0 else 1)
    df_paired[f"nodimensions_weighted_{t_}"] = df_paired[f"nodimensions_{t_}"]

    df_paired[f"anydimensions_{t_}"] = sum(
        [df_paired[f"{d}_{t_}"] for d in dimensions_names_redux])
    df_paired[f"anydimensions_{t_}"] = df_paired[f"anydimensions_{t_}"].apply(
        lambda x: 1 if x > 0 else 0)
    df_paired[f"anydimensions_weighted_{t_}"] = df_paired[f"anydimensions_{t_}"]

    if t_ == "post":
        log_lengths = df_paired[
            df_paired[f"nodimensions_{t_}"] == 1]["log_number_of_words_post"]
        avg = np.mean(log_lengths)
        stdev = np.std(log_lengths)
        df_paired[f"nodimensions_weighted_{t_}"] = df_paired[
            "log_number_of_words_post"].apply(
            lambda x: standardized_dimension_score(x, avg, stdev)
        ) * df_paired[f"nodimensions_{t_}"]
        df_paired[f"anydimensions_weighted_{t_}"] = np.sum(
            [df_paired[f"{d}_weighted_{t_}"] for d in dimensions_names_redux])

    df_paired_shuffled[f"nodimensions_{t_}"] = sum(
        [df_paired_shuffled[f"{d}_{t_}"] for d in dimensions_names_redux])
    df_paired_shuffled[f"nodimensions_{t_}"] = df_paired_shuffled[
        f"nodimensions_{t_}"].apply(lambda x: 0 if x > 0 else 1)
    df_paired_shuffled[f"nodimensions_weighted_{t_}"] = df_paired_shuffled[
        f"nodimensions_{t_}"]

# --------------------------------------------------------------------------- cell 20
field_selection = ["post_id", "comment_id", "has_delta", "has_answer_by_op"]
field_selection += [d + "_post" for d in dimensions_names]
field_selection += [d + "_weighted_post" for d in dimensions_names]
field_selection += ["sentiment_pos_weighted_post", "sentiment_neg_weighted_post"]
field_selection += [
    "nodimensions_comment", "nodimensions_post",
    "nodimensions_weighted_comment", "nodimensions_weighted_post",
]
field_selection += [
    "anydimensions_comment", "anydimensions_post",
    "anydimensions_weighted_comment", "anydimensions_weighted_post",
]
df_posts_delta_nondelta = df_paired[field_selection].drop_duplicates(
    subset=["post_id", "comment_id", "has_delta"])

# --------------------------------------------------------------------------- cell 21
df_posts_delta = df_posts_delta_nondelta[
    df_posts_delta_nondelta["has_delta"] == True
].drop_duplicates().dropna()

# --------------------------------------------------------------------------- cell 22
df_posts_nodelta = df_posts_delta_nondelta[
    ~df_posts_delta_nondelta["post_id"].isin(df_posts_delta["post_id"])
].drop_duplicates().dropna()
print(f"[figure 1b, notebook's literal unit] rows with delta: {len(df_posts_delta):,} "
      f"({df_posts_delta['post_id'].nunique():,} posts); without: "
      f"{len(df_posts_nodelta):,} ({df_posts_nodelta['post_id'].nunique():,} posts)",
      flush=True)

# --------------------------------------------------------------------------- cell 23
panel_b_rows = []
all_dims_b = dimensions_names_redux + ["sentiment_pos", "sentiment_neg", "nodimensions"]
for dim in all_dims_b:
    y = odds_ratio_and_ci(df_posts_delta[f"{dim}_weighted_post"],
                          df_posts_nodelta[f"{dim}_weighted_post"])
    y["dimension"] = dim
    panel_b_rows.append(y)
panel_b = pd.DataFrame(panel_b_rows).sort_values(
    "odds_ratio", ascending=False).reset_index(drop=True)
panel_b.to_csv(OUT_DIR / "figure1b_posts.csv", index=False)
print("[figure 1b] all computed dimensions (the notebook's loop):", flush=True)
print(panel_b[["dimension", "odds_ratio", "ci_log_lower", "ci_log_upper"]]
      .to_string(index=False, float_format=lambda x: f"{x:.4f}"), flush=True)

# --------------------------------------------------------------------------- cell 26
dimensions_names_extended = dimensions_names_redux + ["nodimensions"]
has_answer_filter = False

matrices_discounted = {True: None, False: None}
for is_delta in [True, False]:
    dicts = {"normal": {}, "shuffled": {}}
    for d, data_type in zip([df_paired, df_paired_shuffled], ["normal", "shuffled"]):
        d = d[d["has_delta"] == is_delta]
        if has_answer_filter:
            d = d[(d["has_delta"] == True) | (d["has_answer_by_op"] == True)]
        d = d[d["number_of_words"] > 0]
        d = d[d["number_of_words_post"] > 0]

        d = d[["number_of_words", "number_of_words_post"]
              + [f"{dim}_post" for dim in dimensions_names_extended]
              + [f"{dim}_comment" for dim in dimensions_names_extended]
              + [f"{dim}_weighted_post" for dim in dimensions_names_extended]
              + [f"{dim}_weighted_comment" for dim in dimensions_names_extended]]
        d = d.dropna()
        tot_pairs = len(d)
        print(f"[figure 1c] is_delta={is_delta} {data_type}: {tot_pairs:,} pairs",
              flush=True)

        dim2matches = dicts[data_type]
        for dim1 in dimensions_names_extended:
            dim2matches[dim1] = {}
            for dim2 in dimensions_names_extended:
                # The notebook uses .apply(bool)/.apply(int); astype is the same
                # computation without the per-element Python loop.
                d["in_post"] = d[f"{dim1}_post"].astype(bool)
                d["in_comment"] = d[f"{dim2}_comment"].astype(bool)
                d["in_both"] = d["in_comment"] & d["in_post"]
                d["in_post"] = d["in_post"].astype(int)
                d["in_comment"] = d["in_comment"].astype(int)
                d["in_both"] = d["in_both"].astype(int)

                d["in_both_weighted"] = d[f"{dim2}_weighted_comment"] * d["in_both"]
                p_dimpost_and_dimcomment = (sum(d["in_both_weighted"]), tot_pairs)
                p_dimcomment_given_dimpost = (sum(d["in_both_weighted"]),
                                              sum(d["in_post"]))
                dim2matches[dim1][dim2] = (p_dimpost_and_dimcomment,
                                           p_dimcomment_given_dimpost)

    matrix_discounted = [[0 for _ in dimensions_names_extended]
                         for _ in dimensions_names_extended]
    for i, dim1 in enumerate(dimensions_names_extended):
        for j, dim2 in enumerate(dimensions_names_extended):
            p_joint_pair, p_cond_pair = dicts["normal"][dim1][dim2]
            p_joint = p_joint_pair[0] / p_joint_pair[1]
            p_cond = p_cond_pair[0] / p_cond_pair[1]
            p_joint_shuffled_pair, p_cond_shuffled_pair = dicts["shuffled"][dim1][dim2]
            p_joint_shuffled = p_joint_shuffled_pair[0] / p_joint_shuffled_pair[1]
            p_cond_shuffled = p_cond_shuffled_pair[0] / p_cond_shuffled_pair[1]

            stderr_log_joint_pval = np.sqrt(
                1 / p_joint_pair[0] + 1 / p_joint_pair[1]
                + 1 / p_joint_shuffled_pair[0] + 1 / p_joint_shuffled_pair[1])

            or_joint = (p_joint / (1 - p_joint)) / (
                p_joint_shuffled / (1 - p_joint_shuffled))
            or_cond = (p_cond / (1 - p_cond)) / (
                p_cond_shuffled / (1 - p_cond_shuffled))
            matrix_discounted[i][j] = (or_joint, or_cond, stderr_log_joint_pval)
    matrices_discounted[is_delta] = matrix_discounted

# --------------------------------------------------------------------------- cell 27
p_type = "cond"  # 'cond' as in the notebook

M = {True: None, False: None}
for is_delta in matrices_discounted:
    matrix = matrices_discounted[is_delta]
    matrix_joint = [[0 for _ in dimensions_names_extended]
                    for _ in dimensions_names_extended]
    matrix_cond = [[0 for _ in dimensions_names_extended]
                   for _ in dimensions_names_extended]
    matrix_stderr = [[0 for _ in dimensions_names_extended]
                     for _ in dimensions_names_extended]
    for i, dim1 in enumerate(dimensions_names_extended):
        for j, dim2 in enumerate(dimensions_names_extended):
            matrix_joint[i][j] = matrix[i][j][0]
            matrix_cond[i][j] = matrix[i][j][1]
            matrix_stderr[i][j] = matrix[i][j][2]
    if p_type == "cond":
        matrix = matrix_cond
    elif p_type == "joint":
        matrix = matrix_joint
    M[is_delta] = (matrix, matrix_stderr)

diff_matrix = [[0 for _ in dimensions_names_extended]
               for _ in dimensions_names_extended]
pval_matrix = [[0 for _ in dimensions_names_extended]
               for _ in dimensions_names_extended]
for i, dim1 in enumerate(dimensions_names_extended):
    for j, dim2 in enumerate(dimensions_names_extended):
        difference = M[True][0][i][j] - M[False][0][i][j]
        errors = np.sqrt(M[True][1][i][j] ** 2 + M[False][1][i][j] ** 2)
        Z = difference / errors
        pval = norm.sf(abs(Z))
        pval_matrix[i][j] = pval
        diff_matrix[i][j] = difference if pval <= 0.01 else 0

values = pd.DataFrame(diff_matrix, index=dimensions_names_extended,
                      columns=dimensions_names_extended).rename_axis(
    index="post", columns="comment")
pvalues = pd.DataFrame(pval_matrix, index=dimensions_names_extended,
                       columns=dimensions_names_extended).rename_axis(
    index="post", columns="comment")
values.reset_index().to_csv(OUT_DIR / "figure1c_values.csv", index=False)
pvalues.reset_index().to_csv(OUT_DIR / "figure1c_pvalues.csv", index=False)
values.reset_index().to_csv(OUT_DIR / "figure1c_shown.csv", index=False)
n_significant = int(np.sum(np.array(pval_matrix) <= 0.01))
print(f"[figure 1c] {n_significant} cells with p<=0.01", flush=True)
print(values.round(3).to_string(), flush=True)

# --------------------------------------------------------------------------- figure
def plot_odds_ratio_panel(ax, panel, title, dims):
    panel = panel[panel["dimension"].isin(dims)].sort_values(
        "odds_ratio", ascending=False).reset_index(drop=True)
    y = np.arange(len(panel))
    for yi, (_, row) in zip(y, panel.iterrows()):
        color = "b" if row["avg_delta"] > row["avg_no_delta"] else "r"
        ax.plot([yi], [row["odds_ratio"]], "o", markersize=2, color=color)
        ax.errorbar([yi], [row["odds_ratio"]], yerr=[row["yerr"]],
                    elinewidth=1, fmt=f"{color}-")
    ax.axhline(1, linestyle="--", linewidth=1, color="gray")
    ax.set_xticks(y)
    ax.set_xticklabels([d.capitalize() if d != "nodimensions" else "No dim"
                        for d in panel["dimension"]], rotation=90)
    ax.set_ylabel("odds ratio")
    ax.set_title(title, fontsize=11)


fig = plt.figure(figsize=(17, 6))
grid = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.5], wspace=0.3)
plot_odds_ratio_panel(fig.add_subplot(grid[0, 0]), panel_a,
                      "(a) Odds ratio (comments)",
                      dimensions_names_redux + ["nodimensions"])
plot_odds_ratio_panel(fig.add_subplot(grid[0, 1]), panel_b,
                      "(b) Odds ratio (posts)",
                      dimensions_names_redux + ["nodimensions"])

ax = fig.add_subplot(grid[0, 2])
shown = np.where(np.array(pval_matrix) <= 0.01, np.array(diff_matrix), np.nan)
cmap = plt.get_cmap("PiYG").copy()
cmap.set_bad("white")
cax = ax.matshow(shown, interpolation="nearest", cmap=cmap,
                 norm=mcolors.TwoSlopeNorm(vcenter=0.0))
labels = [d.capitalize() if d != "nodimensions" else "No dim"
          for d in dimensions_names_extended]
ax.set_xticks(range(len(labels)))
ax.set_yticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=90)
ax.set_yticklabels(labels)
fig.colorbar(cax, fraction=0.046, pad=0.04)
ax.set_ylabel("Post")
ax.set_xlabel("Comment")
ax.xaxis.set_label_position("top")
ax.set_title("(c) Post x comment interaction", fontsize=11, pad=30)
for i in range(len(dimensions_names_extended)):
    for j in range(len(dimensions_names_extended)):
        if pval_matrix[i][j] <= 0.01:
            ax.text(j, i, f"{diff_matrix[i][j]:.2f}", va="center", ha="center",
                    fontsize=7)
fig.savefig(OUT_DIR / "figure1_reproduced.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"[saved] {OUT_DIR / 'figure1_reproduced.png'}", flush=True)
