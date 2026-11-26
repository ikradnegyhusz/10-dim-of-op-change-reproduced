from __future__ import annotations

import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm  # noqa: F401  (imported by the notebook)
import statsmodels.formula.api as smf
import statsmodels.stats  # noqa: F401  (imported by the notebook)
from statsmodels.stats.multitest import multipletests

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA_PATH = ROOT / "data" / "changemyview-sociopol-processed.pickle.gz"
OUT_DIR = HERE / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DIMENSION_COLUMNS = [
    "support", "knowledge", "conflict", "power", "similarity",
    "fun", "status", "trust", "identity",
]
Z_COLUMNS = [f"{d}_weighted_comment_z" for d in DIMENSION_COLUMNS] + [
    "sentiment_pos_weighted_comment_z",
    "sentiment_neg_weighted_comment_z",
]
CONFOUNDER_COLUMNS = [
    "post_leftwing", "post_rightwing", "comment_leftwing", "comment_rightwing",
    "both_polarized", "diff_side", "shared_sub",
]


#  cell 4
nice_columns = lambda c: (c
            .replace('_z', '')
            .replace('_weighted_comment', '')
            .replace('poster', 'Post')
            .replace('log_num_links', 'Num. links (log)')
            .replace('_', ' ')
            .replace('republican', 'right-wing')
            .replace('democrat', 'left-wing')
            .replace('twing', 't-wing')
#             .replace('post', 'OP')
            .replace('pos', 'pos.').replace('neg', 'neg.').replace('pos.t', 'post')
            .title()
            .replace("Diff ", "Diff. ")
            .replace("Op And Commenter Opinionated", "Both polarized")
            .replace("Op And Commenter Diff Opinion", "Diff. side")
            .replace(" Sub", " Group")
            .replace("[T.True]", "")
            .replace(":", " \\& ")
)

pvalue2asterisk = lambda x: (
    '***' if x < 0.001 else '**' if x < 0.01 else '*' if x < 0.05 else ''
)


def add_results_to_table(res, df=None, method='fdr_bh'):
    """Cell 4, with ``df.append`` replaced by ``pandas.concat``."""
    K = len(res.params)
    # https://www.statease.com/docs/v12/contents/advanced-topics/glm/adj-mcfadden-pseudo-r-squared/
    adj_pseudo_r2 = 1 - (res.llf - K) / res.llnull
    main_stat = pd.Series('{:.5f}'.format(adj_pseudo_r2), index=['Adj. Pseudo-$R^2$'])
    r = pd.concat([res.pvalues, np.exp(res.params)], axis=1)
    r.columns = ['pvalues_uncorrected', 'logits']
    r['pvalues'] = statsmodels.stats.multitest.multipletests(
        r.pvalues_uncorrected, method=method)[1]
    # .iloc instead of x[0] / x[1]: positional indexing on a labelled Series is
    # not allowed any more in pandas 3.
    new_row = r.transpose().apply(
        lambda x: '{:.3f}'.format(x.iloc[1]) + pvalue2asterisk(x.iloc[0]))
    new_row = pd.concat([main_stat, new_row])
    if df is None:
        return pd.DataFrame([new_row])
    else:
        return pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)


#  cells 2-3
t = time.time()
df = pd.read_pickle(DATA_PATH)[
    ['has_delta'] + CONFOUNDER_COLUMNS + Z_COLUMNS
].copy()
df.has_delta = df.has_delta.astype(np.int64)
print(f"[data] {df.shape[0]:,} rows loaded in {time.time() - t:.1f}s "
      f"({df['has_delta'].mean() * 100:.2f}% with delta)", flush=True)

#  cell 6
ALL_DIMS = [c.replace('_weighted_comment_z', '') for c in df.columns
            if 'weighted_comment_z' in c]

#  cell 7
np.random.seed(1234)

#  cell 8
base_model_results = smf.logit(formula="has_delta ~ " + ' + '.join(
    [c + '_weighted_comment_z' for c in ALL_DIMS[:-2]]
), data=df).fit(disp=0)
print(f"[dims] base model fitted (converged: "
      f"{base_model_results.mle_retvals['converged']})", flush=True)

#  cell 14
DIMS = list(np.array(list(base_model_results.pvalues.index))[
    statsmodels.stats.multitest.multipletests(
        base_model_results.pvalues,
        alpha=0.01,
        method='fdr_by',
    )[0]
])
DIMS.remove('Intercept')
DIMS = [d.replace('_weighted_comment_z', '') for d in DIMS]
print(f"[dims] kept at 0.01 (fdr_by): {', '.join(DIMS)}", flush=True)


#  cell 23
def run_log_reg(common_factors=''):
    np.random.seed(1234)
    homophily_results_df = None
    sentiments = "sentiment_pos_weighted_comment_z + sentiment_neg_weighted_comment_z"
    for confounders, use_social_dim in [
        (f"{sentiments}", False),
        (f"{sentiments} + post_leftwing + post_rightwing + comment_leftwing + comment_rightwing", False),
        (f"{sentiments} + shared_sub", False),
        (f"{sentiments} + both_polarized + diff_side + both_polarized*diff_side", False),
        (f"{sentiments}", True),
        (f"{sentiments} + post_leftwing + post_rightwing + comment_leftwing + comment_rightwing", True),
        (f"{sentiments} + shared_sub", True),
        (f"{sentiments} + both_polarized + diff_side + both_polarized*diff_side", True),
    ]:
        formula = "has_delta ~ " + common_factors
        if use_social_dim:
            formula += ' + '.join([c + '_weighted_comment_z' for c in DIMS])
        if confounders:
            formula += (" + " + confounders)
        t_fit = time.time()
        homophily_results = smf.logit(formula=formula, data=df).fit(disp=0)
        print(f"[model] fitted in {time.time() - t_fit:.1f}s: {formula}", flush=True)
        homophily_results_df = add_results_to_table(homophily_results,
                                                    homophily_results_df)
    return homophily_results_df


homophily_results_df = run_log_reg()

#  cell 24
homophily_results_table = homophily_results_df.sort_values(
    by='Adj. Pseudo-$R^2$').copy()
homophily_results_table.columns = [
    nice_columns(t) for t in homophily_results_table.columns]
homophily_results_table[homophily_results_table.isna()] = ""
homophily_results_table = homophily_results_table.transpose()
homophily_results_table.columns = list(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:len(homophily_results_table.columns)])

#  cell 25
homophily_results_table = homophily_results_table.reindex([
    'Adj. Pseudo-$R^2$', 'Intercept',
    'Support', 'Knowledge', 'Conflict', 'Power', 'Similarity', 'Status',
    'Trust', 'Identity',
    'Sentiment Pos.', 'Sentiment Neg.',
    'Comment Left-Wing', 'Comment Right-Wing', 'Post Left-Wing',
    'Post Right-Wing',
    'Both Polarized', 'Both Polarized \\& Diff. Side', 'Diff. Side',
    'Shared Group',
])
homophily_results_table.to_csv(OUT_DIR / "table2_reproduced.csv")
print("\n=== Reproduced Table 2 ===", flush=True)
print(homophily_results_table.to_string(), flush=True)

#  cell 26
# applymap -> map: DataFrame.applymap was removed in pandas 3.
latex_table = (homophily_results_table
               .map(lambda x: x if x != '' else r'\dotfill')
               .to_latex(header=True, escape=False)
               )
latex_table = re.sub(r'(Adj. Pseudo.*\\\\)', r'\1 \\midrule', latex_table)
latex_table = re.sub(r'(Adj. Pseudo.*\\\\)', r'\1 \\midrule', latex_table)
latex_table = re.sub(r'(Intercept.*\\\\)', r'\1 \\addlinespace', latex_table)
latex_table = re.sub(r'(Identity.*\\\\)', r'\1 \\addlinespace', latex_table)
latex_table = re.sub(r'(Sentiment Neg.*\\\\)', r'\1 \\addlinespace', latex_table)
latex_table = re.sub(r'(Post Right-Wing.*\\\\)', r'\1 \\addlinespace', latex_table)
latex_table = re.sub(r'(Diff side.*\\\\)', r'\1 \\addlinespace', latex_table)
with open(OUT_DIR / "table2_reproduced.tex", 'w') as f:
    f.write(latex_table)

print(f"\n[saved] {OUT_DIR / 'table2_reproduced.csv'}")
print(f"[saved] {OUT_DIR / 'table2_reproduced.tex'}")
