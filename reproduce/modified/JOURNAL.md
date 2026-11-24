# Journal — modified reproduction of Figure 1 and Table 2

This folder contains the **final** versions of the two scripts.  They start from
the literal ports in `reproduce/from_original/` and apply the minimum set of
changes needed to approach the published results.  Every experiment that was run
to decide the changes is recorded below, including the ones that failed.

Headline:

| Target | Modified script | Paper |
| --- | --- | --- |
| Figure 1a | within 0.032 on all 10 values | 2.19 … 0.23 |
| Figure 1b | within 0.029 on all 10 values (one row per post) | 1.46 … 0.74 |
| Figure 1c | 9/14 cells within 0.05; 6 of the paper's cells significant at p<0.01; four "Support" cells **not reproducible** | 14 cells |
| Table 2 | **80/80 non-empty cells exact** | — |

## 1. Figure 1a — comments

Backbone code kept, one change: the 95% interval is drawn asymmetrically from
the two Eq. 9 log-odds bounds.  The notebook drew only `CI_upper - OR` (a
symmetric matplotlib error bar); the paper's bars are asymmetric.

| Dimension | Reproduced | Paper | Δ |
| --- | --- | --- | --- |
| Knowledge | 2.222 | 2.19 | +0.032 |
| Similarity | 1.810 | 1.80 | +0.010 |
| Trust | 1.657 | 1.65 | +0.007 |
| Power | 1.517 | 1.49 | +0.027 |
| Support | 1.475 | 1.47 | +0.005 |
| Identity | 1.383 | 1.38 | +0.003 |
| Conflict | 1.155 | 1.14 | +0.015 |
| Status | 0.978 | 0.98 | −0.002 |
| Fun | 0.938 | 0.93 | +0.008 |
| No dim | 0.226 | 0.23 | −0.004 |

The residual is consistent with the released dataset being a slightly smaller
snapshot (3,678,849 comments / 45,342 posts vs 3,690,687 / 46,046).  No other
change was needed.

## 2. Figure 1b — posts (unit-of-analysis fix)

The literal code kept one row per comment and, because `comment_id` in the
released data is the **parent id** (`t3_<post>` for direct replies,
`t1_<parent comment>` for nested ones; 650,101 direct replies share their post's
id), cell 20's `drop_duplicates(subset=['post_id','comment_id','has_delta'])`
also collapsed sibling comments.

Fix: build the panel from **one row per post** (`drop_duplicates('post_id')`), a
post being a delta post if any of its comments received a delta.  The no-dim
post score uses Eq. 2 with the length distribution of the no-dim posts.  No
`comment_id` is used.

| Dimension | Reproduced | Paper | Δ |
| --- | --- | --- | --- |
| Status | 1.451 | 1.46 | −0.009 |
| Support | 1.177 | 1.19 | −0.013 |
| Trust | 1.137 | 1.12 | +0.017 |
| Fun | 1.126 | 1.14 | −0.014 |
| Similarity | 1.044 | 1.03 | +0.014 |
| Identity | 1.037 | 1.03 | +0.007 |
| Conflict | 1.023 | 1.04 | −0.017 |
| Knowledge | 0.998 | 0.98 | +0.018 |
| Power | 0.994 | 0.97 | +0.024 |
| No dim | 0.711 | 0.74 | −0.029 |

(20,230 delta posts / 25,112 no-delta posts; the paper reports 20,239 / 25,807.)

The no-dim value is the only residual above 0.02.  Using the length distribution
of all posts instead of only the no-dim posts gives 0.724, closer to 0.74 but
contradicting Eq. 2 ("all messages labelled with d"), so the Eq. 2 version is
kept.

## 3. Figure 1c — the difficult panel

### 3.1 What the notebook does

Cell 26 computes `p(d_comment | d_post)` with length-discounted comment weights,
divides the odds by a shuffled null (cell 14, unseeded), and cell 27 plots
`OR_delta - OR_no_delta` for cells with a one-sided `norm.sf` p-value ≤ 0.01.
Cell 16 restricts the no-delta baseline to comments that received an answer by
the OP.  This literal pipeline matches only 3 of the paper's 14 cells (from the
previous phase).

### 3.2 Experiment log

Each row reports the number of the paper's 14 cells recovered (same sign and
|Δ| ≤ 0.05) by a candidate definition.  "W" means the conditional OR of delta vs
no-delta, each against a shuffled null; "weighted" means length-discounted
comment scores.

| # | Hypothesis / variant | Score |
| --- | --- | --- |
| 1 | Literal notebook (answered baseline, weighted, single shuffle) | 3/14 |
| 2 | **All non-delta comments, weighted comment, binary post, mean null** | **9/14** |
| 3 | As #2 but unweighted comment | 4/14 |
| 4 | As #2 but joint probabilities instead of conditional | 1/14 |
| 5 | As #2 but length-weighted post as well | 8/14 |
| 6 | As #2 but OR ratio `OR_delta/OR_no_delta - 1` instead of difference | 5/14 |
| 7 | Null = permute comment intents within each post | 0/14 |
| 8 | Within-post contrast: comment dim vs its complement, conditional on post | 0/14 |
| 9 | Interaction lift of delta probability `(p_ij p)/(p_i p_j) - 1` | 3–4/14 |
| 10 | Same, additive / log version | 5/14 / 3/14 |
| 11 | Logistic-regression interaction coefficients, answered sample | 2/14 |
| 12 | Logistic-regression interactions, 1M/2M all-comment sample | 5/14 / 5/14 |
| 13 | Interactions + sentiment + length controls | 4/14 |
| 14 | Baseline = comments on posts with no delta at all | 5/14 (6/14 without OP) |
| 15 | Moderation metric (does the post dim change the comment-dim effect?) | 7/14 |
| 16 | Two-way centering of the OR-difference matrix | 9/14 (same as #2) |
| 17 | Sample filters: no OP / top-level only / answered top-level | 8 / 1 / 2 (/14) |
| 18 | Seed search over 200 null draws of #2 | best 10/14 |

Why #2 is the chosen definition: it is the notebook's statistic with the one
filter the paper does not mention removed (`has_answer_by_op`; the notebook's own
`has_answer_filter=False` suggests it was not intended), and it is the only
variant that recovers the paper's five positive diagonal cells
(Support, Conflict, Power, Fun, Identity).  The null is averaged over 300
permutations to remove the Monte Carlo noise that the unseeded notebook draw
introduces (per-cell SD over 200 draws: 0.02–0.07).

The logistic-regression variants were run on the answered sample (542k rows,
120 parameters) and on 1M/2M random samples of all comments with
`sklearn.LogisticRegression(C=1e6, solver='lbfgs')`; interaction coefficients
and `exp(coef)-1` were both compared.  Neither the coefficients nor the
percentage-scale versions recover the support column.

### 3.3 Final Figure 1c

Definition shipped in `reproduce_figure1.py`:

* sample: all 3,678,849 comments;
* `p(d_comment | d_post)` with binary post dimensions and length-discounted
  comment weights (Eq. 6–7);
* null: 300 random permutations of the post associated with each comment; the
  conditional probabilities and joint counts are averaged;
* statistic: `OR_delta - OR_no_delta`, each OR vs the null;
* significance: notebook's Eq. 9 standard error and one-sided `p ≤ 0.01`.

| Post x Comment | Reproduced | p | Paper | Status |
| --- | --- | --- | --- | --- |
| Support x Support | 0.086 | 0.0103 | 0.10 | within 0.05, not significant |
| Support x No dim | −0.094 | 0.0153 | −0.10 | within 0.05, not significant |
| Conflict x Conflict | 0.105 | 0.0026 | 0.13 | significant |
| Conflict x Identity | −0.081 | 0.0180 | −0.12 | within 0.05, not significant |
| Power x Support | −0.006 | 0.441 | −0.12 | **not reproduced** |
| Power x Power | 0.194 | 6.5e−8 | 0.22 | significant |
| Power x Status | −0.128 | 0.0097 | −0.14 | significant |
| Power x Fun | −0.129 | 0.0094 | −0.13 | significant |
| Similarity x Support | −0.038 | 0.184 | −0.11 | **not reproduced** |
| Fun x Support | 0.047 | 0.121 | −0.09 | **not reproduced** |
| Fun x Fun | 0.111 | 0.0070 | 0.21 | significant, magnitude −0.10 |
| Status x Support | 0.045 | 0.104 | −0.07 | **not reproduced** |
| Identity x Identity | 0.200 | 1.2e−8 | 0.21 | significant |
| No dim x Knowledge | 0.049 | 0.0359 | 0.07 | within 0.05, not significant |

Result: 9/14 within 0.05, 6/14 significant.  The figure shows those 6 cells plus
one extra cell (`Similarity x Knowledge`, −0.097, p = 0.0027) that is not in the
paper; the paper's 14-cell set is therefore not reproduced exactly.

### 3.4 Why the four "Support" cells are abandoned

The paper reports negative values for Support comments on Power, Similarity, Fun
and Status posts (−0.12, −0.11, −0.09, −0.07).  In the released data, support
comments are **more** likely to receive a delta than non-support comments in
every one of those strata:

| Post dim | Support comments | No support | Support-only (no matching dim) |
| --- | --- | --- | --- |
| Power | 1.90% | 0.90% | 1.36% |
| Similarity | 1.97% | 0.94% | 1.46% |
| Fun | 2.08% | 0.87% | 1.76% |
| Status | 2.17% | 0.90% | 2.10% |

(All comments.)  The raw rates therefore show Support is more successful in every
one of these strata.  The 200-draw distribution of the chosen metric agrees:
Power x Support has mean −0.004 (range −0.06…+0.06, so it cannot reach −0.12),
Fun x Support +0.052 (−0.01…+0.12, cannot reach −0.09), Status x Support +0.046
(−0.02…+0.13, essentially cannot reach −0.07), and Similarity x Support −0.037
(−0.11…+0.03, reaching −0.11 only in the extreme tail).  Power, Fun and Status
Support cells are therefore structurally incompatible with the released data;
Similarity is reported as not reproduced (mean −0.037 vs −0.11).

A seed that matches 10/14 values exists (seed 84 of the 200; Fun x Fun 0.22,
Power x Power 0.25, Identity x Identity 0.19, …), but it still cannot produce
the support column and choosing it would be post-hoc.  The shipped script
therefore uses the permutation average, which is stable and reproducible.

Most plausible explanations for the support column: (i) the published figure
used a different data snapshot or a different support operationalisation;
(ii) it reports a regression-adjusted quantity not described in the Methods; or
(iii) the published cells are an error.  The released repository contains no
code that produces them, so they are abandoned and reported.

## 4. Table 2 — unchanged

`reproduce_table2.py` is the same code as in `from_original` (which reproduced
all 80 non-empty cells exactly: adjusted R² 0.00305 … 0.02265, all odds ratios
and BH stars).  No change was needed; the output CSV is byte-identical to the
`from_original` one.

## 5. Files and outputs

`reproduce/modified/`:

* `reproduce_figure1.py` — panels a/b/c as described;
* `reproduce_table2.py` — exact Table 2;
* `output/figure1_reproduced.png` — panels a, b, c;
* `output/figure1a_comments.csv`, `figure1b_posts.csv`;
* `output/figure1c_values.csv`, `figure1c_pvalues.csv`, `figure1c_shown.csv`,
  `figure1c_vs_paper.csv`;
* `output/table2_reproduced.csv`, `table2_reproduced.tex`.

Run from the repository root:

```bash
env/Scripts/python.exe reproduce/modified/reproduce_figure1.py
env/Scripts/python.exe reproduce/modified/reproduce_table2.py
```

## 6. Summary

* Figure 1a and Table 2 are fully reproduced (within the dataset-version gap).
* Figure 1b is reproduced after the post-level unit fix.
* Figure 1c is partially reproduced: the diagonal "reciprocity" cells and the
  negative Power x Status / Power x Fun cells are recovered (6 significant
  cells, 9 values within 0.05), but the paper's Support column is structurally
  incompatible with the released data and is abandoned with the evidence above.
  The figure shows 7 significant cells instead of 14 and should be read together
  with `figure1c_vs_paper.csv`.
