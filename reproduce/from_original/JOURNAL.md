# Journal — Figure 1 and Table 2 from `original_repo` only

This step ports the original analysis code as literally as possible.  No
methodological fix was applied: the scripts reproduce what the notebooks do,
including their bugs, and the journal reports where that output departs from the
paper.  A corrected/modified version is the next step, not this one.

Sources: `paper/paper.md`, `task/task.txt`,
`original_repo/notebook/3_odds-ratios.ipynb` (Figure 1) and
`original_repo/notebook/2_logistic-regression.ipynb` (Table 2), with the
released dataset `changemyview-sociopol-processed.pickle.gz`.  Notebook `1.x`
(sociopolitical classifier) and the LSTM dimension model
(`lajello/tendimensions`) are not needed: the released table already contains
only sociopolitical posts and pre-computed dimension scores.

## Cell-by-cell mapping

| Script | Notebook cells | What is kept |
| --- | --- | --- |
| `reproduce_figure1.py`, panel (a) | 1, 5, 8, 9, 10 | comment frame, `nodimensions` + length discount, delta/no-delta split, OR + CI loop |
| `reproduce_figure1.py`, panel (b) | 13, 16, 18, 20, 21, 22, 23 | post frame, paired filter, delta/nodelta post frames, OR loop |
| `reproduce_figure1.py`, panel (c) | 14, 16, 18, 26, 27 | shuffled merge, paired filter, post/comment matrices, conditional OR vs. null, delta/no-delta difference |
| `reproduce_table2.py` | 2, 3, 4, 6, 7, 8, 14, 23, 24, 25, 26 | data load, `add_results_to_table`, base model for dimension selection, eight models A–H, sorting/labelling, LaTeX output |

## Minor fixes applied (no change of method)

1. **Seed.**  Cell 14 permutes `post_id` with `np.random.permutation` and no
   seed.  Without a seed the figure cannot be reproduced at all, so
   `np.random.seed(1234)` is set.  The effect of this choice is measured below.
2. **Missing matrix pickles.**  Cell 27 loads
   `matrix_NEW_weighted-comment_discounted`, which is not in the repository, and
   cell 26 never writes it.  The matrices computed in cell 26 are used in
   memory; the pickle round-trip is skipped.
3. **Matplotlib API.**  `matplotlib.cm.get_cmap` was removed in matplotlib 3.9;
   replaced by `plt.get_cmap`.
4. **pandas API (Table 2).**  `DataFrame.append` (removed in pandas 2) is
   replaced by `pandas.concat`; positional `Series[0]/Series[1]` inside `apply`
   is replaced by `.iloc[0]/.iloc[1]` (pandas 3 no longer allows it);
   `DataFrame.applymap` (removed in pandas 3) is replaced by `DataFrame.map`.
5. **Memory.**  `D.copy()` (88 columns, ~2.7 GB) and the full data load are
   replaced by the column subsets actually used downstream.  The selected rows
   and values are identical.
6. **`disp=0`** in the statsmodels fits (output only).
7. **`astype` instead of `.apply(bool)/.apply(int)`** in the panel (c) inner
   loop; identical arithmetic, without a per-element Python call.
8. **Dead code removed.**  Quantities that never reach Figure 1 were dropped:
   the t-interval of the means and the ratio/mean columns in cells 10 and 23;
   the delta and no-delta matrix plots of cell 27 (only the difference matrix is
   kept); the second base model, the base-model LaTeX table/figure and the
   homophily matrix of notebook 2 (cells 8–21); the supplementary length model
   of cells 28–32; the classifier notebooks.  Diagnostic prints are reduced.
9. **Plotting.**  One combined PNG is produced instead of separate figures.  The
   notebook's loops also compute the two sentiment scores for panels (a) and
   (b); they are kept in the CSV but not drawn, because the published figure
   only shows the nine dimensions plus "No dim".  Panels (a)/(b) use the
   notebook's `yerr = CI_upper - OR` (matplotlib draws it symmetrically).

## Environment and data

Python 3.12.10, pandas 3.1.0, numpy 2.5.4, statsmodels 0.15.0, matplotlib
3.11.2, scipy 1.18.1 (the repository pins Python 3.8 / pandas 1.2).

Dataset: 3,678,849 comments, 45,342 unique posts, 1.04% of comments with a
delta.  Paper: 3,690,687 comments and 46,046 posts (20,239 with a delta /
25,807 without; 38,165 delta comments).  The released dataset is a slightly
different snapshot, which explains second/third-decimal differences in Figure
1a–b but not the structural mismatches of Figure 1b–c.

## Verdict

| Target | Literal notebook code | Paper |
| --- | --- | --- |
| Figure 1a | reproduced within 0.035 | 2.19 / 1.80 / … / 0.23 |
| Figure 1b | **not reproduced**: status 1.394, conflict 0.866, power 0.869, no-dim 1.066 | 1.46 / 1.04 / 0.97 / 0.74 |
| Figure 1c | **not reproduced**: 8 significant cells, 3 overlap the paper; unseeded null model | 14 cells, mostly positive diagonal |
| Table 2 | **exact**: 80/80 non-empty cells, including R² and stars | — |

**Correction.**  The paper's negative cells in the Power, Similarity, Fun and
Status rows are in the Knowledge column, not Support.  With that reading the
literal code overlaps 4 of the paper's 14 cells, including Power x Knowledge
(-0.120 vs -0.12); the Figure 1c section below keeps the earlier reading.

## Figure 1a — comments

Cell 5 frame, `nodimensions` from the nine binary columns, length discount on
the no-dimension comments, then the cell 10 loop.  Comments by the OP are
excluded; the notebook also computes `sentiment_pos`/`sentiment_neg` and would
plot them (they are in the CSV, not in the figure).

| Dimension | Reproduced | Paper |
| --- | --- | --- |
| Knowledge | 2.222 | 2.19 |
| Similarity | 1.810 | 1.80 |
| Trust | 1.657 | 1.65 |
| Power | 1.517 | 1.49 |
| Support | 1.475 | 1.47 |
| Identity | 1.383 | 1.38 |
| Conflict | 1.155 | 1.14 |
| Status | 0.978 | 0.98 |
| Fun | 0.938 | 0.93 |
| No dim | 0.226 | 0.23 |

The notebook also computes Sentiment Pos. = 1.675 and Sentiment Neg. = 1.151.
Two code/paper inconsistencies remain visible: (i) the code does **not** apply
the factor 2 of Eq. 3, it uses `p = mean(score)`; using `mean/2` gives Knowledge
2.02 instead of 2.19, so the code and figure agree with the unnormalised
version.  (ii) The code draws `yerr = CI_upper - OR` only (symmetrised by
matplotlib), although Eq. 9 defines a two-sided log-odds interval.

## Figure 1b — posts (literal code does not match the paper)

Two defects compound here.

*Cell 20 assumes `comment_id` is unique per comment.*  In the released data it
is the **parent id**: 650,101 direct replies carry `t3_<post_id>` and the rest
carry `t1_<parent_comment_id>`; there are only 2,331,456 distinct values for
3,678,849 comment rows.  The line
`drop_duplicates(subset=['post_id','comment_id','has_delta'])` therefore collapses
sibling comments: the paired frame goes from 542,669 rows to 339,434, the delta
rows from 38,223 to 31,434 and the no-delta rows from 504,446 to 141,026.

*Cells 21–22 then keep those rows as if they were posts.*  `drop_duplicates()` is
called without `subset=['post_id']`, so a post still contributes once per
surviving parent/comment row and the no-delta set is built at that level.  The
literal code produces:

| Dimension | Literal code | Paper |
| --- | --- | --- |
| Status | 1.394 | 1.46 |
| Support | 1.230 | 1.19 |
| Fun | 1.138 | 1.14 |
| Trust | 1.138 | 1.12 |
| No dim | 1.066 | 0.74 |
| Similarity | 0.988 | 1.03 |
| Knowledge | 0.919 | 0.98 |
| Identity | 0.904 | 1.03 |
| Power | 0.869 | 0.97 |
| Conflict | 0.866 | 1.04 |

(31,434 delta rows over 20,230 posts; 141,026 no-delta rows over 18,834 posts.)
The literal code puts Conflict, Similarity, Identity and No dim on the wrong
side of 1 relative to the paper (and moves Power further below 1).  The causes
are the parent-id dedup and the missing post-level unit; both are reported here
and left for the modified script.  The notebook also computes two sentiment
entries (0.994 / 0.995), which the figure does not show.

## Figure 1c — post x comment matrix (literal code does not match the paper)

Cell 26 computes `p(d_comment | d_post)` with the length-discounted comment
weights, divides its odds by the odds of a shuffled pair (cell 14: permute
`post_id` across comments, merge with the post table), and cell 27 takes
`OR_delta - OR_no_delta`, masks cells with a one-sided `norm.sf` p-value above
0.01, and plots the result.

With the seed fixed at 1234 the literal code gives **8** non-zero cells:

| Post x Comment | Literal code | Paper |
| --- | --- | --- |
| Knowledge x Knowledge | -0.108 | — |
| Conflict x Conflict | -0.122 | +0.13 |
| Conflict x Identity | -0.110 | -0.12 |
| Power x Knowledge | -0.120 | — (text says -12%) |
| Power x Fun | -0.142 | -0.13 |
| Power x Status | -0.160 | -0.14 |
| Similarity x Similarity | -0.119 | — |
| Fun x Support | +0.107 | -0.09 |

Only three cells overlap the paper's 14 (Conflict x Identity, Power x Fun,
Power x Status), and the five positive diagonal cells of the paper
(Support, Conflict, Power, Fun, Identity) are absent or negative here.

This is not only the missing seed.  Repeating the null-model draw 40 times
(vectorised equivalent of the same cells) gives cell-level standard deviations
of 0.02–0.07, and four published cells are **never** within 0.05 of the paper:
Fun x Fun (paper +0.21, mean -0.02), Conflict x Conflict (+0.13, mean -0.09),
Power x Support (-0.12, mean +0.01), Fun x Support (-0.09, mean +0.06).  Three
more match in only 1 of the 40 draws (Support x Support, Status x Support,
Identity x Identity).  Conversely some cells match often (Power x Status 40/40
draws, Power x Fun 37/40, No dim x Knowledge 37/40, Support x No dim 31/40,
Conflict x Identity 24/40).

Additional reasons the panel cannot be recovered from the repository as shipped:

* the matrix pickles read by cell 27 are not shipped (fix 2);
* the null model is unseeded (fix 1) and its noise is large;
* cell 16 hard-codes the no-delta baseline to comments with
  `has_answer_by_op`, while the later `has_answer_filter = False` suggests the
  opposite;
* `nodimensions` comments get a binary weight while the nine dimensions are
  length-discounted (cell 18 only corrects the post side);
* the significance test is one-sided while the caption says p < 0.01; the
  notebook plots the full 10x10 grid (zeros included), the paper only non-zero
  cells;
* the quantity is ambiguous: the paper text quotes percentage changes (+22%,
  -14%, -12%, -13%), the caption says "variation of the probability", and the
  code computes `OR_delta - OR_no_delta`;
* the paper text mentions a Power x Knowledge effect (-12%) that is absent from
  the published matrix.

## Table 2 — exact reproduction

Cells 2–4, 6–8, 14, 23–26.  The base model with the nine dimensions is
fitted, dimensions significant at 0.01 with Benjamini-Yekutieli are kept
(`fun` is dropped, p ~ 0.79, as the paper says), then the eight models A–H are
fitted in the notebook's loop order, sorted by `Adj. Pseudo-$R^2$`, labelled
A–H and reindexed to the paper's rows.

**All 80 non-empty cells of Table 2 are identical to the paper**, including
`Adj. Pseudo-R² = 0.00305 / 0.00446 / 0.00720 / 0.00935 / 0.01643 / 0.01780 /
0.02050 / 0.02265`, the intercepts (0.010*** / 0.011***) and every odds ratio
with its BH stars.

Repository issues met here:

* `requirements.txt`/`environment.yml` omit **statsmodels** and **scipy**, both
  imported by the notebooks;
* `DataFrame.append`/`applymap` and positional `Series[...]` are removed in
  pandas 2/3 (fixes 4);
* the shipped `notebook/output/table2_reproduced.csv` has its columns in the
  loop order (A, B, **D, C**, E, F, **H, G**), not the sorted/paper order
  (A, B, C, D, E, F, G, H): its C/D and G/H labels are swapped;
* `has_delta` must be cast to int (cell 3), otherwise patsy treats it as a
  categorical outcome;
* the regression uses all rows of the processed dataset, including OP comments
  and comments that panel (a) filters out; this is the notebook's behaviour and
  it is what reproduces the table.

## Condensed issue list

1. **Fig. 1b unit bug**: no `post_id` de-duplication (cells 21–22), and cell 20
   collapses siblings because `comment_id` is the parent id.
2. **Fig. 1c missing artefacts**: matrix pickles absent; cell 27 cannot run as
   shipped; `matplotlib.cm.get_cmap` removed.
3. **Fig. 1c non-determinism**: unseeded permutation; cell SD 0.02–0.07 and
   different significant-cell sets across seeds.
4. **Fig. 1c filter inconsistency**: cell 16 vs `has_answer_filter=False`.
5. **Fig. 1c weighting inconsistency**: binary no-dimension comment vs
   length-discounted dimensions.
6. **Fig. 1c metric ambiguity**: percentages in the text, OR differences in the
   code; six published cells are outside the distribution of the code.
7. **Table 2 output labels**: shipped CSV C/D and G/H swapped.
8. **Environment**: missing statsmodels/scipy; pandas 2/3 and matplotlib 3.9
   API removals.
9. **Dataset version drift**: 45,342 posts / 3.68M comments vs 46,046 / 3.69M.
10. **Naming drift**: model `respect` = data/paper `status`; `romance` present
    in the data but excluded; dead code paths (e.g. `apply_weighting='both'`
    references columns never created).
11. **Data column semantics**: `comment_id` is the parent id (`t1_<comment>` /
    `t3_<post>`), not a unique comment id; cell 20's de-duplication therefore
    collapses sibling comments and explains the 38,223 → 31,434 delta rows and
    504,446 → 141,026 no-delta rows.

## What the original repository can do

* Figure 1a as published (up to the dataset-version gap, max |Δ| ≈ 0.035).
* Table 2 exactly, including stars and pseudo-R².
* Figure 1b and Figure 1c **only in the notebook's own form**, which does not
  match the paper: the post panel is comment-weighted, the matrix is the
  significant-cell set of a noisy unseeded null model.

## What it cannot do

* Reproduce the published Figure 1b (needs a post-level rebuild; `comment_id`
  cannot be used as a unique comment key).
* Reproduce the published Figure 1c values, cell selection or metric; the
  missing pickles and unseeded null model make the shipped pipeline
  non-replayable as well.
* Match Figure 1a–b to the last decimal (different dataset snapshot).

## How to run

From the repository root:

```bash
env/Scripts/python.exe reproduce/from_original/reproduce_figure1.py
env/Scripts/python.exe reproduce/from_original/reproduce_table2.py
```

Outputs in `reproduce/from_original/output/`:

* `figure1_reproduced.png` (panels a, b, c)
* `figure1a_comments.csv` (12 rows: nine dimensions + two sentiment scores +
  no-dim), `figure1b_posts.csv` (same structure for posts)
* `figure1c_values.csv` (the cell-27 masked difference matrix),
  `figure1c_pvalues.csv`, `figure1c_shown.csv`
* `table2_reproduced.csv`, `table2_reproduced.tex`
