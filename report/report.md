# Reproducibility Report

**The language of opinion change on social media under the lens of communicative action**

**Original paper:** Corrado Monti, Luca Maria Aiello, Gianmarco De Francisci Morales,
Francesco Bonchi. *The language of opinion change on social media under the lens of
communicative action.* Scientific Reports 12:17920 (2022).
<https://doi.org/10.1038/s41598-022-21720-4>

**Reproduction target:** Figure 1 (panels a, b, c) and Table 2.

**Artifacts used:** the paper transcription (`paper/paper.md`), the authors'
repository (`original_repo/`), the released processed dataset
(`data/changemyview-sociopol-processed.pickle.gz`), and the task description
(`task/task.txt`).

---

## Abstract

We reproduced the central quantitative results of Monti et al. (2022) from the
authors' released code and data. The paper studies which kinds of social intent in
Reddit comments make people change their minds in r/ChangeMyView, a forum where a
poster awards a *delta* (Δ) to a comment that changed their view. Using the
authors' two analysis notebooks, we reproduced **Figure 1a** (comment-level odds
ratios) to within 0.032 and **Table 2** (eight logistic regression models) exactly,
cell for cell. **Figure 1b** (post-level odds ratios) required a correction of the
unit of analysis: the notebook counts comments, while the figure counts posts.
With that correction all ten values are within 0.029. **Figure 1c** (the post ×
comment interaction matrix) is only **partially** reproduced: 9 of its 14 values
are within 0.05 and 6 are significant at the paper's threshold, but the four
negative "Support" cells could not be reproduced and were abandoned after a
systematic search over 18 variants of the statistic and its null. The report
documents what the original code can and cannot do, the exact fixes applied, the
experiments performed, and practical advice for future reproducers.

---

## 1. Introduction

Online political discussion is often studied through the lens of opinion dynamics,
but most models represent an interaction as a single number: agreement,
disagreement, positive or negative sentiment. Monti et al. (2022) argue that this
loses the *pragmatic intent* of a message — what the speaker is trying to do with
language. They operationalise Jürgen Habermas' theory of *communicative action*
using nine "social dimensions" of language (knowledge, power, status, trust,
support, similarity, identity, fun, conflict), extracted automatically from text
by the Ten Dimensions classifier of Choi et al. (2020). They then ask which
dimensions are associated with actual opinion change in Reddit's r/ChangeMyView,
where a comment that changes the poster's mind is explicitly marked with a delta.

This report is a reproducibility study of that paper. It is not a summary of the
theory and it does not re-argue the authors' interpretation; it asks a concrete
question: **can an independent reader, given the paper and the released
repository, reproduce the numbers shown in Figure 1 and Table 2?** We used the
authors' own code and data wherever possible, distinguished carefully between
"what the code actually does" and "what the paper reports", and recorded every
deviation. The report follows the ReScience reproducibility format: scope,
methodology, results, discussion.

---

## 2. Scope of reproducibility

### 2.1 Setting

r/ChangeMyView is a subreddit in which a poster states an opinion and commenters
try to change it. When a comment succeeds, the poster replies with a *delta*
(`Δ`), a public token of merit. The paper restricts the analysis to
*sociopolitical* posts (about political figures/institutions, cultural and social
issues, or national issues) and to nine social dimensions plus a "No dimension"
category. It uses 46,046 sociopolitical posts and 3,690,687 comments, of which
38,165 comments received a delta.

The analysis targets in scope are:

* **Figure 1a** — odds ratios (ORs) of a dimension appearing in comments with a
  delta versus comments without one.
* **Figure 1b** — ORs of a dimension appearing in posts whose author awarded at
  least one delta versus posts that never did.
* **Figure 1c** — a 10 × 10 matrix of post dimension × comment dimension
  interactions, showing how the combination of the two intents relates to the
  probability of a delta.
* **Table 2** — eight logistic regressions of "comment received a delta" on the
  social dimensions and on three sets of political-side confounders (individual
  sides, their interaction, shared groups) plus sentiment, with adjusted
  pseudo-R² and Benjamini–Hochberg significance stars.

### 2.2 Claims tested

We formulate the paper's contributions as falsifiable claims:

* **Claim 1 (comments).** Comments that convey at least one social dimension are
  more likely to receive a delta than comments that convey none; the strongest
  positive associations are knowledge, similarity and trust, and the
  no-dimension OR is about 0.23 (i.e. roughly 77% lower odds).
* **Claim 2 (posts).** Posts whose authors ended up awarding a delta differ in
  social intent from posts that did not: status is the most over-represented
  dimension (OR 1.46 in the paper) and power is the least likely (OR ≈ 0.97).
* **Claim 3 (interaction).** The combination of post intent and comment intent
  matters. Matching intents are associated with a higher probability of a delta
  (e.g. power × power +22%, identity × identity +21%, fun × fun +21%, conflict ×
  conflict +13%, support × support +10%), while some mismatches are negative
  (power × status −14%, power × fun −13%, conflict × identity −12%, and support
  responses to power/similarity/fun/status).
* **Claim 4 (robustness).** The association between social dimensions and delta
  survives controls for positive/negative sentiment and for the political sides
  of the two users, and models with social dimensions fit better than models
  without them; `fun` is not significant and is dropped.

### 2.3 What is in and out of scope

**In scope:** Figure 1a, 1b, 1c and Table 2, using the released processed dataset
and the authors' analysis notebooks.

**Out of scope:** retraining the Ten Dimensions LSTM classifier; retraining the
sociopolitical classifier; re-collecting raw Reddit data from Pushshift;
reproducing the supplementary figures/tables (SI1–SI8); reproducing the exact
visual style of the figures; the qualitative examples quoted in the paper.

**Provided artifacts and their state.** The repository contains two analysis
notebooks (`2_logistic-regression.ipynb`, `3_odds-ratios.ipynb`), the processed
dataset, the sociopolitical classifier vocabulary, and the labelled subreddit
list. It does **not** contain a runnable end-to-end Figure 1c pipeline: the matrix
pickles loaded by the notebook are absent, and the notebook never writes them.
The task description said "no code is available", but the repository does contain
the analysis code, so the study is a *code-and-data* reproduction rather than a
from-description reimplementation.

---

## 3. Methodology

### 3.1 Approach: two phases

We deliberately worked in two phases.

1. **Literal port (`reproduce/from_original/`).** We copied the notebook cells
   into two Python scripts with only compatibility fixes and dead-code removal.
   The goal was to measure what the repository actually produces, including its
   bugs. This phase is the baseline: it shows which paper results the published
   artifacts can and cannot support.
2. **Modified reproduction (`reproduce/modified/`).** Starting from the literal
   scripts, we applied the minimum methodological changes needed to approach the
   published results, and verified each change against the paper. Every change
   and every failed experiment is recorded in `reproduce/modified/JOURNAL.md`.

This separation matters for a reproducibility report: without it, one cannot tell
whether a mismatch comes from the paper, the code, the data, or the reproducer's
own choices.

### 3.2 Models and algorithms

* **Social-dimension extraction (external, not retrained).** The paper uses the
  Ten Dimensions model: an LSTM over 300-dimensional GloVe (Common Crawl)
  embeddings, with one independently trained binary classifier per dimension,
  trained on about 9,000 crowdsourced sentences; average AUC ≈ 0.84. The model
  returns the maximum sentence score per message. The released dataset already
  contains the binarised and length-discounted scores, so we did not retrain or
  re-run it.
* **Sociopolitical post classifier (external, not retrained).** An L1-regularised
  logistic regression on {1,2,3}-grams with the 10,000 most frequent features;
  reported F1 ≈ 89.5% on a held-out test set and ≈ 82% on a manually labelled
  r/ChangeMyView validation set. Its output defines the released dataset; we did
  not retrain it.
* **Comment and post odds ratios (Figure 1a/1b).** Non-parametric estimators from
  the paper's Eq. 5: `OR = [p_d/(1−p_d)] / [p_nd/(1−p_nd)]`, where `p_d` and
  `p_nd` are the mean length-discounted dimension scores in the delta and
  no-delta sets. The 95% confidence interval uses the count-based standard error
  of Eq. 9, `1.96·sqrt(1/n₁₁ + 1/n₁₂ + 1/n₂₁ + 1/n₂₂)` on the log-odds scale.
* **Post × comment interaction (Figure 1c).** For every post dimension `j` and
  comment dimension `i`, the conditional probability
  `p(d_i(comment) | d_j(post))` is computed with length-discounted comment
  weights (Eq. 6–7). Its odds are divided by the odds of a shuffled null model in
  which each comment is randomly assigned a post (Eq. 8), and the resulting
  quantity for delta comments is compared with the same quantity for no-delta
  comments. The difference is the value shown in the matrix.
* **Logistic regressions (Table 2).** `statsmodels` Logit, fitted by maximum
  likelihood (Newton–Raphson), with 3–15 coefficients per model and no
  regularisation. Adjusted McFadden pseudo-R² is
  `1 − (llf − K)/llnull`, where `K` is the number of parameters.

### 3.3 Dataset

The released file is a single pandas pickle,
`changemyview-sociopol-processed.pickle.gz` (149 MB compressed, about 2.7 GB in
memory), with **3,678,849 comment rows × 88 columns** and **45,342 unique
posts**. Each row is one comment, with its own text features and the features of
the post it belongs to. The delta rate is 1.04% (38,223 comments). Key columns:

| Group | Examples | Meaning |
| --- | --- | --- |
| Identifiers / outcome | `post_id`, `comment_id`, `has_delta`, `is_op`, `has_answer_by_op` | `has_delta` = comment received a Δ; `is_op` = comment written by the poster |
| Text / length | `log_number_of_words`, `number_of_words_post` | length controls and the Eq. 2 discount |
| Dimensions | `support_comment`, …, `identity_post` | binary presence of each dimension |
| Weighted dimensions | `support_weighted_comment`, … | length-discounted score in [0, 2] (Eq. 2) |
| Standardised | `support_weighted_comment_z`, … | z-scored weighted scores for the regressions |
| Sentiment | `sentiment_pos_weighted_comment_z`, `sentiment_neg_…` | VADER positive/negative scores |
| Confounders | `post_leftwing`, `comment_rightwing`, `both_polarized`, `diff_side`, `shared_sub` | political-side variables of Table 3 |

Comparison with the paper:

| Quantity | Paper | Released data |
| --- | --- | --- |
| Comments | 3,690,687 | 3,678,849 |
| Posts | 46,046 | 45,342 |
| Posts with ≥ 1 delta | 20,239 | 20,230 |
| Posts without delta | 25,807 | 25,112 |
| Comments with delta | 38,165 | 38,223 (38,217 after excluding OP comments) |
| Delta rate | ≈ 1.03% | 1.04% |

There is **no train/dev/test split** for the reproduction: the analysis is
observational and runs on the full released table. The only split-like
operations are the delta / no-delta partitions. All preprocessing (binarisation
at the 85th percentile of each dimension score, length discounting, z-scoring,
sentiment, political-side variables) was performed by the authors and is already
in the released columns; we did not redo it. The data can be downloaded from the
release page:
<https://github.com/corradomonti/10-dim-of-op-change/releases/tag/changemyview-sociopol-processed>.

**Important data quirk.** The column named `comment_id` is **not** a unique
comment identifier: it is the *parent* id. Direct replies to a post carry
`t3_<post_id>` (650,101 rows), and nested replies carry `t1_<parent_comment_id>`
(3,028,748 rows); there are only 2,331,456 distinct values for 3,678,849 rows.
This matters for Figure 1b (see Results).

### 3.4 Hyperparameters

The analysis has almost no free hyperparameters; the important settings are the
statistical cut-offs and the choices we had to make where the paper/code were
silent.

| Setting | Value | Where it comes from |
| --- | --- | --- |
| Dimension binarisation threshold | 85th percentile per dimension | Authors; already applied in the released columns |
| Length discount | `1/(1+z)` for `z ≥ 0`, `2 − 1/(1−z)` for `z < 0`, with `z` the standardised log-length | Paper Eq. 2 and authors' code |
| Dimension selection (Table 2) | keep dimensions with `p < 0.01` after Benjamini–Yekutieli correction | Authors' notebook |
| Significance stars (Table 2) | Benjamini–Hochberg, thresholds 0.05 / 0.01 / 0.001 | Authors' notebook |
| Regression fitting | MLE, Newton, `maxiter=100` | `statsmodels` defaults / notebook |
| Figure 1c null | 300 permutations, seed 2024 | **Our choice**; the notebook had one unseeded draw |
| Figure 1c significance | one-sided p ≤ 0.01 | Authors' notebook |
| Figure 1a/1b | no hyperparameters | — |

No hyperparameter search was performed: the models are either fixed by the
paper or fitted by maximum likelihood. The only "search" in this study is the
systematic exploration of Figure 1c definitions described in Results.

**Note on Eq. 2.** The transcribed equation shows `1 + z` for `z ≥ 0`, but the
released code uses `1/(1+z)`, which is also what the prose describes (the value
goes to 0 for long messages and to 2 for short ones). We follow the code; the
transcribed form would exceed 1 for long messages and is inconsistent with the
text.

### 3.5 Experimental setup and code

All experiments are CPU-only Python scripts. The environment used was:

* Python 3.12.10, pandas 3.1.0, NumPy 2.5.4, statsmodels 0.15.0,
  matplotlib 3.11.2, SciPy 1.18.1, scikit-learn 1.9.1 (only for exploratory
  logistic regressions);
* Windows; AMD Ryzen 5 5500U (6 cores / 12 threads), 15.4 GB RAM, no GPU.

The two final scripts are:

```bash
env/Scripts/python.exe reproduce/modified/reproduce_figure1.py
env/Scripts/python.exe reproduce/modified/reproduce_table2.py
```

The literal baselines are:

```bash
env/Scripts/python.exe reproduce/from_original/reproduce_figure1.py
env/Scripts/python.exe reproduce/from_original/reproduce_table2.py
```

The evaluation measures are the paper's own: odds ratios, 95% confidence
intervals, Benjamini–Hochberg-adjusted p-values, and adjusted pseudo-R². To
compare a reproduction with the paper we report the absolute difference per value
and count how many values are within 0.05 (and, for Figure 1c, whether the sign
matches and whether the cell is significant at p < 0.01).

### 3.6 Computational requirements

The workload is small by modern standards but memory-heavy: the full table is
about 2.7 GB in pandas, and the Figure 1c code materialises a 3.68M-row frame plus
three 3.68M × 10 float32 matrices (about 0.4 GB) and a permuted copy.

| Step | Measured wall-clock | Notes |
| --- | --- | --- |
| Load the processed dataset | ~14 s | 2.7 GB in memory |
| `reproduce_table2.py` | ~2.8 min | 9 logistic fits (base + 8 models), 3–15 coefficients each |
| `reproduce_figure1.py` (modified) | ~7.4 min | dominated by 300 permutations of the Figure 1c null |
| `reproduce_figure1.py` (literal) | a few minutes | 400 inner-loop cells on the answered subset |
| Figure 1c exploratory searches | ~1–3 min each | 40–200 permutations, logit fits on 0.5–2M rows |

Peak memory is estimated at roughly 5–7 GB for the literal port (it keeps the
full 2.7 GB table, a 1 GB comment frame, a 1 GB shuffled copy and a ~2 GB merged
frame) and 4–5 GB for the modified one. The total compute is about 0.2 CPU-hours
per full run (figure plus table), and there is no GPU use. A reader with 16 GB of
RAM can run everything comfortably; 8 GB is likely too tight for the full-table
frames, and the literal port needs the most memory.

### 3.7 Deviations from the original code

The literal ports required the following compatibility fixes (no change of
method):

1. pandas 2/3: `DataFrame.append` → `pandas.concat`, `DataFrame.applymap` →
   `DataFrame.map`, and positional `Series[0]` → `.iloc[0]` inside `apply`.
2. matplotlib ≥ 3.9: `matplotlib.cm.get_cmap` → `plt.get_cmap`.
3. A random seed (`np.random.seed(1234)` / `default_rng(2024)`) for the
   otherwise unseeded permutation null.
4. The Figure 1c matrix pickles loaded by the notebook are not shipped; the
   matrices are computed in memory instead.
5. Column-subset loads and `disp=0` in the fits (memory and output only).
6. Dead code that never reaches Figure 1/Table 2 was removed (mean t-intervals,
   ratio columns, extra matrix plots, the base-model table/figure, the homophily
   matrix, the supplementary length model, the classifier notebooks).

The **modified** scripts add three methodological changes, all documented in
`reproduce/modified/JOURNAL.md`:

* Figure 1b is computed on **one row per post** rather than one row per comment
  (and does not use `comment_id`).
* Figure 1c uses **all non-delta comments** as the baseline rather than only
  comments that received an answer by the OP, and averages the null over 300
  permutations.
* Figure 1a draws the confidence interval **asymmetrically** from the two
  log-odds bounds instead of the notebook's one-sided `yerr`.

---

## 4. Results

### 4.1 Overview

| Target | Claim | Literal repository code | Modified reproduction | Verdict |
| --- | --- | --- | --- | --- |
| Figure 1a | C1 | matches within 0.035 | matches within 0.032 | **reproduced** |
| Figure 1b | C2 | does not match; conflict, similarity, identity and no-dim fall on the wrong side of 1 | matches within 0.029 | **reproduced after unit fix** |
| Figure 1c | C3 | 3 of 14 cells | 9 of 14 values within 0.05, 6 significant | **partially reproduced** |
| Table 2 | C4 | 80/80 cells exact | 80/80 cells exact | **reproduced exactly** |

### 4.2 Results reproducing the original paper

#### Result 1 — Figure 1a supports Claim 1

The comment-level odds ratios are reproduced from the released data and the
authors' estimator:

| Dimension | Reproduced | Paper | Abs. diff |
| --- | --- | --- | --- |
| Knowledge | 2.222 | 2.19 | 0.032 |
| Similarity | 1.810 | 1.80 | 0.010 |
| Trust | 1.657 | 1.65 | 0.007 |
| Power | 1.517 | 1.49 | 0.027 |
| Support | 1.475 | 1.47 | 0.005 |
| Identity | 1.383 | 1.38 | 0.003 |
| Conflict | 1.155 | 1.14 | 0.015 |
| Status | 0.978 | 0.98 | 0.002 |
| Fun | 0.938 | 0.93 | 0.008 |
| No dim | 0.226 | 0.23 | 0.004 |

Mean absolute difference 0.011, maximum 0.032. The sign and ordering are
identical, the no-dim OR is 0.226 (paper 0.23, i.e. about 77% lower odds), and
the three strongest dimensions are knowledge, similarity and trust. The small
residual is consistent with the released data being a slightly smaller snapshot
(3.68M vs 3.69M comments; 45.3k vs 46.0k posts). **Claim 1 is supported.**

**Note on Eq. 3.** The paper defines `p(d) = sum(d)/(2·|M|)`, but the released
code uses the mean of the weighted scores directly, without the factor 2. The
published values only match the unnormalised version: applying the factor 2 gives
Knowledge 2.02 instead of 2.19. We follow the code; a reproducer who implements
Eq. 3 literally will not match Figure 1a.

#### Result 2 — Figure 1b supports Claim 2 only after fixing the unit of analysis

The literal notebook keeps one row per comment in the post-level panel: cell 20
de-duplicates on `comment_id` (which is the parent id), and cells 21–22 call
`drop_duplicates()` without `subset=['post_id']`. The literal output is:

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

This is a structural failure: Conflict, Similarity, Identity and No dim end up on
the wrong side of 1. Rebuilding the panel on **one row per post** (a post is a
delta post if any of its comments received a delta) gives:

| Dimension | Reproduced | Paper | Abs. diff |
| --- | --- | --- | --- |
| Status | 1.451 | 1.46 | 0.009 |
| Support | 1.177 | 1.19 | 0.013 |
| Trust | 1.137 | 1.12 | 0.017 |
| Fun | 1.126 | 1.14 | 0.014 |
| Similarity | 1.044 | 1.03 | 0.014 |
| Identity | 1.037 | 1.03 | 0.007 |
| Conflict | 1.023 | 1.04 | 0.017 |
| Knowledge | 0.998 | 0.98 | 0.018 |
| Power | 0.994 | 0.97 | 0.024 |
| No dim | 0.711 | 0.74 | 0.029 |

Mean absolute difference 0.016, maximum 0.029. Status is the most over-represented
dimension, power the least; the no-dim odds ratio is far below 1. **Claim 2 is
supported after the unit-of-analysis correction.** (The no-dim residual, 0.711 vs
0.74, is the largest; standardising the length on all posts instead of only
no-dim posts gives 0.724, closer but not exact.)

#### Result 3 — Figure 1c only partially supports Claim 3

The modified matrix uses all non-delta comments as the baseline, binary post
dimensions, length-discounted comment weights, a null averaged over 300
permutations, and the paper's one-sided p ≤ 0.01 mask. It recovers the five
positive diagonal cells and the two reproducible negative Power cells
(Power × Status and Power × Fun), but not the Support column:

| Post × Comment | Reproduced | p-value | Paper | Status |
| --- | --- | --- | --- | --- |
| Support × Support | 0.086 | 0.0103 | 0.10 | within 0.05, not significant |
| Support × No dim | −0.094 | 0.0153 | −0.10 | within 0.05, not significant |
| Conflict × Conflict | 0.105 | 0.0026 | 0.13 | significant |
| Conflict × Identity | −0.081 | 0.0180 | −0.12 | within 0.05, not significant |
| Power × Support | −0.006 | 0.441 | −0.12 | **not reproduced** |
| Power × Power | 0.194 | 6.5e−8 | 0.22 | significant |
| Power × Status | −0.128 | 0.0097 | −0.14 | significant |
| Power × Fun | −0.129 | 0.0094 | −0.13 | significant |
| Similarity × Support | −0.038 | 0.184 | −0.11 | **not reproduced** |
| Fun × Support | 0.047 | 0.121 | −0.09 | **not reproduced** |
| Fun × Fun | 0.111 | 0.0070 | 0.21 | significant, magnitude −0.10 |
| Status × Support | 0.045 | 0.104 | −0.07 | **not reproduced** |
| Identity × Identity | 0.200 | 1.2e−8 | 0.21 | significant |
| No dim × Knowledge | 0.049 | 0.0359 | 0.07 | within 0.05, not significant |

Summary: **9 of 14 values are within 0.05**, 6 are significant at p < 0.01, and
the figure shows a seventh significant cell that is not in the paper
(Similarity × Knowledge, −0.097, p = 0.0027). Mean absolute difference over all
14 cells is 0.049; excluding the four Support cells it is 0.027. **Claim 3 is
supported for the "matching intent" diagonal and for the Power × Status / Power ×
Fun mismatches, but not for the Support column.**

#### Result 4 — Table 2 supports Claim 4 exactly

All eight models were refitted from the released data with the authors' pipeline:
base model with the nine dimensions → keep the dimensions significant at p <
0.01 after Benjamini–Yekutieli (`fun` is dropped, as the paper says) → fit models
A–H with and without confounders → Benjamini–Hochberg stars and adjusted
pseudo-R². The reproduced table is **identical in all 80 non-empty cells**:

| Metric | A | B | C | D | E | F | G | H |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Adj. pseudo-R² | 0.00305 | 0.00446 | 0.00720 | 0.00935 | 0.01643 | 0.01780 | 0.02050 | 0.02265 |
| Knowledge | — | — | — | — | 1.217*** | 1.216*** | 1.216*** | 1.216*** |
| Similarity | — | — | — | — | 1.110*** | 1.108*** | 1.110*** | 1.112*** |
| Trust | — | — | — | — | 1.143*** | 1.143*** | 1.144*** | 1.144*** |
| Power | — | — | — | — | 1.097*** | 1.099*** | 1.097*** | 1.096*** |
| Support | — | — | — | — | 1.096*** | 1.095*** | 1.097*** | 1.098*** |
| Identity | — | — | — | — | 1.085*** | 1.086*** | 1.086*** | 1.086*** |
| Conflict | — | — | — | — | 1.024*** | 1.026*** | 1.025*** | 1.024*** |
| Status | — | — | — | — | 0.930*** | 0.929*** | 0.934*** | 0.937*** |
| Sentiment Pos. | 1.174*** | 1.173*** | 1.175*** | 1.177*** | 1.110*** | 1.110*** | 1.110*** | 1.112*** |
| Sentiment Neg. | 1.080*** | 1.082*** | 1.081*** | 1.081*** | 1.056*** | 1.058*** | 1.058*** | 1.057*** |

The full table, including the political-side and shared-group rows, is in
`reproduce/modified/output/table2_reproduced.csv`; it matches the paper cell for
cell, including the five-decimal pseudo-R² and every asterisk. The best
dimension-free model (D, shared group) has R² 0.00935, while the best model with
dimensions (H) has 0.02265, i.e. about 2.4× larger, as the paper argues. **Claim 4
is supported exactly.**

### 4.3 Results beyond the original paper

#### Additional Result 1 — `comment_id` is the parent id

The released `comment_id` column is the parent identifier, not a unique comment
identifier: 650,101 direct replies carry `t3_<post_id>`, the remaining 3,028,748
rows carry `t1_<parent_comment_id>`, and there are only 2,331,456 distinct values
for 3,678,849 rows. Any analysis that de-duplicates on `comment_id` (as the
notebook's Figure 1b code does) silently collapses sibling comments: the paired
frame drops from 542,669 to 339,434 rows, the delta rows from 38,223 to 31,434,
and the no-delta rows from 504,446 to 141,026. This is a data-documentation issue
that future users of the release should know about.

#### Additional Result 2 — Figure 1c definition search

We tested 18 alternative definitions of the Figure 1c statistic. The table below
reports how many of the paper's 14 cells each variant recovers (same sign and
|Δ| ≤ 0.05):

| # | Variant | Score |
| --- | --- | --- |
| 1 | Literal notebook (answered baseline, weighted, single shuffle) | 3/14 |
| 2 | **All non-delta comments, weighted comment, binary post, mean null** | **9/14** |
| 3 | As #2 but unweighted comment | 4/14 |
| 4 | As #2 but joint probabilities | 1/14 |
| 5 | As #2 but length-weighted post | 8/14 |
| 6 | As #2 but OR ratio instead of difference | 5/14 |
| 7 | Null = permute comment intents within each post | 0/14 |
| 8 | Within-post contrast (comment dim vs its complement) | 0/14 |
| 9 | Interaction lift `(p_ij p)/(p_i p_j) − 1` | 3–4/14 |
| 10 | Same, additive / log version | 5/14 / 3/14 |
| 11 | Logistic-regression interaction coefficients, answered sample | 2/14 |
| 12 | Logistic-regression interactions, 1M / 2M all-comment samples | 5/14 / 5/14 |
| 13 | Interactions + sentiment + length controls | 4/14 |
| 14 | Baseline = comments on posts with no delta at all | 5/14 (6/14 without OP) |
| 15 | Moderation metric (post dim changes the comment-dim effect) | 7/14 |
| 16 | Two-way centering of the OR-difference matrix | 9/14 |
| 17 | Sample filters: no OP / top-level / answered top-level | 8 / 1 / 2 (/14) |
| 18 | Seed search over 200 null draws of #2 | best 10/14 |

Variant #2 is the closest to the notebook's own statistic (the notebook's
statistic with the `has_answer_by_op` baseline filter removed), and it is the one
that recovers the five positive diagonal cells. It is the one shipped.

#### Additional Result 3 — the Figure 1c null is unstable

The notebook draws a single unseeded permutation. Over 200 draws of the chosen
metric, the per-cell standard deviation is 0.02–0.07, and the significant-cell
set changes from draw to draw. The best draw (seed 84) matches 10 of the 14 paper
values, but it was selected post hoc and still cannot produce the Support column;
the shipped script therefore uses the permutation average, which is stable and
reproducible.

#### Additional Result 4 — the Support column contradicts the released data

The paper reports negative values for Support comments on Power, Similarity, Fun
and Status posts. In the released data, Support comments are **more** likely to
receive a delta than non-Support comments in every one of those strata:

| Post dimension | Support comments | No support | Support only (no matching dim) |
| --- | --- | --- | --- |
| Power | 1.90% | 0.90% | 1.36% |
| Similarity | 1.97% | 0.94% | 1.46% |
| Fun | 2.08% | 0.87% | 1.76% |
| Status | 2.17% | 0.90% | 2.10% |

(All comments.) The 200-draw distribution confirms that the paper's values are at
or beyond the edge of the metric: Power × Support has mean −0.004 (range −0.06 to
+0.06, so it cannot reach −0.12), Fun × Support +0.052 (−0.01 to +0.12, so it
cannot reach −0.09), Status × Support +0.046 (−0.02 to +0.13, so it essentially
cannot reach −0.07), and Similarity × Support −0.037 (−0.11 to +0.03, reaching
−0.11 only in the extreme tail of 200 draws). The raw rates show Support is more
successful in every one of these strata, so the paper's negative Support effects
are not a simple consequence of the released data. Power, Fun and Status Support
cells were therefore abandoned; Similarity is reported as not reproduced (its
mean is −0.037, far from −0.11).

#### Additional Result 5 — environment and artifact inventory

Several parts of the released pipeline do not run unmodified on a current
environment:

* `requirements.txt` / `environment.yml` omit **statsmodels** and **SciPy**, both
  imported by the notebooks.
* pandas 2/3 removed `DataFrame.append` and `DataFrame.applymap`; matplotlib 3.9
  removed `matplotlib.cm.get_cmap`.
* The Figure 1c notebook's cell 27 loads `matrix_NEW_weighted-comment_discounted`
  pickles that are not in the repository, and cell 26 never writes them.
* The Figure 1c shuffle has no seed.
* The notebook hard-codes the `has_answer_by_op` baseline filter while a later
  variable (`has_answer_filter=False`) suggests no filter was intended.
* The shipped `notebook/output/table2_reproduced.csv` has its columns in loop
  order (A, B, D, C, E, F, H, G), not the paper's sorted order (A–H), so its C/D
  and G/H labels are swapped.

---

## 5. Discussion

### 5.1 Do the results support the paper's claims?

* **Claim 1 (comments): yes.** The entire Figure 1a ordering, the no-dim effect
  (OR 0.226) and the dominance of knowledge, similarity and trust are reproduced.
  The residual 0.03 is consistent with the data snapshot difference.
* **Claim 2 (posts): yes, but only after correcting the code.** The paper's
  qualitative story (status most over-represented, power least likely) is
  reproduced on one row per post. The literal notebook does not reproduce it, and
  the reason is a unit-of-analysis bug amplified by the parent-id semantics of
  `comment_id`. A reader who ran the notebook as-is would reject Claim 2.
* **Claim 3 (interactions): partially.** The "matching intent" diagonal (power,
  identity, fun, conflict, support) and the negative Power × Status / Power × Fun
  cells are reproduced in sign and roughly in magnitude. The Support column is
  not reproduced, and the published negative values are not obtainable from the
  released data. The paper's interpretation of the Support column should be
  treated with caution until the original computation or data version is
  clarified.
* **Claim 4 (robustness): yes, exactly.** Table 2 is reproduced cell for cell,
  including the pseudo-R² and the BH stars. The confounders do not overturn the
  dimensions, and `fun` is correctly dropped.

### 5.2 What was easy

* **Data access.** The processed dataset is a single file with all derived
  columns (binary dimensions, weighted scores, z-scores, sentiment, political
  sides). No feature engineering or classifier inference was needed for Figure 1
  or Table 2.
* **Table 2.** The regression pipeline is fully specified by the notebook and the
  paper; the model formulas, the dimension-selection rule, the BH correction and
  the adjusted pseudo-R² are unambiguous. The reproduction is exact on the first
  attempt once the pandas API fixes are applied.
* **Figure 1a.** The estimator is simple (means of the released weighted scores
  plus a count-based standard error), and the notebook code maps directly onto the
  paper's Eq. 5 and Eq. 9.

### 5.3 What was difficult

* **The unit-of-analysis bug in Figure 1b.** It is a one-line difference
  (`drop_duplicates()` vs `drop_duplicates(subset=['post_id'])`), but its effect
  is large and it is invisible unless one compares the row counts to the paper.
  The `comment_id` parent-id semantics make it worse: an apparently innocuous
  de-duplication on `comment_id` collapses siblings even before the post-level
  issue is considered.
* **Figure 1c.** Three problems compound: the pipeline cannot be replayed as
  shipped (missing matrix pickles), the null is unseeded and noisy, and the
  published quantity is ambiguous (the text quotes percentage changes, the
  caption says "variation of the probability", and the code computes a
  difference of odds ratios). The Support column then contradicts the data.
* **Environment drift.** The repository targets Python 3.8 / pandas 1.2; the
  current environment is Python 3.12 / pandas 3.1. The API removals are easy to
  fix once identified, but a first-time user may interpret the errors as
  methodological failures.
* **Data semantics.** Nothing in the release documents that `comment_id` is the
  parent id. This is the kind of undocumented detail that changes results and is
  hard to discover without profiling the data.

### 5.4 Communication with the original authors

We did not contact the authors or open issues in the repository. The released
code and data were sufficient for Figure 1a, Figure 1b (after the fix) and Table
2, and the remaining problem in Figure 1c is not a missing parameter but a
contradiction with the released data; a useful question would be: *"Which script
and which data snapshot produced the Support column of Figure 1c, and how was the
cell value defined quantitatively?"* Without that, the four cells cannot be
recovered from the public artifacts. We recommend that future reproducers ask
this question before spending time on alternative definitions.

### 5.5 Strengths and weaknesses of this reproduction

**Strengths.** (i) We used the authors' code and data rather than reimplementing
from the paper, so the reproduction tests the actual artifacts. (ii) We separated
the literal port from the modified reproduction, which makes the effect of each
fix measurable. (iii) Table 2 is exact, and Figure 1a/1b are reproduced with
small, explainable residuals. (iv) Every experiment, including the 18 Figure 1c
variants, is documented in the journals and reproducible from the scripts.

**Weaknesses.** (i) Figure 1c is not fully reproduced; the Support column is
abandoned rather than explained. (ii) The released dataset is a slightly different
snapshot from the paper (≈ 12k comments and ≈ 700 posts fewer), so Figure 1a/1b
cannot be expected to match to the last decimal. (iii) We did not retrain the
Ten Dimensions or sociopolitical classifiers, so the reproduction covers the
analysis, not the full NLP pipeline. (iv) We did not contact the authors.

### 5.6 Recommendations for future reproducers

1. **Check the unit of analysis before trusting a post-level result.** Compare
   the number of rows/posts in the notebook with the paper's counts. In this
   repository, `comment_id` is the parent id; de-duplicate on `post_id` for
   post-level panels.
2. **Seed every permutation null and average over several draws.** The Figure 1c
   null has per-cell SD 0.02–0.07; a single unseeded draw is not a stable
   estimate.
3. **Recompute matrices instead of relying on pickles.** The Figure 1c matrix
   files are not shipped; the notebook's cell 26 already computes them in memory.
4. **Use the released counts as a sanity check.** 3,678,849 comments, 45,342
   posts, 20,230 delta posts and 25,112 no-delta posts are the numbers the
   released file actually contains; differences from the paper are expected but
   small.
5. **Treat the Support column of Figure 1c as unresolved.** The released data
   show a positive Support effect in every post stratum; a reproduction that
   reports the paper's negative cells without qualification would be misleading.
6. **Pin the environment or patch the known API removals.** `statsmodels` and
   `scipy` must be installed; `append`/`applymap`/`get_cmap` need modern
   equivalents.

---

## 6. Reproduction artifacts

| Artifact | Path |
| --- | --- |
| Paper transcription | [`paper/paper.md`](../paper/paper.md) |
| Task description | [`task/task.txt`](../task/task.txt) |
| Authors' repository | [`original_repo/`](../original_repo/) |
| Released dataset | [`data/changemyview-sociopol-processed.pickle.gz`](../data/changemyview-sociopol-processed.pickle.gz) |
| Literal port (scripts + journal) | [`reproduce/from_original/`](../reproduce/from_original/) |
| Modified reproduction (scripts + journal) | [`reproduce/modified/`](../reproduce/modified/) |
| Final Figure 1 | [`reproduce/modified/output/figure1_reproduced.png`](../reproduce/modified/output/figure1_reproduced.png) |
| Final Figure 1c comparison | [`reproduce/modified/output/figure1c_vs_paper.csv`](../reproduce/modified/output/figure1c_vs_paper.csv) |
| Final Table 2 | [`reproduce/modified/output/table2_reproduced.csv`](../reproduce/modified/output/table2_reproduced.csv) |

The literal port's journal is
[`reproduce/from_original/JOURNAL.md`](../reproduce/from_original/JOURNAL.md);
the modified reproduction's journal, including the full experiment log, is
[`reproduce/modified/JOURNAL.md`](../reproduce/modified/JOURNAL.md).

---

## 7. References

1. Monti, C., Aiello, L. M., De Francisci Morales, G., & Bonchi, F. (2022). The
   language of opinion change on social media under the lens of communicative
   action. *Scientific Reports*, 12, 17920.
   <https://doi.org/10.1038/s41598-022-21720-4>
2. Choi, M., Aiello, L. M., Varga, K. Z., & Quercia, D. (2020). Ten social
   dimensions of conversations and relationships. *Proceedings of the Web
   Conference (WWW)*, 1514–1525.
3. Deri, S., Rappaz, J., Aiello, L. M., & Quercia, D. (2018). Coloring in the
   links: Capturing social ties as they are perceived. *Proceedings of the ACM on
   Human–Computer Interaction*, 2, 1–18.
4. Hutto, C. J., & Gilbert, E. (2014). VADER: A parsimonious rule-based model for
   sentiment analysis of social media text. *Proceedings of ICWSM*, 216–225.
5. Monti, C., Aiello, L. M., De Francisci Morales, G., & Bonchi, F. (2022).
   *10-dim-of-op-change* repository and processed dataset release.
   <https://github.com/corradomonti/10-dim-of-op-change>
6. The Ten Dimensions classifier implementation.
   <https://github.com/lajello/tendimensions>

