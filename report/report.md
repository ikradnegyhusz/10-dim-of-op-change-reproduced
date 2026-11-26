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
comment interaction matrix) is **largely** reproduced: 12 of its 14 values are
within 0.05 and 7 are significant at the paper's threshold; only Fun × Fun
(about half its published size) and Status × Knowledge fall outside. The report
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
  (power × status −14%, power × fun −13%, conflict × identity −12%, and knowledge
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

As a pragmatic starting point, we decided to part from the authors' own code and
data rather than reimplementing the method from the paper. Their repository
contains the two notebooks behind Figure 1 and Table 2, and the processed
dataset is published alongside it, so we could test the actual artifacts. Apart
from the paper, these were the only resources we used, so no step honestly
needed a GPU.

We then split the work into two phases.

1. **Literal port (`reproduce/from_original/`).** We first copied the notebook
   cells into two Python scripts, changing only what was needed to make them run
   (see `reproduce/from_original/JOURNAL.md`) to be able to see what the
   repository really produced, bugs included, as our "baseline".
2. **Modified reproduction (`reproduce/modified/`).** Starting from those
   scripts, we made the smallest methodological changes that brought the output
   closer to the paper, and checked each one against the paper before keeping
   it. We recorded every change and every experiment that failed in
   `reproduce/modified/JOURNAL.md`.

To be able to tell whether a difference came from the paper, the code, the data
or our own choices, we kept the two phases separate. This way, every mismatch
can be traced to its source.

### 3.2 Models and algorithms

The pipeline starts with two trained models. We did not retrain or re-run
either of them, because their outputs are already columns in the released
dataset.

* **Ten Dimensions classifier.** The social dimensions come from the model of
  Choi et al. (2020). Each dimension has its own binary classifier, an LSTM over
  300-dimensional GloVe embeddings learned from Common Crawl. The classifiers
  were trained on about 9,000 sentences labelled by crowd workers and reach an
  average AUC of 0.84. A message receives the highest score of any of its
  sentences.
* **Sociopolitical post classifier.** An L1-regularised logistic regression on
  {1,2,3}-grams that keeps the 10,000 most frequent features. The authors
  report an F1 of 89.5% on a held-out test set and 82% on a hand-labelled
  validation set of r/ChangeMyView posts. Its predictions decided which posts
  are in the released dataset.

What we ran ourselves is standard statistics on those columns.

* **Odds ratios for comments and posts (Figure 1a/1b).** For each dimension,
  the code compares the mean length-discounted score in the delta set (`p_d`)
  with the mean in the no-delta set (`p_nd`) through the odds ratio of Eq. 5,
  `OR = [p_d/(1−p_d)] / [p_nd/(1−p_nd)]`. The 95% confidence interval is built
  on the log-odds scale from the four cell counts (Eq. 9):
  `1.96·sqrt(1/n₁₁ + 1/n₁₂ + 1/n₂₁ + 1/n₂₂)`.
* **Post × comment interactions (Figure 1c).** For each post dimension `j` and
  comment dimension `i`, the code estimates how likely a comment is to express
  `i` when its post expresses `j`, using the length-discounted comment scores
  (Eq. 6–7). It turns these probabilities into odds and divides them by the
  odds under a null model in which every comment is attached to a random post
  (Eq. 8). Each cell of the matrix is this ratio for delta comments minus the
  same ratio for no-delta comments.
* **Logistic regressions (Table 2).** Ordinary `statsmodels` Logit models with
  3 to 15 coefficients each, fitted by maximum likelihood (Newton–Raphson)
  without regularisation. Fit is reported as McFadden's adjusted pseudo-R²,
  `1 − (llf − K)/llnull`, where `K` is the number of parameters.

### 3.3 Dataset

Everything runs on a single file, `changemyview-sociopol-processed.pickle.gz`.
Each of its 3,678,849 rows is one comment, with 88 columns covering both the
comment's own features and those of the post it belongs to. The comments come
from 45,342 posts, and 1.04% of them (38,223) received a delta. The columns we
relied on most are:

| Group | Examples | Meaning |
| --- | --- | --- |
| Identifiers / outcome | `post_id`, `comment_id`, `has_delta`, `is_op`, `has_answer_by_op` | `has_delta` = comment received a Δ; `is_op` = comment written by the poster |
| Text / length | `log_number_of_words`, `number_of_words_post` | length controls and the Eq. 2 discount |
| Dimensions | `support_comment`, …, `identity_post` | binary presence of each dimension |
| Weighted dimensions | `support_weighted_comment`, … | length-discounted score in [0, 2] (Eq. 2) |
| Standardised | `support_weighted_comment_z`, … | z-scored weighted scores for the regressions |
| Sentiment | `sentiment_pos_weighted_comment_z`, `sentiment_neg_…` | VADER positive/negative scores |
| Confounders | `post_leftwing`, `comment_rightwing`, `both_polarized`, `diff_side`, `shared_sub` | political-side variables of Table 3 |

The released data is a slightly different snapshot from the one in the paper,
with about 12,000 fewer comments and 700 fewer posts:

| Quantity | Paper | Released data |
| --- | --- | --- |
| Comments | 3,690,687 | 3,678,849 |
| Posts | 46,046 | 45,342 |
| Posts with ≥ 1 delta | 20,239 | 20,230 |
| Posts without delta | 25,807 | 25,112 |
| Comments with delta | 38,165 | 38,223 (38,217 after excluding OP comments) |
| Delta rate | ≈ 1.03% | 1.04% |

There is no train/dev/test split, as the analysis is purely observational. It
measures associations on the full table instead of training a model to predict
new data, so the only partition is into delta and no-delta comments or posts.
All preprocessing was done by the authors and is already in the released
columns: binarising each dimension score at its 85th percentile, length
discounting, z-scoring, sentiment and the political-side variables. The data can
be downloaded from the release page:
<https://github.com/corradomonti/10-dim-of-op-change/releases/tag/changemyview-sociopol-processed>.

One column is easy to misread. Despite its name, `comment_id` does not identify
a comment; it holds the id of the comment's *parent*. Direct replies to a post
carry `t3_<post_id>` (650,101 rows) and nested replies carry
`t1_<parent_comment_id>` (3,028,748 rows), so the column has only 2,331,456
distinct values for 3,678,849 rows. This matters for Figure 1b (see Results).

### 3.4 Hyperparameters

There is very little to tune. The settings that matter are the statistical
cut-offs, plus a couple of choices we made ourselves where the paper and the
code were silent or disagreed.

| Setting | Value | Where it comes from |
| --- | --- | --- |
| Dimension binarisation threshold | 85th percentile per dimension | Authors; already applied in the released columns |
| Length discount | `1/(1+z)` for `z ≥ 0`, `2 − 1/(1−z)` for `z < 0`, with `z` the standardised log-length | Paper Eq. 2 and authors' code (see the note below) |
| Dimension selection (Table 2) | keep dimensions with `p < 0.01` after Benjamini–Yekutieli correction | Authors' notebook |
| Significance stars (Table 2) | Benjamini–Hochberg, thresholds 0.05 / 0.01 / 0.001 | Authors' notebook |
| Regression fitting | MLE, Newton–Raphson, `maxiter=35` | `statsmodels` default; the notebook does not change it |
| Figure 1c null | 300 permutations, seed 2024 | **Our choice**; the notebook had one unseeded draw |
| Figure 1c significance | one-sided p ≤ 0.01 | Authors' notebook |
| Figure 1a/1b | no hyperparameters | — |

We did not search over hyperparameters, since every model is either fixed by
the paper or fitted by maximum likelihood. The closest thing to a search is our
exploration of alternative Figure 1c definitions, described in the Results.

**Note on Eq. 2.** The paper standardises message length in words, but for the
No dim score the authors' code standardises the logarithm of the word count
(`log_number_of_words`) instead. We follow the code. The other nine dimensions
come already discounted in the released data, so we cannot check which version
the authors used for them.

### 3.5 Experimental setup and code

All experiments are Python scripts that run on a CPU. We used Python 3.12.10
with pandas 3.1.0, NumPy 2.5.4, statsmodels 0.15.0, matplotlib 3.11.2 and SciPy
1.18.1, plus Jinja2, which pandas needs to write Table 2 as LaTeX. We also used
scikit-learn 1.9.1, but only for exploratory logistic regressions. Everything
ran on a Windows PC with an AMD Ryzen 5 5500U (6 cores, 12 threads), 15.4 GB of
RAM and no GPU.

The code is at <https://github.com/ikradnegyhusz/10-dim-of-op-change-reproduced>.
To rerun it, download the dataset into `data/` and run the two final scripts
from the repository root, with the environment active:

```bash
python reproduce/modified/reproduce_figure1.py
python reproduce/modified/reproduce_table2.py
```

The literal baselines run the same way:

```bash
python reproduce/from_original/reproduce_figure1.py
python reproduce/from_original/reproduce_table2.py
```

We judge the reproduction with the paper's own measures: odds ratios with 95%
confidence intervals, Benjamini–Hochberg-adjusted p-values and adjusted
pseudo-R². For every value we report the absolute difference from the paper and
count how many fall within 0.05. For Figure 1c we also check whether the sign
matches and whether the cell is significant at p < 0.01.

### 3.6 Computational requirements

The workload is light on computation but heavy on memory. The full table takes
about 2.7 GB in pandas, and the Figure 1c code also builds a 3.68M-row frame,
three 3.68M × 10 float32 matrices (about 0.4 GB) and a permuted copy.

| Step | Wall-clock on the Windows PC | Notes |
| --- | --- | --- |
| Load the processed dataset | ~14 s | 2.7 GB in memory |
| `reproduce_table2.py` | ~2.8 min | 9 logistic fits (base + 8 models), 3–15 coefficients each |
| `reproduce_figure1.py` (modified) | ~7.4 min | dominated by 300 permutations of the Figure 1c null |
| `reproduce_figure1.py` (literal) | a few minutes | 400 inner-loop cells on the answered subset |
| Figure 1c exploratory searches | ~1–3 min each | 40–200 permutations, logit fits on 0.5–2M rows |

A full run of the final scripts (figure plus table) takes about 0.2 CPU-hours,
with no GPU. To check our outputs and measure memory, we also reran all four
scripts on a second machine, an Apple M4 Mac with 16 GB of RAM and the same
package versions. The rerun took under four minutes in total. The output tables
for Figure 1a, Figure 1b and Table 2 came out byte-for-byte identical, and every
Figure 1c value agreed to within 0.0001, with the same significant cells. Peak
memory (the peak footprint reported by `/usr/bin/time -l`) was about 7–8 GB for
the two Table 2 scripts and the modified Figure 1 script, and about 11 GB for
the literal Figure 1 port, which holds several full-size frames at once. A
machine with 16 GB of RAM can therefore run everything, but 8 GB is likely too
tight.

---

## 4. Results

### 4.1 Overview

| Target | Claim | Literal repository code | Modified reproduction | Verdict |
| --- | --- | --- | --- | --- |
| Figure 1a | C1 | matches within 0.035 | matches within 0.032 | **reproduced** |
| Figure 1b | C2 | does not match; conflict, similarity, identity and no-dim fall on the wrong side of 1 | matches within 0.029 | **reproduced after unit fix** |
| Figure 1c | C3 | 4 of 14 cells | 12 of 14 values within 0.05, 7 significant | **largely reproduced** |
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

#### Result 3 — Figure 1c largely supports Claim 3

The modified matrix uses all non-delta comments as the baseline, binary post
dimensions, length-discounted comment weights, a null averaged over 300
permutations, and the paper's one-sided p ≤ 0.01 mask. It recovers the five
positive diagonal cells, the negative Power × Status and Power × Fun cells and
the negative Knowledge responses to Power, Similarity and Fun posts, but not
Status × Knowledge:

| Post × Comment | Reproduced | p-value | Paper | Status |
| --- | --- | --- | --- | --- |
| Support × Support | 0.086 | 0.0103 | 0.10 | within 0.05, not significant |
| Support × No dim | −0.094 | 0.0153 | −0.10 | within 0.05, not significant |
| Conflict × Conflict | 0.105 | 0.0026 | 0.13 | significant |
| Conflict × Identity | −0.081 | 0.0180 | −0.12 | within 0.05, not significant |
| Power × Knowledge | −0.081 | 0.0104 | −0.12 | within 0.05, not significant |
| Power × Power | 0.194 | 6.5e−8 | 0.22 | significant |
| Power × Status | −0.128 | 0.0097 | −0.14 | significant |
| Power × Fun | −0.129 | 0.0094 | −0.13 | significant |
| Similarity × Knowledge | −0.097 | 0.0027 | −0.11 | significant |
| Fun × Knowledge | −0.067 | 0.0297 | −0.09 | within 0.05, not significant |
| Fun × Fun | 0.111 | 0.0070 | 0.21 | significant, magnitude −0.10 |
| Status × Knowledge | −0.002 | 0.473 | −0.07 | **not reproduced** |
| Identity × Identity | 0.200 | 1.2e−8 | 0.21 | significant |
| No dim × Knowledge | 0.049 | 0.0359 | 0.07 | within 0.05, not significant |

Summary: **12 of 14 values are within 0.05**, 7 are significant at p < 0.01, and
every significant cell of the reproduced figure is also in the paper. Mean
absolute difference over all 14 cells is 0.028; excluding Fun × Fun and Status ×
Knowledge it is 0.019. **Claim 3 is supported for the "matching intent" diagonal,
for the Power × Status / Power × Fun mismatches and for the negative Knowledge
responses, but not for Status × Knowledge or the size of Fun × Fun.**

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
| 1 | Literal notebook (answered baseline, weighted, single shuffle) | 4/14 |
| 2 | **All non-delta comments, weighted comment, binary post, mean null** | **12/14** |
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

Scores #3–#18 were computed while we still read the four negative Knowledge
cells of the published figure as Support cells (see Result 3); only #1 and #2
have been rescored against the figure as published.

Variant #2 is the closest to the notebook's own statistic (the notebook's
statistic with the `has_answer_by_op` baseline filter removed), and it is the one
that recovers the five positive diagonal cells. It is the one shipped.

#### Additional Result 3 — the Figure 1c null is unstable

The notebook draws a single unseeded permutation. Over 200 draws of the chosen
metric, the per-cell standard deviation is 0.02–0.07, and the significant-cell
set changes from draw to draw. The best draw (seed 84) matched 10 of the 14 paper
values under the earlier reading of the figure (see Additional Result 2), but it
was selected post hoc; the shipped script therefore uses the permutation average,
which is stable and reproducible.

#### Additional Result 4 — environment and artifact inventory

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
* **Claim 3 (interactions): largely.** The "matching intent" diagonal (power,
  identity, fun, conflict, support), the negative Power × Status / Power × Fun
  cells and the negative Knowledge responses to Power, Similarity and Fun posts
  are reproduced in sign and roughly in magnitude (12 of 14 values within 0.05).
  Fun × Fun is significant but about half its published size, and Status ×
  Knowledge is not reproduced.
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
  difference of odds ratios).
* **Environment drift.** The repository targets Python 3.8 / pandas 1.2; the
  current environment is Python 3.12 / pandas 3.1. The API removals are easy to
  fix once identified, but a first-time user may interpret the errors as
  methodological failures.
* **Data semantics.** Nothing in the release documents that `comment_id` is the
  parent id. This is the kind of undocumented detail that changes results and is
  hard to discover without profiling the data.

### 5.4 Communication with the original authors

We did not contact the authors or open issues in the repository. The released
code and data were sufficient for Figure 1a, Figure 1b (after the fix), Table 2
and most of Figure 1c; for the two Figure 1c cells that are not reproduced, a
useful question would be: *"Which script and which data snapshot produced
Figure 1c, and how was the cell value defined quantitatively?"* We recommend that
future reproducers ask this question before spending time on alternative
definitions.

### 5.5 Strengths and weaknesses of this reproduction

**Strengths.** (i) We used the authors' code and data rather than reimplementing
from the paper, so the reproduction tests the actual artifacts. (ii) We separated
the literal port from the modified reproduction, which makes the effect of each
fix measurable. (iii) Table 2 is exact, and Figure 1a/1b are reproduced with
small, explainable residuals. (iv) Every experiment, including the 18 Figure 1c
variants, is documented in the journals and reproducible from the scripts.

**Weaknesses.** (i) Figure 1c is not fully reproduced; Fun × Fun and Status ×
Knowledge remain unexplained. (ii) The released dataset is a slightly different
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
5. **Align the Figure 1c columns before comparing.** The rotated column labels
   make it easy to read the Knowledge column as Support; the diagonal cells and
   the text (Power × Knowledge −12%) fix the alignment.
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

