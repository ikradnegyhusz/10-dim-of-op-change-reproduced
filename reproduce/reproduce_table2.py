#!/usr/bin/env python3
"""
Replication Script for Table 2
Paper: The language of opinion change on social media under the lens of communicative action
Target: Table 2 (Logistic regression odds ratios across models A to H)
"""

import argparse
import gc
import gzip
import os
import pickle
import sys
import types
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.stats.multitest

# ==============================================================================
# Compatibility Shim: Pandas 1.x -> 2.x Unpickler
# ==============================================================================
dummy_numeric = types.ModuleType("pandas.core.indexes.numeric")
dummy_numeric.Int64Index = pd.Index
dummy_numeric.Float64Index = pd.Index
dummy_numeric.NumericIndex = pd.Index
sys.modules["pandas.core.indexes.numeric"] = dummy_numeric


class LegacyPandasUnpickler(pickle.Unpickler):
    """Fallback class remapper for legacy Pandas pickles."""
    def find_class(self, module, name):
        if module == "pandas.core.indexes.numeric":
            return pd.Index
        return super().find_class(module, name)


def load_data_safe(path: str) -> pd.DataFrame:
    """Stream decompression and load pickle with legacy pandas compatibility."""
    print(f"Opening {path}...")
    gc.collect()

    with gzip.open(path, "rb") as f:
        try:
            df = pickle.load(f)
        except (ModuleNotFoundError, AttributeError, KeyError):
            f.seek(0)
            df = LegacyPandasUnpickler(f).load()

    print(f"Dataset loaded. Total shape: {df.shape}.")

    # Normalize column names between underscore variants
    rename_map = {
        "comment_left_wing": "comment_leftwing",
        "comment_right_wing": "comment_rightwing",
        "post_left_wing": "post_leftwing",
        "post_right_wing": "post_rightwing",
        "diff_side": "diffside",
        "shared_group": "shared_sub",
    }
    for k, v in list(rename_map.items()):
        if k in df.columns and v not in df.columns:
            df[v] = df[k]
        elif v in df.columns and k not in df.columns:
            df[k] = df[v]

    df["has_delta"] = df["has_delta"].astype(np.int8)

    # Columns needed for Table 2
    required_cols = [
        "has_delta",
        "sentiment_pos_weighted_comment_z",
        "sentiment_neg_weighted_comment_z",
        "post_leftwing",
        "post_rightwing",
        "comment_leftwing",
        "comment_rightwing",
        "shared_sub",
        "both_polarized",
        "diffside",
        "support_weighted_comment_z",
        "knowledge_weighted_comment_z",
        "conflict_weighted_comment_z",
        "power_weighted_comment_z",
        "similarity_weighted_comment_z",
        "status_weighted_comment_z",
        "trust_weighted_comment_z",
        "identity_weighted_comment_z",
    ]

    existing_cols = [c for c in required_cols if c in df.columns]
    df = df[existing_cols].copy()
    gc.collect()

    # Downcast float columns to float32 to reduce memory footprint
    float_cols = df.select_dtypes(include=["float64"]).columns
    df[float_cols] = df[float_cols].astype(np.float32)

    print(f"Data ready: {len(df):,} rows x {len(df.columns)} columns.")
    return df


def nice_columns(c: str) -> str:
    """Format variable names to match Table 2 in the paper."""
    return (
        c.replace("_z", "")
        .replace("_weighted_comment", "")
        .replace("poster", "Post")
        .replace("republican", "right-wing")
        .replace("democrat", "left-wing")
        .replace("pos", "pos.")
        .replace("neg", "neg.")
        .replace("pos.t", "post")
        .replace("leftwing", "Left-Wing")
        .replace("rightwing", "Right-Wing")
        .title()
        .replace("Diff ", "Diff. ")
        .replace("Op And Commenter Opinionated", "Both polarized")
        .replace("Op And Commenter Diff Opinion", "Both polarized Diff. side")
        .replace("Both Polarized:Diffside", "Both Polarized Diff. Side")
        .replace("Both_Polarized:Diffside", "Both Polarized Diff. Side")
        .replace("Diffside", "Diff. Side")
        .replace("Shared Sub", "Shared Group")
        .replace("T.True", "")
    )


def pvalue2asterisk(p: float) -> str:
    """Format Benjamini-Hochberg corrected p-values into significance stars."""
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return ""


def add_results_to_table(res, df=None, method="fdr_bh"):
    """Format logistic regression results into a table row with adjusted R2 and odds ratios."""
    k = len(res.params)
    adj_pseudo_r2 = 1.0 - (res.llf - k) / res.llnull
    main_stat = pd.Series([f"{adj_pseudo_r2:.5f}"], index=["Adj. Pseudo-R2"])

    logits = np.exp(res.params)
    pvalues_uncorrected = res.pvalues

    _, pvalues_corrected, _, _ = statsmodels.stats.multitest.multipletests(
        pvalues_uncorrected, method=method
    )

    formatted_params = {}
    for var, or_val, p_corr in zip(res.params.index, logits, pvalues_corrected):
        formatted_params[var] = f"{or_val:.3f}{pvalue2asterisk(p_corr)}"

    row_series = pd.Series(formatted_params)
    new_row = pd.concat([main_stat, row_series])
    new_row_df = pd.DataFrame([new_row])

    if df is None:
        return new_row_df
    return pd.concat([df, new_row_df], ignore_index=True)


def run_table2_regressions(df: pd.DataFrame, dims: list) -> pd.DataFrame:
    """Fit models A through H as defined in Table 2 of the paper."""
    sentiments = "sentiment_pos_weighted_comment_z + sentiment_neg_weighted_comment_z"
    social_dims_formula = " + ".join([f"{c}_weighted_comment_z" for c in dims])

    # In patsy formula, interaction is defined with a colon `:`
    model_specs = [
        (sentiments, False),  # A
        (f"{sentiments} + post_leftwing + post_rightwing + comment_leftwing + comment_rightwing", False),  # B
        (f"{sentiments} + shared_sub", False),  # C
        (f"{sentiments} + both_polarized + diffside + both_polarized:diffside", False),  # D
        (sentiments, True),   # E
        (f"{sentiments} + post_leftwing + post_rightwing + comment_leftwing + comment_rightwing", True),   # F
        (f"{sentiments} + shared_sub", True),   # G
        (f"{sentiments} + both_polarized + diffside + both_polarized:diffside", True),   # H
    ]

    results_df = None
    for idx, (confounders, use_social_dim) in enumerate(model_specs):
        formula_parts = []
        if use_social_dim:
            formula_parts.append(social_dims_formula)
        if confounders:
            formula_parts.append(confounders)

        formula = "has_delta ~ " + " + ".join(formula_parts)
        print(f"Fitting Model {chr(ord('A') + idx)}...")
        model = smf.logit(formula=formula, data=df).fit(disp=False)
        results_df = add_results_to_table(model, results_df)

    results_table = results_df.copy()
    results_table.columns = [nice_columns(c) for c in results_table.columns]
    results_table = results_table.transpose()
    results_table.columns = [chr(ord("A") + i) for i in range(len(results_table.columns))]

    row_order = [
        "Adj. Pseudo-R2",
        "Intercept",
        "Support",
        "Knowledge",
        "Conflict",
        "Power",
        "Similarity",
        "Status",
        "Trust",
        "Identity",
        "Sentiment Pos.",
        "Sentiment Neg.",
        "Comment Left-Wing",
        "Comment Right-Wing",
        "Post Left-Wing",
        "Post Right-Wing",
        "Both Polarized",
        "Both Polarized Diff. Side",
        "Diff. Side",
        "Shared Group",
    ]

    available_rows = [r for r in row_order if r in results_table.index]
    results_table = results_table.reindex(available_rows)
    return results_table.fillna("")


def main():
    parser = argparse.ArgumentParser(description="Reproduce Table 2 Logistic Regressions")
    parser.add_argument(
        "--data_path",
        type=str,
        default="../data/changemyview-sociopol-processed.pickle.gz",
        help="Path to the processed pickle.gz dataset",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Directory to save the reproduced tables",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if not os.path.exists(args.data_path):
        sys.exit(f"Error: Data file not found at {args.data_path}")

    # Load dataset safely
    df = load_data_safe(args.data_path)

    # 8 Dimensions used (excluding 'Fun')
    dims = [
        "support",
        "knowledge",
        "conflict",
        "power",
        "similarity",
        "status",
        "trust",
        "identity",
    ]

    print("\nRunning Table 2 regressions (Models A to H)...")
    table2 = run_table2_regressions(df, dims)

    print("\n=================== REPRODUCED TABLE 2 ===================")
    print(table2.to_string())

    csv_path = os.path.join(args.output_dir, "table2_reproduced.csv")
    tex_path = os.path.join(args.output_dir, "table2_reproduced.tex")

    table2.to_csv(csv_path)
    with open(tex_path, "w") as f:
        f.write(table2.to_latex(escape=False))

    print(f"\nSaved CSV to: {csv_path}")
    print(f"Saved LaTeX to: {tex_path}")


if __name__ == "__main__":
    main()