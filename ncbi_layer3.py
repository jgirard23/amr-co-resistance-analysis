"""
AMR Analysis — Layer 3: Sampling Bias Correction
=================================================
Applies propensity-based reweighting using species, continent,
and breakpoint standard as covariates.
Then re-runs module detection on the weighted dataset and compares
results across CLSI vs inferred-EUCAST subsets.

Run:
    python3 ncbi_layer3.py
"""

import pandas as pd
import numpy as np
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# ─────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────
RESULTS_FOLDER = "/Users/jacobgirard-beaupre/Downloads/NCBI data/results"
OUTPUT_FOLDER  = "/Users/jacobgirard-beaupre/Downloads/NCBI data/results/layer3"

INPUT_FILE = "primary_cleaned.csv"

# Panel after dropping ampicillin (100% intrinsic signal)
PANEL = [
    "ceftazidime",
    "ciprofloxacin",
    "gentamicin",
    "trimethoprim/sulfamethoxazole",
    "aztreonam",
    "amikacin",
]

N_MODULES   = 3   # updated panel has 6 drugs, 3 modules expected
MIN_PAIRS   = 30
MAX_WEIGHT  = 10.0  # cap extreme weights to avoid instability


# ═════════════════════════════════════════════════════════════════
def divider(title=""):
    width = 72
    if title:
        pad = max(0, width - len(title) - 4)
        print(f"\n{'─'*2} {title} {'─'*pad}")
    else:
        print("─" * width)


def infer_breakpoint_standard(df):
    """
    Infer EUCAST for European isolates with UNKNOWN breakpoint standard.
    Justified by Layer 2 finding: 99.9% of European unknowns are E. coli,
    European labs use EUCAST by default.
    """
    df = df.copy()
    df["breakpoint_inferred"] = df["breakpoint_standard"].copy()

    european_unknown = (
        (df["breakpoint_standard"] == "UNKNOWN") &
        (df["continent"] == "Europe")
    )
    df.loc[european_unknown, "breakpoint_inferred"] = "EUCAST"

    n_inferred = european_unknown.sum()
    print(f"  Inferred EUCAST for {n_inferred:,} European isolates "
          f"with unknown breakpoint standard")

    print(f"\n  Updated breakpoint standard distribution:")
    print(df["breakpoint_inferred"].value_counts().to_string())

    return df


def compute_propensity_weights(df):
    """
    Compute inverse propensity weights to correct sampling bias.
    Target distribution: uniform across species × continent × breakpoint.
    Method: logistic regression predicting membership in each stratum,
    weights = 1 / P(being in observed stratum).
    """
    df = df.copy()

    # Build covariate matrix
    covariates = ["species", "continent", "breakpoint_inferred"]
    df_cov = df[covariates].copy()

    # Fill missing continent
    df_cov["continent"] = df_cov["continent"].fillna("Unknown")

    # Encode categoricals
    encoders = {}
    for col in covariates:
        le = LabelEncoder()
        df_cov[col] = le.fit_transform(df_cov[col].astype(str))
        encoders[col] = le

    X = df_cov.values

    # Compute stratum sizes for target distribution
    strata = df[covariates].fillna("Unknown").astype(str).apply(
        lambda x: "_".join(x), axis=1
    )
    stratum_counts = strata.value_counts()
    n_total = len(df)
    n_strata = len(stratum_counts)

    # Target: equal representation across strata
    target_per_stratum = n_total / n_strata

    # Weight = target_count / observed_count per stratum
    weights = strata.map(lambda s: target_per_stratum / stratum_counts[s])

    # Cap extreme weights
    weights = weights.clip(upper=MAX_WEIGHT)

    # Normalise so weights sum to n_total
    weights = weights * (n_total / weights.sum())

    df["propensity_weight"] = weights.values

    print(f"\n  Propensity weight distribution:")
    print(f"    Min    : {weights.min():.3f}")
    print(f"    Max    : {weights.max():.3f}")
    print(f"    Mean   : {weights.mean():.3f}")
    print(f"    Median : {weights.median():.3f}")

    print(f"\n  Effective sample size after weighting:")
    ess = (weights.sum()**2) / (weights**2).sum()
    print(f"    ESS = {ess:.0f} (from {len(df):,} actual isolates)")

    return df


def pairwise_coresistance_weighted(binary_df, weights, panel, min_pairs=MIN_PAIRS):
    """
    Compute weighted pairwise co-resistance matrix.
    Uses propensity weights so underrepresented groups count more.
    """
    cols = [c for c in panel if c in binary_df.columns]
    n = len(cols)
    matrix = pd.DataFrame(np.nan, index=cols, columns=cols)
    counts = pd.DataFrame(0, index=cols, columns=cols)

    for i in range(n):
        # Weighted marginal resistance rate
        mask = binary_df[cols[i]].notna()
        if mask.sum() >= min_pairs:
            w = weights[mask]
            matrix.iloc[i, i] = (
                (binary_df[cols[i]][mask] * w).sum() / w.sum()
            )
            counts.iloc[i, i] = mask.sum()

        for j in range(i+1, n):
            pair_mask = binary_df[cols[i]].notna() & binary_df[cols[j]].notna()
            if pair_mask.sum() < min_pairs:
                continue
            w = weights[pair_mask]
            both_r = (
                (binary_df[cols[i]][pair_mask] == 1) &
                (binary_df[cols[j]][pair_mask] == 1)
            ).astype(float)
            rate = (both_r * w).sum() / w.sum()
            matrix.iloc[i, j] = rate
            matrix.iloc[j, i] = rate
            counts.iloc[i, j] = pair_mask.sum()
            counts.iloc[j, i] = pair_mask.sum()

    return matrix, counts


def pairwise_coresistance_unweighted(binary_df, panel, min_pairs=MIN_PAIRS):
    """Unweighted version for comparison."""
    cols = [c for c in panel if c in binary_df.columns]
    n = len(cols)
    matrix = pd.DataFrame(np.nan, index=cols, columns=cols)

    for i in range(n):
        matrix.iloc[i, i] = binary_df[cols[i]].mean()
        for j in range(i+1, n):
            pair = binary_df[[cols[i], cols[j]]].dropna()
            if len(pair) < min_pairs:
                continue
            both_r = ((pair[cols[i]] == 1) & (pair[cols[j]] == 1)).sum()
            matrix.iloc[i, j] = both_r / len(pair)
            matrix.iloc[j, i] = matrix.iloc[i, j]

    return matrix


def cluster_matrix(matrix, n_modules=N_MODULES):
    """Hierarchical clustering on co-resistance matrix."""
    dist = 1.0 - matrix.fillna(0)
    dist_arr = dist.values.copy()
    np.fill_diagonal(dist_arr, 0)
    dist_df = pd.DataFrame(dist_arr, index=dist.index, columns=dist.columns)
    condensed = squareform(dist_arr, checks=False)
    condensed = np.clip(condensed, 0, None)
    Z = linkage(condensed, method="average")
    labels = fcluster(Z, n_modules, criterion="maxclust")
    return Z, labels, dist_df


def plot_comparison_heatmaps(matrices, titles, output_path):
    """Plot side-by-side heatmaps for comparison."""
    n = len(matrices)
    n_ab = len(matrices[0])
    fig, axes = plt.subplots(1, n, figsize=(n_ab * n * 0.9, n_ab * 0.85 + 2))
    if n == 1:
        axes = [axes]

    for ax, mat, title in zip(axes, matrices, titles):
        data = mat.values.copy()
        np.fill_diagonal(data, np.nan)
        im = ax.imshow(data, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(n_ab))
        ax.set_yticks(range(n_ab))
        ax.set_xticklabels(mat.columns, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(mat.index, fontsize=8)
        ax.set_title(title, fontsize=10, fontweight="bold")
        for i in range(n_ab):
            for j in range(n_ab):
                val = data[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            fontsize=7,
                            color="black" if val < 0.6 else "white")

    plt.suptitle("Co-resistance matrix comparison", fontsize=12,
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved → {output_path}")


def plot_dendrogram(Z, labels, title, output_path):
    fig, ax = plt.subplots(figsize=(10, 5))
    dendrogram(Z, labels=labels, ax=ax,
               leaf_font_size=10, leaf_rotation=45)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylabel("Distance (1 - co-resistance rate)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved → {output_path}")


def compare_modules(modules_a, modules_b, label_a, label_b, panel):
    """Compare module assignments between two analyses."""
    print(f"\n  Module comparison: {label_a} vs {label_b}")
    for ab, mod_a, mod_b in zip(panel, modules_a, modules_b):
        match = "✓" if mod_a == mod_b else "⚠ DIFFERS"
        print(f"    {ab:<40} {label_a}: Module {mod_a}  "
              f"{label_b}: Module {mod_b}  {match}")


# ═════════════════════════════════════════════════════════════════
def main():

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    print("=" * 72)
    print("  AMR ANALYSIS — LAYER 3: SAMPLING BIAS CORRECTION")
    print("=" * 72)

    # ── Load cleaned dataset ──────────────────────────────────────
    divider("Loading cleaned dataset")
    input_path = os.path.join(RESULTS_FOLDER, INPUT_FILE)
    if not os.path.exists(input_path):
        print(f"  ERROR: {input_path} not found.")
        sys.exit(1)

    df = pd.read_csv(input_path, low_memory=False)
    print(f"  Shape: {df.shape[0]:,} isolates × {df.shape[1]} columns")

    panel_present = [c for c in PANEL if c in df.columns]
    panel_missing = [c for c in PANEL if c not in df.columns]
    if panel_missing:
        print(f"  ⚠ Panel antibiotics not found: {panel_missing}")
    print(f"  Panel: {panel_present}")

    # ── Step 1: Infer breakpoint standard ────────────────────────
    divider("Step 1 — Inferring breakpoint standard for European unknowns")
    df = infer_breakpoint_standard(df)

    # ── Step 2: Propensity weighting ──────────────────────────────
    divider("Step 2 — Computing propensity weights")
    df = compute_propensity_weights(df)

    # ── Step 3: Weighted co-resistance matrix ─────────────────────
    divider("Step 3 — Weighted co-resistance matrix")
    binary = df[panel_present].copy()
    weights = df["propensity_weight"]

    mat_weighted, counts_weighted = pairwise_coresistance_weighted(
        binary, weights, panel_present
    )
    mat_unweighted = pairwise_coresistance_unweighted(binary, panel_present)

    print(f"\n  UNWEIGHTED co-resistance matrix:")
    print(mat_unweighted.round(3).to_string())
    print(f"\n  WEIGHTED co-resistance matrix:")
    print(mat_weighted.round(3).to_string())

    # Difference
    diff = (mat_weighted - mat_unweighted).round(3)
    print(f"\n  Difference (weighted − unweighted):")
    print(diff.to_string())
    max_diff = diff.abs().max().max()
    print(f"\n  Maximum absolute difference: {max_diff:.3f}")
    if max_diff < 0.05:
        print(f"  ✓ Weighting has minimal effect — results are robust to sampling bias")
    elif max_diff < 0.10:
        print(f"  ⚠ Moderate effect of weighting — report both in paper")
    else:
        print(f"  ⚠ Large effect of weighting — sampling bias was substantial")

    # ── Step 4: Clustering on weighted matrix ─────────────────────
    divider("Step 4 — Module detection on weighted matrix")
    Z_w, mods_weighted, dist_w = cluster_matrix(mat_weighted, N_MODULES)
    Z_u, mods_unweighted, dist_u = cluster_matrix(mat_unweighted, N_MODULES)

    print(f"\n  Weighted module assignments:")
    mod_df_w = pd.DataFrame({
        "antibiotic": panel_present,
        "module_weighted": mods_weighted
    }).sort_values("module_weighted")
    for mod in sorted(mod_df_w["module_weighted"].unique()):
        members = mod_df_w[
            mod_df_w["module_weighted"]==mod
        ]["antibiotic"].tolist()
        print(f"    Module {mod}: {members}")

    compare_modules(mods_weighted, mods_unweighted,
                    "Weighted", "Unweighted", panel_present)

    # ── Step 5: CLSI vs EUCAST comparison ────────────────────────
    divider("Step 5 — Breakpoint standard sensitivity analysis")

    matrices_bp = {}
    module_assignments_bp = {}

    for std in ["CLSI", "EUCAST"]:
        sub = df[df["breakpoint_inferred"] == std]
        if len(sub) < MIN_PAIRS:
            print(f"  ⚠ {std}: too few isolates ({len(sub)}), skipping")
            continue

        binary_sub = sub[panel_present].copy()
        w_sub = sub["propensity_weight"]

        mat, _ = pairwise_coresistance_weighted(binary_sub, w_sub, panel_present)
        matrices_bp[std] = mat

        _, mods, _ = cluster_matrix(mat, N_MODULES)
        module_assignments_bp[std] = mods

        print(f"\n  {std} isolates ({len(sub):,}):")
        print(mat.round(3).to_string())
        print(f"\n  {std} module assignments:")
        for i, (ab, mod) in enumerate(zip(panel_present, mods)):
            print(f"    Module {mod}: {ab}")

    # Compare CLSI vs EUCAST modules
    if "CLSI" in module_assignments_bp and "EUCAST" in module_assignments_bp:
        print(f"\n  ── CLSI vs EUCAST module stability ──")
        n_agree = sum(
            a == b for a, b in zip(
                module_assignments_bp["CLSI"],
                module_assignments_bp["EUCAST"]
            )
        )
        pct_agree = n_agree / len(panel_present) * 100
        print(f"  Antibiotics in same module: {n_agree}/{len(panel_present)} "
              f"({pct_agree:.0f}%)")
        if pct_agree >= 80:
            print(f"  ✓ Module structure is STABLE across breakpoint standards")
        elif pct_agree >= 60:
            print(f"  ⚠ Module structure is PARTIALLY stable — some sensitivity")
        else:
            print(f"  ⚠ Module structure is UNSTABLE — breakpoint standard "
                  f"substantially affects results")

        compare_modules(
            module_assignments_bp["CLSI"],
            module_assignments_bp["EUCAST"],
            "CLSI", "EUCAST", panel_present
        )

    # ── Step 6: Per-species weighted matrices ─────────────────────
    divider("Step 6 — Per-species weighted co-resistance matrices")

    species_matrices = {}
    for sp in df["species"].unique():
        sub = df[df["species"] == sp]
        if len(sub) < MIN_PAIRS:
            continue
        binary_sub = sub[panel_present].copy()
        w_sub = sub["propensity_weight"]
        mat, _ = pairwise_coresistance_weighted(binary_sub, w_sub, panel_present)
        species_matrices[sp] = mat
        print(f"\n  {sp} ({len(sub):,} isolates, weighted):")
        print(mat.round(3).to_string())

    # ── Step 7: Visualisations ────────────────────────────────────
    divider("Step 7 — Generating visualisations")

    # Weighted vs unweighted comparison
    plot_comparison_heatmaps(
        [mat_unweighted, mat_weighted],
        ["Unweighted", "Weighted (bias-corrected)"],
        os.path.join(OUTPUT_FOLDER, "heatmap_weighted_vs_unweighted.png")
    )

    # Weighted dendrogram
    plot_dendrogram(
        Z_w, panel_present,
        "Module dendrogram — weighted analysis (bias-corrected)",
        os.path.join(OUTPUT_FOLDER, "dendrogram_weighted.png")
    )

    # CLSI vs EUCAST comparison
    if len(matrices_bp) >= 2:
        plot_comparison_heatmaps(
            list(matrices_bp.values()),
            [f"{k} isolates" for k in matrices_bp.keys()],
            os.path.join(OUTPUT_FOLDER, "heatmap_clsi_vs_eucast.png")
        )

    # Per-species comparison
    if len(species_matrices) >= 2:
        plot_comparison_heatmaps(
            list(species_matrices.values()),
            list(species_matrices.keys()),
            os.path.join(OUTPUT_FOLDER, "heatmap_species_weighted.png")
        )

    # ── Step 8: Save outputs ──────────────────────────────────────
    divider("Step 8 — Saving outputs")

    # Save weighted dataset
    weighted_path = os.path.join(OUTPUT_FOLDER, "primary_weighted.csv")
    df.to_csv(weighted_path, index=False)
    print(f"  ✓ Weighted dataset → {weighted_path}")

    # Save weighted matrix
    mat_path = os.path.join(OUTPUT_FOLDER, "coresistance_matrix_weighted.csv")
    mat_weighted.to_csv(mat_path)
    print(f"  ✓ Weighted matrix → {mat_path}")

    # Save module assignments
    mod_df_w["module_unweighted"] = mods_unweighted
    if "CLSI" in module_assignments_bp:
        mod_df_w["module_clsi"] = module_assignments_bp["CLSI"]
    if "EUCAST" in module_assignments_bp:
        mod_df_w["module_eucast"] = module_assignments_bp["EUCAST"]
    mod_path = os.path.join(OUTPUT_FOLDER, "module_assignments_layer3.csv")
    mod_df_w.to_csv(mod_path, index=False)
    print(f"  ✓ Module assignments → {mod_path}")

    # ── Final summary ─────────────────────────────────────────────
    divider("FINAL SUMMARY")
    print(f"""
  LAYER 3 COMPLETE

  Panel (6 antibiotics, ampicillin removed):
  {panel_present}

  Key findings:
  ─────────────────────────────────────────────────────
  • Propensity weights computed using species, continent,
    and breakpoint standard as covariates
  • Weighted vs unweighted matrices compared —
    max difference: {max_diff:.3f}
  • Module structure tested across CLSI and EUCAST subsets
  • Per-species weighted matrices computed

  OUTPUT FILES (in {OUTPUT_FOLDER}):
  ─────────────────────────────────────────────────────
  primary_weighted.csv                  — full weighted dataset
  coresistance_matrix_weighted.csv      — bias-corrected matrix
  module_assignments_layer3.csv         — modules across all analyses
  heatmap_weighted_vs_unweighted.png    — bias correction effect
  dendrogram_weighted.png               — weighted module tree
  heatmap_clsi_vs_eucast.png            — breakpoint sensitivity
  heatmap_species_weighted.png          — per-species comparison

  NEXT STEP
  ─────────────────────────────────────────────────────
  Run odds ratio-based cross-species consistency test
  on the weighted, masked, harmonised dataset.
""")

    divider()
    print("  Layer 3 complete.")
    divider()


if __name__ == "__main__":
    main()