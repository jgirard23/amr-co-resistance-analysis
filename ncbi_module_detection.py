"""
Co-resistance Module Detection Script
======================================
Detects recurring multi-antibiotic resistance modules across species.
Loads primary_analysis.csv and secondary_analysis.csv.

Run:
    python3 ncbi_module_detection.py
"""

import pandas as pd
import numpy as np
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving files
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.stats import chi2_contingency, fisher_exact

# ─────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────
DATA_FOLDER   = "/Users/jacobgirard-beaupre/Downloads/NCBI data"
OUTPUT_FOLDER = "/Users/jacobgirard-beaupre/Downloads/NCBI data/results"

PRIMARY_FILE   = "primary_analysis.csv"
SECONDARY_FILE = "secondary_analysis.csv"

PRIMARY_PANEL = [
    "ampicillin",
    "ceftazidime",
    "ciprofloxacin",
    "gentamicin",
    "trimethoprim/sulfamethoxazole",
    "aztreonam",
    "amikacin",
]

SECONDARY_PANEL = [
    "meropenem",
    "ciprofloxacin",
    "ceftazidime",
]

PRIMARY_SPECIES  = ["E. coli", "K. pneumoniae", "A. baumannii"]
SECONDARY_SPECIES = ["P. aeruginosa"]

# Antibiotics that are EUCAST intrinsic for each species
# These will be flagged but NOT removed — removal is done analytically
EUCAST_INTRINSIC = {
    "A. baumannii": ["ampicillin"],
    "P. aeruginosa": ["ampicillin"],
    "E. coli":       ["ampicillin"],
    "K. pneumoniae": ["ampicillin"],
}

# Clustering threshold — number of modules to extract
# (can be adjusted after seeing the dendrogram)
N_MODULES = 3

# Minimum number of isolates tested against both drugs
# for a pairwise co-resistance estimate to be included
MIN_PAIRS = 30

# Significance threshold for cross-species consistency tests
ALPHA = 0.05


# ═════════════════════════════════════════════════════════════════
def divider(title=""):
    width = 72
    if title:
        pad = max(0, width - len(title) - 4)
        print(f"\n{'─'*2} {title} {'─'*pad}")
    else:
        print("─" * width)


def binarise(df, panel):
    """
    Convert S/I/R columns to binary.
    Resistant = 1, Susceptible/Intermediate/SDD = 0, missing = NaN.
    Returns a dataframe with the same index, binary values.
    """
    binary = pd.DataFrame(index=df.index)
    for col in panel:
        if col not in df.columns:
            binary[col] = np.nan
            continue
        s = df[col].astype(str).str.strip().str.upper()
        binary[col] = np.where(
            s.isin(["RESISTANT", "NONSUSCEPTIBLE"]), 1,
            np.where(s.isin(["SUSCEPTIBLE", "INTERMEDIATE",
                              "SUSCEPTIBLE-DOSE DEPENDENT", "S", "I"]), 0,
                     np.nan)
        )
    return binary.astype(float)


def pairwise_coresistance(binary_df, min_pairs=MIN_PAIRS):
    """
    Compute pairwise co-resistance rates using only isolates
    tested against both drugs (pairwise complete observations).
    Returns a symmetric matrix of co-resistance rates (0-1).
    """
    cols = binary_df.columns.tolist()
    n = len(cols)
    matrix = pd.DataFrame(np.nan, index=cols, columns=cols)
    counts = pd.DataFrame(0, index=cols, columns=cols)

    for i in range(n):
        matrix.iloc[i, i] = binary_df[cols[i]].mean()
        counts.iloc[i, i] = binary_df[cols[i]].notna().sum()
        for j in range(i+1, n):
            pair = binary_df[[cols[i], cols[j]]].dropna()
            if len(pair) < min_pairs:
                continue
            # P(resistant to both | tested against both)
            both_r = ((pair[cols[i]] == 1) & (pair[cols[j]] == 1)).sum()
            rate = both_r / len(pair)
            matrix.iloc[i, j] = rate
            matrix.iloc[j, i] = rate
            counts.iloc[i, j] = len(pair)
            counts.iloc[j, i] = len(pair)

    return matrix, counts


def coresistance_to_distance(matrix):
    """
    Convert co-resistance matrix to a distance matrix for clustering.
    Higher co-resistance = lower distance.
    Uses 1 - co_resistance_rate, clamped to [0, 1].
    Missing pairs get distance 1 (maximum).
    """
    dist = 1.0 - matrix.fillna(0)
    dist_arr = dist.values.copy()
    np.fill_diagonal(dist_arr, 0)
    return pd.DataFrame(dist_arr, index=dist.index, columns=dist.columns)


def run_clustering(dist_matrix, n_modules=N_MODULES):
    """Hierarchical clustering on distance matrix."""
    condensed = squareform(dist_matrix.values, checks=False)
    condensed = np.clip(condensed, 0, None)
    Z = linkage(condensed, method="average")
    labels = fcluster(Z, n_modules, criterion="maxclust")
    return Z, labels


def test_cross_species_consistency(binary_all, panel, species_col):
    """
    For each antibiotic pair, test whether co-resistance rate
    is consistent across species using chi-squared test.
    Returns a dataframe of p-values.
    """
    cols = [c for c in panel if c in binary_all.columns]
    results = []

    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            a, b = cols[i], cols[j]
            species_rates = {}
            contingency_ok = True
            tables = []

            for sp in binary_all[species_col].unique():
                sub = binary_all[binary_all[species_col] == sp][[a, b]].dropna()
                if len(sub) < MIN_PAIRS:
                    contingency_ok = False
                    break
                r_both  = ((sub[a]==1) & (sub[b]==1)).sum()
                r_a_only= ((sub[a]==1) & (sub[b]==0)).sum()
                r_b_only= ((sub[a]==0) & (sub[b]==1)).sum()
                r_none  = ((sub[a]==0) & (sub[b]==0)).sum()
                tables.append([[r_both, r_a_only], [r_b_only, r_none]])
                species_rates[sp] = r_both / len(sub) if len(sub) > 0 else np.nan

            if not contingency_ok or len(tables) < 2:
                continue

            # Cochran-Mantel-Haenszel approximation via chi2 on pooled table
            pooled = np.sum(tables, axis=0)
            try:
                chi2, p, dof, _ = chi2_contingency(pooled)
            except Exception:
                p = np.nan

            results.append({
                "antibiotic_1": a,
                "antibiotic_2": b,
                "p_value": p,
                "consistent_across_species": p > ALPHA if not np.isnan(p) else None,
                **{f"rate_{sp.replace(' ', '_').replace('.', '')}": v
                   for sp, v in species_rates.items()}
            })

    return pd.DataFrame(results)


def plot_heatmap(matrix, title, output_path, module_labels=None):
    """Plot co-resistance heatmap with optional module colour bars."""
    n = len(matrix)
    fig, ax = plt.subplots(figsize=(max(8, n*0.8), max(6, n*0.7)))

    # Fill diagonal with marginal resistance rates if available
    data = matrix.values.copy()
    np.fill_diagonal(data, np.nan)

    im = ax.imshow(data, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, label="Co-resistance rate")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(matrix.index, fontsize=9)

    # Add text annotations
    for i in range(n):
        for j in range(n):
            val = data[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7, color="black" if val < 0.6 else "white")

    # Add module colour bar on top if labels provided
    if module_labels is not None:
        colors = plt.cm.Set1(np.linspace(0, 0.8, max(module_labels)))
        for idx, (ab, mod) in enumerate(zip(matrix.columns, module_labels)):
            ax.add_patch(mpatches.Rectangle(
                (idx - 0.5, -1.2), 1, 0.6,
                color=colors[mod-1], clip_on=False, transform=ax.transData
            ))

    ax.set_title(title, fontsize=12, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved heatmap → {output_path}")


def plot_dendrogram(Z, labels, title, output_path, module_assignments=None):
    """Plot hierarchical clustering dendrogram."""
    fig, ax = plt.subplots(figsize=(10, 5))

    # Colour leaves by module if assignments provided
    if module_assignments is not None:
        colors = plt.cm.Set1(np.linspace(0, 0.8, max(module_assignments)))
        leaf_colors = {i: colors[module_assignments[i]-1]
                       for i in range(len(module_assignments))}

        def leaf_color_fn(id):
            if id < len(labels):
                c = leaf_colors.get(id, (0, 0, 0, 1))
                return f"#{int(c[0]*255):02x}{int(c[1]*255):02x}{int(c[2]*255):02x}"
            return "black"

        dendrogram(Z, labels=labels, ax=ax,
                   leaf_font_size=10, leaf_rotation=45,
                   link_color_func=lambda k: "grey")
    else:
        dendrogram(Z, labels=labels, ax=ax,
                   leaf_font_size=10, leaf_rotation=45)

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylabel("Distance (1 - co-resistance rate)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved dendrogram → {output_path}")


def plot_species_heatmaps(binary_all, panel, species_col, output_path):
    """Plot side-by-side co-resistance heatmaps for each species."""
    species_list = binary_all[species_col].unique()
    n_sp = len(species_list)
    n_ab = len(panel)

    fig, axes = plt.subplots(1, n_sp, figsize=(n_ab * n_sp * 0.9, n_ab * 0.8 + 2))
    if n_sp == 1:
        axes = [axes]

    for ax, sp in zip(axes, species_list):
        sub = binary_all[binary_all[species_col] == sp][panel]
        mat = np.full((n_ab, n_ab), np.nan)
        for i in range(n_ab):
            for j in range(i+1, n_ab):
                pair = sub[[panel[i], panel[j]]].dropna()
                if len(pair) < MIN_PAIRS:
                    continue
                both = ((pair[panel[i]]==1) & (pair[panel[j]]==1)).sum()
                mat[i, j] = both / len(pair)
                mat[j, i] = mat[i, j]

        im = ax.imshow(mat, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(n_ab))
        ax.set_yticks(range(n_ab))
        ax.set_xticklabels(panel, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(panel, fontsize=8)
        ax.set_title(sp, fontsize=10, fontweight="bold")

        for i in range(n_ab):
            for j in range(n_ab):
                if not np.isnan(mat[i, j]):
                    ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center",
                            fontsize=6, color="black" if mat[i,j] < 0.6 else "white")

    plt.suptitle("Co-resistance rates by species", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved species comparison → {output_path}")


# ═════════════════════════════════════════════════════════════════
def main():

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    print("=" * 72)
    print("  CO-RESISTANCE MODULE DETECTION")
    print("=" * 72)

    # ── Load primary dataset ──────────────────────────────────────
    divider("Loading primary dataset")
    primary_path = os.path.join(DATA_FOLDER, PRIMARY_FILE)
    if not os.path.exists(primary_path):
        print(f"  ERROR: {primary_path} not found.")
        sys.exit(1)

    df = pd.read_csv(primary_path, low_memory=False)
    print(f"  Shape: {df.shape[0]:,} isolates × {df.shape[1]} columns")
    print(f"  Species: {df['species'].value_counts().to_dict()}")
    print(f"  Breakpoint standards: {df['breakpoint_standard'].value_counts().to_dict()}")

    panel_present = [c for c in PRIMARY_PANEL if c in df.columns]
    panel_missing = [c for c in PRIMARY_PANEL if c not in df.columns]
    if panel_missing:
        print(f"  ⚠ Panel antibiotics not found: {panel_missing}")
    print(f"  Panel antibiotics available: {panel_present}")

    # ── Binarise ──────────────────────────────────────────────────
    divider("Binarising resistance values")
    binary = binarise(df, panel_present)
    binary["species"] = df["species"].values
    binary["breakpoint_standard"] = df["breakpoint_standard"].values

    # Report resistance rates per antibiotic
    print(f"\n  Overall resistance rates:")
    for ab in panel_present:
        col = binary[ab].dropna()
        rate = col.mean() * 100
        n = len(col)
        print(f"    {ab:<40} {rate:5.1f}%  (n={n:,})")

    # ── Overall co-resistance matrix ──────────────────────────────
    divider("Computing overall co-resistance matrix")
    matrix_overall, counts_overall = pairwise_coresistance(binary[panel_present])

    print(f"\n  Pairwise co-resistance matrix (% isolates resistant to BOTH):")
    print(matrix_overall.round(3).to_string())
    print(f"\n  Pairwise sample sizes (isolates tested against BOTH drugs):")
    print(counts_overall.to_string())

    # ── Clustering ────────────────────────────────────────────────
    divider("Hierarchical clustering — module detection")
    dist_matrix = coresistance_to_distance(matrix_overall)
    Z, module_labels = run_clustering(dist_matrix, N_MODULES)

    module_df = pd.DataFrame({
        "antibiotic": panel_present,
        "module": module_labels,
    }).sort_values("module")

    print(f"\n  Module assignments ({N_MODULES} modules):")
    for mod in sorted(module_df["module"].unique()):
        members = module_df[module_df["module"]==mod]["antibiotic"].tolist()
        print(f"    Module {mod}: {members}")

    # Save module assignments
    module_path = os.path.join(OUTPUT_FOLDER, "module_assignments.csv")
    module_df.to_csv(module_path, index=False)
    print(f"\n  ✓ Saved module assignments → {module_path}")

    # ── Per-species co-resistance matrices ────────────────────────
    divider("Per-species co-resistance matrices")
    species_matrices = {}
    for sp in PRIMARY_SPECIES:
        sub = binary[binary["species"] == sp][panel_present]
        if len(sub) < MIN_PAIRS:
            print(f"  ⚠ {sp}: too few isolates, skipping")
            continue
        mat, cnt = pairwise_coresistance(sub)
        species_matrices[sp] = mat
        print(f"\n  {sp} ({len(sub):,} isolates):")
        print(mat.round(3).to_string())

    # ── Cross-species consistency test ────────────────────────────
    divider("Cross-species consistency test")
    consistency = test_cross_species_consistency(binary, panel_present, "species")

    if len(consistency) > 0:
        consistency = consistency.sort_values("p_value")
        print(f"\n  Pairwise co-resistance consistency across species:")
        print(f"  (p > {ALPHA} = consistent, p < {ALPHA} = species-specific)")
        print(consistency.to_string(index=False))

        consistent_pairs = consistency[consistency["consistent_across_species"]==True]
        print(f"\n  ★ {len(consistent_pairs)} / {len(consistency)} pairs are "
              f"consistent across all species (p > {ALPHA})")

        consistency_path = os.path.join(OUTPUT_FOLDER, "cross_species_consistency.csv")
        consistency.to_csv(consistency_path, index=False)
        print(f"  ✓ Saved → {consistency_path}")

    # ── Breakpoint standard sensitivity check ────────────────────
    divider("Breakpoint standard sensitivity check")
    for std in ["CLSI", "EUCAST"]:
        sub = binary[binary["breakpoint_standard"] == std][panel_present]
        if len(sub) < MIN_PAIRS:
            print(f"  ⚠ {std}: too few isolates ({len(sub)}), skipping")
            continue
        mat, _ = pairwise_coresistance(sub)
        print(f"\n  {std} isolates ({len(sub):,}):")
        print(mat.round(3).to_string())

    # ── Visualisations ────────────────────────────────────────────
    divider("Generating visualisations")

    plot_heatmap(
        matrix_overall,
        "Overall co-resistance matrix — Primary analysis\n"
        "(E. coli, K. pneumoniae, A. baumannii)",
        os.path.join(OUTPUT_FOLDER, "heatmap_overall.png"),
        module_labels=module_labels
    )

    plot_dendrogram(
        Z, panel_present,
        "Co-resistance module dendrogram — Primary analysis",
        os.path.join(OUTPUT_FOLDER, "dendrogram_primary.png"),
        module_assignments=module_labels
    )

    plot_species_heatmaps(
        binary, panel_present, "species",
        os.path.join(OUTPUT_FOLDER, "heatmap_by_species.png")
    )

    # ── Secondary analysis — P. aeruginosa ───────────────────────
    divider("Secondary analysis — P. aeruginosa")
    secondary_path = os.path.join(DATA_FOLDER, SECONDARY_FILE)

    if os.path.exists(secondary_path):
        df_sec = pd.read_csv(secondary_path, low_memory=False)
        print(f"  Shape: {df_sec.shape[0]:,} isolates × {df_sec.shape[1]} columns")

        sec_panel = [c for c in SECONDARY_PANEL if c in df_sec.columns]
        binary_sec = binarise(df_sec, sec_panel)

        print(f"\n  P. aeruginosa resistance rates:")
        for ab in sec_panel:
            col = binary_sec[ab].dropna()
            rate = col.mean() * 100
            print(f"    {ab:<40} {rate:5.1f}%  (n={len(col):,})")

        mat_sec, cnt_sec = pairwise_coresistance(binary_sec[sec_panel])
        print(f"\n  P. aeruginosa co-resistance matrix:")
        print(mat_sec.round(3).to_string())

        # Compare P. aeruginosa to primary modules
        print(f"\n  Echo analysis — comparing to primary modules:")
        for mod in sorted(module_df["module"].unique()):
            members = module_df[module_df["module"]==mod]["antibiotic"].tolist()
            overlap = [ab for ab in members if ab in sec_panel]
            if overlap:
                print(f"    Module {mod} members present in P. aeruginosa panel: {overlap}")
                sub = binary_sec[overlap].dropna()
                if len(sub) > MIN_PAIRS:
                    r = sub.mean()
                    print(f"    Resistance rates: {r.round(3).to_dict()}")
            else:
                print(f"    Module {mod}: no panel overlap with P. aeruginosa")

        plot_heatmap(
            mat_sec,
            "Co-resistance matrix — P. aeruginosa (secondary analysis)",
            os.path.join(OUTPUT_FOLDER, "heatmap_secondary.png")
        )

        # Save secondary matrix
        sec_matrix_path = os.path.join(OUTPUT_FOLDER, "secondary_coresistance_matrix.csv")
        mat_sec.to_csv(sec_matrix_path)
        print(f"  ✓ Saved → {sec_matrix_path}")
    else:
        print(f"  ⚠ Secondary file not found: {secondary_path}")

    # ── Save all matrices ─────────────────────────────────────────
    divider("Saving output files")
    matrix_overall.to_csv(os.path.join(OUTPUT_FOLDER, "coresistance_matrix_overall.csv"))
    dist_matrix.to_csv(os.path.join(OUTPUT_FOLDER, "distance_matrix.csv"))
    for sp, mat in species_matrices.items():
        fname = f"coresistance_matrix_{sp.replace(' ', '_').replace('.', '')}.csv"
        mat.to_csv(os.path.join(OUTPUT_FOLDER, fname))
    print(f"  ✓ All matrices saved to {OUTPUT_FOLDER}")

    # ── Final summary ─────────────────────────────────────────────
    divider("FINAL SUMMARY")
    print(f"""
  OUTPUT FILES (in {OUTPUT_FOLDER}):
  ─────────────────────────────────────────────────────
  module_assignments.csv          — antibiotic → module mapping
  coresistance_matrix_overall.csv — full pairwise co-resistance rates
  distance_matrix.csv             — clustering distance matrix
  cross_species_consistency.csv   — consistency test results
  coresistance_matrix_*.csv       — per-species matrices
  secondary_coresistance_matrix.csv — P. aeruginosa matrix

  VISUALISATIONS:
  ─────────────────────────────────────────────────────
  heatmap_overall.png             — overall co-resistance heatmap
  dendrogram_primary.png          — module dendrogram
  heatmap_by_species.png          — side-by-side species comparison
  heatmap_secondary.png           — P. aeruginosa heatmap

  NEXT STEPS:
  ─────────────────────────────────────────────────────
  1. Review dendrogram_primary.png to confirm N_MODULES={N_MODULES} is appropriate
     (adjust N_MODULES at top of script if needed and re-run)
  2. Review cross_species_consistency.csv to identify which
     co-resistance pairs are consistent across all three species
  3. Add temporal analysis once country/date join is resolved
""")

    divider()
    print("  Module detection complete.")
    divider()


if __name__ == "__main__":
    main()