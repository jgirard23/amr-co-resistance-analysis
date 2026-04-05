"""
AMR Analysis — Permutation Testing + Geographic Split Validation
================================================================
Two additional analyses to strengthen the paper:
 
1. Permutation testing (n=1000): tests whether observed ORs exceed
   chance co-occurrence under the null hypothesis of no association.
 
2. Geographic split validation: replicates the core module analysis
   independently in European vs Asian/other isolate subsets.
 
Run:
    python3 ncbi_validation.py
"""
 
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")
 
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
 
# ─────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────
PIPELINE_FILE = "/Users/jacobgirard-beaupre/Downloads/NCBI data/results/full_pipeline/primary_full_pipeline.csv"
OUTPUT_DIR    = "/Users/jacobgirard-beaupre/Downloads/NCBI data/results/validation"
 
CONSISTENT_PAIRS = [
    ("ceftazidime", "gentamicin"),
    ("ceftazidime", "amikacin"),
    ("trimethoprim/sulfamethoxazole", "amikacin"),
]
 
SPECIES_LIST  = ["E. coli", "K. pneumoniae", "A. baumannii"]
N_PERMUTATIONS = 1000
MIN_PAIRS      = 20
SEED           = 42
 
COLORS = {
    "E. coli":       "#2196F3",
    "K. pneumoniae": "#F44336",
    "A. baumannii":  "#4CAF50",
}
 
 
# ═════════════════════════════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════════════════════════════
 
def divider(title=""):
    width = 72
    if title:
        pad = max(0, width - len(title) - 4)
        print(f"\n{'─'*2} {title} {'─'*pad}")
    else:
        print("─" * width)
 
 
def compute_log_or(sub, a, b):
    """Returns log OR and SE using Haldane-Anscombe correction."""
    pair = sub[[a, b]].dropna()
    if len(pair) < MIN_PAIRS:
        return np.nan, np.nan, len(pair)
    r_both = ((pair[a] == 1) & (pair[b] == 1)).sum()
    r_a    = ((pair[a] == 1) & (pair[b] == 0)).sum()
    r_b    = ((pair[a] == 0) & (pair[b] == 1)).sum()
    r_none = ((pair[a] == 0) & (pair[b] == 0)).sum()
    or_val = ((r_both + 0.5) * (r_none + 0.5)) / \
             ((r_a + 0.5) * (r_b + 0.5))
    se = np.sqrt(1/(r_both+0.5) + 1/(r_a+0.5) +
                 1/(r_b+0.5) + 1/(r_none+0.5))
    return np.log(or_val), se, len(pair)
 
 
def pooled_log_or(df, a, b, species_list):
    """DerSimonian-Laird pooled log OR across species."""
    log_ors, ses = [], []
    for sp in species_list:
        sp_df = df[df["species"] == sp]
        log_or, se, n = compute_log_or(sp_df, a, b)
        if not np.isnan(log_or):
            log_ors.append(log_or)
            ses.append(se)
    if len(log_ors) < 2:
        return np.nan
    log_ors = np.array(log_ors)
    ses     = np.array(ses)
    wi      = 1 / ses**2
    fe_log  = np.sum(wi * log_ors) / np.sum(wi)
    Q       = np.sum(wi * (log_ors - fe_log)**2)
    df_q    = len(log_ors) - 1
    C       = np.sum(wi) - np.sum(wi**2) / np.sum(wi)
    tau2    = max(0, (Q - df_q) / C) if C > 0 else 0
    wi_re   = 1 / (ses**2 + tau2)
    return np.sum(wi_re * log_ors) / np.sum(wi_re)
 
 
# ═════════════════════════════════════════════════════════════════
# 1. PERMUTATION TESTING
# ═════════════════════════════════════════════════════════════════
 
def run_permutation_tests(df, pairs, species_list, n_perm=N_PERMUTATIONS):
    divider("Permutation Testing (n=1,000 permutations)")
    rng = np.random.default_rng(SEED)
    results = []
 
    for a, b in pairs:
        pair_label = f"{a} × {b}"
        print(f"\n  {pair_label}")
 
        # Observed pooled log OR
        obs_log_or = pooled_log_or(df, a, b, species_list)
        if np.isnan(obs_log_or):
            print(f"    Insufficient data — skipping")
            continue
 
        # Permutation null distribution
        null_log_ors = []
        for i in range(n_perm):
            df_perm = df.copy()
            # Shuffle antibiotic b independently within each species
            # (preserves marginal resistance rates per species)
            for sp in species_list:
                sp_idx = df_perm[df_perm["species"] == sp].index
                if b in df_perm.columns and len(sp_idx) > 0:
                    perm_vals = df_perm.loc[sp_idx, b].values.copy()
                    rng.shuffle(perm_vals)
                    df_perm.loc[sp_idx, b] = perm_vals
 
            null_val = pooled_log_or(df_perm, a, b, species_list)
            if not np.isnan(null_val):
                null_log_ors.append(null_val)
 
        null_log_ors = np.array(null_log_ors)
 
        # Permutation p-value (one-tailed: observed > null)
        p_perm = (np.sum(null_log_ors >= obs_log_or) + 1) / (len(null_log_ors) + 1)
 
        obs_or = np.exp(obs_log_or)
        null_or_median = np.exp(np.median(null_log_ors))
        null_or_95 = np.exp(np.percentile(null_log_ors, 95))
 
        print(f"    Observed pooled OR : {obs_or:.3f}")
        print(f"    Null median OR     : {null_or_median:.3f}")
        print(f"    Null 95th pctile   : {null_or_95:.3f}")
        print(f"    Permutation p-value: {p_perm:.4f} "
              f"({'***' if p_perm<0.001 else '**' if p_perm<0.01 else '*' if p_perm<0.05 else 'ns'})")
 
        results.append({
            "pair":           pair_label,
            "antibiotic_1":   a,
            "antibiotic_2":   b,
            "observed_OR":    round(obs_or, 3),
            "null_median_OR": round(null_or_median, 3),
            "null_95th_OR":   round(null_or_95, 3),
            "p_permutation":  round(p_perm, 4),
            "significant":    p_perm < 0.05,
            "n_permutations": len(null_log_ors),
            "null_log_ors":   null_log_ors,
        })
 
    return results
 
 
def plot_permutation_nulls(perm_results, output_path):
    n = len(perm_results)
    fig, axes = plt.subplots(1, n, figsize=(n * 5, 5))
    if n == 1:
        axes = [axes]
 
    for ax, result in zip(axes, perm_results):
        null_ors = np.exp(result["null_log_ors"])
        obs_or   = result["observed_OR"]
 
        ax.hist(null_ors, bins=50, color="#90CAF9", edgecolor="white",
                alpha=0.8, label="Null distribution")
        ax.axvline(obs_or, color="#D32F2F", linewidth=2,
                   label=f"Observed OR = {obs_or:.2f}")
        ax.axvline(result["null_95th_OR"], color="gray",
                   linewidth=1.5, linestyle="--",
                   label=f"Null 95th = {result['null_95th_OR']:.2f}")
 
        p = result["p_permutation"]
        sig = "***" if p < 0.001 else "**" if p < 0.01 else \
              "*" if p < 0.05 else "ns"
        ax.set_title(f"{result['pair']}\np={p} {sig}",
                     fontsize=9, fontweight="bold")
        ax.set_xlabel("Pooled OR (DerSimonian-Laird)")
        ax.set_ylabel("Frequency")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
 
    plt.suptitle("Permutation test — null OR distributions vs observed",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {os.path.basename(output_path)}")
 
 
# ═════════════════════════════════════════════════════════════════
# 2. GEOGRAPHIC SPLIT VALIDATION
# ═════════════════════════════════════════════════════════════════
 
GEO_GROUPS = {
    "Europe":      ["Europe"],
    "Asia":        ["Asia"],
    "N. America":  ["N. America"],
}
 
 
def run_geographic_split(df, pairs, species_list):
    divider("Geographic Split Validation")
 
    # Show continent distribution
    print(f"\n  Continent distribution:")
    print(df["continent"].value_counts().to_string())
 
    results = []
    for geo_label, continents in GEO_GROUPS.items():
        geo_df = df[df["continent"].isin(continents)]
        print(f"\n  ── {geo_label} ({len(geo_df):,} isolates) ──")
        print(f"  Species breakdown:")
        print(geo_df["species"].value_counts().to_string())
 
        for a, b in pairs:
            pair_label = f"{a} × {b}"
            geo_log_ors = []
            geo_ses     = []
            geo_sps     = []
 
            for sp in species_list:
                sp_df = geo_df[geo_df["species"] == sp]
                log_or, se, n = compute_log_or(sp_df, a, b)
                if not np.isnan(log_or):
                    geo_log_ors.append(log_or)
                    geo_ses.append(se)
                    geo_sps.append(sp)
                    or_v = np.exp(log_or)
                    ci_lo = np.exp(log_or - 1.96*se)
                    ci_hi = np.exp(log_or + 1.96*se)
                    print(f"    {pair_label} | {sp:<20} "
                          f"OR={or_v:.3f} ({ci_lo:.3f}–{ci_hi:.3f}) n={n}")
 
            # Pooled OR for this geography
            if len(geo_log_ors) >= 2:
                geo_log_ors = np.array(geo_log_ors)
                geo_ses     = np.array(geo_ses)
                wi      = 1 / geo_ses**2
                fe_log  = np.sum(wi * geo_log_ors) / np.sum(wi)
                Q       = np.sum(wi * (geo_log_ors - fe_log)**2)
                df_q    = len(geo_log_ors) - 1
                C       = np.sum(wi) - np.sum(wi**2) / np.sum(wi)
                tau2    = max(0, (Q - df_q) / C) if C > 0 else 0
                wi_re   = 1 / (geo_ses**2 + tau2)
                pool_log = np.sum(wi_re * geo_log_ors) / np.sum(wi_re)
                se_pool  = np.sqrt(1 / np.sum(wi_re))
                pool_or  = np.exp(pool_log)
                pool_lo  = np.exp(pool_log - 1.96*se_pool)
                pool_hi  = np.exp(pool_log + 1.96*se_pool)
                p_val    = 2*(1 - stats.norm.cdf(abs(pool_log/se_pool)))
                I2       = max(0, (Q - df_q)/Q*100) if Q > 0 else 0
                print(f"    → {pair_label} POOLED ({geo_label}): "
                      f"OR={pool_or:.3f} ({pool_lo:.3f}–{pool_hi:.3f}) "
                      f"p={p_val:.4f} I²={I2:.1f}%")
 
                results.append({
                    "geography":   geo_label,
                    "pair":        pair_label,
                    "antibiotic_1": a,
                    "antibiotic_2": b,
                    "n_species":   len(geo_sps),
                    "pooled_OR":   round(pool_or, 3),
                    "CI_lower":    round(pool_lo, 3),
                    "CI_upper":    round(pool_hi, 3),
                    "p_value":     round(p_val, 4),
                    "I2":          round(I2, 1),
                })
 
    return pd.DataFrame(results)
 
 
def plot_geographic_split(geo_df, pairs, output_path):
    geographies = geo_df["geography"].unique()
    n_pairs     = len(pairs)
 
    fig, axes = plt.subplots(1, n_pairs, figsize=(n_pairs * 5, 5))
    if n_pairs == 1:
        axes = [axes]
 
    geo_colors = {
        "Europe":     "#1565C0",
        "Asia":       "#E65100",
        "N. America": "#2E7D32",
    }
 
    for ax, (a, b) in zip(axes, pairs):
        pair_label = f"{a} × {b}"
        pair_data  = geo_df[geo_df["pair"] == pair_label]
 
        y_pos = list(range(len(geographies), 0, -1))
        for y, geo in zip(y_pos, geographies):
            row = pair_data[pair_data["geography"] == geo]
            if len(row) == 0:
                continue
            row  = row.iloc[0]
            col  = geo_colors.get(geo, "gray")
            or_v = row["pooled_OR"]
            lo   = row["CI_lower"]
            hi   = row["CI_upper"]
            ax.errorbar(or_v, y,
                        xerr=[[or_v - lo], [hi - or_v]],
                        fmt="s", color=col, markersize=10,
                        capsize=5, linewidth=2,
                        label=f"{geo} (I²={row['I2']}%)")
            ax.text(hi * 1.05, y,
                    f"{or_v:.2f} ({lo:.2f}–{hi:.2f})\np={row['p_value']}",
                    va="center", fontsize=7)
 
        ax.axvline(1, color="black", linestyle="--", linewidth=1)
        ax.set_xscale("log")
        ax.set_xlabel("Pooled OR (log scale)", fontsize=9)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(geographies, fontsize=8)
        ax.set_title(pair_label, fontsize=9, fontweight="bold")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3, axis="x")
 
    plt.suptitle("Geographic split validation — pooled OR by region",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {os.path.basename(output_path)}")
 
 
# ═════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════
 
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
 
    print("=" * 72)
    print("  AMR ANALYSIS — PERMUTATION + GEOGRAPHIC SPLIT VALIDATION")
    print("=" * 72)
 
    # ── Load data ─────────────────────────────────────────────────
    divider("Loading data")
    df = pd.read_csv(PIPELINE_FILE, low_memory=False,
                     dtype={"Genome ID": str})
    df["collection_year"] = pd.to_numeric(df["collection_year"],
                                          errors="coerce")
    print(f"  Shape: {df.shape[0]:,} × {df.shape[1]}")
    print(f"  Species: {df['species'].value_counts().to_dict()}")
    print(f"  Continents: {df['continent'].value_counts().to_dict()}")
 
    # ── Permutation tests ─────────────────────────────────────────
    print(f"\n  Running {N_PERMUTATIONS} permutations per pair...")
    print(f"  (This may take 1-2 minutes)")
    perm_results = run_permutation_tests(df, CONSISTENT_PAIRS, SPECIES_LIST)
 
    # ── Geographic split ──────────────────────────────────────────
    geo_df = run_geographic_split(df, CONSISTENT_PAIRS, SPECIES_LIST)
 
    # ── Plots ─────────────────────────────────────────────────────
    divider("Generating plots")
 
    plot_permutation_nulls(
        perm_results,
        os.path.join(OUTPUT_DIR, "permutation_null_distributions.png")
    )
 
    if len(geo_df) > 0:
        plot_geographic_split(
            geo_df, CONSISTENT_PAIRS,
            os.path.join(OUTPUT_DIR, "geographic_split_validation.png")
        )
 
    # ── Save CSVs ─────────────────────────────────────────────────
    divider("Saving outputs")
 
    perm_rows = []
    for r in perm_results:
        perm_rows.append({
            "pair":            r["pair"],
            "observed_OR":     r["observed_OR"],
            "null_median_OR":  r["null_median_OR"],
            "null_95th_OR":    r["null_95th_OR"],
            "p_permutation":   r["p_permutation"],
            "significant":     r["significant"],
            "n_permutations":  r["n_permutations"],
        })
    pd.DataFrame(perm_rows).to_csv(
        os.path.join(OUTPUT_DIR, "permutation_results.csv"), index=False
    )
    print(f"  ✓ permutation_results.csv")
 
    geo_df.drop(columns=[], errors="ignore").to_csv(
        os.path.join(OUTPUT_DIR, "geographic_split_results.csv"), index=False
    )
    print(f"  ✓ geographic_split_results.csv")
 
    # ── Final summary ─────────────────────────────────────────────
    divider("FINAL SUMMARY")
 
    print(f"\n  PERMUTATION TEST RESULTS:")
    for r in perm_results:
        sig = "✓ SIGNIFICANT" if r["significant"] else "✗ not significant"
        print(f"    {r['pair']}")
        print(f"      Observed OR={r['observed_OR']}, "
              f"Null 95th={r['null_95th_OR']}, "
              f"p={r['p_permutation']} — {sig}")
 
    print(f"\n  GEOGRAPHIC SPLIT RESULTS:")
    for _, row in geo_df.iterrows():
        print(f"    {row['geography']:<12} | {row['pair']:<45} "
              f"OR={row['pooled_OR']} ({row['CI_lower']}–{row['CI_upper']}) "
              f"p={row['p_value']}")
 
    print(f"""
  OUTPUT FILES (in {OUTPUT_DIR}):
  ──────────────────────────────────────────────────
  permutation_results.csv          — permutation test summary
  permutation_null_distributions.png — null OR histograms
  geographic_split_results.csv     — pooled ORs by region
  geographic_split_validation.png  — geographic split plot
""")
    divider()
    print("  Validation analysis complete.")
    divider()
 
 
if __name__ == "__main__":
    main()
 