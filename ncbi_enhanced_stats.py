"""
AMR Analysis — Enhanced Statistical Analysis
=============================================
Adds to existing temporal results:
  1. DerSimonian-Laird random-effects meta-analytic OR across species
  2. Cochran-Armitage trend test on yearly ORs per pair per species
  3. Forest plots for meta-analytic ORs
  4. Auto-generated Markdown report

Run:
    python3 ncbi_enhanced_stats.py
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
from datetime import date

# ─────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────
PIPELINE_FILE = "/Users/jacobgirard-beaupre/Downloads/NCBI data/results/full_pipeline/primary_full_pipeline.csv"
TEMPORAL_DIR  = "/Users/jacobgirard-beaupre/Downloads/NCBI data/results/temporal"
OUTPUT_DIR    = "/Users/jacobgirard-beaupre/Downloads/NCBI data/results/enhanced"

CONSISTENT_PAIRS = [
    ("ceftazidime", "gentamicin"),
    ("ceftazidime", "amikacin"),
    ("trimethoprim/sulfamethoxazole", "amikacin"),
]

SPECIES_LIST = ["E. coli", "K. pneumoniae", "A. baumannii"]

YEAR_MIN  = 2005
YEAR_MAX  = 2024
MIN_PAIRS = 20

COLORS = {
    "E. coli":       "#2196F3",
    "K. pneumoniae": "#F44336",
    "A. baumannii":  "#4CAF50",
    "Pooled":        "#9C27B0",
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


def compute_or_with_ci(sub, a, b):
    """Haldane-Anscombe OR with 95% CI and SE of log OR."""
    pair = sub[[a, b]].dropna()
    if len(pair) < MIN_PAIRS:
        return np.nan, np.nan, np.nan, np.nan, len(pair)
    r_both = ((pair[a] == 1) & (pair[b] == 1)).sum()
    r_a    = ((pair[a] == 1) & (pair[b] == 0)).sum()
    r_b    = ((pair[a] == 0) & (pair[b] == 1)).sum()
    r_none = ((pair[a] == 0) & (pair[b] == 0)).sum()
    or_val = ((r_both + 0.5) * (r_none + 0.5)) / \
             ((r_a + 0.5) * (r_b + 0.5))
    se = np.sqrt(1/(r_both+0.5) + 1/(r_a+0.5) +
                 1/(r_b+0.5) + 1/(r_none+0.5))
    log_or = np.log(or_val)
    ci_lo  = np.exp(log_or - 1.96 * se)
    ci_hi  = np.exp(log_or + 1.96 * se)
    return round(or_val, 3), round(ci_lo, 3), round(ci_hi, 3), round(se, 4), len(pair)


# ═════════════════════════════════════════════════════════════════
# 1. DerSimonian-Laird Random-Effects Meta-Analysis
# ═════════════════════════════════════════════════════════════════

def dersimonian_laird(log_ors, ses):
    """
    DerSimonian-Laird random-effects meta-analysis.
    Returns pooled OR, 95% CI, I², Q statistic, p-value for heterogeneity.
    """
    log_ors = np.array(log_ors)
    ses     = np.array(ses)
    wi      = 1 / ses**2  # fixed-effect weights

    # Fixed-effect pooled estimate
    fe_log_or = np.sum(wi * log_ors) / np.sum(wi)

    # Cochran Q
    Q  = np.sum(wi * (log_ors - fe_log_or)**2)
    df = len(log_ors) - 1

    # I²
    I2 = max(0, (Q - df) / Q * 100) if Q > 0 else 0

    # Heterogeneity p-value
    p_het = 1 - stats.chi2.cdf(Q, df) if df > 0 else np.nan

    # Between-study variance τ²
    C    = np.sum(wi) - np.sum(wi**2) / np.sum(wi)
    tau2 = max(0, (Q - df) / C) if C > 0 else 0

    # Random-effects weights
    wi_re     = 1 / (ses**2 + tau2)
    re_log_or = np.sum(wi_re * log_ors) / np.sum(wi_re)
    se_re     = np.sqrt(1 / np.sum(wi_re))

    pooled_or = np.exp(re_log_or)
    ci_lo     = np.exp(re_log_or - 1.96 * se_re)
    ci_hi     = np.exp(re_log_or + 1.96 * se_re)

    # Z-test for pooled OR ≠ 1
    z     = re_log_or / se_re
    p_val = 2 * (1 - stats.norm.cdf(abs(z)))

    return {
        "pooled_OR":    round(pooled_or, 3),
        "CI_lower":     round(ci_lo, 3),
        "CI_upper":     round(ci_hi, 3),
        "SE":           round(se_re, 4),
        "I2":           round(I2, 1),
        "Q":            round(Q, 3),
        "p_het":        round(p_het, 4) if not np.isnan(p_het) else np.nan,
        "p_val":        round(p_val, 4),
        "tau2":         round(tau2, 4),
        "n_studies":    len(log_ors),
    }


def run_meta_analysis(df, pairs, species_list):
    divider("Meta-analytic OR (DerSimonian-Laird random effects)")
    results = []

    for a, b in pairs:
        pair_label = f"{a} × {b}"
        log_ors, ses, ns, sps = [], [], [], []

        for sp in species_list:
            sp_df = df[df["species"] == sp]
            or_v, ci_lo, ci_hi, se, n = compute_or_with_ci(sp_df, a, b)
            if np.isnan(or_v) or or_v <= 0:
                continue
            log_ors.append(np.log(or_v))
            ses.append(se)
            ns.append(n)
            sps.append(sp)

        if len(log_ors) < 2:
            print(f"\n  {pair_label}: insufficient species for meta-analysis")
            continue

        meta = dersimonian_laird(log_ors, ses)
        print(f"\n  {pair_label}")
        print(f"    Species included : {', '.join(sps)}")
        print(f"    Pooled OR        : {meta['pooled_OR']} "
              f"(95% CI {meta['CI_lower']}–{meta['CI_upper']})")
        print(f"    p-value          : {meta['p_val']}")
        print(f"    I²               : {meta['I2']}%")
        print(f"    Q (p_het)        : {meta['Q']} (p={meta['p_het']})")

        results.append({
            "pair":       pair_label,
            "antibiotic_1": a,
            "antibiotic_2": b,
            "species":    sps,
            "log_ors":    log_ors,
            "ses":        ses,
            "ns":         ns,
            "meta":       meta,
        })

    return results


# ═════════════════════════════════════════════════════════════════
# 2. Cochran-Armitage Trend Test
# ═════════════════════════════════════════════════════════════════

def cochran_armitage_trend(years, log_ors, ses):
    """
    Weighted linear regression of log OR on year.
    Returns slope, p-value, and direction.
    """
    years   = np.array(years)
    log_ors = np.array(log_ors)
    ses     = np.array(ses)
    weights = 1 / ses**2

    # Weighted least squares
    W   = np.sum(weights)
    Wx  = np.sum(weights * years)
    Wy  = np.sum(weights * log_ors)
    Wxx = np.sum(weights * years**2)
    Wxy = np.sum(weights * years * log_ors)

    denom = W * Wxx - Wx**2
    if abs(denom) < 1e-10:
        return np.nan, np.nan, "insufficient data"

    slope     = (W * Wxy - Wx * Wy) / denom
    intercept = (Wy - slope * Wx) / W
    se_slope  = np.sqrt(W / denom)
    z         = slope / se_slope
    p_val     = 2 * (1 - stats.norm.cdf(abs(z)))

    direction = "increasing" if slope > 0 else "decreasing"
    return round(slope, 5), round(p_val, 4), direction


def run_trend_tests(df, pairs, species_list):
    divider("Cochran-Armitage Trend Tests (yearly OR trajectories)")
    results = []

    for a, b in pairs:
        pair_label = f"{a} × {b}"
        print(f"\n  {pair_label}")

        for sp in species_list:
            sp_df  = df[df["species"] == sp]
            years  = sorted(sp_df["collection_year"].dropna()
                           .unique().astype(int))
            years  = [y for y in years if YEAR_MIN <= y <= YEAR_MAX]

            yr_data = []
            for yr in years:
                yr_sub = sp_df[sp_df["collection_year"] == yr]
                or_v, ci_lo, ci_hi, se, n = compute_or_with_ci(yr_sub, a, b)
                if not np.isnan(or_v) and or_v > 0:
                    yr_data.append((yr, np.log(or_v), se, n))

            if len(yr_data) < 3:
                print(f"    {sp}: insufficient yearly data (n={len(yr_data)} years)")
                continue

            yrs_arr     = np.array([d[0] for d in yr_data])
            log_ors_arr = np.array([d[1] for d in yr_data])
            ses_arr     = np.array([d[2] for d in yr_data])

            slope, p_val, direction = cochran_armitage_trend(
                yrs_arr, log_ors_arr, ses_arr
            )

            sig = "***" if p_val < 0.001 else \
                  "**"  if p_val < 0.01  else \
                  "*"   if p_val < 0.05  else "ns"

            print(f"    {sp:<20} slope={slope:+.4f}  "
                  f"p={p_val}  {direction}  {sig}")

            results.append({
                "pair":        pair_label,
                "antibiotic_1": a,
                "antibiotic_2": b,
                "species":     sp,
                "n_years":     len(yr_data),
                "slope":       slope,
                "p_trend":     p_val,
                "direction":   direction,
                "significant": p_val < 0.05,
            })

    return pd.DataFrame(results)


# ═════════════════════════════════════════════════════════════════
# 3. Forest Plots
# ═════════════════════════════════════════════════════════════════

def plot_forest(meta_results, output_path):
    n_pairs = len(meta_results)
    fig, axes = plt.subplots(1, n_pairs,
                             figsize=(n_pairs * 5, 5))
    if n_pairs == 1:
        axes = [axes]

    for ax, result in zip(axes, meta_results):
        pair_label = result["pair"]
        sps        = result["species"]
        log_ors    = result["log_ors"]
        ses        = result["ses"]
        ns         = result["ns"]
        meta       = result["meta"]

        y_pos = list(range(len(sps), 0, -1))

        # Per-species ORs
        for i, (sp, log_or, se, n, y) in enumerate(
                zip(sps, log_ors, ses, ns, y_pos)):
            or_v  = np.exp(log_or)
            ci_lo = np.exp(log_or - 1.96 * se)
            ci_hi = np.exp(log_or + 1.96 * se)
            color = COLORS.get(sp, "gray")
            ax.errorbar(or_v, y, xerr=[[or_v - ci_lo], [ci_hi - or_v]],
                        fmt="s", color=color, markersize=8,
                        capsize=4, linewidth=1.5,
                        label=f"{sp} (n={n:,})")
            ax.text(ci_hi * 1.05, y,
                    f"{or_v:.2f} ({ci_lo:.2f}–{ci_hi:.2f})",
                    va="center", fontsize=7)

        # Pooled diamond
        y_pool = 0
        p_or   = meta["pooled_OR"]
        p_lo   = meta["CI_lower"]
        p_hi   = meta["CI_upper"]
        diamond_x = [p_lo, p_or, p_hi, p_or, p_lo]
        diamond_y = [y_pool, y_pool + 0.3, y_pool, y_pool - 0.3, y_pool]
        ax.fill(diamond_x, diamond_y, color=COLORS["Pooled"], alpha=0.8)
        ax.text(p_hi * 1.05, y_pool,
                f"Pooled: {p_or:.2f} ({p_lo:.2f}–{p_hi:.2f})\n"
                f"I²={meta['I2']}%  p={meta['p_val']}",
                va="center", fontsize=7, color=COLORS["Pooled"],
                fontweight="bold")

        ax.axvline(1, color="black", linestyle="--", linewidth=1)
        ax.set_xscale("log")
        ax.set_xlabel("Odds ratio (log scale)", fontsize=9)
        ax.set_yticks(y_pos + [0])
        ax.set_yticklabels(sps + ["Pooled"], fontsize=8)
        ax.set_title(pair_label, fontsize=9, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="x")

    plt.suptitle("Forest plots — Co-resistance meta-analysis across species",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {os.path.basename(output_path)}")


def plot_trend_trajectories(df, pairs, species_list, trend_df, output_dir):
    """OR trajectory plots annotated with trend test results."""
    for a, b in pairs:
        pair_label = f"{a} × {b}"
        safe       = pair_label.replace("/", "_").replace(" ", "_") \
                               .replace("×", "x")
        fig, ax = plt.subplots(figsize=(12, 5))

        for sp in species_list:
            sp_df = df[df["species"] == sp]
            years = sorted(sp_df["collection_year"].dropna()
                          .unique().astype(int))
            years = [y for y in years if YEAR_MIN <= y <= YEAR_MAX]

            yr_data = []
            for yr in years:
                yr_sub = sp_df[sp_df["collection_year"] == yr]
                or_v, ci_lo, ci_hi, se, n = compute_or_with_ci(yr_sub, a, b)
                if not np.isnan(or_v):
                    yr_data.append((yr, or_v, ci_lo, ci_hi))

            if len(yr_data) < 2:
                continue

            yrs_plot  = [d[0] for d in yr_data]
            ors_plot  = [d[1] for d in yr_data]
            cis_lo    = [d[2] for d in yr_data]
            cis_hi    = [d[3] for d in yr_data]
            color     = COLORS.get(sp, "gray")

            # Trend annotation
            trend_row = trend_df[
                (trend_df["antibiotic_1"] == a) &
                (trend_df["antibiotic_2"] == b) &
                (trend_df["species"] == sp)
            ]
            trend_label = ""
            if len(trend_row) > 0:
                row = trend_row.iloc[0]
                sig = "p<0.05" if row["significant"] else "ns"
                trend_label = (f" ({row['direction'][:3]}, "
                               f"p={row['p_trend']}, {sig})")

            ax.plot(yrs_plot, ors_plot, marker="o",
                    label=f"{sp}{trend_label}",
                    color=color, linewidth=2)
            ax.fill_between(yrs_plot, cis_lo, cis_hi,
                            alpha=0.15, color=color)

        ax.axhline(1, color="black", linestyle="--",
                   linewidth=1, label="OR = 1")
        ax.axvline(2016, color="gray", linestyle=":",
                   linewidth=1, label="Period split (2016)")
        ax.set_xlabel("Collection year")
        ax.set_ylabel("Odds ratio (log scale)")
        ax.set_yscale("log")
        ax.set_title(f"Temporal OR trajectory — {pair_label}",
                     fontsize=12, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        out = os.path.join(output_dir,
                           f"trajectory_annotated_{safe}.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  ✓ {os.path.basename(out)}")


# ═════════════════════════════════════════════════════════════════
# 4. Markdown Report
# ═════════════════════════════════════════════════════════════════

def generate_report(df, df_dated, meta_results, trend_df,
                    stability_df, output_path):
    today = date.today().strftime("%B %d, %Y")

    # Summary counts
    n_total  = len(df)
    n_dated  = len(df_dated)
    pct      = n_dated / n_total * 100
    sp_counts = df_dated["species"].value_counts()

    lines = []

    lines += [
        "# AMR Co-resistance Temporal Stability Analysis",
        f"**Generated:** {today}  ",
        f"**Dataset:** BV-BRC (NCBI Pathogen Detection)  ",
        "",
        "---",
        "",
        "## 1. Dataset Overview",
        "",
        f"- Total isolates: **{n_total:,}**",
        f"- Isolates with valid collection year: **{n_dated:,}** ({pct:.1f}%)",
        "",
        "| Species | Dated isolates |",
        "|---------|---------------|",
    ]
    for sp, n in sp_counts.items():
        lines.append(f"| {sp} | {n:,} |")

    lines += [
        "",
        "**Study period:** 2005–2024  ",
        "**Early period:** 2005–2015 | **Late period:** 2016–2024  ",
        "",
        "---",
        "",
        "## 2. Consistent Co-resistance Pairs",
        "",
        "Three antibiotic pairs were identified as cross-species consistent "
        "(OR > 1 in all species, <3-fold variation) in the full pipeline:",
        "",
    ]
    for a, b in CONSISTENT_PAIRS:
        lines.append(f"- **{a} × {b}**")

    lines += [
        "",
        "---",
        "",
        "## 3. Meta-analytic Odds Ratios (DerSimonian-Laird)",
        "",
        "Random-effects meta-analysis pooling OR estimates across species.",
        "",
        "| Pair | Pooled OR | 95% CI | p-value | I² | Heterogeneity p |",
        "|------|-----------|--------|---------|-----|----------------|",
    ]
    for result in meta_results:
        m = result["meta"]
        lines.append(
            f"| {result['pair']} | {m['pooled_OR']} | "
            f"{m['CI_lower']}–{m['CI_upper']} | "
            f"{m['p_val']} | {m['I2']}% | {m['p_het']} |"
        )

    lines += [
        "",
        "**Interpretation:**",
        "",
    ]
    for result in meta_results:
        m   = result["meta"]
        sig = "statistically significant" if m["p_val"] < 0.05 \
              else "not statistically significant"
        het = "substantial heterogeneity" if m["I2"] > 50 \
              else "moderate heterogeneity" if m["I2"] > 25 \
              else "low heterogeneity"
        lines.append(
            f"- **{result['pair']}**: Pooled OR = {m['pooled_OR']} "
            f"(95% CI {m['CI_lower']}–{m['CI_upper']}), {sig} "
            f"(p={m['p_val']}). {het} across species (I²={m['I2']}%)."
        )

    lines += [
        "",
        "---",
        "",
        "## 4. Temporal Trend Tests (Cochran-Armitage)",
        "",
        "Weighted linear regression of log OR on collection year.",
        "",
        "| Pair | Species | n years | Slope | p-trend | Direction | Significant |",
        "|------|---------|---------|-------|---------|-----------|-------------|",
    ]
    for _, row in trend_df.iterrows():
        sig = "Yes *" if row["significant"] else "No"
        lines.append(
            f"| {row['pair']} | {row['species']} | {row['n_years']} | "
            f"{row['slope']:+.4f} | {row['p_trend']} | "
            f"{row['direction']} | {sig} |"
        )

    lines += [
        "",
        "**Interpretation:**",
        "",
    ]
    for _, row in trend_df.iterrows():
        if row["significant"]:
            lines.append(
                f"- **{row['pair']} in {row['species']}**: "
                f"Statistically significant {row['direction']} trend "
                f"over time (slope={row['slope']:+.4f}, p={row['p_trend']})."
            )
    if not trend_df["significant"].any():
        lines.append(
            "- No statistically significant temporal trends detected "
            "across any pair-species combination."
        )

    lines += [
        "",
        "---",
        "",
        "## 5. Period Stability Summary (Early vs Late)",
        "",
        "| Pair | Species | Early OR | Late OR | Ratio | Stable |",
        "|------|---------|----------|---------|-------|--------|",
    ]
    if stability_df is not None:
        for _, row in stability_df.iterrows():
            e_str = f"{row['OR_early']:.3f}" \
                    if pd.notna(row["OR_early"]) else "N/A"
            l_str = f"{row['OR_late']:.3f}" \
                    if pd.notna(row["OR_late"]) else "N/A"
            r_str = f"{row['OR_ratio_period']:.2f}" \
                    if pd.notna(row.get("OR_ratio_period")) else "N/A"
            stab  = "✓" if row["temporally_stable"] else "⚠"
            lines.append(
                f"| {row['antibiotic_1']} × {row['antibiotic_2']} | "
                f"{row['species']} | {e_str} | {l_str} | {r_str} | {stab} |"
            )

    lines += [
        "",
        "---",
        "",
        "## 6. Limitations",
        "",
        "- E. coli temporal coverage is limited (90 dated isolates of 7,789 total) "
        "due to synthetic genome IDs in the primary dataset; "
        "temporal conclusions for E. coli should be interpreted cautiously.",
        "- K. pneumoniae dates are concentrated in 2012–2018, "
        "limiting long-term trend detection.",
        "- A. baumannii and K. pneumoniae temporal windows do not fully overlap, "
        "which may confound cross-species period comparisons.",
        "- The 2018 K. pneumoniae cohort (n=296) shows extreme OR values "
        "(e.g. ceftazidime × amikacin OR=234), likely reflecting a "
        "clonal outbreak rather than a population trend.",
        "- P. aeruginosa could not be included due to missing temporal data.",
        "",
        "---",
        "",
        "## 7. Methods",
        "",
        "**OR estimation:** Haldane-Anscombe correction applied to all "
        "2×2 tables (0.5 added to each cell). 95% CI computed via "
        "log OR ± 1.96 × SE.",
        "",
        "**Meta-analysis:** DerSimonian-Laird random-effects model. "
        "Between-study variance τ² estimated by method-of-moments. "
        "Heterogeneity assessed by Cochran Q and I².",
        "",
        "**Trend test:** Weighted linear regression of log OR on "
        "collection year, with inverse-variance weights. "
        "Significance threshold α=0.05.",
        "",
        "**Temporal stability criterion:** OR > 1 in both periods "
        "and <3-fold change between early (2005–2015) and "
        "late (2016–2024) periods.",
        "",
        "---",
        "",
        "*Report generated automatically by ncbi_enhanced_stats.py*",
    ]

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  ✓ {os.path.basename(output_path)}")


# ═════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 72)
    print("  AMR ANALYSIS — ENHANCED STATISTICAL ANALYSIS")
    print("=" * 72)

    # ── Load data ─────────────────────────────────────────────────
    divider("Loading data")
    df = pd.read_csv(PIPELINE_FILE, low_memory=False,
                     dtype={"Genome ID": str})
    df["collection_year"] = pd.to_numeric(df["collection_year"],
                                          errors="coerce")
    print(f"  Pipeline: {df.shape[0]:,} × {df.shape[1]}")

    df_dated = df[df["collection_year"].between(YEAR_MIN, YEAR_MAX)].copy()
    df_dated["collection_year"] = df_dated["collection_year"].astype(int)
    print(f"  Dated isolates: {len(df_dated):,}")
    print(df_dated["species"].value_counts().to_string())

    # Load existing stability results if available
    stab_path = os.path.join(TEMPORAL_DIR, "temporal_stability.csv")
    stability_df = pd.read_csv(stab_path) if os.path.exists(stab_path) \
                   else None

    # ── Meta-analysis ─────────────────────────────────────────────
    meta_results = run_meta_analysis(df_dated, CONSISTENT_PAIRS,
                                     SPECIES_LIST)

    # ── Trend tests ───────────────────────────────────────────────
    trend_df = run_trend_tests(df_dated, CONSISTENT_PAIRS, SPECIES_LIST)

    # ── Plots ─────────────────────────────────────────────────────
    divider("Generating plots")

    if meta_results:
        plot_forest(
            meta_results,
            os.path.join(OUTPUT_DIR, "forest_plots.png")
        )

    plot_trend_trajectories(
        df_dated, CONSISTENT_PAIRS, SPECIES_LIST, trend_df, OUTPUT_DIR
    )

    # ── Save CSVs ─────────────────────────────────────────────────
    divider("Saving outputs")

    trend_df.to_csv(
        os.path.join(OUTPUT_DIR, "trend_test_results.csv"), index=False
    )
    print(f"  ✓ trend_test_results.csv")

    meta_rows = []
    for r in meta_results:
        m = r["meta"]
        meta_rows.append({
            "pair":        r["pair"],
            "antibiotic_1": r["antibiotic_1"],
            "antibiotic_2": r["antibiotic_2"],
            "n_species":   m["n_studies"],
            "pooled_OR":   m["pooled_OR"],
            "CI_lower":    m["CI_lower"],
            "CI_upper":    m["CI_upper"],
            "p_value":     m["p_val"],
            "I2":          m["I2"],
            "Q":           m["Q"],
            "p_het":       m["p_het"],
            "tau2":        m["tau2"],
        })
    meta_df = pd.DataFrame(meta_rows)
    meta_df.to_csv(
        os.path.join(OUTPUT_DIR, "meta_analysis_results.csv"), index=False
    )
    print(f"  ✓ meta_analysis_results.csv")

    # ── Report ────────────────────────────────────────────────────
    divider("Generating report")
    report_path = os.path.join(OUTPUT_DIR, "amr_analysis_report.md")
    generate_report(df, df_dated, meta_results, trend_df,
                    stability_df, report_path)

    print(f"\n  ✓ All outputs saved to {OUTPUT_DIR}")

    # ── Final summary ─────────────────────────────────────────────
    divider("FINAL SUMMARY")
    print(f"""
  ENHANCED ANALYSIS COMPLETE

  Meta-analysis:
""")
    for r in meta_results:
        m = r["meta"]
        print(f"    {r['pair']}")
        print(f"      Pooled OR={m['pooled_OR']} "
              f"({m['CI_lower']}–{m['CI_upper']}), "
              f"p={m['p_val']}, I²={m['I2']}%")

    print(f"\n  Temporal trends:")
    for _, row in trend_df.iterrows():
        sig = "SIGNIFICANT" if row["significant"] else "ns"
        print(f"    {row['pair']} | {row['species']:<20} "
              f"p={row['p_trend']}  {row['direction']}  {sig}")

    print(f"""
  OUTPUT FILES (in {OUTPUT_DIR}):
  ─────────────────────────────────────────────
  meta_analysis_results.csv   — pooled ORs per pair
  trend_test_results.csv      — Cochran-Armitage results
  forest_plots.png            — forest plot per pair
  trajectory_annotated_*.png  — OR trajectories with trend annotations
  amr_analysis_report.md      — full markdown report
""")
    divider()
    print("  Enhanced analysis complete.")
    divider()


if __name__ == "__main__":
    main()
