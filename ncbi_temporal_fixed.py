"""
AMR Analysis — Temporal Stability Testing (FIXED)
==================================================
Step 0: Patches primary_full_pipeline.csv with real BV-BRC Genome IDs
        and collection years directly from metadata files.
Step 1: Joins date files and runs temporal stability analysis.

Run:
    python3 ncbi_temporal_fixed.py
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────
DATA_FOLDER   = "/Users/jacobgirard-beaupre/Downloads/NCBI data"
PIPELINE_FILE = "/Users/jacobgirard-beaupre/Downloads/NCBI data/results/full_pipeline/primary_full_pipeline.csv"
OUTPUT_FOLDER = "/Users/jacobgirard-beaupre/Downloads/NCBI data/results/temporal"

DATE_FILES = {
    "A. baumannii":  "abaumannii_dates.csv",
    "P. aeruginosa": "paeruginosa_dates.csv",
    "E. coli":       "ecoli_dates.csv",
    "K. pneumoniae": "kpneumoniae_dates.csv",
}

METADATA_FILES = {
    "E. coli":       "ecoli_genome_metadata.csv",
    "K. pneumoniae": "kpneumoniae_genome_metadata.csv",
    "A. baumannii":  "abaumannii_genome_metadata.csv",
}

CONSISTENT_PAIRS = [
    ("ceftazidime", "gentamicin"),
    ("ceftazidime", "amikacin"),
    ("trimethoprim/sulfamethoxazole", "amikacin"),
]

YEAR_MIN     = 2005
YEAR_MAX     = 2024
EARLY_PERIOD = (2005, 2015)
LATE_PERIOD  = (2016, 2024)
MIN_PAIRS    = 20


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


def load_csv(filepath):
    for sep in [",", "\t"]:
        try:
            df = pd.read_csv(filepath, sep=sep, low_memory=False,
                             on_bad_lines="skip")
            if df.shape[1] > 1:
                return df
        except Exception:
            continue
    raise ValueError(f"Could not parse: {filepath}")


def normalise_id(series):
    def _norm(val):
        s = str(val).replace('"', '').strip()
        try:
            return f"{float(s):.5f}"
        except (ValueError, TypeError):
            return s
    return series.apply(_norm)


def extract_year(series):
    return (
        series.astype(str)
        .str.replace('"', '', regex=False)
        .str.strip()
        .str.extract(r'(\d{4})')[0]
        .apply(lambda x: int(x) if pd.notna(x) and x != 'nan' else np.nan)
    )


# ═════════════════════════════════════════════════════════════════
# STEP 0 — PATCH CSV WITH REAL IDs AND YEARS FROM METADATA
# ═════════════════════════════════════════════════════════════════

def patch_pipeline_csv():
    divider("Step 0 — Patching pipeline CSV with real BV-BRC IDs")

    df = pd.read_csv(PIPELINE_FILE, low_memory=False, dtype={"Genome ID": str})
    df["Genome ID"] = df["Genome ID"].astype(str)
    df["collection_year"] = pd.to_numeric(df["collection_year"], errors="coerce")

    for sp, meta_file in METADATA_FILES.items():
        meta_path = os.path.join(DATA_FOLDER, meta_file)
        if not os.path.exists(meta_path):
            print(f"  ⚠ [{sp}] Metadata file not found: {meta_path}")
            continue

        meta = pd.read_csv(meta_path, low_memory=False, on_bad_lines="skip")
        meta["_id"] = normalise_id(meta["genome_id"])

        # Extract year from all available date columns
        meta["_year"] = np.nan
        for col in ["collection_date", "isolation_date", "completion_date"]:
            if col in meta.columns:
                yr = pd.to_numeric(
                    meta[col].astype(str).str.extract(r'(\d{4})')[0],
                    errors="coerce"
                )
                yr[~yr.between(YEAR_MIN, YEAR_MAX)] = np.nan
                meta["_year"] = meta["_year"].fillna(yr)

        id_map   = dict(zip(meta["_id"],
                            meta["genome_id"].astype(str)
                            .str.replace('"', '').str.strip()))
        year_map = dict(zip(meta["_id"], meta["_year"]))

        sp_idx      = df.index[df["species"] == sp]
        sp_norm_ids = normalise_id(df.loc[sp_idx, "Genome ID"])

        # Patch Genome IDs
        real_ids = sp_norm_ids.map(id_map)
        valid    = real_ids.notna() & (real_ids != "nan")
        df.loc[sp_idx[valid], "Genome ID"] = real_ids[valid].values

        # Patch collection_year
        years = sp_norm_ids.map(year_map)
        years_float = pd.to_numeric(years, errors="coerce")
        df.loc[sp_idx, "collection_year"] = years_float.values

        n_real  = df.loc[sp_idx, "Genome ID"].str.match(r'^\d+\.\d{4,}$').sum()
        n_years = df.loc[sp_idx, "collection_year"].notna().sum()
        print(f"  {sp}: {n_real}/{len(sp_idx)} real IDs, "
              f"{n_years}/{len(sp_idx)} with year")

    total_years = df["collection_year"].notna().sum()
    print(f"\n  Total isolates with collection year: {total_years:,}/{len(df):,}")
    df.to_csv(PIPELINE_FILE, index=False)
    print("  ✓ Pipeline CSV patched and saved.")
    return df


# ═════════════════════════════════════════════════════════════════
# STEP 1 — TEMPORAL ANALYSIS
# ═════════════════════════════════════════════════════════════════

def compute_or(sub, a, b):
    pair = sub[[a, b]].dropna()
    if len(pair) < MIN_PAIRS:
        return np.nan, np.nan, np.nan
    r_both = ((pair[a] == 1) & (pair[b] == 1)).sum()
    r_a    = ((pair[a] == 1) & (pair[b] == 0)).sum()
    r_b    = ((pair[a] == 0) & (pair[b] == 1)).sum()
    r_none = ((pair[a] == 0) & (pair[b] == 0)).sum()
    or_val = ((r_both + 0.5) * (r_none + 0.5)) / \
             ((r_a + 0.5) * (r_b + 0.5))
    se = np.sqrt(1/(r_both+0.5) + 1/(r_a+0.5) +
                 1/(r_b+0.5) + 1/(r_none+0.5))
    log_or = np.log(or_val)
    return (round(or_val, 3),
            round(np.exp(log_or - 1.96*se), 3),
            round(np.exp(log_or + 1.96*se), 3))


def period_analysis(df, period_name, year_range, pairs, species_list):
    sub = df[df["collection_year"].between(*year_range)]
    print(f"\n  {period_name}: {len(sub):,} isolates")
    results = []
    for sp in species_list:
        sp_sub = sub[sub["species"] == sp]
        print(f"    {sp}: {len(sp_sub):,} isolates")
        for a, b in pairs:
            or_val, ci_lo, ci_hi = compute_or(sp_sub, a, b)
            results.append({
                "period": period_name, "species": sp,
                "antibiotic_1": a, "antibiotic_2": b,
                "OR": or_val, "CI_lower": ci_lo, "CI_upper": ci_hi,
                "n": len(sp_sub[[a, b]].dropna()),
            })
    return pd.DataFrame(results)


def yearly_or(df, a, b, species_list):
    years = [y for y in sorted(df["collection_year"].dropna()
                               .unique().astype(int))
             if YEAR_MIN <= y <= YEAR_MAX]
    records = []
    for yr in years:
        yr_sub = df[df["collection_year"] == yr]
        for sp in species_list:
            sp_sub = yr_sub[yr_sub["species"] == sp]
            or_val, ci_lo, ci_hi = compute_or(sp_sub, a, b)
            if not np.isnan(or_val):
                records.append({
                    "year": yr, "species": sp,
                    "OR": or_val, "CI_lower": ci_lo, "CI_upper": ci_hi,
                    "n": len(sp_sub[[a, b]].dropna()),
                })
    return pd.DataFrame(records)


COLORS = {"E. coli": "#2196F3",
          "K. pneumoniae": "#F44336",
          "A. baumannii": "#4CAF50"}


def plot_or_trajectory(yearly_df, pair_label, output_path):
    fig, ax = plt.subplots(figsize=(12, 5))
    for sp in yearly_df["species"].unique():
        sp_data = yearly_df[yearly_df["species"] == sp].sort_values("year")
        if len(sp_data) < 2:
            continue
        c = COLORS.get(sp, "gray")
        ax.plot(sp_data["year"], sp_data["OR"],
                marker="o", label=sp, color=c, linewidth=2)
        ax.fill_between(sp_data["year"],
                        sp_data["CI_lower"], sp_data["CI_upper"],
                        alpha=0.15, color=c)
    ax.axhline(1, color="black", linestyle="--", linewidth=1, label="OR = 1")
    ax.axvline(2016, color="gray", linestyle=":", linewidth=1,
               label="Period boundary (2016)")
    ax.set_xlabel("Collection year")
    ax.set_ylabel("Odds ratio (log scale)")
    ax.set_yscale("log")
    ax.set_title(f"Temporal stability — {pair_label}",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {os.path.basename(output_path)}")


def plot_period_comparison(early_df, late_df, pairs, output_path):
    species_list = ["E. coli", "K. pneumoniae", "A. baumannii"]
    fig, axes = plt.subplots(1, len(pairs), figsize=(len(pairs) * 5, 5))
    if len(pairs) == 1:
        axes = [axes]
    period_colors = {"Early (2005–2015)": "#90CAF9",
                     "Late (2016–2024)":  "#1565C0"}
    all_results = pd.concat([early_df, late_df])
    for ax, (a, b) in zip(axes, pairs):
        pair_data = all_results[(all_results["antibiotic_1"] == a) &
                                (all_results["antibiotic_2"] == b)]
        x = np.arange(len(species_list))
        width = 0.35
        for i, (period, color) in enumerate(period_colors.items()):
            period_data = pair_data[pair_data["period"] == period]
            ors, cis_lo, cis_hi = [], [], []
            for sp in species_list:
                row = period_data[period_data["species"] == sp]
                if len(row) > 0 and not np.isnan(row["OR"].values[0]):
                    ors.append(row["OR"].values[0])
                    cis_lo.append(row["CI_lower"].values[0])
                    cis_hi.append(row["CI_upper"].values[0])
                else:
                    ors.append(0); cis_lo.append(0); cis_hi.append(0)
            bars = ax.bar(x + i*width, ors, width,
                          label=period, color=color, alpha=0.85)
            for bar, lo, hi in zip(bars, cis_lo, cis_hi):
                if bar.get_height() > 0:
                    ax.errorbar(bar.get_x() + bar.get_width()/2,
                                bar.get_height(),
                                yerr=[[bar.get_height()-lo],
                                      [hi-bar.get_height()]],
                                fmt='none', color='black', capsize=3)
        ax.set_xticks(x + width/2)
        ax.set_xticklabels([s.split()[0] + "\n" + " ".join(s.split()[1:])
                            for s in species_list], fontsize=8)
        ax.set_ylabel("Odds ratio")
        ax.set_title(f"{a}\n× {b}", fontsize=9, fontweight="bold")
        ax.axhline(1, color="red", linestyle="--", linewidth=0.8)
        ax.legend(fontsize=7)
    plt.suptitle("Co-resistance OR: Early vs Late period by species",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {os.path.basename(output_path)}")


# ═════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════

def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    print("=" * 72)
    print("  AMR ANALYSIS — TEMPORAL STABILITY TESTING (FIXED)")
    print("=" * 72)

    # ── Step 0: Patch the CSV ─────────────────────────────────────
    df = patch_pipeline_csv()

    # ── Step 1: Filter to dated isolates ─────────────────────────
    divider("Temporal analysis")

    df["collection_year"] = pd.to_numeric(df["collection_year"],
                                          errors="coerce")
    df_dated = df[df["collection_year"].notna()].copy()
    df_dated["collection_year"] = df_dated["collection_year"].astype(int)

    print(f"\n  Working dataset: {len(df_dated):,} dated isolates")
    print(f"  Species breakdown:")
    print(df_dated["species"].value_counts().to_string())

    print(f"\n  Year distribution:")
    print(df_dated["collection_year"].value_counts()
          .sort_index().to_string())

    species_list = ["E. coli", "K. pneumoniae", "A. baumannii"]

    # ── Early vs late ─────────────────────────────────────────────
    divider("Early vs Late period — consistent pair ORs")

    early_df = period_analysis(df_dated, "Early (2005–2015)",
                               EARLY_PERIOD, CONSISTENT_PAIRS, species_list)
    late_df  = period_analysis(df_dated, "Late (2016–2024)",
                               LATE_PERIOD, CONSISTENT_PAIRS, species_list)
    all_periods = pd.concat([early_df, late_df])

    print(f"\n  {'Pair':<55} {'Species':<20} "
          f"{'Early OR':>10} {'Late OR':>10} {'Stable?':>10}")
    print(f"  {'─'*55} {'─'*20} {'─'*10} {'─'*10} {'─'*10}")

    stability_results = []
    for a, b in CONSISTENT_PAIRS:
        pair_data = all_periods[(all_periods["antibiotic_1"] == a) &
                                (all_periods["antibiotic_2"] == b)]
        for sp in species_list:
            e_row = pair_data[(pair_data["period"] == "Early (2005–2015)") &
                              (pair_data["species"] == sp)]
            l_row = pair_data[(pair_data["period"] == "Late (2016–2024)") &
                              (pair_data["species"] == sp)]
            e_or = e_row["OR"].values[0] if len(e_row) > 0 else np.nan
            l_or = l_row["OR"].values[0] if len(l_row) > 0 else np.nan

            if (not np.isnan(e_or) and not np.isnan(l_or)
                    and e_or > 0 and l_or > 0):
                ratio  = max(e_or, l_or) / min(e_or, l_or)
                stable = "✓" if ratio < 3 and e_or > 1 and l_or > 1 else "⚠"
            else:
                ratio  = np.nan
                stable = "?"

            e_str = f"{e_or:.3f}" if not np.isnan(e_or) else "N/A"
            l_str = f"{l_or:.3f}" if not np.isnan(l_or) else "N/A"
            print(f"  {a} × {b:<{54-len(a)}} {sp:<20} "
                  f"{e_str:>10} {l_str:>10} {stable:>10}")

            stability_results.append({
                "antibiotic_1": a, "antibiotic_2": b, "species": sp,
                "OR_early": e_or, "OR_late": l_or,
                "OR_ratio_period": round(ratio, 3)
                    if not np.isnan(ratio) else None,
                "temporally_stable": stable == "✓",
            })

    stability_df = pd.DataFrame(stability_results)

    print(f"\n  Temporal stability summary:")
    for a, b in CONSISTENT_PAIRS:
        sub = stability_df[(stability_df["antibiotic_1"] == a) &
                           (stability_df["antibiotic_2"] == b)]
        n_stable = sub["temporally_stable"].sum()
        n_total  = sub.dropna(subset=["OR_early", "OR_late"]).shape[0]
        print(f"    {a} × {b}: {n_stable}/{n_total} temporally stable")

    # ── Year-by-year ──────────────────────────────────────────────
    divider("Year-by-year OR trajectories")

    yearly_results = {}
    for a, b in CONSISTENT_PAIRS:
        pair_label = f"{a} × {b}"
        print(f"\n  {pair_label}")
        yearly = yearly_or(df_dated, a, b, species_list)
        yearly_results[(a, b)] = yearly
        if len(yearly) > 0:
            print(yearly[["year", "species", "OR", "n"]].to_string(index=False))
        else:
            print("  (no data above MIN_PAIRS threshold)")

    # ── Resistance rate trends ────────────────────────────────────
    divider("Resistance rate trends")

    for ab in ["ceftazidime", "ciprofloxacin", "gentamicin", "amikacin"]:
        if ab not in df_dated.columns:
            continue
        print(f"\n  {ab}:")
        for sp in species_list:
            sp_sub = df_dated[df_dated["species"] == sp]
            yr_rates = (
                sp_sub.groupby("collection_year")[ab]
                .agg(lambda x: x.mean() if x.notna().sum() >= MIN_PAIRS
                     else np.nan)
                .dropna()
            )
            if len(yr_rates) > 0:
                print(f"    {sp}:")
                print(yr_rates.round(3).to_string())

    # ── Plots ─────────────────────────────────────────────────────
    divider("Generating visualisations")

    for (a, b), yearly in yearly_results.items():
        pair_label = f"{a} x {b}"
        safe_label = pair_label.replace("/", "_").replace(" ", "_")
        plot_or_trajectory(
            yearly, pair_label,
            os.path.join(OUTPUT_FOLDER, f"or_trajectory_{safe_label}.png")
        )

    plot_period_comparison(
        early_df, late_df, CONSISTENT_PAIRS,
        os.path.join(OUTPUT_FOLDER, "period_comparison.png")
    )

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    for ax, ab in zip(axes, ["ceftazidime", "ciprofloxacin",
                              "gentamicin", "amikacin"]):
        if ab not in df_dated.columns:
            continue
        for sp in species_list:
            sp_sub = df_dated[df_dated["species"] == sp]
            yr_rates = (
                sp_sub.groupby("collection_year")[ab]
                .agg(lambda x: x.mean() if x.notna().sum() >= MIN_PAIRS
                     else np.nan)
                .dropna()
            )
            if len(yr_rates) > 1:
                ax.plot(yr_rates.index, yr_rates.values * 100,
                        marker="o", markersize=4, linewidth=2,
                        label=sp, color=COLORS.get(sp, "gray"))
        ax.set_title(ab, fontweight="bold")
        ax.set_xlabel("Year")
        ax.set_ylabel("Resistance rate (%)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)

    plt.suptitle("Resistance rate trends 2005–2024",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, "resistance_rate_trends.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ resistance_rate_trends.png")

    # ── Save outputs ──────────────────────────────────────────────
    divider("Saving outputs")

    stability_df.to_csv(
        os.path.join(OUTPUT_FOLDER, "temporal_stability.csv"), index=False)
    all_periods.to_csv(
        os.path.join(OUTPUT_FOLDER, "period_or_results.csv"), index=False)
    for (a, b), yearly in yearly_results.items():
        safe = f"{a}_{b}".replace("/", "_").replace(" ", "_")
        yearly.to_csv(
            os.path.join(OUTPUT_FOLDER, f"yearly_or_{safe}.csv"), index=False)

    print(f"  ✓ All outputs saved to {OUTPUT_FOLDER}")

    # ── Summary ───────────────────────────────────────────────────
    divider("FINAL SUMMARY")

    n_stable = stability_df["temporally_stable"].sum()
    n_total  = stability_df["temporally_stable"].notna().sum()

    print(f"""
  TEMPORAL STABILITY RESULTS

  Consistent pairs tested          : {len(CONSISTENT_PAIRS)}
  Species tested per pair          : {len(species_list)}
  Total species-pair combinations  : {n_total}
  Temporally stable (OR>1, <3-fold): {n_stable}/{n_total}
""")

    divider()
    print("  Temporal analysis complete.")
    divider()


if __name__ == "__main__":
    main()
