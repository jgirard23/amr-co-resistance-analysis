"""
NCBI Pathogen Detection — Multi-Species Quality Check Script (BV-BRC version)
==============================================================================
Handles BV-BRC long-format antibiogram data and pivots to wide format.
Set FOLDER_PATH below, then run:
    python3 ncbi_pathogen_qc_all.py
"""

import pandas as pd
import numpy as np
import os
import sys

# ─────────────────────────────────────────────────────────────────
# CONFIGURATION — change this to your folder path
# ─────────────────────────────────────────────────────────────────
FOLDER_PATH = "/Users/jacobgirard-beaupre/Downloads/NCBI data"

FILES = {
    "A. baumannii":  "A. baumannii.csv",
    "P. aeruginosa": "P. aeruginosa.csv",
    "E. coli":       "E. coli.csv",
    "K. pneumoniae": "K. pneumoniae.csv",
}

# BV-BRC column names
COL_GENOME_ID   = "Genome ID"
COL_GENOME_NAME = "Genome Name"
COL_ANTIBIOTIC  = "Antibiotic"
COL_PHENOTYPE   = "Resistant Phenotype"
COL_MIC_VALUE   = "Measurement Value"
COL_MIC_SIGN    = "Measurement Sign"
COL_MIC_UNIT    = "Measurement Unit"
COL_STANDARD    = "Testing Standard"
COL_YEAR        = "Testing Standard Year"
COL_EVIDENCE    = "Evidence"
COL_METHOD      = "Laboratory Typing Method"

# Only keep real lab measurements
LAB_METHODS = {"agar dilution", "broth dilution", "disk diffusion", "mic"}

# EUCAST intrinsic resistance per species
EUCAST_INTRINSIC = {
    "A. baumannii": [
        "ampicillin", "amoxicillin-clavulanate", "amoxicillin/clavulanic",
        "trimethoprim", "cefotaxime", "ceftriaxone",
        "chloramphenicol", "fosfomycin",
    ],
    "P. aeruginosa": [
        "ampicillin", "amoxicillin-clavulanate", "amoxicillin/clavulanic",
        "cefotaxime", "ceftriaxone", "ertapenem",
        "chloramphenicol", "trimethoprim", "fosfomycin",
        "tetracycline", "doxycycline",
    ],
    "E. coli":       ["ampicillin"],
    "K. pneumoniae": ["ampicillin"],
}


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
            df = pd.read_csv(filepath, sep=sep, low_memory=False, on_bad_lines="skip")
            if df.shape[1] > 1:
                return df, ("COMMA" if sep == "," else "TAB")
        except Exception:
            continue
    raise ValueError(f"Could not parse: {filepath}")


def filter_lab_only(df):
    if COL_METHOD not in df.columns:
        return df
    mask = df[COL_METHOD].fillna("").str.strip().str.lower().isin(LAB_METHODS)
    return df[mask].copy()


def pivot_to_wide(df):
    if COL_GENOME_ID not in df.columns or COL_ANTIBIOTIC not in df.columns:
        return None

    df = df.copy()
    df[COL_ANTIBIOTIC] = df[COL_ANTIBIOTIC].str.strip().str.lower()

    df_dedup = (
        df.dropna(subset=[COL_PHENOTYPE])
          .drop_duplicates(subset=[COL_GENOME_ID, COL_ANTIBIOTIC], keep="first")
    )

    wide = df_dedup.pivot(
        index=COL_GENOME_ID,
        columns=COL_ANTIBIOTIC,
        values=COL_PHENOTYPE
    ).reset_index()

    wide.columns.name = None

    if COL_GENOME_NAME in df.columns:
        names = df.drop_duplicates(COL_GENOME_ID)[[COL_GENOME_ID, COL_GENOME_NAME]]
        wide = wide.merge(names, on=COL_GENOME_ID, how="left")

    if COL_STANDARD in df.columns:
        bp = (
            df.groupby(COL_GENOME_ID)[COL_STANDARD]
              .agg(lambda x: x.dropna().mode()[0] if x.dropna().shape[0] > 0 else np.nan)
              .reset_index()
        )
        wide = wide.merge(bp, on=COL_GENOME_ID, how="left")

    return wide


def get_antibiotic_cols(wide_df):
    non_ab = {COL_GENOME_ID, COL_GENOME_NAME, COL_STANDARD, COL_YEAR, "Taxon ID"}
    return [c for c in wide_df.columns if c not in non_ab]


# ═════════════════════════════════════════════════════════════════
def analyse_species(label, filepath):
    divider(f"FILE: {label}")

    if not os.path.exists(filepath):
        print(f"  ⚠ FILE NOT FOUND: {filepath}")
        return None

    df_raw, sep_label = load_csv(filepath)
    print(f"\n  Separator    : {sep_label}")
    print(f"  Raw shape    : {df_raw.shape[0]:,} rows × {df_raw.shape[1]} columns")
    print(f"  Columns      : {list(df_raw.columns)}")

    df_lab = filter_lab_only(df_raw)
    n_removed = len(df_raw) - len(df_lab)
    print(f"\n  After filtering to lab measurements only:")
    print(f"    Kept    : {len(df_lab):,} rows")
    print(f"    Removed : {n_removed:,} computational predictions")

    if len(df_lab) == 0:
        print("  ⚠ No laboratory measurements found after filtering.")
        return None

    if COL_STANDARD in df_lab.columns:
        bp_dist = df_lab[COL_STANDARD].value_counts(dropna=False)
        print(f"\n  Breakpoint standard distribution:")
        print(bp_dist.to_string())

    if COL_ANTIBIOTIC in df_lab.columns:
        ab_counts = df_lab[COL_ANTIBIOTIC].str.lower().value_counts()
        print(f"\n  Unique antibiotics tested ({len(ab_counts)}):")
        print(ab_counts.to_string())

    print(f"\n  Pivoting to wide format (one row per isolate) …")
    wide = pivot_to_wide(df_lab)

    if wide is None:
        print("  ⚠ Could not pivot — missing Genome ID or Antibiotic column.")
        return None

    ab_cols = get_antibiotic_cols(wide)
    print(f"  Wide shape   : {wide.shape[0]:,} isolates × {len(ab_cols)} antibiotics")

    tested = wide[ab_cols].notna().sum(axis=1)
    print(f"\n  Antibiotics tested per isolate:")
    print(f"    Min {tested.min()}  Max {tested.max()}  "
          f"Mean {tested.mean():.1f}  Median {tested.median():.0f}")

    dist = tested.value_counts().sort_index()
    print(f"\n  Distribution (# antibiotics → # isolates):")
    print(dist.to_string())

    pct5  = (tested >= 5).mean()  * 100
    pct10 = (tested >= 10).mean() * 100
    print(f"\n  Isolates with ≥5  antibiotics tested : {pct5:.1f}%")
    print(f"  Isolates with ≥10 antibiotics tested : {pct10:.1f}%")

    sample_vals = wide[ab_cols].stack().dropna().astype(str).str.upper().unique()[:20]
    print(f"\n  Unique resistance values found: {sorted(sample_vals)}")

    missing_pct = wide[ab_cols].isnull().mean().sort_values(ascending=False) * 100
    print(f"\n  Top 15 antibiotics by % missing:")
    print(missing_pct.head(15).apply(lambda x: f"{x:.1f}%").to_string())

    intrinsic = EUCAST_INTRINSIC.get(label, [])
    hits = [c for c in ab_cols if any(k in c.lower() for k in intrinsic)]
    print(f"\n  EUCAST intrinsic resistance antibiotics present ({len(hits)}):")
    print(f"    {hits if hits else 'none matched'}")
    if hits:
        print(f"  ⚠ These must be masked before co-resistance module analysis.")

    return {
        "label":      label,
        "wide":       wide,
        "ab_cols":    ab_cols,
        "pct5":       pct5,
        "n_isolates": len(wide),
    }


# ═════════════════════════════════════════════════════════════════
def main():

    if not os.path.exists(FOLDER_PATH):
        print(f"ERROR: Folder not found → {FOLDER_PATH}")
        print("Edit FOLDER_PATH at the top of the script and re-run.")
        sys.exit(1)

    print("=" * 72)
    print("  NCBI PATHOGEN DETECTION — MULTI-SPECIES QUALITY CHECK (BV-BRC)")
    print("=" * 72)

    results = {}
    for label, filename in FILES.items():
        filepath = os.path.join(FOLDER_PATH, filename)
        r = analyse_species(label, filepath)
        if r is not None:
            results[label] = r

    # ── Cross-species antibiotic panel overlap ────────────────────
    divider("CROSS-SPECIES ANTIBIOTIC PANEL OVERLAP (CRITICAL)")

    coverage_threshold = 0.50
    species_ab_sets = {}

    for label, r in results.items():
        wide    = r["wide"]
        ab_cols = r["ab_cols"]
        covered = [c for c in ab_cols if wide[c].notna().mean() >= coverage_threshold]
        species_ab_sets[label] = set(covered)
        print(f"\n  {label} — {len(covered)} antibiotics with ≥50% coverage:")
        print(f"    {sorted(covered)}")

    if len(species_ab_sets) >= 2:
        common = set.intersection(*species_ab_sets.values())
        print(f"\n  ★ INTERSECTION across ALL {len(species_ab_sets)} species "
              f"({len(common)} antibiotics):")
        print(f"    {sorted(common)}")

        if len(common) == 0:
            print("  ⚠ No shared antibiotics at 50% threshold.")
            print("  Checking at 20% threshold …")
            species_ab_sets_20 = {}
            for label, r in results.items():
                wide    = r["wide"]
                ab_cols = r["ab_cols"]
                covered20 = [c for c in ab_cols if wide[c].notna().mean() >= 0.20]
                species_ab_sets_20[label] = set(covered20)
            common20 = set.intersection(*species_ab_sets_20.values())
            print(f"  At 20% threshold: {len(common20)} shared antibiotics:")
            print(f"    {sorted(common20)}")
        elif len(common) < 5:
            print("  ⚠ WARNING: Fewer than 5 shared antibiotics.")
        else:
            print("  ✓ Sufficient shared panel for cross-species analysis.")

        print(f"\n  Pairwise overlaps (≥50% coverage threshold):")
        labels = list(species_ab_sets.keys())
        for i in range(len(labels)):
            for j in range(i+1, len(labels)):
                a, b = labels[i], labels[j]
                overlap = species_ab_sets[a] & species_ab_sets[b]
                print(f"    {a} ∩ {b}: {len(overlap)} → {sorted(overlap)}")

    # ── Geographic coverage note ──────────────────────────────────
    divider("GEOGRAPHIC COVERAGE NOTE")
    print("  BV-BRC antibiogram files do not contain country information.")
    print("  Geographic data will be joined from your original NCBI metadata")
    print("  files using Genome ID as the linking key in the next step.")
    print("  Your original NCBI files confirmed coverage across all 6 continents.")

    # ── Final YES/NO summary ──────────────────────────────────────
    divider("FINAL YES/NO ASSESSMENT")

    n_species = len(results)
    q1 = "YES" if n_species >= 2 else "NO"

    avg_pct5 = np.mean([r["pct5"] for r in results.values()]) if results else 0
    q2 = "YES" if avg_pct5 >= 50 else "NO"
    q2_detail = f"Average {avg_pct5:.1f}% of isolates have ≥5 antibiotics tested"

    q3 = "YES (confirmed from NCBI metadata files — join needed)"

    bp_found = any(COL_STANDARD in r["wide"].columns for r in results.values())
    q4 = "YES" if bp_found else "NO — harmonisation must be done blindly"

    if len(species_ab_sets) >= 2:
        shared = set.intersection(*species_ab_sets.values())
        q5 = "YES" if len(shared) >= 5 else f"NO — only {len(shared)} shared antibiotics at 50% threshold"
    else:
        q5 = "UNCERTAIN"

    print(f"""
  ┌─────────────────────────────────────────────────────────────────────┐
  │  Q1  Multi-species coverage for cross-species analysis?
  │      {q1} — {n_species} species loaded
  │
  │  Q2  Per-isolate antibiotic coverage ≥5 drugs for co-resistance?
  │      {q2} — {q2_detail}
  │
  │  Q3  Multi-continental geographic coverage?
  │      {q3}
  │
  │  Q4  Breakpoint standard recorded (CLSI/EUCAST)?
  │      {q4}
  │
  │  Q5  Sufficient shared antibiotic panel across all species?
  │      {q5}
  └─────────────────────────────────────────────────────────────────────┘
""")

    divider("NEXT STEP RECOMMENDATION")

    if avg_pct5 < 50:
        print("  Per-isolate coverage is below 50% for ≥5 antibiotics.")
        print("  Check the antibiotic distribution table above for each species.")
        print("  If most isolates were only tested against 1-3 antibiotics,")
        print("  co-resistance profiling will not be possible without additional data.")

    if len(species_ab_sets) >= 2:
        shared = set.intersection(*species_ab_sets.values())
        if len(shared) >= 5:
            print("  ✓ Proceed to join BV-BRC antibiogram data with NCBI metadata")
            print("  on Genome ID to attach country and date to each isolate.")
            print("  Then filter to CLSI-only or EUCAST-only for harmonisation.")
        else:
            print("  Review the antibiotic lists per species above.")
            print("  Antibiotic name differences across species may be reducing overlap.")

    divider()
    print("  Quality check complete.")
    divider()


if __name__ == "__main__":
    main()