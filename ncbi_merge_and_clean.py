"""
NCBI Pathogen Detection — Merge, Clean, and Join Script
========================================================
Combines BV-BRC antibiogram data with NCBI metadata,
produces two analysis-ready files:
  - primary_analysis.csv   (E. coli, K. pneumoniae, A. baumannii)
  - secondary_analysis.csv (P. aeruginosa)

Run:
    python3 ncbi_merge_and_clean.py
"""

import pandas as pd
import numpy as np
import os
import sys

# ─────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────
BVBRC_FOLDER = "/Users/jacobgirard-beaupre/Downloads/NCBI data"
NCBI_FOLDER  = "/Users/jacobgirard-beaupre/Documents/NCBI old data"
OUTPUT_FOLDER = "/Users/jacobgirard-beaupre/Downloads/NCBI data"

BVBRC_FILES = {
    "A. baumannii":  "A. baumannii.csv",
    "P. aeruginosa": "P. aeruginosa.csv",
    "E. coli":       "E. coli.csv",
    "K. pneumoniae": "K. pneumoniae.csv",
}

NCBI_FILES = {
    "A. baumannii":  "A. baumannii",
    "P. aeruginosa": "P. aeruginosa",
    "E. coli":       "E.coli + Shigella",
    "K. pneumoniae": "K. pneumoniae",
}

# Analysis panels
PRIMARY_SPECIES = ["E. coli", "K. pneumoniae", "A. baumannii"]
PRIMARY_PANEL = [
    "ampicillin",
    "ceftazidime",
    "ciprofloxacin",
    "gentamicin",
    "trimethoprim/sulfamethoxazole",
    "aztreonam",
    "amikacin",
]

SECONDARY_SPECIES = ["P. aeruginosa"]
SECONDARY_PANEL = [
    "meropenem",
    "ciprofloxacin",
    "ceftazidime",
]

# BV-BRC column names
COL_GENOME_ID   = "Genome ID"
COL_GENOME_NAME = "Genome Name"
COL_ANTIBIOTIC  = "Antibiotic"
COL_PHENOTYPE   = "Resistant Phenotype"
COL_STANDARD    = "Testing Standard"
COL_METHOD      = "Laboratory Typing Method"

# NCBI metadata column names
NCBI_BIOSAMPLE  = "BioSample"
NCBI_COUNTRY    = "Location"
NCBI_DATE       = "Create date"
NCBI_ORGANISM   = "#Organism group"

# Lab methods to keep
LAB_METHODS = {"agar dilution", "broth dilution", "disk diffusion", "mic"}

# EUCAST intrinsic resistances to flag per species
EUCAST_INTRINSIC = {
    "A. baumannii": [
        "ampicillin", "amoxicillin/clavulanic acid", "amoxicillin-clavulanate",
        "trimethoprim", "cefotaxime", "ceftriaxone",
        "chloramphenicol", "fosfomycin", "ampicillin/sulbactam",
        "trimethoprim/sulfamethoxazole",
    ],
    "P. aeruginosa": [
        "ampicillin", "amoxicillin/clavulanic acid", "amoxicillin-clavulanate",
        "cefotaxime", "ceftriaxone", "ertapenem",
        "chloramphenicol", "trimethoprim", "fosfomycin",
        "tetracycline", "doxycycline", "trimethoprim/sulfamethoxazole",
        "ampicillin/sulbactam",
    ],
    "E. coli":       ["ampicillin", "ampicillin/sulbactam"],
    "K. pneumoniae": ["ampicillin", "ampicillin/sulbactam"],
}


# ═════════════════════════════════════════════════════════════════
def divider(title=""):
    width = 72
    if title:
        pad = max(0, width - len(title) - 4)
        print(f"\n{'─'*2} {title} {'─'*pad}")
    else:
        print("─" * width)


def load_file(filepath):
    for sep in [",", "\t"]:
        try:
            df = pd.read_csv(filepath, sep=sep, low_memory=False, on_bad_lines="skip")
            if df.shape[1] > 1:
                return df
        except Exception:
            continue
    raise ValueError(f"Could not parse: {filepath}")


def find_file(folder, name):
    """Try exact name, then with .tsv and .csv extensions."""
    for candidate in [name, name + ".csv", name + ".tsv", name + ".txt"]:
        path = os.path.join(folder, candidate)
        if os.path.exists(path):
            return path
    return None


def normalise_breakpoint(val):
    """Standardise breakpoint standard values to CLSI, EUCAST, MIXED, or UNKNOWN."""
    if pd.isna(val):
        return "UNKNOWN"
    v = str(val).strip().upper()
    has_clsi   = "CLSI" in v or "NARMS" in v
    has_eucast = "EUCAST" in v or "SFM" in v or "BSAC" in v
    if has_clsi and has_eucast:
        return "MIXED"
    elif has_clsi:
        return "CLSI"
    elif has_eucast:
        return "EUCAST"
    else:
        return "UNKNOWN"


def filter_lab_only(df):
    if COL_METHOD not in df.columns:
        return df
    mask = df[COL_METHOD].fillna("").str.strip().str.lower().isin(LAB_METHODS)
    return df[mask].copy()


def pivot_to_wide(df, species_label):
    """Pivot long format to wide, one row per isolate."""
    df = df.copy()
    df[COL_ANTIBIOTIC] = df[COL_ANTIBIOTIC].str.strip().str.lower()

    # Normalise breakpoint per row then aggregate per isolate
    df["bp_norm"] = df[COL_STANDARD].apply(normalise_breakpoint)

    # Deduplicate: one result per isolate-antibiotic pair
    df_dedup = (
        df.dropna(subset=[COL_PHENOTYPE])
          .drop_duplicates(subset=[COL_GENOME_ID, COL_ANTIBIOTIC], keep="first")
    )

    # Pivot
    wide = df_dedup.pivot(
        index=COL_GENOME_ID,
        columns=COL_ANTIBIOTIC,
        values=COL_PHENOTYPE
    ).reset_index()
    wide.columns.name = None

    # Attach genome name
    if COL_GENOME_NAME in df.columns:
        names = df.drop_duplicates(COL_GENOME_ID)[[COL_GENOME_ID, COL_GENOME_NAME]]
        wide = wide.merge(names, on=COL_GENOME_ID, how="left")

    # Attach normalised breakpoint standard (most common per isolate)
    bp_agg = (
        df.groupby(COL_GENOME_ID)["bp_norm"]
          .agg(lambda x: x.mode()[0] if len(x) > 0 else "UNKNOWN")
          .reset_index()
          .rename(columns={"bp_norm": "breakpoint_standard"})
    )
    wide = wide.merge(bp_agg, on=COL_GENOME_ID, how="left")

    # Add species column
    wide["species"] = species_label

    return wide


def extract_year_series(series):
    """Vectorised year extraction from a date series."""
    return series.astype(str).str.extract(r'(\d{4})')[0].apply(
        lambda x: int(x) if pd.notna(x) and x != 'nan' else np.nan
    )


def load_ncbi_metadata(species_label):
    """Load original NCBI metadata file and return relevant columns."""
    name = NCBI_FILES.get(species_label)
    if name is None:
        return None
    path = find_file(NCBI_FOLDER, name)
    if path is None:
        print(f"  ⚠ NCBI metadata not found for {species_label}")
        return None
    df = load_file(path)

    # Keep only useful columns
    keep = []
    for col in [NCBI_BIOSAMPLE, NCBI_COUNTRY, NCBI_DATE, NCBI_ORGANISM, "Assembly"]:
        if col in df.columns:
            keep.append(col)
    df = df[keep].copy()

    # Extract year
    if NCBI_DATE in df.columns:
        df["collection_year"] = extract_year_series(df[NCBI_DATE])

    return df


# ═════════════════════════════════════════════════════════════════
def main():

    # Validate folders
    for folder, name in [(BVBRC_FOLDER, "BV-BRC"), (NCBI_FOLDER, "NCBI metadata")]:
        if not os.path.exists(folder):
            print(f"ERROR: {name} folder not found → {folder}")
            sys.exit(1)

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    print("=" * 72)
    print("  NCBI PATHOGEN — MERGE, CLEAN, AND JOIN")
    print("=" * 72)

    all_wide = {}

    # ── Process each species ──────────────────────────────────────
    for species, filename in BVBRC_FILES.items():
        divider(f"Processing {species}")

        bvbrc_path = os.path.join(BVBRC_FOLDER, filename)
        if not os.path.exists(bvbrc_path):
            print(f"  ⚠ BV-BRC file not found: {bvbrc_path}")
            continue

        # Load and filter BV-BRC
        df_raw = load_file(bvbrc_path)
        df_lab = filter_lab_only(df_raw)
        print(f"  BV-BRC rows (lab only): {len(df_lab):,}")

        # Pivot to wide
        wide = pivot_to_wide(df_lab, species)
        print(f"  Isolates after pivot  : {wide.shape[0]:,}")

        # Load NCBI metadata
        meta = load_ncbi_metadata(species)

        # Join on Genome ID ↔ Assembly (best available key)
        if meta is not None:
            meta_renamed = meta.rename(columns={
                NCBI_COUNTRY:   "country",
                NCBI_DATE:      "collection_date",
                NCBI_ORGANISM:  "organism_group",
                NCBI_BIOSAMPLE: "biosample_id",
            })

            # Print sample keys to diagnose join
            print(f"  Sample Genome IDs (BV-BRC): {list(wide[COL_GENOME_ID].head(3))}")
            if "Assembly" in meta_renamed.columns:
                print(f"  Sample Assembly IDs (NCBI): {list(meta_renamed['Assembly'].dropna().head(3))}")
            if "biosample_id" in meta_renamed.columns:
                print(f"  Sample BioSample IDs (NCBI): {list(meta_renamed['biosample_id'].dropna().head(3))}")

            merged = wide.copy()
            merged["country"] = np.nan
            merged["collection_year"] = np.nan
            merged["collection_date"] = np.nan

            # Try joining on Assembly
            if "Assembly" in meta_renamed.columns:
                wide["genome_id_clean"] = wide[COL_GENOME_ID].astype(str).str.strip()
                meta_renamed["assembly_clean"] = meta_renamed["Assembly"].astype(str).str.strip()
                test_merge = wide.merge(
                    meta_renamed[["assembly_clean", "country", "collection_year", "collection_date", "biosample_id"]].dropna(subset=["assembly_clean"]),
                    left_on="genome_id_clean",
                    right_on="assembly_clean",
                    how="left"
                )
                n_matched = test_merge["country"].notna().sum()
                print(f"  Join on Assembly: {n_matched:,} / {len(test_merge):,} matched")
                if n_matched > 0:
                    merged = test_merge
                    
            n_matched = merged["country"].notna().sum()
            print(f"  Isolates with country : {n_matched:,} / {len(merged):,} "
                  f"({n_matched/len(merged)*100:.1f}%)")
        else:
            merged = wide.copy()
            merged["country"] = np.nan
            merged["collection_year"] = np.nan
            print(f"  ⚠ Could not join metadata — country will be NaN")

        all_wide[species] = merged

    # ── Build primary dataset ─────────────────────────────────────
    divider("Building PRIMARY dataset (E. coli, K. pneumoniae, A. baumannii)")

    primary_frames = []
    for sp in PRIMARY_SPECIES:
        if sp in all_wide:
            primary_frames.append(all_wide[sp])

    if primary_frames:
        primary = pd.concat(primary_frames, axis=0, ignore_index=True)

        # Keep only panel antibiotics + metadata columns
        meta_cols = [
            COL_GENOME_ID, COL_GENOME_NAME, "species",
            "breakpoint_standard", "country", "collection_year",
            "collection_date", "biosample_id", "organism_group"
        ]
        meta_cols = [c for c in meta_cols if c in primary.columns]
        panel_cols_present = [c for c in PRIMARY_PANEL if c in primary.columns]
        missing_panel = [c for c in PRIMARY_PANEL if c not in primary.columns]

        if missing_panel:
            print(f"  ⚠ Panel antibiotics not found in data: {missing_panel}")

        primary_out = primary[meta_cols + panel_cols_present].copy()

        # Add EUCAST intrinsic flag columns
        for sp in PRIMARY_SPECIES:
            intrinsic = EUCAST_INTRINSIC.get(sp, [])
            for ab in panel_cols_present:
                if any(k in ab.lower() for k in intrinsic):
                    flag_col = f"intrinsic_{ab.replace('/', '_').replace(' ', '_')}"
                    if flag_col not in primary_out.columns:
                        primary_out[flag_col] = (
                            primary_out["species"] == sp
                        ).astype(int)

        # Summary
        print(f"\n  Primary dataset shape : {primary_out.shape[0]:,} isolates × "
              f"{primary_out.shape[1]} columns")
        print(f"  Panel antibiotics     : {panel_cols_present}")
        print(f"\n  Isolates per species:")
        print(primary_out["species"].value_counts().to_string())
        print(f"\n  Breakpoint standard distribution:")
        print(primary_out["breakpoint_standard"].value_counts().to_string())

        # Complete panel profiles
        complete = primary_out[panel_cols_present].notna().all(axis=1).sum()
        print(f"\n  Isolates with COMPLETE panel profile : {complete:,} "
              f"({complete/len(primary_out)*100:.1f}%)")

        # Geographic distribution
        if "country" in primary_out.columns:
            n_with_country = primary_out["country"].notna().sum()
            print(f"  Isolates with country data          : {n_with_country:,} "
                  f"({n_with_country/len(primary_out)*100:.1f}%)")
            if n_with_country > 0:
                print(f"\n  Top 15 countries:")
                print(primary_out["country"].value_counts().head(15).to_string())

        # Temporal distribution
        if "collection_year" in primary_out.columns:
            n_with_year = primary_out["collection_year"].notna().sum()
            if n_with_year > 0:
                print(f"\n  Year distribution:")
                print(primary_out["collection_year"].value_counts()
                      .sort_index().to_string())

        # Save
        out_path = os.path.join(OUTPUT_FOLDER, "primary_analysis.csv")
        primary_out.to_csv(out_path, index=False)
        print(f"\n  ✓ Saved → {out_path}")

    # ── Build secondary dataset ───────────────────────────────────
    divider("Building SECONDARY dataset (P. aeruginosa)")

    secondary_frames = []
    for sp in SECONDARY_SPECIES:
        if sp in all_wide:
            secondary_frames.append(all_wide[sp])

    if secondary_frames:
        secondary = pd.concat(secondary_frames, axis=0, ignore_index=True)

        meta_cols = [
            COL_GENOME_ID, COL_GENOME_NAME, "species",
            "breakpoint_standard", "country", "collection_year",
            "collection_date", "biosample_id", "organism_group"
        ]
        meta_cols = [c for c in meta_cols if c in secondary.columns]
        panel_cols_present = [c for c in SECONDARY_PANEL if c in secondary.columns]
        missing_panel = [c for c in SECONDARY_PANEL if c not in secondary.columns]

        if missing_panel:
            print(f"  ⚠ Panel antibiotics not found in data: {missing_panel}")

        secondary_out = secondary[meta_cols + panel_cols_present].copy()

        print(f"\n  Secondary dataset shape : {secondary_out.shape[0]:,} isolates × "
              f"{secondary_out.shape[1]} columns")
        print(f"  Panel antibiotics       : {panel_cols_present}")
        print(f"\n  Breakpoint standard distribution:")
        print(secondary_out["breakpoint_standard"].value_counts().to_string())

        complete = secondary_out[panel_cols_present].notna().all(axis=1).sum()
        print(f"\n  Isolates with COMPLETE panel profile : {complete:,} "
              f"({complete/len(secondary_out)*100:.1f}%)")

        if "country" in secondary_out.columns:
            n_with_country = secondary_out["country"].notna().sum()
            print(f"  Isolates with country data          : {n_with_country:,} "
                  f"({n_with_country/len(secondary_out)*100:.1f}%)")

        out_path = os.path.join(OUTPUT_FOLDER, "secondary_analysis.csv")
        secondary_out.to_csv(out_path, index=False)
        print(f"\n  ✓ Saved → {out_path}")

    # ── Final summary ─────────────────────────────────────────────
    divider("FINAL SUMMARY")

    print(f"""
  PRIMARY ANALYSIS
  ────────────────
  Species  : E. coli, K. pneumoniae, A. baumannii
  Panel    : {PRIMARY_PANEL}
  File     : primary_analysis.csv

  SECONDARY ANALYSIS
  ──────────────────
  Species  : P. aeruginosa
  Panel    : {SECONDARY_PANEL}
  File     : secondary_analysis.csv

  NEXT STEP
  ─────────
  Both files are now ready for resistance module analysis.
  The breakpoint_standard column flags each isolate as
  CLSI, EUCAST, MIXED, or UNKNOWN — include this as a
  covariate in all downstream statistical models.

  Intrinsic resistance flag columns (intrinsic_*) mark
  species-level expected resistances that must be excluded
  from acquired co-resistance module detection.
""")

    divider()
    print("  Merge and clean complete.")
    divider()


if __name__ == "__main__":
    main()