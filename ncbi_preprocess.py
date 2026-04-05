"""
AMR Analysis — Layer 1 + Layer 2 Preprocessing
================================================
Layer 1: EUCAST Expected Phenotype Removal
Layer 2: Breakpoint Harmonisation + Unknown Characterisation
Also: Metadata join using BV-BRC genome metadata API files

Run:
    python3 ncbi_preprocess.py
"""

import pandas as pd
import numpy as np
import os
import sys
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────
DATA_FOLDER   = "/Users/jacobgirard-beaupre/Downloads/NCBI data"
OUTPUT_FOLDER = "/Users/jacobgirard-beaupre/Downloads/NCBI data/results"

PRIMARY_FILE   = "primary_analysis.csv"
SECONDARY_FILE = "secondary_analysis.csv"

METADATA_FILES = {
    "A. baumannii":  "abaumannii_genome_metadata.csv",
    "P. aeruginosa": "paeruginosa_genome_metadata.csv",
    "E. coli":       "ecoli_genome_metadata.csv",
    "K. pneumoniae": "kpneumoniae_genome_metadata.csv",
}

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

# ─────────────────────────────────────────────────────────────────
# EUCAST INTRINSIC RESISTANCE TABLES
# Source: EUCAST Expected Resistant Phenotypes
# These resistances are species-level, not acquired
# ─────────────────────────────────────────────────────────────────
EUCAST_INTRINSIC = {
    "A. baumannii": [
        "ampicillin",
        "amoxicillin/clavulanic acid",
        "ampicillin/sulbactam",
        "cefotaxime",
        "ceftriaxone",
        "trimethoprim",
        "trimethoprim/sulfamethoxazole",
        "chloramphenicol",
        "fosfomycin",
    ],
    "P. aeruginosa": [
        "ampicillin",
        "amoxicillin/clavulanic acid",
        "ampicillin/sulbactam",
        "cefotaxime",
        "ceftriaxone",
        "ertapenem",
        "trimethoprim",
        "trimethoprim/sulfamethoxazole",
        "chloramphenicol",
        "fosfomycin",
        "tetracycline",
        "doxycycline",
    ],
    "E. coli": [
        "ampicillin",
        "ampicillin/sulbactam",
    ],
    "K. pneumoniae": [
        "ampicillin",
        "ampicillin/sulbactam",
    ],
}

# Country to continent mapping
COUNTRY_TO_CONTINENT = {
    # North America
    "usa": "N. America", "united states": "N. America",
    "canada": "N. America", "mexico": "N. America",
    # South America
    "brazil": "S. America", "argentina": "S. America",
    "colombia": "S. America", "peru": "S. America",
    "chile": "S. America", "venezuela": "S. America",
    # Europe
    "germany": "Europe", "france": "Europe", "italy": "Europe",
    "spain": "Europe", "netherlands": "Europe", "sweden": "Europe",
    "denmark": "Europe", "norway": "Europe", "belgium": "Europe",
    "united kingdom": "Europe", "uk": "Europe", "switzerland": "Europe",
    "austria": "Europe", "portugal": "Europe", "greece": "Europe",
    "poland": "Europe", "czech republic": "Europe", "finland": "Europe",
    "romania": "Europe", "hungary": "Europe", "croatia": "Europe",
    "serbia": "Europe", "turkey": "Europe",
    # Asia
    "china": "Asia", "india": "Asia", "japan": "Asia",
    "south korea": "Asia", "korea": "Asia", "thailand": "Asia",
    "vietnam": "Asia", "pakistan": "Asia", "bangladesh": "Asia",
    "singapore": "Asia", "taiwan": "Asia", "malaysia": "Asia",
    "indonesia": "Asia", "philippines": "Asia", "iran": "Asia",
    "saudi arabia": "Asia", "israel": "Asia", "iraq": "Asia",
    "jordan": "Asia", "lebanon": "Asia", "qatar": "Asia",
    "united arab emirates": "Asia", "kuwait": "Asia",
    # Africa
    "nigeria": "Africa", "south africa": "Africa", "kenya": "Africa",
    "ethiopia": "Africa", "ghana": "Africa", "egypt": "Africa",
    "senegal": "Africa", "tanzania": "Africa", "uganda": "Africa",
    "cameroon": "Africa", "mozambique": "Africa", "zambia": "Africa",
    "tunisia": "Africa", "morocco": "Africa", "algeria": "Africa",
    # Oceania
    "australia": "Oceania", "new zealand": "Oceania",
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
            df = pd.read_csv(filepath, sep=sep, low_memory=False,
                             on_bad_lines="skip")
            if df.shape[1] > 1:
                return df
        except Exception:
            continue
    raise ValueError(f"Could not parse: {filepath}")


def get_continent(country_str):
    if pd.isna(country_str):
        return "Unknown"
    c = str(country_str).lower().strip()
    for key, continent in COUNTRY_TO_CONTINENT.items():
        if key in c:
            return continent
    return "Other"


def normalise_breakpoint(val):
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
    return "UNKNOWN"


# ═════════════════════════════════════════════════════════════════
def join_metadata(df, species):
    """Join BV-BRC genome metadata to add geographic and temporal data."""
    meta_file = METADATA_FILES.get(species)
    if meta_file is None:
        return df

    meta_path = os.path.join(DATA_FOLDER, meta_file)
    if not os.path.exists(meta_path):
        print(f"  ⚠ Metadata file not found: {meta_path}")
        return df

    meta = load_csv(meta_path)

    # Clean genome_id for joining
    meta["genome_id"] = (
        meta["genome_id"].astype(str)
        .str.replace('"', '').str.strip()
    )
    df["Genome ID clean"] = (
        df["Genome ID"].astype(str)
        .str.replace('"', '').str.strip()
    )

    # Select metadata columns
    meta_cols = ["genome_id"]
    for col in ["geographic_location", "isolation_country", "collection_date"]:
        if col in meta.columns:
            meta_cols.append(col)

    meta_sub = meta[meta_cols].drop_duplicates("genome_id")

    merged = df.merge(
        meta_sub,
        left_on="Genome ID clean",
        right_on="genome_id",
        how="left"
    )

    # Consolidate country
    if "isolation_country" in merged.columns:
        merged["country"] = merged["isolation_country"].fillna(
            merged.get("geographic_location", pd.Series(dtype=str))
        )
    elif "geographic_location" in merged.columns:
        merged["country"] = merged["geographic_location"]
    else:
        merged["country"] = np.nan

    # Extract year from collection_date
    if "collection_date" in merged.columns:
        merged["collection_year"] = (
            merged["collection_date"].astype(str)
            .str.extract(r'(\d{4})')[0]
            .apply(lambda x: int(x) if pd.notna(x) and x != 'nan' else np.nan)
        )
    else:
        merged["collection_year"] = np.nan

    # Add continent
    merged["continent"] = merged["country"].apply(get_continent)

    n_matched = merged["country"].notna().sum()
    print(f"  Metadata join: {n_matched:,} / {len(merged):,} isolates with country "
          f"({n_matched/len(merged)*100:.1f}%)")

    return merged


def apply_eucast_masking(df, panel, species_col):
    """
    Layer 1: Flag intrinsic resistances as 'EXPECTED' instead of 'RESISTANT'.
    Creates new columns with suffix _masked where:
      - EXPECTED = intrinsic resistance for that species
      - RESISTANT = acquired resistance (informative)
      - SUSCEPTIBLE/INTERMEDIATE = as before
      - NaN = not tested
    """
    df = df.copy()

    for ab in panel:
        if ab not in df.columns:
            continue

        masked_col = ab + "_masked"
        df[masked_col] = df[ab].copy()

        for species, intrinsic_list in EUCAST_INTRINSIC.items():
            if ab.lower() in [i.lower() for i in intrinsic_list]:
                # Flag as EXPECTED for this species
                species_mask = df[species_col] == species
                resistant_mask = df[ab].astype(str).str.upper().isin(
                    ["RESISTANT", "NONSUSCEPTIBLE"]
                )
                df.loc[species_mask & resistant_mask, masked_col] = "EXPECTED"

    return df


def binarise_masked(df, panel):
    """
    Convert masked columns to binary for analysis.
    RESISTANT = 1 (acquired)
    EXPECTED = NaN (excluded from module detection)
    SUSCEPTIBLE/INTERMEDIATE = 0
    Missing = NaN
    """
    binary = pd.DataFrame(index=df.index)
    for ab in panel:
        masked_col = ab + "_masked"
        if masked_col not in df.columns:
            if ab in df.columns:
                masked_col = ab
            else:
                binary[ab] = np.nan
                continue

        s = df[masked_col].astype(str).str.strip().str.upper()
        binary[ab] = np.where(
            s == "RESISTANT", 1,
            np.where(s == "NONSUSCEPTIBLE", 1,
            np.where(s.isin(["SUSCEPTIBLE", "INTERMEDIATE",
                              "SUSCEPTIBLE-DOSE DEPENDENT"]), 0,
                     np.nan))  # EXPECTED and missing both become NaN
        )

    return binary.astype(float)


# ═════════════════════════════════════════════════════════════════
def characterise_unknowns(df, panel):
    """
    Layer 2: Characterise isolates with UNKNOWN breakpoint standard.
    Cross-tabulate against species and continent.
    """
    divider("Layer 2 — Unknown breakpoint characterisation")

    total = len(df)
    unknown = df[df["breakpoint_standard"] == "UNKNOWN"]
    known   = df[df["breakpoint_standard"] != "UNKNOWN"]

    print(f"\n  Total isolates   : {total:,}")
    print(f"  Known standard   : {len(known):,} ({len(known)/total*100:.1f}%)")
    print(f"  Unknown standard : {len(unknown):,} ({len(unknown)/total*100:.1f}%)")

    print(f"\n  Breakpoint standard × species:")
    ct_species = pd.crosstab(
        df["species"],
        df["breakpoint_standard"],
        margins=True
    )
    print(ct_species.to_string())

    if "continent" in df.columns:
        print(f"\n  Breakpoint standard × continent:")
        ct_continent = pd.crosstab(
            df["continent"],
            df["breakpoint_standard"],
            margins=True
        )
        print(ct_continent.to_string())

        print(f"\n  Unknown isolates by continent (% of continent total):")
        for cont in df["continent"].unique():
            sub = df[df["continent"] == cont]
            n_unk = (sub["breakpoint_standard"] == "UNKNOWN").sum()
            pct = n_unk / len(sub) * 100
            print(f"    {cont:<15} {n_unk:>6,} / {len(sub):>6,} ({pct:.1f}% unknown)")

    # Test whether unknowns are randomly distributed
    print(f"\n  Are unknowns randomly distributed across species?")
    expected_unknown_rate = len(unknown) / total
    for sp in df["species"].unique():
        sub = df[df["species"] == sp]
        actual_rate = (sub["breakpoint_standard"] == "UNKNOWN").mean()
        flag = "⚠ BIASED" if abs(actual_rate - expected_unknown_rate) > 0.10 else "OK"
        print(f"    {sp:<20} unknown rate: {actual_rate*100:.1f}% "
              f"(overall: {expected_unknown_rate*100:.1f}%) {flag}")

    return unknown, known


# ═════════════════════════════════════════════════════════════════
def main():

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    print("=" * 72)
    print("  AMR PREPROCESSING — LAYER 1 + LAYER 2")
    print("=" * 72)

    # ── Load primary dataset ──────────────────────────────────────
    divider("Loading primary dataset")
    primary_path = os.path.join(DATA_FOLDER, PRIMARY_FILE)
    df = load_csv(primary_path)
    print(f"  Shape: {df.shape[0]:,} isolates × {df.shape[1]} columns")
    print(f"  Species: {df['species'].value_counts().to_dict()}")

    # ── Metadata join ─────────────────────────────────────────────
    divider("Joining BV-BRC genome metadata")
    frames = []
    for sp in df["species"].unique():
        sub = df[df["species"] == sp].copy()
        sub = join_metadata(sub, sp)
        frames.append(sub)
    df = pd.concat(frames, ignore_index=True)

    print(f"\n  Overall country coverage: "
          f"{df['country'].notna().sum():,} / {len(df):,} "
          f"({df['country'].notna().mean()*100:.1f}%)")

    if "continent" in df.columns:
        print(f"\n  Continent distribution:")
        print(df["continent"].value_counts().to_string())

    if "collection_year" in df.columns:
        print(f"\n  Year distribution:")
        yr = df["collection_year"].dropna().astype(int).value_counts().sort_index()
        print(yr.to_string())

    # ── Layer 2: Unknown characterisation ─────────────────────────
    unknown_df, known_df = characterise_unknowns(df, PRIMARY_PANEL)

    # ── Layer 1: EUCAST masking ───────────────────────────────────
    divider("Layer 1 — EUCAST Expected Phenotype Removal")

    panel_present = [c for c in PRIMARY_PANEL if c in df.columns]
    df = apply_eucast_masking(df, panel_present, "species")

    # Report how many resistances were flagged as EXPECTED
    print(f"\n  Resistances flagged as EXPECTED (intrinsic) per antibiotic:")
    total_expected = 0
    for ab in panel_present:
        masked_col = ab + "_masked"
        if masked_col in df.columns:
            n_expected = (df[masked_col] == "EXPECTED").sum()
            n_resistant = df[ab].astype(str).str.upper().isin(
                ["RESISTANT", "NONSUSCEPTIBLE"]).sum()
            total_expected += n_expected
            if n_expected > 0:
                print(f"    {ab:<40} {n_expected:>6,} flagged as EXPECTED "
                      f"({n_expected/n_resistant*100:.1f}% of resistant calls)")

    print(f"\n  Total resistance calls flagged as EXPECTED: {total_expected:,}")
    print(f"  These will be excluded from module detection.")

    # ── Binarise after masking ────────────────────────────────────
    divider("Binarising masked resistance matrix")
    binary = binarise_masked(df, panel_present)
    binary["species"] = df["species"].values
    binary["breakpoint_standard"] = df["breakpoint_standard"].values
    if "country" in df.columns:
        binary["country"] = df["country"].values
    if "continent" in df.columns:
        binary["continent"] = df["continent"].values
    if "collection_year" in df.columns:
        binary["collection_year"] = df["collection_year"].values
    if "Genome ID" in df.columns:
        binary["genome_id"] = df["Genome ID"].values

    print(f"\n  Acquired resistance rates after EUCAST masking:")
    for ab in panel_present:
        col = binary[ab].dropna()
        rate = col.mean() * 100
        n = len(col)
        print(f"    {ab:<40} {rate:5.1f}%  (n={n:,})")

    # ── Save cleaned datasets ─────────────────────────────────────
    divider("Saving output files")

    # Full cleaned dataset
    full_path = os.path.join(OUTPUT_FOLDER, "primary_cleaned.csv")
    binary.to_csv(full_path, index=False)
    print(f"  ✓ Full cleaned dataset → {full_path}")

    # CLSI only
    clsi = binary[binary["breakpoint_standard"] == "CLSI"]
    clsi_path = os.path.join(OUTPUT_FOLDER, "primary_clsi_only.csv")
    clsi.to_csv(clsi_path, index=False)
    print(f"  ✓ CLSI only ({len(clsi):,} isolates) → {clsi_path}")

    # EUCAST only
    eucast = binary[binary["breakpoint_standard"] == "EUCAST"]
    eucast_path = os.path.join(OUTPUT_FOLDER, "primary_eucast_only.csv")
    eucast.to_csv(eucast_path, index=False)
    print(f"  ✓ EUCAST only ({len(eucast):,} isolates) → {eucast_path}")

    # Unknown only
    unk = binary[binary["breakpoint_standard"] == "UNKNOWN"]
    unk_path = os.path.join(OUTPUT_FOLDER, "primary_unknown_only.csv")
    unk.to_csv(unk_path, index=False)
    print(f"  ✓ Unknown only ({len(unk):,} isolates) → {unk_path}")

    # Also save full df with masked columns for audit
    masked_cols = ["species", "breakpoint_standard", "country",
                   "continent", "collection_year"] + \
                  [ab + "_masked" for ab in panel_present
                   if ab + "_masked" in df.columns]
    masked_cols = [c for c in masked_cols if c in df.columns]
    audit_path = os.path.join(OUTPUT_FOLDER, "primary_masked_audit.csv")
    df[masked_cols].to_csv(audit_path, index=False)
    print(f"  ✓ Masked audit file → {audit_path}")

    # ── Process secondary dataset ─────────────────────────────────
    divider("Processing secondary dataset — P. aeruginosa")
    secondary_path = os.path.join(DATA_FOLDER, SECONDARY_FILE)
    if os.path.exists(secondary_path):
        df_sec = load_csv(secondary_path)
        df_sec = join_metadata(df_sec, "P. aeruginosa")
        df_sec = apply_eucast_masking(df_sec, SECONDARY_PANEL, "species")

        sec_panel_present = [c for c in SECONDARY_PANEL if c in df_sec.columns]
        binary_sec = binarise_masked(df_sec, sec_panel_present)
        binary_sec["species"] = df_sec["species"].values
        binary_sec["breakpoint_standard"] = df_sec["breakpoint_standard"].values
        if "country" in df_sec.columns:
            binary_sec["country"] = df_sec["country"].values
        if "continent" in df_sec.columns:
            binary_sec["continent"] = df_sec["continent"].values
        if "collection_year" in df_sec.columns:
            binary_sec["collection_year"] = df_sec["collection_year"].values

        sec_path = os.path.join(OUTPUT_FOLDER, "secondary_cleaned.csv")
        binary_sec.to_csv(sec_path, index=False)
        print(f"  ✓ Secondary cleaned → {sec_path}")

        print(f"\n  P. aeruginosa acquired resistance rates after masking:")
        for ab in sec_panel_present:
            col = binary_sec[ab].dropna()
            rate = col.mean() * 100
            print(f"    {ab:<40} {rate:5.1f}%  (n={len(col):,})")

    # ── Final summary ─────────────────────────────────────────────
    divider("FINAL SUMMARY")
    print(f"""
  LAYER 1 COMPLETE — EUCAST intrinsic resistances flagged and excluded
  LAYER 2 COMPLETE — Breakpoint standard characterised, unknowns profiled

  OUTPUT FILES:
  ─────────────────────────────────────────────────────
  primary_cleaned.csv       — all isolates, masked, with geo/temporal data
  primary_clsi_only.csv     — CLSI isolates only
  primary_eucast_only.csv   — EUCAST isolates only
  primary_unknown_only.csv  — unknown standard isolates
  primary_masked_audit.csv  — full audit trail of masking decisions
  secondary_cleaned.csv     — P. aeruginosa cleaned

  NEXT STEP — Layer 3: Sampling Bias Correction
  ─────────────────────────────────────────────────────
  Apply propensity-based reweighting using species,
  continent, and collection_year as covariates.
  Then re-run module detection on the weighted dataset.
""")

    divider()
    print("  Preprocessing complete.")
    divider()


if __name__ == "__main__":
    main()
