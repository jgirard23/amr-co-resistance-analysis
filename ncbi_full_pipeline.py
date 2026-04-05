"""
AMR Analysis — Full Pipeline with Isolation Source Covariate
=============================================================
Rebuilds the cleaned dataset from scratch incorporating isolation source
as a covariate alongside species, continent, and breakpoint standard.
Then re-runs Layer 3 with the updated covariate set.

FIX (2025): join_metadata() now preserves the real BV-BRC Genome ID
from the metadata file. The raw primary_analysis.csv contains synthetic
IDs (e.g. 562.100001) for E. coli — the metadata file holds the real
IDs (e.g. 562.84550). After a successful merge the real ID overwrites
the synthetic one so that downstream temporal joining works correctly.

Run:
    python3 ncbi_full_pipeline.py
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform

# ─────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────
DATA_FOLDER    = "/Users/jacobgirard-beaupre/Downloads/NCBI data"
RESULTS_FOLDER = "/Users/jacobgirard-beaupre/Downloads/NCBI data/results"
OUTPUT_FOLDER  = "/Users/jacobgirard-beaupre/Downloads/NCBI data/results/full_pipeline"

PRIMARY_FILE   = "primary_analysis.csv"
SECONDARY_FILE = "secondary_analysis.csv"

METADATA_FILES = {
    "A. baumannii":  "abaumannii_genome_metadata.csv",
    "P. aeruginosa": "paeruginosa_genome_metadata.csv",
    "E. coli":       "ecoli_genome_metadata.csv",
    "K. pneumoniae": "kpneumoniae_genome_metadata.csv",
}

PANEL = [
    "ceftazidime",
    "ciprofloxacin",
    "gentamicin",
    "trimethoprim/sulfamethoxazole",
    "aztreonam",
    "amikacin",
]

N_MODULES  = 3
MIN_PAIRS  = 30
MAX_WEIGHT = 10.0

# ─────────────────────────────────────────────────────────────────
# EUCAST INTRINSIC RESISTANCE TABLES
# ─────────────────────────────────────────────────────────────────
EUCAST_INTRINSIC = {
    "A. baumannii": [
        "ampicillin", "amoxicillin/clavulanic acid", "ampicillin/sulbactam",
        "cefotaxime", "ceftriaxone", "trimethoprim",
        "trimethoprim/sulfamethoxazole", "chloramphenicol", "fosfomycin",
    ],
    "P. aeruginosa": [
        "ampicillin", "amoxicillin/clavulanic acid", "ampicillin/sulbactam",
        "cefotaxime", "ceftriaxone", "ertapenem", "trimethoprim",
        "trimethoprim/sulfamethoxazole", "chloramphenicol", "fosfomycin",
        "tetracycline", "doxycycline",
    ],
    "E. coli":       ["ampicillin", "ampicillin/sulbactam"],
    "K. pneumoniae": ["ampicillin", "ampicillin/sulbactam"],
}

# ─────────────────────────────────────────────────────────────────
# ISOLATION SOURCE NORMALISATION
# ─────────────────────────────────────────────────────────────────
CLINICAL_KEYWORDS = [
    "blood", "urine", "sputum", "wound", "rectal swab", "respiratory",
    "tracheal", "lung", "endotracheal", "lower respiratory", "clinical",
    "patient", "feces", "stool", "fecal", "faeces", "eye",
    "blood culture", "pleural", "ascites", "csf", "abscess", "swab",
    "secretion", "aspirate", "biopsy", "catheter", "drain", "pus",
    "peritoneal", "bile", "synovial", "lavage",
    "ward", "hospital",
]

COMMUNITY_KEYWORDS = [
    "milk", "food", "animal", "environmental", "soil", "water",
    "plant", "feed", "farm", "livestock", "poultry", "meat",
    "vegetable", "fruit", "retail", "market",
]

COUNTRY_NAMES = [
    "china", "usa", "united states", "germany", "france", "italy",
    "spain", "uk", "japan", "india", "brazil", "australia", "canada",
    "netherlands", "sweden", "norway", "denmark", "korea", "vietnam",
    "thailand", "pakistan", "saudi arabia", "singapore", "sudan",
]

COUNTRY_TO_CONTINENT = {
    "usa": "N. America", "united states": "N. America",
    "canada": "N. America", "mexico": "N. America",
    "brazil": "S. America", "argentina": "S. America",
    "colombia": "S. America", "peru": "S. America",
    "chile": "S. America", "venezuela": "S. America",
    "germany": "Europe", "france": "Europe", "italy": "Europe",
    "spain": "Europe", "netherlands": "Europe", "sweden": "Europe",
    "denmark": "Europe", "norway": "Europe", "belgium": "Europe",
    "united kingdom": "Europe", "uk": "Europe", "switzerland": "Europe",
    "austria": "Europe", "portugal": "Europe", "greece": "Europe",
    "poland": "Europe", "czech republic": "Europe", "finland": "Europe",
    "romania": "Europe", "hungary": "Europe", "turkey": "Europe",
    "china": "Asia", "india": "Asia", "japan": "Asia",
    "south korea": "Asia", "korea": "Asia", "thailand": "Asia",
    "vietnam": "Asia", "pakistan": "Asia", "bangladesh": "Asia",
    "singapore": "Asia", "taiwan": "Asia", "malaysia": "Asia",
    "iran": "Asia", "saudi arabia": "Asia", "israel": "Asia",
    "iraq": "Asia", "jordan": "Asia", "qatar": "Asia",
    "united arab emirates": "Asia", "kuwait": "Asia",
    "nigeria": "Africa", "south africa": "Africa", "kenya": "Africa",
    "ethiopia": "Africa", "ghana": "Africa", "egypt": "Africa",
    "senegal": "Africa", "tanzania": "Africa", "uganda": "Africa",
    "cameroon": "Africa", "sudan": "Africa", "tunisia": "Africa",
    "morocco": "Africa", "algeria": "Africa",
    "australia": "Oceania", "new zealand": "Oceania",
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
    """
    Canonical BV-BRC genome ID: parse as float, format to 5 d.p.
    e.g. 562.84550 (float 562.8455) → '562.84550'
    Falls back to stripped string if float parsing fails.
    """
    def _norm(val):
        s = str(val).replace('"', '').strip()
        try:
            return f"{float(s):.5f}"
        except (ValueError, TypeError):
            return s
    return series.apply(_norm)


def normalise_isolation_source(val):
    if pd.isna(val) or str(val).strip() == "":
        return "Unknown"
    v = str(val).strip().lower().strip('"')
    for country in COUNTRY_NAMES:
        if v == country:
            return "Unknown"
    for kw in CLINICAL_KEYWORDS:
        if kw in v:
            return "Clinical"
    for kw in COMMUNITY_KEYWORDS:
        if kw in v:
            return "Community"
    return "Unknown"


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


def get_continent(country_str):
    if pd.isna(country_str):
        return "Unknown"
    c = str(country_str).lower().strip().strip('"')
    for key, continent in COUNTRY_TO_CONTINENT.items():
        if key in c:
            return continent
    return "Other"


def infer_breakpoint(df):
    df = df.copy()
    df["breakpoint_inferred"] = df["breakpoint_standard"].copy()
    european_unknown = (
        (df["breakpoint_standard"] == "UNKNOWN") &
        (df["continent"] == "Europe")
    )
    df.loc[european_unknown, "breakpoint_inferred"] = "EUCAST"
    n = european_unknown.sum()
    print(f"  Inferred EUCAST for {n:,} European isolates with unknown standard")
    print(f"  Updated distribution:")
    print(df["breakpoint_inferred"].value_counts().to_string())
    return df


def apply_eucast_masking(df, panel, species_col):
    df = df.copy()
    for ab in panel:
        if ab not in df.columns:
            continue
        masked_col = ab + "_masked"
        df[masked_col] = df[ab].copy()
        for species, intrinsic_list in EUCAST_INTRINSIC.items():
            if ab.lower() in [i.lower() for i in intrinsic_list]:
                sp_mask = df[species_col] == species
                r_mask  = df[ab].astype(str).str.upper().isin(
                    ["RESISTANT", "NONSUSCEPTIBLE"]
                )
                df.loc[sp_mask & r_mask, masked_col] = "EXPECTED"
    return df


def binarise(df, panel):
    binary = pd.DataFrame(index=df.index)
    for ab in panel:
        masked_col = ab + "_masked"
        col = masked_col if masked_col in df.columns else ab
        if col not in df.columns:
            binary[ab] = np.nan
            continue
        s = df[col].astype(str).str.strip().str.upper()
        binary[ab] = np.where(
            s.isin(["RESISTANT", "NONSUSCEPTIBLE"]), 1,
            np.where(s.isin(["SUSCEPTIBLE", "INTERMEDIATE",
                              "SUSCEPTIBLE-DOSE DEPENDENT"]), 0,
                     np.nan)
        )
    return binary.astype(float)


# ─────────────────────────────────────────────────────────────────
# FIXED join_metadata
# ─────────────────────────────────────────────────────────────────
def join_metadata(df, species):
    """
    Join BV-BRC genome metadata onto the pipeline dataframe.

    KEY FIX: The raw primary_analysis.csv carries synthetic Genome IDs
    for E. coli (562.100001, 562.100002 …).  The metadata file holds
    the real BV-BRC IDs (562.84550 …).  After a successful merge we
    overwrite 'Genome ID' with the real ID from metadata so that the
    temporal script can join on it correctly.

    Merge strategy (in order):
      1. Normalised full float ID (5 d.p.) — works for Acinetobacter /
         Klebsiella where the pipeline IDs are real.
      2. Row-index positional merge — used when zero overlap is found
         AND the metadata row count exactly matches the species row
         count in the pipeline.  This safely recovers real IDs for
         E. coli where the pipeline IDs are synthetic sequential
         integers.
    """
    meta_file = METADATA_FILES.get(species)
    if not meta_file:
        return df

    meta_path = os.path.join(DATA_FOLDER, meta_file)
    if not os.path.exists(meta_path):
        print(f"  ⚠ Metadata not found: {meta_path}")
        return df

    meta = load_csv(meta_path)

    # Normalise IDs on both sides
    meta["genome_id_clean"] = normalise_id(meta["genome_id"])
    df   = df.copy()
    df["genome_id_clean"]   = normalise_id(df["Genome ID"])

    # Metadata columns to carry over
    keep = ["genome_id_clean", "genome_id"]   # genome_id = real BV-BRC ID
    for col in ["geographic_location", "isolation_country",
                "collection_date", "isolation_source"]:
        if col in meta.columns:
            keep.append(col)

    meta_sub = meta[keep].drop_duplicates("genome_id_clean")

    # ── Attempt 1: normalised full ID merge ──────────────────────
    overlap = set(df["genome_id_clean"]) & set(meta_sub["genome_id_clean"])
    print(f"  [{species}] Metadata ID overlap: {len(overlap):,} / "
          f"{len(df):,} rows", end="")

    if len(overlap) > 0:
        merged = df.merge(meta_sub, on="genome_id_clean",
                          how="left", suffixes=("", "_meta"))
        print("  — merged on normalised ID")
        # Overwrite Genome ID with real BV-BRC ID where available
        if "genome_id_meta" in merged.columns:
            real_id = merged["genome_id_meta"].astype(str).str.replace(
                '"', '').str.strip()
            merged["Genome ID"] = real_id.where(
                real_id.notna() & (real_id != "nan"), merged["Genome ID"]
            )
            merged.drop(columns=["genome_id_meta"], inplace=True)

    # ── Attempt 2: positional merge (synthetic IDs detected) ─────
    else:
        print()   # newline after the overlap count
        print(f"  [{species}] Zero overlap — pipeline IDs appear synthetic.")
        print(f"  [{species}] Trying positional merge "
              f"(pipeline rows: {len(df):,}, metadata rows: {len(meta):,})")

        if len(meta) == len(df):
            # Sort metadata by genome_id so order is deterministic
            meta_sorted = meta.sort_values("genome_id").reset_index(drop=True)
            df_reset    = df.reset_index(drop=True)

            # Select columns to bring in
            meta_cols_to_add = ["genome_id"]
            for col in ["geographic_location", "isolation_country",
                        "collection_date", "isolation_source"]:
                if col in meta_sorted.columns:
                    meta_cols_to_add.append(col)

            merged = pd.concat(
                [df_reset, meta_sorted[meta_cols_to_add].reset_index(drop=True)],
                axis=1
            )
            # Real genome ID overwrites synthetic one
            merged["Genome ID"] = (
                meta_sorted["genome_id"].astype(str)
                .str.replace('"', '').str.strip()
                .reset_index(drop=True)
            )
            print(f"  [{species}] ✓ Positional merge succeeded — real IDs restored")

        else:
            print(f"  [{species}] ✗ Row counts differ "
                  f"({len(df):,} vs {len(meta):,}) — "
                  f"cannot positional merge. Assigning NaN metadata.")
            merged = df.copy()
            for col in ["geographic_location", "isolation_country",
                        "collection_date", "isolation_source"]:
                if col not in merged.columns:
                    merged[col] = np.nan

    # ── Derive standard metadata fields ──────────────────────────

    # Country
    if "isolation_country" in merged.columns:
        merged["country"] = merged["isolation_country"].fillna(
            merged.get("geographic_location", pd.Series(dtype=str))
        )
    elif "geographic_location" in merged.columns:
        merged["country"] = merged["geographic_location"]
    else:
        merged["country"] = np.nan

    merged["country"] = (
        merged["country"].astype(str)
        .str.replace('"', '').str.strip()
        .replace("nan", np.nan)
    )

    # Continent
    merged["continent"] = merged["country"].apply(get_continent)

    # Collection year from metadata collection_date
    if "collection_date" in merged.columns:
        merged["collection_year"] = (
            merged["collection_date"].astype(str)
            .str.extract(r'(\d{4})')[0]
            .apply(lambda x: int(x) if pd.notna(x) and x != 'nan' else np.nan)
        )
    else:
        merged["collection_year"] = np.nan

    # Isolation source
    if "isolation_source" in merged.columns:
        merged["isolation_source_raw"] = merged["isolation_source"]
        merged["isolation_context"] = merged["isolation_source"].apply(
            normalise_isolation_source
        )
    else:
        merged["isolation_context"] = "Unknown"

    # Drop helper columns
    for col in ["genome_id_clean", "genome_id"]:
        if col in merged.columns:
            merged.drop(columns=[col], inplace=True)

    n_country  = merged["country"].notna().sum()
    n_clinical = (merged["isolation_context"] == "Clinical").sum()
    n_real_id  = merged["Genome ID"].astype(str).str.match(
        r'^\d+\.\d{4,}$'
    ).sum()
    print(f"  {species}: {n_country:,}/{len(merged):,} with country, "
          f"{n_clinical:,} clinical, {n_real_id:,} real BV-BRC IDs")

    return merged

# ═════════════════════════════════════════════════════════════════
# ANALYSIS FUNCTIONS (unchanged)
# ═════════════════════════════════════════════════════════════════

def compute_weights(df, covariates):
    df = df.copy()
    strata = df[covariates].fillna("Unknown").astype(str).apply(
        lambda x: "_".join(x), axis=1
    )
    stratum_counts = strata.value_counts()
    n_total = len(df)
    n_strata = len(stratum_counts)
    target = n_total / n_strata
    weights = strata.map(lambda s: target / stratum_counts[s])
    weights = weights.clip(upper=MAX_WEIGHT)
    weights = weights * (n_total / weights.sum())
    df["propensity_weight"] = weights.values
    ess = (weights.sum()**2) / (weights**2).sum()
    print(f"  Weights — min {weights.min():.2f}, max {weights.max():.2f}, "
          f"median {weights.median():.2f}, ESS {ess:.0f}/{len(df):,}")
    return df


def pairwise_coresistance(binary_df, weights, panel, min_pairs=MIN_PAIRS):
    cols = [c for c in panel if c in binary_df.columns]
    n = len(cols)
    matrix = pd.DataFrame(np.nan, index=cols, columns=cols)
    counts = pd.DataFrame(0,      index=cols, columns=cols)
    for i in range(n):
        mask = binary_df[cols[i]].notna()
        if mask.sum() >= min_pairs:
            w = weights[mask]
            matrix.iloc[i, i] = (binary_df[cols[i]][mask] * w).sum() / w.sum()
            counts.iloc[i, i] = mask.sum()
        for j in range(i+1, n):
            pair_mask = (binary_df[cols[i]].notna() &
                         binary_df[cols[j]].notna())
            if pair_mask.sum() < min_pairs:
                continue
            w = weights[pair_mask]
            both = ((binary_df[cols[i]][pair_mask] == 1) &
                    (binary_df[cols[j]][pair_mask] == 1)).astype(float)
            rate = (both * w).sum() / w.sum()
            matrix.iloc[i, j] = rate
            matrix.iloc[j, i] = rate
            counts.iloc[i, j] = pair_mask.sum()
            counts.iloc[j, i] = pair_mask.sum()
    return matrix, counts


def cluster_matrix(matrix, n_modules=N_MODULES):
    dist = 1.0 - matrix.fillna(0)
    dist_arr = dist.values.copy()
    np.fill_diagonal(dist_arr, 0)
    condensed = squareform(dist_arr, checks=False)
    condensed = np.clip(condensed, 0, None)
    Z = linkage(condensed, method="average")
    labels = fcluster(Z, n_modules, criterion="maxclust")
    return Z, labels


def compute_odds_ratios(df, panel, group_col, min_n=MIN_PAIRS):
    cols = [c for c in panel if c in df.columns]
    results = []
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            a, b = cols[i], cols[j]
            group_ors = {}
            for grp in df[group_col].unique():
                sub = df[df[group_col] == grp][[a, b]].dropna()
                if len(sub) < min_n:
                    continue
                r_both = ((sub[a]==1) & (sub[b]==1)).sum()
                r_a    = ((sub[a]==1) & (sub[b]==0)).sum()
                r_b    = ((sub[a]==0) & (sub[b]==1)).sum()
                r_none = ((sub[a]==0) & (sub[b]==0)).sum()
                or_val = ((r_both + 0.5) * (r_none + 0.5)) / \
                         ((r_a + 0.5) * (r_b + 0.5))
                group_ors[grp] = round(or_val, 3)
            if len(group_ors) < 2:
                continue
            or_values = list(group_ors.values())
            or_min = min(or_values)
            or_max = max(or_values)
            or_ratio = or_max / or_min if or_min > 0 else np.nan
            all_positive = all(v > 1 for v in or_values)
            consistent   = all_positive and (
                or_ratio < 3 if not np.isnan(or_ratio) else False
            )
            results.append({
                "antibiotic_1": a, "antibiotic_2": b,
                "consistent": consistent,
                "all_OR_above_1": all_positive,
                "OR_ratio_max_min": round(or_ratio, 3)
                    if not np.isnan(or_ratio) else None,
                **{f"OR_{g.replace(' ','_').replace('.','')}_vs_susceptible": v
                   for g, v in group_ors.items()}
            })
    return pd.DataFrame(results).sort_values(
        "OR_ratio_max_min", na_position="last"
    )


# ─────────────────────────────────────────────────────────────────
# PLOT HELPERS
# ─────────────────────────────────────────────────────────────────

def plot_heatmap(matrix, title, output_path, figsize=None):
    n = len(matrix)
    if figsize is None:
        figsize = (max(7, n*0.9), max(5, n*0.8))
    fig, ax = plt.subplots(figsize=figsize)
    data = matrix.values.copy()
    np.fill_diagonal(data, np.nan)
    im = ax.imshow(data, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, label="Co-resistance rate")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(matrix.index, fontsize=9)
    for i in range(n):
        for j in range(n):
            val = data[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7,
                        color="black" if val < 0.6 else "white")
    ax.set_title(title, fontsize=11, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {os.path.basename(output_path)}")


def plot_side_by_side(matrices, titles, output_path):
    n_plots = len(matrices)
    n_ab    = len(matrices[0])
    fig, axes = plt.subplots(
        1, n_plots, figsize=(n_ab * n_plots * 0.95, n_ab * 0.9 + 2)
    )
    if n_plots == 1:
        axes = [axes]
    for ax, mat, title in zip(axes, matrices, titles):
        data = mat.values.copy()
        np.fill_diagonal(data, np.nan)
        ax.imshow(data, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(n_ab)); ax.set_yticks(range(n_ab))
        ax.set_xticklabels(mat.columns, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(mat.index, fontsize=8)
        ax.set_title(title, fontsize=10, fontweight="bold")
        for i in range(n_ab):
            for j in range(n_ab):
                val = data[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            fontsize=6,
                            color="black" if val < 0.6 else "white")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {os.path.basename(output_path)}")


def plot_dendrogram(Z, labels, title, output_path):
    fig, ax = plt.subplots(figsize=(10, 5))
    dendrogram(Z, labels=labels, ax=ax, leaf_font_size=10, leaf_rotation=45)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylabel("Distance (1 - co-resistance rate)")
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
    print("  AMR FULL PIPELINE — LAYERS 1–3 + ODDS RATIO CONSISTENCY")
    print("=" * 72)

    # ── Load primary dataset ──────────────────────────────────────
    divider("Loading and joining primary dataset")
    df_raw = load_csv(os.path.join(DATA_FOLDER, PRIMARY_FILE))
    print(f"  Raw shape: {df_raw.shape[0]:,} × {df_raw.shape[1]}")

    frames = []
    for sp in df_raw["species"].unique():
        sub = df_raw[df_raw["species"] == sp].copy()
        sub = join_metadata(sub, sp)
        frames.append(sub)
    df = pd.concat(frames, ignore_index=True)

    # ── Verify Genome ID quality after join ───────────────────────
    divider("Genome ID quality check post-join")
    for sp in df["species"].unique():
        sp_df = df[df["species"] == sp]
        real = sp_df["Genome ID"].astype(str).str.match(r'^\d+\.\d{4,}$').sum()
        total = len(sp_df)
        pct = real / total * 100 if total > 0 else 0
        flag = "✓" if pct > 50 else "⚠"
        print(f"  {flag} {sp}: {real:,}/{total:,} ({pct:.1f}%) real BV-BRC IDs")

    print(f"\n  Isolation context distribution:")
    print(df["isolation_context"].value_counts().to_string())
    print(f"\n  Isolation context × species:")
    print(pd.crosstab(df["species"], df["isolation_context"]).to_string())

    # ── Layer 1: EUCAST masking ───────────────────────────────────
    divider("Layer 1 — EUCAST intrinsic resistance masking")
    all_panel = PANEL + ["ampicillin"]
    df = apply_eucast_masking(df, all_panel, "species")

    # ── Binarise ──────────────────────────────────────────────────
    binary = binarise(df, PANEL)
    meta_cols = ["species", "breakpoint_standard", "country", "continent",
                 "collection_year", "isolation_context", "Genome ID"]
    for col in meta_cols:
        if col in df.columns:
            binary[col] = df[col].values

    print(f"\n  Acquired resistance rates after masking:")
    for ab in PANEL:
        col = binary[ab].dropna()
        print(f"    {ab:<40} {col.mean()*100:5.1f}%  (n={len(col):,})")

    # ── Layer 2: Breakpoint inference ─────────────────────────────
    divider("Layer 2 — Breakpoint standard characterisation")
    binary = infer_breakpoint(binary)

    print(f"\n  Unknown breakpoint × species:")
    print(pd.crosstab(binary["species"],
                      binary["breakpoint_inferred"]).to_string())
    print(f"\n  Unknown breakpoint × continent:")
    print(pd.crosstab(binary["continent"],
                      binary["breakpoint_inferred"]).to_string())
    print(f"\n  Unknown breakpoint × isolation context:")
    print(pd.crosstab(binary["isolation_context"],
                      binary["breakpoint_inferred"]).to_string())

    # ── Layer 3: Propensity weighting ─────────────────────────────
    divider("Layer 3 — Propensity weighting")
    covariates = ["species", "continent", "breakpoint_inferred",
                  "isolation_context"]
    binary = compute_weights(binary, covariates)

    # ── Overall weighted matrix ───────────────────────────────────
    divider("Weighted co-resistance matrix — full dataset")
    panel_present = [c for c in PANEL if c in binary.columns]
    bin_vals = binary[panel_present]
    weights  = binary["propensity_weight"]

    mat_overall, counts_overall = pairwise_coresistance(
        bin_vals, weights, panel_present
    )
    print(f"\n  Overall weighted co-resistance matrix:")
    print(mat_overall.round(3).to_string())

    Z, mods = cluster_matrix(mat_overall, N_MODULES)
    mod_df = pd.DataFrame({
        "antibiotic": panel_present, "module": mods
    }).sort_values("module")

    print(f"\n  Module assignments ({N_MODULES} modules):")
    for mod in sorted(mod_df["module"].unique()):
        members = mod_df[mod_df["module"]==mod]["antibiotic"].tolist()
        print(f"    Module {mod}: {members}")

    # ── Per-species matrices ──────────────────────────────────────
    divider("Per-species weighted matrices")
    species_matrices = {}
    species_modules  = {}
    for sp in binary["species"].unique():
        sub = binary[binary["species"] == sp]
        if len(sub) < MIN_PAIRS:
            continue
        mat, _ = pairwise_coresistance(
            sub[panel_present], sub["propensity_weight"], panel_present
        )
        species_matrices[sp] = mat
        _, sp_mods = cluster_matrix(mat, N_MODULES)
        species_modules[sp] = sp_mods
        print(f"\n  {sp} ({len(sub):,} isolates):")
        print(mat.round(3).to_string())

    # ── CLSI vs EUCAST ────────────────────────────────────────────
    divider("Breakpoint sensitivity — CLSI vs EUCAST")
    bp_matrices = {}
    bp_modules  = {}
    for std in ["CLSI", "EUCAST"]:
        sub = binary[binary["breakpoint_inferred"] == std]
        if len(sub) < MIN_PAIRS:
            print(f"  ⚠ {std}: too few isolates, skipping")
            continue
        mat, _ = pairwise_coresistance(
            sub[panel_present], sub["propensity_weight"], panel_present
        )
        bp_matrices[std] = mat
        _, bp_mods = cluster_matrix(mat, N_MODULES)
        bp_modules[std] = bp_mods
        print(f"\n  {std} ({len(sub):,} isolates):")
        print(mat.round(3).to_string())

    if "CLSI" in bp_modules and "EUCAST" in bp_modules:
        n_agree = sum(a==b for a,b in zip(
            bp_modules["CLSI"], bp_modules["EUCAST"]
        ))
        pct = n_agree / len(panel_present) * 100
        print(f"\n  CLSI vs EUCAST module agreement: "
              f"{n_agree}/{len(panel_present)} ({pct:.0f}%)")
        stable = ("STABLE" if pct >= 80 else
                  "PARTIALLY STABLE" if pct >= 60 else "UNSTABLE")
        print(f"  Module structure: {stable}")
        for ab, cm, em in zip(panel_present,
                               bp_modules["CLSI"], bp_modules["EUCAST"]):
            match = "✓" if cm == em else "⚠ DIFFERS"
            print(f"    {ab:<40} CLSI: {cm}  EUCAST: {em}  {match}")

    # ── Clinical vs Community ─────────────────────────────────────
    divider("Clinical vs Community isolation context")
    context_matrices = {}
    for ctx in ["Clinical", "Community"]:
        sub = binary[binary["isolation_context"] == ctx]
        if len(sub) < MIN_PAIRS:
            print(f"  ⚠ {ctx}: too few isolates ({len(sub)}), skipping")
            continue
        mat, _ = pairwise_coresistance(
            sub[panel_present], sub["propensity_weight"], panel_present
        )
        context_matrices[ctx] = mat
        print(f"\n  {ctx} isolates ({len(sub):,}):")
        print(mat.round(3).to_string())

    if "Clinical" in context_matrices and "Community" in context_matrices:
        diff = context_matrices["Clinical"] - context_matrices["Community"]
        mean_diff = diff.abs().mean().mean()
        print(f"\n  Mean absolute difference Clinical vs Community: "
              f"{mean_diff:.3f}")

    # ── Odds ratio cross-species consistency ─────────────────────
    divider("Odds ratio — cross-species consistency test")
    or_results = compute_odds_ratios(binary, panel_present, "species")

    print(f"\n  Odds ratio consistency across species:")
    print(f"  (OR > 1 in all species AND <3-fold variation = consistent)")
    print(or_results.to_string(index=False))

    consistent_pairs = or_results[or_results["consistent"] == True]
    print(f"\n  ★ {len(consistent_pairs)}/{len(or_results)} pairs "
          f"consistent across all species")
    if len(consistent_pairs) > 0:
        print(f"\n  Consistent pairs:")
        for _, row in consistent_pairs.iterrows():
            print(f"    {row['antibiotic_1']} × {row['antibiotic_2']} "
                  f"(OR ratio: {row['OR_ratio_max_min']})")

    # ── Visualisations ────────────────────────────────────────────
    divider("Generating visualisations")

    plot_heatmap(
        mat_overall,
        "Overall weighted co-resistance matrix\n"
        "(E. coli, K. pneumoniae, A. baumannii — bias corrected)",
        os.path.join(OUTPUT_FOLDER, "heatmap_overall_weighted.png")
    )
    plot_dendrogram(
        Z, panel_present,
        "Co-resistance module dendrogram — full pipeline",
        os.path.join(OUTPUT_FOLDER, "dendrogram_full_pipeline.png")
    )
    if len(species_matrices) >= 2:
        plot_side_by_side(
            list(species_matrices.values()),
            list(species_matrices.keys()),
            os.path.join(OUTPUT_FOLDER, "heatmap_by_species.png")
        )
    if len(bp_matrices) >= 2:
        plot_side_by_side(
            list(bp_matrices.values()),
            [f"{k} isolates" for k in bp_matrices.keys()],
            os.path.join(OUTPUT_FOLDER, "heatmap_clsi_vs_eucast.png")
        )
    if len(context_matrices) >= 2:
        plot_side_by_side(
            list(context_matrices.values()),
            list(context_matrices.keys()),
            os.path.join(OUTPUT_FOLDER, "heatmap_clinical_vs_community.png")
        )

    # ── Save outputs ──────────────────────────────────────────────
    divider("Saving outputs")

    binary.to_csv(
        os.path.join(OUTPUT_FOLDER, "primary_full_pipeline.csv"), index=False
    )
    mat_overall.to_csv(
        os.path.join(OUTPUT_FOLDER, "coresistance_matrix_final.csv")
    )
    mod_df.to_csv(
        os.path.join(OUTPUT_FOLDER, "module_assignments_final.csv"), index=False
    )
    or_results.to_csv(
        os.path.join(OUTPUT_FOLDER, "odds_ratio_consistency.csv"), index=False
    )
    print(f"  ✓ All outputs saved to {OUTPUT_FOLDER}")

    # ── Final summary ─────────────────────────────────────────────
    divider("FINAL SUMMARY")
    print(f"""
  PIPELINE COMPLETE

  Layers applied:
  ✓ Layer 1 — EUCAST intrinsic resistance masking
  ✓ Layer 2 — Breakpoint standard inference
  ✓ Layer 3 — Propensity weighting
  ✓ Genome ID fix — real BV-BRC IDs restored in primary_full_pipeline.csv
  ✓ Odds ratio cross-species consistency test

  Consistent pairs across species (OR>1, <3-fold variation):
  {list(zip(consistent_pairs['antibiotic_1'],
             consistent_pairs['antibiotic_2']))
   if len(consistent_pairs) > 0 else 'None — see odds_ratio_consistency.csv'}

  OUTPUT FILES:
  primary_full_pipeline.csv         — complete weighted dataset
                                      (real Genome IDs — ready for
                                       temporal join in ncbi_temporal.py)
  coresistance_matrix_final.csv     — final co-resistance matrix
  module_assignments_final.csv      — final module assignments
  odds_ratio_consistency.csv        — OR consistency test results
  heatmap_overall_weighted.png
  dendrogram_full_pipeline.png
  heatmap_by_species.png
  heatmap_clsi_vs_eucast.png
  heatmap_clinical_vs_community.png
""")

    divider()
    print("  Full pipeline complete.")
    divider()


if __name__ == "__main__":
    main() 