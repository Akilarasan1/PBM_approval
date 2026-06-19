"""
history.py — Monthly snapshot persistence for the PBM Emerging Patterns engine.

Why this exists
----------------
The dashboard previously analyzed exactly one uploaded file in isolation.
To discover *drift* (provider behavior changing, a diagnosis suddenly
spiking, a rejection code share growing) we need a baseline made of
several PRIOR months. This module is the thin persistence layer that
makes that possible without requiring a database:

    1. Every time a file is processed, we compute small, cheap, per-dimension
       aggregate tables (a "snapshot") — NOT the raw claim rows.
    2. The snapshot is written to disk under a month label (YYYY-MM),
       inferred from SERVICE_DT or the filename.
    3. On every dashboard run, we load up to N prior month snapshots and
       hand them to anomaly.py as the statistical baseline.

This keeps storage tiny (a few KB per month, regardless of whether the
source file had 5,000 or 5,000,000 rows) and keeps the comparison engine
fast, since it never has to re-read raw historical claim files.

Swapping storage backends
--------------------------
`HISTORY_DIR` is a local folder, which is fine for on-prem / desktop
Streamlit deployments where the filesystem persists between sessions.
If you deploy on ephemeral infra (e.g. Streamlit Community Cloud), replace
the four I/O functions at the bottom (`save_snapshot`, `load_snapshot`,
`list_available_months`, and the parquet read/write calls) with calls to
S3 / a database table / etc. Nothing in anomaly.py needs to change — it
only consumes the `dict[str, DataFrame]` shape returned by `load_baseline`.
"""

from pathlib import Path
import re

import pandas as pd

HISTORY_DIR = Path(__file__).parent / "pbm_history"

# Keys are arbitrary — anomaly.py just reads whichever ones it needs out of
# the dict, so adding a new aggregate table later requires no schema change
# here, only a new key produced by build_snapshot().
SNAPSHOT_TABLES = (
    "provider_stats", "drug_stats", "diag_stats", "code_stats",
    "gender_stats", "age_stats", "gender_diag_stats", "age_drug_stats",
    "combo_stats",
)


# ── MONTH LABELING ──────────────────────────────────────────────
def infer_month_label(drug_df, filename=None):
    """Best-effort YYYY-MM label for an uploaded file.

    Priority: SERVICE_DT (most reliable) -> filename pattern -> today.
    """
    if "SERVICE_DT" in drug_df.columns and drug_df["SERVICE_DT"].notna().any():
        modal_period = drug_df["SERVICE_DT"].dropna().dt.to_period("M").mode()
        if len(modal_period):
            return str(modal_period.iloc[0])

    if filename:
        m = re.search(r"(20\d{2})[-_]?(0[1-9]|1[0-2])", filename)
        if m:
            return f"{m.group(1)}-{m.group(2)}"

    return pd.Timestamp.today().strftime("%Y-%m")


# ── SNAPSHOT CONSTRUCTION ───────────────────────────────────────
KEY_COLUMNS = (
    "DOC_LIC_NO", "DRUG_CODE", "PA_PRIMARY_DIAG", "REJ_CODE_PREFIX",
    "MEM_GENDER", "AGE_GROUP",
)


def build_snapshot(drug_df):
    """Aggregate a full claims dataframe into small per-dimension summaries.

    Each table is keyed by the entity column(s) and carries only the
    counts needed for drift math (Claims, Rejections, RejRate, Share).
    Designed to degrade gracefully — any column missing from the source
    file simply means that table is omitted from the snapshot, and
    anomaly.py skips detectors that depend on it.
    """
    snap = {}
    if "IS_REJECTED" not in drug_df.columns:
        return snap

    # Real-world PBM exports are messy — e.g. a DOC_LIC_NO column can mix
    # string license numbers ("D959") with stray raw integers. Mixed-type
    # object columns break parquet writes and silently break equality
    # joins against future months. Force every key column to a clean
    # string up front so grouping, persistence, and later cross-month
    # comparisons are all consistent.
    drug_df = drug_df.copy()
    for col in KEY_COLUMNS:
        if col in drug_df.columns:
            drug_df[col] = drug_df[col].astype(str)

    def _rate_table(group_cols):
        g = (
            drug_df.groupby(group_cols)
            .agg(Claims=("IS_REJECTED", "count"), Rejections=("IS_REJECTED", "sum"))
            .reset_index()
        )
        g["RejRate"] = (g["Rejections"] / g["Claims"] * 100).round(2)
        return g

    if "DOC_LIC_NO" in drug_df.columns:
        snap["provider_stats"] = _rate_table(["DOC_LIC_NO"])

    if "DRUG_CODE" in drug_df.columns:
        snap["drug_stats"] = _rate_table(["DRUG_CODE"])

    if "PA_PRIMARY_DIAG" in drug_df.columns:
        snap["diag_stats"] = _rate_table(["PA_PRIMARY_DIAG"])

    if "REJ_CODE_PREFIX" in drug_df.columns:
        rejected = drug_df[drug_df["IS_REJECTED"] == 1]
        if len(rejected):
            g = rejected.groupby("REJ_CODE_PREFIX").size().reset_index(name="Count")
            g["Share"] = (g["Count"] / len(rejected) * 100).round(2)
            snap["code_stats"] = g

    if "MEM_GENDER" in drug_df.columns:
        snap["gender_stats"] = _rate_table(["MEM_GENDER"])

    if "AGE_GROUP" in drug_df.columns:
        snap["age_stats"] = _rate_table(["AGE_GROUP"])

    if {"MEM_GENDER", "PA_PRIMARY_DIAG"}.issubset(drug_df.columns):
        snap["gender_diag_stats"] = (
            drug_df.groupby(["MEM_GENDER", "PA_PRIMARY_DIAG"])
            .size().reset_index(name="Claims")
        )

    if {"AGE_GROUP", "DRUG_CODE"}.issubset(drug_df.columns):
        snap["age_drug_stats"] = (
            drug_df.groupby(["AGE_GROUP", "DRUG_CODE"])
            .size().reset_index(name="Claims")
        )

    if "DRUG_DIAG_COMBO" in drug_df.columns:
        snap["combo_stats"] = (
            drug_df.assign(DRUG_DIAG_COMBO=drug_df["DRUG_DIAG_COMBO"].astype(str))
            .groupby("DRUG_DIAG_COMBO").size().reset_index(name="Claims")
        )

    return snap


# ── PERSISTENCE ──────────────────────────────────────────────────
def _month_dir(month_label):
    d = HISTORY_DIR / month_label
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_snapshot(month_label, snapshot):
    """Persist a snapshot dict to disk, one small parquet file per table."""
    d = _month_dir(month_label)
    for name, df in snapshot.items():
        if df is not None and not df.empty:
            df.to_parquet(d / f"{name}.parquet", index=False)


def list_available_months(exclude=None):
    if not HISTORY_DIR.exists():
        return []
    months = sorted(p.name for p in HISTORY_DIR.iterdir() if p.is_dir())
    if exclude:
        months = [m for m in months if m != exclude]
    return months


def load_snapshot(month_label):
    d = HISTORY_DIR / month_label
    if not d.exists():
        return {}
    return {f.stem: pd.read_parquet(f) for f in d.glob("*.parquet")}


def load_baseline(current_month, n_months=6):
    """Load up to `n_months` of snapshots strictly before `current_month`.

    Returns (history_dict, months_used) where history_dict maps
    month_label -> snapshot dict, ready to hand to anomaly.run_emerging_pattern_scan.
    """
    months = [m for m in list_available_months(exclude=current_month) if m < current_month]
    months = months[-n_months:]
    return {m: load_snapshot(m) for m in months}, months


def delete_month(month_label):
    """Remove a stored month (e.g. to re-seed a corrected upload)."""
    d = HISTORY_DIR / month_label
    if d.exists():
        for f in d.glob("*.parquet"):
            f.unlink()
        d.rmdir()


def split_by_month(drug_df):
    """Split a claims dataframe into per-month sub-dataframes using SERVICE_DT.

    This is what makes the single uploader dynamic: a file with 1 month,
    3 months, 6 months, or a full year of claims all flow through the same
    path. Each calendar month found becomes its own bucket; rows with a
    missing SERVICE_DT (if any) are folded into the most recent bucket
    rather than silently dropped.

    Returns
    -------
    dict[str, DataFrame] ordered chronologically, e.g.
        {"2025-11": df_nov, "2025-12": df_dec, "2026-01": df_jan}
    If SERVICE_DT is absent or entirely null, returns a single bucket
    under today's month label (preserves old single-month behavior for
    files that don't carry dates).
    """
    if "SERVICE_DT" not in drug_df.columns or drug_df["SERVICE_DT"].notna().sum() == 0:
        label = pd.Timestamp.today().strftime("%Y-%m")
        return {label: drug_df.copy()}

    periods = drug_df["SERVICE_DT"].dt.to_period("M")
    labels = periods.astype(str)
    valid_labels = sorted(labels[periods.notna()].unique())
    if not valid_labels:
        label = pd.Timestamp.today().strftime("%Y-%m")
        return {label: drug_df.copy()}

    latest_label = valid_labels[-1]
    undated_mask = drug_df["SERVICE_DT"].isna()

    out = {}
    for label in valid_labels:
        out[label] = drug_df[labels == label].copy()
    if undated_mask.any():
        out[latest_label] = pd.concat(
            [out[latest_label], drug_df[undated_mask]], ignore_index=False
        )
    return out
