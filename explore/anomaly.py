"""
anomaly.py — Statistical "Emerging Patterns" engine for the PBM dashboard.
[rest of docstring remains the same]
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Callable, Any, Tuple

# ── TUNABLE THRESHOLDS ────────────────────────────────────────────
# [thresholds remain the same]

MIN_BASELINE_MONTHS = 2
MIN_CLAIMS = 20
MIN_BASELINE_CLAIMS = 5
MIN_Z_SCORE = 2.0
MIN_PERCENT_CHANGE = 50
MIN_NOVEL_VOLUME = 10
MIN_SCORE_TO_REPORT = 25

Z_SCORE_WEIGHT = 0.40
PCT_CHANGE_WEIGHT = 0.25
VOLUME_WEIGHT = 0.25
NOVELTY_WEIGHT = 0.10

VOLUME_REF_CAP = 100
CRITICAL_THRESHOLD = 80
WARNING_THRESHOLD = 55

FINDINGS_COLUMNS = [
    "Rank", "Dimension", "Entity", "Metric", "Current", "Baseline_Mean",
    "Baseline_Months", "Novel", "Anomaly_Score",
    "Severity", "Reason",
]


# ── COMBO RULES CONFIGURATION ─────────────────────────────────────
# Define all cross-dimensional novelty checks declaratively.

ComboRule = Dict[str, Any]

COMBO_RULES: List[ComboRule] = [
    {
        "table": "gender_diag_stats",
        "columns": ["MEM_GENDER", "PA_PRIMARY_DIAG"],
        "dimension": "Gender \u00d7 Diagnosis",
        "reason": lambda key, claims: (
            f"{key[0]} / {key[1]}: this gender-diagnosis pairing has never occurred together "
            f"in the baseline months, although the diagnosis itself may not be new. "
            f"{claims:,.0f} claim(s) this month."
        ),
        "min_claims": MIN_NOVEL_VOLUME,
    },
    {
        "table": "age_drug_stats",
        "columns": ["AGE_GROUP", "DRUG_CODE"],
        "dimension": "Age \u00d7 Drug",
        "reason": lambda key, claims: (
            f"{key[0]} / {key[1]}: this age-drug pairing has no precedent in the baseline period, "
            f"although the drug itself may already be in use elsewhere. "
            f"{claims:,.0f} claim(s) this month."
        ),
        "min_claims": MIN_NOVEL_VOLUME,
    },
    {
        "table": "provider_drug_stats",
        "columns": ["DOC_LIC_NO", "DRUG_CODE"],
        "dimension": "Provider \u00d7 Drug",
        "reason": lambda key, claims: (
            f"Provider {key[0]} has never billed drug '{key[1]}' before. "
            f"{claims:,.0f} claim(s) this month — worth checking against the provider's "
            f"usual prescribing pattern."
        ),
        "min_claims": MIN_NOVEL_VOLUME,
    },
    {
        "table": "provider_diag_stats",
        "columns": ["DOC_LIC_NO", "PA_PRIMARY_DIAG"],
        "dimension": "Provider \u00d7 Diagnosis",
        "reason": lambda key, claims: (
            f"Provider {key[0]} has never billed diagnosis '{key[1]}' before. "
            f"{claims:,.0f} claim(s) this month — outside the provider's usual case mix."
        ),
        "min_claims": MIN_NOVEL_VOLUME,
    },
    {
        "table": "combo_stats",
        "columns": ["DRUG_DIAG_COMBO"],
        "dimension": "New Drug-Diagnosis Combo",
        "reason": lambda key, claims: (
            f"Drug-diagnosis combination '{key[0]}' did not exist in any baseline month and "
            f"already has {claims:,.0f} claim(s) this month."
        ),
        "min_claims": 3,  # Lower threshold for exact drug-diagnosis combos
    },
]


# ── SCORING FUNCTIONS ─────────────────────────────────────────────
# [All scoring functions remain the same]

def _z_component(z):
    """0-100. How many standard deviations from baseline, capped at 6."""
    if pd.isna(z):
        return 0.0
    return min(abs(z), 6.0) / 6.0 * 100.0


def _pct_component(pct):
    """0-100. Magnitude of percent change, capped at 500%."""
    if pd.isna(pct) or pct == 0:
        return 0.0
    return min(abs(pct), 500.0) / 500.0 * 100.0


def _volume_component(current_vol):
    """0-100. Saturates at VOLUME_REF_CAP claims."""
    return min(current_vol / VOLUME_REF_CAP, 1.0) * 100.0


def anomaly_score(z, pct, current_vol, novel=False, combo_novel=False, baseline_vol=0):
    """
    Composite 0-100 score. This is the one change that matters most:
    volume is now a first-class input, weighted equally with z-score
    contribution-wise, instead of being a multiplier applied after the
    fact. A 5-claim, 500% spike no longer outranks a 5,000-claim, 30%
    trend — see the docstring at the top of the file for the example.
    """
    z_part = _z_component(z)
    pct_part = _pct_component(pct)
    vol_part = _volume_component(current_vol)

    novelty_part = 0.0
    if novel:
        novelty_part = 100.0       # first-ever occurrence is the strongest signal
    elif combo_novel:
        novelty_part = 50.0        # a new pairing of two existing things is a weaker signal

    score = (
        z_part * Z_SCORE_WEIGHT +
        pct_part * PCT_CHANGE_WEIGHT +
        vol_part * VOLUME_WEIGHT +
        novelty_part * NOVELTY_WEIGHT
    )

    # A drift finding compared against a thin baseline is weaker evidence,
    # even if the math says it's extreme — halve it rather than drop it,
    # so the analyst can still see it lower in the list if they want to.
    if not novel and baseline_vol < MIN_BASELINE_CLAIMS:
        score *= 0.5

    return round(min(score, 100.0), 1)

def severity_from_score(score):
    if score >= CRITICAL_THRESHOLD:
        return "critical"
    if score >= WARNING_THRESHOLD:
        return "warning"
    if score >= MIN_SCORE_TO_REPORT:
        return "info"
    return "success"


def confidence_label(z):
    """Turn a z-score into something a manager can read at a glance."""
    if pd.isna(z):
        return "Unknown — no statistical baseline yet"
    sigma = abs(z)
    if sigma >= 3.0:
        return "Very high (>99.7% confidence)"
    if sigma >= 2.5:
        return "High (>99% confidence)"
    if sigma >= 2.0:
        return "Medium-high (>95% confidence)"
    if sigma >= 1.5:
        return "Medium (>87% confidence)"
    return "Low (<87% confidence)"


# ── BASELINE ASSEMBLY ─────────────────────────────────────────────
# [These functions remain the same]

def _collect_history_long(historical, table_name, entity_cols, value_col):
    """Stack one table across all historical months."""
    frames = []
    for month, snap in historical.items():
        df = snap.get(table_name)
        if df is None or df.empty or value_col not in df.columns:
            continue
        sub = df[entity_cols + [value_col]].copy()
        sub["month"] = month
        frames.append(sub)
    if not frames:
        return pd.DataFrame(columns=entity_cols + ["month", value_col])
    return pd.concat(frames, ignore_index=True)


def _baseline_stats(history_long, entity_cols, value_col):
    """Per-entity mean / std / number-of-months-seen across history_long."""
    if history_long.empty:
        return pd.DataFrame(columns=entity_cols + ["baseline_mean", "baseline_std", "n_months"])
    g = (
        history_long.groupby(entity_cols)[value_col]
        .agg(baseline_mean="mean", baseline_std="std", n_months="count")
        .reset_index()
    )
    g["baseline_std"] = g["baseline_std"].fillna(0)
    return g


def _row_z(current_val, baseline_mean, baseline_std, n_months):
    if n_months < MIN_BASELINE_MONTHS or pd.isna(baseline_mean):
        return np.nan
    if baseline_std == 0:
        if current_val == baseline_mean:
            return 0.0
        return float(np.sign(current_val - baseline_mean) * 6.0)
    return (current_val - baseline_mean) / baseline_std


def _pct_change(current_val, baseline_mean):
    if baseline_mean is None or pd.isna(baseline_mean) or baseline_mean == 0:
        return np.nan
    return (current_val - baseline_mean) / baseline_mean * 100


def _build_reason(entity_label, metric_label, current_val, baseline_mean,
                   n_months, pct, z, novel, confidence):
    """One paragraph, plain language."""
    if novel:
        return (
            f"{entity_label}: never appeared in any of the {n_months if n_months else 'available'} "
            f"baseline months. {current_val:,.2f} this month. Confidence: {confidence}."
        )

    direction = "up" if current_val > baseline_mean else "down"
    abs_change = current_val - baseline_mean
    pct_text = f", {pct:+.0f}%" if pd.notna(pct) else ""
    z_text = f", {z:.1f}\u03c3 from baseline" if pd.notna(z) else ""

    return (
        f"{entity_label}: {metric_label} went {direction} from {baseline_mean:,.2f} "
        f"({n_months}-month baseline) to {current_val:,.2f} "
        f"({abs_change:+.2f} absolute{pct_text}{z_text}). Confidence: {confidence}."
    )


# ── GENERIC DRIFT DETECTOR ────────────────────────────────────────
# [FIXED: Now tracks baseline_vol and uses it for the penalty]

def _baseline_stats_with_volume(history_long, entity_cols, value_col):
    """Per-entity mean / std / months / total volume."""
    if history_long.empty:
        return pd.DataFrame(columns=entity_cols + [
            "baseline_mean", "baseline_std", "n_months", "baseline_vol"
        ])
    g = (
        history_long.groupby(entity_cols)[value_col]
        .agg(
            baseline_mean="mean",
            baseline_std="std",
            n_months="count",
            baseline_vol="sum"  # Added: total claim volume in baseline
        )
        .reset_index()
    )
    g["baseline_std"] = g["baseline_std"].fillna(0)
    return g


def _generic_drift(current_df, historical, table_name, entity_cols, value_col,
                    dimension_label, metric_label, name_map=None):
    """
    Compare every entity's current `value_col` against its own trailing
    baseline, with volume filtering applied before scoring.
    """
    if current_df is None or current_df.empty or value_col not in current_df.columns:
        return pd.DataFrame()

    hist_long = _collect_history_long(historical, table_name, entity_cols, value_col)
    base = _baseline_stats_with_volume(hist_long, entity_cols, value_col)

    merged = current_df[entity_cols + [value_col]].merge(base, on=entity_cols, how="left")
    merged["n_months"] = pd.to_numeric(merged["n_months"], errors="coerce").fillna(0)

    peer_vals = current_df[value_col]
    peer_mean, peer_std = peer_vals.mean(), peer_vals.std(ddof=0)

    rows = []
    for _, r in merged.iterrows():
        current_val = r[value_col]
        n_months = int(r["n_months"])
        baseline_mean = r["baseline_mean"]
        baseline_std = r["baseline_std"]
        baseline_vol = r.get("baseline_vol", 0) or 0
        novelty = n_months == 0

        # Volume gate
        min_vol = MIN_NOVEL_VOLUME if novelty else MIN_CLAIMS
        if current_val < min_vol:
            continue

        if novelty:
            z = np.nan if peer_std == 0 else (current_val - peer_mean) / peer_std
            pct = np.nan
        else:
            z = _row_z(current_val, baseline_mean, baseline_std, n_months)
            pct = _pct_change(current_val, baseline_mean)

            # Suppress drift findings that clear neither bar
            passes_z = pd.notna(z) and abs(z) >= MIN_Z_SCORE
            passes_pct = pd.notna(pct) and abs(pct) >= MIN_PERCENT_CHANGE
            if not (passes_z or passes_pct):
                continue

        if pd.isna(z) and pd.isna(pct) and not novelty:
            continue

        # FIXED: Pass baseline_vol to anomaly_score
        score = anomaly_score(
            z, pct,
            current_vol=current_val,
            novel=novelty,
            baseline_vol=baseline_vol  # Added parameter
        )
        if score < MIN_SCORE_TO_REPORT:
            continue

        key = tuple(r[c] for c in entity_cols)
        entity_label = " / ".join(str(v) for v in key)
        if name_map and key[0] in name_map and name_map[key[0]]:
            entity_label = f"{name_map[key[0]]} ({entity_label})"

        confidence = confidence_label(z)
        reason = _build_reason(
            entity_label, metric_label, current_val, baseline_mean,
            n_months, pct, z, novelty, confidence,
        )

        rows.append({
            "Dimension": dimension_label,
            "Entity": entity_label,
            "Metric": metric_label,
            "Current": round(float(current_val), 2),
            "Baseline_Mean": round(float(baseline_mean), 2) if pd.notna(baseline_mean) else None,
            "Baseline_Months": n_months,
            "Novel": novelty,
            "Anomaly_Score": score,
            "Severity": severity_from_score(score),
            "Reason": reason,
        })

    return pd.DataFrame(rows)


# ── TRUE NOVELTY DETECTOR ────────────────────────────────────────
# [detect_true_novelty remains the same]

def detect_true_novelty(current_snapshot, historical, table_name, entity_col,
                         dimension_label, metric_label="Claims"):
    """
    Flag entities that never appeared in ANY baseline month.
    """
    if len(historical) == 0:
        return pd.DataFrame()

    cur = current_snapshot.get(table_name)
    if cur is None or cur.empty or entity_col not in cur.columns:
        return pd.DataFrame()

    seen_before = set()
    for snap in historical.values():
        hist_df = snap.get(table_name)
        if hist_df is not None and not hist_df.empty and entity_col in hist_df.columns:
            seen_before.update(hist_df[entity_col].astype(str))

    cur = cur.copy()
    cur["_key"] = cur[entity_col].astype(str)
    new_entities = cur[~cur["_key"].isin(seen_before)]
    if new_entities.empty:
        return pd.DataFrame()

    value_col = "Claims" if "Claims" in new_entities.columns else (
        "Count" if "Count" in new_entities.columns else None
    )
    if value_col is None:
        return pd.DataFrame()

    candidates = new_entities[new_entities[value_col] >= MIN_NOVEL_VOLUME]
    if candidates.empty:
        return pd.DataFrame()

    rows = []
    for _, r in candidates.iterrows():
        claims = float(r[value_col])
        score = anomaly_score(z=np.nan, pct=np.nan, current_vol=claims, novel=True)
        if score < MIN_SCORE_TO_REPORT:
            continue

        entity_label = str(r[entity_col])
        rows.append({
            "Dimension": dimension_label,
            "Entity": entity_label,
            "Metric": metric_label,
            "Current": claims,
            "Baseline_Months": len(historical),
            "Novel": True,
            "Anomaly_Score": score,
            "Severity": severity_from_score(score),
            "Reason": (
                f"{entity_label}: never appeared in any of the {len(historical)} baseline "
                f"months — this is a genuinely new {dimension_label.lower()}, not just a new "
                f"pairing. {claims:,.0f} claims this month. Confidence: {confidence_label(np.nan)}."
            ),
        })

    return pd.DataFrame(rows)


# ── COMBO NOVELTY DETECTOR ───────────────────────────────────────
# [Updated to use COMBO_RULES]

def detect_combo_novelty(current_snapshot, historical, rule: ComboRule):
    """
    Flag combinations present this month with ZERO occurrences across
    every historical month, using a configuration dictionary.
    """
    table_name = rule["table"]
    entity_cols = rule["columns"]
    dimension_label = rule["dimension"]
    reason_fn = rule["reason"]
    min_claims = rule.get("min_claims", MIN_NOVEL_VOLUME)

    current_df = current_snapshot.get(table_name)
    if current_df is None or current_df.empty or "Claims" not in current_df.columns:
        return pd.DataFrame()

    if len(historical) == 0:
        return pd.DataFrame()

    hist_long = _collect_history_long(historical, table_name, entity_cols, "Claims")

    def _key(df):
        key = df[entity_cols[0]].astype(str)
        for c in entity_cols[1:]:
            key = key + "\x1f" + df[c].astype(str)
        return key

    seen_before = set(_key(hist_long)) if not hist_long.empty else set()
    current_keys = _key(current_df)
    is_novel = ~current_keys.isin(seen_before)

    candidates = current_df.loc[is_novel & (current_df["Claims"] >= min_claims)].copy()
    if candidates.empty:
        return pd.DataFrame()

    peer_mean, peer_std = current_df["Claims"].mean(), current_df["Claims"].std(ddof=0)

    rows = []
    for _, r in candidates.iterrows():
        claims = float(r["Claims"])
        z = (claims - peer_mean) / peer_std if peer_std else np.nan

        score = anomaly_score(z=z, pct=np.nan, current_vol=claims, combo_novel=True)
        if score < MIN_SCORE_TO_REPORT:
            continue

        key = tuple(r[c] for c in entity_cols)

        rows.append({
            "Dimension": dimension_label,
            "Entity": " / ".join(str(v) for v in key),
            "Metric": "Claims (new combination)",
            "Current": claims,
            "Baseline_Months": len(historical),
            "Novel": True,
            "Anomaly_Score": score,
            "Severity": severity_from_score(score),
            "Reason": reason_fn(key, claims),
        })

    return pd.DataFrame(rows)


# ── MASTER ORCHESTRATOR ───────────────────────────────────────────

def run_emerging_pattern_scan(current_snapshot, historical, drug_df=None):
    """
    Run every drift + novelty detector and return one ranked findings table.

    Parameters
    ----------
    current_snapshot : dict   -- output of history.build_snapshot(drug_df)
    historical       : dict   -- {month_label: snapshot_dict}, output of
                                  history.load_baseline()
    drug_df          : DataFrame, optional -- used only to attach human
                                  -readable names (DOC_NAME) to provider rows
    """
    findings = []

    name_map = None
    if drug_df is not None and {"DOC_LIC_NO", "DOC_NAME"}.issubset(drug_df.columns):
        name_map = (
            drug_df[["DOC_LIC_NO", "DOC_NAME"]].dropna().drop_duplicates("DOC_LIC_NO")
            .set_index("DOC_LIC_NO")["DOC_NAME"].to_dict()
        )

    # ── DRIFT: an entity we've seen before has changed ─────────────
    if "provider_stats" in current_snapshot:
        prov = current_snapshot["provider_stats"]
        findings.append(_generic_drift(prov, historical, "provider_stats", ["DOC_LIC_NO"],
                                        "RejRate", "Provider Behavior", "Rejection Rate %", name_map))
        findings.append(_generic_drift(prov, historical, "provider_stats", ["DOC_LIC_NO"],
                                        "Claims", "Provider Behavior", "Claim Volume", name_map))

    if "drug_stats" in current_snapshot:
        drug = current_snapshot["drug_stats"]
        findings.append(_generic_drift(drug, historical, "drug_stats", ["DRUG_CODE"],
                                        "Claims", "Drug Utilization", "Claim Volume"))
        findings.append(_generic_drift(drug, historical, "drug_stats", ["DRUG_CODE"],
                                        "RejRate", "Drug Utilization", "Rejection Rate %"))

    if "diag_stats" in current_snapshot:
        findings.append(_generic_drift(current_snapshot["diag_stats"], historical, "diag_stats",
                                        ["PA_PRIMARY_DIAG"], "Claims", "Diagnosis Drift", "Claim Volume"))

    if "code_stats" in current_snapshot:
        findings.append(_generic_drift(current_snapshot["code_stats"], historical, "code_stats",
                                        ["REJ_CODE_PREFIX"], "Share", "Rejection Code Drift",
                                        "% Share of All Rejections"))

    if "gender_stats" in current_snapshot:
        gen = current_snapshot["gender_stats"]
        findings.append(_generic_drift(gen, historical, "gender_stats", ["MEM_GENDER"],
                                        "RejRate", "Gender Anomaly", "Rejection Rate %"))
        findings.append(_generic_drift(gen, historical, "gender_stats", ["MEM_GENDER"],
                                        "Claims", "Gender Anomaly", "Claim Volume"))

    if "age_stats" in current_snapshot:
        findings.append(_generic_drift(current_snapshot["age_stats"], historical, "age_stats",
                                        ["AGE_GROUP"], "RejRate", "Age Anomaly", "Rejection Rate %"))

    if "drug_concentration_stats" in current_snapshot:
        findings.append(_generic_drift(current_snapshot["drug_concentration_stats"], historical,
                                        "drug_concentration_stats", ["DRUG_CODE"],
                                        "Top_Provider_Share", "Provider Dominance",
                                        "Top Provider Share %"))

    # ── TRUE NOVELTY: the entity itself never existed before ───────
    findings.append(detect_true_novelty(current_snapshot, historical, "diag_stats",
                                         "PA_PRIMARY_DIAG", "True Novelty - Diagnosis"))
    findings.append(detect_true_novelty(current_snapshot, historical, "code_stats",
                                         "REJ_CODE_PREFIX", "True Novelty - Rejection Code"))

    # ── COMBO NOVELTY: old entities paired in a new way ────────────
    for rule in COMBO_RULES:
        findings.append(detect_combo_novelty(current_snapshot, historical, rule))

    findings = [f for f in findings if f is not None and not f.empty]
    if not findings:
        return pd.DataFrame(columns=FINDINGS_COLUMNS[1:])

    result = pd.concat(findings, ignore_index=True)
    result = result.sort_values("Anomaly_Score", ascending=False).reset_index(drop=True)
    result.insert(0, "Rank", range(1, len(result) + 1))
    return result


# ── SUMMARY AND FRAUD SUBSET ──────────────────────────────────────
# [These functions remain the same]

def summarize_scan(findings_df):
    """Small KPI summary dict for the top of the Emerging Patterns tab."""
    if findings_df.empty:
        return {"total": 0, "critical": 0, "warning": 0, "novel": 0, "avg_score": 0.0}
    return {
        "total": len(findings_df),
        "critical": int((findings_df["Severity"] == "critical").sum()),
        "warning": int((findings_df["Severity"] == "warning").sum()),
        "novel": int(findings_df["Novel"].sum()),
        "avg_score": round(findings_df["Anomaly_Score"].mean(), 1),
    }


FRAUD_DIMENSIONS = {
    "Gender \u00d7 Diagnosis",
    "Age \u00d7 Drug",
    "Provider \u00d7 Drug",
    "Provider \u00d7 Diagnosis",
    "New Drug-Diagnosis Combo",
    "Provider Behavior",
    "True Novelty - Diagnosis",
    "True Novelty - Rejection Code",
}


def extract_fraud_signals(findings_df):
    """
    Pull out the findings most relevant to fraud/safety review and rank
    them by a fraud-specific risk score.
    """
    if findings_df.empty:
        return pd.DataFrame()

    fraud_relevant = findings_df[findings_df["Dimension"].isin(FRAUD_DIMENSIONS)].copy()
    if fraud_relevant.empty:
        return pd.DataFrame()

    fraud_relevant["Fraud_Risk"] = fraud_relevant.apply(_fraud_risk, axis=1)
    return fraud_relevant.sort_values("Fraud_Risk", ascending=False).reset_index(drop=True)


def _fraud_risk(row):
    """0-100 risk score layered on top of Anomaly_Score."""
    risk = 0.0
    dimension = row.get("Dimension", "")
    entity = str(row.get("Entity", "")).lower()
    current = row.get("Current", 0) or 0
    pct_change = row.get("Pct_Change", 0) or 0

    if dimension == "Gender \u00d7 Diagnosis":
        risk += 30
        if "male" in entity and ("maternity" in entity or "pregnan" in entity):
            risk += 50
        elif "female" in entity and ("prostat" in entity or "testis" in entity):
            risk += 40

    elif dimension == "Age \u00d7 Drug":
        risk += 20
        if ("elderly" in entity or "senior" in entity) and ("pediatric" in entity or "child" in entity):
            risk += 25
        if ("infant" in entity or "0-2" in entity) and ("adult" in entity or "senior" in entity):
            risk += 35

    elif dimension in ("Provider \u00d7 Drug", "Provider \u00d7 Diagnosis"):
        risk += 20
        if current > 20:
            risk += 15

    elif dimension == "Provider Behavior":
        if pct_change and pct_change > 500:
            risk += 50
        elif pct_change and pct_change > 200:
            risk += 35
        elif pct_change and pct_change > 100:
            risk += 20
        if current > 80:
            risk += 35
        elif current > 60:
            risk += 25

    elif dimension.startswith("True Novelty"):
        if current > 100:
            risk += 50
        elif current > 50:
            risk += 40

    if current > 500:
        risk += 25
    elif current > 100:
        risk += 15

    return round(min(risk, 100.0), 1)