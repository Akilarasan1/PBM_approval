"""
anomaly.py — Statistical "Emerging Patterns" engine for the PBM dashboard.

Design principle
-----------------
Every detector in this module compares the CURRENT month's aggregated
metrics for an entity (a provider, a drug, a diagnosis, a rejection code,
a gender, an age group, or a cross-dimension combination) against a
BASELINE distribution built from the trailing N historical months for
that *same* entity. No PBM-specific or fraud-specific business rule is
hardcoded anywhere in this file — only generic statistical drift tests:

    Z-score            (current - baseline_mean) / baseline_std
    Percent change     (current - baseline_mean) / baseline_mean
    Novelty            entity/combination existed in 0 baseline months,
                       i.e. a frequency deviation from "never" to "now"
    Peer outlier check for novel entities: is THIS month's volume unusual
                       compared to every other entity that's also new
                       or active this month (handles "no history yet"
                       gracefully instead of silently ignoring it)

A single entity can be flagged by more than one detector (e.g. a provider
can show both a rejection-rate drift finding and a volume drift finding) —
that's intentional; the analyst sees every angle, ranked by severity.

Adding a new dimension later (e.g. pharmacy chain, plan type) requires:
  1. A new aggregate table added to history.build_snapshot()
  2. One new call to `_generic_drift` or `detect_cross_novelty` in
     `run_emerging_pattern_scan`
No new "rule" logic — the math is identical for every dimension.
"""

import numpy as np
import pandas as pd

# ── GENERIC STATISTICAL SENSITIVITY KNOBS (not fraud rules) ─────
MIN_BASELINE_MONTHS = 2      # need >=2 historical months for a meaningful std
MIN_SCORE_TO_REPORT = 25     # suppress noise below this anomaly score
NOVELTY_MIN_VOLUME = 3       # ignore single-claim novelty (pure noise floor)
VOLUME_REF_CAP = 50          # claim volume above which we stop down-weighting
NOVELTY_BASE_SCORE = 45      # any first-ever occurrence (above noise floor) starts here —
                              # novelty is itself the anomaly; volume below only modulates severity
NOVELTY_VOLUME_SCALE = 20    # claim count at which the novelty volume-bonus saturates

FINDINGS_COLUMNS = [
    "Rank", "Dimension", "Entity", "Metric", "Current", "Baseline_Mean",
    "Pct_Change", "ZScore", "Baseline_Months", "Novel", "Anomaly_Score",
    "Severity", "Reason",
]


# ── CORE STATISTICAL PRIMITIVES ──────────────────────────────────
def _row_z(current_val, baseline_mean, baseline_std, n_months):
    """Z-score of current value vs this entity's own historical mean/std."""
    if n_months < MIN_BASELINE_MONTHS or pd.isna(baseline_mean):
        return np.nan
    if baseline_std == 0:
        if current_val == baseline_mean:
            return 0.0
        return float(np.sign(current_val - baseline_mean) * 6.0)  # flat history, any move = extreme
    return (current_val - baseline_mean) / baseline_std


def _pct_change(current_val, baseline_mean):
    if baseline_mean is None or pd.isna(baseline_mean) or baseline_mean == 0:
        return np.nan
    return (current_val - baseline_mean) / baseline_mean * 100


def _volume_weight(volume, cap=VOLUME_REF_CAP):
    """Down-weight (don't exclude) low-volume entities so a single odd claim
    from a tiny provider doesn't outrank a real pattern, while still
    surfacing it at lower severity for analyst visibility."""
    if volume >= cap:
        return 1.0
    if volume <= 1:
        return 0.2
    return 0.2 + 0.8 * (volume - 1) / (cap - 1)


def _volume_weight_vec(volume_series, cap=VOLUME_REF_CAP):
    """Vectorized version of _volume_weight for whole-column use (avoids
    .apply() row-by-row overhead on large tables)."""
    v = volume_series.astype(float)
    w = 0.2 + 0.8 * (v - 1) / (cap - 1)
    return w.clip(lower=0.2, upper=1.0)


def _anomaly_score(z, pct, novelty=False, volume_weight=1.0):
    """Blend z-score, percent change, and novelty into one 0-100 score."""
    z_part = 0 if pd.isna(z) else min(abs(z), 6) / 6 * 60        # up to 60 pts
    pct_part = 0 if pd.isna(pct) else min(abs(pct), 300) / 300 * 25  # up to 25 pts
    novelty_part = 15 if novelty else 0                            # up to 15 pts
    return round(min((z_part + pct_part + novelty_part) * min(volume_weight, 1.0), 100), 1)


def _novelty_score(claims, volume_weight=1.0):
    """Score a 'never seen before' occurrence on its own terms.

    Comparing a brand-new combination's volume against other combinations
    in a sparse, high-cardinality cross-tab (gender x diagnosis, age x drug)
    is a weak signal — in a typical month *most* specific pairings are thin,
    so a real novelty rarely stands out as a peer outlier even though the
    fact that it has NEVER happened before is itself the anomaly. Here,
    novelty supplies a strong base score; volume only scales severity on
    top of that, capturing "and it's already growing fast" without making
    that growth a precondition for being flagged at all.
    """
    volume_bonus = min(claims / NOVELTY_VOLUME_SCALE, 1.0) * (100 - NOVELTY_BASE_SCORE)
    return round(min((NOVELTY_BASE_SCORE + volume_bonus) * min(volume_weight, 1.0), 100), 1)


def _severity_from_score(score):
    if score >= 80:
        return "critical"
    if score >= 55:
        return "warning"
    if score >= MIN_SCORE_TO_REPORT:
        return "info"
    return "success"


# ── BASELINE ASSEMBLY ─────────────────────────────────────────────
def _collect_history_long(historical, table_name, entity_cols, value_col):
    """Stack one table across all historical months: entity_cols..., month, value."""
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


# ── GENERIC DRIFT DETECTOR (used by every "existing entity changed" check) ─
def _generic_drift(current_df, historical, table_name, entity_cols, value_col,
                    dimension_label, metric_label, name_map=None):
    """
    Compare every entity's current `value_col` against its own trailing
    baseline. Returns a tidy findings frame (possibly empty).

    name_map: optional dict {entity_key: display_name} to enrich the Entity
    label (e.g. DOC_LIC_NO -> DOC_NAME) without changing the join keys.
    """
    if current_df is None or current_df.empty or value_col not in current_df.columns:
        return pd.DataFrame()

    hist_long = _collect_history_long(historical, table_name, entity_cols, value_col)
    base = _baseline_stats(hist_long, entity_cols, value_col)

    merged = current_df[entity_cols + [value_col]].merge(base, on=entity_cols, how="left")
    # merged["n_months"] = merged["n_months"].fillna(0)
    merged["n_months"] = pd.to_numeric(merged["n_months"],errors="coerce").fillna(0)

    peer_vals = current_df[value_col]
    peer_mean, peer_std = peer_vals.mean(), peer_vals.std(ddof=0)

    rows = []
    for _, r in merged.iterrows():
        current_val = r[value_col]
        n_months = int(r["n_months"])
        baseline_mean = r["baseline_mean"]
        baseline_std = r["baseline_std"]
        novelty = n_months == 0

        if novelty:
            z = np.nan if peer_std == 0 else (current_val - peer_mean) / peer_std
            pct = np.nan
        else:
            z = _row_z(current_val, baseline_mean, baseline_std, n_months)
            pct = _pct_change(current_val, baseline_mean)

        if pd.isna(z) and pd.isna(pct) and not novelty:
            continue

        vw = _volume_weight(current_val if current_val > 0 else 0)
        score = _anomaly_score(z, pct, novelty=novelty, volume_weight=vw)
        if score < MIN_SCORE_TO_REPORT:
            continue

        key = tuple(r[c] for c in entity_cols)
        entity_label = " / ".join(str(v) for v in key)
        if name_map and key[0] in name_map and name_map[key[0]]:
            entity_label = f"{name_map[key[0]]} ({entity_label})"

        if novelty:
            reason = (
                f"First appearance in any baseline month for this entity. "
                f"Current {metric_label} = {current_val:,.2f}"
                + (f", {abs(z):.1f}\u03c3 above peer average this month." if pd.notna(z) and z > 0 else ".")
            )
        else:
            direction = "increased" if current_val > baseline_mean else "decreased"
            reason = (
                f"{metric_label} {direction} to {current_val:,.2f} from a "
                f"{n_months}-month baseline average of {baseline_mean:,.2f}"
                + (f" ({pct:+.0f}% change)" if pd.notna(pct) else "")
                + (f", {abs(z):.1f}\u03c3 from baseline." if pd.notna(z) else ".")
            )

        rows.append({
            "Dimension": dimension_label,
            "Entity": entity_label,
            "Metric": metric_label,
            "Current": round(float(current_val), 2),
            "Baseline_Mean": round(float(baseline_mean), 2) if pd.notna(baseline_mean) else None,
            "Pct_Change": round(float(pct), 1) if pd.notna(pct) else None,
            "ZScore": round(float(z), 2) if pd.notna(z) else None,
            "Baseline_Months": n_months,
            "Novel": novelty,
            "Anomaly_Score": score,
            "Severity": _severity_from_score(score),
            "Reason": reason,
        })

    return pd.DataFrame(rows)


# ── CROSS-DIMENSION NOVELTY DETECTOR (used for "never seen before" combos) ─
def detect_cross_novelty(current_df, historical, table_name, entity_cols,
                          dimension_label, reason_fn, min_claims=NOVELTY_MIN_VOLUME):
    """
    Flag combinations present this month with ZERO occurrences across every
    historical month — pure frequency-deviation novelty. This is the generic
    mechanism behind "maternity claim for a male member", "new drug-diagnosis
    combo", "pediatric drug for an elderly patient", etc. The same function
    handles all of them; only the column pair and explanation text differ.
    """
    if current_df is None or current_df.empty or "Claims" not in current_df.columns:
        return pd.DataFrame()

    if len(historical) == 0:
        # With NO historical months at all, "never seen before" is true for
        # every combination by definition — that's not a finding, it's an
        # empty reference set. Cross-combination novelty needs at least one
        # real baseline month before "never happened before" carries any
        # information. Single-entity drift (_generic_drift) doesn't have
        # this problem because it falls back to genuine same-month outlier
        # detection via the peer-comparison branch instead.
        return pd.DataFrame()

    hist_long = _collect_history_long(historical, table_name, entity_cols, "Claims")

    # Vectorized anti-join: build a single string key per row using plain
    # Series concatenation (NOT .agg(join, axis=1) / .apply(), which are
    # row-wise Python loops in disguise and just as slow as iterrows() —
    # this matters once combo tables run into the tens of thousands of
    # rows at 50k+ claims/month).
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

    vw = _volume_weight_vec(candidates["Claims"])
    bonus = (candidates["Claims"] / NOVELTY_VOLUME_SCALE).clip(upper=1.0) * (100 - NOVELTY_BASE_SCORE)
    scores = ((NOVELTY_BASE_SCORE + bonus) * vw).clip(upper=100).round(1)

    keep = scores >= MIN_SCORE_TO_REPORT
    candidates, scores = candidates.loc[keep], scores.loc[keep]
    if candidates.empty:
        return pd.DataFrame()

    peer_mean, peer_std = current_df["Claims"].mean(), current_df["Claims"].std(ddof=0)
    zs = ((candidates["Claims"] - peer_mean) / peer_std).round(2) if peer_std else pd.Series(np.nan, index=candidates.index)

    keys = [tuple(row) for row in candidates[entity_cols].itertuples(index=False)]
    out = pd.DataFrame({
        "Dimension": dimension_label,
        "Entity": [" / ".join(str(v) for v in k) for k in keys],
        "Metric": "Claims (new combination)",
        "Current": candidates["Claims"].astype(int).values,
        "Baseline_Mean": 0,
        "Pct_Change": None,
        "ZScore": zs.where(pd.notna(zs), None).values,
        "Baseline_Months": len(historical),
        "Novel": True,
        "Anomaly_Score": scores.values,
        "Severity": [_severity_from_score(s) for s in scores],
        "Reason": [reason_fn(k, c) for k, c in zip(keys, candidates["Claims"].values)],
    })
    return out.reset_index(drop=True)


def detect_gender_diagnosis_anomaly(current_snapshot, historical):
    cur = current_snapshot.get("gender_diag_stats")

    def reason(key, claims):
        gender, diag = key
        return (
            f"{claims} claim(s) with diagnosis '{diag}' under gender '{gender}' have "
            f"never occurred together in the baseline months — first-time "
            f"gender/diagnosis pairing this month."
        )

    return detect_cross_novelty(
        cur, historical, "gender_diag_stats", ["MEM_GENDER", "PA_PRIMARY_DIAG"],
        "Gender \u00d7 Diagnosis", reason,
    )


def detect_age_drug_anomaly(current_snapshot, historical):
    cur = current_snapshot.get("age_drug_stats")

    def reason(key, claims):
        age, drug = key
        return (
            f"{claims} claim(s) of drug '{drug}' for age group '{age}' have no "
            f"historical precedent in the baseline period."
        )

    return detect_cross_novelty(
        cur, historical, "age_drug_stats", ["AGE_GROUP", "DRUG_CODE"],
        "Age \u00d7 Drug", reason,
    )


def detect_new_combo_growth(current_snapshot, historical):
    cur = current_snapshot.get("combo_stats")

    def reason(key, claims):
        (combo,) = key
        return (
            f"Drug-diagnosis combination '{combo}' did not exist in any baseline "
            f"month and already has {claims} claim(s) this month."
        )

    return detect_cross_novelty(
        cur, historical, "combo_stats", ["DRUG_DIAG_COMBO"],
        "New Drug-Diagnosis Combo", reason, min_claims=3,
    )


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

    findings.append(detect_gender_diagnosis_anomaly(current_snapshot, historical))
    findings.append(detect_age_drug_anomaly(current_snapshot, historical))
    findings.append(detect_new_combo_growth(current_snapshot, historical))

    findings = [f for f in findings if f is not None and not f.empty]
    if not findings:
        return pd.DataFrame(columns=FINDINGS_COLUMNS[1:])

    # result = pd.concat(findings, ignore_index=True)

    valid_findings = [df for df in findings if not df.empty and not df.isna().all().all()]
    if valid_findings:
        result = pd.concat(valid_findings, ignore_index=True)
    else:
        result = pd.DataFrame()


    result = result.sort_values("Anomaly_Score", ascending=False).reset_index(drop=True)
    result.insert(0, "Rank", range(1, len(result) + 1))
    return result


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
