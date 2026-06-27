"""
fraud_detection.py — Fraud-risk lens on statistical anomalies.

This module wraps the generic anomaly.py engine with fraud-specific filtering,
scoring, and prioritization. It reuses emerging_findings but frames them through
a fraud risk lens:

  - Spike magnitude (>100%, >200%, >500%) → fraud risk signals
  - Novel entities (new drugs, new diagnoses) → potential evasion attempts
  - Provider volume surges → claim pattern abuse
  - Facility rejection rate changes → formulary gaming
  
No hardcoded business rules — purely statistical drift with fraud context.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


# ── FRAUD-SPECIFIC SEVERITY THRESHOLDS ──────────────────────────
FRAUD_THRESHOLDS = {
    "signal_p70": 20,
    "signal_p85": 40,
    "signal_p95": 70,

    "new_entity_min_claims": 5,
    "provider_surge_min_volume": 30,

    "fraud_score_info": 25,
    "fraud_score_warning": 50,
    "fraud_score_critical": 75,
}


# ── FRAUD SIGNAL TYPES ──────────────────────────────────────────
FRAUD_SIGNAL_TYPES = {
    "Diagnosis Spike": {
        "dimension": "Diagnosis Drift",
        "metric": "Claim Volume",
        "icon": "📈",
        "risk_level": "medium",
    },
    "Drug Spike": {
        "dimension": "Drug Utilization",
        "metric": "Claim Volume",
        "icon": "💊",
        "risk_level": "medium",
    },
    "Provider Surge": {
        "dimension": "Provider Behavior",
        "metric": "Claim Volume",
        "icon": "🏥",
        "risk_level": "high",
    },
    "Provider Rejection Rate Jump": {
        "dimension": "Provider Behavior",
        "metric": "Rejection Rate %",
        "icon": "📊",
        "risk_level": "high",
    },
    "New Diagnosis": {
        "dimension": "Diagnosis Drift",
        "metric": "New Entity",
        "icon": "🆕",
        "risk_level": "medium",
    },
    "New Drug": {
        "dimension": "Drug Utilization",
        "metric": "New Entity",
        "icon": "🆕",
        "risk_level": "medium",
    },
    "New Drug-Diagnosis Combo": {
        "dimension": "New Drug-Diagnosis Combo",
        "metric": "New Combination",
        "icon": "⚠️",
        "risk_level": "medium",
    },
}


# ── EXTRACT FRAUD-RELEVANT FINDINGS ────────────────────────────
def extract_fraud_spikes(emerging_findings: pd.DataFrame) -> pd.DataFrame:
    """
    Filter emerging_findings for fraud-relevant signals and rank them by a
    fraud-specific score.

    Note: this works entirely off `Anomaly_Score` (already a plain 0-100
    number). The engine intentionally doesn't expose raw Pct_Change/ZScore
    columns to keep findings stakeholder-readable, so fraud scoring here
    must not depend on them either.

    Parameters
    ----------
    emerging_findings : DataFrame
        Output from anomaly.run_emerging_pattern_scan()
    
    Returns
    -------
    DataFrame
        Fraud-relevant findings with additional fraud-specific scoring
    """
    if emerging_findings.empty:
        return pd.DataFrame()
    
    findings = emerging_findings.copy()
    
    # Filter to volume/rate changes (not just any anomaly)
    findings = findings[
        findings["Metric"].isin([
            "Claim Volume",
            "Rejection Rate %",
            "New Entity",
            "New Combination",
            "% Share of All Rejections",
        ])
    ].copy()
    
    if findings.empty:
        return pd.DataFrame()
    
    # How strong is this finding relative to everything else found this
    # month? A plain rank-percentile of Anomaly_Score — no raw stats needed.
    findings["Anomaly_Percentile"] = findings["Anomaly_Score"].rank(pct=True) * 100

    # Add fraud-specific scoring
    findings["Fraud_Score"] = findings.apply(_compute_fraud_score, axis=1)
    findings["Fraud_Severity"] = findings["Fraud_Score"].apply(_fraud_severity)
    findings["Signal_Type"] = findings.apply(_classify_signal_type, axis=1)
    
    # Rank by fraud score
    findings = findings.sort_values("Fraud_Score", ascending=False).reset_index(drop=True)
    # Insert or overwrite `Rank`, then ensure it's the first column
    rank_values = range(1, len(findings) + 1)
    if "Rank" in findings.columns:
        findings["Rank"] = list(rank_values)
        cols = findings.columns.tolist()
        cols.remove("Rank")
        findings = findings[["Rank"] + cols]
    else:
        findings.insert(0, "Rank", rank_values)
    
    return findings


def _fraud_severity(score: float) -> str:
    """Convert fraud score to severity level."""
    if score >= FRAUD_THRESHOLDS["fraud_score_critical"]:
        return "critical"
    if score >= FRAUD_THRESHOLDS["fraud_score_warning"]:
        return "warning"
    if score >= FRAUD_THRESHOLDS["fraud_score_info"]:
        return "info"
    return "success"


def _classify_signal_type(row: pd.Series) -> str:
    """Classify the fraud signal type."""
    dimension = row["Dimension"]
    metric = row["Metric"]
    is_novel = row["Novel"]
    
    if is_novel:
        if "Drug" in dimension or "Drug" in row["Entity"]:
            return "New Drug"
        elif "Diagnosis" in dimension:
            return "New Diagnosis"
        elif "Combo" in dimension:
            return "New Drug-Diagnosis Combo"
    
    if "Diagnosis" in dimension and "Volume" in metric:
        return "Diagnosis Spike"
    elif "Drug" in dimension and "Volume" in metric:
        return "Drug Spike"
    elif "Provider" in dimension and "Volume" in metric:
        return "Provider Surge"
    elif "Provider" in dimension and "Rejection" in metric:
        return "Provider Rejection Rate Jump"
    
    return "Other Anomaly"


# ── CATEGORIZED EXTRACTION ──────────────────────────────────────
def extract_diagnosis_spikes(emerging_findings: pd.DataFrame) -> pd.DataFrame:
    """Extract diagnosis volume spikes, ranked by signal strength."""
    if emerging_findings.empty:
        return pd.DataFrame()
    
    findings = emerging_findings[
        (emerging_findings["Dimension"] == "Diagnosis Drift")
        & (emerging_findings["Metric"] == "Claim Volume")
    ].copy()
    
    if findings.empty:
        return pd.DataFrame()
    
    findings["Signal_Strength"] = findings["Anomaly_Score"].apply(_categorize_signal_strength)
    findings = findings[findings["Signal_Strength"] != "Normal (<25)"].copy()
    
    return findings.sort_values("Anomaly_Score", ascending=False)


def extract_drug_spikes(emerging_findings: pd.DataFrame) -> pd.DataFrame:
    """Extract drug utilization spikes, ranked by signal strength."""
    if emerging_findings.empty:
        return pd.DataFrame()
    
    findings = emerging_findings[
        (emerging_findings["Dimension"] == "Drug Utilization")
        & (emerging_findings["Metric"] == "Claim Volume")
    ].copy()
    
    if findings.empty:
        return pd.DataFrame()
    
    findings["Signal_Strength"] = findings["Anomaly_Score"].apply(_categorize_signal_strength)
    findings = findings[findings["Signal_Strength"] != "Normal (<25)"].copy()
    
    return findings.sort_values("Anomaly_Score", ascending=False)


def extract_provider_surges(emerging_findings: pd.DataFrame) -> pd.DataFrame:
    """Extract provider volume surges, ranked by signal strength."""
    if emerging_findings.empty:
        return pd.DataFrame()
    
    findings = emerging_findings[
        (emerging_findings["Dimension"] == "Provider Behavior")
        & (emerging_findings["Metric"] == "Claim Volume")
        & (emerging_findings["Current"] >= FRAUD_THRESHOLDS["provider_surge_min_volume"])
    ].copy()
    
    if findings.empty:
        return pd.DataFrame()
    
    findings["Signal_Strength"] = findings["Anomaly_Score"].apply(_categorize_signal_strength)
    findings = findings[findings["Signal_Strength"] != "Normal (<25)"].copy()
    
    return findings.sort_values("Anomaly_Score", ascending=False)


def extract_provider_rejection_spikes(emerging_findings: pd.DataFrame) -> pd.DataFrame:
    """Extract provider rejection rate anomalies."""
    if emerging_findings.empty:
        return pd.DataFrame()
    
    findings = emerging_findings[
        (emerging_findings["Dimension"] == "Provider Behavior")
        & (emerging_findings["Metric"] == "Rejection Rate %")
    ].copy()
    
    if findings.empty:
        return pd.DataFrame()
    
    return findings.sort_values("Anomaly_Score", ascending=False)


def extract_new_entities(emerging_findings: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract novel diagnoses and drugs.
    
    Returns
    -------
    (new_diagnoses, new_drugs)
    """
    if emerging_findings.empty:
        return pd.DataFrame(), pd.DataFrame()
    
    novel = emerging_findings[emerging_findings["Novel"] == True].copy()
    
    new_diagnoses = novel[
        (novel["Dimension"] == "Diagnosis Drift")
        | (novel["Dimension"].str.contains("Diagnosis", na=False))
    ].copy()
    
    new_drugs = novel[
        (novel["Dimension"] == "Drug Utilization")
        | (novel["Dimension"].str.contains("Drug", na=False))
    ].copy()
    
    return new_diagnoses, new_drugs


def _categorize_signal_strength(score) -> str:
    """Categorize signal strength from the plain 0-100 Anomaly_Score —
    no raw stats needed, matches the same bands anomaly.py uses for
    Severity so labels stay consistent across the dashboard."""
    if pd.isna(score):
        return "Unknown"
    s = float(score)
    if s >= 80:
        return "Critical (80-100)"
    elif s >= 55:
        return "High (55-79)"
    elif s >= 25:
        return "Moderate (25-54)"
    else:
        return "Normal (<25)"
    

def _compute_fraud_score(row):
    base_score = float(row["Anomaly_Score"]) if pd.notna(row["Anomaly_Score"]) else 0
    percentile = float(row["Anomaly_Percentile"]) if pd.notna(row["Anomaly_Percentile"]) else 0
    is_novel = bool(row["Novel"]) if pd.notna(row["Novel"]) else False
    current_val = float(row["Current"]) if pd.notna(row["Current"]) else 0

    fraud_boost = 0
    if percentile >= 95:
        fraud_boost += FRAUD_THRESHOLDS["signal_p95"]
    elif percentile >= 85:
        fraud_boost += FRAUD_THRESHOLDS["signal_p85"]
    elif percentile >= 70:
        fraud_boost += FRAUD_THRESHOLDS["signal_p70"]

    if is_novel:
        fraud_boost += 20
    if current_val > 100:
        fraud_boost += min(15, current_val / 50)

    return round(min(base_score + fraud_boost, 100), 1)



# ── INVESTIGATION QUEUE PRIORITIZATION ──────────────────────────
def build_investigation_queue(emerging_findings: pd.DataFrame,max_findings: int = 50,) -> pd.DataFrame:
    """
    Build a prioritized investigation queue combining all fraud signals.
    
    Parameters
    ----------
    emerging_findings : DataFrame
        Output from anomaly.run_emerging_pattern_scan()
    max_findings : int
        Max findings to return
    
    Returns
    -------
    DataFrame
        Ranked findings with fraud context, ready for analyst investigation
    """
    if emerging_findings.empty:
        return pd.DataFrame()
    
    findings = extract_fraud_spikes(emerging_findings).copy()
    
    if findings.empty:
        return pd.DataFrame()
    
    # Select key columns for display
    display_cols = [
        "Rank",
        "Signal_Type",
        "Entity",
        "Current",
        "Baseline_Mean",
        "Anomaly_Score",
        "Fraud_Score",
        "Fraud_Severity",
        "Reason",
    ]
    
    available_cols = [c for c in display_cols if c in findings.columns]
    queue = findings[available_cols].head(max_findings).copy()
    
    # Format for display
    if "Current" in queue.columns:
        queue["Current"] = queue["Current"].apply(lambda x: f"{x:,.1f}")
    if "Baseline_Mean" in queue.columns:
        queue["Baseline_Mean"] = queue["Baseline_Mean"].apply(
            lambda x: f"{x:,.1f}" if pd.notna(x) else "N/A"
        )
    if "Anomaly_Score" in queue.columns:
        queue["Anomaly_Score"] = queue["Anomaly_Score"].apply(lambda x: f"{x:.0f}/100")
    if "Fraud_Score" in queue.columns:
        queue["Fraud_Score"] = queue["Fraud_Score"].apply(lambda x: f"{x:.0f}/100")
    
    return queue


# ── SUMMARY STATISTICS ──────────────────────────────────────────
def summarize_fraud_findings(emerging_findings: pd.DataFrame) -> Dict:
    """
    Quick summary of fraud findings for KPI display.
    
    Returns dict with counts by severity and type.
    """
    if emerging_findings.empty:
        return {
            "total": 0,
            "critical": 0,
            "warning": 0,
            "info": 0,
            "diagnosis_spikes": 0,
            "drug_spikes": 0,
            "provider_surges": 0,
            "new_entities": 0,
        }
    
    fraud_findings = extract_fraud_spikes(emerging_findings)
    
    if fraud_findings.empty:
        return {
            "total": 0,
            "critical": 0,
            "warning": 0,
            "info": 0,
            "diagnosis_spikes": 0,
            "drug_spikes": 0,
            "provider_surges": 0,
            "new_entities": 0,
        }
    
    return {
        "total": len(fraud_findings),
        "critical": int((fraud_findings["Fraud_Severity"] == "critical").sum()),
        "warning": int((fraud_findings["Fraud_Severity"] == "warning").sum()),
        "info": int((fraud_findings["Fraud_Severity"] == "info").sum()),
        "diagnosis_spikes": len(extract_diagnosis_spikes(emerging_findings)),
        "drug_spikes": len(extract_drug_spikes(emerging_findings)),
        "provider_surges": len(extract_provider_surges(emerging_findings)),
        "new_entities": len(extract_new_entities(emerging_findings)[0])
        + len(extract_new_entities(emerging_findings)[1]),
    }
