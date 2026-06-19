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
    "spike_100pct": 25,      # Anomaly score boost for >100% changes
    "spike_200pct": 50,      # Boost for >200%
    "spike_500pct": 80,      # Boost for >500%
    "new_entity_min_claims": 5,    # Min claims for new entity to flag
    "high_rejection_jump": 15,     # % point jump in rejection rate
    "provider_surge_min_volume": 30,  # Min claims for provider volume to matter
    "fraud_score_critical": 75,
    "fraud_score_warning": 50,
    "fraud_score_info": 25,
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
    Filter emerging_findings for fraud-signal spikes (>100%, >200%, >500%).
    
    Parameters
    ----------
    emerging_findings : DataFrame
        Output from anomaly.run_emerging_pattern_scan()
    
    Returns
    -------
    DataFrame
        Fraud-relevant spikes with additional fraud-specific scoring
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


def _compute_fraud_score(row: pd.Series) -> float:
    """
    Blend statistical anomaly score with fraud-specific context.
    
    Higher score = higher fraud risk. Factors:
    - Base anomaly score (0-100 from anomaly.py)
    - Spike magnitude (>500% is worse than >100%)
    - Novelty (first-ever appearance)
    - Volume (higher volume = higher impact)
    """
    base_score = float(row["Anomaly_Score"]) if pd.notna(row["Anomaly_Score"]) else 0
    pct_change = float(row["Pct_Change"]) if pd.notna(row["Pct_Change"]) else 0
    is_novel = bool(row["Novel"]) if pd.notna(row["Novel"]) else False
    current_val = float(row["Current"]) if pd.notna(row["Current"]) else 0
    
    fraud_boost = 0
    
    # Spike magnitude boosts
    if abs(pct_change) > 500:
        fraud_boost += FRAUD_THRESHOLDS["spike_500pct"]
    elif abs(pct_change) > 200:
        fraud_boost += FRAUD_THRESHOLDS["spike_200pct"]
    elif abs(pct_change) > 100:
        fraud_boost += FRAUD_THRESHOLDS["spike_100pct"]
    
    # Novelty boost
    if is_novel:
        fraud_boost += 20
    
    # Volume impact (higher volume = higher fraud risk potential)
    if current_val > 100:
        fraud_boost += min(15, current_val / 100)
    
    final_score = min(base_score + fraud_boost, 100)
    return round(final_score, 1)


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
    """Extract diagnosis volume spikes with >100%, >200%, >500% flags."""
    if emerging_findings.empty:
        return pd.DataFrame()
    
    findings = emerging_findings[
        (emerging_findings["Dimension"] == "Diagnosis Drift")
        & (emerging_findings["Metric"] == "Claim Volume")
    ].copy()
    
    if findings.empty:
        return pd.DataFrame()
    
    findings["Spike_Category"] = findings["Pct_Change"].apply(_categorize_spike)
    findings = findings[findings["Spike_Category"] != "Normal"].copy()
    
    return findings.sort_values("Pct_Change", ascending=False)


def extract_drug_spikes(emerging_findings: pd.DataFrame) -> pd.DataFrame:
    """Extract drug utilization spikes."""
    if emerging_findings.empty:
        return pd.DataFrame()
    
    findings = emerging_findings[
        (emerging_findings["Dimension"] == "Drug Utilization")
        & (emerging_findings["Metric"] == "Claim Volume")
    ].copy()
    
    if findings.empty:
        return pd.DataFrame()
    
    findings["Spike_Category"] = findings["Pct_Change"].apply(_categorize_spike)
    findings = findings[findings["Spike_Category"] != "Normal"].copy()
    
    return findings.sort_values("Pct_Change", ascending=False)


def extract_provider_surges(emerging_findings: pd.DataFrame) -> pd.DataFrame:
    """Extract provider volume surges."""
    if emerging_findings.empty:
        return pd.DataFrame()
    
    findings = emerging_findings[
        (emerging_findings["Dimension"] == "Provider Behavior")
        & (emerging_findings["Metric"] == "Claim Volume")
        & (emerging_findings["Current"] >= FRAUD_THRESHOLDS["provider_surge_min_volume"])
    ].copy()
    
    if findings.empty:
        return pd.DataFrame()
    
    findings["Spike_Category"] = findings["Pct_Change"].apply(_categorize_spike)
    findings = findings[findings["Spike_Category"] != "Normal"].copy()
    
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


def _categorize_spike(pct_change) -> str:
    """Categorize spike magnitude."""
    if pd.isna(pct_change):
        return "Unknown"
    pct = abs(float(pct_change))
    if pct >= 500:
        return "Critical (>500%)"
    elif pct >= 200:
        return "High (200-500%)"
    elif pct >= 100:
        return "Moderate (100-200%)"
    else:
        return "Normal (<100%)"


# ── INVESTIGATION QUEUE PRIORITIZATION ──────────────────────────
def build_investigation_queue(
    emerging_findings: pd.DataFrame,
    max_findings: int = 50,
) -> pd.DataFrame:
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
        "Pct_Change",
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
    if "Pct_Change" in queue.columns:
        queue["Pct_Change"] = queue["Pct_Change"].apply(
            lambda x: f"{x:+.0f}%" if pd.notna(x) else "N/A"
        )
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
