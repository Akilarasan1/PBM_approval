import pandas as pd
from config.utils import *

def compute_code_stats(rejected):
    """Compute rejection code statistics."""
    
    if len(rejected) == 0:
        return pd.DataFrame(columns=['REJ_CODE_PREFIX', 'Count', 'Rejected_Amt', 'Pct', 'Description'])
    
    agg_dict = {'IS_REJECTED': 'count'}
    if 'TREAT_REJ_AMT' in rejected.columns:
        agg_dict['TREAT_REJ_AMT'] = 'sum'
    
    code_stats = (rejected.groupby('REJ_CODE_PREFIX')
                  .agg(agg_dict)
                  .reset_index())
    
    rename_map = {'IS_REJECTED': 'Count'}
    if 'TREAT_REJ_AMT' in code_stats.columns:
        rename_map['TREAT_REJ_AMT'] = 'Rejected_Amt'
    code_stats = code_stats.rename(columns=rename_map)
    
    if 'Rejected_Amt' not in code_stats.columns:
        code_stats['Rejected_Amt'] = 0
    
    code_stats['Pct'] = (code_stats['Count'] / len(rejected) * 100).round(1)
    code_stats['Description'] = code_stats['REJ_CODE_PREFIX'].map(CODE_DESC).fillna('Unknown')
    return code_stats.sort_values('Count', ascending=False)


def compute_drug_diag_combos(drug_df):
    """Compute drug-diagnosis combination statistics."""
    from config.utils import THRESHOLDS

    min_claims = THRESHOLDS['combo_min_claims']
    high_rate = THRESHOLDS['combo_high_risk_rate']
    gray_min = THRESHOLDS['combo_gray_min_rate']

    if 'DRUG_DIAG_COMBO' not in drug_df.columns or 'IS_REJECTED' not in drug_df.columns:
        empty = pd.DataFrame(columns=['DRUG_DIAG_COMBO', 'Total', 'Rejected', 'RejRate'])
        return empty, empty, empty
    
    combo = (drug_df.groupby('DRUG_DIAG_COMBO')
             .agg(Total=('IS_REJECTED','count'), Rejected=('IS_REJECTED','sum'))
             .reset_index())
    combo['RejRate'] = (combo['Rejected'] / combo['Total'] * 100).round(1)
    
    high_risk = combo[(combo['Total'] >= min_claims) & (combo['RejRate'] >= high_rate)].sort_values('RejRate', ascending=False)
    safe_combos = combo[(combo['Total'] >= min_claims) & (combo['RejRate'] == 0)].sort_values('Total', ascending=False)
    gray_combos = combo[(combo['Total'] >= min_claims) & (combo['RejRate'] >= gray_min) & (combo['RejRate'] < high_rate)].sort_values('RejRate', ascending=False)
    
    return high_risk, safe_combos, gray_combos


def compute_provider_stats(drug_df):
    """Compute provider statistics."""
    if 'DOC_LIC_NO' not in drug_df.columns or 'IS_REJECTED' not in drug_df.columns:
        return pd.DataFrame(columns=['DOC_LIC_NO', 'Claims', 'UniqueDrugs', 'RejRate', 'RejAmt'])
    
    agg_dict = {
        'IS_REJECTED': ('count', 'Claims'),
        'DRUG_CODE': ('nunique', 'UniqueDrugs') if 'DRUG_CODE' in drug_df.columns else None,
    }
    
    prov = drug_df.groupby('DOC_LIC_NO').agg(
        Claims=('IS_REJECTED', 'count'),
        UniqueDrugs=('DRUG_CODE', 'nunique') if 'DRUG_CODE' in drug_df.columns else ('IS_REJECTED', 'count'),
        RejRate=('IS_REJECTED', 'mean'),
        RejAmt=('TREAT_REJ_AMT', 'sum') if 'TREAT_REJ_AMT' in drug_df.columns else ('IS_REJECTED', 'sum')
    ).reset_index()
    
    prov['RejRate'] = (prov['RejRate'] * 100).round(1)
    prov['RejAmt'] = prov['RejAmt'].round(0)
    
    return prov


def compute_weekly_trends(drug_df):
    """Compute weekly claim volume and rejection rate."""
    if 'SERVICE_DT' not in drug_df.columns or drug_df['SERVICE_DT'].isna().all():
        return pd.DataFrame(columns=['Week', 'Total', 'Rejected', 'RejRate'])

    weekly = (
        drug_df.dropna(subset=['SERVICE_DT'])
        .assign(Week=lambda d: d['SERVICE_DT'].dt.to_period('W').astype(str))
        .groupby('Week', as_index=False)
        .agg(Total=('IS_REJECTED', 'count'), Rejected=('IS_REJECTED', 'sum'))
        .sort_values('Week')
    )
    weekly['RejRate'] = (weekly['Rejected'] / weekly['Total'] * 100).round(1)
    return weekly


#new ly added need to check


# ── REJECTION REASON KEYWORD HOTSPOTS (Documentation, Medical Necessity) ──
TEXT_REASON_COLUMNS = ['REJ_REMARKS', 'PBM_REJ_DESC', 'PAT_REJ_REMARKS']

REJECTION_REASON_KEYWORDS = {
    'Documentation': ['document'],  # catches documentation/absence of documentation/
                                      # lack of documentation/missing documentation, etc.
    'Medical Necessity': ['medically necessary', 'medical necessity'],
}


def _text_reason_mask(df, keywords):
    """OR-combined, case-insensitive keyword match across every available
    rejection-text column. Broader and more robust than matching one exact
    phrase in one column — the same reason is phrased differently across
    files/months in real exports (see: medical-necessity phrasing varies
    month to month), so an exact-string filter silently misses real cases."""
    text_cols = [c for c in TEXT_REASON_COLUMNS if c in df.columns]
    if not text_cols:
        return pd.Series(False, index=df.index)

    pattern = '|'.join(keywords)
    mask = pd.Series(False, index=df.index)
    for col in text_cols:
        mask = mask | df[col].str.contains(pattern, case=False, na=False, regex=True)
    return mask


def compute_rejection_reason_hotspots(drug_df, keywords):
    """
    Generic keyword-bucket rejection-reason analysis. Works identically for
    Documentation, Medical Necessity, or any other keyword list — no
    per-category hardcoded logic.

    Args:
        drug_df: DataFrame with claims
        keywords: list of substrings to match (case-insensitive, OR'd)

    Returns:
        (matched_claims, summary) tuple:
        matched_claims — the filtered claim-level rows (rejected only)
        summary — DataFrame: PA_PRIMARY_DIAG, DRUG_CODE, DOC_LIC_NO,
        Frequency — sorted by Frequency descending
    """
    if 'IS_REJECTED' not in drug_df.columns:
        return pd.DataFrame(), pd.DataFrame()

    rejected = drug_df[drug_df['IS_REJECTED'] == 1].copy()
    if rejected.empty:
        return pd.DataFrame(), pd.DataFrame()

    mask = _text_reason_mask(rejected, keywords)
    matched = rejected[mask]
    if matched.empty:
        return matched, pd.DataFrame()

    group_cols = [c for c in ['PA_PRIMARY_DIAG', 'DRUG_CODE', 'DOC_LIC_NO'] if c in matched.columns]
    if not group_cols:
        return matched, pd.DataFrame()

    summary = (
        matched.groupby(group_cols).size().reset_index(name='Frequency')
        .sort_values('Frequency', ascending=False).reset_index(drop=True)
    )
    return matched, summary


def compute_rejection_reason_monthly_trend(full_df, keywords):
    """Month-by-month count of rejections matching the given keyword bucket,
    across the full uploaded date range."""
    if 'SERVICE_DT' not in full_df.columns or 'IS_REJECTED' not in full_df.columns:
        return pd.DataFrame()

    dated = full_df[full_df['SERVICE_DT'].notna() & (full_df['IS_REJECTED'] == 1)].copy()
    if dated.empty:
        return pd.DataFrame()

    mask = _text_reason_mask(dated, keywords)
    matched = dated[mask].copy()
    if matched.empty:
        return pd.DataFrame()

    matched['Month'] = matched['SERVICE_DT'].dt.to_period('M').astype(str)
    return matched.groupby('Month').size().reset_index(name='Frequency').sort_values('Month')