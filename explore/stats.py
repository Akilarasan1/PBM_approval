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