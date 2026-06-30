"""Data processing functions for PBM Dashboard."""

import pandas as pd
import numpy as np
from config.utils import CODE_DESC
import io
from explore.stats import compute_code_stats, compute_drug_diag_combos, compute_provider_stats

# ── REQUIRED & OPTIONAL COLUMNS ────────────────────────────────
REQUIRED_COLUMNS = {
    'INS_TREAT_DESC', 'PBM_APPR_STS', 'PBM_REJ_CODE',
    'DRUG_CODE', 'PA_PRIMARY_DIAG', 'TREAT_REJ_AMT'
}

OPTIONAL_COLUMNS = {
    'SERVICE_DT', "SERVICE_TYPE", 'PA_MEM_AGE', 'MEM_GENDER', 'DOC_LIC_NO', 'DOC_NAME',
    'PA_FLAG_REASON', 'REJ_REMARKS', 'PBM_REJ_DESC', 'PAT_REJ_REMARKS',
    'PROV_TREAT_DESC', 'PRIMARY_DIAG', 'TREAT_EST_AMT', 'TREAT_APPR_AMT',"PA_QTY" 
}


def validate_columns(df):
    """
    Validate DataFrame has required columns and return missing columns.
    
    Returns:
        tuple: (missing_required, missing_optional)
    """
    missing_required = REQUIRED_COLUMNS - set(df.columns)
    missing_optional = OPTIONAL_COLUMNS - set(df.columns)
    return missing_required, missing_optional


def compute_ncov_coverage_trend(full_df, min_claims=5):
    """
    Monthly Not-Covered (NCOV) trend per drug across the full uploaded date
    range, plus the drugs that genuinely FLIP — NCOV-rejected in one month,
    approved in a different month for the same drug. This is the broader,
    multi-month version of the Ever_Approved check in
    compute_new_drugs_always_ncov: that function only looks at drugs that
    are 100%-rejected in the CURRENT month; this one tracks every drug's
    coverage behavior across every month on record, regardless of whether
    any single month hits 100%.

    Args:
        full_df: DataFrame spanning the full uploaded date range
        min_claims: minimum total claims (summed across all months) for a
            drug to be considered — filters out single-claim noise

    Returns:
        (trend, flips) tuple of DataFrames:
        trend — one row per (DRUG_CODE, Month): Claims, NCOV_Rejections,
            Approved_Claims, NCOV_Rate_%
        flips — one row per drug with a genuine month-to-month flip:
            DRUG_CODE, DRUG_NAME (if available), NCOV_Months, Approved_Months,
            Total_NCOV_Rejections, Total_Approved_Claims, Top_Diagnosis,
            Top_Provider
    """
    required = {'SERVICE_DT', 'DRUG_CODE', 'IS_REJECTED', 'REJ_CODE_PREFIX'}
    empty = (pd.DataFrame(), pd.DataFrame())
    if full_df is None or not required.issubset(full_df.columns):
        return empty

    dated = full_df[full_df['SERVICE_DT'].notna()].copy()
    if dated.empty:
        return empty

    dated['Month'] = dated['SERVICE_DT'].dt.to_period('M').astype(str)
    dated['REJ_CODE_PREFIX'] = dated['REJ_CODE_PREFIX'].astype(str).str.upper().str.strip()
    dated['IS_NCOV'] = (dated['IS_REJECTED'] == 1) & (dated['REJ_CODE_PREFIX'] == 'NCOV')

    trend = (
        dated.groupby(['DRUG_CODE', 'Month'])
        .agg(
            Claims=('IS_REJECTED', 'count'),
            NCOV_Rejections=('IS_NCOV', 'sum'),
            Approved_Claims=('IS_REJECTED', lambda s: int((s == 0).sum())),
        )
        .reset_index()
    )
    trend['NCOV_Rate_%'] = (trend['NCOV_Rejections'] / trend['Claims'] * 100).round(1)

    drug_totals = trend.groupby('DRUG_CODE')['Claims'].sum()
    eligible_drugs = set(drug_totals[drug_totals >= min_claims].index)

    flip_rows = []
    for drug, g in trend[trend['DRUG_CODE'].isin(eligible_drugs)].groupby('DRUG_CODE'):
        ncov_months = set(g.loc[g['NCOV_Rejections'] > 0, 'Month'])
        approved_months = set(g.loc[g['Approved_Claims'] > 0, 'Month'])
        flip_months = approved_months - ncov_months
        if not ncov_months or not flip_months:
            continue

        drug_rows = dated[dated['DRUG_CODE'] == drug]
        top_diag = None
        if 'PA_PRIMARY_DIAG' in drug_rows.columns:
            mode_vals = drug_rows['PA_PRIMARY_DIAG'].dropna().mode()
            top_diag = mode_vals.iloc[0] if len(mode_vals) else None
        top_provider = None
        if 'DOC_LIC_NO' in drug_rows.columns:
            mode_vals = drug_rows['DOC_LIC_NO'].dropna().mode()
            top_provider = mode_vals.iloc[0] if len(mode_vals) else None

        flip_rows.append({
            'DRUG_CODE': drug,
            'NCOV_Months': ', '.join(sorted(ncov_months)),
            'Approved_Months': ', '.join(sorted(flip_months)),
            'Total_NCOV_Rejections': int(g['NCOV_Rejections'].sum()),
            'Total_Approved_Claims': int(g['Approved_Claims'].sum()),
            'Top_Diagnosis': top_diag if top_diag else '—',
            'Top_Provider': top_provider if top_provider else '—',
        })

    flips = pd.DataFrame(flip_rows)
    if not flips.empty:
        if 'DRUG_NAME' in full_df.columns:
            names = full_df[['DRUG_CODE', 'DRUG_NAME']].drop_duplicates('DRUG_CODE')
            flips = flips.merge(names, on='DRUG_CODE', how='left')
        flips = flips.sort_values('Total_NCOV_Rejections', ascending=False).reset_index(drop=True)

    return trend, flips


def ensure_datetime(df, col):
    """Convert column to datetime, filter out NaT."""
    if col in df.columns:
        df = df.copy()
        df[col] = pd.to_datetime(df[col], errors='coerce')
        return df[df[col].notna()]
    return df

def get_current_month(df):
    """Filter to current month claims."""
    df = ensure_datetime(df, 'SERVICE_DT')
    if df.empty:
        return df
    end = df['SERVICE_DT'].max()
    start = end.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    return df[(df['SERVICE_DT'] >= start) & (df['SERVICE_DT'] <= end)]

def calc_ncov_stats(df, min_claims):
    """Calculate NCOV statistics for current month."""
    if df.empty:
        return pd.DataFrame()
    
    df = df.copy()
    df['REJ_CODE_PREFIX'] = df['REJ_CODE_PREFIX'].astype(str).str.upper().str.strip()
    df['IS_NCOV'] = df['REJ_CODE_PREFIX'].str.contains('NCOV', na=False)
    df['IS_REJECTED'] = df['IS_REJECTED'].astype(int)
    
    stats = df.groupby('DRUG_CODE').agg(
        Total_Claims=('IS_REJECTED', 'count'),
        Rejected_Claims=('IS_REJECTED', 'sum'),
        NCOV_Rejections=('IS_NCOV', 'sum'),
        Rejected_Amount=('TREAT_REJ_AMT', 'sum') if 'TREAT_REJ_AMT' in df.columns else ('IS_REJECTED', 'sum')
    ).reset_index()
    
    stats['RejRate_%'] = (stats['Rejected_Claims'] / stats['Total_Claims'] * 100).round(1)
    
    return stats[
        (stats['Total_Claims'] >= min_claims) &
        (stats['Rejected_Claims'] == stats['Total_Claims']) &
        (stats['NCOV_Rejections'] == stats['Rejected_Claims']) &
        (stats['NCOV_Rejections'] > 0)
    ].copy()

def add_history(result, full_df, current_month):
    """Add historical NCOV and approval data."""
    if full_df is None or full_df.empty:
        return result
    
    full_df = ensure_datetime(full_df, 'SERVICE_DT')
    if full_df.empty:
        return result
    
    # Create combos if needed
    if 'DRUG_DIAG_COMBO' not in current_month.columns:
        return result
    
    full_df = full_df.copy()
    if 'DRUG_DIAG_COMBO' not in full_df.columns:
        return result
    
    # Get flagged combos
    combos = current_month[current_month['DRUG_CODE'].isin(set(result['DRUG_CODE']))][['DRUG_CODE', 'DRUG_DIAG_COMBO']].drop_duplicates()
    
    # NCOV history
    ncov = full_df[full_df['REJ_CODE_PREFIX'].astype(str).str.upper().str.contains('NCOV', na=False)][['DRUG_DIAG_COMBO', 'SERVICE_DT']]
    ncov_sum = ncov.groupby('DRUG_DIAG_COMBO')['SERVICE_DT'].agg(First_NCOV_Date='min', Last_NCOV_Date='max').reset_index()
    
    # Approval history
    full_df['IS_REJECTED'] = full_df['IS_REJECTED'].astype(int)
    approved = full_df[full_df['IS_REJECTED'] == 0][['DRUG_DIAG_COMBO', 'SERVICE_DT']]
    app_sum = approved.groupby('DRUG_DIAG_COMBO')['SERVICE_DT'].agg(
        Last_Approval_Date='max', Approved_Claims_Count='count'
    ).reset_index()
    
    # Combine
    combo = combos.merge(ncov_sum, on='DRUG_DIAG_COMBO', how='left').merge(app_sum, on='DRUG_DIAG_COMBO', how='left')
    combo['Approved_After_First_NCOV'] = combo['Last_Approval_Date'] > combo['First_NCOV_Date']
    combo['NCOV_Duration_Days'] = (combo['Last_NCOV_Date'] - combo['First_NCOV_Date']).dt.days
    
    # Aggregate to drug level
    drug = combo.groupby('DRUG_CODE').agg(
        First_NCOV_Date=('First_NCOV_Date', 'min'),
        Last_NCOV_Date=('Last_NCOV_Date', 'max'),
        NCOV_Duration_Days=('NCOV_Duration_Days', 'max'),
        Ever_Approved=('Approved_Claims_Count', lambda s: s.notna().any()),
        Last_Approval_Date=('Last_Approval_Date', 'max'),
        Approved_Claims_Count=('Approved_Claims_Count', 'sum'),
        Approved_After_First_NCOV=('Approved_After_First_NCOV', 'any')
    ).reset_index()
    
    drug['Approved_Claims_Count'] = drug['Approved_Claims_Count'].fillna(0).astype(int)
    
    # Merge
    drop_cols = ['First_NCOV_Date', 'Last_NCOV_Date', 'NCOV_Duration_Days', 'Ever_Approved', 
                 'Last_Approval_Date', 'Approved_Claims_Count', 'Approved_After_First_NCOV']
    result = result.drop(columns=[c for c in drop_cols if c in result.columns], errors='ignore')
    result = result.merge(drug, on='DRUG_CODE', how='left')
    
    # Fill missing
    for col in ['Ever_Approved', 'Approved_After_First_NCOV']:
        result[col] = result[col].fillna(False)
    result['Approved_Claims_Count'] = result['Approved_Claims_Count'].fillna(0).astype(int)
    
    return result

def format_output(df):
    """Format and sort final output."""
    if df.empty:
        return df
    
    df = df.copy()
    for col in ['First_NCOV_Date', 'Last_NCOV_Date', 'Last_Approval_Date']:
        if col in df.columns:
            df[col] = df[col].dt.strftime('%Y-%m-%d')
    
    df['Coverage_Status'] = np.select(
        [~df['Ever_Approved'], 
         pd.to_datetime(df['Last_Approval_Date'], errors='coerce') < pd.to_datetime(df['First_NCOV_Date'], errors='coerce')],
        ['Never Covered', 'Coverage Stopped'], 
        'Mixed History'
    )
    
    return df.sort_values(['Ever_Approved', 'NCOV_Rejections'], ascending=[False, False])

def compute_new_drugs_always_ncov(drug_df, full_df=None, min_claims=5):
    """Find ALL drug codes with 100% NCOV rejection in the current month."""
    required = ['SERVICE_DT', 'DRUG_CODE', 'IS_REJECTED', 'REJ_CODE_PREFIX']
    if not all(c in drug_df.columns for c in required):
        return pd.DataFrame()
    
    current = get_current_month(drug_df)
    if current.empty:
        return pd.DataFrame()
    
    result = calc_ncov_stats(current, min_claims)
    if result.empty:
        return pd.DataFrame()
    
    if 'DRUG_NAME' in drug_df.columns:
        names = drug_df[['DRUG_CODE', 'DRUG_NAME']].drop_duplicates('DRUG_CODE')
        result = result.merge(names, on='DRUG_CODE', how='left')
    
    result = add_history(result, full_df, current)
    return format_output(result)


def compute_provider_investigation(drug_df, min_claims=20):
    """Build prioritized provider investigation queue."""
    required = {'DOC_LIC_NO', 'REJ_CODE_PREFIX', 'IS_REJECTED'}
    if not required.issubset(drug_df.columns):
        return pd.DataFrame()

    investigation = (drug_df.groupby('DOC_LIC_NO').agg(Total_Claims=('IS_REJECTED', 'count'),Rejections=('IS_REJECTED', 'sum'),RejAmt=('TREAT_REJ_AMT', 'sum'),
        ).reset_index())

    mnec_counts = (drug_df[drug_df['REJ_CODE_PREFIX'] == 'MNEC'].groupby('DOC_LIC_NO')
        .size().reset_index(name='MNEC_Count'))
    
    ncov_counts = (
        drug_df[drug_df['REJ_CODE_PREFIX'] == 'NCOV'].groupby('DOC_LIC_NO').size().reset_index(name='NCOV_Count'))

    investigation = investigation.merge(mnec_counts, on='DOC_LIC_NO', how='left')
    investigation = investigation.merge(ncov_counts, on='DOC_LIC_NO', how='left')
    investigation = investigation.fillna(0)
    investigation['RejRate_%'] = (
        investigation['Rejections'] / investigation['Total_Claims'] * 100).round(1)

    investigation = investigation[investigation['Total_Claims'] >= min_claims]

    # --- Weighted Risk Score Calculation ---
    # Define max values for normalization (cap at 100% for rejection rate)
    MAX_REJ_RATE = 100
    MAX_REJ_AMT = investigation['RejAmt'].quantile(0.95)  # Cap outliers at 95th percentile
    MAX_MNEC = investigation['MNEC_Count'].quantile(0.95)
    MAX_NCOV = investigation['NCOV_Count'].quantile(0.95)
    
    # Cap extreme values to prevent outliers from dominating
    investigation['RejAmt_capped'] = investigation['RejAmt'].clip(upper=MAX_REJ_AMT)
    investigation['MNEC_Count_capped'] = investigation['MNEC_Count'].clip(upper=MAX_MNEC)
    investigation['NCOV_Count_capped'] = investigation['NCOV_Count'].clip(upper=MAX_NCOV)
    
    # Normalize each metric to 0-100 scale
    investigation['Rejection_Score'] = (investigation['RejRate_%'] / MAX_REJ_RATE * 100).round(1)
    
    investigation['Financial_Score'] = (investigation['RejAmt_capped'] / MAX_REJ_AMT * 100).round(1)
    
    investigation['MNEC_Score'] = (investigation['MNEC_Count_capped'] / MAX_MNEC * 100).round(1)
    
    investigation['NCOV_Score'] = (investigation['NCOV_Count_capped'] / MAX_NCOV * 100).round(1)
    
    # Weighted composite score (configurable weights)
    weights = {
        'rejection': 0.40,   # Highest weight - primary indicator
        'financial': 0.20,   # Financial impact matters
        'mnec': 0.20,        # Specific fraud pattern
        'ncov': 0.20,        # Specific fraud pattern
    }
    
    investigation['RiskScore'] = (
        investigation['Rejection_Score'] * weights['rejection'] +
        investigation['Financial_Score'] * weights['financial'] +
        investigation['MNEC_Score'] * weights['mnec'] +
        investigation['NCOV_Score'] * weights['ncov']
    ).round(1)
    
    # Risk categorization for business users
    investigation['RiskLevel'] = pd.cut(
        investigation['RiskScore'],
        bins=[0, 25, 50, 75, 100],
        labels=['Low', 'Medium', 'High', 'Critical'],
        include_lowest=True
    )

    # --- Build all reasons (not just first match) ---
    investigation['flag_high_rejection'] = investigation['RejRate_%'] > 35
    investigation['flag_high_mnec'] = investigation['MNEC_Count'] > 80
    investigation['flag_frequent_ncov'] = investigation['NCOV_Count'] > 20
    investigation['flag_high_financial'] = investigation['RejAmt'] > 100000
    
    reason_mapping = [
        ('flag_high_rejection', '🔴 High Rejection Rate'),
        ('flag_high_mnec', '⚠️ High MNEC Activity'),
        ('flag_frequent_ncov', '🟡 Frequent NCOV'),
        ('flag_high_financial', '💰 High Financial Impact'),
    ]
    
    def combine_reasons(row):
        reasons = [label for col, label in reason_mapping if row[col]]
        return ' | '.join(reasons) if reasons else 'Monitor'
    
    investigation['InvestigationReason'] = investigation.apply(combine_reasons, axis=1)

    if 'DOC_NAME' in drug_df.columns:
        names = drug_df[['DOC_LIC_NO', 'DOC_NAME']].dropna().drop_duplicates('DOC_LIC_NO')
        investigation = investigation.merge(names, on='DOC_LIC_NO', how='left')

    # Sort by RiskScore descending (highest risk first)
    return investigation.sort_values('RiskScore', ascending=False)


def compute_payment_anomalies(drug_df):
    """
    Detect financial data-integrity anomalies in the paid/requested/rejected
    amount fields. Two independent issues, kept separate because they need
    different follow-up:

    1. rejected_paid — claims marked PBM_REJECT (IS_REJECTED=1) that still
       carry a non-zero TREAT_APPR_AMT (paid amount). A rejected claim
       should show $0 paid; this is almost certainly a field-population
       bug rather than a real payment.
    2. Overpayment — TREAT_APPR_AMT exceeds TREAT_EST_AMT (paid more than
       was requested), split into:
         - zero_requested: TREAT_EST_AMT == 0 but something was still paid.
           Usually a missing/zero source value, not a genuine overpayment.
         - genuine_overpayment: TREAT_EST_AMT > 0 and still exceeded — the
           system paid out more than was actually requested.

    Args:
        drug_df: DataFrame with claims

    Returns:
        dict of DataFrames: {'rejected_paid', 'zero_requested', 'genuine_overpayment'}
    """
    empty = {'rejected_paid': pd.DataFrame(), 'zero_requested': pd.DataFrame(),
             'genuine_overpayment': pd.DataFrame()}

    required = {'TREAT_APPR_AMT', 'TREAT_EST_AMT', 'IS_REJECTED'}
    if not required.issubset(drug_df.columns):
        return empty
    
    dru = drug_df['IS_REJECTED'].value_counts(dropna=False)


    df = drug_df.copy()
    df['TREAT_APPR_AMT'] = df['TREAT_APPR_AMT'].fillna(0)
    df['TREAT_EST_AMT'] = df['TREAT_EST_AMT'].fillna(0)

    rejected_paid = df[(df['IS_REJECTED'] == 1) & (df['TREAT_APPR_AMT'] > 0)].copy()

    overpaid = df[df['TREAT_APPR_AMT'] > df['TREAT_EST_AMT']].copy()
    overpaid['Excess_Amt'] = (overpaid['TREAT_APPR_AMT'] - overpaid['TREAT_EST_AMT']).round(2)

    zero_requested = overpaid[overpaid['TREAT_EST_AMT'] == 0].copy()
    genuine_overpayment = overpaid[overpaid['TREAT_EST_AMT'] > 0].copy()

    return {
        'rejected_paid': rejected_paid,
        'zero_requested': zero_requested,
        'genuine_overpayment': genuine_overpayment,
    }


def summarize_payment_anomalies(anomalies):
    """Small KPI summary dict for the Payment Integrity section."""
    rp = anomalies.get('rejected_paid', pd.DataFrame())
    zr = anomalies.get('zero_requested', pd.DataFrame())
    go = anomalies.get('genuine_overpayment', pd.DataFrame())
    return {
        'rejected_paid_count': len(rp),
        'rejected_paid_amt': round(rp['TREAT_APPR_AMT'].sum(), 2) if len(rp) else 0,
        'zero_requested_count': len(zr),
        'zero_requested_amt': round(zr['Excess_Amt'].sum(), 2) if len(zr) else 0,
        'genuine_overpayment_count': len(go),
        'genuine_overpayment_amt': round(go['Excess_Amt'].sum(), 2) if len(go) else 0,
    }


def compute_provider_detail(drug_df, doc_lic_no):
    """Return drill-down stats for a single provider."""
    subset = drug_df[drug_df['DOC_LIC_NO'] == doc_lic_no].copy()
    if subset.empty:
        return None

    top_codes = (
        subset[subset['IS_REJECTED'] == 1]
        .groupby('REJ_CODE_PREFIX')
        .size()
        .reset_index(name='Count')
        .sort_values('Count', ascending=False)
        .head(10)
    )

    drug_cols = ['DRUG_CODE']
    if 'DRUG_NAME' in subset.columns:
        drug_cols.append('DRUG_NAME')

    top_drugs = (
        subset[subset['IS_REJECTED'] == 1]
        .groupby(drug_cols)
        .agg(Claims=('IS_REJECTED', 'count'), RejAmt=('TREAT_REJ_AMT', 'sum'))
        .reset_index()
        .sort_values('RejAmt', ascending=False)
        .head(10)
    )

    top_combos = (
        subset[subset['IS_REJECTED'] == 1]
        .groupby('DRUG_DIAG_COMBO')
        .size()
        .reset_index(name='Count')
        .sort_values('Count', ascending=False)
        .head(10)
    )

    doc_name = None
    if 'DOC_NAME' in subset.columns:
        names = subset['DOC_NAME'].dropna()
        doc_name = names.iloc[0] if len(names) else None

    return {
        'doc_name': doc_name,
        'total_claims': len(subset),
        'rejections': int(subset['IS_REJECTED'].sum()),
        'rej_rate': round(subset['IS_REJECTED'].mean() * 100, 1),
        'rej_amt': round(subset['TREAT_REJ_AMT'].sum(), 0),
        'top_codes': top_codes,
        'top_drugs': top_drugs,
        'top_combos': top_combos,
    }


def compute_diagnosis_treatment_matrix(drug_df, top_n_diag=15, top_n_drug=15):
    """
    Diagnosis x Drug cross-section, restricted to the top N diagnoses and
    top N drugs by claim volume (full cardinality is thousands of each —
    unrestricted would be unreadable as a heatmap/Sankey and likely too
    large to render). Feeds both the heatmap and the Sankey: every link in
    each chart is derived from this same table, so Diagnosis->Drug and
    Drug->Outcome flows stay numerically consistent with each other.

    Args:
        drug_df: DataFrame with claims
        top_n_diag: number of top diagnoses (by volume) to include
        top_n_drug: number of top drugs (by volume) to include

    Returns:
        DataFrame: PA_PRIMARY_DIAG, DRUG_CODE, Claims, Approved, Rejected,
        RejRate_% — one row per (diagnosis, drug) pair that actually
        co-occurs within the top-N x top-N subset.
    """
    required = {'PA_PRIMARY_DIAG', 'DRUG_CODE', 'IS_REJECTED'}
    if not required.issubset(drug_df.columns):
        return pd.DataFrame()

    top_diag = drug_df['PA_PRIMARY_DIAG'].value_counts().head(top_n_diag).index
    top_drug = drug_df['DRUG_CODE'].value_counts().head(top_n_drug).index

    sub = drug_df[drug_df['PA_PRIMARY_DIAG'].isin(top_diag) & drug_df['DRUG_CODE'].isin(top_drug)]
    if sub.empty:
        return pd.DataFrame()

    matrix = (
        sub.groupby(['PA_PRIMARY_DIAG', 'DRUG_CODE'])
        .agg(Claims=('IS_REJECTED', 'count'), Rejected=('IS_REJECTED', 'sum'))
        .reset_index()
    )
    matrix['Approved'] = matrix['Claims'] - matrix['Rejected']
    matrix['RejRate_%'] = (matrix['Rejected'] / matrix['Claims'] * 100).round(1)
    return matrix


def compute_mixed_outcome_diagnoses(drug_df, min_claims=20, top_k_drugs=3):
    """
    Diagnoses where the SAME diagnosis led to both approved and rejected
    claims — answers "can the same diagnosis lead to both outcomes?"
    directly, with the specific drugs on each side so it's actionable
    rather than just a yes/no.

    Args:
        drug_df: DataFrame with claims
        min_claims: minimum total claims for a diagnosis to be reported
        top_k_drugs: how many top drugs to list on each side (approved/rejected)

    Returns:
        DataFrame: PA_PRIMARY_DIAG, Total_Claims, Approved_Claims,
        Rejected_Claims, Approved_Drugs, Rejected_Drugs, Top_Rejection_Reason
        — one row per diagnosis with a genuinely mixed outcome, sorted by
        Total_Claims descending.
    """
    required = {'PA_PRIMARY_DIAG', 'DRUG_CODE', 'IS_REJECTED'}
    if not required.issubset(drug_df.columns):
        return pd.DataFrame()

    agg = (
        drug_df.groupby('PA_PRIMARY_DIAG')
        .agg(Total_Claims=('IS_REJECTED', 'count'), Rejected_Claims=('IS_REJECTED', 'sum'))
        .reset_index()
    )
    agg['Approved_Claims'] = agg['Total_Claims'] - agg['Rejected_Claims']
    mixed = agg[
        (agg['Total_Claims'] >= min_claims) & (agg['Rejected_Claims'] > 0) & (agg['Approved_Claims'] > 0)
    ].copy()
    if mixed.empty:
        return pd.DataFrame()

    def _top_drugs(rows):
        return ', '.join(f"{d} ({n})" for d, n in rows['DRUG_CODE'].value_counts().head(top_k_drugs).items())

    rows = []
    for _, r in mixed.iterrows():
        diag = r['PA_PRIMARY_DIAG']
        diag_rows = drug_df[drug_df['PA_PRIMARY_DIAG'] == diag]
        approved_drugs = _top_drugs(diag_rows[diag_rows['IS_REJECTED'] == 0])
        rejected_rows = diag_rows[diag_rows['IS_REJECTED'] == 1]
        rejected_drugs = _top_drugs(rejected_rows)

        top_reason = None
        if 'REJ_CODE_PREFIX' in rejected_rows.columns and len(rejected_rows):
            mode_vals = rejected_rows['REJ_CODE_PREFIX'].mode()
            top_reason = mode_vals.iloc[0] if len(mode_vals) else None

        rows.append({
            'PA_PRIMARY_DIAG': diag,
            'Total_Claims': int(r['Total_Claims']),
            'Approved_Claims': int(r['Approved_Claims']),
            'Rejected_Claims': int(r['Rejected_Claims']),
            'Approved_Drugs': approved_drugs if approved_drugs else '—',
            'Rejected_Drugs': rejected_drugs if rejected_drugs else '—',
            'Top_Rejection_Reason': top_reason if top_reason else '—',
        })

    return pd.DataFrame(rows).sort_values('Total_Claims', ascending=False).reset_index(drop=True)




def enrich_combo_display(combo_df, drug_df):
    """Add drug labels to combo tables when available."""
    if combo_df.empty:
        return combo_df

    display = combo_df.copy()
    parts = display['DRUG_DIAG_COMBO'].str.split(' | ', n=1, expand=True)
    display['Drug Code'] = parts[0]

    if 'DRUG_NAME' in drug_df.columns:
        names = drug_df[['DRUG_CODE', 'DRUG_NAME']].drop_duplicates('DRUG_CODE')
        display = display.merge(names, left_on='Drug Code', right_on='DRUG_CODE', how='left')
        display = display.drop(columns=['DRUG_CODE'], errors='ignore')
        display = display.rename(columns={'DRUG_NAME': 'Drug Name'})

    cols = [c for c in [
        'Drug Code', 'Drug Name', 'DRUG_DIAG_COMBO', 'Total', 'Rejected', 'RejRate',
    ] if c in display.columns]
    return display[cols]


def add_top_rejection_columns(df, entity_col, full_df):
    """
    Add top rejection code and description to entity summary.
    Vectorized implementation - O(n) instead of O(n * distinct_entities).
    """    
    # Fast path: no rejections or no rejection codes
    if 'IS_REJECTED' not in full_df.columns or full_df['IS_REJECTED'].sum() == 0:
        df['Top_Rejection_Code'] = 'N/A'
        df['Top_Rejection_Desc'] = 'N/A'
        return df
    
    rej_df = full_df[full_df['IS_REJECTED'] == 1]
    
    if 'REJ_CODE_PREFIX' not in rej_df.columns or rej_df.empty:
        df['Top_Rejection_Code'] = 'N/A'
        df['Top_Rejection_Desc'] = 'N/A'
        return df
    
    # ── VECTORIZED APPROACH ──────────────────────────────────
    # Count rejections by (entity, rejection_code)
    rejection_counts = (
        rej_df.groupby([entity_col, 'REJ_CODE_PREFIX'])
        .size()
        .reset_index(name='count')
    )
    
    # Find the rejection code with maximum count for each entity
    # Using idxmax on groupby is much faster than looping
    top_rejection_idx = (
        rejection_counts
        .groupby(entity_col)['count']
        .idxmax()
    )
    
    top_rejection = rejection_counts.loc[top_rejection_idx, [entity_col, 'REJ_CODE_PREFIX']]
    
    # Merge back to the main DataFrame
    df = df.merge(
        top_rejection,
        on=entity_col,
        how='left'
    )
    
    # Fill missing values and add descriptions
    df['REJ_CODE_PREFIX'] = df['REJ_CODE_PREFIX'].fillna('N/A')
    df['Top_Rejection_Code'] = df['REJ_CODE_PREFIX']
    df['Top_Rejection_Desc'] = df['REJ_CODE_PREFIX'].map(CODE_DESC).fillna('Unknown')
    df = df.drop(columns=['REJ_CODE_PREFIX'])
    
    return df


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def build_baseline_sets(historical_snapshots):
    """
    Build baseline entity sets efficiently.
    Returns dict with all baseline sets.
    """
    baseline = {
        'drugs': set(),
        'diagnoses': set(),
        'providers': set(),
        'combos': set()
    }
    
    mapping = {
        'drug_stats': ('drugs', 'DRUG_CODE'),
        'diag_stats': ('diagnoses', 'PA_PRIMARY_DIAG'),
        'provider_stats': ('providers', 'DOC_LIC_NO'),
        'combo_stats': ('combos', 'DRUG_DIAG_COMBO')
    }
    
    for snapshot in historical_snapshots.values():
        for key, (target, col) in mapping.items():
            if key in snapshot and snapshot[key] is not None and not snapshot[key].empty:
                baseline[target].update(
                    str(x) for x in snapshot[key][col].dropna().values
                )
    
    return baseline


def get_current_month_data(drug_df):
    """Filter to current month data."""
    period_end = pd.to_datetime(drug_df['SERVICE_DT']).max()
    period_start = period_end.replace(day=1)
    return drug_df[
        (drug_df['SERVICE_DT'] >= period_start) & 
        (drug_df['SERVICE_DT'] <= period_end)
    ].copy()


def _build_summary_bytes():
    out = io.StringIO()
    out.write("Rejection Code Breakdown\n")
    try:
        compute_code_stats(rejected).to_csv(out, index=False)
    except Exception:
        out.write("No rejection code data\n")
    out.write("\nHigh Risk Combos\n")
    try:
        hr, sf, gf = compute_drug_diag_combos(drug_df)
        (hr if not hr.empty else pd.DataFrame()).to_csv(out, index=False)
    except Exception:
        out.write("No combos data\n")
    out.write("\nProvider Stats\n")
    try:
        compute_provider_stats(drug_df).to_csv(out, index=False)
    except Exception:
        out.write("No provider stats\n")
    return out.getvalue().encode('utf-8')



def add_top_rejection_columns_fast(df, entity_col, full_df):
    """
    Add top rejection code and description to entity summary.
    Vectorized implementation using sort + drop_duplicates.
    
    Parameters:
    -----------
    df : DataFrame
        Aggregated entity statistics (must contain entity_col)
    entity_col : str
        Column name for the entity
    full_df : DataFrame
        Original full dataset with rejection details
    
    Returns:
    --------
    DataFrame with added Top_Rejection_Code and Top_Rejection_Desc columns
    """
    # Fast path: no rejections or no rejection codes
    if 'IS_REJECTED' not in full_df.columns or full_df['IS_REJECTED'].sum() == 0:
        df['Top_Rejection_Code'] = 'N/A'
        df['Top_Rejection_Desc'] = 'N/A'
        return df
    
    rej_df = full_df[full_df['IS_REJECTED'] == 1]
    
    if 'REJ_CODE_PREFIX' not in rej_df.columns or rej_df.empty:
        df['Top_Rejection_Code'] = 'N/A'
        df['Top_Rejection_Desc'] = 'N/A'
        return df
    
    # Count rejections by (entity, rejection_code)
    # Sort by count descending so drop_duplicates keeps the highest count
    top_rejection = (
        rej_df
        .groupby([entity_col, 'REJ_CODE_PREFIX'])
        .size()
        .reset_index(name='count')
        .sort_values([entity_col, 'count'], ascending=[True, False])
        .drop_duplicates(entity_col)  # Keeps first (highest count due to sort)
        [[entity_col, 'REJ_CODE_PREFIX']]
    )
    
    # Merge back to the main DataFrame
    df = df.merge(top_rejection, on=entity_col, how='left')
    
    # Fill missing values and add descriptions
    df['Top_Rejection_Code'] = df['REJ_CODE_PREFIX'].fillna('N/A')
    df['Top_Rejection_Desc'] = df['REJ_CODE_PREFIX'].map(CODE_DESC).fillna('Unknown')
    df = df.drop(columns=['REJ_CODE_PREFIX'])
    
    return df


def add_top_rejection_columns_idxmax(df, entity_col, full_df):
    """
    Alternative implementation using idxmax.
    This can be faster for smaller datasets.
    """
    if 'IS_REJECTED' not in full_df.columns or full_df['IS_REJECTED'].sum() == 0:
        df['Top_Rejection_Code'] = 'N/A'
        df['Top_Rejection_Desc'] = 'N/A'
        return df
    
    rej_df = full_df[full_df['IS_REJECTED'] == 1]
    
    if 'REJ_CODE_PREFIX' not in rej_df.columns or rej_df.empty:
        df['Top_Rejection_Code'] = 'N/A'
        df['Top_Rejection_Desc'] = 'N/A'
        return df
    
    # Count rejections by (entity, rejection_code)
    rejection_counts = (
        rej_df.groupby([entity_col, 'REJ_CODE_PREFIX'])
        .size()
        .reset_index(name='count')
    )
    
    # Find the rejection code with maximum count for each entity
    top_rejection_idx = (
        rejection_counts
        .groupby(entity_col)['count']
        .idxmax()
    )
    
    top_rejection = rejection_counts.loc[top_rejection_idx, [entity_col, 'REJ_CODE_PREFIX']]
    
    # Merge back
    df = df.merge(top_rejection, on=entity_col, how='left')
    df['Top_Rejection_Code'] = df['REJ_CODE_PREFIX'].fillna('N/A')
    df['Top_Rejection_Desc'] = df['REJ_CODE_PREFIX'].map(CODE_DESC).fillna('Unknown')
    df = df.drop(columns=['REJ_CODE_PREFIX'])
    
    return df


def compute_new_entity_summary_optimized(
    df,
    entity_col,
    baseline_entities,
    name_col=None,
    min_claims=1
):
    """
    Compute summary statistics for new entities appearing in the current period.
    Fully vectorized - no Python loops.
    
    Parameters:
    -----------
    df : DataFrame - Current period claims data
    entity_col : str - Column name for the entity
    baseline_entities : set - Entities seen in baseline months
    name_col : str, optional - Column name for entity names
    min_claims : int - Minimum claims required to include entity
    
    Returns:
    --------
    DataFrame with entity statistics and rejection patterns
    """
    # Input validation
    if entity_col not in df.columns or 'IS_REJECTED' not in df.columns:
        return pd.DataFrame()
    
    # Convert to string for set operations
    # Use a temp column to avoid modifying original
    entity_str_col = entity_col + '_str'
    df[entity_str_col] = df[entity_col].astype(str)
    
    # Identify new entities using set operations
    current_entities = set(df[entity_str_col].dropna().values)
    new_entities = current_entities - baseline_entities
    
    if not new_entities:
        # Clean up temp column
        df.drop(columns=[entity_str_col], inplace=True)
        return pd.DataFrame()
    
    # Filter to new entities - use boolean indexing with isin
    mask = df[entity_str_col].isin(new_entities)
    new_df = df[mask].copy()
    
    # Drop temp column from filtered df
    new_df.drop(columns=[entity_str_col], inplace=True)
    df.drop(columns=[entity_str_col], inplace=True)  # Clean up original too
    
    # ── AGGREGATE ──────────────────────────────────────────────
    agg_df = (
        new_df.groupby(entity_col)
        .agg(
            Claims=('IS_REJECTED', 'count'),
            Approved=('IS_REJECTED', lambda x: (x == 0).sum()),
            Rejected=('IS_REJECTED', 'sum'),
        )
        .reset_index()
    )
    
    # Filter by minimum claims
    agg_df = agg_df[agg_df['Claims'] >= min_claims]
    
    if agg_df.empty:
        return pd.DataFrame()
    
    # Calculate rejection rate (avoid division by zero)
    agg_df['Rejection_Rate'] = (
        (agg_df['Rejected'] / agg_df['Claims'] * 100)
        .round(1)
        .astype(str) + '%'
    )
    
    # ── ADD TOP REJECTION CODES (VECTORIZED) ──────────────────
    agg_df = add_top_rejection_columns_fast(agg_df, entity_col, new_df)
    
    # ── ADD NAME COLUMN ──────────────────────────────────────
    if name_col and name_col in df.columns:
        # Use drop_duplicates with subset for efficiency
        names = df[[entity_col, name_col]].drop_duplicates(subset=[entity_col])
        agg_df = agg_df.merge(names, on=entity_col, how='left')
        cols = [entity_col, name_col, 'Claims', 'Approved', 'Rejected', 
                'Rejection_Rate', 'Top_Rejection_Code', 'Top_Rejection_Desc']
    else:
        cols = [entity_col, 'Claims', 'Approved', 'Rejected', 
                'Rejection_Rate', 'Top_Rejection_Code', 'Top_Rejection_Desc']
    
    # Clean up and sort
    result = agg_df[[c for c in cols if c in agg_df.columns]]
    return result.sort_values('Claims', ascending=False)


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def compute_new_entities_appearing(drug_df, historical_snapshots):
    """
    Detect entities appearing in the current month that were NEVER seen in baseline.
    
    Returns dict with keys: 'drugs', 'diagnoses', 'providers', 'combos'
    Each contains a DataFrame with entity statistics and rejection patterns.
    
    Business meaning: Brand-new entities entering the population for the first time.
    Immediately see whether they're covered or getting specific rejection patterns.
    
    With no baseline months, returns empty dict (too early to identify "new" vs "old").
    """
    results = {}
    
    if not historical_snapshots:
        return results
    
    # Build baseline sets efficiently
    baseline = build_baseline_sets(historical_snapshots)
    
    # Filter to current month
    current_df = get_current_month_data(drug_df)
    
    # Compute summaries - all vectorized operations
    results['drugs'] = compute_new_entity_summary_optimized(
        current_df, 'DRUG_CODE', baseline['drugs'], 'DRUG_NAME'
    )
    
    results['diagnoses'] = compute_new_entity_summary_optimized(
        current_df, 'PA_PRIMARY_DIAG', baseline['diagnoses']
    )
    
    results['providers'] = compute_new_entity_summary_optimized(
        current_df, 'DOC_LIC_NO', baseline['providers'], 'DOC_NAME'
    )
    
    results['combos'] = compute_new_entity_summary_optimized(
        current_df, 'DRUG_DIAG_COMBO', baseline['combos'], min_claims=4
    )
    
    # Additional combo filtering (business rules)
    if not results['combos'].empty:
        results['combos'] = results['combos'][
            (results['combos']['Rejected'] > 0) | 
            (results['combos']['Claims'] >= 10)
        ]
    
    return results