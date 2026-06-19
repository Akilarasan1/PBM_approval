"""Data processing functions for PBM Dashboard."""

import pandas as pd
import io
import streamlit as st


# ── REQUIRED & OPTIONAL COLUMNS ────────────────────────────────
REQUIRED_COLUMNS = {
    'INS_TREAT_DESC', 'PBM_APPR_STS', 'PBM_REJ_CODE',
    'DRUG_CODE', 'PA_PRIMARY_DIAG', 'TREAT_REJ_AMT'
}

OPTIONAL_COLUMNS = {
    'SERVICE_DT', 'PA_MEM_AGE', 'MEM_GENDER', 'DOC_LIC_NO', 'DOC_NAME',
    'PA_FLAG_REASON', 'REJ_REMARKS', 'PBM_REJ_DESC', 'PAT_REJ_REMARKS',
    'PROV_TREAT_DESC', 'PRIMARY_DIAG', 'TREAT_EST_AMT', 'TREAT_APPR_AMT',
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


@st.cache_data(show_spinner=False)
def process(file_bytes, filename):
    """
    Process uploaded PBM claims file and add derived columns.
    
    Args:
        file_bytes: Binary content of uploaded file
        filename: Name of the file (determines if CSV or Excel)
    
    Returns:
        Processed DataFrame with derived columns for analysis
    """
    # Load file
    if filename.endswith('.csv'):
        # Read with low_memory=False to avoid mixed-type dtype warnings
        df = pd.read_csv(io.BytesIO(file_bytes), low_memory=False)
    else:
        df = pd.read_excel(io.BytesIO(file_bytes))

    # Validate required columns
    missing_required, missing_optional = validate_columns(df)
    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")
    
    # Filter to prescribed drugs only
    if 'INS_TREAT_DESC' in df.columns:
        drug_df = df[df['INS_TREAT_DESC'] == 'Prescribed Drugs'].copy()
    else:
        drug_df = df.copy()

    # Some PBM exports mix string license numbers ("D959") with stray raw
    # integers in the same column. Left as-is, this breaks any sorted()/
    # comparison call downstream (e.g. the provider drill-down selector)
    # with "'<' not supported between instances of 'int' and 'str'".
    # Normalize once, here, so every consumer sees a clean string.
    if 'DOC_LIC_NO' in drug_df.columns:
        drug_df['DOC_LIC_NO'] = drug_df['DOC_LIC_NO'].astype(str)
    
    # Add rejection indicator
    if 'PBM_APPR_STS' in drug_df.columns:
        drug_df['IS_REJECTED'] = (drug_df['PBM_APPR_STS'] == 'PBM_REJECT').astype(int)
    else:
        drug_df['IS_REJECTED'] = 0
    
    # Extract rejection code prefix
    if 'PBM_REJ_CODE' in drug_df.columns:
        drug_df['REJ_CODE_PREFIX'] = (drug_df['PBM_REJ_CODE'].fillna('NO_CODE')
                                       .astype(str).str.split('-').str[0])
    else:
        drug_df['REJ_CODE_PREFIX'] = 'NO_CODE'
    
    # Create drug-diagnosis combo column
    drug_code_col = drug_df['DRUG_CODE'].fillna('UNK').astype(str) if 'DRUG_CODE' in drug_df.columns else 'UNK'
    diag_col = drug_df['PA_PRIMARY_DIAG'].fillna('UNK').astype(str) if 'PA_PRIMARY_DIAG' in drug_df.columns else 'UNK'
    drug_df['DRUG_DIAG_COMBO'] = drug_code_col + ' | ' + diag_col
    
    # Convert rejected amount to absolute value
    if 'TREAT_REJ_AMT' in drug_df.columns:
        drug_df['TREAT_REJ_AMT'] = drug_df['TREAT_REJ_AMT'].abs()
    else:
        drug_df['TREAT_REJ_AMT'] = 0

    # Parse service date and extract week
    if 'SERVICE_DT' in drug_df.columns:
        drug_df['SERVICE_DT'] = pd.to_datetime(drug_df['SERVICE_DT'], errors='coerce')
        drug_df['WEEK'] = drug_df['SERVICE_DT'].dt.isocalendar().week.astype(str)

    # Normalize rejection remarks for MNEC/refill analysis
    if 'REJ_REMARKS' not in drug_df.columns or drug_df['REJ_REMARKS'].isna().all():
        if 'PBM_REJ_DESC' in drug_df.columns:
            drug_df['REJ_REMARKS'] = drug_df['PBM_REJ_DESC']
        elif 'PAT_REJ_REMARKS' in drug_df.columns:
            drug_df['REJ_REMARKS'] = drug_df['PAT_REJ_REMARKS']

    # Human-readable drug label when available
    if 'PROV_TREAT_DESC' in drug_df.columns:
        drug_df['DRUG_NAME'] = drug_df['PROV_TREAT_DESC'].fillna('').astype(str).str.strip()
        drug_df.loc[drug_df['DRUG_NAME'] == '', 'DRUG_NAME'] = drug_df['DRUG_CODE'].astype(str)

    # Add age group categories
    if 'PA_MEM_AGE' in drug_df.columns:
        drug_df['AGE_GROUP'] = drug_df['PA_MEM_AGE'].apply(_categorize_age)

    return drug_df


def _categorize_age(a):
    """Categorize age into age groups."""
    if pd.isna(a):
        return 'UNKNOWN'
    a = int(a)
    if a <= 2: return 'INFANT (0-2)'
    if a <= 12: return 'CHILD (3-12)'
    if a <= 17: return 'TEEN (13-17)'
    if a <= 35: return 'YOUNG ADULT (18-35)'
    if a <= 55: return 'ADULT (36-55)'
    if a <= 70: return 'SENIOR (56-70)'
    return 'ELDERLY (71+)'


def compute_code_stats(rejected):
    """Compute rejection code statistics."""
    from utils import CODE_DESC
    
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
    from utils import THRESHOLDS

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


def compute_new_drugs_always_ncov(drug_df, min_claims=5):
    """
    Find drug codes first seen in the reporting month with 100% NCOV rejection.
    """
    if 'SERVICE_DT' not in drug_df.columns or 'DRUG_CODE' not in drug_df.columns:
        return pd.DataFrame()

    dated = drug_df[drug_df['SERVICE_DT'].notna()].copy()
    if dated.empty:
        return pd.DataFrame()

    period_end = dated['SERVICE_DT'].max()
    period_start = period_end.replace(day=1)

    first_seen = (
        dated.groupby('DRUG_CODE')['SERVICE_DT']
        .min()
        .reset_index()
        .rename(columns={'SERVICE_DT': 'FIRST_SEEN'})
    )
    new_codes = first_seen.loc[
        (first_seen['FIRST_SEEN'] >= period_start) & (first_seen['FIRST_SEEN'] <= period_end),
        'DRUG_CODE',
    ]
    new_drugs = dated[dated['DRUG_CODE'].isin(new_codes)]
    if new_drugs.empty:
        return pd.DataFrame()

    ncov_stats = (
        new_drugs.groupby('DRUG_CODE')
        .agg(
            Total=('IS_REJECTED', 'count'),
            Rejected=('IS_REJECTED', 'sum'),
            NCOV=('REJ_CODE_PREFIX', lambda x: (x == 'NCOV').sum()),
            Amt=('TREAT_REJ_AMT', 'sum'),
            First=('SERVICE_DT', 'min'),
        )
        .reset_index()
    )
    ncov_stats['RejRate'] = (ncov_stats['Rejected'] / ncov_stats['Total'] * 100).round(1)

    always_ncov = ncov_stats[
        (ncov_stats['NCOV'] == ncov_stats['Rejected'])
        & (ncov_stats['Rejected'] == ncov_stats['Total'])
        & (ncov_stats['Total'] >= min_claims)
    ].sort_values('Total', ascending=False)

    if 'DRUG_NAME' in drug_df.columns:
        names = drug_df[['DRUG_CODE', 'DRUG_NAME']].drop_duplicates('DRUG_CODE')
        always_ncov = always_ncov.merge(names, on='DRUG_CODE', how='left')

    return always_ncov


def compute_provider_investigation(drug_df, min_claims=20):
    """Build prioritized provider investigation queue."""
    required = {'DOC_LIC_NO', 'REJ_CODE_PREFIX', 'IS_REJECTED'}
    if not required.issubset(drug_df.columns):
        return pd.DataFrame()

    investigation = (
        drug_df.groupby('DOC_LIC_NO')
        .agg(
            Total_Claims=('IS_REJECTED', 'count'),
            Rejections=('IS_REJECTED', 'sum'),
            RejAmt=('TREAT_REJ_AMT', 'sum'),
        )
        .reset_index()
    )

    mnec_counts = (
        drug_df[drug_df['REJ_CODE_PREFIX'] == 'MNEC']
        .groupby('DOC_LIC_NO')
        .size()
        .reset_index(name='MNEC_Count')
    )
    ncov_counts = (
        drug_df[drug_df['REJ_CODE_PREFIX'] == 'NCOV']
        .groupby('DOC_LIC_NO')
        .size()
        .reset_index(name='NCOV_Count')
    )

    investigation = investigation.merge(mnec_counts, on='DOC_LIC_NO', how='left')
    investigation = investigation.merge(ncov_counts, on='DOC_LIC_NO', how='left')
    investigation = investigation.fillna(0)
    investigation['RejRate_%'] = (
        investigation['Rejections'] / investigation['Total_Claims'] * 100
    ).round(1)

    investigation = investigation[investigation['Total_Claims'] >= min_claims]

    def _reason(row):
        if row['RejRate_%'] > 35:
            return 'High Rejection Rate'
        if row['MNEC_Count'] > 80:
            return 'High MNEC Activity'
        if row['NCOV_Count'] > 20:
            return 'Frequent NCOV'
        if row['RejAmt'] > 100000:
            return 'High Financial Impact'
        return 'Monitor'

    investigation['InvestigationReason'] = investigation.apply(_reason, axis=1)

    if 'DOC_NAME' in drug_df.columns:
        names = drug_df[['DOC_LIC_NO', 'DOC_NAME']].dropna().drop_duplicates('DOC_LIC_NO')
        investigation = investigation.merge(names, on='DOC_LIC_NO', how='left')

    return investigation.sort_values(['MNEC_Count', 'RejAmt'], ascending=False)


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
