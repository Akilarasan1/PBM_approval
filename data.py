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
    'SERVICE_DT', 'PA_MEM_AGE', 'MEM_GENDER', 'DOC_LIC_NO',
    'PA_FLAG_REASON', 'REJ_REMARKS', 'TREAT_EST_AMT', 'TREAT_APPR_AMT'
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
        df = pd.read_csv(io.BytesIO(file_bytes))
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
        drug_df['SERVICE_DT'] = pd.to_datetime(drug_df['SERVICE_DT'], dayfirst=True, errors='coerce')
        drug_df['WEEK'] = drug_df['SERVICE_DT'].dt.isocalendar().week.astype(str)

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
    if 'DRUG_DIAG_COMBO' not in drug_df.columns or 'IS_REJECTED' not in drug_df.columns:
        empty = pd.DataFrame(columns=['DRUG_DIAG_COMBO', 'Total', 'Rejected', 'RejRate'])
        return empty, empty, empty
    
    combo = (drug_df.groupby('DRUG_DIAG_COMBO')
             .agg(Total=('IS_REJECTED','count'), Rejected=('IS_REJECTED','sum'))
             .reset_index())
    combo['RejRate'] = (combo['Rejected'] / combo['Total'] * 100).round(1)
    
    high_risk = combo[(combo['Total'] >= 20) & (combo['RejRate'] >= 70)].sort_values('RejRate', ascending=False)
    safe_combos = combo[(combo['Total'] >= 20) & (combo['RejRate'] == 0)].sort_values('Total', ascending=False)
    gray_combos = combo[(combo['Total'] >= 20) & (combo['RejRate'] >= 30) & (combo['RejRate'] < 70)].sort_values('RejRate', ascending=False)
    
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
