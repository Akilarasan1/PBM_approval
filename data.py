"""Data processing functions for PBM Dashboard."""

import pandas as pd
import io
import streamlit as st
import time
import numpy as np
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


@st.cache_data(show_spinner="Processing uploaded file — large Excel files can take a minute or more...")
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
    t = time.perf_counter()
    if filename.endswith('.csv'):
        # Read with low_memory=False to avoid mixed-type dtype warnings
        df = pd.read_csv(io.BytesIO(file_bytes), low_memory=False)
    else:
        df = pd.read_excel(io.BytesIO(file_bytes))

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
        # drug_df['SERVICE_DT'] = pd.to_datetime(drug_df['SERVICE_DT'], errors='coerce')
        # drug_df['SERVICE_DT'] = pd.to_datetime(drug_df['SERVICE_DT'],format='%d-%m-%Y %H:%M',errors='coerce')
        parsed = pd.to_datetime(drug_df['SERVICE_DT'], format='%d-%m-%Y %H:%M', errors='coerce')
        still_bad = parsed.isna() & drug_df['SERVICE_DT'].notna()
        if still_bad.any():
            parsed.loc[still_bad] = pd.to_datetime(drug_df['SERVICE_DT'][still_bad], errors='coerce')
        drug_df['SERVICE_DT'] = parsed

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


def compute_new_drugs_always_ncov(drug_df, full_df=None, min_claims=5):
    """
    Find ALL drug codes with 100% NCOV rejection in the current month.
    
    Logic:
    1. Filter to current month claims only
    2. For each drug, count total claims and NCOV rejections
    3. Flag if: 100% of claims are rejected AND 100% of rejections are NCOV

    If `full_df` is supplied (the full uploaded date range, across every
    month on record), each flagged drug is additionally checked against
    its own drug-diagnosis combination(s) to see whether that exact combo
    was EVER approved anywhere in the upload. This distinguishes a
    genuinely brand-new/never-covered drug from one that WAS covered
    before and something changed (formulary drop, miscoding, etc.).
    
    Args:
        drug_df: DataFrame with claims (the scoped/current-month view)
        full_df: DataFrame spanning the full uploaded date range, used only
            for the "ever approved" history lookup. If None, the
            Ever_Approved columns are omitted.
        min_claims: Minimum claims threshold
        
    Returns:
        DataFrame with suspicious drugs (100% NCOV rejection this month)
    """

    required_cols = ['SERVICE_DT', 'DRUG_CODE', 'IS_REJECTED', 'REJ_CODE_PREFIX']
    missing = [c for c in required_cols if c not in drug_df.columns]

    if missing:
        return pd.DataFrame()

    # Filter to dated records only
    dated = drug_df[drug_df['SERVICE_DT'].notna()].copy()
    if dated.empty:
        return pd.DataFrame()

    # Ensure datetime
    dated['SERVICE_DT'] = pd.to_datetime(dated['SERVICE_DT'], errors='coerce')
    dated = dated[dated['SERVICE_DT'].notna()]
    if dated.empty:
        return pd.DataFrame()

    # ── KEY CHANGE: Filter to CURRENT MONTH (not just new drugs) ──
    # period_end = dated['SERVICE_DT'].max()
    # period_start = period_end.replace(day=1)

    period_end = dated['SERVICE_DT'].max()
    period_start = period_end.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

    
    # Get ALL drugs in current month (new OR existing)
    current_month_drugs = dated[
        (dated['SERVICE_DT'] >= period_start) & (dated['SERVICE_DT'] <= period_end)
    ].copy()
    
    if current_month_drugs.empty:
        return pd.DataFrame()

    # ── NCOV COUNTING: From rejected claims only ──────────────────
    rejected_claims = current_month_drugs[current_month_drugs['IS_REJECTED'] == 1].copy()
    
    if rejected_claims.empty:
        return pd.DataFrame()
    
    # Clean REJ_CODE_PREFIX for consistent matching
    rejected_claims['REJ_CODE_PREFIX'] = (
        rejected_claims['REJ_CODE_PREFIX']
        .astype(str)
        .str.upper()
        .str.strip()
    )
    
    # Aggregate statistics for ALL drugs in current month
    ncov_stats = (
        current_month_drugs.groupby('DRUG_CODE')
        .agg(
            Total=('IS_REJECTED', 'count'),
            Rejected=('IS_REJECTED', 'sum'),
            Rejected_Amount=('TREAT_REJ_AMT', 'sum') if 'TREAT_REJ_AMT' in current_month_drugs.columns else ('IS_REJECTED', 'sum')
        )
        .reset_index()
    )
    
    # Count NCOV in rejected claims only
    ncov_count = (
        rejected_claims.groupby('DRUG_CODE')['REJ_CODE_PREFIX']
        .apply(lambda x: x.str.contains('NCOV', na=False).sum())
        .reset_index()
        .rename(columns={'REJ_CODE_PREFIX': 'NCOV_Count'})
    )
    
    # Merge NCOV counts
    ncov_stats = ncov_stats.merge(ncov_count, on='DRUG_CODE', how='left')
    ncov_stats['NCOV_Count'] = ncov_stats['NCOV_Count'].fillna(0).astype(int)
    
    # Calculate rejection rate
    ncov_stats['RejRate_%'] = (ncov_stats['Rejected'] / ncov_stats['Total'] * 100).round(1)
    
    # ── FLAG CRITERIA ──────────────────────────────────────────────
    always_ncov = ncov_stats[
        (ncov_stats['Total'] >= min_claims)
        & (ncov_stats['Rejected'] == ncov_stats['Total'])
        & (ncov_stats['NCOV_Count'] == ncov_stats['Rejected'])
        & (ncov_stats['NCOV_Count'] > 0)
    ].copy()
    
    if always_ncov.empty:
        return pd.DataFrame()

    if 'DRUG_NAME' in drug_df.columns:
        names = drug_df[['DRUG_CODE', 'DRUG_NAME']].drop_duplicates('DRUG_CODE')
        always_ncov = always_ncov.merge(names, on='DRUG_CODE', how='left')

    # Sort by NCOV count (highest first)
    always_ncov = always_ncov.sort_values('NCOV_Count', ascending=False)

    # Rename for display
    always_ncov = always_ncov.rename(columns={
        'Total': 'Total_Claims',
        'Rejected': 'Rejected_Claims',
        'NCOV_Count': 'NCOV_Rejections',
        'Amt': 'Rejected_Amount'
    })
    # ── NCOV HISTORY + APPROVAL HISTORY (drug-diagnosis combo) ─────────────

    always_ncov['First_NCOV_Date'] = pd.NaT
    always_ncov['Last_NCOV_Date'] = pd.NaT
    always_ncov['NCOV_Duration_Days'] = 0

    always_ncov['Ever_Approved'] = False
    always_ncov['Last_Approval_Date'] = pd.NaT
    always_ncov['Approved_Claims_Count'] = 0
    always_ncov['Approved_After_First_NCOV'] = False

    has_full_history = (
        full_df is not None
        and 'DRUG_DIAG_COMBO' in full_df.columns
        and 'SERVICE_DT' in full_df.columns
        and 'IS_REJECTED' in full_df.columns
        and 'REJ_CODE_PREFIX' in full_df.columns
        and 'DRUG_DIAG_COMBO' in current_month_drugs.columns
    )

    if has_full_history:

        full_df = full_df.copy()
        full_df['SERVICE_DT'] = pd.to_datetime(
            full_df['SERVICE_DT'],
            errors='coerce'
        )

        flagged_drugs = set(always_ncov['DRUG_CODE'])

        flagged_combos = (
            current_month_drugs[
                current_month_drugs['DRUG_CODE'].isin(flagged_drugs)
            ][['DRUG_CODE', 'DRUG_DIAG_COMBO']]
            .drop_duplicates()
        )

        # ---------------------------------------------------------
        # NCOV HISTORY
        # ---------------------------------------------------------

        ncov_history = full_df[
            full_df['REJ_CODE_PREFIX']
            .astype(str)
            .str.upper()
            .str.contains('NCOV', na=False)
        ][['DRUG_DIAG_COMBO', 'SERVICE_DT']].copy()

        ncov_summary = (
            ncov_history
            .groupby('DRUG_DIAG_COMBO')['SERVICE_DT']
            .agg(
                First_NCOV_Date='min',
                Last_NCOV_Date='max'
            )
            .reset_index()
        )

        # ---------------------------------------------------------
        # APPROVAL HISTORY
        # ---------------------------------------------------------

        approved_history = full_df[
            full_df['IS_REJECTED'] == 0
        ][['DRUG_DIAG_COMBO', 'SERVICE_DT']].copy()

        approval_summary = (
            approved_history
            .groupby('DRUG_DIAG_COMBO')['SERVICE_DT']
            .agg(
                Last_Approval_Date='max',
                Approved_Claims_Count='count'
            )
            .reset_index()
        )

        combo_history = (
            flagged_combos
            .merge(ncov_summary, on='DRUG_DIAG_COMBO', how='left')
            .merge(approval_summary, on='DRUG_DIAG_COMBO', how='left')
        )

        # Was there an approval AFTER NCOV started?
        combo_history['Approved_After_First_NCOV'] = (
            combo_history['Last_Approval_Date']
            > combo_history['First_NCOV_Date']
        )

        combo_history['NCOV_Duration_Days'] = (
            combo_history['Last_NCOV_Date']
            - combo_history['First_NCOV_Date']
        ).dt.days

        drug_level = (
            combo_history.groupby('DRUG_CODE')
            .agg(
                First_NCOV_Date=('First_NCOV_Date', 'min'),
                Last_NCOV_Date=('Last_NCOV_Date', 'max'),
                NCOV_Duration_Days=('NCOV_Duration_Days', 'max'),

                Ever_Approved=(
                    'Approved_Claims_Count',
                    lambda s: s.notna().any()
                ),

                Last_Approval_Date=('Last_Approval_Date', 'max'),

                Approved_Claims_Count=(
                    'Approved_Claims_Count',
                    'sum'
                ),

                Approved_After_First_NCOV=(
                    'Approved_After_First_NCOV',
                    'any'
                )
            )
            .reset_index()
        )

        drug_level['Approved_Claims_Count'] = (
            drug_level['Approved_Claims_Count']
            .fillna(0)
            .astype(int)
        )

        always_ncov = always_ncov.drop(
            columns=[
                'First_NCOV_Date',
                'Last_NCOV_Date',
                'NCOV_Duration_Days',
                'Ever_Approved',
                'Last_Approval_Date',
                'Approved_Claims_Count',
                'Approved_After_First_NCOV'
            ],
            errors='ignore'
        ).merge(
            drug_level,
            on='DRUG_CODE',
            how='left'
        )

        always_ncov['Ever_Approved'] = (
            always_ncov['Ever_Approved']
            .fillna(False)
        )

        always_ncov['Approved_After_First_NCOV'] = (
            always_ncov['Approved_After_First_NCOV']
            .fillna(False)
        )

        always_ncov['Approved_Claims_Count'] = (
            always_ncov['Approved_Claims_Count']
            .fillna(0)
            .astype(int)
        )

        always_ncov['First_NCOV_Date'] = (
            always_ncov['First_NCOV_Date']
            .dt.strftime('%Y-%m-%d')
        )

        always_ncov['Last_NCOV_Date'] = (
            always_ncov['Last_NCOV_Date']
            .dt.strftime('%Y-%m-%d')
        )

        always_ncov['Last_Approval_Date'] = (
            always_ncov['Last_Approval_Date']
            .dt.strftime('%Y-%m-%d')
        )

        always_ncov = always_ncov.sort_values(
            ['Ever_Approved', 'NCOV_Rejections'],
            ascending=[False, False]
        )

        always_ncov['Coverage_Status'] = np.select(
            [~always_ncov['Ever_Approved'],
                pd.to_datetime(always_ncov['Last_Approval_Date'], errors='coerce')
                < pd.to_datetime(always_ncov['First_NCOV_Date'], errors='coerce')],
            ['Never Covered','Coverage Stopped'],default='Mixed History')
        

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


def compute_new_entities_appearing(drug_df, historical_snapshots):
    """
    Detect entities (drugs, diagnoses, providers, drug-diagnosis combos) 
    appearing in the current month that were NEVER seen in any baseline month.
    
    Returns a dict with keys 'drugs', 'diagnoses', 'providers', 'combos', each 
    containing a DataFrame with:
    - Entity (code/name)
    - Claims (total this month)
    - Approved (count)
    - Rejected (count)
    - Rejection_Rate % (formatted)
    - Top_Rejection_Code
    - Top_Rejection_Desc
    
    Business meaning: Brand-new entities entering the population for the first time.
    Immediately see whether they're covered or getting specific rejection patterns.
    
    With no baseline months, returns empty dict (too early to identify "new" vs "old").
    """
    
    from utils import CODE_DESC
    
    results = {}
    
    if not historical_snapshots or len(historical_snapshots) == 0:
        return results
    
    # Collect all entities ever seen in baseline months
    baseline_drugs = set()
    baseline_diags = set()
    baseline_provs = set()
    baseline_combos = set()
    
    for month, snapshot in historical_snapshots.items():
        if 'drug_stats' in snapshot and snapshot['drug_stats'] is not None and not snapshot['drug_stats'].empty:
            baseline_drugs.update(
                snapshot['drug_stats']['DRUG_CODE'].dropna().astype(str).unique()
            )
        if 'diag_stats' in snapshot and snapshot['diag_stats'] is not None and not snapshot['diag_stats'].empty:
            baseline_diags.update(
                snapshot['diag_stats']['PA_PRIMARY_DIAG'].dropna().astype(str).unique()
            )
        if 'provider_stats' in snapshot and snapshot['provider_stats'] is not None and not snapshot['provider_stats'].empty:
            baseline_provs.update(
                snapshot['provider_stats']['DOC_LIC_NO'].dropna().astype(str).unique()
            )
        if 'combo_stats' in snapshot and snapshot['combo_stats'] is not None and not snapshot['combo_stats'].empty:
            baseline_combos.update(
                snapshot['combo_stats']['DRUG_DIAG_COMBO'].dropna().astype(str).unique()
            )
    
    def _top_rejection_for_entity(subset):
        """Helper: find most common rejection code in a subset, with description."""
        if len(subset) == 0 or subset['IS_REJECTED'].sum() == 0:
            return 'N/A', 'N/A'
        rej = subset[subset['IS_REJECTED'] == 1]
        if 'REJ_CODE_PREFIX' in rej.columns and len(rej) > 0:
            top_code = rej['REJ_CODE_PREFIX'].value_counts().index[0]
            desc = CODE_DESC.get(top_code, 'Unknown')
            return top_code, desc
        return 'N/A', 'N/A'
    
    # ── NEW DRUGS ──────────────────────────────────────────────────
    if 'DRUG_CODE' in drug_df.columns and 'IS_REJECTED' in drug_df.columns:
        period_end = pd.to_datetime(drug_df['SERVICE_DT']).max()
        period_start = period_end.replace(day=1)

        drug_df = drug_df[(drug_df['SERVICE_DT'] >= period_start) & (drug_df['SERVICE_DT'] <= period_end)].copy()

        current_drugs = set(drug_df['DRUG_CODE'].dropna().astype(str).unique())

        new_drugs = current_drugs - baseline_drugs
        
        if new_drugs:
            new_drug_df = drug_df[drug_df['DRUG_CODE'].astype(str).isin(new_drugs)].copy()
            
            drug_agg = (
                new_drug_df.groupby('DRUG_CODE')
                .agg(
                    Claims=('IS_REJECTED', 'count'),
                    Approved=('IS_REJECTED', lambda x: (x == 0).sum()),
                    Rejected=('IS_REJECTED', 'sum'),
                )
                .reset_index()
            )
            
            drug_agg['Rejection_Rate'] = (
                (drug_agg['Rejected'] / drug_agg['Claims'] * 100).round(1).astype(str) + '%'
            )
            
            top_codes = []
            top_descs = []
            for drug_code in drug_agg['DRUG_CODE']:
                code, desc = _top_rejection_for_entity(
                    new_drug_df[new_drug_df['DRUG_CODE'] == drug_code]
                )
                top_codes.append(code)
                top_descs.append(desc)
            
            drug_agg['Top_Rejection_Code'] = top_codes
            drug_agg['Top_Rejection_Desc'] = top_descs
            
            # Add drug name if available
            if 'DRUG_NAME' in drug_df.columns:
                drug_names = drug_df[['DRUG_CODE', 'DRUG_NAME']].drop_duplicates('DRUG_CODE')
                drug_agg = drug_agg.merge(drug_names, on='DRUG_CODE', how='left')
                display_cols = ['DRUG_CODE', 'DRUG_NAME', 'Claims', 'Approved', 'Rejected', 
                                'Rejection_Rate', 'Top_Rejection_Code', 'Top_Rejection_Desc']
            else:
                display_cols = ['DRUG_CODE', 'Claims', 'Approved', 'Rejected', 
                                'Rejection_Rate', 'Top_Rejection_Code', 'Top_Rejection_Desc']
            
            results['drugs'] = drug_agg[[c for c in display_cols if c in drug_agg.columns]].sort_values(
                'Claims', ascending=False
            )
    
    # ── NEW DIAGNOSES ──────────────────────────────────────────────
    if 'PA_PRIMARY_DIAG' in drug_df.columns and 'IS_REJECTED' in drug_df.columns:
        current_diags = set(drug_df['PA_PRIMARY_DIAG'].dropna().astype(str).unique())
        new_diags = current_diags - baseline_diags
        
        if new_diags:
            new_diag_df = drug_df[drug_df['PA_PRIMARY_DIAG'].astype(str).isin(new_diags)].copy()
            
            diag_agg = (
                new_diag_df.groupby('PA_PRIMARY_DIAG')
                .agg(
                    Claims=('IS_REJECTED', 'count'),
                    Approved=('IS_REJECTED', lambda x: (x == 0).sum()),
                    Rejected=('IS_REJECTED', 'sum'),
                )
                .reset_index()
            )
            
            diag_agg['Rejection_Rate'] = (
                (diag_agg['Rejected'] / diag_agg['Claims'] * 100).round(1).astype(str) + '%'
            )
            
            top_codes = []
            top_descs = []
            for diag in diag_agg['PA_PRIMARY_DIAG']:
                code, desc = _top_rejection_for_entity(
                    new_diag_df[new_diag_df['PA_PRIMARY_DIAG'] == diag]
                )
                top_codes.append(code)
                top_descs.append(desc)
            
            diag_agg['Top_Rejection_Code'] = top_codes
            diag_agg['Top_Rejection_Desc'] = top_descs
            
            results['diagnoses'] = diag_agg[[
                'PA_PRIMARY_DIAG', 'Claims', 'Approved', 'Rejected', 
                'Rejection_Rate', 'Top_Rejection_Code', 'Top_Rejection_Desc'
            ]].sort_values('Claims', ascending=False)
    
    # ── NEW PROVIDERS ──────────────────────────────────────────────
    if 'DOC_LIC_NO' in drug_df.columns and 'IS_REJECTED' in drug_df.columns:
        current_provs = set(drug_df['DOC_LIC_NO'].dropna().astype(str).unique())
        new_provs = current_provs - baseline_provs
        
        if new_provs:
            new_prov_df = drug_df[drug_df['DOC_LIC_NO'].astype(str).isin(new_provs)].copy()
            
            prov_agg = (
                new_prov_df.groupby('DOC_LIC_NO')
                .agg(
                    Claims=('IS_REJECTED', 'count'),
                    Approved=('IS_REJECTED', lambda x: (x == 0).sum()),
                    Rejected=('IS_REJECTED', 'sum'),
                )
                .reset_index()
            )
            
            prov_agg['Rejection_Rate'] = (
                (prov_agg['Rejected'] / prov_agg['Claims'] * 100).round(1).astype(str) + '%'
            )
            
            top_codes = []
            top_descs = []
            for prov in prov_agg['DOC_LIC_NO']:
                code, desc = _top_rejection_for_entity(
                    new_prov_df[new_prov_df['DOC_LIC_NO'] == prov]
                )
                top_codes.append(code)
                top_descs.append(desc)
            
            prov_agg['Top_Rejection_Code'] = top_codes
            prov_agg['Top_Rejection_Desc'] = top_descs
            
            # Add provider name if available
            if 'DOC_NAME' in drug_df.columns:
                prov_names = drug_df[['DOC_LIC_NO', 'DOC_NAME']].drop_duplicates('DOC_LIC_NO')
                prov_agg = prov_agg.merge(prov_names, on='DOC_LIC_NO', how='left')
                display_cols = ['DOC_LIC_NO', 'DOC_NAME', 'Claims', 'Approved', 'Rejected',
                                'Rejection_Rate', 'Top_Rejection_Code', 'Top_Rejection_Desc']
            else:
                display_cols = ['DOC_LIC_NO', 'Claims', 'Approved', 'Rejected',
                                'Rejection_Rate', 'Top_Rejection_Code', 'Top_Rejection_Desc']
            
            results['providers'] = prov_agg[[c for c in display_cols if c in prov_agg.columns]].sort_values(
                'Claims', ascending=False
            )
    
    # ── NEW DRUG-DIAGNOSIS COMBOS ──────────────────────────────────
    if 'DRUG_DIAG_COMBO' in drug_df.columns and 'IS_REJECTED' in drug_df.columns:
        current_combos = set(drug_df['DRUG_DIAG_COMBO'].dropna().astype(str).unique())
        new_combos = current_combos - baseline_combos
        
        if new_combos:
            new_combo_df = drug_df[drug_df['DRUG_DIAG_COMBO'].astype(str).isin(new_combos)].copy()
            
            combo_agg = (
                new_combo_df.groupby('DRUG_DIAG_COMBO')
                .agg(
                    Claims=('IS_REJECTED', 'count'),
                    Approved=('IS_REJECTED', lambda x: (x == 0).sum()),
                    Rejected=('IS_REJECTED', 'sum'),
                )
                .reset_index()
            )
            
            combo_agg['Rejection_Rate'] = (
                (combo_agg['Rejected'] / combo_agg['Claims'] * 100).round(1).astype(str) + '%'
            )
            
            top_codes = []
            top_descs = []
            for combo in combo_agg['DRUG_DIAG_COMBO']:
                code, desc = _top_rejection_for_entity(
                    new_combo_df[new_combo_df['DRUG_DIAG_COMBO'] == combo]
                )
                top_codes.append(code)
                top_descs.append(desc)
            
            combo_agg['Top_Rejection_Code'] = top_codes
            combo_agg['Top_Rejection_Desc'] = top_descs

            # Remove tiny combos
            combo_agg = combo_agg[(combo_agg['Claims'] >= 4)]

            # Optional: keep only interesting combos
            combo_agg = combo_agg[(combo_agg['Rejected'] > 0) |(combo_agg['Claims'] >= 10)]

            results['combos'] = combo_agg[[
                'DRUG_DIAG_COMBO',
                'Claims',
                'Approved',
                'Rejected',
                'Rejection_Rate',
                'Top_Rejection_Code',
                'Top_Rejection_Desc'
            ]].sort_values('Claims', ascending=False)
    
    return results
 
