import pandas as pd



# ── OUTPATIENT vs INPATIENT ──────────────────────────────────────────────
SERVICE_TYPE_LABELS = {'O': 'Outpatient', 'I': 'Inpatient'}


def compute_service_type_stats(drug_df):
    """
    Rejection count/amount/rate by service type (Outpatient vs Inpatient).
    Sample sizes are usually very lopsided (inpatient is typically a small
    fraction of pharmacy claims) — Claims is always returned alongside the
    rate so a tiny-sample rate isn't read as a strong signal on its own.

    Returns:
        DataFrame: Service_Type, Claims, Approved, Rejected, RejRate_%,
        Rejected_Amt
    """
    if 'SERVICE_TYPE' not in drug_df.columns or 'IS_REJECTED' not in drug_df.columns:
        return pd.DataFrame()

    g = drug_df.groupby('SERVICE_TYPE').agg(
        Claims=('IS_REJECTED', 'count'),
        Rejected=('IS_REJECTED', 'sum'),
        Rejected_Amt=('TREAT_REJ_AMT', 'sum') if 'TREAT_REJ_AMT' in drug_df.columns else ('IS_REJECTED', 'sum'),
    ).reset_index()
    g['Approved'] = g['Claims'] - g['Rejected']
    g['RejRate_%'] = (g['Rejected'] / g['Claims'] * 100).round(1)
    g['Service_Type'] = g['SERVICE_TYPE'].map(SERVICE_TYPE_LABELS).fillna(g['SERVICE_TYPE'])
    return g[['Service_Type', 'Claims', 'Approved', 'Rejected', 'RejRate_%', 'Rejected_Amt']].sort_values(
        'Claims', ascending=False
    ).reset_index(drop=True)


def compute_service_type_monthly_trend(full_df):
    """Month-by-month rejection rate per service type, across the full upload."""
    if 'SERVICE_TYPE' not in full_df.columns or 'SERVICE_DT' not in full_df.columns:
        return pd.DataFrame()

    dated = full_df[full_df['SERVICE_DT'].notna()].copy()
    if dated.empty:
        return pd.DataFrame()

    dated['Month'] = dated['SERVICE_DT'].dt.to_period('M').astype(str)
    g = dated.groupby(['Month', 'SERVICE_TYPE']).agg(
        Claims=('IS_REJECTED', 'count'), Rejected=('IS_REJECTED', 'sum'),
    ).reset_index()
    g['RejRate_%'] = (g['Rejected'] / g['Claims'] * 100).round(1)
    g['Service_Type'] = g['SERVICE_TYPE'].map(SERVICE_TYPE_LABELS).fillna(g['SERVICE_TYPE'])
    return g.sort_values('Month')


# ── QUANTITY IMPACT ───────────────────────────────────────────────────────
QUANTITY_BAND_ORDER = ['1', '2-5', '6-10', '10+']


def _quantity_band(qty):
    if pd.isna(qty):
        return None
    if qty <= 1:
        return '1'
    if qty <= 5:
        return '2-5'
    if qty <= 10:
        return '6-10'
    return '10+'


def compute_quantity_bands(drug_df):
    """
    Claims/approval/rejection broken into quantity bands (1, 2-5, 6-10, 10+).
    Note: a small number of extreme outlier quantities (data entry errors,
    e.g. a single claim with qty=7000) fall into the open-ended "10+" band —
    they're rare enough not to distort the aggregate rate, but are visible
    if you drill into that band's claims directly.

    Returns:
        DataFrame: Band, Claims, Approved, Rejected, Approval_Rate_%,
        Rejection_Rate_% — ordered 1 -> 2-5 -> 6-10 -> 10+
    """
    if 'PA_QTY' not in drug_df.columns or 'IS_REJECTED' not in drug_df.columns:
        return pd.DataFrame()

    df = drug_df.copy()
    df['Band'] = df['PA_QTY'].apply(_quantity_band)
    df = df[df['Band'].notna()]
    if df.empty:
        return pd.DataFrame()

    g = df.groupby('Band').agg(
        Claims=('IS_REJECTED', 'count'), Rejected=('IS_REJECTED', 'sum'),
    ).reset_index()
    g['Approved'] = g['Claims'] - g['Rejected']
    g['Rejection_Rate_%'] = (g['Rejected'] / g['Claims'] * 100).round(1)
    g['Approval_Rate_%'] = (100 - g['Rejection_Rate_%']).round(1)
    g['order'] = g['Band'].apply(lambda b: QUANTITY_BAND_ORDER.index(b) if b in QUANTITY_BAND_ORDER else 99)
    return g.sort_values('order').drop(columns='order').reset_index(drop=True)


def compute_quantity_rejection_by_drug(drug_df, min_claims_per_band=5, min_delta=15.0):
    """
    For each drug with claims spread across multiple quantity bands, check
    whether the rejection rate is meaningfully higher at higher quantities
    than at lower quantities for THAT SAME drug — a per-drug comparison,
    not a population-wide one, since different drugs have very different
    baseline rejection rates and typical quantities.

    Args:
        drug_df: DataFrame with claims
        min_claims_per_band: minimum claims in both the lowest and highest
            engaged band for a drug to be considered
        min_delta: minimum percentage-point gap (highest band rate minus
            lowest band rate) to be reported

    Returns:
        DataFrame: DRUG_CODE, Lowest_Band, Lowest_Band_RejRate_%,
        Highest_Band, Highest_Band_RejRate_%, Delta_Pts, Total_Claims —
        sorted by Delta_Pts descending
    """
    if 'PA_QTY' not in drug_df.columns or 'DRUG_CODE' not in drug_df.columns:
        return pd.DataFrame()

    df = drug_df.copy()
    df['Band'] = df['PA_QTY'].apply(_quantity_band)
    df = df[df['Band'].notna()]
    if df.empty:
        return pd.DataFrame()

    band_stats = (
        df.groupby(['DRUG_CODE', 'Band'])
        .agg(Claims=('IS_REJECTED', 'count'), Rejected=('IS_REJECTED', 'sum'))
        .reset_index()
    )
    band_stats['RejRate_%'] = (band_stats['Rejected'] / band_stats['Claims'] * 100).round(1)
    band_stats['order'] = band_stats['Band'].apply(
        lambda b: QUANTITY_BAND_ORDER.index(b) if b in QUANTITY_BAND_ORDER else 99
    )

    rows = []
    for drug, g in band_stats.groupby('DRUG_CODE'):
        eligible = g[g['Claims'] >= min_claims_per_band].sort_values('order')
        if len(eligible) < 2:
            continue
        lowest = eligible.iloc[0]
        highest = eligible.iloc[-1]
        delta = highest['RejRate_%'] - lowest['RejRate_%']
        if delta >= min_delta:
            rows.append({
                'DRUG_CODE': drug,
                'Lowest Quantity Range': lowest['Band'],
                'Rejection Rate (Lowest Quantity)': lowest['RejRate_%'],
                'Highest Quantity Range': highest['Band'],
                'Rejection Rate (Highest Quantity)': highest['RejRate_%'],
                'Increase in Rejection Rate (%)': round(delta, 1),
                'Total_Claims': int(g['Claims'].sum()),
            })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values('Increase in Rejection Rate (%)', ascending=False).reset_index(drop=True)