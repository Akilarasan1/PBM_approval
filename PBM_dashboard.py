"""PBM Claims Intelligence Dashboard - Streamlit App"""

import streamlit as st
import pandas as pd
from insight import gen_insights

# Import modular functions
from utils import STREAMLIT_CSS
from data import process, compute_code_stats, compute_drug_diag_combos, compute_provider_stats, validate_columns
from viz import (
    plot_rejection_codes_volume_financial, plot_mnec_breakdown,
    plot_gender_rejection_rate, plot_age_rejection_rate,
    plot_rejected_amount_by_code, plot_amount_distribution_treemap,
    plot_top_rejected_drugs, plot_high_risk_combos,
    plot_provider_volume_and_rejection, plot_provider_risk_map,
    plot_age_violations
)

# ── PAGE CONFIG ────────────────────────────────────────────────
st.set_page_config(
    page_title="PBM Claims Intelligence",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── APPLY DESIGN TOKENS ────────────────────────────────────────
st.markdown(STREAMLIT_CSS, unsafe_allow_html=True)

# ── SIDEBAR ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:16px 0 24px'>
        <div style='font-size:20px;font-weight:700;color:#E6EDF3;letter-spacing:-0.5px'>💊 PBM Intelligence</div>
        <div style='font-size:11px;color:#8B949E;margin-top:4px;letter-spacing:0.05em;text-transform:uppercase'>Claims Analytics Platform</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Data Source</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Upload monthly CSV or Excel",
        type=['csv', 'xlsx'],
        help="Upload your monthly PBM claims file"
    )
    st.markdown("---")
    st.markdown('<div class="section-header">Filters</div>', unsafe_allow_html=True)

    if uploaded:
        try:
            drug_df = process(uploaded.read(), uploaded.name)
            
            # Validate columns and show warnings
            missing_required, missing_optional = validate_columns(drug_df)
            if missing_optional:
                with st.sidebar.expander("⚠️ Missing Optional Features"):
                    st.warning(f"Some features disabled due to missing columns: {', '.join(sorted(missing_optional))}")
            
            rejected = drug_df[drug_df['IS_REJECTED'] == 1].copy()
            approved = drug_df[drug_df['IS_REJECTED'] == 0].copy()

            gender_opts = ['All'] + sorted(drug_df['MEM_GENDER'].dropna().unique().tolist()) if 'MEM_GENDER' in drug_df.columns else ['All']
            sel_gender = st.selectbox('Gender', gender_opts)

            age_opts = ['All'] + sorted(drug_df['AGE_GROUP'].dropna().unique().tolist()) if 'AGE_GROUP' in drug_df.columns else ['All']
            sel_age = st.selectbox('Age Group', age_opts)

            if sel_gender != 'All':
                drug_df = drug_df[drug_df['MEM_GENDER'] == sel_gender]
                rejected = drug_df[drug_df['IS_REJECTED'] == 1]
                approved = drug_df[drug_df['IS_REJECTED'] == 0]
            if sel_age != 'All' and 'AGE_GROUP' in drug_df.columns:
                drug_df = drug_df[drug_df['AGE_GROUP'] == sel_age]
                rejected = drug_df[drug_df['IS_REJECTED'] == 1]
                approved = drug_df[drug_df['IS_REJECTED'] == 0]

            st.markdown("---")
            # Quick stats
            total = len(drug_df)
            rej_count = len(rejected)
            rej_rate = rej_count / total * 100 if total > 0 else 0
            st.markdown(f"""
            <div style='font-size:11px;color:#8B949E;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:12px'>Quick Stats</div>
            <div style='font-size:13px;color:#C9D1D9;line-height:2'>
                📊 Total claims: <b style='color:#E6EDF3'>{total:,}</b><br>
                ❌ Rejected: <b style='color:#FF4444'>{rej_count:,}</b><br>
                ✅ Approved: <b style='color:#00C853'>{total-rej_count:,}</b><br>
                📈 Rejection rate: <b style='color:{'#FF4444' if rej_rate>17 else '#FF8C00' if rej_rate>14 else '#00C853'}'>{rej_rate:.1f}%</b>
            </div>
            """, unsafe_allow_html=True)
        
        except Exception as e:
            st.sidebar.error(f"Error processing file: {str(e)}")
            st.stop()

# ── MAIN CONTENT ───────────────────────────────────────────────
if not uploaded or not 'drug_df' in locals():
    # Landing screen
    st.markdown("""
    <div style='display:flex;flex-direction:column;align-items:center;justify-content:center;min-height:70vh;text-align:center'>
        <div style='font-size:72px;margin-bottom:24px'>💊</div>
        <h1 style='font-size:36px;font-weight:700;color:#E6EDF3;letter-spacing:-1px;margin:0 0 12px'>
            PBM Claims Intelligence
        </h1>
        <p style='font-size:16px;color:#8B949E;max-width:480px;line-height:1.6;margin:0 0 40px'>
            Upload your monthly prescribed drug claims file to instantly surface rejection patterns, 
            financial impact, fraud signals, and actionable insights.
        </p>
        <div style='display:grid;grid-template-columns:repeat(3,1fr);gap:16px;max-width:640px;margin-bottom:48px'>
            <div style='background:#161B22;border:1px solid #21262D;border-radius:12px;padding:20px;'>
                <div style='font-size:28px;margin-bottom:8px'>🔍</div>
                <div style='font-size:13px;font-weight:600;color:#E6EDF3;margin-bottom:4px'>Pattern Detection</div>
                <div style='font-size:12px;color:#8B949E'>Drug-diagnosis exclusion rules built automatically</div>
            </div>
            <div style='background:#161B22;border:1px solid #21262D;border-radius:12px;padding:20px;'>
                <div style='font-size:28px;margin-bottom:8px'>💰</div>
                <div style='font-size:13px;font-weight:600;color:#E6EDF3;margin-bottom:4px'>Financial Impact</div>
                <div style='font-size:12px;color:#8B949E'>Rejected amount breakdown by rejection code</div>
            </div>
            <div style='background:#161B22;border:1px solid #21262D;border-radius:12px;padding:20px;'>
                <div style='font-size:28px;margin-bottom:8px'>🚨</div>
                <div style='font-size:13px;font-weight:600;color:#E6EDF3;margin-bottom:4px'>Auto Insights</div>
                <div style='font-size:12px;color:#8B949E'>Fraud flags, safety alerts, policy signals</div>
            </div>
        </div>
        <div style='font-size:13px;color:#8B949E'>
            ↑ Upload your CSV or Excel file in the sidebar to begin
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── COMPUTE CORE STATS ─────────────────────────────────────────
total = len(drug_df)
rej_count = len(rejected)
rej_rate = rej_count / total * 100 if total > 0 else 0

code_stats = compute_code_stats(rejected)
high_risk, safe_combos, gray_combos = compute_drug_diag_combos(drug_df)

insights = gen_insights(drug_df, rejected, code_stats, high_risk)

total_rej_amt = rejected['TREAT_REJ_AMT'].sum() if 'TREAT_REJ_AMT' in rejected.columns else 0
total_est_amt = drug_df['TREAT_EST_AMT'].sum() if 'TREAT_EST_AMT' in drug_df.columns else 0

# ── TABS ───────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview", "💰 Financial", "🔍 Combos", "👨‍⚕️ Providers", "🚨 Fraud & Safety"
])

# ════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ════════════════════════════════════════════════════
with tab1:
    # KPI row
    cols = st.columns(5)
    kpis = [
        ('red',    'TOTAL CLAIMS',    f'{total:,}',            f'{rej_count:,} rejected'),
        ('red',    'REJECTION RATE',  f'{rej_rate:.1f}%',      'of all claims'),
        ('green',  'APPROVAL RATE',   f'{100-rej_rate:.1f}%',  f'{total-rej_count:,} approved'),
        ('blue',   'UNIQUE DRUGS',    f'{drug_df["DRUG_CODE"].nunique():,}', 'distinct drug codes'),
        ('amber',  'HIGH RISK COMBOS',f'{len(high_risk)}',      'drug-diag exclusions'),
    ]
    for col, (color, label, value, sub) in zip(cols, kpis):
        with col:
            st.markdown(f"""
            <div class="metric-card {color}">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
                <div class="metric-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns([1.4, 1])

    with col1:
        st.markdown('<div class="section-header">Rejection Codes — Volume & Financial</div>', unsafe_allow_html=True)
        fig = plot_rejection_codes_volume_financial(code_stats)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info('No rejection code data available.')

    with col2:
        st.markdown('<div class="section-header">Auto Insights</div>', unsafe_allow_html=True)
        for sev, msg in insights:
            st.markdown(f'<div class="insight-card {sev}">{msg}</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-header">Rejection Code Breakdown</div>', unsafe_allow_html=True)
    display_df = code_stats.copy()
    display_df['Rejected_Amt'] = display_df['Rejected_Amt'].apply(lambda x: f'{x:,.0f}')
    display_df['Count'] = display_df['Count'].apply(lambda x: f'{x:,}')
    display_df['Pct'] = display_df['Pct'].apply(lambda x: f'{x:.1f}%')
    display_df.columns = ['Code', 'Claims', 'Rejected Amt', '% of Rejections', 'Description']
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # MNEC breakdown
    if 'MNEC' in rejected['REJ_CODE_PREFIX'].values and 'REJ_REMARKS' in rejected.columns:
        st.markdown('<div class="section-header">MNEC Breakdown (Auto-decoded)</div>', unsafe_allow_html=True)
        mnec_df = rejected[rejected['REJ_CODE_PREFIX'] == 'MNEC']
        fig = plot_mnec_breakdown(mnec_df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info('MNEC data unavailable for breakdown.')

    # Gender & Age
    c1, c2 = st.columns(2)
    with c1:
        if 'MEM_GENDER' in drug_df.columns:
            st.markdown('<div class="section-header">Rejection Rate by Gender</div>', unsafe_allow_html=True)
            fig = plot_gender_rejection_rate(drug_df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)

    with c2:
        if 'AGE_GROUP' in drug_df.columns:
            st.markdown('<div class="section-header">Rejection Rate by Age Group</div>', unsafe_allow_html=True)
            fig = plot_age_rejection_rate(drug_df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)

# ════════════════════════════════════════════════════
# TAB 2 — FINANCIAL
# ════════════════════════════════════════════════════
with tab2:
    f1, f2, f3 = st.columns(3)
    with f1:
        st.markdown(f"""<div class="metric-card red">
            <div class="metric-label">Total Rejected Amount</div>
            <div class="metric-value">{total_rej_amt:,.0f}</div>
            <div class="metric-sub">{total_rej_amt/total_est_amt*100:.1f}% of estimated</div>
        </div>""", unsafe_allow_html=True)
    with f2:
        st.markdown(f"""<div class="metric-card green">
            <div class="metric-label">Total Approved Amount</div>
            <div class="metric-value">{drug_df['TREAT_APPR_AMT'].sum() if 'TREAT_APPR_AMT' in drug_df.columns else 0:,.0f}</div>
            <div class="metric-sub">Net paid out</div>
        </div>""", unsafe_allow_html=True)
    with f3:
        avg_rej = rejected['TREAT_REJ_AMT'].mean() if len(rejected) > 0 else 0
        st.markdown(f"""<div class="metric-card amber">
            <div class="metric-label">Avg Rejected Per Claim</div>
            <div class="metric-value">{avg_rej:,.0f}</div>
            <div class="metric-sub">Per rejection event</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-header">Rejected Amount by Code</div>', unsafe_allow_html=True)
        fig = plot_rejected_amount_by_code(code_stats)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown('<div class="section-header">Amount Distribution (Treemap)</div>', unsafe_allow_html=True)
        if (not code_stats.empty and code_stats['Rejected_Amt'].fillna(0).sum() > 0):
            fig = plot_amount_distribution_treemap(code_stats)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info('No rejected amount data available for treemap.')

    # Top rejected drugs by amount
    st.markdown('<div class="section-header">Top 15 Most Expensive Rejected Drugs</div>', unsafe_allow_html=True)
    fig = plot_top_rejected_drugs(rejected)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info('No rejected drug data available.')

# ════════════════════════════════════════════════════
# TAB 3 — COMBOS
# ════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">High Risk Drug-Diagnosis Combinations (>70% rejection, min 20 claims)</div>', unsafe_allow_html=True)
    if len(high_risk) > 0:
        fig = plot_high_risk_combos(high_risk)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        st.dataframe(high_risk[['DRUG_DIAG_COMBO','Total','Rejected','RejRate']].head(30),
                     use_container_width=True, hide_index=True)
    else:
        st.info('No high-risk combos found this month (min 20 claims, >70% rejection).')

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-header">Safe Combos (0% rejection, min 20 claims)</div>', unsafe_allow_html=True)
        if len(safe_combos) > 0:
            st.dataframe(safe_combos[['DRUG_DIAG_COMBO','Total','Rejected','RejRate']].head(20),
                         use_container_width=True, hide_index=True)
        else:
            st.info('No safe combos found.')

    with c2:
        st.markdown('<div class="section-header">Gray Area (30-70% rejection, min 20 claims)</div>', unsafe_allow_html=True)
        if len(gray_combos) > 0:
            st.dataframe(gray_combos[['DRUG_DIAG_COMBO','Total','Rejected','RejRate']].head(20),
                         use_container_width=True, hide_index=True)
        else:
            st.info('No gray area combos found.')

# ════════════════════════════════════════════════════
# TAB 4 — PROVIDERS
# ════════════════════════════════════════════════════
with tab4:
    if 'DOC_LIC_NO' in drug_df.columns:
        prov = compute_provider_stats(drug_df)

        p1, p2, p3 = st.columns(3)
        with p1:
            st.markdown(f"""<div class="metric-card blue">
                <div class="metric-label">Unique Providers</div>
                <div class="metric-value">{len(prov):,}</div>
                <div class="metric-sub">Active this month</div>
            </div>""", unsafe_allow_html=True)
        with p2:
            flagged_provs = prov[(prov['RejRate'] > 30) & (prov['Claims'] >= 50)]
            st.markdown(f"""<div class="metric-card red">
                <div class="metric-label">Flagged Providers</div>
                <div class="metric-value">{len(flagged_provs)}</div>
                <div class="metric-sub">&gt;30% rejection, min 50 claims</div>
            </div>""", unsafe_allow_html=True)
        with p3:
            top_prov = prov.sort_values('RejAmt', ascending=False).iloc[0]
            st.markdown(f"""<div class="metric-card amber">
                <div class="metric-label">Highest Rejected Amt</div>
                <div class="metric-value">{top_prov['RejAmt']:,.0f}</div>
                <div class="metric-sub">Provider {top_prov['DOC_LIC_NO']}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        c1, c2 = st.columns([1.2, 1])
        with c1:
            st.markdown('<div class="section-header">Top 20 Providers by Volume</div>', unsafe_allow_html=True)
            fig = plot_provider_volume_and_rejection(prov)
            if fig:
                st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.markdown('<div class="section-header">⚠️ Flagged Providers (&gt;30% rejection)</div>', unsafe_allow_html=True)
            if len(flagged_provs) > 0:
                flagged_show = flagged_provs.sort_values('RejRate', ascending=False)[['DOC_LIC_NO','Claims','RejRate','RejAmt']]
                flagged_show.columns = ['Provider', 'Claims', 'Rej Rate %', 'Rej Amount']
                st.dataframe(flagged_show, use_container_width=True, hide_index=True)
            else:
                st.success('No providers flagged this month.')

        # Provider scatter
        st.markdown('<div class="section-header">Provider Risk Map (Claims vs Rejection Rate)</div>', unsafe_allow_html=True)
        fig = plot_provider_risk_map(prov)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning('DOC_LIC_NO column not found in dataset.')

# ════════════════════════════════════════════════════
# TAB 5 — FRAUD & SAFETY
# ════════════════════════════════════════════════════
with tab5:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">🔍 Fraud Flagged Claims</div>', unsafe_allow_html=True)
        if ('PA_FLAG_REASON' in drug_df.columns and drug_df['PA_FLAG_REASON'].notna().any()):
            fraud_df = drug_df[drug_df['PA_FLAG_REASON'].str.contains('fraud', case=False, na=False)].copy()
            if len(fraud_df) > 0:
                st.markdown(f"""<div class="insight-card critical">
                    🔴 <b>{len(fraud_df):,} claims flagged as potential fraud</b><br>
                    Rejection rate: {fraud_df['IS_REJECTED'].mean()*100:.1f}%<br>
                    Avg claim amount: {fraud_df['TREAT_EST_AMT'].mean():.0f} 
                    vs dataset avg {drug_df['TREAT_EST_AMT'].mean():.0f}
                    ({fraud_df['TREAT_EST_AMT'].mean()/drug_df['TREAT_EST_AMT'].mean():.1f}x higher)
                </div>""", unsafe_allow_html=True)

                show_cols = [c for c in ['DRUG_CODE','PA_PRIMARY_DIAG','PA_MEM_AGE','MEM_GENDER',
                                         'DOC_LIC_NO','TREAT_EST_AMT','PBM_APPR_STS','REJ_CODE_PREFIX',
                                         'PA_FLAG_REASON'] if c in fraud_df.columns]
                st.dataframe(fraud_df[show_cols], use_container_width=True, hide_index=True)
            else:
                st.success('No fraud-flagged claims this month.')
        else:
            st.info('PA_FLAG_REASON column not found.')

    with col2:
        st.markdown('<div class="section-header">🚨 Children Age Rule Violations</div>', unsafe_allow_html=True)
        if 'AGE_GROUP' in drug_df.columns:
            child_v = rejected[(rejected['REJ_CODE_PREFIX'] == 'CODE') &
                               (rejected['AGE_GROUP'].isin(['INFANT (0-2)','CHILD (3-12)','TEEN (13-17)']))]
            if len(child_v) > 0:
                st.markdown(f"""<div class="insight-card critical">
                    🚨 <b>{len(child_v):,} age rule violations for children</b><br>
                    Adult drugs prescribed to minors. Patient safety concern.
                </div>""", unsafe_allow_html=True)

                fig = plot_age_violations(child_v)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)

                st.markdown('**Top drugs violating age rules:**')
                top_age_drugs = child_v['DRUG_CODE'].value_counts().head(10).reset_index()
                top_age_drugs.columns = ['Drug Code', 'Violations']
                st.dataframe(top_age_drugs, use_container_width=True, hide_index=True)
            else:
                st.success('No age rule violations for children this month.')
        else:
            st.info('AGE_GROUP column requires PA_MEM_AGE in dataset.')

    # New drugs 100% NCOV
    st.markdown('<div class="section-header">🆕 New Drugs with 100% Not-Covered Rejection</div>', unsafe_allow_html=True)
    if 'SERVICE_DT' in drug_df.columns:
        dated = drug_df[drug_df['SERVICE_DT'].notna()].copy()
        first_seen = dated.groupby('DRUG_CODE')['SERVICE_DT'].min().reset_index()
        first_seen.columns = ['DRUG_CODE','FIRST_SEEN']
        dated = dated.merge(first_seen, on='DRUG_CODE')
        dated['IS_NEW'] = (dated['FIRST_SEEN'] >= dated['SERVICE_DT'].min()).astype(int)
        new_drugs = dated[dated['IS_NEW'] == 1]
        ncov_stats = (new_drugs.groupby('DRUG_CODE')
                      .agg(Total=('IS_REJECTED','count'),
                           Rejected=('IS_REJECTED','sum'),
                           NCOV=('REJ_CODE_PREFIX', lambda x: (x=='NCOV').sum()),
                           Amt=('TREAT_REJ_AMT','sum'),
                           First=('FIRST_SEEN','min'))
                      .reset_index())
        ncov_stats['RejRate'] = (ncov_stats['Rejected']/ncov_stats['Total']*100).round(1)
        always_ncov = ncov_stats[
            (ncov_stats['NCOV'] == ncov_stats['Rejected']) &
            (ncov_stats['Rejected'] == ncov_stats['Total']) &
            (ncov_stats['Total'] >= 5)
        ].sort_values('Total', ascending=False)

        if len(always_ncov) > 0:
            st.markdown(f"""<div class="insight-card warning">
                🚫 <b>{len(always_ncov)} new drug codes with 100% NCOV rejection</b><br>
                These drugs are being prescribed but are not in the coverage formulary.
                Immediate formulary review needed.
            </div>""", unsafe_allow_html=True)
            always_ncov['Amt'] = always_ncov['Amt'].round(0)
            st.dataframe(always_ncov[['DRUG_CODE','First','Total','NCOV','Amt']].head(20),
                         use_container_width=True, hide_index=True)
        else:
            st.success('No new drugs with 100% NCOV rejection found.')
