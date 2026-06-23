"""PBM Claims Intelligence Dashboard - Streamlit App"""

import streamlit as st
import pandas as pd
import io
from pathlib import Path
from insight import gen_insights
import time
from utils import STREAMLIT_CSS, THRESHOLDS, SAMPLE_DATA_FILE

from data import (
    process, compute_code_stats, compute_drug_diag_combos, compute_provider_stats,
    compute_weekly_trends, compute_new_drugs_always_ncov, compute_provider_investigation,
    compute_provider_detail, enrich_combo_display, validate_columns,
    compute_payment_anomalies, summarize_payment_anomalies,
)



from viz import (
    plot_rejection_codes_volume_financial, plot_mnec_breakdown,
    plot_age_rejection_rate, plot_rejected_amount_by_code,
    plot_top_rejected_drugs, plot_high_risk_combos,
    plot_provider_volume_and_rejection, plot_provider_risk_map,
    plot_age_violations, plot_weekly_trends,
    plot_anomaly_scatter, plot_findings_by_dimension,
)
import history
import anomaly

APP_DIR = Path(__file__).parent
SAMPLE_PATH = APP_DIR / SAMPLE_DATA_FILE

# ── PAGE CONFIG ────────────────────────────────────────────────
st.set_page_config(
    page_title="PBM Claims Intelligence",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── APPLY DESIGN TOKENS ────────────────────────────────────────
st.markdown(STREAMLIT_CSS, unsafe_allow_html=True)

file_bytes = None
filename = None

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
        help="Upload your monthly PBM claims file",
    )


    if uploaded is not None:
        st.session_state['use_sample_data'] = False
        file_bytes = uploaded.read()
        filename = uploaded.name
    # elif st.session_state.get('use_sample_data') and SAMPLE_PATH.exists():
    #     file_bytes = SAMPLE_PATH.read_bytes()
    #     filename = SAMPLE_PATH.name

    st.markdown("---")
    st.markdown('<div class="section-header">Filters</div>', unsafe_allow_html=True)

    if file_bytes and filename:
        try:
            t1 = time.perf_counter()
            print("fileed entered intto system",t1)
            drug_df_full_raw = process(file_bytes, filename)
            print(f' time taking to file load :: {(time.perf_counter() - t1) / 60:.2f} minutes')
            missing_required, missing_optional = validate_columns(drug_df_full_raw)
            if missing_optional:
                with st.sidebar.expander("⚠️ Missing Optional Features"):
                    st.warning(f"Some features disabled due to missing columns: {', '.join(sorted(missing_optional))}")

            # ── DYNAMIC PERIOD DETECTION ─────────────────────────────
            # Split whatever was uploaded into per-calendar-month buckets
            # and persist every one as a snapshot — this is what lets one
            # uploader handle 1, 3, 6, or 12 months interchangeably.
            month_buckets = history.split_by_month(drug_df_full_raw)
            month_labels_sorted = sorted(month_buckets.keys())

            for m_label, m_df in month_buckets.items():
                history.save_snapshot(m_label, history.build_snapshot(m_df))

            OVERALL_LABEL = "📊 Overall (All Months)"
            is_multi_month = len(month_labels_sorted) > 1

            st.markdown("---")
            st.markdown('<div class="section-header">Reporting Period</div>', unsafe_allow_html=True)

            if is_multi_month:
                st.caption(
                    f"📥 Detected **{len(month_labels_sorted)} months** "
                    f"({', '.join(month_labels_sorted)}) — all saved automatically as history."
                )
                period_options = [OVERALL_LABEL] + list(reversed(month_labels_sorted))
                selected_period = st.selectbox(
                    "Period",
                    options=period_options,
                    index=0,  # default = Overall
                    help="'Overall' combines every uploaded month into one summary. "
                         "Pick a specific month to drill into just that period.",
                )
            else:
                selected_period = month_labels_sorted[0]
                st.caption(f"📅 Single month detected: **{selected_period}**")

            is_overall = is_multi_month and selected_period == OVERALL_LABEL
            latest_month_label = month_labels_sorted[-1]

            # The month used for drift/novelty math is always one specific
            # month — in Overall view that's the most recent one uploaded.
            emerging_month_label = latest_month_label if is_overall else selected_period

            # ── FILTER OPTIONS (derived from the full unfiltered upload) ─
            st.markdown("---")
            st.markdown('<div class="section-header">Filters</div>', unsafe_allow_html=True)

            gender_opts = ['All'] + sorted(drug_df_full_raw['MEM_GENDER'].dropna().unique().tolist()) if 'MEM_GENDER' in drug_df_full_raw.columns else ['All']
            sel_gender = st.selectbox('Gender', gender_opts)

            age_opts = ['All'] + sorted(drug_df_full_raw['AGE_GROUP'].dropna().unique().tolist()) if 'AGE_GROUP' in drug_df_full_raw.columns else ['All']
            sel_age = st.selectbox('Age Group', age_opts)

            def _apply_filters(df):
                if sel_gender != 'All' and 'MEM_GENDER' in df.columns:
                    df = df[df['MEM_GENDER'] == sel_gender]
                if sel_age != 'All' and 'AGE_GROUP' in df.columns:
                    df = df[df['AGE_GROUP'] == sel_age]
                return df

            # drug_df_full = entire uploaded range, filtered — drives the
            # weekly trend chart regardless of which period is selected.
            drug_df_full = _apply_filters(drug_df_full_raw.copy())

            # drug_df = the scoped view every other tab consumes.
            drug_df = drug_df_full if is_overall else _apply_filters(month_buckets[selected_period].copy())

            period_label = (
                f"Overall ({month_labels_sorted[0]} → {month_labels_sorted[-1]})"
                if is_overall else selected_period
            )

            rejected = drug_df[drug_df['IS_REJECTED'] == 1].copy()
            approved = drug_df[drug_df['IS_REJECTED'] == 0].copy()

            st.markdown("---")
            st.markdown('<div class="section-header">Emerging Patterns</div>', unsafe_allow_html=True)
            enable_patterns = st.toggle(
                "🔍 Enable Pattern Detection",
                value=True,
                help="ON: compares your latest month against historical months to "
                     "automatically find new patterns (no fixed rules) — shown in "
                     "Fraud & Safety and the Emerging Patterns tab. "
                     "OFF: skip this, just show plain claims analytics."
            )
            baseline_n = st.slider(
                'Baseline months to compare against', min_value=1, max_value=12, value=6,
                help='Drift/novelty always compares ONE month against its trailing baseline. '
                     'In Overall view this is automatically the most recent month uploaded.',
                disabled=not enable_patterns,
            )
            current_snapshot = history.build_snapshot(month_buckets[emerging_month_label])
            historical_snapshots, baseline_months_used = history.load_baseline(
                emerging_month_label, n_months=baseline_n
            )

            st.markdown("---")
            total = len(drug_df)
            rej_count = len(rejected)
            rej_rate = rej_count / total * 100 if total > 0 else 0
            st.markdown(f"""
            <div style='font-size:11px;color:#8B949E;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:12px'>Quick Stats</div>
            <div style='font-size:13px;color:#C9D1D9;line-height:2'>
                🗓 Period: <b style='color:#E6EDF3'>{period_label}</b><br>
                📊 Total claims: <b style='color:#E6EDF3'>{total:,}</b><br>
                ❌ Rejected: <b style='color:#FF4444'>{rej_count:,}</b><br>
                ✅ Approved: <b style='color:#00C853'>{total-rej_count:,}</b><br>
                📈 Rejection rate: <b style='color:{'#FF4444' if rej_rate>17 else '#FF8C00' if rej_rate>14 else '#00C853'}'>{rej_rate:.1f}%</b>
            </div>
            """, unsafe_allow_html=True)

            with st.sidebar.expander("Export"):
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

                tag = period_label.replace(' ', '_').replace('(', '').replace(')', '').replace('→', 'to')
                st.download_button("Download Summary CSV", data=_build_summary_bytes(),
                                    file_name=f"pbm_summary_{tag}_{filename}", mime="text/csv")
                try:
                    st.download_button("Download Rejected Claims CSV",
                                        data=rejected.to_csv(index=False).encode('utf-8'),
                                        file_name=f"rejected_claims_{tag}_{filename}", mime="text/csv")
                except Exception:
                    st.warning('Unable to prepare rejected claims CSV for download.')

            with st.sidebar.expander("Analysis thresholds"):
                st.markdown(
                    f"- Rejection rate: **>{THRESHOLDS['rejection_rate_critical']}%** critical, "
                    f"**>{THRESHOLDS['rejection_rate_warning']}%** monitor\n"
                    f"- High-risk combo: **≥{THRESHOLDS['combo_high_risk_rate']}%** rejection, "
                    f"min **{THRESHOLDS['combo_min_claims']}** claims\n"
                    f"- Flagged provider: **>{THRESHOLDS['provider_flag_rate']}%** rejection, "
                    f"min **{THRESHOLDS['provider_flag_min_claims']}** claims\n"
                    f"- New drug NCOV alert: min **{THRESHOLDS['new_drug_ncov_min_claims']}** claims"
                )

        except Exception as e:
            st.sidebar.error(f"Error processing file: {str(e)}")
            st.stop()


# ── MAIN CONTENT ───────────────────────────────────────────────
if not file_bytes or not filename or 'drug_df' not in locals():
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

# weekly_trends = compute_weekly_trends(drug_df)
weekly_trends = compute_weekly_trends(drug_df_full)


provider_investigation = compute_provider_investigation(drug_df)
# new_drugs_ncov = compute_new_drugs_always_ncov(drug_df, min_claims=THRESHOLDS['new_drug_ncov_min_claims'])


payment_anomalies = compute_payment_anomalies(drug_df)
payment_anomaly_summary = summarize_payment_anomalies(payment_anomalies)

# Computed once here (not inside a tab) so both Tab 5 (Fraud & Safety —
# condensed "newly discovered" callout) and Tab 6 (Emerging Patterns —
# full ranked investigation queue) can reuse the same scan instead of
# running the statistical engine twice per page load.

if enable_patterns:
    emerging_findings = anomaly.run_emerging_pattern_scan(current_snapshot, historical_snapshots, drug_df=drug_df)
else:
    emerging_findings = pd.DataFrame(columns=anomaly.FINDINGS_COLUMNS[1:])


new_drugs_ncov = compute_new_drugs_always_ncov(
    drug_df, full_df=drug_df_full, min_claims=THRESHOLDS['new_drug_ncov_min_claims']
)

insights = gen_insights(drug_df, rejected, code_stats, high_risk)

total_rej_amt = rejected['TREAT_REJ_AMT'].sum() if 'TREAT_REJ_AMT' in rejected.columns else 0
total_est_amt = drug_df['TREAT_EST_AMT'].sum() if 'TREAT_EST_AMT' in drug_df.columns else 0

# ── TABS ───────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Overview", "💰 Financial", "🔍 Combos", "👨‍⚕️ Providers", "🚨 Fraud & Safety",
    "🧬 Emerging Patterns",
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
            st.plotly_chart(fig,  width="stretch")
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
    st.dataframe(display_df,  width="stretch", hide_index=True)

    # if not weekly_trends.empty:
    #     st.markdown('<div class="section-header">Weekly Trend — Volume & Rejection Rate</div>', unsafe_allow_html=True)
    #     fig = plot_weekly_trends(weekly_trends)
    #     if fig:
    #         st.plotly_chart(fig, use_container_width=True)

    if not weekly_trends.empty:
        st.markdown('<div class="section-header">Weekly Trend — Full Uploaded Range</div>', unsafe_allow_html=True)
        fig = plot_weekly_trends(weekly_trends)
        if fig:
            st.plotly_chart(fig,  width="stretch")


    # MNEC breakdown
    if 'MNEC' in rejected['REJ_CODE_PREFIX'].values and 'REJ_REMARKS' in rejected.columns:
        st.markdown('<div class="section-header">MNEC Breakdown (Auto-decoded)</div>', unsafe_allow_html=True)
        mnec_df = rejected[rejected['REJ_CODE_PREFIX'] == 'MNEC']
        fig = plot_mnec_breakdown(mnec_df)
        if fig:
            st.plotly_chart(fig,  width="stretch")
        else:
            st.info('MNEC data unavailable for breakdown.')

    # Age breakdown (gender moved to Advanced expander — low actionability)
    if 'AGE_GROUP' in drug_df.columns:
        st.markdown('<div class="section-header">Rejection Rate by Age Group</div>', unsafe_allow_html=True)
        fig = plot_age_rejection_rate(drug_df)
        if fig:
            st.plotly_chart(fig,  width="stretch")

    with st.expander("Advanced demographics"):
        if 'MEM_GENDER' in drug_df.columns:
            gen = (
                drug_df.groupby('MEM_GENDER')
                .agg(Total=('IS_REJECTED', 'count'), Rejected=('IS_REJECTED', 'sum'))
                .reset_index()
            )
            gen['Rej Rate %'] = (gen['Rejected'] / gen['Total'] * 100).round(1)
            st.dataframe(gen,  width="stretch", hide_index=True)
        else:
            st.info('Gender column not available.')

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

    st.markdown('<div class="section-header">Rejected Amount by Code</div>', unsafe_allow_html=True)
    fig = plot_rejected_amount_by_code(code_stats)
    if fig:
        st.plotly_chart(fig,  width="stretch")

    st.markdown('<div class="section-header">Top 15 Most Expensive Rejected Drugs</div>', unsafe_allow_html=True)
    fig = plot_top_rejected_drugs(rejected)
    if fig:
        st.plotly_chart(fig,  width="stretch")
    else:
        st.info('No rejected drug data available.')

# ════════════════════════════════════════════════════
# TAB 3 — COMBOS
# ════════════════════════════════════════════════════
with tab3:
    st.markdown(
        f'<div class="section-header">High Risk Drug-Diagnosis Combinations '
        f'(>{THRESHOLDS["combo_high_risk_rate"]}% rejection, min {THRESHOLDS["combo_min_claims"]} claims)</div>',
        unsafe_allow_html=True,
    )
    if len(high_risk) > 0:
        fig = plot_high_risk_combos(high_risk)
        if fig:
            st.plotly_chart(fig,  width="stretch")
        st.dataframe(
            enrich_combo_display(high_risk, drug_df).head(30),
             width="stretch",
            hide_index=True,
        )
    else:
        st.info(
            f'No high-risk combos found this month (min {THRESHOLDS["combo_min_claims"]} claims, '
            f'>{THRESHOLDS["combo_high_risk_rate"]}% rejection).'
        )

    with st.expander("Gray area combos (30–70% rejection)"):
        if len(gray_combos) > 0:
            st.dataframe(
                enrich_combo_display(gray_combos, drug_df).head(20),
                width="stretch",
                hide_index=True,
            )
        else:
            st.info('No gray area combos found.')

    with st.expander("Safe combos (0% rejection — reference only)"):
        if len(safe_combos) > 0:
            st.dataframe(
                enrich_combo_display(safe_combos, drug_df).head(20),
                width="stretch",
                hide_index=True,
            )
        else:
            st.info('No safe combos found.')

# ════════════════════════════════════════════════════
# TAB 4 — PROVIDERS
# ════════════════════════════════════════════════════
with tab4:
    if 'DOC_LIC_NO' in drug_df.columns:
        prov = compute_provider_stats(drug_df)
        flagged_provs = prov[
            (prov['RejRate'] > THRESHOLDS['provider_flag_rate'])
            & (prov['Claims'] >= THRESHOLDS['provider_flag_min_claims'])
        ]

        p1, p2, p3 = st.columns(3)
        with p1:
            st.markdown(f"""<div class="metric-card blue">
                <div class="metric-label">Unique Providers</div>
                <div class="metric-value">{len(prov):,}</div>
                <div class="metric-sub">Active this month</div>
            </div>""", unsafe_allow_html=True)
        with p2:
            st.markdown(f"""<div class="metric-card red">
                <div class="metric-label">Flagged Providers</div>
                <div class="metric-value">{len(flagged_provs)}</div>
                <div class="metric-sub">&gt;{THRESHOLDS['provider_flag_rate']}% rejection, min {THRESHOLDS['provider_flag_min_claims']} claims</div>
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
                st.plotly_chart(fig,  width="stretch")

        with c2:
            st.markdown(
                f'<div class="section-header">Flagged Providers (&gt;{THRESHOLDS["provider_flag_rate"]}% rejection)</div>',
                unsafe_allow_html=True,
            )
            if len(flagged_provs) > 0:
                flagged_show = flagged_provs.sort_values('RejRate', ascending=False)[
                    ['DOC_LIC_NO', 'Claims', 'RejRate', 'RejAmt']
                ]
                flagged_show.columns = ['Provider', 'Claims', 'Rej Rate %', 'Rej Amount']
                st.dataframe(flagged_show,  width="stretch", hide_index=True)
            else:
                st.success('No providers flagged this month.')

        st.markdown('<div class="section-header">Provider Risk Map (Claims vs Rejection Rate)</div>', unsafe_allow_html=True)
        fig = plot_provider_risk_map(prov)
        if fig:
            st.plotly_chart(fig,  width="stretch")

        st.markdown(
            '<div class="section-header">Investigation Queue (prioritized)</div>',
            unsafe_allow_html=True,
        )
        if not provider_investigation.empty:
            queue_cols = [
                c for c in [
                    'DOC_LIC_NO', 'DOC_NAME', 'InvestigationReason',
                    'Total_Claims', 'RejRate_%', 'MNEC_Count', 'NCOV_Count', 'RejAmt',
                ] if c in provider_investigation.columns
            ]
            st.dataframe(
                provider_investigation[queue_cols].head(20),
                 width="stretch",
                hide_index=True,
            )
        else:
            st.info('Not enough provider data for investigation queue.')

        st.markdown('<div class="section-header">Provider Drill-Down</div>', unsafe_allow_html=True)
        provider_options = sorted(drug_df['DOC_LIC_NO'].dropna().unique().tolist())
        default_provider = (
            provider_investigation.iloc[0]['DOC_LIC_NO']
            if not provider_investigation.empty
            else provider_options[0]
        )
        selected_provider = st.selectbox(
            'Select provider',
            provider_options,
            index=provider_options.index(default_provider) if default_provider in provider_options else 0,
        )
        detail = compute_provider_detail(drug_df, selected_provider)
        if detail:
            title = detail['doc_name'] or selected_provider
            st.markdown(
                f"**{title}** (`{selected_provider}`) — "
                f"{detail['total_claims']:,} claims | "
                f"{detail['rej_rate']}% rejection | "
                f"{detail['rej_amt']:,.0f} rejected amount"
            )
            d1, d2, d3 = st.columns(3)
            with d1:
                st.markdown("**Top rejection codes**")
                st.dataframe(detail['top_codes'],  width="stretch", hide_index=True)
            with d2:
                st.markdown("**Top rejected drugs**")
                st.dataframe(detail['top_drugs'],  width="stretch", hide_index=True)
            with d3:
                st.markdown("**Top rejected combos**")
                st.dataframe(detail['top_combos'],  width="stretch", hide_index=True)
    else:
        st.warning('DOC_LIC_NO column not found in dataset.')


# ════════════════════════════════════════════════════
# TAB 5 — FRAUD & SAFETY (DYNAMIC VERSION)
# ════════════════════════════════════════════════════
with tab5:
    st.markdown(
        '<div class="section-header">🚨 Fraud & Safety — Dynamic Pattern Detection</div>',
        unsafe_allow_html=True
    )
    st.caption(
        'Automatically detects anomalies and unusual patterns using statistical methods, '
        'not hardcoded rules. Catches unseen fraud types and safety issues.'
    )

    
    fraud_findings = anomaly.run_emerging_pattern_scan(
        current_snapshot, historical_snapshots, drug_df=drug_df
    )
    
    # Filter for fraud-relevant dimensions
    fraud_relevant_dimensions = [
        'Gender × Diagnosis',  # Maternity for males, etc.
        'Age × Drug',          # Pediatric drug for elderly, etc.
        'New Drug-Diagnosis Combo',  # Brand new prescription patterns
        'Provider Behavior',    # Provider rejection rate changes
        'Drug Utilization',     # Drug volume spikes
        'Diagnosis Drift',      # New diagnosis spikes
        'Rejection Code Drift', # Rejection reason pattern changes
    ]
    
    fraud_critical = fraud_findings[
        (fraud_findings['Dimension'].isin(fraud_relevant_dimensions)) &
        (fraud_findings['Severity'].isin(['critical', 'warning']))
    ].copy()

    # ── KPI SUMMARY ──────────────────────────────────────────────────
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    
    with kpi1:
        st.markdown(f"""<div class="metric-card red">
            <div class="metric-label">Critical Patterns</div>
            <div class="metric-value">{len(fraud_critical[fraud_critical['Severity']=='critical']):,}</div>
            <div class="metric-sub">Investigate immediately</div>
        </div>""", unsafe_allow_html=True)
    
    with kpi2:
        st.markdown(f"""<div class="metric-card amber">
            <div class="metric-label">Warning Patterns</div>
            <div class="metric-value">{len(fraud_critical[fraud_critical['Severity']=='warning']):,}</div>
            <div class="metric-sub">Monitor closely</div>
        </div>""", unsafe_allow_html=True)
    
    with kpi3:
        novel_count = fraud_critical['Novel'].sum()
        st.markdown(f"""<div class="metric-card purple">
            <div class="metric-label">Never-Seen-Before</div>
            <div class="metric-value">{novel_count:,}</div>
            <div class="metric-sub">No historical precedent</div>
        </div>""", unsafe_allow_html=True)
    
    with kpi4:
        baseline_msg = f"{len(baseline_months_used)} months" if len(baseline_months_used) > 0 else "None yet"
        st.markdown(f"""<div class="metric-card blue">
            <div class="metric-label">Baseline Months</div>
            <div class="metric-value">{len(baseline_months_used)}</div>
            <div class="metric-sub">{baseline_msg} for comparison</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── BASELINE STATUS ──────────────────────────────────────────────
    if len(baseline_months_used) == 0:
        st.info(
            f"📅 **{current_month_label}** is the first month on record. "
            "Month-over-month drift detection will activate once you upload another month. "
            "For now, same-month outlier detection is active."
        )
    elif len(baseline_months_used) < anomaly.MIN_BASELINE_MONTHS:
        st.warning(
            f"📅 Baseline has **{len(baseline_months_used)} month(s)**. "
            f"Need {anomaly.MIN_BASELINE_MONTHS}+ months for full z-score accuracy. "
            f"Upload one more month to unlock complete drift detection."
        )
    else:
        st.success(
            f"📅 Full statistical baseline active: {', '.join(baseline_months_used)}"
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── DYNAMIC FINDINGS ─────────────────────────────────────────────
    if not fraud_critical.empty:
        
        # Section 1: Critical Patterns (Prioritized Queue)
        st.markdown(
            '<div class="section-header">🎯 Critical & Warning Findings (Ranked by Severity)</div>',
            unsafe_allow_html=True
        )
        
        display_cols = [
            'Rank', 'Dimension', 'Entity', 'Metric', 'Current', 'Baseline_Mean',
            'Pct_Change', 'ZScore', 'Anomaly_Score', 'Severity', 'Reason'
        ]
        
        st.dataframe(
            fraud_critical[display_cols].head(25),
            # use_container_width=True,
            width = "stretch",
            hide_index=True
        )
        
        st.markdown("<br>", unsafe_allow_html=True)

        # Section 2: Specific Pattern Detection
        col_a, col_b = st.columns(2)
        
        # ── GENDER × DIAGNOSIS ANOMALIES (Maternity for males, etc.) ──
        with col_a:
            st.markdown(
                '<div class="section-header">👥 Gender × Diagnosis Anomalies</div>',
                unsafe_allow_html=True
            )
            st.caption('Unexpected gender-diagnosis combinations (e.g., maternity for males)')
            
            gender_diag = fraud_critical[
                fraud_critical['Dimension'] == 'Gender × Diagnosis'
            ].copy()
            
            if len(gender_diag) > 0:
                for idx, row in gender_diag.head(10).iterrows():
                    severity_icon = "🔴" if row['Severity'] == 'critical' else "🟡"
                    st.markdown(f"""
                    {severity_icon} **{row['Entity']}**
                    - Metric: {row['Metric']}
                    - Score: {row['Anomaly_Score']:.0f}/100
                    - {row['Reason'][:100]}...
                    """)
            else:
                st.info("No gender-diagnosis anomalies detected.")
        
        # ── AGE × DRUG ANOMALIES (Pediatric for elderly, etc.) ────────
        with col_b:
            st.markdown(
                '<div class="section-header">💊 Age × Drug Anomalies</div>',
                unsafe_allow_html=True
            )
            st.caption('Age-inappropriate drug prescriptions (e.g., pediatric drug for elderly)')
            
            age_drug = fraud_critical[
                fraud_critical['Dimension'] == 'Age × Drug'
            ].copy()
            
            if len(age_drug) > 0:
                for idx, row in age_drug.head(10).iterrows():
                    severity_icon = "🔴" if row['Severity'] == 'critical' else "🟡"
                    st.markdown(f"""
                    {severity_icon} **{row['Entity']}**
                    - Metric: {row['Metric']}
                    - Score: {row['Anomaly_Score']:.0f}/100
                    - {row['Reason'][:100]}...
                    """)
            else:
                st.info("No age-drug anomalies detected.")

        st.markdown("<br>", unsafe_allow_html=True)

        # Section 3: New Patterns & Volume Spikes
        col_c, col_d = st.columns(2)
        
        # ── NEW DIAGNOSIS SPIKES (COVID-style) ────────────────────────
        with col_c:
            st.markdown(
                '<div class="section-header">📊 New Diagnosis Spikes</div>',
                unsafe_allow_html=True
            )
            st.caption('Brand-new diagnoses appearing (like COVID-19 in 2020)')
            
            new_diag = fraud_critical[
                fraud_critical['Dimension'] == 'Diagnosis Drift'
            ].copy()
            
            if len(new_diag) > 0:
                for idx, row in new_diag.head(10).iterrows():
                    severity_icon = "🔴" if row['Severity'] == 'critical' else "🟡"
                    pct_change = f"{row['Pct_Change']}%" if pd.notna(row['Pct_Change']) else "N/A"
                    st.markdown(f"""
                    {severity_icon} **{row['Entity']}**
                    - Change: {pct_change}%
                    - Score: {row['Anomaly_Score']:.0f}/100
                    - {row['Reason'][:100]}...
                    """)
            else:
                st.info("No diagnosis spikes detected.")
        
        # ── PROVIDER BEHAVIOR CHANGES ──────────────────────────────────
        with col_d:
            st.markdown(
                '<div class="section-header">🏥 Provider Behavior Changes</div>',
                unsafe_allow_html=True
            )
            st.caption('Provider rejection rates or volumes changing dramatically')
            
            prov_behavior = fraud_critical[
                fraud_critical['Dimension'] == 'Provider Behavior'
            ].copy()
            
            if len(prov_behavior) > 0:
                for idx, row in prov_behavior.head(10).iterrows():
                    severity_icon = "🔴" if row['Severity'] == 'critical' else "🟡"
                    pct_change_display = f"{row['Pct_Change']:+.0f}%" if pd.notna(row['Pct_Change']) else "N/A"


                    st.markdown(f"""
                    {severity_icon} **{row['Entity']}**
                    - Metric: {row['Metric']}
                    - Change: {pct_change_display}
                    - Score: {row['Anomaly_Score']:.0f}/100
                    - {row['Reason'][:100]}...
                    """)
            else:
                st.info("No provider behavior changes detected.")

        st.markdown("<br>", unsafe_allow_html=True)

        # Section 4: Drill-Down Investigation
        st.markdown(
            '<div class="section-header">🔍 Drill-Down Investigation</div>',
            unsafe_allow_html=True
        )
        
        findings_list = (fraud_critical['Entity'].astype(str) + ' — ' + 
                        fraud_critical['Dimension'].astype(str)).tolist()
        
        if findings_list:
            selected_finding = st.selectbox(
                'Select a finding to investigate:',
                findings_list,
                key='fraud_finding_selector'
            )
            
            if selected_finding:
                finding_idx = findings_list.index(selected_finding)
                finding_row = fraud_critical.iloc[finding_idx]
                
                st.markdown(f"### {finding_row['Entity']} — {finding_row['Dimension']}")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Anomaly Score",
                        f"{finding_row['Anomaly_Score']:.0f}/100",
                        delta=f"{finding_row['Pct_Change']:+.0f}%" if pd.notna(finding_row['Pct_Change']) else None
                    )
                
                with col2:
                    st.metric(
                        "Severity",
                        finding_row['Severity'].upper(),
                        delta=f"Z-Score: {finding_row['ZScore']:.1f}" if pd.notna(finding_row['ZScore']) else None
                    )
                
                with col3:
                    st.metric(
                        "Baseline Months",
                        finding_row['Baseline_Months'],
                        delta="Novel" if finding_row['Novel'] else "Historical"
                    )
                
                st.markdown(f"""
                **Investigation Details:**
                
                {finding_row['Reason']}
                
                **Metrics:**
                - Current Value: {finding_row['Current']:.2f}
                - Baseline Average: {finding_row['Baseline_Mean']:.2f}
                - Percent Change: {finding_row['Pct_Change']:.1f}% (if available)
                - Z-Score: {finding_row['ZScore']:.2f} (if available)
                """)

    else:
        st.success('✅ No critical or warning patterns detected this month.')

    st.markdown("<br>", unsafe_allow_html=True)

    # ── PAYMENT INTEGRITY (Keep existing section) ──────────────────────
    st.markdown('<div class="section-header">💵 Payment Integrity Anomalies</div>', unsafe_allow_html=True)
    st.caption('Financial data-integrity issues (rejected-but-paid, overpayments, etc.)')

    pa1, pa2, pa3 = st.columns(3)
    with pa1:
        st.markdown(f"""<div class="metric-card red">
            <div class="metric-label">Rejected But Paid</div>
            <div class="metric-value">{payment_anomaly_summary['rejected_paid_count']:,}</div>
            <div class="metric-sub">{payment_anomaly_summary['rejected_paid_amt']:,.0f} paid on rejected claims</div>
        </div>""", unsafe_allow_html=True)
    with pa2:
        st.markdown(f"""<div class="metric-card amber">
            <div class="metric-label">Overpaid (Real Request)</div>
            <div class="metric-value">{payment_anomaly_summary['genuine_overpayment_count']:,}</div>
            <div class="metric-sub">{payment_anomaly_summary['genuine_overpayment_amt']:,.0f} excess paid</div>
        </div>""", unsafe_allow_html=True)
    with pa3:
        st.markdown(f"""<div class="metric-card blue">
            <div class="metric-label">Paid With $0 Requested</div>
            <div class="metric-value">{payment_anomaly_summary['zero_requested_count']:,}</div>
            <div class="metric-sub">{payment_anomaly_summary['zero_requested_amt']:,.0f} — likely missing source value</div>
        </div>""", unsafe_allow_html=True)

    if any(payment_anomaly_summary[k] for k in ('rejected_paid_count', 'genuine_overpayment_count', 'zero_requested_count')):
        if payment_anomaly_summary['rejected_paid_count'] > 0:
            with st.expander(f"🔴 Rejected-but-paid claims ({payment_anomaly_summary['rejected_paid_count']:,}) — should be $0 paid"):
                rp = payment_anomalies['rejected_paid']
                cols = [c for c in ['DRUG_CODE', 'DRUG_NAME', 'DOC_LIC_NO', 'REJ_CODE_PREFIX',
                                    'TREAT_EST_AMT', 'TREAT_APPR_AMT', 'TREAT_REJ_AMT', 'SERVICE_DT']
                        if c in rp.columns]
                st.dataframe(rp[cols].sort_values('TREAT_APPR_AMT', ascending=False), width = "stretch",
                               hide_index=True)

        if payment_anomaly_summary['genuine_overpayment_count'] > 0:
            with st.expander(f"🟠 Genuine overpayments — paid more than requested ({payment_anomaly_summary['genuine_overpayment_count']:,})"):
                go = payment_anomalies['genuine_overpayment']
                cols = [c for c in ['DRUG_CODE', 'DRUG_NAME', 'DOC_LIC_NO',
                                    'TREAT_EST_AMT', 'TREAT_APPR_AMT', 'Excess_Amt', 'SERVICE_DT']
                        if c in go.columns]
                st.dataframe(go[cols].sort_values('Excess_Amt', ascending=False),
                            width = "stretch" ,hide_index=True)

        if payment_anomaly_summary['zero_requested_count'] > 0:
            with st.expander(f"🔵 Paid with $0 requested — likely missing source value ({payment_anomaly_summary['zero_requested_count']:,})"):
                zr = payment_anomalies['zero_requested']
                cols = [c for c in ['DRUG_CODE', 'DRUG_NAME', 'DOC_LIC_NO',
                                    'TREAT_EST_AMT', 'TREAT_APPR_AMT', 'SERVICE_DT']
                        if c in zr.columns]
                st.dataframe(zr[cols].sort_values('TREAT_APPR_AMT', ascending=False),
                            width = "stretch", hide_index=True)
    else:
        st.success('No payment integrity anomalies found this month.')

    st.markdown("<br>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════
# TAB 6 — EMERGING PATTERNS (statistical discovery, no hardcoded rules)
# ════════════════════════════════════════════════════
with tab6:
    if not enable_patterns:
        st.info("🔌 Pattern Detection is turned off. Enable it in the sidebar (Emerging Patterns section) to use this tab.")

    if is_overall:
        st.info(
            f"📊 You're viewing **Overall**. Drift/novelty detection always compares one "
            f"specific month against its trailing baseline, so this tab is analyzing the "
            f"most recent month in your upload: **{emerging_month_label}**. Pick that month "
            f"directly in the Period selector if you want to drill into a different one."
        )
    st.markdown('<div class="section-header">Baseline Status</div>', unsafe_allow_html=True)
    n_baseline = len(baseline_months_used)

    if n_baseline == 0:
        st.info(
            f"📅 **{emerging_month_label}** is the first month on record — there's no baseline yet. "
            "Month-over-month drift (z-scores, % change, new-combination novelty) will activate "
            "automatically once at least one more month has been uploaded or seeded. "
            "What you see below is same-month outlier detection: providers, drugs, and "
            "diagnoses that already look unusual *relative to their peers this month*."
        )
    elif n_baseline < anomaly.MIN_BASELINE_MONTHS:
        st.warning(
            f"📅 Baseline currently has **{n_baseline} month** ({', '.join(baseline_months_used)}). "
            f"Z-score drift needs {anomaly.MIN_BASELINE_MONTHS}+ months to compute a meaningful "
            "standard deviation, so percent-change and novelty findings are active, but z-score-based "
            "findings are limited. Upload or seed one more historical month to unlock full drift detection."
        )
    else:
        st.success(
            f"📅 Comparing **{emerging_month_label}** against a **{n_baseline}-month** baseline: "
            f"{', '.join(baseline_months_used)}."
        )

    emerging_findings = anomaly.run_emerging_pattern_scan(
        current_snapshot, historical_snapshots, drug_df=drug_df
    )
    summary = anomaly.summarize_scan(emerging_findings)

    e1, e2, e3, e4 = st.columns(4)
    with e1:
        st.markdown(f"""<div class="metric-card blue">
            <div class="metric-label">Total Findings</div>
            <div class="metric-value">{summary['total']:,}</div>
            <div class="metric-sub">across all dimensions</div>
        </div>""", unsafe_allow_html=True)
    with e2:
        st.markdown(f"""<div class="metric-card red">
            <div class="metric-label">Critical</div>
            <div class="metric-value">{summary['critical']:,}</div>
            <div class="metric-sub">investigate first</div>
        </div>""", unsafe_allow_html=True)
    with e3:
        st.markdown(f"""<div class="metric-card amber">
            <div class="metric-label">Warning</div>
            <div class="metric-value">{summary['warning']:,}</div>
            <div class="metric-sub">monitor closely</div>
        </div>""", unsafe_allow_html=True)
    with e4:
        st.markdown(f"""<div class="metric-card purple">
            <div class="metric-label">Never-Seen-Before</div>
            <div class="metric-value">{summary['novel']:,}</div>
            <div class="metric-sub">no historical precedent</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if emerging_findings.empty:
        st.success("No statistically significant emerging patterns this month.")
    else:
        c1, c2 = st.columns([1.3, 1])
        with c1:
            st.markdown('<div class="section-header">All Findings (ranked by severity)</div>', unsafe_allow_html=True)
            fig = plot_anomaly_scatter(emerging_findings)
            if fig:
                st.plotly_chart(fig, width="stretch")
        with c2:
            st.markdown('<div class="section-header">Findings by Dimension</div>', unsafe_allow_html=True)
            fig = plot_findings_by_dimension(emerging_findings)
            if fig:
                st.plotly_chart(fig, width="stretch")

        st.markdown('<div class="section-header">Investigation Queue</div>', unsafe_allow_html=True)
        fcol1, fcol2 = st.columns(2)
        with fcol1:
            dim_filter = st.multiselect(
                'Filter by dimension', sorted(emerging_findings['Dimension'].unique()),
                default=[],
            )
        with fcol2:
            sev_filter = st.multiselect(
                'Filter by severity', ['critical', 'warning', 'info'], default=[],
            )

        show = emerging_findings.copy()
        if dim_filter:
            show = show[show['Dimension'].isin(dim_filter)]
        if sev_filter:
            show = show[show['Severity'].isin(sev_filter)]

        display_cols = ['Rank', 'Severity', 'Dimension', 'Entity', 'Metric', 'Current',
                         'Baseline_Mean', 'Pct_Change', 'ZScore', 'Novel', 'Anomaly_Score', 'Reason']
        # st.dataframe(
        #     show[display_cols].head(100),
        #     width="stretch", hide_index=True,
        # )

        available_cols = [c for c in display_cols if c in show.columns]

        st.dataframe(
            show[available_cols].head(100),width = "stretch", 
            hide_index=True,
        )


        st.caption(f"Showing {min(len(show), 100):,} of {len(show):,} matching findings.")

        st.download_button(
            "Download Full Findings CSV",
            data=emerging_findings.to_csv(index=False).encode('utf-8'),
            file_name=f"emerging_patterns_{emerging_month_label}.csv",
            mime="text/csv",
        )

        st.markdown('<div class="section-header">Drill-Down</div>', unsafe_allow_html=True)
        finding_options = show.head(50)['Entity'] + ' — ' + show.head(50)['Dimension']
        if len(finding_options):
            chosen = st.selectbox('Inspect a finding', finding_options.tolist())
            chosen_idx = finding_options.tolist().index(chosen)
            chosen_row = show.head(50).iloc[chosen_idx]
            st.markdown(f"**{chosen_row['Entity']}** — {chosen_row['Dimension']} / {chosen_row['Metric']}")
            st.markdown(
                f"<div class='insight-card {chosen_row['Severity']}'>{chosen_row['Reason']}</div>",
                unsafe_allow_html=True,
            )

    with st.expander("How Emerging Patterns works"):
        st.markdown(
            """
Every finding compares a metric for **this month** against a statistical baseline,
using one of four generic methods — never a hand-written fraud rule:

- **Z-score** — how many standard deviations this month's value is from that
  entity's own trailing average (provider rejection rate, drug volume, etc.)
- **Percent change** — straightforward magnitude of the move
- **Frequency deviation / novelty** — combinations (gender × diagnosis, age × drug,
  drug × diagnosis) that have **zero** occurrences in every baseline month
- **Outlier detection** — for brand-new entities with no history yet, this month's
  volume is compared against all its current peers instead

Findings are blended into a single 0–100 **Anomaly Score** and ranked. The same
math runs identically across every dimension — providers, drugs, diagnoses,
rejection codes, gender, age — so a new pattern type doesn't require a new rule,
only more historical months to compare against.
            """
        )

