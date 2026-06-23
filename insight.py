import pandas as pd
import numpy as np
from utils import CODE_DESC, THRESHOLDS


def summarize_finding(row):
    """
    Convert one technical anomaly-finding row (from anomaly.run_emerging_pattern_scan)
    into a single plain-English sentence — no sigma, no z-score, no "absolute delta"
    jargon. Built for a stakeholder scanning Fraud & Safety who doesn't do statistics,
    not for an analyst who already wants the raw Reason column in the ranked table.
    """
    entity = row.get('Entity', 'Unknown')
    metric = str(row.get('Metric', '')).lower()
    current = row.get('Current')
    baseline = row.get('Baseline_Mean')
    pct = row.get('Pct_Change')
    novel = bool(row.get('Novel', False))

    if novel and (baseline is None or pd.isna(baseline)):
        current_txt = f"{current:,.0f}" if pd.notna(current) else "some"
        return f"{entity} — never happened before. {current_txt} claim(s) this month."

    if pd.notna(pct) and pd.notna(baseline) and pd.notna(current):
        direction = "up" if pct >= 0 else "down"
        return (
            f"{entity} — {metric} went {direction} from {baseline:,.1f} to "
            f"{current:,.1f} ({abs(pct):,.0f}% {direction})."
        )

    if pd.notna(current):
        return f"{entity} — {metric} is {current:,.1f}, unusual compared to its own history."

    return f"{entity} — flagged as unusual this month."



def gen_insights(drug_df, rejected, code_stats, high_risk,
                  new_drugs_ncov=None, payment_anomaly_summary=None, emerging_findings=None):
    insights = []
    total = len(drug_df)
    rej_count = len(rejected)
    rej_rate = rej_count / total * 100

    crit = THRESHOLDS['rejection_rate_critical']
    warn = THRESHOLDS['rejection_rate_warning']

    if rej_rate > crit:
        insights.append(('critical', f'🔴 HIGH rejection rate {rej_rate:.1f}% — above {crit}% threshold.'))
    elif rej_rate > warn:
        insights.append(('warning', f'🟡 MODERATE rejection rate {rej_rate:.1f}% — monitor trend.'))
    else:
        insights.append(('success', f'🟢 NORMAL rejection rate {rej_rate:.1f}% — within expected range.'))


    if code_stats.empty:
        insights.append(("warning", "No rejection codes found."))
        return insights

    top_code = code_stats.iloc[0]['REJ_CODE_PREFIX']
    top_count = int(code_stats.iloc[0]['Count'])
    top_amt = code_stats.iloc[0]['Rejected_Amt']
    insights.append(('info', f'💰 {top_code} ({CODE_DESC.get(top_code,"")}) is #1 rejection: {top_count:,} claims | {top_amt:,.0f} rejected amount.'))

    if 'MNEC' in rejected['REJ_CODE_PREFIX'].values and 'REJ_REMARKS' in rejected.columns:
        mnec_df = rejected[rejected['REJ_CODE_PREFIX'] == 'MNEC']
        refill_n = mnec_df['REJ_REMARKS'].str.contains('refill', case=False, na=False).sum()
        refill_p = refill_n / len(mnec_df) * 100 if len(mnec_df) > 0 else 0
        if refill_p > 50:
            insights.append(('critical', f'⚠️ MNEC MISCODING: {refill_p:.1f}% of MNEC is Refill Too Soon ({refill_n:,} claims). System coding bug.'))

    if 'REJ_REMARKS' in rejected.columns:
        rf = rejected['REJ_REMARKS'].str.contains('refill', case=False, na=False).sum()
        rf_amt = rejected[rejected['REJ_REMARKS'].str.contains('refill', case=False, na=False)]['TREAT_REJ_AMT'].sum()
        insights.append(('warning', f'🔄 Refill Too Soon: {rf:,} rejections | {rf_amt:,.0f} amount → Alert doctors of next allowed refill date.'))

    ncov_n = len(rejected[rejected['REJ_CODE_PREFIX'] == 'NCOV'])
    ncov_a = rejected[rejected['REJ_CODE_PREFIX'] == 'NCOV']['TREAT_REJ_AMT'].sum()
    if ncov_n > 0:
        insights.append(('warning', f'📋 Not Covered (NCOV): {ncov_n:,} claims | {ncov_a:,.0f} → Show covered drug list at prescription time.'))

    elig_n = len(rejected[rejected['REJ_CODE_PREFIX'] == 'ELIG'])
    elig_p = elig_n / rej_count * 100
    if elig_p > 5:
        insights.append(('warning', f'👤 ELIG spike: {elig_n:,} eligibility rejections ({elig_p:.1f}%) → Check for expired memberships.'))

    if 'AGE_GROUP' in drug_df.columns:
        child_v = rejected[(rejected['REJ_CODE_PREFIX'] == 'CODE') &
                           (rejected['AGE_GROUP'].isin(['INFANT (0-2)', 'CHILD (3-12)', 'TEEN (13-17)']))]
        if len(child_v) > 0:
            insights.append(('critical', f'🚨 PATIENT SAFETY: {len(child_v):,} age rule violations for children → Adult drugs to minors. Urgent review.'))

    if len(high_risk) > 0:
        insights.append(('info', f'🚫 {len(high_risk)} drug-diagnosis combos with >70% rejection → Add to hard exclusion rules.'))

   

    if ('PA_FLAG_REASON' in drug_df.columns and drug_df['PA_FLAG_REASON'].notna().any()):
        fraud_n = drug_df[drug_df['PA_FLAG_REASON'].str.contains('fraud', case=False, na=False)]
        if len(fraud_n) > 0:
            insights.append(('critical', f'🔍 FRAUD FLAG: {len(fraud_n):,} claims flagged ({fraud_n["IS_REJECTED"].mean()*100:.1f}% rejection) → Review with compliance.'))

    # ── NCOV drugs that WERE approved before (formulary drop / coding bug) ──
    if new_drugs_ncov is not None and not new_drugs_ncov.empty and 'Ever_Approved' in new_drugs_ncov.columns:
        was_approved = new_drugs_ncov[new_drugs_ncov['Ever_Approved']]
        if len(was_approved) > 0:
            insights.append(('critical',
                f'⚠️ {len(was_approved)} drug(s) now 100% NCOV-rejected WERE approved before for the '
                f'same diagnosis → likely a formulary drop or coding bug, not a genuinely new drug.'))

    # ── Payment integrity anomalies ──────────────────────────────────────
    if payment_anomaly_summary is not None:
        rp_n = payment_anomaly_summary.get('rejected_paid_count', 0)
        if rp_n > 0:
            rp_amt = payment_anomaly_summary.get('rejected_paid_amt', 0)
            insights.append(('critical',
                f'💰 {rp_n:,} rejected claims still show a paid amount ({rp_amt:,.0f} total) → '
                f'audit for payment leakage.'))

        go_n = payment_anomaly_summary.get('genuine_overpayment_count', 0)
        if go_n > 0:
            go_amt = payment_anomaly_summary.get('genuine_overpayment_amt', 0)
            insights.append(('warning',
                f'💰 {go_n:,} claims paid MORE than requested ({go_amt:,.0f} excess) → review for overpayment.'))

    # ── Dynamically discovered patterns (no hardcoded rule) ──────────────
    if emerging_findings is not None and not emerging_findings.empty:
        crit_n = int((emerging_findings['Severity'] == 'critical').sum())
        if crit_n > 0:
            insights.append(('critical',
                f'🆕 {crit_n} critical statistically-discovered pattern(s) this month (new combos, '
                f'demographic mismatches, volume spikes) → see Fraud & Safety / Emerging Patterns.'))

    return insights