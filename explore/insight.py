import pandas as pd
import numpy as np
from config.utils import CODE_DESC, THRESHOLDS


def summarize_finding(row):
# def summarize_emerging_pattern(row):
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


  # Rule registry - defines the execution order

def gen_insights(drug_df, rejected, code_stats, high_risk,
                  new_drugs_ncov=None, payment_anomaly_summary=None, emerging_findings=None):
    

    RULES = [
    rejection_rate_rule,
    top_rejection_rule,
    mnec_rule,
    refill_too_soon_rule,
    ncov_rule,
    eligibility_rule,
    age_rule,
    high_risk_rule,
    fraud_rule,
    new_drugs_ncov_rule,
    payment_anomaly_rule,
    emerging_patterns_rule,
    documentation_rejection_rule,
    medical_necessity_rule,
    service_type_rule,
    quantity_impact_rule,
]

    """
    Generate insights from pharmacy claims data.
    
    Each rule in RULES is called with a context dictionary containing all the
    data it needs. Rules that don't need certain parameters can ignore them.
    """
    context = {
        'drug_df': drug_df,
        'rejected': rejected,
        'code_stats': code_stats,
        'high_risk': high_risk,
        'new_drugs_ncov': new_drugs_ncov,
        'payment_anomaly_summary': payment_anomaly_summary,
        'emerging_findings': emerging_findings,
    }
    
    insights = []
    
    for rule in RULES:
        insights.extend(rule(context))
    
    return insights



def documentation_rejection_rule(ctx):
    """Flag documentation-related rejection hotspots."""
    from explore.stats import compute_rejection_reason_hotspots, REJECTION_REASON_KEYWORDS
    drug_df = ctx['drug_df']

    matched, summary = compute_rejection_reason_hotspots(drug_df, REJECTION_REASON_KEYWORDS['Documentation'])
    if matched.empty:
        return []

    top_n = len(summary)
    return [('warning',
            f'📄 {len(matched):,} rejection(s) cite documentation issues, across {top_n} '
            f'diagnosis/drug/provider combination(s) → see Fraud & Safety for the breakdown.')]


def medical_necessity_rule(ctx):
    """Flag treatment codes repeatedly rejected as medically unnecessary."""
    from explore.stats import compute_rejection_reason_hotspots, REJECTION_REASON_KEYWORDS
    drug_df = ctx['drug_df']

    matched, summary = compute_rejection_reason_hotspots(drug_df, REJECTION_REASON_KEYWORDS['Medical Necessity'])
    if matched.empty:
        return []

    repeat_drugs = summary.groupby('DRUG_CODE')['Frequency'].sum() if 'DRUG_CODE' in summary.columns else None
    repeat_n = int((repeat_drugs > 1).sum()) if repeat_drugs is not None else 0

    msg = f'⚕️ {len(matched):,} rejection(s) cite medical necessity'
    if repeat_n > 0:
        msg += f', including {repeat_n} drug(s) with repeated denials'
    msg += ' → see Fraud & Safety for the breakdown.'
    return [('warning', msg)]


def service_type_rule(ctx):
    """Compare outpatient vs inpatient rejection rates; flag a meaningful gap."""
    from explore.patient import compute_service_type_stats
    drug_df = ctx['drug_df']

    stats = compute_service_type_stats(drug_df)
    if stats.empty or len(stats) < 2:
        return []

    stats_sorted = stats.sort_values('RejRate_%', ascending=False)
    highest, lowest = stats_sorted.iloc[0], stats_sorted.iloc[-1]
    gap = highest['RejRate_%'] - lowest['RejRate_%']

    if gap >= 10 and lowest['Claims'] >= 50:
        return [('info',
                f"🏨 {highest['Service_Type']} claims reject at {highest['RejRate_%']:.1f}% vs "
                f"{lowest['Service_Type']} at {lowest['RejRate_%']:.1f}% ({gap:.0f} point gap) → "
                f"see Overview for the trend.")]
    return []


def quantity_impact_rule(ctx):
    """Flag drugs where higher prescribed quantity meaningfully raises rejection odds."""
    from explore.patient import compute_quantity_rejection_by_drug
    drug_df = ctx['drug_df']

    qty_by_drug = compute_quantity_rejection_by_drug(drug_df, min_claims_per_band=5, min_delta=15.0)
    if qty_by_drug.empty:
        return []

    top = qty_by_drug.iloc[0]
    return [('warning',
            f"📦 {len(qty_by_drug)} drug(s) reject more often at higher quantities — worst: "
            f"{top['DRUG_CODE']} ({top['Lowest_Band_RejRate_%']:.0f}% at qty {top['Lowest_Band']} → "
            f"{top['Highest_Band_RejRate_%']:.0f}% at qty {top['Highest_Band']}) → see Overview for details.")]



# Updated rules to accept context dictionary
def rejection_rate_rule(ctx):
    """Check if rejection rate is within expected thresholds."""
    drug_df = ctx['drug_df']
    rejected = ctx['rejected']
    
    total = len(drug_df)
    rej_count = len(rejected)
    rej_rate = rej_count / total * 100 if total > 0 else 0
    
    crit = THRESHOLDS['rejection_rate_critical']
    warn = THRESHOLDS['rejection_rate_warning']
    
    if rej_rate > crit:
        return [('critical', f'🔴 HIGH rejection rate {rej_rate:.1f}% — above {crit}% threshold.')]
    elif rej_rate > warn:
        return [('warning', f'🟡 MODERATE rejection rate {rej_rate:.1f}% — monitor trend.')]
    else:
        return [('success', f'🟢 NORMAL rejection rate {rej_rate:.1f}% — within expected range.')]


def top_rejection_rule(ctx):
    """Identify the most common rejection code."""
    code_stats = ctx['code_stats']
    
    if code_stats.empty:
        return [("warning", "No rejection codes found.")]
    
    top_code = code_stats.iloc[0]['REJ_CODE_PREFIX']
    top_count = int(code_stats.iloc[0]['Count'])
    top_amt = code_stats.iloc[0]['Rejected_Amt']
    return [('info', f'💰 {top_code} ({CODE_DESC.get(top_code,"")}) is #1 rejection: {top_count:,} claims | {top_amt:,.0f} rejected amount.')]


def mnec_rule(ctx):
    """Check for MNEC miscoding (refill too soon misclassified)."""
    rejected = ctx['rejected']
    
    if 'MNEC' not in rejected['REJ_CODE_PREFIX'].values or 'REJ_REMARKS' not in rejected.columns:
        return []
    
    mnec_df = rejected[rejected['REJ_CODE_PREFIX'] == 'MNEC']
    if len(mnec_df) == 0:
        return []
    
    refill_n = mnec_df['REJ_REMARKS'].str.contains('refill', case=False, na=False).sum()
    refill_p = refill_n / len(mnec_df) * 100
    
    if refill_p > 50:
        return [('critical', f'⚠️ MNEC MISCODING: {refill_p:.1f}% of MNEC is Refill Too Soon ({refill_n:,} claims). System coding bug.')]
    return []


def refill_too_soon_rule(ctx):
    """Alert about refill too soon rejections."""
    rejected = ctx['rejected']
    
    if 'REJ_REMARKS' not in rejected.columns:
        return []
    
    rf = rejected['REJ_REMARKS'].str.contains('refill', case=False, na=False).sum()
    if rf == 0:
        return []
    
    rf_amt = rejected[rejected['REJ_REMARKS'].str.contains('refill', case=False, na=False)]['TREAT_REJ_AMT'].sum()
    return [('warning', f'🔄 Refill Too Soon: {rf:,} rejections | {rf_amt:,.0f} amount → Alert doctors of next allowed refill date.')]


def ncov_rule(ctx):
    """Monitor not covered rejections."""
    rejected = ctx['rejected']
    
    ncov_n = len(rejected[rejected['REJ_CODE_PREFIX'] == 'NCOV'])
    if ncov_n == 0:
        return []
    
    ncov_a = rejected[rejected['REJ_CODE_PREFIX'] == 'NCOV']['TREAT_REJ_AMT'].sum()
    return [('warning', f'📋 Not Covered (NCOV): {ncov_n:,} claims | {ncov_a:,.0f} → Show covered drug list at prescription time.')]


def eligibility_rule(ctx):
    """Check for eligibility rejection spikes."""
    rejected = ctx['rejected']
    
    rej_count = len(rejected)
    if rej_count == 0:
        return []
    
    elig_n = len(rejected[rejected['REJ_CODE_PREFIX'] == 'ELIG'])
    elig_p = elig_n / rej_count * 100
    
    if elig_p > 5:
        return [('warning', f'👤 ELIG spike: {elig_n:,} eligibility rejections ({elig_p:.1f}%) → Check for expired memberships.')]
    return []


def age_rule(ctx):
    """Check for age-related safety violations (adult drugs to minors)."""
    drug_df = ctx['drug_df']
    rejected = ctx['rejected']
    
    if 'AGE_GROUP' not in drug_df.columns:
        return []
    
    child_v = rejected[(rejected['REJ_CODE_PREFIX'] == 'CODE') &
                       (rejected['AGE_GROUP'].isin(['INFANT (0-2)', 'CHILD (3-12)', 'TEEN (13-17)']))]
    
    if len(child_v) > 0:
        return [('critical', f'🚨 PATIENT SAFETY: {len(child_v):,} age rule violations for children → Adult drugs to minors. Urgent review.')]
    return []


def high_risk_rule(ctx):
    """Alert on drug-diagnosis combos with very high rejection rates."""
    high_risk = ctx['high_risk']
    
    if len(high_risk) > 0:
        return [('info', f'🚫 {len(high_risk)} drug-diagnosis combos with >70% rejection → Add to hard exclusion rules.')]
    return []


def fraud_rule(ctx):
    """Check for fraud flags in the data."""
    drug_df = ctx['drug_df']
    
    if 'PA_FLAG_REASON' not in drug_df.columns or not drug_df['PA_FLAG_REASON'].notna().any():
        return []
    
    fraud_n = drug_df[drug_df['PA_FLAG_REASON'].str.contains('fraud', case=False, na=False)]
    if len(fraud_n) > 0:
        return [('critical', f'🔍 FRAUD FLAG: {len(fraud_n):,} claims flagged ({fraud_n["IS_REJECTED"].mean()*100:.1f}% rejection) → Review with compliance.')]
    return []


def new_drugs_ncov_rule(ctx):
    """Detect drugs that were previously approved but now always rejected (formulary changes)."""
    new_drugs_ncov = ctx['new_drugs_ncov']
    
    if new_drugs_ncov is None or new_drugs_ncov.empty:
        return []
    
    if 'Ever_Approved' not in new_drugs_ncov.columns:
        return []
    
    was_approved = new_drugs_ncov[new_drugs_ncov['Ever_Approved']]
    if len(was_approved) > 0:
        return [('critical',
                f'⚠️ {len(was_approved)} drug(s) now 100% NCOV-rejected WERE approved before for the '
                f'same diagnosis → likely a formulary drop or coding bug, not a genuinely new drug.')]
    return []


def payment_anomaly_rule(ctx):
    """Check for payment integrity issues."""
    payment_anomaly_summary = ctx['payment_anomaly_summary']
    
    if payment_anomaly_summary is None:
        return []
    
    insights = []
    
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
    
    return insights


def emerging_patterns_rule(ctx):
    """Report dynamically discovered patterns."""
    emerging_findings = ctx['emerging_findings']
    
    if emerging_findings is None or emerging_findings.empty:
        return []
    
    crit_n = int((emerging_findings['Severity'] == 'critical').sum())
    if crit_n > 0:
        return [('critical',
                f'🆕 {crit_n} critical statistically-discovered pattern(s) this month (new combos, '
                f'demographic mismatches, volume spikes) → see Fraud & Safety / Emerging Patterns.')]
    return []