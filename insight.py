
import pandas as pd
import numpy as np
from utils import CODE_DESC, THRESHOLDS

def gen_insights(drug_df, rejected, code_stats, high_risk):
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

    
    return insights