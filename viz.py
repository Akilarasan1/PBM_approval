"""Visualization and plotting functions for PBM Dashboard."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils import PLOT_THEME, COLORS, CODE_KEYWORDS_MNEC, AGE_ORDER


def plot_rejection_codes_volume_financial(code_stats):
    """Plot rejection codes by volume and financial impact."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    colors_bar = [COLORS.get(c, '#607D8B') for c in code_stats['REJ_CODE_PREFIX']]
    
    fig.add_trace(go.Bar(
        x=code_stats['REJ_CODE_PREFIX'],
        y=code_stats['Count'],
        name='Claims',
        marker_color=colors_bar,
        marker_line_width=0,
    ), secondary_y=False)
    
    fig.add_trace(go.Scatter(
        x=code_stats['REJ_CODE_PREFIX'],
        y=code_stats['Rejected_Amt'],
        name='Rejected Amt',
        mode='lines+markers',
        line=dict(color='#FFC107', width=2),
        marker=dict(size=8, color='#FFC107'),
    ), secondary_y=True)
    
    fig.update_layout(**PLOT_THEME, showlegend=True)
    fig.update_yaxes(title_text='Claims', secondary_y=False,
                     gridcolor='#21262D', tickfont=dict(color='#8B949E'))
    fig.update_yaxes(title_text='Rejected Amount', secondary_y=True,
                     gridcolor='#21262D', tickfont=dict(color='#8B949E'))
    return fig


def plot_mnec_breakdown(mnec_df):
    """Plot MNEC rejection breakdown by category."""
    if 'REJ_REMARKS' not in mnec_df.columns:
        return None
    
    assigned = set()
    buckets = {}
    
    for label, patterns in CODE_KEYWORDS_MNEC.items():
        mask = mnec_df['REJ_REMARKS'].str.contains('|'.join(patterns), case=False, na=False)
        subset = mnec_df[mask & ~mnec_df.index.isin(assigned)]
        buckets[label] = len(subset)
        assigned.update(subset.index)
    
    buckets['Other'] = len(mnec_df) - sum(buckets.values())

    fig = go.Figure(go.Pie(
        labels=list(buckets.keys()),
        values=list(buckets.values()),
        hole=0.6,
        marker_colors=['#FF4444','#FF8C00','#FFC107','#607D8B'],
        textfont=dict(color='#E6EDF3', size=12),
    ))

    fig.update_layout(**PLOT_THEME, height=280,
                      annotations=[dict(text='MNEC', x=0.5, y=0.5,
                                       font_size=16, font_color='#E6EDF3',
                                       showarrow=False)])
    return fig


def plot_gender_rejection_rate(drug_df):
    """Plot rejection rate by gender."""
    if 'MEM_GENDER' not in drug_df.columns or 'IS_REJECTED' not in drug_df.columns:
        return None
    
    gen = (drug_df.groupby('MEM_GENDER')
           .agg(Total=('IS_REJECTED','count'), Rejected=('IS_REJECTED','sum'))
           .reset_index())
    gen['RejRate'] = (gen['Rejected'] / gen['Total'] * 100).round(1)
    
    fig = go.Figure(go.Bar(
        x=gen['MEM_GENDER'], y=gen['RejRate'],
        marker_color=['#2196F3','#E91E63','#607D8B'],
        text=gen['RejRate'].apply(lambda x: f'{x:.1f}%'),
        textposition='outside', textfont=dict(color='#E6EDF3'),
        marker_line_width=0,
    ))
    fig.update_layout(**PLOT_THEME, height=260, yaxis_title='Rejection Rate %')
    return fig


def plot_age_rejection_rate(drug_df):
    """Plot rejection rate by age group."""
    if 'AGE_GROUP' not in drug_df.columns or 'IS_REJECTED' not in drug_df.columns:
        return None
    
    age_s = (drug_df.groupby('AGE_GROUP')
             .agg(Total=('IS_REJECTED','count'), Rejected=('IS_REJECTED','sum'))
             .reset_index())
    age_s['RejRate'] = (age_s['Rejected'] / age_s['Total'] * 100).round(1)
    age_s['order'] = age_s['AGE_GROUP'].apply(lambda x: AGE_ORDER.index(x) if x in AGE_ORDER else 99)
    age_s = age_s.sort_values('order')
    
    fig = go.Figure(go.Bar(
        x=age_s['AGE_GROUP'], y=age_s['RejRate'],
        marker_color='#388BFD',
        text=age_s['RejRate'].apply(lambda x: f'{x:.1f}%'),
        textposition='outside', textfont=dict(color='#E6EDF3'),
        marker_line_width=0,
    ))
    fig.update_layout(**PLOT_THEME, height=260,
                      yaxis_title='Rejection Rate %',
                      xaxis_tickangle=-30)
    return fig


def plot_rejected_amount_by_code(code_stats):
    """Plot rejected amount by rejection code."""
    if code_stats.empty or 'Rejected_Amt' not in code_stats.columns:
        return None
    
    fig = go.Figure(go.Bar(
        x=code_stats['Rejected_Amt'],
        y=code_stats['REJ_CODE_PREFIX'],
        orientation='h',
        marker_color=[COLORS.get(c, '#607D8B') for c in code_stats['REJ_CODE_PREFIX']],
        text=code_stats['Rejected_Amt'].apply(lambda x: f'{x:,.0f}'),
        textposition='outside', textfont=dict(color='#E6EDF3', size=11),
        marker_line_width=0,
    ))
    fig.update_layout(**PLOT_THEME, height=360, xaxis_title='Rejected Amount')
    return fig


def plot_amount_distribution_treemap(code_stats):
    """Plot amount distribution as treemap."""
    if code_stats.empty or 'Rejected_Amt' not in code_stats.columns:
        return None
    
    # Filter to only non-zero amounts to avoid normalization error
    filtered = code_stats[code_stats['Rejected_Amt'] > 0].copy()
    if filtered.empty:
        return None
    
    fig = px.treemap(
        filtered,
        path=['REJ_CODE_PREFIX'],
        values='Rejected_Amt',
        color='Rejected_Amt',
        color_continuous_scale=['#1A2332','#FF4444'],
        custom_data=['Description', 'Count']
    )

    fig.update_traces(
        texttemplate='<b>%{label}</b><br>%{value:,.0f}',
        hovertemplate='<b>%{label}</b><br>%{customdata[0]}<br>Claims: %{customdata[1]:,}<br>Amount: %{value:,.0f}',
        textfont=dict(color='#E6EDF3', size=13)
    )
    fig.update_layout(**PLOT_THEME, height=360, coloraxis_showscale=False)
    return fig


def plot_top_rejected_drugs(rejected):
    """Plot top 15 most expensive rejected drugs."""
    if 'DRUG_CODE' not in rejected.columns or 'TREAT_REJ_AMT' not in rejected.columns:
        return None
    
    drug_amt = (rejected.groupby('DRUG_CODE')
                .agg(Claims=('TREAT_REJ_AMT','count'),
                     Total_Amt=('TREAT_REJ_AMT','sum'),
                     Avg_Amt=('TREAT_REJ_AMT','mean'))
                .reset_index()
                .query('Claims >= 5')
                .sort_values('Total_Amt', ascending=False)
                .head(15))
    
    if len(drug_amt) == 0:
        return None
    
    drug_amt['Total_Amt'] = drug_amt['Total_Amt'].round(0)
    
    fig = go.Figure(go.Bar(
        y=drug_amt['DRUG_CODE'],
        x=drug_amt['Total_Amt'],
        orientation='h',
        marker_color='#FF4444',
        marker_line_width=0,
        text=drug_amt['Total_Amt'].apply(lambda x: f'{x:,.0f}'),
        textposition='outside', textfont=dict(color='#E6EDF3', size=11)
    ))
    fig.update_layout(**PLOT_THEME, height=420, xaxis_title='Total Rejected Amount')
    return fig


def plot_high_risk_combos(high_risk):
    """Plot high-risk drug-diagnosis combinations."""
    if high_risk.empty:
        return None
    
    fig = go.Figure(go.Bar(
        x=high_risk.head(20)['DRUG_DIAG_COMBO'],
        y=high_risk.head(20)['RejRate'],
        marker_color='#FF4444',
        marker_line_width=0,
        text=high_risk.head(20)['RejRate'].apply(lambda x: f'{x:.0f}%'),
        textposition='outside', textfont=dict(color='#E6EDF3'),
    ))
    fig.add_hline(y=70, line_dash='dash', line_color='#FF8C00',
                  annotation_text='70% threshold', annotation_font_color='#FF8C00')
    fig.update_layout(**PLOT_THEME, height=380, xaxis_tickangle=-45,
                      yaxis_title='Rejection Rate %')
    return fig


def plot_provider_volume_and_rejection(prov):
    """Plot top 20 providers by volume with rejection rate overlay."""
    if prov.empty:
        return None
    
    top20 = prov.sort_values('Claims', ascending=False).head(20)
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=top20['DOC_LIC_NO'], y=top20['Claims'],
        name='Claims', marker_color='#388BFD', marker_line_width=0,
    ))
    fig.add_trace(go.Scatter(
        x=top20['DOC_LIC_NO'], y=top20['RejRate'],
        name='Rej Rate %', mode='lines+markers',
        line=dict(color='#FF4444', width=2),
        marker=dict(size=6), yaxis='y2'
    ))
    fig.update_layout(
        **PLOT_THEME, height=360,
        yaxis=dict(title='Claims', gridcolor='#21262D', tickfont=dict(color='#8B949E')),
        yaxis2=dict(title='Rejection Rate %', overlaying='y', side='right',
                    gridcolor='#21262D', tickfont=dict(color='#8B949E')),
        xaxis_tickangle=-45, showlegend=True
    )
    return fig


def plot_provider_risk_map(prov):
    """Plot provider risk map (claims vs rejection rate)."""
    if prov.empty:
        return None
    
    prov_plot = prov[prov['Claims'] >= 20].copy()
    if prov_plot.empty:
        return None
    
    fig = go.Figure(go.Scatter(
        x=prov_plot['Claims'],
        y=prov_plot['RejRate'],
        mode='markers',
        marker=dict(
            size=prov_plot['RejAmt'].apply(lambda x: max(6, min(30, x/5000))),
            color=prov_plot['RejRate'],
            colorscale=[[0,'#00C853'],[0.3,'#FFC107'],[1,'#FF4444']],
            showscale=True,
            colorbar=dict(title='Rej Rate %', tickfont=dict(color='#8B949E')),
            line=dict(width=0)
        ),
        text=prov_plot['DOC_LIC_NO'],
        hovertemplate='<b>%{text}</b><br>Claims: %{x:,}<br>Rej Rate: %{y:.1f}%<extra></extra>'
    ))
    fig.add_hline(y=30, line_dash='dash', line_color='#FF8C00',
                  annotation_text='30% threshold', annotation_font_color='#FF8C00')
    fig.update_layout(**PLOT_THEME, height=380,
                      xaxis_title='Total Claims', yaxis_title='Rejection Rate %')
    return fig


def plot_age_violations(child_v):
    """Plot age rule violations by age group."""
    if child_v.empty or 'AGE_GROUP' not in child_v.columns:
        return None
    
    age_breakdown = child_v['AGE_GROUP'].value_counts().reset_index()
    age_breakdown.columns = ['Age Group', 'Violations']
    
    fig = go.Figure(go.Bar(
        x=age_breakdown['Age Group'], y=age_breakdown['Violations'],
        marker_color='#FF4444', marker_line_width=0,
        text=age_breakdown['Violations'], textposition='outside',
        textfont=dict(color='#E6EDF3')
    ))
    fig.update_layout(**PLOT_THEME, height=220)
    return fig


def plot_anomaly_scatter(findings):
    """Bubble chart: anomaly score vs current value, colored by dimension,
    sized by how many baseline months back it — gives analysts a single
    glance at where the emerging-pattern findings cluster."""
    if findings.empty:
        return None

    dim_colors = {
        'Provider Behavior': '#00BCD4', 'Drug Utilization': '#9C27B0',
        'Diagnosis Drift': '#FF8C00', 'Rejection Code Drift': '#FF4444',
        'Gender Anomaly': '#E91E63', 'Age Anomaly': '#FFC107',
        'Gender × Diagnosis': '#FF4444', 'Age × Drug': '#FF4444',
        'New Drug-Diagnosis Combo': '#4CAF50',
    }

    fig = go.Figure()
    for dim, sub in findings.groupby('Dimension'):
        fig.add_trace(go.Scatter(
            x=sub['Rank'], y=sub['Anomaly_Score'],
            mode='markers', name=dim,
            marker=dict(
                size=(sub['Current'].clip(upper=200) / 8 + 6),
                color=dim_colors.get(dim, '#607D8B'),
                line=dict(width=0), opacity=0.85,
            ),
            text=sub['Entity'],
            hovertemplate='<b>%{text}</b><br>Score: %{y:.0f}<extra></extra>',
        ))
    fig.add_hline(y=80, line_dash='dash', line_color='#FF4444',
                  annotation_text='critical', annotation_font_color='#FF4444')
    fig.add_hline(y=55, line_dash='dash', line_color='#FF8C00',
                  annotation_text='warning', annotation_font_color='#FF8C00')
    fig.update_layout(**PLOT_THEME, height=380, showlegend=True,
                      xaxis_title='Rank (by severity)', yaxis_title='Anomaly Score')
    return fig


def plot_findings_by_dimension(findings):
    """Stacked bar: count of findings per dimension, split by severity."""
    if findings.empty:
        return None

    sev_colors = {'critical': '#FF4444', 'warning': '#FF8C00', 'info': '#2196F3'}
    counts = findings.groupby(['Dimension', 'Severity']).size().reset_index(name='Count')

    fig = go.Figure()
    for sev in ['critical', 'warning', 'info']:
        sub = counts[counts['Severity'] == sev]
        fig.add_trace(go.Bar(
            x=sub['Dimension'], y=sub['Count'], name=sev.title(),
            marker_color=sev_colors[sev], marker_line_width=0,
        ))
    fig.update_layout(**PLOT_THEME, height=320, barmode='stack',
                      xaxis_tickangle=-25, showlegend=True)
    return fig


def plot_entity_trend(history_long, entity_label, current_month, current_val, value_col):
    """Small line chart: one entity's metric across baseline months + current
    month, for the drill-down view on a single finding."""
    if history_long is None or history_long.empty:
        return None

    trend = history_long.sort_values('month')[['month', value_col]].copy()
    trend = pd.concat([trend, pd.DataFrame({'month': [current_month], value_col: [current_val]})],
                       ignore_index=True)

    fig = go.Figure(go.Scatter(
        x=trend['month'], y=trend[value_col],
        mode='lines+markers',
        line=dict(color='#388BFD', width=2),
        marker=dict(size=10, color=['#388BFD'] * (len(trend) - 1) + ['#FF4444']),
    ))
    fig.update_layout(**PLOT_THEME, height=220, title=f'{entity_label} — {value_col} over time')
    return fig


def plot_weekly_trends(weekly):
    """Plot weekly claim volume and rejection rate."""
    if weekly.empty:
        return None

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            x=weekly['Week'],
            y=weekly['Total'],
            name='Claims',
            marker_color='#388BFD',
            marker_line_width=0,
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=weekly['Week'],
            y=weekly['RejRate'],
            name='Rej Rate %',
            mode='lines+markers',
            line=dict(color='#FF4444', width=2),
            marker=dict(size=7, color='#FF4444'),
        ),
        secondary_y=True,
    )
    fig.update_layout(**PLOT_THEME, height=320, showlegend=True)
    fig.update_yaxes(title_text='Claims', secondary_y=False, gridcolor='#21262D')
    fig.update_yaxes(title_text='Rejection Rate %', secondary_y=True, gridcolor='#21262D')
    fig.update_xaxes(tickangle=-30)
    return fig
