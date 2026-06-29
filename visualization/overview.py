"""Visualization and plotting functions for PBM Dashboard."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from config.utils import PLOT_THEME, COLORS, CODE_KEYWORDS_MNEC, AGE_ORDER


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
                                       showarrow=False)
                                       ],title_text = " ")
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
    fig.update_layout(**PLOT_THEME, height=260, yaxis_title='Rejection Rate %',title_text="")
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
                      xaxis_tickangle=-30,title_text="")
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
    fig.update_layout(**PLOT_THEME, height=360, xaxis_title='Rejected Amount',title_text="")
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
    fig.update_layout(**PLOT_THEME, height=360, coloraxis_showscale=False,title_text="")
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
    fig.update_layout(**PLOT_THEME, height=420, xaxis_title='Total Rejected Amount',title_text=" ")
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
        xaxis_tickangle=-45, showlegend=True,title_text=" "
    )
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
    fig.update_layout(**PLOT_THEME, height=320, showlegend=True,title_text="")
    fig.update_yaxes(title_text='Claims', secondary_y=False, gridcolor='#21262D')
    fig.update_yaxes(title_text='Rejection Rate %', secondary_y=True, gridcolor='#21262D')
    fig.update_xaxes(tickangle=-30)
    return fig



def plot_service_type_rejection(service_type_stats):
    """Bar chart: rejection rate by service type, with claim count labeled
    so a tiny inpatient sample isn't misread as a strong signal."""
    if service_type_stats.empty:
        return None

    fig = go.Figure(go.Bar(
        x=service_type_stats['Service_Type'], y=service_type_stats['RejRate_%'],
        marker_color=['#388BFD', '#9C27B0'][:len(service_type_stats)],
        marker_line_width=0,
        text=[f"{r:.1f}% (n={c:,})" for r, c in zip(service_type_stats['RejRate_%'], service_type_stats['Claims'])],
        textposition='outside', textfont=dict(color='#E6EDF3'),
    ))
    fig.update_layout(**PLOT_THEME, height=300, yaxis_title='Rejection Rate %',title_text="")
    return fig


def plot_service_type_monthly_trend(trend):
    """Line chart: rejection rate per service type, month by month."""
    if trend.empty:
        return None

    fig = go.Figure()
    colors = {'Outpatient': '#388BFD', 'Inpatient': '#9C27B0'}
    for stype, sub in trend.groupby('Service_Type'):
        fig.add_trace(go.Scatter(
            x=sub['Month'], y=sub['RejRate_%'], name=stype,
            mode='lines+markers', line=dict(color=colors.get(stype, '#607D8B'), width=2),
            marker=dict(size=8),
        ))
    fig.update_layout(**PLOT_THEME, height=300, showlegend=True, yaxis_title='Rejection Rate %',title_text="")
    return fig


def plot_quantity_bands(band_stats):
    """Claims volume (bars) + rejection rate (line) per quantity band."""
    if band_stats.empty:
        return None

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        x=band_stats['Band'], y=band_stats['Claims'], name='Claims',
        marker_color='#388BFD', marker_line_width=0,
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=band_stats['Band'], y=band_stats['Rejection_Rate_%'], name='Rejection Rate %',
        mode='lines+markers', line=dict(color='#FF4444', width=2), marker=dict(size=9),
    ), secondary_y=True)
    fig.update_layout(**PLOT_THEME, height=320, showlegend=True,title_text=" ")
    fig.update_yaxes(title_text='Claims', secondary_y=False, gridcolor='#21262D')
    fig.update_yaxes(title_text='Rejection Rate %', secondary_y=True, gridcolor='#21262D')
    return fig