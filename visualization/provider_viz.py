import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from config.utils import PLOT_THEME, COLORS


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
    fig.update_layout(**PLOT_THEME, height=220, title=f'{entity_label} — {value_col} over time',title_text="  ")
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
                      xaxis_title='Total Claims', yaxis_title='Rejection Rate %',title_text="  ")
    return fig



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
    
    fig.update_layout(**PLOT_THEME, showlegend=True, title_text = " ")
    fig.update_yaxes(title_text='Claims', secondary_y=False,
                     gridcolor='#21262D', tickfont=dict(color='#8B949E'))
    fig.update_yaxes(title_text='Rejected Amount', secondary_y=True,
                     gridcolor='#21262D', tickfont=dict(color='#8B949E'))
    


    return fig

