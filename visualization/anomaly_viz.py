import plotly.graph_objects as go
from config.utils import PLOT_THEME
from plotly.subplots import make_subplots


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
                      xaxis_title='Rank (by severity)', yaxis_title='Anomaly Score',title_text="")
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
                      xaxis_tickangle=-25, showlegend=True,title_text="")
    return fig




def plot_ncov_coverage_trend(trend, drug_code):
    """Line chart: claims/approved/NCOV-rejected for one drug, month by month —
    the visual for a coverage-flip drug, showing exactly when it changed."""
    sub = trend[trend['DRUG_CODE'] == drug_code].sort_values('Month')
    if sub.empty:
        return None

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=sub['Month'], y=sub['Claims'], name='Total Claims',
        marker_color='#388BFD', opacity=0.35, marker_line_width=0,
    ))
    fig.add_trace(go.Scatter(
        x=sub['Month'], y=sub['Approved_Claims'], name='Approved',
        mode='lines+markers', line=dict(color='#00C853', width=2), marker=dict(size=9),
    ))
    fig.add_trace(go.Scatter(
        x=sub['Month'], y=sub['NCOV_Rejections'], name='NCOV Rejected',
        mode='lines+markers', line=dict(color='#FF4444', width=2), marker=dict(size=9),
    ))
    fig.update_layout(**PLOT_THEME, height=300, showlegend=True,
                      yaxis_title='Claims', title=f'{drug_code} — coverage over time',title_text="")
    return fig



 
 
def plot_rejection_reason_trend(trend_df, label):
    """Generic monthly-trend bar chart for any keyword-bucket rejection-reason
    analysis (Documentation, Medical Necessity, etc.)."""
    if trend_df.empty:
        return None
 
    fig = go.Figure(go.Bar(
        x=trend_df['Month'], y=trend_df['Frequency'],
        marker_color='#FF8C00', marker_line_width=0,
        text=trend_df['Frequency'], textposition='outside', textfont=dict(color='#E6EDF3'),
    ))
    fig.update_layout(**PLOT_THEME, height=260, yaxis_title='Rejections', title=label,title_text="")
    return fig
 
 
