
import plotly.graph_objects as go
from config.utils import PLOT_THEME


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




def plot_diagnosis_drug_sankey(matrix):
    """Three-level Sankey: Diagnosis -> Drug -> Approved/Rejected. Every link
    is derived from the same matrix table, so flows stay consistent —
    a drug's outflow to Approved+Rejected always sums to its total inflow."""
    if matrix.empty:
        return None

    diagnoses = matrix['PA_PRIMARY_DIAG'].unique().tolist()
    drugs = matrix['DRUG_CODE'].unique().tolist()
    outcomes = ['Approved', 'Rejected']

    labels = [f"Dx: {d}" for d in diagnoses] + [f"Rx: {d}" for d in drugs] + outcomes
    diag_idx = {d: i for i, d in enumerate(diagnoses)}
    drug_idx = {d: i + len(diagnoses) for i, d in enumerate(drugs)}
    outcome_idx = {'Approved': len(diagnoses) + len(drugs), 'Rejected': len(diagnoses) + len(drugs) + 1}

    sources, targets, values, colors = [], [], [], []

    for _, r in matrix.iterrows():
        sources.append(diag_idx[r['PA_PRIMARY_DIAG']])
        targets.append(drug_idx[r['DRUG_CODE']])
        values.append(r['Claims'])
        colors.append('rgba(56,139,253,0.35)')

    by_drug = matrix.groupby('DRUG_CODE')[['Approved', 'Rejected']].sum().reset_index()
    for _, r in by_drug.iterrows():
        if r['Approved'] > 0:
            sources.append(drug_idx[r['DRUG_CODE']])
            targets.append(outcome_idx['Approved'])
            values.append(r['Approved'])
            colors.append('rgba(0,200,83,0.35)')
        if r['Rejected'] > 0:
            sources.append(drug_idx[r['DRUG_CODE']])
            targets.append(outcome_idx['Rejected'])
            values.append(r['Rejected'])
            colors.append('rgba(255,68,68,0.35)')

    node_colors = (
        ['#9C27B0'] * len(diagnoses) + ['#388BFD'] * len(drugs) + ['#00C853', '#FF4444']
    )

    fig = go.Figure(go.Sankey(
        node=dict(label=labels, color=node_colors, pad=12, thickness=14,
                  line=dict(color='#21262D', width=0.5)),
        link=dict(source=sources, target=targets, value=values, color=colors),
    ))
    theme = {**PLOT_THEME, 'font': dict(color='#E6EDF3', size=11, family='Inter')}
    fig.update_layout(**theme, height=520)
    return fig

def plot_diagnosis_drug_heatmap(matrix):
    """Diagnosis x Drug heatmap, color = rejection rate %."""
    if matrix.empty:
        return None

    pivot = matrix.pivot(index='PA_PRIMARY_DIAG', columns='DRUG_CODE', values='RejRate_%')

    fig = go.Figure(go.Heatmap(
        z=pivot.values, x=pivot.columns, y=pivot.index,
        colorscale=[[0, '#0D1117'], [0.5, '#FF8C00'], [1, '#FF4444']],
        colorbar=dict(title='Rej Rate %', tickfont=dict(color='#8B949E')),
        hovertemplate='Diagnosis: %{y}<br>Drug: %{x}<br>Rej Rate: %{z:.1f}%<extra></extra>',
    ))
    fig.update_layout(**PLOT_THEME, height=460, xaxis_tickangle=-45,
                      xaxis_title='Drug Code', yaxis_title='Diagnosis')
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



