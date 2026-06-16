"""Configuration, constants, and styling for PBM Dashboard."""

# ── PLOTLY THEME ───────────────────────────────────────────────
PLOT_THEME = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='Inter', color='#8B949E', size=12),
    title_font=dict(color='#E6EDF3', size=14, family='Inter'),
    legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor='#21262D', borderwidth=1),
    margin=dict(t=40, b=40, l=40, r=20)
)

# ── COLOR SCHEME ────────────────────────────────────────────────
COLORS = {
    'MNEC':'#FF4444','NCOV':'#FF8C00','ELIG':'#FFC107',
    'MTQ':'#9C27B0','BENX':'#E91E63','AUTH':'#00BCD4',
    'CODE':'#4CAF50','DUPL':'#607D8B','NO_CODE':'#37474F',
    'approved':'#00C853','rejected':'#FF4444'
}

CODE_DESC = {
    'MNEC':'Not Clinically Indicated','NCOV':'Not Covered',
    'ELIG':'Eligibility Issue','MTQ':'Benefit / Qty Limit',
    'BENX':'Benefit Exhausted','AUTH':'Auth Required',
    'CODE':'Age / Gender Rule','DUPL':'Duplicate','NO_CODE':'No Code'
}

# ── MNEC KEYWORD PATTERNS ──────────────────────────────────────
CODE_KEYWORDS_MNEC = {
    'Clinical Mismatch': ['clinically indicated','non indicated','not related','medically indicated','not clinically'],
    'Refill Too Soon':   ['refill'],
    'Duplicate':         ['duplicate','overlaps','same day']
}

# ── AGE GROUP ORDER ─────────────────────────────────────────────
AGE_ORDER = [
    'INFANT (0-2)', 'CHILD (3-12)', 'TEEN (13-17)',
    'YOUNG ADULT (18-35)', 'ADULT (36-55)',
    'SENIOR (56-70)', 'ELDERLY (71+)'
]

# ── STREAMLIT DESIGN CSS ───────────────────────────────────────
STREAMLIT_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* Base */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #0D1117; color: #E6EDF3; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #161B22;
    border-right: 1px solid #21262D;
}
[data-testid="stSidebar"] * { color: #E6EDF3 !important; }

/* Metric cards */
.metric-card {
    background: #161B22;
    border: 1px solid #21262D;
    border-radius: 12px;
    padding: 20px 24px;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
}
.metric-card.red::before   { background: linear-gradient(90deg, #FF4444, #FF8C00); }
.metric-card.green::before { background: linear-gradient(90deg, #00C853, #00BCD4); }
.metric-card.blue::before  { background: linear-gradient(90deg, #2196F3, #9C27B0); }
.metric-card.amber::before { background: linear-gradient(90deg, #FF8C00, #FFC107); }
.metric-card.purple::before{ background: linear-gradient(90deg, #9C27B0, #E91E63); }

.metric-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #8B949E;
    margin-bottom: 8px;
}
.metric-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 28px;
    font-weight: 700;
    color: #E6EDF3;
    line-height: 1;
}
.metric-sub {
    font-size: 12px;
    color: #8B949E;
    margin-top: 6px;
}

/* Insight cards */
.insight-card {
    background: #161B22;
    border: 1px solid #21262D;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 10px;
    font-size: 13px;
    line-height: 1.6;
    color: #C9D1D9;
}
.insight-card.critical { border-left: 3px solid #FF4444; }
.insight-card.warning  { border-left: 3px solid #FF8C00; }
.insight-card.info     { border-left: 3px solid #2196F3; }
.insight-card.success  { border-left: 3px solid #00C853; }

/* Section headers */
.section-header {
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: #8B949E;
    padding: 8px 0;
    border-bottom: 1px solid #21262D;
    margin-bottom: 16px;
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    background: #161B22;
    border-radius: 10px;
    padding: 4px;
    gap: 2px;
    border: 1px solid #21262D;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #8B949E;
    border-radius: 8px;
    font-size: 13px;
    font-weight: 500;
    padding: 8px 16px;
}
.stTabs [aria-selected="true"] {
    background: #21262D !important;
    color: #E6EDF3 !important;
}

/* DataFrames */
[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }
.stDataFrame { background: #161B22; }

/* Upload area */
[data-testid="stFileUploader"] {
    background: #161B22;
    border: 2px dashed #21262D;
    border-radius: 12px;
    padding: 20px;
}
[data-testid="stFileUploader"]:hover { border-color: #388BFD; }

/* Plotly charts transparent bg */
.js-plotly-plot { border-radius: 12px; }

/* Hide streamlit branding */
#MainMenu, footer, header { visibility: hidden; }

/* Custom scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0D1117; }
::-webkit-scrollbar-thumb { background: #21262D; border-radius: 3px; }

/* Alert boxes */
.stAlert { border-radius: 10px; }
</style>
"""
