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

THRESHOLDS = {
    'rejection_rate_critical': 17,
    'rejection_rate_warning': 14,
    'combo_high_risk_rate': 70,
    'combo_gray_min_rate': 30,
    'combo_min_claims': 20,
    'provider_flag_rate': 30,
    'provider_flag_min_claims': 50,
    'provider_investigation_min_claims': 20,
    'new_drug_ncov_min_claims': 5,
    'emerging_baseline_months': 6,
    'emerging_min_current_claims': 10,
    'emerging_min_baseline_months': 2,
    'emerging_zscore_warning': 2.0,
    'emerging_zscore_critical': 3.0,
    'emerging_pct_change_warning': 50,
    'emerging_pct_change_critical': 100,
    'emerging_new_combo_min_claims': 5,
    'emerging_financial_min_delta': 10000,
    'dynamic_fraud_min_claims': 3,
    'dynamic_fraud_baseline_min_claims': 20,
    'dynamic_fraud_dominance_pct': 95,
}

SAMPLE_DATA_FILE = 'Pharmacy_QLM_February.csv'

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




/* Upload box */
[data-testid="stFileUploader"] {
    background-color: #161B22 !important;
    border: 2px dashed #388BFD !important;
    border-radius: 12px !important;
    padding: 20px !important;
}

/* Inner drop area */
[data-testid="stFileUploader"] section {
    background-color: #161B22 !important;
}

/* Upload text */
[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] p,
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] div { color: #E6EDF3 !important; font-weight: 500 !important;}

/* Browse files button */
[data-testid="stFileUploader"] button {
    background-color: #21262D !important;
    color: #E6EDF3 !important;
    border: 1px solid #388BFD !important;
}
   



/* Selectbox and all input labels - aggressive targeting */
[data-testid="stSelectbox"] label,
[data-testid="stSelectbox"] *,
[data-testid="stNumberInput"] label,
[data-testid="stNumberInput"] *,
[data-testid="stTextInput"] label,
[data-testid="stTextInput"] * {
    color: #E6EDF3 !important;
    font-weight: 500 !important;
}

/* Selectbox styling */
[data-testid="stSelectbox"],
[data-testid="stNumberInput"],
[data-testid="stTextInput"] {
    color: #E6EDF3 !important;
}

/* Force all div and span text in sidebar to be light */
[data-testid="stSidebar"] div,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label {
    color: #E6EDF3 !important;
}



/* ── FIX: white-on-white select/multiselect contrast ───────────
   Trying to force the TEXT to a fixed color kept losing to other
   rules, and Streamlit/BaseWeb sometimes shows the value through a
   placeholder (which `color` on the input does NOT style — that
   needs `::placeholder`). Instead of fighting over text color,
   make the control itself dark so it matches the rest of the app
   and contrast is guaranteed no matter which DOM trick renders the
   value (div, span, or input/placeholder).

   This covers Gender, Age Group, Period (sidebar) AND Provider
   drill-down, "Inspect a finding", dimension/severity filters
   (main-content tabs) — the closed control lives in the normal
   component tree, so these selectors reach it everywhere. */


[data-baseweb="select"] > div {
    background-color: #161B22 !important;
    border-color: #30363D !important;
}
[data-baseweb="select"] > div *,
[data-baseweb="select"] input {
    color: #E6EDF3 !important;
    -webkit-text-fill-color: #E6EDF3 !important;
    font-weight: 500 !important;
}
[data-baseweb="select"] input::placeholder {
    color: #E6EDF3 !important;
    opacity: 1 !important;
}
[data-baseweb="select"] svg {
    fill: #8B949E !important;
}

/* The OPEN dropdown menu is rendered by BaseWeb in a portal attached
   directly to <body> — it is NOT nested inside stSidebar/stSelectbox,
   so none of the rules above can ever reach it. It needs its own
   top-level selector. */


   
[data-baseweb="popover"] [data-baseweb="menu"] {
    background-color: #acb2b9 !important;
}


[data-baseweb="popover"] li,
[data-baseweb="popover"] li * {
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
    opacity: 1 !important;
}

[data-baseweb="popover"] li:hover {
    background-color: #E5E5E5 !important;
    color: #000000 !important;
}

[data-baseweb="popover"] li[aria-selected="true"] {
    background-color: #21262D !important;
    color: #FFFFFF !important;
}






/* Plotly charts transparent bg */
.js-plotly-plot { border-radius: 12px; }

#MainMenu { visibility: hidden;}

footer { visibility: hidden;}

/* Custom scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0D1117; }
::-webkit-scrollbar-thumb { background: #21262D; border-radius: 3px; }

/* Alert boxes */
.stAlert { border-radius: 10px; }


/* Spinner / cache message */
[data-testid="stSpinner"] {
    color: #000000 !important;
}

[data-testid="stSpinner"] * {
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
    opacity: 1 !important;
}




</style>
"""

