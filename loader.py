import pandas as pd
import streamlit as st
import time, io
from explore.analytics import *


def _categorize_age(a):
    """Categorize age into age groups."""
    if pd.isna(a):
        return 'UNKNOWN'
    a = int(a)
    if a <= 2: return 'INFANT (0-2)'
    if a <= 12: return 'CHILD (3-12)'
    if a <= 17: return 'TEEN (13-17)'
    if a <= 35: return 'YOUNG ADULT (18-35)'
    if a <= 55: return 'ADULT (36-55)'
    if a <= 70: return 'SENIOR (56-70)'
    return 'ELDERLY (71+)'



@st.cache_data(show_spinner="Processing uploaded file — large Excel files can take a minute or more...")
def process(file_bytes, filename):
    """
    Process uploaded PBM claims file and add derived columns.
    
    Args:
        file_bytes: Binary content of uploaded file
        filename: Name of the file (determines if CSV or Excel)
    
    Returns:
        Processed DataFrame with derived columns for analysis
    """
    # Load file
    t = time.perf_counter()
    if filename.endswith('.csv'):
        # Read with low_memory=False to avoid mixed-type dtype warnings
        start = time.perf_counter()
        df = pd.read_csv(io.BytesIO(file_bytes),engine="pyarrow")
        print(f"Read CSV Time: {time.perf_counter()-start:.2f}s")

    else:
        start = time.perf_counter()
        # df = pd.read_excel(io.BytesIO(file_bytes),engine="calamine")
        USE_COLS = list(REQUIRED_COLUMNS | OPTIONAL_COLUMNS)

        df = pd.read_excel(
            io.BytesIO(file_bytes),
            engine="calamine",
            usecols=lambda c: c in USE_COLS
        )

        print(f"Read Excel Time: {time.perf_counter()-start:.2f}s")

        

    missing_required, missing_optional = validate_columns(df)
    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")


    # Filter to prescribed drugs only
    if 'INS_TREAT_DESC' in df.columns:
        drug_df = df[df['INS_TREAT_DESC'] == 'Prescribed Drugs'].copy()
    else:
        drug_df = df.copy()

    # Some PBM exports mix string license numbers ("D959") with stray raw
    # integers in the same column. Left as-is, this breaks any sorted()/
    # comparison call downstream (e.g. the provider drill-down selector)
    # with "'<' not supported between instances of 'int' and 'str'".
    # Normalize once, here, so every consumer sees a clean string.
    if 'DOC_LIC_NO' in drug_df.columns:
        drug_df['DOC_LIC_NO'] = drug_df['DOC_LIC_NO'].astype(str)
    
    # Add rejection indicator
    if 'PBM_APPR_STS' in drug_df.columns:
        drug_df['IS_REJECTED'] = (drug_df['PBM_APPR_STS'] == 'PBM_REJECT').astype(int)
    else:
        drug_df['IS_REJECTED'] = 0
    
    # Extract rejection code prefix
    if 'PBM_REJ_CODE' in drug_df.columns:
        drug_df['REJ_CODE_PREFIX'] = (drug_df['PBM_REJ_CODE'].fillna('NO_CODE')
                                       .astype(str).str.split('-').str[0])
    else:
        drug_df['REJ_CODE_PREFIX'] = 'NO_CODE'
    
    # Create drug-diagnosis combo column
    drug_code_col = drug_df['DRUG_CODE'].fillna('UNK').astype(str) if 'DRUG_CODE' in drug_df.columns else 'UNK'
    diag_col = drug_df['PA_PRIMARY_DIAG'].fillna('UNK').astype(str) if 'PA_PRIMARY_DIAG' in drug_df.columns else 'UNK'
    drug_df['DRUG_DIAG_COMBO'] = drug_code_col + ' | ' + diag_col
    
    # Convert rejected amount to absolute value
    if 'TREAT_REJ_AMT' in drug_df.columns:
        drug_df['TREAT_REJ_AMT'] = drug_df['TREAT_REJ_AMT'].abs()
    else:
        drug_df['TREAT_REJ_AMT'] = 0

    # Parse service date and extract week
    if 'SERVICE_DT' in drug_df.columns:
        # drug_df['SERVICE_DT'] = pd.to_datetime(drug_df['SERVICE_DT'], errors='coerce')
        # drug_df['SERVICE_DT'] = pd.to_datetime(drug_df['SERVICE_DT'],format='%d-%m-%Y %H:%M',errors='coerce')
        parsed = pd.to_datetime(drug_df['SERVICE_DT'], format='%d-%m-%Y %H:%M', errors='coerce')
        still_bad = parsed.isna() & drug_df['SERVICE_DT'].notna()
        if still_bad.any():
            parsed.loc[still_bad] = pd.to_datetime(drug_df['SERVICE_DT'][still_bad], errors='coerce')
        drug_df['SERVICE_DT'] = parsed

        drug_df['WEEK'] = drug_df['SERVICE_DT'].dt.isocalendar().week.astype(str)
        

    # Normalize rejection remarks for MNEC/refill analysis
    if 'REJ_REMARKS' not in drug_df.columns or drug_df['REJ_REMARKS'].isna().all():
        if 'PBM_REJ_DESC' in drug_df.columns:
            drug_df['REJ_REMARKS'] = drug_df['PBM_REJ_DESC']
        elif 'PAT_REJ_REMARKS' in drug_df.columns:
            drug_df['REJ_REMARKS'] = drug_df['PAT_REJ_REMARKS']

    # Human-readable drug label when available
    if 'PROV_TREAT_DESC' in drug_df.columns:
        drug_df['DRUG_NAME'] = drug_df['PROV_TREAT_DESC'].fillna('').astype(str).str.strip()
        drug_df.loc[drug_df['DRUG_NAME'] == '', 'DRUG_NAME'] = drug_df['DRUG_CODE'].astype(str)

    # Add age group categories
    if 'PA_MEM_AGE' in drug_df.columns:
        drug_df['AGE_GROUP'] = drug_df['PA_MEM_AGE'].apply(_categorize_age)

    return drug_df