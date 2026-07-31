import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import streamlit as st

REQUIRED_COLS = {
    "invoice": ["invoiceno", "invoice"],
    "customer": ["customerid", "customer_id"],
    "date": ["invoicedate", "date"],
    "quantity": ["quantity", "qty"],
    "price": ["unitprice", "price", "unit_price"],
}

@st.cache_data(show_spinner=False)
def clean_data(df: pd.DataFrame, missing_strategy: str = "none") -> pd.DataFrame:
    df = df.copy().drop_duplicates()
    
    # 1. Numerical Casting: Convert Object columns to Numeric (int/float) where possible
    for col in df.select_dtypes(include=['object']).columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except (ValueError, TypeError):
            pass 
            
    # 2. Temporal Standardization: Format Date columns to ISO 8601
    for col in df.columns:
        if df[col].dtype == 'object':
            if 'date' in col.lower() or 'time' in col.lower():
                df[col] = pd.to_datetime(df[col], errors='coerce').dt.strftime('%Y-%m-%d')

    # 3. Missing Value Imputation: Implement explicit 'None' strategy
    if missing_strategy == "none":
        df = df.fillna('None')
    else:
        for col in df.columns:
            if df[col].isna().sum() > 0:
                if missing_strategy == "b-fill": df[col] = df[col].bfill()
                elif missing_strategy == "f-fill": df[col] = df[col].ffill()
                elif missing_strategy == "mode":
                    df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else np.nan)
                elif missing_strategy == "mean" and pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(df[col].mean())
                elif missing_strategy == "median" and pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(df[col].median())
                else:
                    df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown")
    return df

def _find_col(df: pd.DataFrame, candidates: list) -> str | None:
    lower_map = {c.lower().replace(" ", "").replace("_", ""): c for c in df.columns}
    for cand in candidates:
        key = cand.replace("_", "")
        if key in lower_map: return lower_map[key]
    return None

def process_rfm_if_transactional(df: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    cols = {key: _find_col(df, candidates) for key, candidates in REQUIRED_COLS.items()}
    if not all(cols.values()) or df[cols["customer"]].nunique() == len(df):
        return df, False

    work = df.dropna(subset=[cols["date"], cols["customer"]]).copy()
    is_return = work[cols["invoice"]].astype(str).str.startswith("C") | (work[cols["quantity"]] <= 0)
    work = work[~is_return]
    work["LineTotal"] = work[cols["quantity"]] * work[cols["price"]]
    
    work[cols["date"]] = pd.to_datetime(work[cols["date"]]).dt.normalize()
    reference_date = pd.Timestamp.now().normalize()
    
    agg = work.groupby(cols["customer"]).agg(
        LastPurchaseDate=(cols["date"], "max"),
        Frequency=(cols["invoice"], "nunique"),
        Monetary=("LineTotal", "sum"),
    ).reset_index()

    agg["Recency"] = (reference_date - agg["LastPurchaseDate"]).dt.days
    agg["Monetary"] = agg["Monetary"].round(2)
    agg = agg.rename(columns={cols["customer"]: "CustomerID"})
    
    agg["Churn_Risk"] = pd.qcut(agg["Recency"].rank(method='first'), q=3, labels=["Low", "Medium", "High"])
    agg["CLV_12_Months"] = (agg["Monetary"] * 1.20).round(2)
    
    return agg, True

def scale_features(df: pd.DataFrame, feature_cols: list):
    # Filter out 'None' strings before scaling if they exist in numeric feature columns
    X = df[feature_cols].replace('None', 0).copy().fillna(0)
    scaler = StandardScaler()
    return scaler.fit_transform(X), scaler