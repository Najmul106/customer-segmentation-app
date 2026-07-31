"""
upload.py - Dataset Loading and Structural Validation
"""
import pandas as pd
import numpy as np
import streamlit as st

@st.cache_data(show_spinner=False)
def load_csv(file_or_path) -> pd.DataFrame:
    encodings = ["utf-8", "utf-8-sig", "latin1", "cp1252"]
    is_upload = hasattr(file_or_path, "read")
    last_error = None

    for enc in encodings:
        try:
            if is_upload: file_or_path.seek(0)
            return pd.read_csv(file_or_path, encoding=enc)
        except UnicodeDecodeError as e:
            last_error = e
        except Exception as e:
            try:
                if is_upload: file_or_path.seek(0)
                return pd.read_csv(file_or_path, encoding=enc, sep=None, engine="python")
            except Exception as e2:
                last_error = e2
    raise ValueError(f"Failed to read CSV. Last error: {last_error}")

def validate_dataset(df: pd.DataFrame) -> dict:
    missing_counts = df.isna().sum()
    missing_cols = missing_counts[missing_counts > 0].to_dict()
    return {
        "n_rows": len(df),
        "n_cols": df.shape[1],
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_summary": missing_cols,
        "total_missing": int(missing_counts.sum())
    }