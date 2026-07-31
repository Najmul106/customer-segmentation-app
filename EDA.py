"""
EDA.py - Exploratory Data Analysis Visualization & Reports
"""
import pandas as pd
import plotly.express as px
import numpy as np

def plot_dynamic_scatter(df: pd.DataFrame, x_col: str, y_col: str, hue_col: str = None):
    return px.scatter(df, x=x_col, y=y_col, color=hue_col, title=f"{y_col} vs {x_col}", opacity=0.7)

def plot_box(df: pd.DataFrame, y_col: str, x_col: str = None):
    return px.box(df, x=x_col, y=y_col, title=f"Box Plot of {y_col}")

def plot_correlation(df: pd.DataFrame):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr().round(2)
        return px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", title="Correlation Heatmap")
    return None

def generate_html_report(df: pd.DataFrame):
    try:
        from ydata_profiling import ProfileReport
        profile = ProfileReport(df, minimal=True, title="EDA Profiling Report")
        return profile.to_html()
    except ImportError:
        return None