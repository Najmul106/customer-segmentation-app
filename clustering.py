import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def compute_cluster_metrics(X: np.ndarray, k_max: int = 10) -> pd.DataFrame:
    rows = []
    actual_k_max = min(k_max, len(X) - 1)
    if actual_k_max < 2: return pd.DataFrame([{"k": 2, "inertia": 0, "silhouette": 0}])
        
    for k in range(2, actual_k_max + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(X)
        if len(X) > 10000:
            sil_score = silhouette_score(X, labels, sample_size=10000, random_state=42)
        else:
            sil_score = silhouette_score(X, labels)
        rows.append({"k": k, "inertia": km.inertia_, "silhouette": sil_score})
    return pd.DataFrame(rows)

def suggest_best_k(metrics_df: pd.DataFrame) -> int:
    return int(metrics_df.loc[metrics_df["silhouette"].idxmax(), "k"])

def run_kmeans(X: np.ndarray, k: int) -> tuple:
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    return km, km.fit_predict(X)

def profile_clusters(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    profile = df.groupby("Cluster")[feature_cols].mean().round(2)
    profile["CustomerCount"] = df.groupby("Cluster").size()
    return profile.reset_index()

def plot_elbow(metrics_df: pd.DataFrame):
    return px.line(metrics_df, x="k", y="inertia", markers=True, title="Elbow Method (K-Means Exclusive)")

def plot_cluster_scatter(df: pd.DataFrame, x_col: str, y_col: str):
    return px.scatter(df, x=x_col, y=y_col, color=df["Cluster"].astype(str), title="2D Cluster Scatter")

def plot_rfm_subplots(df: pd.DataFrame, feature_cols: list, cluster_col: str = "Cluster"):
    r_col = next((c for c in feature_cols if 'recency' in c.lower()), None)
    f_col = next((c for c in feature_cols if 'freq' in c.lower() or 'orders' in c.lower()), None)
    m_col = next((c for c in feature_cols if 'monetary' in c.lower() or 'spend' in c.lower() or 'value' in c.lower()), None)
    
    if not (r_col and f_col and m_col) and len(feature_cols) >= 3:
        r_col, f_col, m_col = feature_cols[0], feature_cols[1], feature_cols[2]
        
    if r_col and f_col and m_col:
        fig = make_subplots(
            rows=1, cols=3, 
            subplot_titles=(f"{m_col.title()} vs {r_col.title()}", f"{f_col.title()} vs {r_col.title()}", f"{m_col.title()} vs {f_col.title()}"),
            horizontal_spacing=0.1
        )
        clusters = sorted(df[cluster_col].unique())
        colors = px.colors.qualitative.Set2 

        for i, cluster in enumerate(clusters):
            cluster_data = df[df[cluster_col] == cluster]
            color = colors[i % len(colors)]
            name = f"Cluster {cluster}"
            
            fig.add_trace(go.Scatter(x=cluster_data[r_col], y=cluster_data[m_col], mode='markers', marker=dict(size=9, color=color, opacity=0.8, line=dict(width=1, color='white')), name=name, legendgroup=name), row=1, col=1)
            fig.add_trace(go.Scatter(x=cluster_data[r_col], y=cluster_data[f_col], mode='markers', marker=dict(size=9, color=color, opacity=0.8, line=dict(width=1, color='white')), name=name, legendgroup=name, showlegend=False), row=1, col=2)
            fig.add_trace(go.Scatter(x=cluster_data[f_col], y=cluster_data[m_col], mode='markers', marker=dict(size=9, color=color, opacity=0.8, line=dict(width=1, color='white')), name=name, legendgroup=name, showlegend=False), row=1, col=3)

        fig.update_layout(height=450, plot_bgcolor='white', legend_title_text="ClusterName", margin=dict(t=50, b=50, l=50, r=50))
        return fig
    return None

def plot_radar_chart(df: pd.DataFrame, feature_cols: list):
    means = df.groupby("Cluster")[feature_cols].mean()
    norm_means = (means - means.min()) / (means.max() - means.min() + 1e-9)
    fig = go.Figure()
    for cluster_id in norm_means.index:
        fig.add_trace(go.Scatterpolar(
            r=norm_means.loc[cluster_id].values.tolist() + [norm_means.loc[cluster_id].values[0]],
            theta=feature_cols + [feature_cols[0]], fill="toself", name=f"Cluster {int(cluster_id)}"
        ))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), title="Segment Profiles")
    return fig