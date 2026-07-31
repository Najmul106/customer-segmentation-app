"""
app.py - Main Streamlit Dashboard (6 Tabs Structure)
"""
import streamlit as st
import pandas as pd
import upload
import EDA
import preprocessing
import clustering
import reports
import database

st.set_page_config(layout="wide", page_title="TrendTribe")
st.title("🛍️ TrendTribe")
st.markdown("#### *AI-Powered E-Commerce Customer Segmentation & Insights*")

# State Management (Removed algo_comparison)
for key in ["raw_df", "clean_df", "clustered_df", "metrics_df", "best_k", "feature_cols", "cluster_profile"]:
    if key not in st.session_state:
        st.session_state[key] = None

tabs = st.tabs([
    "1. Upload & Overview", 
    "2. Exploratory Data Analysis", 
    "3. Preprocessing & RFM", 
    "4. Determine Optimal K", 
    "5. Cluster Visualization", 
    "6. Reports & Export"
])

# ================= TAB 1 =================
with tabs[0]:
    st.header("1. Dataset Upload")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        df = upload.load_csv(uploaded_file)
        st.session_state.raw_df = df
        st.session_state.dataset_name = uploaded_file.name
        
    if st.session_state.raw_df is not None:
        st.subheader("Data Preview")
        st.dataframe(st.session_state.raw_df.head(10))
        report = upload.validate_dataset(st.session_state.raw_df)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", report["n_rows"])
        c2.metric("Columns", report["n_cols"])
        c3.metric("Duplicates", report["duplicate_rows"])
        c4.metric("Missing Values", report["total_missing"])

# ================= TAB 2 =================
with tabs[1]:
    st.header("2. Exploratory Data Analysis")
    if st.session_state.raw_df is not None:
        df = st.session_state.raw_df
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        if st.button("Generate Full EDA HTML Report"):
            with st.spinner("Generating Report..."):
                html_report = EDA.generate_html_report(df)
                if html_report:
                    st.download_button("⬇️ Download Full EDA Report (.html)", html_report, "EDA_Report.html", "text/html")
                else:
                    st.error("Please run: pip install ydata-profiling")
        st.divider()
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Dynamic Scatter Plot")
            x_col = st.selectbox("X-Axis", numeric_cols, index=0)
            y_col = st.selectbox("Y-Axis", numeric_cols, index=min(1, len(numeric_cols)-1))
            hue_col = st.selectbox("Hue (Color)", ["None"] + df.columns.tolist())
            hue = None if hue_col == "None" else hue_col
            st.plotly_chart(EDA.plot_dynamic_scatter(df, x_col, y_col, hue), width="stretch")
            
        with col2:
            st.subheader("Box Plot")
            box_y = st.selectbox("Analyze Distribution of:", numeric_cols)
            st.plotly_chart(EDA.plot_box(df, box_y), width="stretch")
            
        st.subheader("Correlation Heatmap")
        st.plotly_chart(EDA.plot_correlation(df), width="stretch")
    else:
        st.info("Upload data in Tab 1 first.")

# ================= TAB 3 =================
with tabs[2]:
    st.header("3. Preprocessing & Feature Engineering")
    if st.session_state.raw_df is not None:
        # Added "none" to the selectbox options
        strategy = st.selectbox("Handle Missing Values Method:", ["none", "mean", "median", "mode", "b-fill", "f-fill"])
        if st.button("Apply Preprocessing & Check RFM"):
            cleaned = preprocessing.clean_data(st.session_state.raw_df, strategy)
            processed, is_rfm = preprocessing.process_rfm_if_transactional(cleaned)
            st.session_state.clean_df = processed
            if is_rfm: st.success("Converted transactional data to Customer-Level RFM!")
            else: st.success(f"Data cleaned using {strategy} method.")
                
        if st.session_state.clean_df is not None:
            st.dataframe(st.session_state.clean_df.head(10))
    else:
        st.info("Upload data in Tab 1 first.")

# ================= TAB 4 =================
with tabs[3]:
    st.header("4. Determine Optimal Clusters (k)")
    if st.session_state.clean_df is not None:
        df = st.session_state.clean_df
        num_cols = [c for c in df.select_dtypes(include=['number']).columns if 'id' not in c.lower()]
        selected_features = st.multiselect("Select Features:", num_cols, default=num_cols)
        
        if len(selected_features) >= 2 and st.button("Calculate Metrics"):
            scaled_X, _ = preprocessing.scale_features(df, selected_features)
            st.session_state.scaled_features = scaled_X
            st.session_state.feature_cols = selected_features
            
            metrics = clustering.compute_cluster_metrics(scaled_X)
            st.session_state.metrics_df = metrics
            st.session_state.best_k = clustering.suggest_best_k(metrics)
            
        if st.session_state.metrics_df is not None:
            st.plotly_chart(clustering.plot_elbow(st.session_state.metrics_df), width="stretch")
            st.info(f"💡 Suggested k based on K-Means Silhouette Score: **{st.session_state.best_k}**")
    else:
        st.info("Complete preprocessing in Tab 3 first.")

# ================= TAB 5 =================
with tabs[4]:
    st.header("5. Run Clustering & Visualize")
    if st.session_state.metrics_df is not None:
        k = st.slider("Select Number of Clusters (k)", 2, 10, st.session_state.best_k)
        
        if st.button("Run K-Means"):
            km, labels = clustering.run_kmeans(st.session_state.scaled_features, k)
            clustered = st.session_state.clean_df.copy()
            clustered["Cluster"] = labels
            st.session_state.clustered_df = clustered
            st.session_state.cluster_profile = clustering.profile_clusters(clustered, st.session_state.feature_cols)
            st.success("Clustering complete!")
            
        if st.session_state.clustered_df is not None:
            feat = st.session_state.feature_cols
            st.subheader("RFM Combination Analysis")
            if len(feat) >= 3:
                fig_rfm = clustering.plot_rfm_subplots(st.session_state.clustered_df, feat)
                if fig_rfm: st.plotly_chart(fig_rfm, width="stretch")
                else: st.info("Could not generate RFM plots.")
            else:
                st.info("Select 3 or more features in Tab 4 to enable the 3-panel visualization.")
            
            st.divider()
            if len(feat) >= 3:
                st.subheader("Segment Radar Profiles")
                st.plotly_chart(clustering.plot_radar_chart(st.session_state.clustered_df, feat), width="stretch")
    else:
        st.info("Find optimal k in Tab 4 first.")

# ================= TAB 6 =================
with tabs[5]:
    st.header("6. Generate Reports & Export Data")
    if st.session_state.cluster_profile is not None:
        st.subheader("Cluster Profile (Feature Means)")
        st.dataframe(st.session_state.cluster_profile)
        
        rep_type = st.radio("Report Format:", ["Normal", "AI Generated"])
        api_key = st.text_input("Gemini API Key (For AI Report):", type="password") if rep_type == "AI Generated" else None
        
        if st.button("Generate Report"):
            try:
                if rep_type == "Normal":
                    report_md = reports.generate_markdown_report(st.session_state.dataset_name, len(st.session_state.cluster_profile), st.session_state.cluster_profile)
                else:
                    with st.spinner("AI is analyzing cluster data..."):
                        report_md = reports.generate_ai_markdown_report(st.session_state.dataset_name, st.session_state.cluster_profile, api_key)
                st.download_button("Download Report (.md)", report_md, "report.md", "text/markdown")
                with st.expander("Preview Report", expanded=True):
                    st.markdown(report_md)
            except Exception as e:
                st.error(str(e))
                
        st.divider()
        st.subheader("Data Export")
        c1, c2 = st.columns(2)
        with c1:
            if st.session_state.clustered_df is not None:
                csv_data = st.session_state.clustered_df.to_csv(index=False).encode('utf-8')
                st.download_button("⬇️ Download Clustered Data (CSV)", csv_data, "clustered_data.csv", "text/csv")
        with c2:
            if st.button("💾 Save Results to SQLite Database"):
                try:
                    db_path = database.save_results_to_db(st.session_state.clustered_df, st.session_state.cluster_profile)
                    st.success(f"Successfully saved to {db_path}!")
                except Exception as e:
                    st.error(f"Failed to save to database: {e}")
    else:
        st.info("Run clustering in Tab 5 first.")