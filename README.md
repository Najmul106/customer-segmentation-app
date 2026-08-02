# 🛍️ TrendTribe: AI-Powered E-Commerce Customer Segmentation & Insights

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.38%2B-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-K--Means-F7931E.svg)](https://scikit-learn.org/)
[![Google Gemini](https://img.shields.io/badge/AI-Gemini%202.5%20Flash-4285F4.svg)](https://ai.google.dev/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**TrendTribe** is an enterprise-grade, end-to-end web application built using **Streamlit**, **Scikit-Learn**, and **Google Gemini AI**. It transforms raw e-commerce transaction datasets into actionable business intelligence through automated preprocessing, Recency-Frequency-Monetary (RFM) feature engineering, K-Means unsupervised clustering, dynamic Plotly visualizations, and AI-generated executive summaries.

---

## 🌟 Key Features

* **📥 Multi-Encoding Dataset Ingestion (`upload.py`)**
  * Supports automatic encoding resolution (`utf-8`, `latin1`, `cp1252`) and structural validation (row counts, duplicate checks, missing value summaries).

* **📊 Exploratory Data Analysis (`EDA.py`)**
  * Dynamic interactive scatter plots, box plot distributions, and correlation heatmaps.
  * One-click full HTML profiling reports powered by `ydata-profiling`.

* **🧹 Preprocessing & RFM Transformation (`preprocessing.py`)**
  * Flexible missing-value imputation (Mean, Median, Mode, Forward/Backward fill, or Explicit 'None').
  * Automated detection of transactional data and conversion to Customer-Level **RFM** metrics:
    * **Recency**: Days elapsed since last transaction.
    * **Frequency**: Count of distinct purchases/invoices.
    * **Monetary**: Total spend across transactions.
  * Automatic derivation of **Churn Risk** quantile buckets and **12-Month CLV** predictions.

* **🎯 Optimal Cluster Determination (`clustering.py`)**
  * Computes K-Means Inertia (Elbow Method) and **Silhouette Scores** across $k = 2 \dots 10$.
  * Algorithmically suggests the mathematically optimal cluster count ($k$).

* **🎨 Advanced Cluster Visualizations**
  * 3-panel 2D/3D Plotly comparative scatter plots ($M \text{ vs } R$, $F \text{ vs } R$, $M \text{ vs } F$).
  * Multi-dimensional **Radar Profiles** (normalized segment comparisons).

* **🤖 AI Intelligence Reports & Export (`reports.py` & `database.py`)**
  * Generates markdown executive reports or deep **Gemini 2.5 Flash** AI insights with segment naming and strategy recommendations.
  * Export capabilities: CSV download and persistent SQLite database saving (`outputs/ecommerce_segmentation.db`).

---

## 🏗️ Project Architecture

```text
customer_segmentation_app/
├── app.py              # Main Streamlit dashboard (6-Tab UI)
├── upload.py           # Dataset loading & validation
├── preprocessing.py    # Data cleaning & RFM transformation
├── EDA.py              # Exploratory data visualizers & profiling
├── clustering.py       # K-Means engine, Elbow, Silhouette, Radar charts
├── reports.py          # Markdown & Gemini AI report generators
├── database.py         # SQLite persistence module
├── requirements.txt    # Application dependencies
└── outputs/            # Persistent storage (Database, CSVs, Reports)
