import pandas as pd

def generate_markdown_report(dataset_name: str, k: int, profile_df: pd.DataFrame) -> str:
    md = f"""# 🏢 Enterprise Customer Segmentation Report

**Dataset Analysis:** `{dataset_name}`
**Model Used:** `K-Means Clustering`
**Optimal Segments Found:** `{k}`

---

## 📑 Executive Summary
This report outlines the customer segments generated exclusively using K-Means unsupervised machine learning. By analyzing these distinct mathematical groups, your business can tailor targeted marketing strategies, optimize resource allocation, reduce churn risk, and enhance overall Customer Lifetime Value (CLV).

## 📊 Segment Profiles & Characteristics

Below are the average feature values for each identified customer segment:

{profile_df.to_markdown(index=False)}

---

## 💡 Strategic Recommendations

*   🏆 **Identify High-Value Clusters:** Target the segment with the highest monetary value and transaction frequency for exclusive VIP loyalty programs.
*   ⚠️ **Re-engage At-Risk Customers:** Segments displaying high recency (dormant users) but solid historical value should immediately receive targeted win-back campaigns or retention discounts.
*   🌱 **Nurture New Leads:** Lower frequency, low recency users represent fresh acquisitions. Focus on onboarding sequences and first-time engagement offers to build habit loops.

---
*Generated automatically by the K-Means Insights Engine*
"""
    return md

def generate_ai_markdown_report(dataset_name: str, profile_df: pd.DataFrame, api_key: str) -> str:
    try:
        from google import genai
    except ImportError:
        raise RuntimeError("Run 'pip install google-genai' to use AI features.")
        
    if not api_key: raise RuntimeError("API Key is missing.")

    data_string = profile_df.to_markdown(index=False)
    prompt = f"""
    You are an expert E-Commerce Data Analyst. I have clustered a customer dataset named '{dataset_name}' exclusively using the K-Means algorithm.
    Here is the exact numerical profile for each K-Means cluster:
    {data_string}
    Based strictly on this data, write a highly professional, website-grade Markdown business report. Include:
    1. Executive Summary
    2. Segment Insights (Give each cluster an exciting business name based on its features)
    3. Actionable Marketing Recommendations per segment
    Use beautiful formatting, emojis, bolding, and bullet points.
    """
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
    return f"# 🤖 AI-Generated Intelligence Report\n\n{response.text}"