"""
database.py - SQLite Database Integration for Data Persistence
"""
import sqlite3
import pandas as pd
import os

def save_results_to_db(customer_df: pd.DataFrame, cluster_profile: pd.DataFrame, db_name: str = "ecommerce_segmentation.db") -> str:
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    db_path = os.path.join(output_dir, db_name)
    conn = sqlite3.connect(db_path)
    
    try:
        customer_df.to_sql("Customer", conn, if_exists="replace", index=False)
        cluster_profile.to_sql("Cluster", conn, if_exists="replace", index=False)
        return db_path
    finally:
        conn.commit()
        conn.close()