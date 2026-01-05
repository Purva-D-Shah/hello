import os
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
import json
import streamlit as st

load_dotenv()

client = None
try:
    # Try getting key from environment (local) or Streamlit secrets (cloud)
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key and "OPENAI_API_KEY" in st.secrets:
        api_key = st.secrets["OPENAI_API_KEY"]
        
    if api_key:
        client = OpenAI(api_key=api_key)
except Exception as e:
    print(f"OpenAI Client Init Error: {e}")

def get_openai_client():
    return client

def analyze_profit_loss(stats_dict):
    """
    Sends the calculated statistics to OpenAI for a qualitative analysis.
    """
    if not client:
        return "⚠️ AI Analysis Unavailable (Add API Key)"

    prompt = f"""
    You are a financial analyst for a Meesho seller. 
    Analyze the following profit/loss statement and provide a brief, professional summary.
    Highlight the profit margin, major cost drivers, and suggestive actions.
    
    Data:
    {json.dumps(stats_dict, indent=2)}
    
    Output Format:
    - **Executive Summary**
    - **Key Metrics Analysis** (Focus on Returns/RTO impact if high)
    - **Recommendations**
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful financial assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error connecting to OpenAI: {str(e)}"

def intelligent_structure_mapping(df_head_csv, expected_columns, file_description):
    """
    Uses OpenAI to analyze the raw top rows of a file to find the header row and map columns.
    Returns: json { "header_row_index": int, "col_mapping": dict }
    """
    if not client:
        # Fallback if no AI
        return None

    prompt = f"""
    I have a file ({file_description}) that might have changed format.
    Here are the first 15 rows of the data (raw):
    
    {df_head_csv}
    
    My code expects to find these columns (or similar concepts): {expected_columns}.
    
    Task:
    1. Identify which row (0-indexed) seems to be the HEADER row containing the column names.
    2. Map the actual column names in that row to my expected column names.
    
    Return ONLY valid JSON:
    {{
        "header_row_index": <int or -1 if no header>,
        "column_mapping": {{ "<actual_col_name_found_in_file>": "<expected_col_name>" }}
    }}
    Important: Only map columns if you are confident they match the concept.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a Data Engineer expert in parsing messy Excel/CSV files."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        result = json.loads(response.choices[0].message.content)
        return result
    except Exception as e:
        print(f"AI Parse failed: {e}")
        return None
