import pandas as pd
import numpy as np
import re
from io import BytesIO
from dateutil import parser
from collections import Counter
from ai_service import intelligent_structure_mapping

# Expected schemas
EXPECTED_PAYMENT_COLS = ["Sub Order No", "Live Order Status", "Final Settlement Amount"]
# Added 'Order Date' to help identify the month
EXPECTED_ORDER_COLS = ["Sub Order No", "SKU", "Quantity", "Product Name", "Order Date"]
EXPECTED_COST_COLS = ["SKU", "Cost", "Product Name"]

def detect_file_month(file, df=None):
    """
    Attempts to guess the month/year of a file based on:
    1. Filename (e.g. "Jan Payment.xlsx")
    2. Data (Date columns)
    Returns: (year, month) tuple or None
    """
    # 1. Filename Search (Order matters: YYYY-MM is stronger than just 'jan')
    filename = file.name.lower()
    
    # Check for YYYY-MM pattern (e.g., 2025-09)
    # matches 2020-2029 dash 01-12
    ym_match = re.search(r'(202[0-9])[-_](0[1-9]|1[0-2])', filename)
    if ym_match:
        return (int(ym_match.group(1)), int(ym_match.group(2)))

    # Check for Month names
    months = {
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
        'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
    }
    for m_str, m_int in months.items():
        if m_str in filename:
            # Try to find year? Assume current or look for 202x
            import datetime
            year = datetime.datetime.now().year
            # find 202x in filename
            yr_match = re.search(r'202[0-9]', filename)
            if yr_match:
                year = int(yr_match.group(0))
            return (year, m_int)

    # 2. Data Search
    if df is not None:
        # Look for 'Order Date' or similar
        date_cols = [c for c in df.columns if 'date' in str(c).lower()]
        if date_cols:
            col = date_cols[0]
            try:
                # parser.parse might be slow for whole series, take sample
                sample = df[col].dropna().head(50).astype(str) # Increased sample size
                dates = []
                for d in sample:
                    try:
                        dt = parser.parse(d, fuzzy=True, dayfirst=True)
                        dates.append(dt)
                    except:
                        pass
                
                if dates:
                    # Most common month
                    path_counts = Counter([(d.year, d.month) for d in dates])
                    most_common = path_counts.most_common(1)[0][0]
                    return most_common
            except:
                pass

    return None

def smart_read_with_ai(file, expected_cols, description, sheet_name=None):
    """
    Reads a file using AI to detect structure.
    Returns: (DataFrame, LogMessage)
    """
    log = []
    
    # 1. Read Raw Preview
    file.seek(0)
    try:
        if file.name.endswith('.csv'):
            df_raw = pd.read_csv(file, header=None, nrows=20)
        else:
            try:
                sheet = sheet_name if sheet_name else 0
                df_raw = pd.read_excel(file, sheet_name=sheet, header=None, nrows=20)
            except:
                file.seek(0)
                df_raw = pd.read_excel(file, sheet_name=0, header=None, nrows=20)
    except Exception as e:
        return pd.DataFrame(), f"❌ Failed to read raw file {file.name}: {e}"

    # 2. Ask AI Structure (only if simple check fails)
    # Optimization: Check if headers already exist in row 0 or 1
    # This saves tokens.
    
    def check_header_match(row_vals):
        row_str = " ".join([str(x).lower() for x in row_vals])
        # Check if at least 2 expected cols are present
        matches = sum(1 for c in expected_cols if c.lower() in row_str)
        return matches >= min(2, len(expected_cols))

    header_idx = -1
    mapping = {}
    
    # Simple Heuristic Loop
    for i, row in df_raw.iterrows():
        if check_header_match(row.values):
            header_idx = i
            break
            
    # If heuristic failed, Use AI
    if header_idx == -1:
        csv_preview = df_raw.to_csv(index=False)
        structure = intelligent_structure_mapping(csv_preview, expected_cols, description)
        
        if structure:
            header_idx = structure.get("header_row_index", 0)
            mapping = structure.get("column_mapping", {})
            log.append(f"🤖 AI found header in '{file.name}' at row {header_idx}")
            if mapping:
                log.append(f"🤖 AI Mapped columns: {mapping}")
    
    if header_idx == -1:
        header_idx = 0 # Fallback
        log.append(f"⚠️ Could not detect header for {file.name}, assuming row 0.")

    # 3. Reload Full File with Header
    file.seek(0)
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file, header=header_idx)
        else:
            sheet = sheet_name if sheet_name else 0
            df = pd.read_excel(file, sheet_name=sheet, header=header_idx)
            
        # 4. Apply Mapping
        if mapping:
            df.rename(columns=mapping, inplace=True)
            
        # Deduplicate columns if any
        df = df.loc[:, ~df.columns.duplicated()]
            
        return df, "\n".join(log)
        
    except Exception as e:
        return pd.DataFrame(), f"❌ Error re-loading {file.name}: {e}"


def process_data(orders_files, payment_files, cost_file, packaging_cost, misc_cost):
    logs = []
    
    # --- 1. Process Orders (Multiple Files) ---
    all_orders = []
    ref_month = None
    
    # Ensure lists
    if not isinstance(orders_files, list): orders_files = [orders_files]
    if not isinstance(payment_files, list): payment_files = [payment_files]

    for f in orders_files:
        df, msg = smart_read_with_ai(f, EXPECTED_ORDER_COLS, "Orders File")
        if msg: logs.append(msg)
        if not df.empty:
            all_orders.append(df)
            # Try detect month from first file
            if ref_month is None:
                ref_month = detect_file_month(f, df)
                if ref_month: logs.append(f"📅 Detected Reference Order Month: {ref_month}")

    if not all_orders:
         return None, None, None, ["❌ No valid order data found."]
    
    df_orders = pd.concat(all_orders, ignore_index=True)
    
    # Clean Orders
    cols_to_keep = [c for c in EXPECTED_ORDER_COLS if c in df_orders.columns]
    df_orders = df_orders[cols_to_keep].copy()
    
    if 'Quantity' in df_orders.columns:
        df_orders['Quantity'] = pd.to_numeric(df_orders['Quantity'], errors='coerce').fillna(0)
    else:
        df_orders['Quantity'] = 1

    # --- 2. Process Payments & Ads (Multiple Files) ---
    # We need to bucket them into "Same Month" vs "Next Month"
    # Logic: 
    #   Same Month = ref_month
    #   Next Month = (ref_month.month + 1)
    
    # Containers
    dfs_same = []
    dfs_next = []
    same_ads_sum = 0
    next_ads_sum = 0
    
    def get_ads_cost_from_file(file):
        try:
            file.seek(0)
            df = pd.read_excel(file, sheet_name='Ads Cost')
            num_df = df.select_dtypes(include=[np.number])
            if not num_df.empty:
                return num_df.iloc[:, -1].sum() 
            return 0
        except:
            return 0

    for pf in payment_files:
        # Read Data
        df_pay, msg = smart_read_with_ai(pf, EXPECTED_PAYMENT_COLS, "Payment File", sheet_name='Order Payments')
        if msg: logs.append(msg)
        
        # Detect Month
        f_month = detect_file_month(pf, df_pay)
        is_next_month = False
        
        if ref_month and f_month:
            # Compare
            # Simple check: is it same year/month?
            if f_month == ref_month:
                is_next_month = False
            elif f_month > ref_month: # Basic tuition, assumes next
                is_next_month = True
            else:
                 # It's an older file? Treat as Same or Ignore? Treat as Same for now to be safe
                 is_next_month = False
        else:
             # Fallback: if we have 2 payment files, and this is the second one?
             # No, unreliable. Just default to 'Same' if unknown, or maybe check filename for "Next"
             if "next" in pf.name.lower():
                 is_next_month = True
             else:
                 is_next_month = False
        
        bucket = "Next Month" if is_next_month else "Same Month"
        logs.append(f"📂 Categorized '{pf.name}' as -> {bucket}")

        # Ads Cost
        ads = get_ads_cost_from_file(pf)
        
        if is_next_month:
            dfs_next.append(df_pay)
            next_ads_sum += ads
        else:
            dfs_same.append(df_pay)
            same_ads_sum += ads

    # Combine Payment Dataframes
    def aggregate_payments(dfs):
        if not dfs: return pd.DataFrame(columns=['Sub Order No', 'Amount', 'Live Order Status'])
        combined = pd.concat(dfs, ignore_index=True)
        
        # Normalize
        if "Final Settlement Amount" in combined.columns:
            combined['Final Settlement Amount'] = pd.to_numeric(combined['Final Settlement Amount'], errors='coerce').fillna(0)
            combined.rename(columns={'Final Settlement Amount': 'Amount'}, inplace=True)
        return combined

    # Process Same
    df_same_comb = aggregate_payments(dfs_same)
    if not df_same_comb.empty and 'Sub Order No' in df_same_comb.columns:
        df_same_pivot = df_same_comb.groupby('Sub Order No')['Amount'].sum().reset_index().rename(columns={'Amount': 'same month pay'})
    else:
        df_same_pivot = pd.DataFrame(columns=['Sub Order No', 'same month pay'])
        
    # Process Next
    df_next_comb = aggregate_payments(dfs_next)
    if not df_next_comb.empty and 'Sub Order No' in df_next_comb.columns:
        df_next_pivot = df_next_comb.groupby('Sub Order No')['Amount'].sum().reset_index().rename(columns={'Amount': 'next month pay'})
    else:
        df_next_pivot = pd.DataFrame(columns=['Sub Order No', 'next month pay'])
        
    # Status (Combine both)
    all_status_dfs = []
    if 'Live Order Status' in df_same_comb.columns: all_status_dfs.append(df_same_comb[['Sub Order No', 'Live Order Status']])
    if 'Live Order Status' in df_next_comb.columns: all_status_dfs.append(df_next_comb[['Sub Order No', 'Live Order Status']])
    
    if all_status_dfs:
        df_status = pd.concat(all_status_dfs).drop_duplicates('Sub Order No', keep='last')
    else:
        df_status = pd.DataFrame(columns=['Sub Order No', 'Live Order Status'])

    # --- 3. Process Cost ---
    df_cost, msg_c = smart_read_with_ai(cost_file, EXPECTED_COST_COLS, "Cost Sheet")
    if msg_c: logs.append(msg_c)
    
    if "SKU" in df_cost.columns and "Cost" in df_cost.columns:
        df_cost['SKU'] = df_cost['SKU'].astype(str).str.strip()
        df_cost['Cost'] = pd.to_numeric(df_cost['Cost'], errors='coerce').fillna(0)
    else:
        df_cost = pd.DataFrame(columns=['SKU', 'Cost'])

    # --- 4. Merge Logic (Reference Implementation) ---
    if 'Sub Order No' not in df_orders.columns:
        return None, None, None, ["❌ Critical: 'Sub Order No' not found in orders."]

    # A. Prepare Base Orders
    df_orders_final = df_orders.copy()
    if 'Quantity' in df_orders_final.columns:
        df_orders_final['Quantity'] = pd.to_numeric(df_orders_final['Quantity'], errors='coerce').fillna(0)
    else:
        df_orders_final['Quantity'] = 1
        
    df_orders_final['Sub Order No'] = df_orders_final['Sub Order No'].astype(str).str.strip()
    
    # B. Merge Payments
    df_orders_final = df_orders_final.merge(df_same_pivot, on='Sub Order No', how='left')
    df_orders_final = df_orders_final.merge(df_next_pivot, on='Sub Order No', how='left')
    df_orders_final['total'] = df_orders_final['same month pay'].fillna(0) + df_orders_final['next month pay'].fillna(0)

    # C. Merge Status
    df_status['Sub Order No'] = df_status['Sub Order No'].astype(str).str.strip()
    # Ensure uniqueness in status to prevent explosion
    status_lookup = df_status.drop_duplicates(subset=['Sub Order No'], keep='last')
    
    df_orders_final = df_orders_final.merge(status_lookup, on='Sub Order No', how='left')
    df_orders_final.rename(columns={'Live Order Status': 'status'}, inplace=True)
    
    # D. COST LOGIC (Exact Reference Mirror)
    # The reference code uses iloc[:, :2] assuming first col is SKU, second is Cost.
    # We will try to respect that if our named search failed, or just use our named df if it worked.
    
    # Create clean lookup (Copy reference logic: "cost_lookup.columns = ['SKU_Lookup', 'Cost_Value']")
    if not df_cost.empty and 'SKU' in df_cost.columns and 'Cost' in df_cost.columns:
        cost_lookup = df_cost[['SKU', 'Cost']].copy()
    else:
        # Fallback to iloc if names matched poorly
        cost_lookup = df_cost.iloc[:, :2].copy()
        
    cost_lookup.columns = ['SKU_Lookup', 'Cost_Value']
    
    # String Conversion (Exact Reference)
    df_orders_final['SKU'] = df_orders_final['SKU'].astype(str).str.strip()
    cost_lookup['SKU_Lookup'] = cost_lookup['SKU_Lookup'].astype(str).str.strip()
    
    # Drop duplicates in lookup to prevent explode
    cost_lookup = cost_lookup.drop_duplicates(subset=['SKU_Lookup'])

    # Merge
    df_orders_final = pd.merge(df_orders_final, cost_lookup, left_on='SKU', right_on='SKU_Lookup', how='left')
    
    # ----------------------------------------------------
    # IDENTIFY MISSING SKUS & PREPARE DETAILS
    # ----------------------------------------------------
    missing_cost_mask = df_orders_final['Cost_Value'].isna()
    
    # Create the Detail Dataframe for the Dashboard
    cols_for_missing = ['Sub Order No', 'SKU', 'status', 'Quantity', 'total']
    # ensure cols exist
    curr_cols = [c for c in cols_for_missing if c in df_orders_final.columns]
    
    missing_details_df = df_orders_final.loc[missing_cost_mask, curr_cols].copy()
    if 'total' in missing_details_df.columns:
        missing_details_df.rename(columns={'total': 'Total Payment'}, inplace=True)

    # Fill NaN with 0 temporarily for Calculation
    df_orders_final['Cost_Value'] = df_orders_final['Cost_Value'].fillna(0)

    # 1. Product Cost Calculation (Only for Delivered and Exchange)
    # Ensure status column is clean
    df_orders_final['status'] = df_orders_final['status'].fillna('Unknown').astype(str).str.strip()
    condition_product = df_orders_final['status'].isin(['Delivered', 'Exchange'])

    # Calculate numeric cost
    df_orders_final['cost'] = np.where(condition_product, df_orders_final['Cost_Value'], 0)
    df_orders_final['actual cost'] = df_orders_final['cost'] * df_orders_final['Quantity']
    
    # 2. Packaging Cost Calculation
    condition_packaging = df_orders_final['status'].isin(['Delivered', 'Exchange', 'Return'])
    df_orders_final['packaging cost'] = np.where(condition_packaging, packaging_cost, 0)
    
    if 'SKU_Lookup' in df_orders_final.columns:
        df_orders_final.drop(columns=['SKU_Lookup', 'Cost_Value'], inplace=True)

    # --- Calculate Final Stats ---
    total_payment_sum = df_orders_final['total'].sum()
    total_product_cost = df_orders_final['actual cost'].sum()
    total_packaging = df_orders_final['packaging cost'].sum()
    profit = total_payment_sum - total_product_cost - total_packaging - abs(same_ads_sum) - abs(next_ads_sum) - misc_cost

    status_series = df_orders_final['status']
    stats = {
        "Total Payments": total_payment_sum,
        "Total Product Cost": total_product_cost,
        "Total Packaging Cost": total_packaging,
        "Ads Cost (Same Month)": same_ads_sum,
        "Ads Cost (Next Month)": next_ads_sum,
        "Miscellaneous Cost": misc_cost,
        "Profit / Loss": profit,
        "Order Count": len(df_orders_final),
        "count_delivered": len(df_orders_final[status_series == 'Delivered']),
        "count_return": len(df_orders_final[status_series == 'Return']),
        "count_rto": len(df_orders_final[status_series == 'RTO']),
        "count_exchange": len(df_orders_final[status_series == 'Exchange']),
        "count_cancelled": len(df_orders_final[status_series == 'Cancelled']),
        "count_shipped": len(df_orders_final[status_series == 'Shipped']),
        "count_ready_to_ship": len(df_orders_final[status_series == 'Ready_to_ship'])
    }
    
    # EXPORT PREP: Replace 0 with "SKU Not Found" (Visual only)
    # We create a display version for excel
    df_export = df_orders_final.copy()
    # We cast to object to allow string "SKU Not Found"
    df_export['cost'] = df_export['cost'].astype(object)
    df_export['actual cost'] = df_export['actual cost'].astype(object)
    
    # Re-evaluate condition for display
    # Note: missing_cost_mask indices match df_export since it is a copy
    condition_display_error = missing_cost_mask & condition_product
    df_export.loc[condition_display_error, 'cost'] = "SKU Not Found"
    df_export.loc[condition_display_error, 'actual cost'] = "SKU Not Found"
    
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df_export.to_excel(writer, sheet_name='Result', index=False)
        pd.DataFrame([stats]).T.to_excel(writer, sheet_name='Overview')
    output.seek(0)
    
    return output, stats, missing_details_df, logs
