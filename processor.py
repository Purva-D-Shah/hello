import pandas as pd
import numpy as np
import re
from io import BytesIO
from dateutil import parser
from collections import Counter
from ai_service import intelligent_structure_mapping

# Expected schemas
EXPECTED_PAYMENT_COLS = ["Sub Order No", "Live Order Status", "Final Settlement Amount"]
EXPECTED_ORDER_COLS = ["Sub Order No", "SKU", "Quantity", "Product Name", "Order Date"]
EXPECTED_COST_COLS = ["SKU", "Cost", "Product Name"]

def detect_file_month(file, df=None):
    filename = file.name.lower()
    ym_match = re.search(r'(202[0-9])[-_](0[1-9]|1[0-2])', filename)
    if ym_match:
        return (int(ym_match.group(1)), int(ym_match.group(2)))
    months = {
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
        'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
    }
    for m_str, m_int in months.items():
        if m_str in filename:
            import datetime
            year = datetime.datetime.now().year
            yr_match = re.search(r'202[0-9]', filename)
            if yr_match:
                year = int(yr_match.group(0))
            return (year, m_int)
    if df is not None:
        date_cols = [c for c in df.columns if 'date' in str(c).lower()]
        if date_cols:
            col = date_cols[0]
            try:
                sample = df[col].dropna().head(50).astype(str)
                dates = []
                for d in sample:
                    try:
                        dt = parser.parse(d, fuzzy=True, dayfirst=True)
                        dates.append(dt)
                    except:
                        pass
                if dates:
                    path_counts = Counter([(d.year, d.month) for d in dates])
                    most_common = path_counts.most_common(1)[0][0]
                    return most_common
            except:
                pass
    return None

def smart_read_with_ai(file, expected_cols, description, sheet_name=None):
    log = []
    try:
        # 1. Read Raw Preview
        file.seek(0)
        if file.name.endswith('.csv'):
            df_raw = pd.read_csv(file, header=None, nrows=20)
        else:
            try:
                sheet = sheet_name if sheet_name else 0
                df_raw = pd.read_excel(file, sheet_name=sheet, header=None, nrows=20)
            except:
                file.seek(0)
                df_raw = pd.read_excel(file, sheet_name=0, header=None, nrows=20)

        # 2. Header Detection
        def check_header_match(row_vals):
            row_str = " ".join([str(x).lower() for x in row_vals])
            matches = sum(1 for c in expected_cols if c.lower() in row_str)
            return matches >= min(2, len(expected_cols))

        header_idx = -1
        mapping = {}
        for i, row in df_raw.iterrows():
            if check_header_match(row.values):
                header_idx = i
                break
                
        if header_idx == -1:
            log.append(f"🤖 Calling AI for '{file.name}' structure...")
            csv_preview = df_raw.to_csv(index=False)
            structure = intelligent_structure_mapping(csv_preview, expected_cols, description)
            if structure:
                header_idx = structure.get("header_row_index", 0)
                mapping = structure.get("column_mapping", {})
        
        if header_idx == -1:
            header_idx = 0
            log.append(f"⚠️ Fallback to row 0 for {file.name}")

        # 3. Reload Full
        file.seek(0)
        if file.name.endswith('.csv'):
            df = pd.read_csv(file, header=header_idx)
        else:
            sheet = sheet_name if sheet_name else 0
            df = pd.read_excel(file, sheet_name=sheet, header=header_idx)
            
        if mapping:
            df.rename(columns=mapping, inplace=True)
            
        df = df.loc[:, ~df.columns.duplicated()]
        return df, "\n".join(log)

    except Exception as e:
        # log error but continue
        return pd.DataFrame(), f"❌ Error: {e}"

def process_data(orders_files, payment_files, cost_file, packaging_cost, misc_cost, smart_match=False):
    logs = []
    all_orders = []
    ref_month = None
    if not isinstance(orders_files, list): orders_files = [orders_files]
    if not isinstance(payment_files, list): payment_files = [payment_files]

    for f in orders_files:
        df, msg = smart_read_with_ai(f, EXPECTED_ORDER_COLS, "Orders File")
        if msg: logs.append(msg)
        if not df.empty:
            all_orders.append(df)
            if ref_month is None:
                ref_month = detect_file_month(f, df)

    if not all_orders:
         return None, None, None, ["❌ No valid order data found."]
    
    df_orders = pd.concat(all_orders, ignore_index=True)
    cols_to_keep = [c for c in EXPECTED_ORDER_COLS if c in df_orders.columns]
    df_orders = df_orders[cols_to_keep].copy()
    if 'Quantity' in df_orders.columns:
        df_orders['Quantity'] = pd.to_numeric(df_orders['Quantity'], errors='coerce').fillna(0)
    else:
        df_orders['Quantity'] = 1

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
        df_pay, msg = smart_read_with_ai(pf, EXPECTED_PAYMENT_COLS, "Payment File", sheet_name='Order Payments')
        if msg: logs.append(msg)
        
        f_month = detect_file_month(pf, df_pay)
        is_next_month = False
        if ref_month and f_month:
            if f_month == ref_month: is_next_month = False
            elif f_month > ref_month: is_next_month = True
        else:
             if "next" in pf.name.lower(): is_next_month = True
        
        ads = get_ads_cost_from_file(pf)
        if is_next_month:
            dfs_next.append(df_pay)
            next_ads_sum += ads
        else:
            dfs_same.append(df_pay)
            same_ads_sum += ads

    def aggregate_payments(dfs):
        if not dfs: return pd.DataFrame(columns=['Sub Order No', 'Amount', 'Live Order Status'])
        combined = pd.concat(dfs, ignore_index=True)
        # Look for Payment column case-insensitively
        for col in combined.columns:
            if col.lower() in ["final settlement amount", "final settlement", "settlement amount"]:
                combined.rename(columns={col: 'Amount'}, inplace=True)
                break
        if "Amount" in combined.columns:
            combined['Amount'] = pd.to_numeric(combined['Amount'], errors='coerce').fillna(0)
        return combined

    df_same_comb = aggregate_payments(dfs_same)
    df_next_comb = aggregate_payments(dfs_next)
    
    if not df_same_comb.empty and 'Sub Order No' in df_same_comb.columns:
        df_same_pivot = df_same_comb.groupby('Sub Order No')['Amount'].sum().reset_index().rename(columns={'Amount': 'same month pay'})
    else:
        df_same_pivot = pd.DataFrame(columns=['Sub Order No', 'same month pay'])
        
    if not df_next_comb.empty and 'Sub Order No' in df_next_comb.columns:
        df_next_pivot = df_next_comb.groupby('Sub Order No')['Amount'].sum().reset_index().rename(columns={'Amount': 'next month pay'})
    else:
        df_next_pivot = pd.DataFrame(columns=['Sub Order No', 'next month pay'])
        
    all_status_dfs = []
    if not df_same_comb.empty and 'Live Order Status' in df_same_comb.columns: all_status_dfs.append(df_same_comb[['Sub Order No', 'Live Order Status']])
    if not df_next_comb.empty and 'Live Order Status' in df_next_comb.columns: all_status_dfs.append(df_next_comb[['Sub Order No', 'Live Order Status']])
    df_status = pd.concat(all_status_dfs) if all_status_dfs else pd.DataFrame(columns=['Sub Order No', 'Live Order Status'])

    df_cost, msg_c = smart_read_with_ai(cost_file, EXPECTED_COST_COLS, "Cost Sheet")
    if msg_c: logs.append(msg_c)
    
    cols_map = {c.lower(): c for c in df_cost.columns}
    if "sku" in cols_map: df_cost.rename(columns={cols_map["sku"]: "SKU"}, inplace=True)
    if "cost" in cols_map: df_cost.rename(columns={cols_map["cost"]: "Cost"}, inplace=True)

    if not ("SKU" in df_cost.columns and "Cost" in df_cost.columns):
        df_cost = pd.DataFrame(columns=['SKU', 'Cost'])

    if 'Sub Order No' not in df_orders.columns:
        return None, None, None, ["❌ Critical: 'Sub Order No' not found in orders."]

    df_orders_final = df_orders.copy()
    df_orders_final['Sub Order No'] = df_orders_final['Sub Order No'].astype(str).str.strip()
    
    # Merge Payments
    df_orders_final = df_orders_final.merge(df_same_pivot, on='Sub Order No', how='left')
    df_orders_final = df_orders_final.merge(df_next_pivot, on='Sub Order No', how='left')
    df_orders_final['total'] = df_orders_final['same month pay'].fillna(0) + df_orders_final['next month pay'].fillna(0)

    # Merge Status
    df_status['Sub Order No'] = df_status['Sub Order No'].astype(str).str.strip()
    status_lookup = df_status.drop_duplicates(subset=['Sub Order No'], keep='last')
    df_orders_final = df_orders_final.merge(status_lookup, on='Sub Order No', how='left')
    df_orders_final.rename(columns={'Live Order Status': 'status'}, inplace=True)
    
    # Merge Cost
    if not df_cost.empty and 'SKU' in df_cost.columns and 'Cost' in df_cost.columns:
        cost_lookup = df_cost[['SKU', 'Cost']].copy()
    else:
        cost_lookup = df_cost.iloc[:, :2].copy()
    cost_lookup.columns = ['SKU_Lookup', 'Cost_Value']
    
    def clean_sku(s):
        return s.astype(str).str.strip().str.replace(r'\.0$', '', regex=True)

    if smart_match:
        def normalize(s): return clean_sku(s).str.lower().str.replace(r'\s+', '', regex=True)
        df_orders_final['SKU_Join'] = normalize(df_orders_final['SKU'])
        cost_lookup['SKU_Join'] = normalize(cost_lookup['SKU_Lookup'])
    else:
        df_orders_final['SKU_Join'] = clean_sku(df_orders_final['SKU'])
        cost_lookup['SKU_Join'] = clean_sku(cost_lookup['SKU_Lookup'])

    cost_lookup = cost_lookup.drop_duplicates(subset=['SKU_Join'])
    df_orders_final = pd.merge(df_orders_final, cost_lookup[['SKU_Join', 'Cost_Value']], on='SKU_Join', how='left')
    
    missing_cost_mask = df_orders_final['Cost_Value'].isna()
    df_orders_final['Cost_Value'] = df_orders_final['Cost_Value'].fillna(0)
    df_orders_final['status'] = df_orders_final['status'].fillna('Unknown').astype(str).str.strip()
    
    condition_product = df_orders_final['status'].isin(['Delivered', 'Exchange', 'DELIVERED'])
    df_orders_final['cost'] = np.where(condition_product, df_orders_final['Cost_Value'], 0)
    df_orders_final['actual cost'] = df_orders_final['cost'] * df_orders_final['Quantity']
    
    condition_packaging = df_orders_final['status'].isin(['Delivered', 'Exchange', 'Return', 'DELIVERED'])
    df_orders_final['packaging cost'] = np.where(condition_packaging, packaging_cost, 0)
    
    total_payment_sum = df_orders_final['total'].sum()
    total_product_cost = df_orders_final['actual cost'].sum()
    total_packaging = df_orders_final['packaging cost'].sum()
    profit = total_payment_sum - total_product_cost - total_packaging - abs(same_ads_sum) - abs(next_ads_sum) - misc_cost

    status_series = df_orders_final['status'].str.upper()
    stats = {
        "Total Payments": total_payment_sum,
        "Total Product Cost": total_product_cost,
        "Total Packaging Cost": total_packaging,
        "Ads Cost (Same Month)": same_ads_sum,
        "Ads Cost (Next Month)": next_ads_sum,
        "Miscellaneous Cost": misc_cost,
        "Profit / Loss": profit,
        "Order Count": len(df_orders_final),
        "count_delivered": len(df_orders_final[status_series == 'DELIVERED']),
        "count_return": len(df_orders_final[status_series == 'RETURN']),
        "count_rto": len(df_orders_final[status_series == 'RTO']),
        "count_exchange": len(df_orders_final[status_series == 'EXCHANGE']),
        "count_cancelled": len(df_orders_final[status_series == 'CANCELLED']),
        "count_shipped": len(df_orders_final[status_series == 'SHIPPED']),
        "count_ready_to_ship": len(df_orders_final[status_series == 'READY_TO_SHIP'])
    }
    
    df_export = df_orders_final.copy()
    df_export['cost'] = df_export['cost'].astype(object)
    df_export['actual cost'] = df_export['actual cost'].astype(object)
    df_export.loc[missing_cost_mask & condition_product, 'cost'] = "SKU Not Found"
    df_export.loc[missing_cost_mask & condition_product, 'actual cost'] = "SKU Not Found"
    
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df_export.to_excel(writer, sheet_name='Result', index=False)
        pd.DataFrame([stats]).T.to_excel(writer, sheet_name='Overview')
    output.seek(0)
    
    curr_cols = ['Sub Order No', 'SKU', 'status', 'Quantity', 'total']
    curr_cols = [c for c in curr_cols if c in df_orders_final.columns]
    missing_details_df = df_orders_final.loc[missing_cost_mask, curr_cols].copy()

    return output, stats, missing_details_df, logs
