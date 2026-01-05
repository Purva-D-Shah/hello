import streamlit as st
import pandas as pd
from processor import process_data
from ai_service import analyze_profit_loss, get_openai_client
import os

# --- Page Config ---
st.set_page_config(
    page_title="Dashboard Data Processor",
    layout="wide",
    initial_sidebar_state="collapsed",
    page_icon="📊"
)

# --- Custom Styling ---
st.markdown("""
<style>
    /* Global Styles */
    .stApp {
        background-color: #f8f9fa;
        color: #212529;
    }
    
    /* Metrics */
    div[data-testid="metric-container"] {
        background-color: white;
        padding: 10px 15px;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        text-align: center;
    }
    div[data-testid="metric-container"] label {
        font-size: 0.9rem;
        color: #6c757d;
    }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        font-size: 1.6rem;
        font-weight: 700;
        color: #212529;
    }

    /* Success Message */
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d1e7dd;
        color: #0f5132;
        border: 1px solid #badbcc;
        margin-bottom: 2rem;
        font-weight: 500;
        display: flex;
        align-items: center;
    }

    /* Error Box */
    .error-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f8d7da;
        color: #842029;
        border: 1px solid #f5c2c7;
        margin: 2rem 0;
        font-weight: 600;
    }

    /* Buttons */
    div.stButton > button {
        width: 100%;
        border-radius: 6px;
        font-weight: 600;
    }
    
    .big-profit {
        font-size: 3rem;
        font-weight: 800;
        color: #2c3e50;
    }
    
    h3 {
        color: #343a40;
        margin-top: 2rem;
        margin-bottom: 1rem;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# --- Password Logic (Simplified for Demo) ---
def check_password():
    """Simple password check"""
    # If no passwords in secrets, allow access
    if "passwords" not in st.secrets:
        return True
    
    # Check if logged in
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False

    if st.session_state["password_correct"]:
        return True

    # Login Form
    col1, col2, col3 = st.columns([1,1,1])
    with col2:
        st.markdown("### 🔒 Please Login")
        with st.form("credentials_form"):
            user = st.text_input("Username")
            pwd = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Log In")
            
            if submitted:
                if user in st.secrets["passwords"] and st.secrets["passwords"][user] == pwd:
                    st.session_state["password_correct"] = True
                    st.rerun()
                else:
                    st.error("Invalid credentials")
    return False

if check_password():
    st.title("📊 Dashboard Data Processor")
    st.markdown("---")
    
    # --- Configuration Section (Moved from Sidebar) ---
    with st.container():
        st.markdown("### 🛠️ Configuration")
        c1, c2, c3 = st.columns(3)
        with c1:
            packaging_cost = st.number_input("📦 Packaging Cost (₹/Order)", value=5.0, step=1.0)
        with c2:
            misc_cost = st.number_input("💸 Miscellaneous Costs (₹ Total)", value=0.0, step=100.0)
        with c3:
            # Smart Match Toggle
            use_smart_match = st.checkbox("🧠 Smart Match SKUs", value=False, help="Ignores case and spaces when matching SKUs. (e.g. 'Blue Shirt' == 'blueshirt')")
            
            # API Status Check
            client = get_openai_client()
            api_status = "✅ AI Connected" if client else "⚠️ AI Disconnected"
            st.caption(f"🤖 AI Status: {api_status}")

    st.markdown("---")

    # --- Main Input Section ---
    st.markdown("### 📂 Upload Data Files")
    
    col1, col2 = st.columns(2)
    with col1:
        orders_files = st.file_uploader("1️⃣ Upload ALL Orders Files (CSV/XLSX)", type=["csv", "xlsx"], accept_multiple_files=True)
        cost_file = st.file_uploader("2️⃣ Upload Cost File (Master Sheet)", type=["csv", "xlsx"])
    with col2:
        payment_files = st.file_uploader("3️⃣ Upload ALL Payment Files (XLSX)", type=["xlsx"], accept_multiple_files=True)
        st.info("ℹ️ System auto-detects 'Same Month' vs 'Next Month' payments.")

    # --- Processing ---
    if orders_files and cost_file and payment_files:
        if st.button("🚀 Process Data and Generate Report", type="primary"):
            with st.spinner("Processing multiple files..."):
                excel_data, stats, missing_details_df, logs = process_data(
                    orders_files, payment_files, cost_file, packaging_cost, misc_cost, smart_match=use_smart_match
                )
            
            if excel_data:
                # --- Success Message ---
                st.markdown('<div class="success-box">✅ Processing Complete!</div>', unsafe_allow_html=True)
                
                # Show Logs if any mappings happened
                if logs:
                    with st.expander("📝 AI Processing Logs"):
                        for log in logs:
                            st.info(log)

                # --- Financial Metrics ---
                st.markdown("### 📈 Financial Summary")
                
                # Profit/Loss Hero Metric
                pl = stats['Profit / Loss']
                st.markdown("PROFIT / LOSS")
                st.markdown(f'<div class="big-profit">₹{pl:,.2f}</div>', unsafe_allow_html=True)
                
                # Secondary Metrics
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total Payments", f"₹{stats['Total Payments']:,.2f}")
                m2.metric("Actual Cost", f"₹{stats['Total Product Cost']:,.2f}")
                m3.metric("Packaging", f"₹{stats['Total Packaging Cost']:,.2f}")
                m4.metric("Ads (Same Month)", f"₹{stats['Ads Cost (Same Month)']:,.2f}")
                
                # --- Missing SKUs Alert ---
                if isinstance(missing_details_df, pd.DataFrame) and not missing_details_df.empty:
                    st.markdown(f'<div class="error-box">⚠️ {len(missing_details_df)} Orders Missing SKU Cost</div>', unsafe_allow_html=True)
                    st.caption("The following orders have SKUs that were not found in your cost sheet. They are calculated as 0 cost.")
                    
                    st.dataframe(
                        missing_details_df,
                        use_container_width=True,
                        hide_index=True,
                         column_config={
                            "Total Payment": st.column_config.NumberColumn(format="₹%.2f")
                        }
                    )
                
                # --- Order Status Breakdown ---
                st.markdown("### 📦 Order Status Breakdown")
                
                # Metrics Row
                s1, s2, s3, s4, s5, s6, s7, s8 = st.columns(8)
                s1.metric("Total Orders", stats.get('Order Count', 0))
                s2.metric("Delivered", stats.get('count_delivered', 0))
                s3.metric("Return", stats.get('count_return', 0))
                s4.metric("RTO", stats.get('count_rto', 0))
                s5.metric("Exchange", stats.get('count_exchange', 0))
                s6.metric("Cancelled", stats.get('count_cancelled', 0))
                s7.metric("Shipped", stats.get('count_shipped', 0))
                s8.metric("Ready To Ship", stats.get('count_ready_to_ship', 0))
                
                # --- Visualizations (New) ---
                st.divider()
                st.markdown("### 📊 Visual Insights")
                
                chart_col1, chart_col2 = st.columns(2)
                
                with chart_col1:
                    st.caption("Order Status Distribution")
                    # Prepare Data for Pie Chart
                    status_labels = ["Delivered", "Return", "RTO", "Cancelled"]
                    status_values = [
                        stats.get('count_delivered', 0),
                        stats.get('count_return', 0),
                        stats.get('count_rto', 0),
                        stats.get('count_cancelled', 0)
                    ]
                    
                    if sum(status_values) > 0:
                        import plotly.express as px
                        fig_pie = px.pie(
                            names=status_labels, 
                            values=status_values, 
                            color=status_labels,
                            color_discrete_map={
                                "Delivered": "#2ecc71", 
                                "Return": "#f1c40f", 
                                "RTO": "#e74c3c", 
                                "Cancelled": "#95a5a6"
                            },
                            hole=0.4
                        )
                        fig_pie.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=300)
                        st.plotly_chart(fig_pie, use_container_width=True)
                
                with chart_col2:
                    st.caption("Cost vs Revenue Breakdown")
                    # Waterfall or Bar Chart
                    fin_data = {
                        "Metric": ["Revenue", "Product Cost", "Packaging", "Ads (Same)", "Ads (Next)"],
                        "Amount": [
                            stats['Total Payments'],
                            -stats['Total Product Cost'],
                            -stats['Total Packaging Cost'],
                            -stats['Ads Cost (Same Month)'],
                            -stats['Ads Cost (Next Month)']
                        ]
                    }
                    fig_bar = px.bar(
                        fin_data, 
                        x="Metric", 
                        y="Amount", 
                        color="Amount",
                        color_continuous_scale=["#e74c3c", "#2ecc71"],
                        text_auto='.2s'
                    )
                    fig_bar.update_layout(showlegend=False, margin=dict(t=0, b=0, l=0, r=0), height=300)
                    st.plotly_chart(fig_bar, use_container_width=True)

                # --- Product Performance (New) ---
                st.markdown("### 🏆 Product Performance")
                # We need to access df_final for this, but process_data currently returns only excel bytes.
                # To do this cleanly, we'd need process_data to return the DF or calculate this inside process_data.
                # For now, let's trust the Excel report has it, or we rely on the summary stats for now.
                # NOTE: For a future update, we should refactor process_data to return the DataFrame so we can show "Top Profitable SKUs" here.
                
                st.info("💡 Pro Tip: Check the 'Detailed Report' Excel sheet for per-SKU profit analysis.")

                st.divider()

                # --- AI Analysis ---
                if client:
                    st.markdown("### 🤖 Strategy & Insights (AI)")
                    with st.spinner("AI analyzing financials..."):
                        ai_summary = analyze_profit_loss(stats)
                    st.info(ai_summary)
                else:
                     st.caption("Add OPENAI_API_KEY to secrets for AI Analysis")

                st.divider()
                
                # --- Download ---
                st.download_button(
                    label="⬇️ Download Excel Report",
                    data=excel_data,
                    file_name="Final_Report.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    type="primary",
                    use_container_width=True
                )
    else:
        st.info("👆 Please upload all required files (Orders, Payments, Cost) to begin.")
