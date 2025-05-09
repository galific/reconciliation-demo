import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import random
import io
import base64

# Configuration variables
SHOW_DATA_SOURCES = False  # Set to True to show data sources panel

# Set page configuration
st.set_page_config(
    page_title="Galific Auto-Reconciliation Suite",
    page_icon="💸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #1E3A8A;
        font-weight: 600;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .stat-card {
        background-color: #F3F4F6;
        border-radius: 10px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stat-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1E3A8A;
    }
    .stat-label {
        font-size: 1rem;
        font-weight: 500;
        color: #6B7280;
    }
    .success {
        color: #059669;
    }
    .warning {
        color: #D97706;
    }
    .danger {
        color: #DC2626;
    }
    .mt-2{
            margin-top: 2rem !important;
    }
    .highlight {
        background-color: #FFEDD5;
        padding: 0.2rem 0.5rem;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar for controls and file uploads
with st.sidebar:
    st.image("https://galificsolutions.com/assets/logo.png", width=200)
    
    if SHOW_DATA_SOURCES:
        st.markdown("## Data Sources")
        # File uploaders
        bank_file = st.file_uploader("Bank Statement (CSV)", type=["csv"])
        erp_file = st.file_uploader("ERP Export (CSV)", type=["csv"])
        payout_file = st.file_uploader("Payout System Export (CSV)", type=["csv"])
        crm_file = st.file_uploader("CRM Data (CSV)", type=["csv"])
        st.markdown("---")
    
    # Reconciliation settings
    st.markdown("## Reconciliation Settings")
    
    matching_criteria = st.multiselect(
        "Matching Criteria:",
        ["Amount", "Date", "Reference ID", "Customer Name"],
        default=["Amount", "Date"]
    )
    
    tolerance_days = st.slider(
        "Date Tolerance (days):",
        min_value=0,
        max_value=7,
        value=2
    )
    
    amount_tolerance = st.slider(
        "Amount Tolerance (%):",
        min_value=0.0,
        max_value=5.0,
        value=0.0,
        step=0.1
    )
    
    run_button = st.button("Run Reconciliation", type="primary")

# Main content
st.markdown('<h1 class="main-header">Galific Auto-Reconciliation Suite</h1>', unsafe_allow_html=True)

# Function to generate sample data
def generate_sample_data(months=6, error_rate=7, volume="Medium (100-300)"):
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Determine volume range
    if volume == "Low (50-100)":
        min_tx, max_tx = 50, 100
    elif volume == "Medium (100-300)":
        min_tx, max_tx = 100, 300
    else:  # High
        min_tx, max_tx = 300, 500
    
    # Date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30*months)
    
    # Common transaction types
    tx_types = ["Payment", "Refund", "Subscription", "One-time", "Renewal"]
    payment_methods = ["Credit Card", "Debit Card", "UPI", "NetBanking", "Wallet"]
    customers = [f"Customer_{i}" for i in range(1, 51)]
    
    # Generate random dates within the range
    num_transactions = np.random.randint(min_tx * months, max_tx * months)
    dates = [start_date + timedelta(days=np.random.randint(0, 30*months)) for _ in range(num_transactions)]
    dates.sort()
    
    # Generate transaction IDs
    tx_ids = [f"TX{i+10000}" for i in range(num_transactions)]
    
    # Generate random amounts (in rupees)
    amounts = np.random.choice([
        np.random.randint(1000, 5000),  # Small transactions
        np.random.randint(5000, 20000),  # Medium transactions
        np.random.randint(20000, 100000)  # Large transactions
    ], num_transactions, p=[0.6, 0.3, 0.1])
    
    # Create base DataFrame with all matching transactions
    base_df = pd.DataFrame({
        'Date': dates,
        'Transaction_ID': tx_ids,
        'Amount': amounts,
        'Type': np.random.choice(tx_types, num_transactions),
        'Customer': np.random.choice(customers, num_transactions),
        'Payment_Method': np.random.choice(payment_methods, num_transactions)
    })
    
    # Bank statement data (mostly complete, but might miss some)
    bank_df = base_df.copy()
    bank_df['Source'] = 'Bank'
    bank_df['Bank_Reference'] = [f"BNK{i+20000}" for i in range(len(bank_df))]
    
    # Randomly drop some rows to simulate missed transactions
    drop_indices = np.random.choice(bank_df.index, size=int(len(bank_df) * (error_rate/100) * 0.3), replace=False)
    bank_df = bank_df.drop(drop_indices)
    
    # ERP data (might have extra or missing entries, or different amounts)
    erp_df = base_df.copy()
    erp_df['Source'] = 'ERP'
    erp_df['ERP_Reference'] = [f"ERP{i+30000}" for i in range(len(erp_df))]
    
    # Randomly alter some amounts to create discrepancies
    alter_indices = np.random.choice(erp_df.index, size=int(len(erp_df) * (error_rate/100) * 0.4), replace=False)
    for idx in alter_indices:
        if idx in erp_df.index:  # Check if index exists after potential drops
            # Randomly alter by ±1-5%
            factor = 1 + (np.random.randint(-5, 6) / 100)
            erp_df.loc[idx, 'Amount'] = round(erp_df.loc[idx, 'Amount'] * factor)
    
    # Randomly alter some dates to create timing discrepancies
    date_alter_indices = np.random.choice(erp_df.index, size=int(len(erp_df) * (error_rate/100) * 0.3), replace=False)
    for idx in date_alter_indices:
        if idx in erp_df.index:  # Check if index exists after potential drops
            # Shift by 1-3 days
            shift_days = int(np.random.randint(1, 4) * np.random.choice([-1, 1]))
            erp_df.loc[idx, 'Date'] = erp_df.loc[idx, 'Date'] + timedelta(days=shift_days)
    
    # Payout system data
    payout_df = base_df.copy()
    payout_df['Source'] = 'Payout'
    payout_df['Payout_Reference'] = [f"PAY{i+40000}" for i in range(len(payout_df))]
    
    # Add some transactions that might be in payout but not in ERP
    unique_payout = base_df.sample(int(len(base_df) * (error_rate/100) * 0.2))
    unique_payout['Source'] = 'Payout'
    unique_payout['Payout_Reference'] = [f"PAY{i+50000}" for i in range(len(unique_payout))]
    payout_df = pd.concat([payout_df, unique_payout])
    
    # CRM data (focus on refunds and customer info)
    crm_df = base_df[base_df['Type'] == 'Refund'].copy()
    crm_df['Source'] = 'CRM'
    crm_df['CRM_Reference'] = [f"CRM{i+60000}" for i in range(len(crm_df))]
    
    # Add some refunds that might be in CRM but not in other systems
    extra_refunds = pd.DataFrame({
        'Date': [start_date + timedelta(days=np.random.randint(0, 30*months)) for _ in range(int(len(crm_df) * 0.2))],
        'Transaction_ID': [f"TX{i+90000}" for i in range(int(len(crm_df) * 0.2))],
        'Amount': [np.random.randint(1000, 10000) for _ in range(int(len(crm_df) * 0.2))],
        'Type': ['Refund'] * int(len(crm_df) * 0.2),
        'Customer': np.random.choice(customers, int(len(crm_df) * 0.2)),
        'Payment_Method': np.random.choice(payment_methods, int(len(crm_df) * 0.2)),
        'Source': ['CRM'] * int(len(crm_df) * 0.2),
        'CRM_Reference': [f"CRM{i+70000}" for i in range(int(len(crm_df) * 0.2))]
    })
    crm_df = pd.concat([crm_df, extra_refunds])
    
    # Format dates to string for consistency
    bank_df['Date'] = bank_df['Date'].dt.strftime('%Y-%m-%d')
    erp_df['Date'] = erp_df['Date'].dt.strftime('%Y-%m-%d')
    payout_df['Date'] = payout_df['Date'].dt.strftime('%Y-%m-%d')
    crm_df['Date'] = crm_df['Date'].dt.strftime('%Y-%m-%d')
    
    return bank_df, erp_df, payout_df, crm_df

# Function to perform reconciliation
def reconcile_data(bank_df, erp_df, payout_df, crm_df, matching_criteria, tolerance_days=2, amount_tolerance=0):
    # Combine all data for reconciliation
    bank_df['System'] = 'Bank'
    erp_df['System'] = 'ERP'
    payout_df['System'] = 'Payout'
    crm_df['System'] = 'CRM'
    
    # Standardize column names
    bank_df = bank_df.rename(columns={'Bank_Reference': 'System_Reference'})
    erp_df = erp_df.rename(columns={'ERP_Reference': 'System_Reference'})
    payout_df = payout_df.rename(columns={'Payout_Reference': 'System_Reference'})
    crm_df = crm_df.rename(columns={'CRM_Reference': 'System_Reference'})
    
    # Combine all data
    all_data = pd.concat([bank_df, erp_df, payout_df, crm_df])

    # Convert date strings to datetime for comparison
    bank_df['Date'] = pd.to_datetime(bank_df['Date'])
    erp_df['Date'] = pd.to_datetime(erp_df['Date']) 
    payout_df['Date'] = pd.to_datetime(payout_df['Date'])
    crm_df['Date'] = pd.to_datetime(crm_df['Date'])
    all_data['Date'] = pd.to_datetime(all_data['Date'])

    # Initialize reconciliation results
    reconciliation_results = []
    
    # For each transaction in bank data, try to find matching in other systems
    for _, bank_tx in bank_df.iterrows():
        # Create a base record for this transaction
        tx_record = {
            'Transaction_ID': bank_tx['Transaction_ID'],
            'Date': bank_tx['Date'],
            'Amount': bank_tx['Amount'],
            'Type': bank_tx['Type'],
            'Customer': bank_tx['Customer'],
            'Bank_Present': 'Yes',
            'Bank_Reference': bank_tx['System_Reference'],
            'Bank_Date': bank_tx['Date'],
            'Bank_Amount': bank_tx['Amount'],
            'ERP_Present': 'No',
            'ERP_Reference': '',
            'ERP_Date': None,
            'ERP_Amount': None,
            'Payout_Present': 'No',
            'Payout_Reference': '',
            'Payout_Date': None,
            'Payout_Amount': None,
            'CRM_Present': 'No',
            'CRM_Reference': '',
            'CRM_Date': None,
            'CRM_Amount': None,
            'Fully_Matched': False,
            'Mismatches': []
        }
        
        # Match criteria based on user selection
        match_conditions = {}

        if "Date" in matching_criteria:
            # Create date range for matching with tolerance
            date_lower = bank_tx['Date'] - timedelta(days=tolerance_days)
            date_upper = bank_tx['Date'] + timedelta(days=tolerance_days)
            match_conditions['Date'] = lambda x: (pd.to_datetime(x) >= date_lower) & (pd.to_datetime(x) <= date_upper)
        
        if "Amount" in matching_criteria:
            if amount_tolerance > 0:
                # Create amount range with tolerance
                amount_lower = bank_tx['Amount'] * (1 - amount_tolerance/100)
                amount_upper = bank_tx['Amount'] * (1 + amount_tolerance/100)
                match_conditions['Amount'] = lambda x: (x >= amount_lower) & (x <= amount_upper)
            else:
                match_conditions['Amount'] = bank_tx['Amount']
        
        if "Reference ID" in matching_criteria:
            match_conditions['Transaction_ID'] = bank_tx['Transaction_ID']
            
        if "Customer Name" in matching_criteria:
            match_conditions['Customer'] = bank_tx['Customer']
        
        # Find matches in ERP
        erp_matches = erp_df.copy()
        for col, condition in match_conditions.items():
            if callable(condition):
                erp_matches = erp_matches[condition(erp_matches[col])]
            else:
                erp_matches = erp_matches[erp_matches[col] == condition]
        
        if not erp_matches.empty:
            # Take the first match (could be enhanced to handle multiple matches)
            erp_match = erp_matches.iloc[0]
            tx_record['ERP_Present'] = 'Yes'
            tx_record['ERP_Reference'] = erp_match['System_Reference']
            tx_record['ERP_Date'] = erp_match['Date']
            tx_record['ERP_Amount'] = erp_match['Amount']
            
            # Check for specific mismatches
            date_diff = abs((bank_tx['Date'] - erp_match['Date']).days)
            if date_diff > 0:
                tx_record['Mismatches'].append('ERP_Date_Mismatch')
            
            if bank_tx['Amount'] != erp_match['Amount']:
                tx_record['Mismatches'].append('ERP_Amount_Mismatch')
        else:
            tx_record['Mismatches'].append('Missing_In_ERP')
        
        # Find matches in Payout
        payout_matches = payout_df.copy()
        for col, condition in match_conditions.items():
            if callable(condition):
                payout_matches = payout_matches[condition(payout_matches[col])]
            else:
                payout_matches = payout_matches[payout_matches[col] == condition]
        
        if not payout_matches.empty:
            payout_match = payout_matches.iloc[0]
            tx_record['Payout_Present'] = 'Yes'
            tx_record['Payout_Reference'] = payout_match['System_Reference']
            tx_record['Payout_Date'] = payout_match['Date']
            tx_record['Payout_Amount'] = payout_match['Amount']
            
            # Check for specific mismatches
            date_diff = abs((bank_tx['Date'] - payout_match['Date']).days)
            if date_diff > 0:
                tx_record['Mismatches'].append('Payout_Date_Mismatch')
            
            if bank_tx['Amount'] != payout_match['Amount']:
                tx_record['Mismatches'].append('Payout_Amount_Mismatch')
        else:
            tx_record['Mismatches'].append('Missing_In_Payout')
        
        # For refunds, check in CRM
        if bank_tx['Type'] == 'Refund':
            crm_matches = crm_df.copy()
            for col, condition in match_conditions.items():
                if callable(condition):
                    crm_matches = crm_matches[condition(crm_matches[col])]
                else:
                    crm_matches = crm_matches[crm_matches[col] == condition]
            
            if not crm_matches.empty:
                crm_match = crm_matches.iloc[0]
                tx_record['CRM_Present'] = 'Yes'
                tx_record['CRM_Reference'] = crm_match['System_Reference']
                tx_record['CRM_Date'] = crm_match['Date']
                tx_record['CRM_Amount'] = crm_match['Amount']
                
                # Check for specific mismatches
                date_diff = abs((bank_tx['Date'] - crm_match['Date']).days)
                if date_diff > 0:
                    tx_record['Mismatches'].append('CRM_Date_Mismatch')
                
                if bank_tx['Amount'] != crm_match['Amount']:
                    tx_record['Mismatches'].append('CRM_Amount_Mismatch')
            elif bank_tx['Type'] == 'Refund':  # Only mark as mismatch for refunds
                tx_record['Mismatches'].append('Refund_Missing_In_CRM')
        
        # Check if fully matched across all applicable systems
        if bank_tx['Type'] == 'Refund':
            tx_record['Fully_Matched'] = (tx_record['ERP_Present'] == 'Yes' and
                                        tx_record['Payout_Present'] == 'Yes' and
                                        tx_record['CRM_Present'] == 'Yes' and
                                        not tx_record['Mismatches'])
        else:
            tx_record['Fully_Matched'] = (tx_record['ERP_Present'] == 'Yes' and
                                        tx_record['Payout_Present'] == 'Yes' and
                                        not tx_record['Mismatches'])
        
        reconciliation_results.append(tx_record)
    
    # Check for transactions in ERP but not in Bank
    for _, erp_tx in erp_df.iterrows():
        bank_match = bank_df[bank_df['Transaction_ID'] == erp_tx['Transaction_ID']]
        
        if bank_match.empty:
            # Create a record for this orphaned transaction
            tx_record = {
                'Transaction_ID': erp_tx['Transaction_ID'],
                'Date': erp_tx['Date'],
                'Amount': erp_tx['Amount'],
                'Type': erp_tx['Type'],
                'Customer': erp_tx['Customer'],
                'Bank_Present': 'No',
                'Bank_Reference': '',
                'Bank_Date': None,
                'Bank_Amount': None,
                'ERP_Present': 'Yes',
                'ERP_Reference': erp_tx['System_Reference'],
                'ERP_Date': erp_tx['Date'],
                'ERP_Amount': erp_tx['Amount'],
                'Payout_Present': 'No',
                'Payout_Reference': '',
                'Payout_Date': None,
                'Payout_Amount': None,
                'CRM_Present': 'No',
                'CRM_Reference': '',
                'CRM_Date': None,
                'CRM_Amount': None,
                'Fully_Matched': False,
                'Mismatches': ['Missing_In_Bank']
            }
            
            # Check if it's in payout
            payout_match = payout_df[payout_df['Transaction_ID'] == erp_tx['Transaction_ID']]
            if not payout_match.empty:
                payout_tx = payout_match.iloc[0]
                tx_record['Payout_Present'] = 'Yes'
                tx_record['Payout_Reference'] = payout_tx['System_Reference']
                tx_record['Payout_Date'] = payout_tx['Date']
                tx_record['Payout_Amount'] = payout_tx['Amount']
            else:
                tx_record['Mismatches'].append('Missing_In_Payout')
            
            # For refunds, check in CRM
            if erp_tx['Type'] == 'Refund':
                crm_match = crm_df[crm_df['Transaction_ID'] == erp_tx['Transaction_ID']]
                if not crm_match.empty:
                    crm_tx = crm_match.iloc[0]
                    tx_record['CRM_Present'] = 'Yes'
                    tx_record['CRM_Reference'] = crm_tx['System_Reference']
                    tx_record['CRM_Date'] = crm_tx['Date']
                    tx_record['CRM_Amount'] = crm_tx['Amount']
                else:
                    tx_record['Mismatches'].append('Refund_Missing_In_CRM')
            
            reconciliation_results.append(tx_record)
    
    # Check for refunds in CRM that aren't in other systems
    for _, crm_tx in crm_df.iterrows():
        bank_match = bank_df[bank_df['Transaction_ID'] == crm_tx['Transaction_ID']]
        erp_match = erp_df[erp_df['Transaction_ID'] == crm_tx['Transaction_ID']]
        
        if bank_match.empty and erp_match.empty:
            # Create a record for this orphaned refund
            tx_record = {
                'Transaction_ID': crm_tx['Transaction_ID'],
                'Date': crm_tx['Date'],
                'Amount': crm_tx['Amount'],
                'Type': crm_tx['Type'],
                'Customer': crm_tx['Customer'],
                'Bank_Present': 'No',
                'Bank_Reference': '',
                'Bank_Date': None,
                'Bank_Amount': None,
                'ERP_Present': 'No',
                'ERP_Reference': '',
                'ERP_Date': None,
                'ERP_Amount': None,
                'Payout_Present': 'No',
                'Payout_Reference': '',
                'Payout_Date': None,
                'Payout_Amount': None,
                'CRM_Present': 'Yes',
                'CRM_Reference': crm_tx['System_Reference'],
                'CRM_Date': crm_tx['Date'],
                'CRM_Amount': crm_tx['Amount'],
                'Fully_Matched': False,
                'Mismatches': ['Refund_Missing_In_Bank', 'Refund_Missing_In_ERP']
            }
            
            # Check if it's in payout
            payout_match = payout_df[payout_df['Transaction_ID'] == crm_tx['Transaction_ID']]
            if not payout_match.empty:
                payout_tx = payout_match.iloc[0]
                tx_record['Payout_Present'] = 'Yes'
                tx_record['Payout_Reference'] = payout_tx['System_Reference']
                tx_record['Payout_Date'] = payout_tx['Date']
                tx_record['Payout_Amount'] = payout_tx['Amount']
            else:
                tx_record['Mismatches'].append('Refund_Missing_In_Payout')
            
            reconciliation_results.append(tx_record)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(reconciliation_results)
    
    # Calculate match status
    def determine_status(row):
        if row['Fully_Matched']:
            return 'Matched'
        elif not row['Mismatches']:
            return 'Matched'  # No mismatches = matched
        elif 'Missing_In_Bank' in row['Mismatches'] or 'Missing_In_ERP' in row['Mismatches']:
            return 'Missing'
        else:
            return 'Mismatch'
    
    # Apply status determination
    if not results_df.empty:
        results_df['Status'] = results_df.apply(determine_status, axis=1)
        
        # Convert date columns back to string for display
        results_df['Date'] = pd.to_datetime(results_df['Date'])
        date_cols = [col for col in results_df.columns if 'Date' in col and col != 'Date']
        for col in date_cols:
            results_df[col] = results_df[col].apply(lambda x: x.strftime('%Y-%m-%d') if pd.notnull(x) else None)
    
    return results_df

# Function to create downloadable link
def get_table_download_link(df, filename, text):
    """Generates a link allowing the data in a given pandas dataframe to be downloaded"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" class="download-link">{text}</a>'
    return href

# Main app logic
if run_button:
    with st.spinner('Running reconciliation...'):
        # Process data based on uploaded files or generate sample data if no files uploaded
        if SHOW_DATA_SOURCES and bank_file and erp_file and payout_file and crm_file:
            # Process uploaded files
            bank_df = pd.read_csv(bank_file)
            erp_df = pd.read_csv(erp_file)
            payout_df = pd.read_csv(payout_file)
            crm_df = pd.read_csv(crm_file)
        else:
            # Generate sample data with default parameters
            bank_df, erp_df, payout_df, crm_df = generate_sample_data(
                months=6,  # Default to 6 months
                error_rate=7,  # Default error rate
                volume="Medium (100-300)"  # Default volume
            )
            # st.info("Using sample data for demonstration. Upload your own files to process real data.")
        
        # Perform reconciliation
        results_df = reconcile_data(
            bank_df, erp_df, payout_df, crm_df,
            matching_criteria=matching_criteria,
            tolerance_days=tolerance_days,
            amount_tolerance=amount_tolerance
        )
        
        # Calculate summary statistics
        total_transactions = len(results_df)
        matched_count = len(results_df[results_df['Status'] == 'Matched'])
        mismatch_count = len(results_df[results_df['Status'] == 'Mismatch'])
        missing_count = len(results_df[results_df['Status'] == 'Missing'])
        
        match_rate = (matched_count / total_transactions) * 100 if total_transactions > 0 else 0
        
        # Calculate financial impact
        mismatch_amount = 0
        missing_amount = 0
        
        if not results_df.empty:
            mismatch_rows = results_df[results_df['Status'] == 'Mismatch']
            for _, row in mismatch_rows.iterrows():
                if 'ERP_Amount_Mismatch' in row['Mismatches']:
                    # Calculate the difference between bank and ERP amounts
                    bank_amt = row['Bank_Amount'] or 0
                    erp_amt = row['ERP_Amount'] or 0
                    mismatch_amount += abs(bank_amt - erp_amt)
            
            missing_rows = results_df[results_df['Status'] == 'Missing']
            for _, row in missing_rows.iterrows():
                if 'Missing_In_Bank' in row['Mismatches']:
                    missing_amount += row['ERP_Amount'] or 0
                elif 'Missing_In_ERP' in row['Mismatches']:
                    missing_amount += row['Bank_Amount'] or 0
    
    
    # Summary cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">₹{"{:,.2f}".format(mismatch_amount + missing_amount)}</div>
            <div class="stat-label">Total Financial Impact</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        match_color = "success" if match_rate >= 95 else "warning" if match_rate >= 85 else "danger"
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value {match_color}">{match_rate:.1f}%</div>
            <div class="stat-label">Reconciliation Rate</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{matched_count}/{total_transactions}</div>
            <div class="stat-label">Matched Transactions</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value danger">{mismatch_count + missing_count}</div>
            <div class="stat-label">Discrepancies Found</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Visualization of reconciliation results
    st.markdown('<h3 class="sub-header mt-2">Reconciliation Analysis</h3>', unsafe_allow_html=True)
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["Summary Charts", "Detailed Table", "System Comparison"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            # Pie chart of status
            status_counts = results_df['Status'].value_counts().reset_index()
            status_counts.columns = ['Status', 'Count']
            
            fig = px.pie(
                status_counts, 
                values='Count', 
                names='Status',
                color='Status',
                color_discrete_map={
                    'Matched': '#059669',
                    'Mismatch': '#D97706',
                    'Missing': '#DC2626'
                },
                title='Reconciliation Status'
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Bar chart of mismatch types
            if not results_df.empty:
                # Extract all mismatch types
                mismatch_types = []
                for mismatches in results_df['Mismatches']:
                    if mismatches:  # Check if the list is not empty
                        mismatch_types.extend(mismatches)
                
                mismatch_counts = pd.Series(mismatch_types).value_counts().reset_index()
                mismatch_counts.columns = ['Mismatch Type', 'Count']
                
                fig = px.bar(
                    mismatch_counts,
                    x='Count',
                    y='Mismatch Type',
                    orientation='h',
                    title='Mismatch Categories',
                    color_discrete_sequence=['#D97706']
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Financial impact breakdown
        st.subheader("Financial Impact Breakdown")
        
        impact_data = [
            {"Category": "Amount Mismatches", "Value": mismatch_amount},
            {"Category": "Missing Transactions", "Value": missing_amount}
        ]
        impact_df = pd.DataFrame(impact_data)
        
        fig = px.bar(
            impact_df,
            x='Category',
            y='Value',
            title='Financial Impact (₹)',
            color_discrete_sequence=['#DC2626']
        )
        fig.update_layout(yaxis_title="Amount (₹)")
        st.plotly_chart(fig, use_container_width=True)
        
        # Time series of match rate
        if not results_df.empty:
            # Group by date and calculate match rate
            daily_stats = results_df.groupby(results_df['Date'].dt.floor('D')).apply(
                lambda x: pd.Series({
                    'Total': len(x),
                    'Matched': sum(x['Status'] == 'Matched'),
                    'Match_Rate': sum(x['Status'] == 'Matched') / len(x) * 100
                })
            ).reset_index()
            
            fig = px.line(
                daily_stats,
                x='Date',
                y='Match_Rate',
                title='Daily Reconciliation Rate (%)',
                markers=True
            )
            fig.update_layout(yaxis_title="Match Rate (%)")
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Filter options
        st.subheader("Filter Results")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            status_filter = st.multiselect(
                "Status:",
                options=["Matched", "Mismatch", "Missing"],
                default=["Mismatch", "Missing"]
            )
        
        with col2:
            amount_range = st.slider(
                "Amount Range (₹):",
                min_value=0,
                max_value=int(results_df['Amount'].max()) if not results_df.empty else 100000,
                value=(0, int(results_df['Amount'].max()) if not results_df.empty else 100000)
            )
        
        with col3:
            if not results_df.empty:
                date_min = results_df['Date'].min().date()
                date_max = results_df['Date'].max().date()
                
                date_range = st.date_input(
                    "Date Range:",
                    value=(date_min, date_max),
                    min_value=date_min,
                    max_value=date_max
                )
        
        # Apply filters
        filtered_df = results_df.copy()
        
        if status_filter:
            filtered_df = filtered_df[filtered_df['Status'].isin(status_filter)]
        
        filtered_df = filtered_df[(filtered_df['Amount'] >= amount_range[0]) & 
                                (filtered_df['Amount'] <= amount_range[1])]
        
        if 'date_range' in locals() and len(date_range) == 2:
            date_lower = pd.to_datetime(date_range[0])
            date_upper = pd.to_datetime(date_range[1])
            filtered_df = filtered_df[(filtered_df['Date'] >= date_lower) & 
                                    (filtered_df['Date'] <= date_upper)]
        
        # Display the filtered table
        if not filtered_df.empty:
            # Reformat and select columns for display
            display_df = filtered_df[[
                'Transaction_ID', 'Date', 'Amount', 'Type', 'Customer', 'Status',
                'Bank_Present', 'ERP_Present', 'Payout_Present', 'CRM_Present'
            ]].copy()
            
            # Format Date
            display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
            
            # Add color to Status column
            def color_status(val):
                color = 'white'
                if val == 'Matched':
                    color = '#DCFCE7'  # Light green
                elif val == 'Mismatch':
                    color = '#FEF3C7'  # Light yellow
                elif val == 'Missing':
                    color = '#FEE2E2'  # Light red
                return f'background-color: {color}'
            
            # Apply styling
            styled_df = display_df.style.applymap(color_status, subset=['Status'])
            
            st.dataframe(styled_df, use_container_width=True)
            
            # Provide download link
            st.markdown(get_table_download_link(filtered_df, 'reconciliation_results.csv', 
                                             '📥 Download Filtered Results as CSV'), 
                       unsafe_allow_html=True)
            
            # Detailed view of selected transaction
            st.subheader("Transaction Details")
            selected_tx = st.selectbox(
                "Select Transaction ID for Detailed View:",
                options=filtered_df['Transaction_ID'].tolist()
            )
            
            if selected_tx:
                tx_details = filtered_df[filtered_df['Transaction_ID'] == selected_tx].iloc[0]
                
                # Create 2x2 grid for system details
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Bank Data")
                    if tx_details['Bank_Present'] == 'Yes':
                        st.markdown(f"""
                        - **Reference**: {tx_details['Bank_Reference']}
                        - **Date**: {tx_details['Bank_Date']}
                        - **Amount**: ₹{tx_details['Bank_Amount']:,.2f}
                        """)
                    else:
                        st.markdown("❌ **Not Present in Bank**")
                    
                    st.markdown("#### ERP Data")
                    if tx_details['ERP_Present'] == 'Yes':
                        st.markdown(f"""
                        - **Reference**: {tx_details['ERP_Reference']}
                        - **Date**: {tx_details['ERP_Date']}
                        - **Amount**: ₹{tx_details['ERP_Amount']:,.2f}
                        """)
                    else:
                        st.markdown("❌ **Not Present in ERP**")
                
                with col2:
                    st.markdown("#### Payout Data")
                    if tx_details['Payout_Present'] == 'Yes':
                        st.markdown(f"""
                        - **Reference**: {tx_details['Payout_Reference']}
                        - **Date**: {tx_details['Payout_Date']}
                        - **Amount**: ₹{tx_details['Payout_Amount']:,.2f}
                        """)
                    else:
                        st.markdown("❌ **Not Present in Payout System**")
                    
                    st.markdown("#### CRM Data")
                    if tx_details['CRM_Present'] == 'Yes':
                        st.markdown(f"""
                        - **Reference**: {tx_details['CRM_Reference']}
                        - **Date**: {tx_details['CRM_Date']}
                        - **Amount**: ₹{tx_details['CRM_Amount']:,.2f}
                        """)
                    else:
                        st.markdown("❓ **Not Present in CRM**" if tx_details['Type'] == 'Refund' else "")
                
                # Display mismatch details
                if tx_details['Mismatches']:
                    st.markdown("#### Identified Issues")
                    for mismatch in tx_details['Mismatches']:
                        st.markdown(f"- ⚠️ **{mismatch.replace('_', ' ')}**")
                else:
                    st.markdown("#### ✅ No Issues Detected")
    
    with tab3:
        st.subheader("System Comparison")
        
        # Create a Sankey diagram showing flow between systems
        if not results_df.empty:
            # Prepare data for Sankey diagram
            # Create a simplified view of where transactions are present
            system_df = results_df[['Transaction_ID', 'Bank_Present', 'ERP_Present', 
                                  'Payout_Present', 'CRM_Present']].copy()
            
            # Convert Yes/No to 1/0
            for col in ['Bank_Present', 'ERP_Present', 'Payout_Present', 'CRM_Present']:
                system_df[col] = (system_df[col] == 'Yes').astype(int)
            
            # Count transactions present in different combinations of systems
            bank_to_erp = len(system_df[(system_df['Bank_Present'] == 1) & 
                                      (system_df['ERP_Present'] == 1)])
            bank_not_erp = len(system_df[(system_df['Bank_Present'] == 1) & 
                                        (system_df['ERP_Present'] == 0)])
            erp_not_bank = len(system_df[(system_df['Bank_Present'] == 0) & 
                                        (system_df['ERP_Present'] == 1)])
            
            bank_to_payout = len(system_df[(system_df['Bank_Present'] == 1) & 
                                        (system_df['Payout_Present'] == 1)])
            bank_not_payout = len(system_df[(system_df['Bank_Present'] == 1) & 
                                          (system_df['Payout_Present'] == 0)])
            payout_not_bank = len(system_df[(system_df['Bank_Present'] == 0) & 
                                          (system_df['Payout_Present'] == 1)])
            
            erp_to_payout = len(system_df[(system_df['ERP_Present'] == 1) & 
                                        (system_df['Payout_Present'] == 1)])
            erp_not_payout = len(system_df[(system_df['ERP_Present'] == 1) & 
                                         (system_df['Payout_Present'] == 0)])
            payout_not_erp = len(system_df[(system_df['ERP_Present'] == 0) & 
                                         (system_df['Payout_Present'] == 1)])
            
            # Create Sankey diagram
            labels = ['Bank', 'ERP', 'Payout', 'CRM',
                     'Bank ✓', 'Bank ✗', 'ERP ✓', 'ERP ✗', 'Payout ✓', 'Payout ✗']
            
            source = [0, 0, 1, 1, 2, 2, 0, 1, 2]
            target = [1, 2, 0, 2, 0, 1, 6, 7, 8]
            value = [bank_to_erp, bank_to_payout, bank_to_erp, erp_to_payout, 
                    bank_to_payout, erp_to_payout, 
                    bank_not_erp, erp_not_bank, payout_not_bank]
            
            # Create figure
            fig = go.Figure(data=[go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=labels,
                    color=["#1E40AF", "#047857", "#B45309", "#BE185D",
                          "#1E40AF", "#1E40AF", "#047857", "#047857", "#B45309", "#B45309"]
                ),
                link=dict(
                    source=source,
                    target=target,
                    value=value,
                    color=["rgba(30, 64, 175, 0.4)" for _ in range(len(source))]
                )
            )])
            
            fig.update_layout(title_text="Transaction Flow Between Systems", font_size=12)
            st.plotly_chart(fig, use_container_width=True)
            
            # System comparison matrix
            st.subheader("System Presence Matrix")
            
            # Create a matrix showing transaction presence across systems
            matrix_data = {
                'Bank & ERP': [bank_to_erp, bank_not_erp, erp_not_bank],
                'Bank & Payout': [bank_to_payout, bank_not_payout, payout_not_bank],
                'ERP & Payout': [erp_to_payout, erp_not_payout, payout_not_erp]
            }
            
            matrix_df = pd.DataFrame(matrix_data, 
                                    index=['Present in Both', 'In First Only', 'In Second Only'])
            
            st.table(matrix_df)
            
            # Venn diagram representation (since Plotly doesn't have a built-in Venn diagram)
            # We'll visualize the system presence using a bar chart instead
            st.subheader("System Coverage")
            
            # Count transactions by which systems they're present in
            system_counts = []
            system_counts.append({'Systems': 'Bank Only', 
                                'Count': len(system_df[(system_df['Bank_Present'] == 1) & 
                                                      (system_df['ERP_Present'] == 0) & 
                                                      (system_df['Payout_Present'] == 0)])})
            system_counts.append({'Systems': 'ERP Only', 
                                'Count': len(system_df[(system_df['Bank_Present'] == 0) & 
                                                      (system_df['ERP_Present'] == 1) & 
                                                      (system_df['Payout_Present'] == 0)])})
            system_counts.append({'Systems': 'Payout Only', 
                                'Count': len(system_df[(system_df['Bank_Present'] == 0) & 
                                                      (system_df['ERP_Present'] == 0) & 
                                                      (system_df['Payout_Present'] == 1)])})
            system_counts.append({'Systems': 'Bank & ERP', 
                                'Count': len(system_df[(system_df['Bank_Present'] == 1) & 
                                                      (system_df['ERP_Present'] == 1) & 
                                                      (system_df['Payout_Present'] == 0)])})
            system_counts.append({'Systems': 'Bank & Payout', 
                                'Count': len(system_df[(system_df['Bank_Present'] == 1) & 
                                                      (system_df['ERP_Present'] == 0) & 
                                                      (system_df['Payout_Present'] == 1)])})
            system_counts.append({'Systems': 'ERP & Payout', 
                                'Count': len(system_df[(system_df['Bank_Present'] == 0) & 
                                                      (system_df['ERP_Present'] == 1) & 
                                                      (system_df['Payout_Present'] == 1)])})
            system_counts.append({'Systems': 'All Systems', 
                                'Count': len(system_df[(system_df['Bank_Present'] == 1) & 
                                                      (system_df['ERP_Present'] == 1) & 
                                                      (system_df['Payout_Present'] == 1)])})
            
            system_counts_df = pd.DataFrame(system_counts)
            
            fig = px.bar(
                system_counts_df,
                x='Systems',
                y='Count',
                title='Transaction Presence Across Systems',
                color='Systems',
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Explanation and recommendations
    st.markdown('<h2 class="sub-header">Analysis & Recommendations</h2>', unsafe_allow_html=True)
    
    # Create expandable sections
    with st.expander("Key Findings"):
        st.markdown(f"""
        Based on the reconciliation of {total_transactions} transactions across your financial systems:
        
        - **Reconciliation Rate**: {match_rate:.1f}% of transactions are fully matched across systems
        - **Financial Impact**: Total discrepancies amount to ₹{"{:,.2f}".format(mismatch_amount + missing_amount)}
        - **Primary Issues**:
          - {mismatch_count} transactions have data mismatches between systems
          - {missing_count} transactions are missing in one or more systems
        """)
    
    with st.expander("Common Patterns"):
        if not results_df.empty:
            # Identify the top 3 mismatch types
            mismatch_types = []
            for mismatches in results_df['Mismatches']:
                if mismatches:
                    mismatch_types.extend(mismatches)
            
            top_mismatches = pd.Series(mismatch_types).value_counts().head(3)
            
            st.markdown("""
            Common patterns identified in the reconciliation data:
            """)
            
            for mismatch, count in top_mismatches.items():
                st.markdown(f"- **{mismatch.replace('_', ' ')}**: Found in {count} transactions")
            
            # Provide pattern analysis
            if 'ERP_Amount_Mismatch' in mismatch_types:
                st.markdown("""
                - **Amount Discrepancies**: The ERP system consistently shows different amounts compared to bank data, 
                  suggesting potential calculation issues or manual entry errors.
                """)
            
            if 'Missing_In_ERP' in mismatch_types:
                st.markdown("""
                - **Missing ERP Entries**: Several transactions exist in the bank but are not recorded in the ERP system,
                  indicating possible delays in data entry or integration issues.
                """)
            
            if 'Payout_Date_Mismatch' in mismatch_types:
                st.markdown("""
                - **Timing Differences**: Date mismatches between payout system and other systems suggest 
                  reconciliation issues due to different processing dates.
                """)
            
            if 'Refund_Missing_In_CRM' in mismatch_types:
                st.markdown("""
                - **Refund Tracking Issues**: Refunds processed in financial systems are not consistently tracked
                  in the CRM, which can lead to customer service issues and financial reporting discrepancies.
                """)
    
    with st.expander("Actionable Recommendations"):
        st.markdown("""
        Based on the reconciliation results, here are key recommendations to improve financial data integrity:
        
        1. **Implement Automated Daily Reconciliation**
           - Set up this tool to run daily reconciliation checks
           - Configure alerts for high-impact discrepancies
        
        2. **Address System Integration Gaps**
           - Review API connections between systems
           - Ensure consistent data formats and field mappings
        
        3. **Standardize Financial Processing**
           - Create clear guidelines for transaction recording timelines
           - Implement validation rules to prevent amount discrepancies
        
        4. **Recover Identified Revenue Leakages**
           - Investigate transactions marked "Missing" to recover unrecorded revenue
           - Validate and correct amount mismatches
        
        5. **Improve Refund Tracking**
           - Create an automated workflow that ensures refunds are properly tracked in all systems
           - Implement verification steps before refund processing
        """)
    
    with st.expander("Next Steps"):
        st.markdown("""
        To maximize the value of this reconciliation analysis:
        
        1. **Schedule a Deep-Dive Workshop**
           - Review detailed findings with finance and IT teams
           - Prioritize issues based on financial impact
        
        2. **Develop System Enhancement Plan**
           - Address root causes of recurring discrepancies
           - Optimize data flow between systems
        
        3. **Set Up Continuous Monitoring**
           - Implement this dashboard as a permanent monitoring solution
           - Schedule weekly review of reconciliation metrics
        
        4. **Calculate Potential ROI**
           - Quantify labor savings from automated reconciliation
           - Estimate recoverable revenue from identified discrepancies
        
        **Contact your Galific account manager to schedule implementation support.**
        """)

# If the app isn't running reconciliation, show intro content
else:
    # Display welcome message and instructions
    st.markdown("""

    This powerful tool automatically reconciles financial data across your bank, ERP, CRM, and payout systems to:
    
    - **Detect mismatches** in transaction data between systems
    - **Identify potential revenue leakages** worth ₹10L-₹5Cr annually
    - **Reduce manual reconciliation effort** by up to 95%
    - **Minimize audit pressure** with clear, verifiable records
    
    ### How to use:
    
    1. Configure your data source in the sidebar (sample or your own)
    2. Adjust reconciliation settings if needed
    3. Click "Run Reconciliation" to analyze the data
    4. Explore the results through visualizations and detailed tables
    
    Ready to see how much revenue you could recover? **Get started by configuring the sidebar options!**
    """)
    
    # Display key features
    st.markdown("""
    ### Key Features
    
    - **Multi-System Reconciliation**: Simultaneously compare data across bank statements, ERP exports, payout systems, and CRM records
    - **Intelligent Matching**: Flexible matching criteria with adjustable tolerance for dates and amounts
    - **Comprehensive Analysis**: Visual dashboards and detailed reports highlighting discrepancies
    - **Financial Impact Assessment**: Quantify the monetary value of identified discrepancies
    - **Actionable Insights**: Clear recommendations based on reconciliation patterns
    
    Reconcile with confidence. Recover your revenue.
    """)