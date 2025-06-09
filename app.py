import streamlit as st
import joblib
import pandas as pd
import numpy as np
import sys
import os
import time
from PIL import Image
from collections import defaultdict

# Set page configuration
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
    }
    .result-box {
        padding: 20px;
        border-radius: 5px;
        margin: 10px 0px;
    }
    .fraud {
        background-color: #ffcdd2;
        border: 2px solid #c62828;
    }
    .legitimate {
        background-color: #c8e6c9;
        border: 2px solid #2e7d32;
    }
    .info-text {
        font-size: 1rem;
    }
     
</style>
""", unsafe_allow_html=True)

# Function to load the model
@st.cache_resource
def load_model():
    try:
        model_path = 'optimized_model_v1.1.pkl'
        
        if os.path.exists(model_path):
            model_data = joblib.load(model_path)
            st.success("Model loaded successfully!")
            return model_data
        else:
            st.warning("Model file not found. Using default values for demonstration.")
            return {'threshold': 0.5, 'customer_stats_overall': None}

    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return {'threshold': 0.5, 'customer_stats_overall': None}

# Function to analyze transaction and calculate fraud score
def analyze_transaction(customer_id, amount, model_data):
    # Extract threshold and customer stats
    if isinstance(model_data, dict):
        threshold = model_data.get('threshold', 0.5)
        customer_stats_overall = model_data.get('customer_stats_overall')
    else:
        threshold = getattr(model_data, 'threshold', 0.5)
        customer_stats_overall = getattr(model_data, 'customer_stats_overall', None)
    
    # Get customer history if available
    customer_history = None
    if customer_stats_overall is not None:
        try:
            customer_history = customer_stats_overall[
                customer_stats_overall['customer_id'] == customer_id
            ]
        except Exception as e:
            st.error(f"Error getting customer history: {str(e)}")
            customer_history = None
    
    # Calculate basic statistics
    if customer_history is None or len(customer_history) == 0:
        customer_stats = {
            'transaction_count': 0,
            'avg_amount': 0,
            'max_amount': 0,
            'min_amount': 0,
            'total_amount': 0
        }
    else:
        # Get statistics from customer history
        stats = customer_history.iloc[0]
        customer_stats = {
            'transaction_count': stats['transaction_count'],
            'avg_amount': stats['avg_amount'],
            'max_amount': stats['max_amount'],
            'min_amount': stats['min_amount'],
            'total_amount': stats['total_amount']
        }
    
    # Calculate fraud indicators
    fraud_indicators = []
    
    # Calculate amount ratios
    if customer_stats['avg_amount'] > 0:
        amount_to_avg_ratio = amount / customer_stats['avg_amount']
        if amount_to_avg_ratio > 5.0:
            fraud_indicators.append(f"Amount is {amount_to_avg_ratio:.1f}x higher than customer's average")
        elif amount_to_avg_ratio > 3.0:
            fraud_indicators.append(f"Amount is {amount_to_avg_ratio:.1f}x higher than customer's average")
    else:
        amount_to_avg_ratio = 0
        if amount > 1000:
            fraud_indicators.append("High amount for a new customer")
    
    if customer_stats['max_amount'] > 0:
        amount_to_max_ratio = amount / customer_stats['max_amount']
        if amount_to_max_ratio > 1.5:
            fraud_indicators.append(f"Amount is {amount_to_max_ratio:.1f}x higher than customer's maximum")
        elif amount_to_max_ratio > 1.0:
            fraud_indicators.append(f"Amount exceeds customer's maximum")
    else:
        amount_to_max_ratio = 0
    
    # Calculate basic fraud score
    fraud_score = 0.0
    
    # New customer risk
    if customer_stats['transaction_count'] < 1:
        fraud_score += 0.5  # Moderate risk for new customers
    
    # Amount ratio risk
    if amount_to_avg_ratio > 5.0:
        fraud_score += 0.3  # High ratio to average
    elif amount_to_avg_ratio > 3.0:
        fraud_score += 0.2  # Moderately high ratio
    
    if amount_to_max_ratio > 1.5:
        fraud_score += 0.4  # Much higher than max
    elif amount_to_max_ratio > 1.0:
        fraud_score += 0.2  # Higher than max
    
    # Absolute amount risk
    if amount > 5000:
        fraud_score += 0.3  # Very high amount
    elif amount > 1000:
        fraud_score += 0.1  # High amount
    
    # Cap fraud score at 1.0
    fraud_score = min(fraud_score, 1.0)
    
    # Adjust for frequent customers (reduce risk)
    if customer_stats['transaction_count'] > 10:
        fraud_score *= 0.8  # 20% reduction for established customers
    
    # Predict fraud based on threshold
    predicted_fraud = fraud_score > threshold
    
    # Return all the data needed for display
    return {
        'predicted_fraud': predicted_fraud,
        'fraud_score': fraud_score,
        'threshold': threshold,
        'customer_stats': customer_stats,
        'fraud_indicators': fraud_indicators,
        'amount_to_avg_ratio': amount_to_avg_ratio if customer_stats['avg_amount'] > 0 else None,
        'amount_to_max_ratio': amount_to_max_ratio if customer_stats['max_amount'] > 0 else None
    }

# Function to display the prediction result
def display_result(analysis_result):
    predicted_fraud = analysis_result['predicted_fraud']
    fraud_score = analysis_result['fraud_score']
    
    if predicted_fraud:
        st.markdown('<div class="result-box fraud">', unsafe_allow_html=True)
        st.error("⚠️ **Potential Fraud Detected!**")
        st.write("This transaction has been flagged as potentially fraudulent and should be reviewed.")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="result-box legitimate">', unsafe_allow_html=True)
        st.success("✅ **Transaction Appears Legitimate**")
        st.write("This transaction appears to be legitimate based on our analysis.")
        st.markdown('</div>', unsafe_allow_html=True)

# Function to simulate processing
def simulate_processing():
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    stages = [
        "Initializing fraud detection system...",
        "Analyzing transaction patterns...",
        "Comparing with historical data...",
        "Calculating fraud indicators...",
        "Finalizing fraud assessment..."
    ]
    
    for i, stage in enumerate(stages):
        # Update progress bar and status text
        progress = (i+1) / len(stages)
        progress_bar.progress(progress)
        status_text.text(stage)
        time.sleep(0.5)
    
    progress_bar.empty()
    status_text.empty()

# Header section
st.markdown('<h1 class="main-header">Credit Card Fraud Detection System</h1>', unsafe_allow_html=True)

# Create sidebar for additional information
with st.sidebar:
    st.image("img/logo_app.png", width=100)
    st.markdown("## About")
    st.info(
        "This application uses AI and Data science models to detect potentially " 
        "fraudulent credit card transactions based on customer ID and transaction amount."
    )
    
    st.markdown("## How It Works")
    st.write(
        "1. Enter a customer ID\n"
        "2. Enter a transaction amount\n"
        "3. Click 'Check Transaction'\n"
        "4. The system will analyze the data and provide a fraud assessment"
    )
    


# Create two columns for the input form
col1, col2 = st.columns(2)

# Input form
with st.container():
    st.markdown('<h2 class="sub-header">Transaction Details</h2>', unsafe_allow_html=True)
    
    with col1:
        customer_id = st.number_input("Customer ID", min_value=0, help="Enter the customer's unique identifier")
    
    with col2:
        amount = st.number_input("Transaction Amount ($)", min_value=0.00, step=10.0, format="%.2f", help="Enter the transaction amount")

    # Additional transaction details that could be collected in a real app
   

# Submit button
if st.button("Check Transaction", type="primary", use_container_width=True):
    with st.spinner("Analyzing transaction..."):
        # Load the model data
        model_data = load_model()
        
        # Simulate processing for better UX
        simulate_processing()
        
        # Analyze transaction
        analysis_result = analyze_transaction(customer_id, amount, model_data)
        
        # Display transaction details
        st.markdown('<h2 class="sub-header">Analysis Result</h2>', unsafe_allow_html=True)
        st.write("**Transaction Summary:**")
        st.write(f"* Customer ID: {customer_id}")
        st.write(f"* Amount: ${amount:.2f}")
        
        # Display prediction
        display_result(analysis_result)
        
        # Display detailed analysis
        st.markdown("### Detailed Analysis")
        
        # Customer history info
        if analysis_result['customer_stats']['transaction_count'] > 0:
            st.write("**Customer History:**")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Previous Transactions", int(analysis_result['customer_stats']['transaction_count']))
                st.metric("Total Spent", f"${analysis_result['customer_stats']['total_amount']:.2f}")
            with col2:
                st.metric("Average Amount", f"${analysis_result['customer_stats']['avg_amount']:.2f}")
                if analysis_result['amount_to_avg_ratio']:
                    st.metric("Amount/Avg Ratio", f"{analysis_result['amount_to_avg_ratio']:.2f}x")
            with col3:
                st.metric("Maximum Amount", f"${analysis_result['customer_stats']['max_amount']:.2f}")
                if analysis_result['amount_to_max_ratio']:
                    st.metric("Amount/Max Ratio", f"{analysis_result['amount_to_max_ratio']:.2f}x")
        else:
            st.info("No transaction history for this customer.")
        
        # Risk indicators
        if analysis_result['fraud_indicators']:
            st.write("**Risk Indicators:**")
            for indicator in analysis_result['fraud_indicators']:
                st.warning(f"• {indicator}")
        else:
            st.success("No risk indicators detected.")
        
        # Recommendation based on the prediction
        if analysis_result['predicted_fraud']:
            st.markdown("### Recommended Actions")
            st.markdown("""
            * Contact the account holder immediately
            * Temporarily freeze the account
            * Verify recent transactions with the customer
            * Request additional verification for future transactions
            """)
        else:
            # Show a recent transactions table for context (mock data)
            st.markdown("")
           
footer_style = """
    <style>
        footer {
            visibility: hidden;
        }
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: dark;
            text-align: center;
            padding: 10px;
            font-size: 18px;
        }
    </style>
    <div class="footer">
        © 2025 OrbiDefence
    </div>
"""

# Inject CSS with Streamlit
st.markdown(footer_style, unsafe_allow_html=True)

   
   


