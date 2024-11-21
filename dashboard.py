import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import pandas as pd
from analytics import Analytics
from cloudwatch import get_service_req_counts
def render_dashboard(db):
    st.title("Usage Dashboard")
    
    # Date range selector
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", datetime.now() - timedelta(days=30))
    with col2:
        end_date = st.date_input("End Date", datetime.now())

    # Convert dates to ISO format for database query
    start_iso = start_date.isoformat()
    end_iso = end_date.isoformat()

    # Get usage data
    analytics = Analytics(db)
    usage_summary = analytics.get_usage_summary(st.session_state.user_id, start_iso, end_iso)

    # Display summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Cost", f"${usage_summary['total_cost']:.2f}")
    with col2:
        st.metric("Total Images Generated", int(usage_summary['total_images']))
    with col3:
        st.metric("Total Tokens Used", int(usage_summary['total_tokens']))

    # Get usage data and create DataFrame
    usage_data = db.get_user_usage(st.session_state.user_id, start_iso, end_iso)
    
    if usage_data:  # Only process if we have data
        # Convert DynamoDB items to DataFrame with explicit float conversion
        df = pd.DataFrame([
            {
                'date': item['timestamp'].split('T')[0],  # Extract date part
                'timestamp': item['timestamp'],
                'usage_type': item['usage_type'],
                'amount': float(item['amount']),  # Explicitly convert to float
                'cost': float(item['cost'])  # Explicitly convert to float
            }
            for item in usage_data
        ])
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['date'] = pd.to_datetime(df['date'])
        
        # Daily usage line chart
        daily_usage = df.groupby(['date', 'usage_type'])['amount'].sum().reset_index()
        
        if not daily_usage.empty:
            fig_line = px.line(
                daily_usage,
                x='date',
                y='amount',
                color='usage_type',
                title='Daily Usage Trends',
                labels={
                    'date': 'Date',
                    'amount': 'Amount',
                    'usage_type': 'Type'
                }
            )
            st.plotly_chart(fig_line)

            # Detailed usage table
            st.subheader("Detailed Usage")
            df_display = df.sort_values('timestamp', ascending=False)
            
            # Round cost column after converting to float
            df_display['cost'] = df_display['cost'].round(4)
            
            # Format the timestamp for display
            df_display['timestamp'] = df_display['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            st.dataframe(
                df_display[['timestamp', 'usage_type', 'amount', 'cost']],
                hide_index=True,
                column_config={
                    'timestamp': 'Time',
                    'usage_type': 'Type',
                    'amount': st.column_config.NumberColumn('Amount', format='%.2f'),
                    'cost': st.column_config.NumberColumn('Cost ($)', format='$%.4f')
                }
            )
        else:
            st.info("No usage trends available for the selected period.")
    else:
        st.info("No usage data available for the selected period.")
    if st.button("Cloud Usage"):
        st.subheader("Cloud Usage Metrics")
        
        # Fetch AWS CloudWatch metrics
        region = "ap-south-1"  # Adjust this as needed
        cloud_metrics = get_service_req_counts(region)
        
        # Display metrics in a table
        for service, metrics in cloud_metrics.items():
            st.markdown(f"### {service}")
            if metrics:
                df = pd.DataFrame(list(metrics.items()), columns=["Metric", "Value"])
                st.dataframe(df)
            else:
                st.write(f"No data available for {service}.")

    # Add usage explanation
    st.subheader("Usage Information")
    st.markdown("""
    **Cost Breakdown:**
    - Chat: $0.001 per 1000 tokens
    - Image Generation: $0.02 per image
    - Storage: $0.023 per GB per month
    
    Usage is tracked in real-time as you use the chat and image generation services.
    """)