# analytics_pages.py
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pandas as pd
from cloudwatch import get_service_req_counts
# from load_testing import LoadTester
def render_analytics_page(analytics):
    st.title("Resource Analytics")
    
    # Time range selector
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=datetime.now() - timedelta(days=30)
        )
    with col2:
        end_date = st.date_input(
            "End Date",
            value=datetime.now()
        )

    # Service selection
    selected_services = st.multiselect(
        "Select Services to Analyze",
        ["ChatGPT API", "Image Generation", "Storage", "All"],
        default="All"
    )

    # Get analytics data
    metrics = analytics.get_user_metrics(
        st.session_state.user_id,
        start_date,
        end_date
    )

    # Display cost overview
    st.subheader("Cost Overview")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Total Cost",
            f"${metrics['total_cost']:.2f}",
            delta=_calculate_cost_delta(analytics, st.session_state.user_id, start_date)
        )
    with col2:
        st.metric(
            "Average Daily Cost",
            f"${_calculate_average_daily_cost(metrics, start_date, end_date):.2f}"
        )
    with col3:
        st.metric(
            "Projected Monthly Cost",
            f"${_project_monthly_cost(metrics):.2f}"
        )
   # Add "Cloud Usage" Button
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
    # Resource usage graphs
    st.subheader("Resource Usage")
    
    # Create tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(["Usage Trends", "Cost Distribution", "Resource Metrics"])
    
    with tab1:
        usage_data = _get_usage_trends(analytics, st.session_state.user_id, start_date, end_date)
        fig = px.line(usage_data, x='timestamp', y='value', color='metric',
                     title='Resource Usage Over Time')
        st.plotly_chart(fig)

    with tab2:
        cost_data = _get_cost_distribution(metrics)
        fig = px.pie(cost_data, values='cost', names='service',
                    title='Cost Distribution by Service')
        st.plotly_chart(fig)

    with tab3:
        _render_resource_metrics(metrics)

 
# Helper functions
def _calculate_cost_delta(analytics, user_id, start_date):
    """Calculate cost change from previous period"""
    previous_start = start_date - timedelta(days=30)
    previous_metrics = analytics.get_user_metrics(user_id, previous_start, start_date)
    return float(previous_metrics['total_cost'])

def _calculate_average_daily_cost(metrics, start_date, end_date):
    """Calculate average daily cost"""
    days = (end_date - start_date).days or 1
    return float(metrics['total_cost']) / days

def _project_monthly_cost(metrics):
    """Project monthly cost based on current usage"""
    daily_cost = float(metrics['total_cost']) / 30  # Assuming 30 day period
    return daily_cost * 30

def _get_usage_trends(analytics, user_id, start_date, end_date):
    """Get usage trends for visualization"""
    # This would normally query your database for time-series data
    # Returning mock data for demonstration
    dates = pd.date_range(start_date, end_date, freq='D')
    data = {
        'timestamp': [],
        'value': [],
        'metric': []
    }
    
    for date in dates:
        data['timestamp'].extend([date] * 3)
        data['value'].extend([100 * (1 + date.day/30), 
                            50 * (1 + date.day/30),
                            200 * (1 + date.day/30)])
        data['metric'].extend(['API Calls', 'Images Generated', 'Storage Used'])
    
    return pd.DataFrame(data)

def _get_cost_distribution(metrics):
    """Get cost distribution data"""
    return pd.DataFrame({
        'service': ['ChatGPT API', 'Image Generation', 'Storage'],
        'cost': [
            metrics['total_chat_tokens'] * 0.002,
            metrics['total_images'] * 0.02,
            metrics['storage_used'] * 0.023
        ]
    })

def _render_resource_metrics(metrics):
    """Render detailed resource metrics"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("API Usage")
        st.metric("Total Tokens", metrics['total_chat_tokens'])
        st.metric("API Calls", metrics['total_images'])  # Assuming this is tracked
        
    with col2:
        st.subheader("Storage")
        st.metric("Storage Used", f"{metrics['storage_used']:.2f} GB")
        st.metric("Files Stored", metrics['total_images'])

def _get_latency_distribution(results):
    """Convert load test results to latency distribution data"""
    return pd.DataFrame({
        'latency': [results['avg_latency']] * 100  # Mock data
    })

def _get_endpoint_performance(results):
    """Get endpoint performance comparison data"""
    return pd.DataFrame({
        'endpoint': ['/chat', '/image-generation', '/dashboard'],
        'avg_latency': [results['avg_latency']] * 3,
        'p95_latency': [results['p95_latency']] * 3
    })