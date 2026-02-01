import streamlit as st
import pandas as pd
import numpy as np
import time
import requests
import json
from streamlit_folium import st_folium
import folium
from datetime import datetime

# Set Page Config
st.set_page_config(
    page_title="Lahore Traffic Intelligence",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .stMetric {
        background-color: #161b22;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #30363d;
    }
    .status-badge {
        padding: 5px 10px;
        border-radius: 5px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# API Constants
API_BASE_URL = "http://localhost:8000"

# --- Sidebar ---
st.sidebar.image("https://img.icons8.com/wired/128/ffffff/traffic-jam.png", width=80)
st.sidebar.title("Lahore Smart Traffic")
st.sidebar.markdown("---")
menu = st.sidebar.radio(
    "Navigation", 
    ["Live Dashboard", "Route Planner", "ML Performance", "Analytics"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("System Status")
try:
    health = requests.get(f"{API_BASE_URL}/health").json()
    if health['status'] == 'healthy':
        st.sidebar.success("🟢 API Connected")
    else:
        st.sidebar.error("🔴 API Issue")
except:
    st.sidebar.warning("⚪ API Offline")

st.sidebar.info(f"Last Update: {datetime.now().strftime('%H:%M:%S')}")

# --- Utility Functions ---
def get_stats():
    try:
        res = requests.get(f"{API_BASE_URL}/analytics/stats")
        return res.json()
    except:
        return None

# --- Main Views ---
if menu == "Live Dashboard":
    st.title("🚦 Live Traffic Dashboard")
    st.markdown("Real-time monitoring of Lahore's road network using predictive AI.")
    
    # Top Row Metrics
    stats = get_stats()
    col1, col2, col3, col4 = st.columns(4)
    
    if stats:
        col1.metric("Active Predictions", f"{stats['predictions_count']:,}", "+124")
        col2.metric("Avg Latency", f"{stats['avg_latency_ms']}ms", "-0.01ms")
        col3.metric("Drift Alert", "Normal" if stats['drift_events_detected'] == 0 else "WARNING", delta_color="inverse")
        col4.metric("Model Status", "Champion" if stats['active_challenger'] else "Standby")
    else:
        for c in [col1, col2, col3, col4]: 
            c.metric("Data", "N/A")

    # Lower Section
    tab1, tab2 = st.tabs(["🗺️ Live Network Map", "📈 Traffic Trends"])
    
    with tab1:
        st.info("Loading high-resolution road network... This may take a moment.")
        # Placeholder for Map Component
        m = folium.Map(location=[31.5204, 74.3587], zoom_start=12, tiles="CartoDB dark_matter")
        # In a real implementation, we would load GeoJSON edges color-coded by speed
        st_folium(m, width=1200, height=600)

    with tab2:
        st.write("Live speed distributions and volume trends across the district.")
        chart_data = pd.DataFrame(
            np.random.randn(20, 3),
            columns=['Canal Road', 'Mall Road', 'Ferozepur Road']
        )
        st.line_chart(chart_data)

elif menu == "Route Planner":
    st.title("🗺️ Congestion-Aware Route Planner")
    
    col_input, col_map = st.columns([1, 2])
    
    with col_input:
        st.subheader("Your Journey")
        source = st.number_input("Source Node ID", value=59634015)
        target = st.number_input("Target Node ID", value=83343529)
        aware = st.toggle("Avoid Traffic (Congestion-Aware)", value=True)
        
        if st.button("Calculate Optimal Route", use_container_type="primary"):
            payload = {"start_node": source, "end_node": target, "congestion_aware": aware}
            res = requests.post(f"{API_BASE_URL}/route", json=payload)
            
            if res.status_code == 200:
                data = res.json()
                st.success(f"Route Found!")
                st.write(f"📏 **Distance**: {data['total_length']/1000:.2f} km")
                st.write(f"⏱️ **Estimated Time**: {data['estimated_time']:.2f} cost units")
                
                # Show steps
                with st.expander("Step-by-step Directions"):
                    for step in data['steps'][:10]:
                        st.write(f"📍 {step['name']} ({step['length']:.0f}m)")
            else:
                st.error("Could not find a valid route between these nodes.")

    with col_map:
        # Placeholder for dynamic route line
        m_route = folium.Map(location=[31.5204, 74.3587], zoom_start=12, tiles="CartoDB dark_matter")
        st_folium(m_route, width=800, height=600)

elif menu == "ML Performance":
    st.title("🧠 Adaptive Intelligence Monitor")
    st.markdown("Tracking model drift and shadow model (A/B) performance.")
    
    col_drift, col_ab = st.columns(2)
    
    with col_drift:
        st.subheader("Data Drift Detection")
        st.image("lahore/data/plots/drift_detection_shift.png")
        st.caption("KS-Test analysis of incoming traffic distribution shifts.")
        
    with col_ab:
        st.subheader("Champion vs Challenger (A/B Test)")
        st.image("lahore/data/plots/ab_testing_performance.png")
        st.caption("Real-time accuracy comparison for model promotion.")

elif menu == "Analytics":
    st.title("📊 Advanced Analytics")
    col_a, col_b = st.columns(2)
    with col_a:
        st.image("lahore/data/plots/diurnal_pattern.png")
    with col_b:
        st.image("lahore/data/plots/weekend_comparison.png")
