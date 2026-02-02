import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
from streamlit_folium import st_folium
import folium
from folium.plugins import HeatMap
from datetime import datetime
import asyncio
import websockets
import threading
from queue import Queue

# Page Configuration
st.set_page_config(
    page_title="Lahore Traffic Intelligence",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern Professional CSS
st.markdown("""
<style>
    /* Import Inter font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* CSS Variables */
    :root {
        --bg-primary: #ffffff;
        --bg-secondary: #f8fafc;
        --bg-tertiary: #f1f5f9;
        --text-primary: #0f172a;
        --text-secondary: #475569;
        --text-muted: #94a3b8;
        --border: #e2e8f0;
        --accent-primary: #2563eb;
        --accent-primary-light: #dbeafe;
        --accent-success: #10b981;
        --accent-success-light: #d1fae5;
        --accent-warning: #f59e0b;
        --accent-warning-light: #fef3c7;
        --accent-danger: #ef4444;
        --accent-danger-light: #fee2e2;
        --radius-sm: 6px;
        --radius-md: 8px;
        --radius-lg: 12px;
        --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05);
        --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.1);
    }

    /* Global Styles */
    .main {
        background-color: var(--bg-secondary) !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    .block-container {
        padding: 2rem 2.5rem !important;
        max-width: 1400px !important;
    }

    /* Sidebar Styling - Force Light Theme */
    [data-testid="stSidebar"] {
        background: #ffffff !important;
        border-right: 1px solid #e2e8f0 !important;
    }

    [data-testid="stSidebar"] > div:first-child {
        padding: 1.5rem 1rem !important;
        background: #ffffff !important;
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
        background: transparent !important;
    }

    section[data-testid="stSidebar"] > div {
        background: #ffffff !important;
    }

    /* Sidebar Brand */
    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding: 0.5rem 0 1.5rem 0;
        border-bottom: 1px solid var(--border);
        margin-bottom: 1.5rem;
    }

    .sidebar-brand-icon {
        width: 40px;
        height: 40px;
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        border-radius: var(--radius-md);
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-size: 1.25rem;
    }

    .sidebar-brand-text {
        font-size: 1rem;
        font-weight: 600;
        color: #0f172a;
        line-height: 1.2;
    }

    .sidebar-brand-subtitle {
        font-size: 0.75rem;
        color: #64748b;
        font-weight: 400;
    }

    /* Navigation Label */
    .nav-label {
        font-size: 0.7rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.75rem;
        padding-left: 0.25rem;
    }

    /* Radio buttons in sidebar - hide default label */
    [data-testid="stSidebar"] .stRadio > label {
        display: none !important;
    }

    [data-testid="stSidebar"] .stRadio > div {
        display: flex !important;
        flex-direction: column !important;
        gap: 0.375rem !important;
    }

    [data-testid="stSidebar"] .stRadio > div > label {
        font-family: 'Inter', sans-serif !important;
        background: transparent !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.625rem 0.875rem !important;
        transition: all 0.15s ease !important;
        cursor: pointer !important;
        display: flex !important;
        align-items: center !important;
        gap: 0.5rem !important;
    }

    [data-testid="stSidebar"] .stRadio > div > label:hover {
        background: #f1f5f9 !important;
    }

    [data-testid="stSidebar"] .stRadio > div > label[data-checked="true"] {
        background: #dbeafe !important;
    }

    /* Radio button text - force visibility */
    [data-testid="stSidebar"] .stRadio > div > label p,
    [data-testid="stSidebar"] .stRadio > div > label span {
        font-family: 'Inter', sans-serif !important;
        font-size: 0.875rem !important;
        font-weight: 500 !important;
        color: #475569 !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    [data-testid="stSidebar"] .stRadio > div > label[data-checked="true"] p,
    [data-testid="stSidebar"] .stRadio > div > label[data-checked="true"] span {
        color: #2563eb !important;
    }

    /* Hide the radio circle indicator */
    [data-testid="stSidebar"] .stRadio > div > label > div:first-child {
        display: none !important;
    }

    /* System Status Box */
    .system-status {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
        margin-top: 1.5rem;
    }

    .system-status-title {
        font-size: 0.7rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.75rem;
    }

    .status-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.375rem 0;
    }

    .status-label {
        font-size: 0.8125rem;
        color: #475569;
    }

    .status-indicator {
        display: flex;
        align-items: center;
        gap: 0.375rem;
        font-size: 0.8125rem;
        font-weight: 500;
        color: #1e293b !important;
    }

    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        flex-shrink: 0;
    }

    .status-dot.online { background: #10b981; }
    .status-dot.offline { background: #ef4444; }

    /* Page Header */
    .page-header {
        display: flex;
        align-items: flex-start;
        gap: 1rem;
        margin-bottom: 2rem;
        padding-bottom: 1.5rem;
        border-bottom: 1px solid var(--border);
    }

    .page-icon {
        width: 48px;
        height: 48px;
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        border-radius: var(--radius-lg);
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-size: 1.5rem;
        flex-shrink: 0;
    }

    .page-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--text-primary);
        margin: 0;
        line-height: 1.2;
    }

    .page-subtitle {
        font-size: 0.875rem;
        color: var(--text-muted);
        margin-top: 0.25rem;
    }

    /* Metric Cards */
    .metric-card {
        background: var(--bg-primary);
        border: 1px solid var(--border);
        border-radius: var(--radius-lg);
        padding: 1.25rem;
        box-shadow: var(--shadow-sm);
    }

    .metric-label {
        font-size: 0.75rem;
        font-weight: 500;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.025em;
        margin-bottom: 0.5rem;
    }

    .metric-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: var(--text-primary);
        line-height: 1;
    }

    .metric-delta {
        font-size: 0.75rem;
        font-weight: 500;
        margin-top: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.25rem;
    }

    .metric-delta.positive { color: var(--accent-success); }
    .metric-delta.negative { color: var(--accent-danger); }
    .metric-delta.neutral { color: var(--text-muted); }

    /* Section Headers */
    .section-header {
        font-size: 0.7rem;
        font-weight: 600;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin: 2rem 0 1rem 0;
    }

    /* Content Cards */
    .content-card {
        background: var(--bg-primary);
        border: 1px solid var(--border);
        border-radius: var(--radius-lg);
        overflow: hidden;
        box-shadow: var(--shadow-sm);
    }

    .card-header {
        padding: 1rem 1.25rem;
        border-bottom: 1px solid var(--border);
        background: var(--bg-secondary);
    }

    .card-title {
        font-size: 0.875rem;
        font-weight: 600;
        color: var(--text-primary);
    }

    .card-body {
        padding: 1.25rem;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background: var(--bg-secondary);
        padding: 0.375rem;
        border-radius: var(--radius-md);
        border: 1px solid var(--border);
    }

    .stTabs [data-baseweb="tab"] {
        font-family: 'Inter', sans-serif !important;
        font-size: 0.8125rem !important;
        font-weight: 500 !important;
        padding: 0.5rem 1rem !important;
        border-radius: var(--radius-sm) !important;
        color: var(--text-secondary) !important;
        background: transparent !important;
    }

    .stTabs [data-baseweb="tab"]:hover {
        background: var(--bg-tertiary) !important;
    }

    .stTabs [aria-selected="true"] {
        background: var(--bg-primary) !important;
        color: var(--text-primary) !important;
        box-shadow: var(--shadow-sm) !important;
    }

    .stTabs [data-baseweb="tab-highlight"] {
        display: none !important;
    }

    .stTabs [data-baseweb="tab-border"] {
        display: none !important;
    }

    /* Button Styling */
    .stButton > button {
        font-family: 'Inter', sans-serif !important;
        font-weight: 500 !important;
        border-radius: var(--radius-md) !important;
        padding: 0.5rem 1rem !important;
        transition: all 0.15s ease !important;
    }

    .stButton > button[kind="primary"] {
        background: var(--accent-primary) !important;
        border: none !important;
    }

    .stButton > button[kind="primary"]:hover {
        background: #1d4ed8 !important;
    }

    /* Input Styling */
    .stNumberInput > div > div > input,
    .stTextInput > div > div > input {
        font-family: 'Inter', sans-serif !important;
        border-radius: var(--radius-md) !important;
        border: 1px solid var(--border) !important;
    }

    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Chart containers */
    .stPlotlyChart, .stLineChart {
        background: var(--bg-primary);
        border: 1px solid var(--border);
        border-radius: var(--radius-lg);
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# API Configuration
API_BASE_URL = "http://localhost:8000"
WS_URL = "ws://localhost:8000/ws/traffic"

# Initialize session state
if 'ws_data' not in st.session_state:
    st.session_state.ws_data = {
        'predictions': 0,
        'avg_speed': 0,
        'congestion_level': 'Normal',
        'active_alerts': 0
    }

if 'api_status' not in st.session_state:
    st.session_state.api_status = 'offline'

# --- Utility Functions ---
def check_api_health():
    """Check API health status"""
    try:
        res = requests.get(f"{API_BASE_URL}/health", timeout=2)
        if res.status_code == 200:
            return 'online'
    except:
        pass
    return 'offline'

def get_stats():
    """Fetch analytics stats from API"""
    try:
        res = requests.get(f"{API_BASE_URL}/analytics/stats", timeout=5)
        if res.status_code == 200:
            return res.json()
    except:
        pass
    return None

def get_realtime_data():
    """Fetch real-time traffic data"""
    try:
        res = requests.get(f"{API_BASE_URL}/realtime/summary", timeout=5)
        if res.status_code == 200:
            return res.json()
    except:
        pass
    return None

# --- Sidebar ---
st.sidebar.markdown("""
<div class="sidebar-brand">
    <div class="sidebar-brand-icon">🚦</div>
    <div>
        <div class="sidebar-brand-text">Lahore Traffic</div>
        <div class="sidebar-brand-subtitle">Urban Intelligence Platform</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown('<div class="nav-label">Navigation</div>', unsafe_allow_html=True)

menu = st.sidebar.radio(
    "Navigation",
    ["Live Dashboard", "Route Planner", "ML Performance", "Analytics"],
    label_visibility="collapsed"
)

# System Status
api_status = check_api_health()
st.session_state.api_status = api_status

st.sidebar.markdown(f"""
<div class="system-status">
    <div class="system-status-title">System Status</div>
    <div class="status-row">
        <span class="status-label">API Status</span>
        <span class="status-indicator">
            <span class="status-dot {'online' if api_status == 'online' else 'offline'}"></span>
            {'Online' if api_status == 'online' else 'Offline'}
        </span>
    </div>
    <div class="status-row">
        <span class="status-label">Last Updated</span>
        <span class="status-indicator">{datetime.now().strftime('%H:%M:%S')}</span>
    </div>
</div>
""", unsafe_allow_html=True)

# --- Main Views ---
if menu == "Live Dashboard":
    # Page Header
    st.markdown("""
    <div class="page-header">
        <div class="page-icon">🚦</div>
        <div>
            <h1 class="page-title">Live Traffic Dashboard</h1>
            <p class="page-subtitle">Real-time monitoring of Lahore's road network</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Metrics Row
    stats = get_stats()
    realtime = get_realtime_data()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Data</div>
            <div class="metric-value">{}</div>
            <div class="metric-delta neutral">Awaiting connection</div>
        </div>
        """.format(stats['predictions_count'] if stats else "—"), unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Data</div>
            <div class="metric-value">{}</div>
            <div class="metric-delta neutral">Awaiting connection</div>
        </div>
        """.format(f"{stats['avg_latency_ms']}ms" if stats else "—"), unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Data</div>
            <div class="metric-value">{}</div>
            <div class="metric-delta neutral">Awaiting connection</div>
        </div>
        """.format("Normal" if stats and stats.get('drift_events_detected', 0) == 0 else "—"), unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Data</div>
            <div class="metric-value">{}</div>
            <div class="metric-delta neutral">Awaiting connection</div>
        </div>
        """.format("Active" if stats else "—"), unsafe_allow_html=True)

    # Map Section
    st.markdown('<div class="section-header">Network Visualization</div>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Live Network Map", "Traffic Trends"])

    with tab1:
        # Create Folium map with modern styling
        m = folium.Map(
            location=[31.5204, 74.3587],
            zoom_start=12,
            tiles="CartoDB positron"
        )

        # Add sample markers for key locations
        locations = [
            {"name": "Walled City", "lat": 31.5820, "lng": 74.3239, "status": "moderate"},
            {"name": "Gulberg", "lat": 31.5129, "lng": 74.3426, "status": "clear"},
            {"name": "DHA", "lat": 31.4697, "lng": 74.4078, "status": "clear"},
            {"name": "Model Town", "lat": 31.4833, "lng": 74.3166, "status": "congested"},
        ]

        for loc in locations:
            color = {"clear": "green", "moderate": "orange", "congested": "red"}[loc["status"]]
            folium.CircleMarker(
                location=[loc["lat"], loc["lng"]],
                radius=10,
                popup=f"{loc['name']}: {loc['status'].title()}",
                color=color,
                fill=True,
                fillColor=color,
                fillOpacity=0.7
            ).add_to(m)

        st_folium(m, height=500, use_container_width=True)

    with tab2:
        st.markdown("**Speed Distribution Across Major Corridors**")

        # Generate sample trend data
        chart_data = pd.DataFrame({
            'Time': pd.date_range(start='06:00', periods=18, freq='h'),
            'Canal Road': np.random.uniform(25, 55, 18),
            'Mall Road': np.random.uniform(20, 45, 18),
            'Ferozepur Road': np.random.uniform(15, 40, 18)
        })
        chart_data.set_index('Time', inplace=True)

        st.line_chart(chart_data, height=400)

elif menu == "Route Planner":
    # Page Header
    st.markdown("""
    <div class="page-header">
        <div class="page-icon">🗺️</div>
        <div>
            <h1 class="page-title">Congestion-Aware Route Planner</h1>
            <p class="page-subtitle">Find optimal routes using real-time traffic data</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_input, col_map = st.columns([1, 2])

    with col_input:
        st.markdown("""
        <div class="content-card">
            <div class="card-header">
                <div class="card-title">Route Configuration</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        source = st.number_input("Source Node ID", value=59634015)
        target = st.number_input("Target Node ID", value=83343529)
        aware = st.toggle("Congestion-Aware Routing", value=True)

        if st.button("Calculate Route", type="primary", use_container_width=True):
            try:
                payload = {"start_node": source, "end_node": target, "congestion_aware": aware}
                res = requests.post(f"{API_BASE_URL}/route", json=payload, timeout=10)

                if res.status_code == 200:
                    data = res.json()
                    st.success("Route calculated successfully")

                    st.markdown(f"""
                    <div class="metric-card" style="margin-top: 1rem;">
                        <div class="metric-label">Total Distance</div>
                        <div class="metric-value">{data['total_length']/1000:.2f} km</div>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown(f"""
                    <div class="metric-card" style="margin-top: 0.5rem;">
                        <div class="metric-label">Estimated Time</div>
                        <div class="metric-value">{data['estimated_time']:.1f} min</div>
                    </div>
                    """, unsafe_allow_html=True)

                    with st.expander("Route Details"):
                        for i, step in enumerate(data.get('steps', [])[:10], 1):
                            st.write(f"{i}. {step['name']} ({step['length']:.0f}m)")
                else:
                    st.error("Could not calculate route")
            except Exception as e:
                st.error("API connection failed")

    with col_map:
        m_route = folium.Map(
            location=[31.5204, 74.3587],
            zoom_start=12,
            tiles="CartoDB positron"
        )

        # Add sample route visualization
        folium.Marker(
            [31.5820, 74.3239],
            popup="Start",
            icon=folium.Icon(color="green", icon="play")
        ).add_to(m_route)

        folium.Marker(
            [31.4697, 74.4078],
            popup="End",
            icon=folium.Icon(color="red", icon="stop")
        ).add_to(m_route)

        st_folium(m_route, height=550, use_container_width=True)

elif menu == "ML Performance":
    # Page Header
    st.markdown("""
    <div class="page-header">
        <div class="page-icon">🧠</div>
        <div>
            <h1 class="page-title">ML Performance Monitor</h1>
            <p class="page-subtitle">Model drift detection and A/B testing analytics</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_drift, col_ab = st.columns(2)

    with col_drift:
        st.markdown("""
        <div class="content-card">
            <div class="card-header">
                <div class="card-title">Data Drift Detection</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        try:
            st.image("lahore/data/plots/drift_detection_shift.png", width='stretch')
            st.caption("KS-Test analysis of incoming traffic distribution shifts")
        except:
            st.info("Drift detection visualization not available")

    with col_ab:
        st.markdown("""
        <div class="content-card">
            <div class="card-header">
                <div class="card-title">Champion vs Challenger</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        try:
            st.image("lahore/data/plots/ab_testing_performance.png", width='stretch')
            st.caption("Real-time accuracy comparison for model promotion")
        except:
            st.info("A/B testing visualization not available")

elif menu == "Analytics":
    # Page Header
    st.markdown("""
    <div class="page-header">
        <div class="page-icon">📊</div>
        <div>
            <h1 class="page-title">Traffic Analytics</h1>
            <p class="page-subtitle">Historical patterns and statistical insights</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("""
        <div class="content-card">
            <div class="card-header">
                <div class="card-title">Diurnal Traffic Patterns</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        try:
            st.image("lahore/data/plots/diurnal_pattern.png", width='stretch')
        except:
            st.info("Diurnal pattern visualization not available")

    with col_b:
        st.markdown("""
        <div class="content-card">
            <div class="card-header">
                <div class="card-title">Weekday vs Weekend</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        try:
            st.image("lahore/data/plots/weekend_comparison.png", width='stretch')
        except:
            st.info("Weekend comparison visualization not available")
