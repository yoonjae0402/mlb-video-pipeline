"""
MLB Video Pipeline - Monitoring Dashboard

Streamlit dashboard for monitoring pipeline status, costs, and outputs.

Run: streamlit run dashboard/streamlit_app.py
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd

from config.settings import settings


# =============================================================================
# Page Configuration
# =============================================================================

st.set_page_config(
    page_title="MLB Video Pipeline",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# Sidebar
# =============================================================================

with st.sidebar:
    st.title("⚾ MLB Pipeline")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["Overview", "Data", "Videos", "Costs", "Settings"],
        index=0,
    )

    st.markdown("---")
    st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")

    if st.button("Refresh"):
        st.rerun()


# =============================================================================
# Helper Functions
# =============================================================================

def load_json_file(path: Path) -> dict:
    """Load JSON file safely."""
    try:
        if path.exists():
            return json.loads(path.read_text())
    except Exception:
        pass
    return {}


def get_cost_data() -> dict:
    """Load cost tracking data."""
    return load_json_file(settings.logs_dir / "costs.json")


def get_usage_data() -> dict:
    """Load API usage data."""
    return load_json_file(settings.logs_dir / "api_usage.json")


def get_video_files() -> list[Path]:
    """Get list of generated videos."""
    video_dir = settings.outputs_dir / "videos"
    if video_dir.exists():
        return sorted(video_dir.glob("*.mp4"), reverse=True)
    return []


def get_data_files() -> list[Path]:
    """Get list of processed data files."""
    data_dir = settings.processed_data_dir
    if data_dir.exists():
        return sorted(data_dir.glob("*.*"), reverse=True)
    return []


# =============================================================================
# Overview Page
# =============================================================================

def show_overview():
    """Display overview page."""
    st.title("Pipeline Overview")

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    # Video count
    videos = get_video_files()
    col1.metric("Videos Generated", len(videos))

    # Data files
    data_files = get_data_files()
    col2.metric("Data Files", len(data_files))

    # Total cost
    costs = get_cost_data()
    total_cost = costs.get("total", 0)
    col3.metric("Total Spend", f"${total_cost:.2f}")

    # Daily limit
    daily_limit = settings.daily_cost_limit
    usage = get_usage_data()
    today = datetime.now().strftime("%Y-%m-%d")
    daily_cost = sum(
        api.get("cost", 0)
        for api in usage.get("daily_calls", {}).get(today, {}).values()
    )
    col4.metric(
        "Daily Budget",
        f"${daily_cost:.2f} / ${daily_limit:.2f}",
        delta=f"${daily_limit - daily_cost:.2f} remaining"
    )

    st.markdown("---")

    # Recent activity
    st.subheader("Recent Videos")

    if videos:
        for video in videos[:5]:
            col1, col2, col3 = st.columns([3, 1, 1])
            col1.text(video.name)
            col2.text(f"{video.stat().st_size / 1024 / 1024:.1f} MB")
            col3.text(datetime.fromtimestamp(video.stat().st_mtime).strftime("%Y-%m-%d %H:%M"))
    else:
        st.info("No videos generated yet.")

    # System status
    st.markdown("---")
    st.subheader("System Status")

    col1, col2 = st.columns(2)

    with col1:
        api_keys = settings.validate_api_keys()
        st.write("**API Keys Configured:**")
        for key, configured in api_keys.items():
            status = "✅" if configured else "❌"
            st.write(f"  {status} {key.title()}")

    with col2:
        st.write("**Directories:**")
        dirs = [
            ("Data", settings.data_dir),
            ("Models", settings.models_dir),
            ("Outputs", settings.outputs_dir),
            ("Logs", settings.logs_dir),
        ]
        for name, path in dirs:
            status = "✅" if path.exists() else "❌"
            st.write(f"  {status} {name}")


# =============================================================================
# Data Page
# =============================================================================

def show_data():
    """Display data management page."""
    st.title("Data Management")

    # Data files
    st.subheader("Processed Data Files")

    data_files = get_data_files()

    if data_files:
        for f in data_files[:10]:
            col1, col2, col3 = st.columns([4, 1, 1])
            col1.text(f.name)
            col2.text(f"{f.stat().st_size / 1024:.1f} KB")

            if st.button("View", key=f.name):
                if f.suffix == ".csv":
                    df = pd.read_csv(f)
                    st.dataframe(df)
                elif f.suffix == ".json":
                    data = json.loads(f.read_text())
                    st.json(data)
    else:
        st.info("No processed data files found.")

    # Cache management
    st.markdown("---")
    st.subheader("Cache Management")

    cache_dir = settings.cache_dir
    if cache_dir.exists():
        cache_files = list(cache_dir.glob("*.json"))
        st.write(f"Cache files: {len(cache_files)}")

        if st.button("Clear Cache"):
            for f in cache_files:
                f.unlink()
            st.success(f"Cleared {len(cache_files)} cache files")
            st.rerun()
    else:
        st.info("Cache directory not found.")


# =============================================================================
# Videos Page
# =============================================================================

def show_videos():
    """Display video management page."""
    st.title("Generated Videos")

    videos = get_video_files()

    if not videos:
        st.info("No videos generated yet.")
        return

    # Video list
    for video in videos:
        with st.expander(f"📹 {video.name}"):
            col1, col2 = st.columns(2)

            with col1:
                st.write(f"**Size:** {video.stat().st_size / 1024 / 1024:.1f} MB")
                st.write(f"**Created:** {datetime.fromtimestamp(video.stat().st_mtime).strftime('%Y-%m-%d %H:%M')}")

            with col2:
                # Check for related files
                base_name = video.stem
                audio_dir = settings.outputs_dir / "audio"
                script_dir = settings.outputs_dir / "scripts"
                thumb_dir = settings.outputs_dir / "thumbnails"

                related = []
                for pattern, dir_path in [
                    ("*.mp3", audio_dir),
                    ("*.txt", script_dir),
                    ("*.png", thumb_dir),
                ]:
                    if dir_path.exists():
                        matches = list(dir_path.glob(f"*{base_name}*"))
                        related.extend(matches)

                if related:
                    st.write("**Related files:**")
                    for r in related:
                        st.write(f"  - {r.name}")


# =============================================================================
# Costs Page
# =============================================================================

def show_costs():
    """Display cost tracking page."""
    st.title("Cost Tracking")

    costs = get_cost_data()
    usage = get_usage_data()

    # Overview
    col1, col2, col3 = st.columns(3)

    col1.metric("OpenAI", f"${costs.get('openai', 0):.2f}")
    col2.metric("ElevenLabs", f"${costs.get('elevenlabs', 0):.2f}")
    col3.metric("Total", f"${costs.get('total', 0):.2f}")

    st.markdown("---")

    # Daily breakdown
    st.subheader("Daily Usage")

    daily_calls = usage.get("daily_calls", {})

    if daily_calls:
        # Create dataframe
        records = []
        for date, apis in daily_calls.items():
            for api, data in apis.items():
                records.append({
                    "Date": date,
                    "API": api,
                    "Calls": data.get("calls", 0),
                    "Cost": data.get("cost", 0),
                })

        df = pd.DataFrame(records)

        if not df.empty:
            # Chart
            pivot = df.pivot_table(
                index="Date",
                columns="API",
                values="Cost",
                aggfunc="sum",
                fill_value=0
            )
            st.bar_chart(pivot)

            # Table
            st.dataframe(df.sort_values("Date", ascending=False))
    else:
        st.info("No usage data yet.")

    # Budget alerts
    st.markdown("---")
    st.subheader("Budget Settings")

    daily_limit = st.number_input(
        "Daily Cost Limit ($)",
        min_value=1.0,
        max_value=100.0,
        value=settings.daily_cost_limit,
        step=1.0,
    )

    if daily_limit != settings.daily_cost_limit:
        st.warning(f"To change the limit, update DAILY_COST_LIMIT in .env")


# =============================================================================
# Settings Page
# =============================================================================

def show_settings():
    """Display settings page."""
    st.title("Settings")

    # Current settings
    st.subheader("Current Configuration")

    settings_dict = {
        "Base Directory": str(settings.base_dir),
        "Data Directory": str(settings.data_dir),
        "Output Directory": str(settings.outputs_dir),
        "Video Dimensions": f"{settings.video_width}x{settings.video_height}",
        "Video FPS": settings.video_fps,
        "Log Level": settings.log_level,
        "Daily Cost Limit": f"${settings.daily_cost_limit}",
        "Dry Run Mode": settings.dry_run,
    }

    for key, value in settings_dict.items():
        st.write(f"**{key}:** {value}")

    # API key status
    st.markdown("---")
    st.subheader("API Keys")

    api_keys = settings.validate_api_keys()

    for key, configured in api_keys.items():
        if configured:
            st.success(f"{key.title()}: Configured")
        else:
            st.error(f"{key.title()}: Not configured")

    st.markdown("---")
    st.caption("Edit .env file to change configuration.")


# =============================================================================
# Main Router
# =============================================================================

if page == "Overview":
    show_overview()
elif page == "Data":
    show_data()
elif page == "Videos":
    show_videos()
elif page == "Costs":
    show_costs()
elif page == "Settings":
    show_settings()
