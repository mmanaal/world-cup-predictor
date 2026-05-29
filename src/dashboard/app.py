import pandas as pd
import streamlit as st

from src.dashboard.cluster import cluster_players
from src.dashboard.visualise import plot_radar, plot_scatter

st.set_page_config(page_title="World Cup Player Dashboard", layout="wide")
st.title("World Cup 2026 — Player Dashboard")


@st.cache_data
def load_player_data(path: str = "data/processed/players.csv") -> pd.DataFrame:
    return pd.read_csv(path)


try:
    df = load_player_data()
except FileNotFoundError:
    st.warning("No player data found. Place `players.csv` in `data/processed/`.")
    st.stop()

with st.sidebar:
    st.header("Filters")
    teams = ["All"] + sorted(df["team"].unique().tolist())
    selected_team = st.selectbox("Team", teams)
    selected_positions = st.multiselect("Position", df["position"].unique().tolist())
    n_clusters = st.slider("Player clusters", min_value=2, max_value=8, value=4)

filtered = df.copy()
if selected_team != "All":
    filtered = filtered[filtered["team"] == selected_team]
if selected_positions:
    filtered = filtered[filtered["position"].isin(selected_positions)]

filtered = cluster_players(filtered, n_clusters=n_clusters)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Player Clusters — Passing vs Shooting")
    st.plotly_chart(plot_scatter(filtered), use_container_width=True)

with col2:
    player_names = filtered["name"].tolist()
    if player_names:
        selected_player = st.selectbox("Radar chart — select player", player_names)
        player_row = filtered[filtered["name"] == selected_player].iloc[0]
        st.subheader(f"{selected_player}")
        st.plotly_chart(plot_radar(player_row), use_container_width=True)

st.subheader("Player Table")
st.dataframe(filtered.drop(columns=["cluster"], errors="ignore"), use_container_width=True)
