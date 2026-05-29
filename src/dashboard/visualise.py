import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

_RADAR_ATTRS = ["pace", "shooting", "passing", "dribbling", "defending", "physical"]


def plot_scatter(df: pd.DataFrame) -> go.Figure:
    return px.scatter(
        df,
        x="passing",
        y="shooting",
        color="cluster",
        hover_data=["name", "team", "position"],
        title="Players — Passing vs Shooting",
    )


def plot_radar(player: pd.Series) -> go.Figure:
    attrs = [a for a in _RADAR_ATTRS if a in player.index]
    if not attrs:
        return go.Figure()
    values = [float(player[a]) for a in attrs]
    values_closed = values + [values[0]]
    attrs_closed = attrs + [attrs[0]]
    fig = go.Figure(
        go.Scatterpolar(r=values_closed, theta=attrs_closed, fill="toself", name=str(player.get("name", "")))
    )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=False,
    )
    return fig


def plot_team_form(form_series: pd.Series, team: str) -> go.Figure:
    fig = px.line(
        x=form_series.index,
        y=form_series.values,
        labels={"x": "Match", "y": "Form (0–1)"},
        title=f"{team} — Recent Form",
    )
    fig.update_traces(mode="lines+markers")
    return fig
