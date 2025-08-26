import dash
from dash import dcc, html
import plotly.express as px
import pandas as pd

from extensions.stage_6_trace_buffer import TraceBuffer
from extensions.stage_1_observability import get_observability_collector

# Initialize components
trace_buffer = TraceBuffer(max_size=2000)
collector = get_observability_collector()

# Sample data extraction
def get_trace_metrics():
    metrics = collector.get_metrics(event_type="trace_quality")
    return pd.DataFrame(metrics)

# Create Dash app
app = dash.Dash(__name__)
app.title = "Trace Quality Dashboard"

app.layout = html.Div([
    html.H1("🧠 AI Agent Trace Quality Dashboard"),
    dcc.Graph(id="confidence-plot"),
    dcc.Graph(id="reward-plot"),
    dcc.Graph(id="novelty-centrality-scatter"),
    dcc.Interval(id="interval-refresh", interval=5000, n_intervals=0)
])

@app.callback(
    dash.dependencies.Output("confidence-plot", "figure"),
    dash.dependencies.Output("reward-plot", "figure"),
    dash.dependencies.Output("novelty-centrality-scatter", "figure"),
    dash.dependencies.Input("interval-refresh", "n_intervals")
)
def update_dashboard(n):
    df = get_trace_metrics()
    if df.empty:
        return px.line(), px.line(), px.scatter()

    fig_conf = px.histogram(df, x="confidence", nbins=20, title="Confidence Distribution")
    fig_reward = px.histogram(df, x="reward", nbins=20, title="Reward Distribution")
    fig_scatter = px.scatter(df, x="novelty", y="centrality", color="confidence",
                             title="Novelty vs Centrality (Colored by Confidence)")

    return fig_conf, fig_reward, fig_scatter

if __name__ == "__main__":
    app.run_server(debug=True)

app.layout = html.Div([
    html.H1("🧠 AI Agent Trace Quality Dashboard"),
    
    html.Div([
        html.Label("Filter by Session ID"),
        dcc.Dropdown(id="session-filter", options=[], multi=False),
        
        html.Label("Filter by Trace Type"),
        dcc.Dropdown(id="trace-type-filter", options=[], multi=False),
    ], style={"width": "40%", "display": "inline-block", "verticalAlign": "top"}),

    dcc.Graph(id="confidence-plot"),
    dcc.Graph(id="reward-plot"),
    dcc.Graph(id="novelty-centrality-scatter"),
    dcc.Graph(id="reasoning-timeline"),

    dcc.Interval(id="interval-refresh", interval=5000, n_intervals=0)
])

@app.callback(
    dash.dependencies.Output("session-filter", "options"),
    dash.dependencies.Output("trace-type-filter", "options"),
    dash.dependencies.Input("interval-refresh", "n_intervals")
)
def update_filter_options(n):
    df = get_trace_metrics()
    session_ids = [{"label": s, "value": s} for s in df["session_id"].unique()]
    trace_types = [{"label": t, "value": t} for t in df["trace_type"].unique()]
    return session_ids, trace_types

@app.callback(
    dash.dependencies.Output("confidence-plot", "figure"),
    dash.dependencies.Output("reward-plot", "figure"),
    dash.dependencies.Output("novelty-centrality-scatter", "figure"),
    dash.dependencies.Output("reasoning-timeline", "figure"),
    dash.dependencies.Input("interval-refresh", "n_intervals"),
    dash.dependencies.Input("session-filter", "value"),
    dash.dependencies.Input("trace-type-filter", "value")
)
def update_dashboard(n, session_id, trace_type):
    df = get_trace_metrics()
    if session_id:
        df = df[df["session_id"] == session_id]
    if trace_type:
        df = df[df["trace_type"] == trace_type]

    fig_conf = px.histogram(df, x="confidence", nbins=20, title="Confidence Distribution")
    fig_reward = px.histogram(df, x="reward", nbins=20, title="Reward Distribution")
    fig_scatter = px.scatter(df, x="novelty", y="centrality", color="confidence",
                             title="Novelty vs Centrality")

    fig_timeline = px.line(df.sort_values("timestamp"), x="timestamp", y="reward",
                           color="trace_type", title="Timeline of Reasoning Rewards")

    return fig_conf, fig_reward, fig_scatter, fig_timeline
