"""
Aegis Dashboard - Main Application
Enterprise-grade trading analytics and monitoring dashboard
"""
import dash
from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# Initialize the Dash app with Bootstrap theme
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    suppress_callback_exceptions=True,
    title="Aegis Alpha",
    update_title="Updating..."
)

# Navigation bar
navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Overview", href="/")),
        dbc.NavItem(dbc.NavLink("Risk Analytics", href="/risk")),
        dbc.NavItem(dbc.NavLink("Alpha Analysis", href="/alpha")),
        dbc.NavItem(dbc.NavLink("Trades", href="/trades")),
    ],
    brand="Aegis Analytics",
    brand_href="#",
    color="primary",
    dark=True,
    className="mb-4",
)

# App layout
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    navbar,
    html.Div(id='page-content', className="p-4"),
    
    # Hidden div for storing data
    dcc.Store(id='portfolio-store'),
    dcc.Store(id='trades-store'),
    
    # Interval components
    dcc.Interval(
        id='interval-component',
        interval=5*1000,  # in milliseconds
        n_intervals=0
    )
])

# Import pages
from dashboard.pages import overview, risk, alpha, trades

# Page routing
@callback(Output('page-content', 'children'),
          [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/risk':
        return risk.layout
    elif pathname == '/alpha':
        return alpha.layout
    elif pathname == '/trades':
        return trades.layout
    else:
        return overview.layout

# Load sample data (replace with your data source)
@callback(
    [Output('portfolio-store', 'data'),
     Output('trades-store', 'data')],
    [Input('interval-component', 'n_intervals')]
)
def load_data(n):
    # Generate sample portfolio data
    dates = pd.date_range(end=datetime.now(), periods=100)
    portfolio = pd.DataFrame({
        'date': dates,
        'equity': np.cumprod(1 + np.random.normal(0.001, 0.02, 100)) * 10000,
        'drawdown': np.random.random(100) * 0.2,
        'sharpe': np.random.normal(2.5, 0.5, 100).clip(0, None)
    })
    
    # Generate sample trades
    trades = pd.DataFrame({
        'timestamp': pd.date_range(end=datetime.now(), periods=50),
        'symbol': ['BTC/USDT'] * 25 + ['ETH/USDT'] * 25,
        'side': np.random.choice(['LONG', 'SHORT'], 50),
        'size': np.random.uniform(0.1, 5, 50),
        'price': np.random.normal(50000, 1000, 50).round(2),
        'pnl': np.random.normal(50, 200, 50).round(2)
    })
    
    return portfolio.to_dict('records'), trades.to_dict('records')

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8050)
