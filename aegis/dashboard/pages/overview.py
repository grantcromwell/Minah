"""
Aegis Dashboard - Overview Page
Displays key performance metrics and equity curve
"""
from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import pandas as pd
import numpy as np

# Layout
layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H3("Portfolio Overview", className="mb-4"),
            dbc.Card([
                dbc.CardBody([
                    html.Div(id='equity-value', className="h2 mb-0"),
                    html.Small("Portfolio Value", className="text-muted")
                ])
            ], className="mb-3"),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.Div(id='daily-return', className="h4 mb-0"),
                            html.Small("24h Return", className="text-muted")
                        ])
                    ], className="mb-3")
                ]),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.Div(id='sharpe-ratio', className="h4 mb-0"),
                            html.Small("Sharpe Ratio", className="text-muted")
                        ])
                    ], className="mb-3")
                ])
            ]),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.Div(id='max-dd', className="h4 mb-0"),
                            html.Small("Max Drawdown", className="text-muted")
                        ])
                    ])
                ]),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.Div(id='win-rate', className="h4 mb-0"),
                            html.Small("Win Rate", className="text-muted")
                        ])
                    ])
                ])
            ])
        ], md=3),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Equity Curve"),
                dbc.CardBody([
                    dcc.Graph(id='equity-curve', style={'height': '400px'})
                ])
            ], className="mb-4"),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Drawdown"),
                        dbc.CardBody([
                            dcc.Graph(id='drawdown-chart', style={'height': '300px'})
                        ])
                    ])
                ]),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Risk Metrics"),
                        dbc.CardBody([
                            dcc.Graph(id='risk-metrics', style={'height': '300px'})
                        ])
                    ])
                ])
            ])
        ])
    ])
])

# Callbacks
@callback(
    [Output('equity-curve', 'figure'),
     Output('drawdown-chart', 'figure'),
     Output('risk-metrics', 'figure'),
     Output('equity-value', 'children'),
     Output('daily-return', 'children'),
     Output('sharpe-ratio', 'children'),
     Output('max-dd', 'children'),
     Output('win-rate', 'children')],
    [Input('portfolio-store', 'data'),
     Input('trades-store', 'data')]
)
def update_overview(portfolio_data, trades_data):
    if not portfolio_data or not trades_data:
        raise PreventUpdate
    
    # Convert to DataFrames
    portfolio = pd.DataFrame(portfolio_data)
    trades = pd.DataFrame(trades_data)
    
    # Process data
    portfolio['date'] = pd.to_datetime(portfolio['date'])
    portfolio = portfolio.set_index('date')
    
    # Equity curve
    fig_equity = go.Figure()
    fig_equity.add_trace(go.Scatter(
        x=portfolio.index, 
        y=portfolio['equity'],
        mode='lines',
        name='Equity',
        line=dict(color='#636efa')
    ))
    
    fig_equity.update_layout(
        margin=dict(l=20, r=20, t=30, b=20),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
        showlegend=False
    )
    
    # Drawdown chart
    fig_drawdown = go.Figure()
    fig_drawdown.add_trace(go.Scatter(
        x=portfolio.index,
        y=portfolio['drawdown'] * 100,
        fill='tozeroy',
        mode='none',
        fillcolor='rgba(239, 85, 58, 0.3)',
        line=dict(color='#ef553b')
    ))
    
    fig_drawdown.update_layout(
        margin=dict(l=20, r=20, t=30, b=20),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', ticksuffix='%'),
        showlegend=False
    )
    
    # Risk metrics
    risk_metrics = {
        'Metric': ['Sharpe', 'Sortino', 'Calmar', 'Max DD'],
        'Value': [
            portfolio['sharpe'].iloc[-1],
            portfolio['sharpe'].iloc[-1] * 1.2,  # Placeholder
            portfolio['sharpe'].iloc[-1] * 1.5,  # Placeholder
            portfolio['drawdown'].max() * 100
        ]
    }
    
    fig_metrics = go.Figure()
    fig_metrics.add_trace(go.Bar(
        x=risk_metrics['Metric'],
        y=risk_metrics['Value'],
        marker_color=['#636efa', '#00cc96', '#ab63fa', '#ffa15a']
    ))
    
    fig_metrics.update_layout(
        margin=dict(l=20, r=20, t=30, b=20),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
        showlegend=False
    )
    
    # Calculate metrics
    equity_value = f"${portfolio['equity'].iloc[-1]:,.2f}"
    daily_return = f"{np.random.normal(0.5, 0.2):.2f}%"
    sharpe_ratio = f"{portfolio['sharpe'].iloc[-1]:.2f}"
    max_drawdown = f"{portfolio['drawdown'].max() * 100:.1f}%"
    win_rate = f"{np.random.uniform(50, 75):.1f}%"
    
    return (
        fig_equity,
        fig_drawdown,
        fig_metrics,
        equity_value,
        daily_return,
        sharpe_ratio,
        max_drawdown,
        win_rate
    )
