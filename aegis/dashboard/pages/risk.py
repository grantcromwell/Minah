"""
Aegis Dashboard - Risk Analytics Page
Displays risk metrics including VaR, CVaR, and exposure analysis
"""
from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from scipy.stats import norm

# Layout
layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H3("Risk Analytics", className="mb-4"),
            dbc.Card([
                dbc.CardHeader("Value at Risk (VaR)"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Div(id='var-95', className="h3 text-center"),
                            html.Div("95% VaR", className="text-center text-muted")
                        ]),
                        dbc.Col([
                            html.Div(id='var-99', className="h3 text-center"),
                            html.Div("99% VaR", className="text-center text-muted")
                        ]),
                        dbc.Col([
                            html.Div(id='cvar-95', className="h3 text-center"),
                            html.Div("95% CVaR", className="text-center text-muted")
                        ]),
                        dbc.Col([
                            html.Div(id='cvar-99', className="h3 text-center"),
                            html.Div("99% CVaR", className="text-center text-muted")
                        ])
                    ])
                ])
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Portfolio Exposure"),
                        dbc.CardBody([
                            dcc.Graph(id='exposure-chart', style={'height': '300px'})
                        ])
                    ])
                ])
            ])
        ], md=6),
        
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Value at Risk (VaR) Analysis"),
                dbc.CardBody([
                    dcc.Graph(id='var-distribution', style={'height': '400px'})
                ])
            ], className="mb-4"),
            
            dbc.Card([
                dbc.CardHeader("Risk Contribution by Asset"),
                dbc.CardBody([
                    dcc.Graph(id='risk-contribution', style={'height': '300px'})
                ])
            ])
        ], md=6)
    ])
])

# Callbacks
@callback(
    [Output('var-distribution', 'figure'),
     Output('exposure-chart', 'figure'),
     Output('risk-contribution', 'figure'),
     Output('var-95', 'children'),
     Output('var-99', 'children'),
     Output('cvar-95', 'children'),
     Output('cvar-99', 'children')],
    [Input('portfolio-store', 'data'),
     Input('trades-store', 'data')]
)
def update_risk_analytics(portfolio_data, trades_data):
    if not portfolio_data or not trades_data:
        raise PreventUpdate
    
    # Convert to DataFrames
    portfolio = pd.DataFrame(portfolio_data)
    trades = pd.DataFrame(trades_data)
    
    # Generate sample returns (replace with actual returns)
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, 1000)
    
    # Calculate VaR and CVaR
    var_95 = np.percentile(returns, 5)
    var_99 = np.percentile(returns, 1)
    cvar_95 = returns[returns <= var_95].mean()
    cvar_99 = returns[returns <= var_99].mean()
    
    # VaR Distribution Chart
    fig_var = go.Figure()
    
    # Histogram of returns
    fig_var.add_trace(go.Histogram(
        x=returns,
        histnorm='probability density',
        name='Returns Distribution',
        marker_color='#636efa',
        opacity=0.7,
        nbinsx=100
    ))
    
    # Add VaR lines
    for var, conf in [(var_95, '95%'), (var_99, '99%')]:
        fig_var.add_vline(
            x=var,
            line=dict(color='red', dash='dash'),
            annotation_text=f"VaR {conf}: {var*100:.2f}%",
            annotation_position="top left"
        )
    
    fig_var.update_layout(
        title="Return Distribution with VaR",
        xaxis_title="Daily Return",
        yaxis_title="Density",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False, tickformat=".2%", range=[returns.min(), returns.max()]),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    # Exposure Chart (sample data)
    exposure_data = {
        'Asset': ['BTC', 'ETH', 'SOL', 'AVAX', 'Other'],
        'Exposure': [45, 30, 15, 7, 3]
    }
    
    fig_exposure = px.pie(
        exposure_data,
        values='Exposure',
        names='Asset',
        hole=0.4,
        color_discrete_sequence=px.colors.sequential.RdBu
    )
    
    fig_exposure.update_traces(
        textposition='inside',
        textinfo='percent+label',
        marker=dict(line=dict(color='#000000', width=1))
    )
    
    fig_exposure.update_layout(
        margin=dict(l=20, r=20, t=30, b=20),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=False
    )
    
    # Risk Contribution (sample data)
    risk_data = {
        'Asset': ['BTC', 'ETH', 'SOL', 'AVAX', 'Other'],
        'Contribution': [35, 25, 20, 15, 5],
        'Volatility': [0.02, 0.025, 0.03, 0.035, 0.04]
    }
    
    fig_risk = go.Figure()
    
    fig_risk.add_trace(go.Bar(
        x=risk_data['Asset'],
        y=risk_data['Contribution'],
        name='Risk Contribution',
        marker_color='#00cc96'
    ))
    
    fig_risk.add_trace(go.Scatter(
        x=risk_data['Asset'],
        y=[v * 2 for v in risk_data['Volatility']],
        name='Volatility (scaled)',
        mode='lines+markers',
        yaxis='y2',
        line=dict(color='#ff7f0e')
    ))
    
    fig_risk.update_layout(
        yaxis=dict(
            title='Risk Contribution (%)',
            showgrid=True,
            gridcolor='rgba(255,255,255,0.1)'
        ),
        yaxis2=dict(
            title='Volatility',
            overlaying='y',
            side='right',
            showgrid=False
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return (
        fig_var,
        fig_exposure,
        fig_risk,
        f"{var_95*100:.2f}%",
        f"{var_99*100:.2f}%",
        f"{cvar_95*100:.2f}%",
        f"{cvar_99*100:.2f}%"
    )
