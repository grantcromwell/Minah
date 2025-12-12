"""
Aegis Dashboard - Alpha Analysis Page
Displays performance attribution and alpha generation metrics
"""
from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from scipy import stats

# Layout
layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H3("Alpha Analysis", className="mb-4"),
            dbc.Card([
                dbc.CardHeader("Performance Attribution"),
                dbc.CardBody([
                    dcc.Graph(id='alpha-returns', style={'height': '400px'}),
                    html.Hr(),
                    dbc.Row([
                        dbc.Col([
                            html.Div(id='alpha-value', className="h3 text-center"),
                            html.Div("Annualized Alpha", className="text-center text-muted")
                        ]),
                        dbc.Col([
                            html.Div(id='beta-value', className="h3 text-center"),
                            html.Div("Beta", className="text-center text-muted")
                        ]),
                        dbc.Col([
                            html.Div(id='r-squared', className="h3 text-center"),
                            html.Div("R²", className="text-center text-muted")
                        ]),
                        dbc.Col([
                            html.Div(id='info-ratio', className="h3 text-center"),
                            html.Div("Information Ratio", className="text-center text-muted")
                        ])
                    ], className="mt-3")
                ])
            ], className="mb-4"),
            
            dbc.Card([
                dbc.CardHeader("Factor Exposure"),
                dbc.CardBody([
                    dcc.Graph(id='factor-exposure', style={'height': '300px'})
                ])
            ])
        ], md=6),
        
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Rolling Performance"),
                dbc.CardBody([
                    dcc.Graph(id='rolling-performance', style={'height': '350px'})
                ])
            ], className="mb-4"),
            
            dbc.Card([
                dbc.CardHeader("Drawdown Analysis"),
                dbc.CardBody([
                    dcc.Graph(id='drawdown-analysis', style={'height': '350px'})
                ])
            ])
        ], md=6)
    ])
])

# Callbacks
@callback(
    [Output('alpha-returns', 'figure'),
     Output('rolling-performance', 'figure'),
     Output('drawdown-analysis', 'figure'),
     Output('factor-exposure', 'figure'),
     Output('alpha-value', 'children'),
     Output('beta-value', 'children'),
     Output('r-squared', 'children'),
     Output('info-ratio', 'children')],
    [Input('portfolio-store', 'data'),
     Input('trades-store', 'data')]
)
def update_alpha_analysis(portfolio_data, trades_data):
    if not portfolio_data or not trades_data:
        raise PreventUpdate
    
    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range(end=pd.Timestamp.now(), periods=252)
    
    # Strategy returns (slightly better than benchmark with some correlation)
    benchmark_returns = np.random.normal(0.0005, 0.015, 252)
    strategy_returns = benchmark_returns * 0.6 + np.random.normal(0.0003, 0.01, 252)
    
    # Create DataFrames
    returns = pd.DataFrame({
        'Date': dates,
        'Strategy': np.cumprod(1 + strategy_returns) - 1,
        'Benchmark': np.cumprod(1 + benchmark_returns) - 1,
        'Active': np.cumprod(1 + (strategy_returns - benchmark_returns)) - 1
    })
    
    # Calculate performance metrics
    def calculate_metrics(strat_ret, bench_ret):
        excess_returns = strat_ret - bench_ret
        beta, alpha, _, _, _ = stats.linregress(bench_ret, strat_ret)
        r_squared = beta ** 2 * np.var(bench_ret) / np.var(strat_ret)
        info_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
        return {
            'alpha': alpha * 252,
            'beta': beta,
            'r_squared': r_squared,
            'info_ratio': info_ratio
        }
    
    metrics = calculate_metrics(strategy_returns, benchmark_returns)
    
    # Alpha vs Benchmark chart
    fig_alpha = go.Figure()
    
    fig_alpha.add_trace(go.Scatter(
        x=returns['Date'],
        y=returns['Strategy'] * 100,
        name='Strategy',
        line=dict(color='#636efa')
    ))
    
    fig_alpha.add_trace(go.Scatter(
        x=returns['Date'],
        y=returns['Benchmark'] * 100,
        name='Benchmark',
        line=dict(color='#ff7f0e')
    ))
    
    fig_alpha.add_trace(go.Scatter(
        x=returns['Date'],
        y=returns['Active'] * 100,
        name='Active Return',
        line=dict(color='#2ca02c', dash='dot')
    ))
    
    fig_alpha.update_layout(
        title="Cumulative Returns vs Benchmark",
        xaxis_title="Date",
        yaxis_title="Cumulative Return (%)",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', ticksuffix='%'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    # Rolling performance
    rolling_window = 21  # 1 month
    rolling_metrics = returns[['Strategy', 'Benchmark']].rolling(rolling_window).apply(
        lambda x: (1 + x).prod() - 1
    ) * 100
    
    fig_rolling = go.Figure()
    
    fig_rolling.add_trace(go.Scatter(
        x=returns['Date'],
        y=rolling_metrics['Strategy'],
        name='Strategy',
        line=dict(color='#636efa')
    ))
    
    fig_rolling.add_trace(go.Scatter(
        x=returns['Date'],
        y=rolling_metrics['Benchmark'],
        name='Benchmark',
        line=dict(color='#ff7f0e')
    ))
    
    fig_rolling.update_layout(
        title=f"{rolling_window}-Day Rolling Returns",
        xaxis_title="Date",
        yaxis_title=f"{rolling_window}-Day Return (%)",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
        showlegend=True,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    # Drawdown analysis
    def calculate_drawdown(cum_returns):
        peak = cum_returns.cummax()
        return (cum_returns - peak) / (1 + peak)
    
    drawdowns = pd.DataFrame({
        'Strategy': calculate_drawdown(returns['Strategy'] + 1) * 100,
        'Benchmark': calculate_drawdown(returns['Benchmark'] + 1) * 100
    })
    
    fig_drawdown = go.Figure()
    
    fig_drawdown.add_trace(go.Scatter(
        x=returns['Date'],
        y=drawdowns['Strategy'],
        name='Strategy',
        fill='tozeroy',
        line=dict(color='#636efa'),
        fillcolor='rgba(99, 110, 250, 0.2)'
    ))
    
    fig_drawdown.add_trace(go.Scatter(
        x=returns['Date'],
        y=drawdowns['Benchmark'],
        name='Benchmark',
        fill='tozeroy',
        line=dict(color='#ff7f0e'),
        fillcolor='rgba(255, 127, 14, 0.2)'
    ))
    
    fig_drawdown.update_layout(
        title="Drawdown Analysis",
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
        showlegend=True,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    # Factor exposure (sample data)
    factors = ['MKT', 'SMB', 'HML', 'MOM', 'LIQ']
    exposures = np.random.normal(0, 0.5, len(factors))
    t_stats = np.random.normal(2, 0.5, len(factors))
    
    fig_factors = go.Figure()
    
    colors = ['rgba(99, 110, 250, 0.8)' if t > 1.96 or t < -1.96 else 'rgba(200, 200, 200, 0.6)' 
              for t in t_stats]
    
    fig_factors.add_trace(go.Bar(
        x=factors,
        y=exposures,
        marker_color=colors,
        error_y=dict(
            type='data',
            array=1.96 / np.array(t_stats) * np.abs(exposures),
            visible=True
        )
    ))
    
    fig_factors.add_hline(y=0, line_dash='dash', line_color='white')
    
    fig_factors.update_layout(
        title="Factor Exposures (95% CI)",
        xaxis_title="Factor",
        yaxis_title="Exposure",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return (
        fig_alpha,
        fig_rolling,
        fig_drawdown,
        fig_factors,
        f"{metrics['alpha']*100:.2f}%",
        f"{metrics['beta']:.2f}",
        f"{metrics['r_squared']*100:.1f}%",
        f"{metrics['info_ratio']:.2f}"
    )
