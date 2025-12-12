"""
Aegis Dashboard - Trades Page
Displays trade history and performance metrics
"""
from dash import dcc, html, Input, Output, callback, dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Layout
layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H3("Trade Analysis", className="mb-4"),
            dbc.Card([
                dbc.CardHeader("Trade History"),
                dbc.CardBody([
                    dcc.Loading(
                        id="loading-trades",
                        type="circle",
                        children=[
                            html.Div(id='trades-table')
                        ]
                    )
                ])
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Win Rate by Asset"),
                        dbc.CardBody([
                            dcc.Graph(id='win-rate-asset', style={'height': '300px'})
                        ])
                    ])
                ]),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Win Rate by Time of Day"),
                        dbc.CardBody([
                            dcc.Graph(id='win-rate-time', style={'height': '300px'})
                        ])
                    ])
                ])
            ])
        ], md=6),
        
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Trade Statistics"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Div(id='total-trades', className="h3 text-center"),
                            html.Div("Total Trades", className="text-center text-muted")
                        ]),
                        dbc.Col([
                            html.Div(id='win-rate', className="h3 text-center"),
                            html.Div("Win Rate", className="text-center text-muted")
                        ]),
                        dbc.Col([
                            html.Div(id='avg-win', className="h3 text-center"),
                            html.Div("Avg. Win", className="text-center text-muted")
                        ]),
                        dbc.Col([
                            html.Div(id='avg-loss', className="h3 text-center"),
                            html.Div("Avg. Loss", className="text-center text-muted")
                        ])
                    ], className="mb-4"),
                    
                    dcc.Graph(id='pnl-distribution', style={'height': '250px'})
                ])
            ], className="mb-4"),
            
            dbc.Card([
                dbc.CardHeader("Trade Duration Analysis"),
                dbc.CardBody([
                    dcc.Graph(id='trade-duration', style={'height': '300px'})
                ])
            ])
        ], md=6)
    ])
])

# Callbacks
@callback(
    [Output('trades-table', 'children'),
     Output('win-rate-asset', 'figure'),
     Output('win-rate-time', 'figure'),
     Output('pnl-distribution', 'figure'),
     Output('trade-duration', 'figure'),
     Output('total-trades', 'children'),
     Output('win-rate', 'children'),
     Output('avg-win', 'children'),
     Output('avg-loss', 'children')],
    [Input('trades-store', 'data')]
)
def update_trade_analysis(trades_data):
    if not trades_data:
        raise PreventUpdate
    
    # Convert to DataFrame
    trades = pd.DataFrame(trades_data)
    
    # Ensure timestamp is datetime
    if 'timestamp' in trades.columns:
        trades['timestamp'] = pd.to_datetime(trades['timestamp'])
    else:
        # Generate sample timestamps if not present
        trades['timestamp'] = pd.date_range(
            end=datetime.now(), 
            periods=len(trades),
            freq='4H'  # One trade every 4 hours
        )
    
    # Ensure pnl is numeric
    if 'pnl' not in trades.columns:
        trades['pnl'] = np.random.normal(0, 100, len(trades))
    
    # Calculate trade statistics
    total_trades = len(trades)
    winning_trades = (trades['pnl'] > 0).sum()
    win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
    
    avg_win = trades[trades['pnl'] > 0]['pnl'].mean() if (trades['pnl'] > 0).any() else 0
    avg_loss = trades[trades['pnl'] < 0]['pnl'].mean() if (trades['pnl'] < 0).any() else 0
    
    # Create trades table
    table_columns = [
        {"name": "Time", "id": "timestamp", "type": "datetime"},
        {"name": "Symbol", "id": "symbol"},
        {"name": "Side", "id": "side"},
        {"name": "Size", "id": "size", "type": "numeric", "format": {"specifier": ",.2f"}},
        {"name": "Price", "id": "price", "type": "numeric", "format": {"specifier": ",.2f"}},
        {"name": "P&L", "id": "pnl", "type": "numeric", "format": {"specifier": "+,.2f"}}
    ]
    
    # Format table data
    table_data = trades.sort_values('timestamp', ascending=False).to_dict('records')
    
    # Create trades table component
    trades_table = dash_table.DataTable(
        id='trades-datatable',
        columns=table_columns,
        data=table_data,
        page_size=10,
        style_table={'overflowX': 'auto'},
        style_cell={
            'textAlign': 'left',
            'padding': '8px',
            'fontSize': '12px',
            'fontFamily': 'sans-serif'
        },
        style_header={
            'backgroundColor': 'rgb(30, 30, 30)',
            'color': 'white',
            'fontWeight': 'bold'
        },
        style_data_conditional=[
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': 'rgb(30, 30, 30)'
            },
            {
                'if': {
                    'filter_query': '{pnl} > 0',
                    'column_id': 'pnl'
                },
                'color': '#00cc96',
                'fontWeight': 'bold'
            },
            {
                'if': {
                    'filter_query': '{pnl} < 0',
                    'column_id': 'pnl'
                },
                'color': '#ef553b',
                'fontWeight': 'bold'
            }
        ]
    )
    
    # Win rate by asset
    if 'symbol' in trades.columns:
        win_rate_asset = trades.groupby('symbol')['pnl'].apply(
            lambda x: (x > 0).mean() * 100
        ).reset_index()
        
        fig_win_asset = px.bar(
            win_rate_asset,
            x='symbol',
            y='pnl',
            title='Win Rate by Asset',
            labels={'pnl': 'Win Rate (%)', 'symbol': 'Asset'},
            color='pnl',
            color_continuous_scale='RdYlGn',
            range_color=[30, 70]
        )
    else:
        # Sample data if no symbol column
        fig_win_asset = go.Figure()
        fig_win_asset.add_annotation(
            text="No asset data available",
            x=0.5,
            y=0.5,
            showarrow=False
        )
    
    # Win rate by time of day
    if 'timestamp' in trades.columns:
        trades['hour'] = trades['timestamp'].dt.hour
        win_rate_time = trades.groupby('hour')['pnl'].apply(
            lambda x: (x > 0).mean() * 100
        ).reset_index()
        
        fig_win_time = go.Figure()
        
        fig_win_time.add_trace(go.Scatter(
            x=win_rate_time['hour'],
            y=win_rate_time['pnl'],
            mode='lines+markers',
            line=dict(color='#636efa', width=2),
            marker=dict(size=8, color='#636efa')
        ))
        
        fig_win_time.update_layout(
            title='Win Rate by Hour of Day',
            xaxis_title='Hour of Day',
            yaxis_title='Win Rate (%)',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=False, dtick=2),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
            showlegend=False,
            margin=dict(l=20, r=20, t=50, b=20)
        )
    else:
        fig_win_time = go.Figure()
        fig_win_time.add_annotation(
            text="No timestamp data available",
            x=0.5,
            y=0.5,
            showarrow=False
        )
    
    # P&L distribution
    fig_pnl = go.Figure()
    
    fig_pnl.add_trace(go.Histogram(
        x=trades['pnl'],
        nbinsx=30,
        marker_color='#636efa',
        opacity=0.7,
        name='P&L Distribution'
    ))
    
    fig_pnl.add_vline(
        x=0,
        line_dash='dash',
        line_color='white',
        opacity=0.7
    )
    
    fig_pnl.update_layout(
        title='P&L Distribution',
        xaxis_title='P&L',
        yaxis_title='Count',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    # Trade duration analysis (sample data)
    if 'timestamp' in trades.columns and 'side' in trades.columns:
        # Generate random durations for the example
        np.random.seed(42)
        durations = np.random.lognormal(mean=3, sigma=0.8, size=len(trades)).clip(1, 24*7)  # 1 hour to 1 week
        trades['duration'] = durations
        
        # Group by side if available
        if 'side' in trades.columns:
            fig_duration = px.box(
                trades,
                x='side',
                y='duration',
                color='side',
                title='Trade Duration by Side',
                labels={'duration': 'Duration (hours)', 'side': 'Side'},
                color_discrete_map={
                    'LONG': '#00cc96',
                    'SHORT': '#ef553b'
                }
            )
        else:
            fig_duration = px.box(
                trades,
                y='duration',
                title='Trade Duration',
                labels={'duration': 'Duration (hours)'}
            )
            
        fig_duration.update_traces(
            boxpoints=False,
            line_width=1.5
        )
        
        fig_duration.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            showlegend=False,
            margin=dict(l=20, r=20, t=50, b=20)
        )
    else:
        fig_duration = go.Figure()
        fig_duration.add_annotation(
            text="Insufficient data for duration analysis",
            x=0.5,
            y=0.5,
            showarrow=False
        )
    
    return (
        trades_table,
        fig_win_asset,
        fig_win_time,
        fig_pnl,
        fig_duration,
        f"{total_trades}",
        f"{win_rate:.1f}%",
        f"${avg_win:,.2f}",
        f"${avg_loss:,.2f}"
    )
