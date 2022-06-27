import os
import pandas as pd
import numpy as np


import dash
from dash import dcc
from dash import html, dash_table
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State

from statsmodels.tsa.stattools import acf, pacf
from pmdarima import auto_arima

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import flask
server = flask.Flask(__name__)
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY], server=server)


ROOT_DIR = os.path.dirname(os.path.realpath(__file__))

# Create the shop index
SHOP_INDEX = pd.read_csv(os.path.join(ROOT_DIR, 'data', 'shops.csv'))
SHOP_INDEX = SHOP_INDEX[~ SHOP_INDEX.shop_id.isin([0, 1, 10])]
SHOP_INDEX = [{'label': i['shop_name'], 'value': i['shop_id']} for i in SHOP_INDEX.to_dict('records')]

# Create the Item category index
ICAT_INDEX = pd.read_csv(os.path.join(ROOT_DIR, 'data', 'item_categories.csv'))
ICAT_INDEX = [{'label': i['item_category_name'], 'value': i['item_category_id']} for i in ICAT_INDEX.to_dict('records')]


# DATA
DATA = pd.read_csv(os.path.join(ROOT_DIR, 'data', 'sales.zip'), sep=';')
DATA.date = pd.to_datetime(DATA.date)



layout = dbc.Container([
    dcc.Store(id='data-sample'),
    dbc.Row([html.H1('Salles prediction data app')]),
    dbc.Row(html.H6('A toy interactive tool for predicting future sales via an ARIMA model. ')),


    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col(html.H6("Target metric:"), width=2),
                        dbc.Col([
                            dcc.Dropdown(id='input-target-metric',
                                         options=[{'label': 'Revenue', 'value': 'revenue'},
                                                  {'label': 'Items Count', 'value': 'item_cnt_day'},
                                                  ],
                                         value ='item_cnt_day')
                        ], width=4)
                    ], style={"margin-top": "10px", "margin-bottom": "10px"}),

                    dbc.Row([
                        dbc.Col(html.H6("Shop:"), width=2),
                        dbc.Col(
                            dcc.Dropdown(id='input-shop',
                                         options=[{'label': 'All', 'value': -1}, *SHOP_INDEX],
                                         value=-1),
                            width=6
                        ),
                    ], style={"margin-top": "10px", "margin-bottom": "10px"}),

                    dbc.Row([
                        dbc.Col(html.H6("Item Category:"), width=2),
                        dbc.Col(
                            dcc.Dropdown(id='input-item-category',
                                         options=[{'label': 'All', 'value': -1}, *ICAT_INDEX],
                                         value=-1),
                            width=6
                        )
                    ], style={"margin-top": "10px", "margin-bottom": "10px"}),

                ], className='split-test-id-card')
            ], color='secondary')
        ], md=12)


    ]),

    dbc.Row([
        dbc.Col(html.H6('Date Granularity:'), width=1),
        dbc.Col([
            dbc.RadioItems(
                options=[
                    {"label": "Monthly", "value": 'M'},
                    {"label": "Daily", "value": 'D'},
                ],
                value='M',
                id="input-date-gran",
                inline=True
            ),

        ], width=2),

        dbc.Col([
            dcc.Slider(
                id='input-diff-slider',
                min=0, max=3, step=1,value=0
            )
        ], width=3),
        dbc.Col([
            html.H6('Prediction horizon (periods):')
        ], width=2),
        dbc.Col([
            dcc.Input(
                id='input-horizon-number',
                type='number',
                placeholder="Debounce False",
                min=1, max=31, step=1, value=3
            )
        ], width=1),
        dbc.Col(dbc.Button('P R E D I C T', color = 'secondary', id='predict-but'), width = 2)
    ], style={"margin-top": "25px", "margin-bottom": "10px"}),

    # Output :
    dbc.Row([
        dbc.Col([
            dcc.Graph('st-ts')
        ], width=12),
    ], style={"margin-top": "20px", "margin-bottom": "10px"})
])

app.layout = layout



@app.callback(
    Output('data-sample', 'data'),
    Input('input-target-metric', 'value'),
    Input('input-shop', 'value'),
    Input('input-item-category', 'value'),
    Input('input-date-gran', 'value'),
)
def load_data_from_sf(metric, shopid, categoryid, gran):

    # This is hack
    shop_filt = (DATA.shop_id > shopid) if shopid == -1 else (DATA.shop_id == shopid)
    category_filt = (DATA.item_category_id > categoryid) if categoryid == -1 else (
                DATA.item_category_id == categoryid)

    ser = (
        DATA[(shop_filt) & (category_filt)].
            assign(date=lambda x: x.date.dt.to_period(gran).dt.to_timestamp()).
            groupby(['date']).
            agg(target=(metric, 'sum')).
            reset_index()
    )

    # ser.index = pd.PeriodIndex(ser.index, freq=gran)
    return ser.to_json(date_format='iso', orient='split')

@app.callback(
    Output('st-ts', 'figure'),
    Input('data-sample', 'data'),
    Input('input-diff-slider', 'value'),
    Input('input-date-gran', 'value'),
    Input('input-target-metric', 'value'),
    Input('input-shop', 'value'),
    Input('input-item-category', 'value'),
    State('input-horizon-number', 'value'),
    Input('predict-but', 'n_clicks')
)
def plot_ts(data, diff, dategran, metric_lab, shop_lab, cat_lab, horizon, predict):

    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    METRIC_MAP = {'item_cnt_day': 'Items Count', 'revenue': 'Revenue'}
    metric = METRIC_MAP[metric_lab]


    # Define titles
    shop_label = "" if shop_lab == -1 else f" {[i['label'] for i in SHOP_INDEX if i['value'] ==shop_lab]} "
    category_label = "" if cat_lab == -1 else f" {[i['label'] for i in ICAT_INDEX if i['value'] ==shop_lab]} "
    main_title = f'{metric}{shop_label}{category_label}'

    fig = make_subplots(rows=3, cols=2, specs=[[{"rowspan":3}, {}], [None, {}], [None, {}]],
                        subplot_titles=[main_title, f'{metric} diff {diff}', 'ACF', 'PACF'])



    ROYAL_BLUE = dict(color="#4169e1")

    if data:
        data_parsed = pd.read_json(data, orient='split')
        data_parsed.date = pd.to_datetime(data_parsed.date)
        ts = pd.Series(data_parsed.target.values, index=pd.to_datetime(data_parsed.date))
        ts.index = pd.PeriodIndex(ts.index, freq=dategran)

        # Apply diff:
        while diff > 0:
            diff -= 1
            ts = ts.diff().dropna()

        # Calculate acf, pacf:
        acf_val = acf(ts)
        pacf_val = pacf(ts)


        # Build trace
        t1 = go.Scatter(x=data_parsed.date, y=data_parsed.target, name ="Main Timeseries", marker= ROYAL_BLUE)
        t_diff = go.Scatter(x=ts.index.to_timestamp(), y=ts.values, name= f"DIFF {diff}", marker=ROYAL_BLUE, showlegend=False)
        t_acf = go.Bar(x=list(range(0, len(acf_val))), y=acf_val, name='ACF', marker=ROYAL_BLUE, showlegend=False)
        t_pacf = go.Bar(x=list(range(0, len(acf_val))), y=pacf_val, name='PACF',  marker=ROYAL_BLUE, showlegend=False)

        fig.add_trace(t1, row=1, col=1)
        fig.add_trace(t_diff, row=1, col=2)
        fig.add_trace(t_acf, row=2, col=2)
        fig.add_trace(t_pacf, row=3, col=2)

        ## Model

        if 'predict-but.n_clicks' in changed_id:

            # Seasonality:
            ses_setup = {'m': 12, 'seasonal': True} if dategran == 'M' else {'m': 1, 'seasonal': False}
            print(ses_setup)

            val_size = int(len(ts) * 0.1)
            oos_size= 0 if len(ts)< 25 else val_size

            # Perform the prediction
            model = auto_arima(ts, start_p=0, start_q=0, test='adf',
                              max_p=10, max_q=10, d=None, start_P=0,
                              D=0, error_action='ignore', stepwise=True,
                              out_of_sample_size=oos_size, trace=True,
                              **ses_setup)

            model_name = f"ARIMA: {model.get_params()['order']} | {model.get_params()['seasonal_order']}"
            fc, intrvl = model.predict(horizon, return_conf_int=True)
            fc_tstamp = pd.date_range(ts.index.max().to_timestamp(), periods=horizon+1, freq=dategran).to_pydatetime()


            # Create the traces
            # They varry a bit thats why we don't put them in loop
            trace_fc = go.Scatter(x=fc_tstamp[1:], y=np.where(fc<0, 0, fc),
                                  name=model_name, mode='lines+markers', marker=dict(color="#FF2400"))

            trace_lo = go.Scatter(x=fc_tstamp[1:], y=np.where(intrvl[:, 0] < 0, 0, intrvl[:, 0]),
                                  showlegend=False, mode='lines+markers', marker=dict(color="#FFA500"))

            trace_up = go.Scatter(x=fc_tstamp[1:], y=np.where(intrvl[:, 1] < 0, 0, intrvl[:, 1]),
                                  showlegend=False, mode='lines+markers', marker=dict(color="#FFA500"))

            fig.add_trace(trace_fc)
            fig.add_trace(trace_lo)
            fig.add_trace(trace_up)

        fig.update_layout(autosize=False,
                          template='simple_white',
                          legend=dict(
                          orientation="h",
                          yanchor="bottom",
                          y=-0.2,
                          xanchor="right",
                          x=0.4
                           )
                          )

        return fig