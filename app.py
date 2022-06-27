import os
import pandas as pd


import dash
from dash import dcc
from dash import html, dash_table
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State



import flask
server = flask.Flask(__name__)
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.COSMO], server=server)







layout = dbc.Container([
    dbc.Row([html.H1('Salles prediction data app'),
             html.P()])])

app.layout = layout