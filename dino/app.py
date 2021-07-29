import plotly.express as px
import dash
import dash_core_components as dcc
import dash_html_components as html
from skimage import data
from dash.dependencies import Input, Output, State
import json
import dash_reusable_components as drc

import base64
import os
import sys
import argparse
import cv2
import random
import colorsys
import requests
from io import BytesIO
from imageio import imread

import skimage.io
from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms as pth_transforms
import numpy as np
from PIL import Image

import utils
import vision_transformer as vits

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

img = data.chelsea()
fig = px.imshow(img)

fig.add_shape(x0 = 0, x1 = 16, y0 = 0, y1 = 16, type = 'rect', name= 'sq')
fig.add_shape(x0 = 0, x1 = 16, y0 = 0, y1 = 16, type = 'rect', name= 'sel', line = {'color':'red'})
fig.update_layout(clickmode="select+event", dragmode= False)
fig.update_traces(hovertemplate='Select Patch')
mini = px.imshow(img[:16,:16,:])


config = {"modeBarButtonsToAdd": [], "displayModeBar": False}

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.layout = html.Div(
    [html.H3("ViT Patch Embedding Observer"), 
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=False
    ),
    dcc.Graph(figure=fig, config=config, id = 'main-graph'),
    dcc.RadioItems(id = 'patchsize', options=[{'label': '8', 'value': '8'}, {'label': '16', 'value': '16'}], value = '16'),
    dcc.Graph(figure=mini, config=config, id = 'mini'),
    ]
)

def parse_contents(contents):
    print('in Parse contetns')
    content_type, content_string = contents.split(',')
    try:
        im = imread(BytesIO(base64.b64decode(content_string)))
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])
    return im


def update_mini(ya, xa, ps):
    string = fig['data'][0]['source'].split(";base64,")[-1]
    im_pil = drc.b64_to_numpy(string)
    print(im_pil[ya:ya+ps,xa:xa+ps,:].shape)
    return px.imshow(im_pil[ya:ya+ps,xa:xa+ps,:3])

@app.callback(
    Output('main-graph', 'figure'),
    Output('mini', 'figure'),
    Input('main-graph', 'clickData'),
    Input('patchsize', 'value'),
    Input('upload-data', 'contents'),
    )
def update_callback(clickData, patchsize, contents):
    ctx = dash.callback_context
    ps = int(patchsize)
    print(fig)
    im = data.chelsea()
    if 'upload-data.contents' == ctx.triggered[0]['prop_id']:
        im = parse_contents(contents)
    
        fig.update({'data' : [{'source':  contents}]})
        # print(contents)
    if clickData is not None:
        print(mini)
        points = clickData['points'][0]
        # print(points)
        x = points['x']
        y = points['y']
        xa = x - x % ps # xanchor is x - mod 16
        ya = y - y % ps
        fig.update_shapes(x0 = xa, x1 =xa + ps, y0 = ya, y1 = ya + ps, name= 'sel')
        # mini.update_traces({'source': px.imshow(img[ya:ya+ps,xa:xa+ps,:])})
        # print(ya, ya+ps, xa, xa+ps)
        #mini = px.imshow(img[ya:ya+ps,xa:xa+ps,:])
    return fig, update_mini(ya, xa, ps)

if __name__ == "__main__":
    app.run_server(debug=True)