# from dino.dash_reusable_components import numpy_to_b64
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
from io import BytesIO, StringIO
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
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

img = data.chelsea()
fig = px.imshow(img)

fig.add_shape(x0 = 0, x1 = 16, y0 = 0, y1 = 16, type = 'rect', name= 'sq')
fig.add_shape(x0 = 0, x1 = 16, y0 = 0, y1 = 16, type = 'rect', name= 'sel', line = {'color':'red'})
fig.update_layout(clickmode="select+event", dragmode= False)
fig.update_traces(hovertemplate='Select Patch')
mini = px.imshow(img[:16,:16,:])

config = {"modeBarButtonsToAdd": [], "displayModeBar": False}

####### MODEL code

arch = 'vit_small'
patch_size = 16
pretrained_weights = 'vit_small'
checkpoint_key = None
image_path = 'surf.png'

image_size = (592, 1184)
output_dir = '.'
threshold = None

## Load model
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = vits.__dict__[arch](patch_size=patch_size, num_classes=0)
for p in model.parameters():
    p.requires_grad = False
model.eval()
model.to(device)
if os.path.isfile(pretrained_weights):
    state_dict = torch.load(pretrained_weights, map_location="cpu")
    if checkpoint_key is not None and checkpoint_key in state_dict:
        print(f"Take key {checkpoint_key} in provided checkpoint dict")
        state_dict = state_dict[checkpoint_key]
    # remove `module.` prefix
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    # remove `backbone.` prefix induced by multicrop wrapper
    state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
    msg = model.load_state_dict(state_dict, strict=False)
    print('Pretrained weights found at {} and loaded with msg: {}'.format(pretrained_weights, msg))
else:
    print("Please use the `--pretrained_weights` argument to indicate the path of the checkpoint to evaluate.")
    url = None
    if arch == "vit_small" and patch_size == 16:
        url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
    elif arch == "vit_small" and patch_size == 8:
        url = "dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"  # model used for visualizations in our paper
    elif arch == "vit_base" and patch_size == 16:
        url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
    elif arch == "vit_base" and patch_size == 8:
        url = "dino_vitbase8_pretrain/dino_vitbase8c_pretrain.pth"
    if url is not None:
        print("Since no pretrained weights have been provided, we load the reference pretrained DINO weights.")
        state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
        model.load_state_dict(state_dict, strict=True)
    else:
        print("There is no reference weights available for this model => We use random weights.")


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
    html.Button('Heatmap', id='heatmap-but'),
    dcc.RadioItems(id = 'patchsize', options=[{'label': '8', 'value': '8'}, {'label': '16', 'value': '16'}], value = '16'),
    dcc.Graph(figure=mini, config=config, id = 'mini'),
    ]
)

def parse_contents(contents):
    print('in Parse contetns')
    content_type, content_string = contents.split(',')
    try:
        im = imread(BytesIO(base64.b64decode(content_string)))
        im = np.asarray(im)
        # print('imshape', im.shape)
        h, w = im.shape[0], im.shape[1]
        size = 112
        # print(h//2 -size,h//2 + size,w//2 - size,w//2 + size)
        im = im[h//2 -size:h//2 + size,w//2 - 5*size:w//2 + 5*size,:3]

        # transform = pth_transforms.Compose([
        #     pth_transforms.Resize(image_size),
        #     pth_transforms.ToTensor(),
        #     pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        # ])
        # im = transform(im)
        ## Error is : img should be PIL Image. Got <class 'imageio.core.util.Array'>
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])
    return im


def update_mini(ya, xa, ps):
    string = fig['data'][0]['source'].split(";base64,")[-1]
    im_pil = drc.b64_to_numpy(string)
    # print(im_pil[ya:ya+ps,xa:xa+ps,:].shape)
    return px.imshow(im_pil[ya:ya+ps,xa:xa+ps,:3])

@app.callback(
    Output('main-graph', 'figure'),
    Output('mini', 'figure'),
    Input('main-graph', 'clickData'),
    Input('patchsize', 'value'),
    Input('upload-data', 'contents'),
    Input('heatmap-but', 'n_clicks'),
    )
def update_callback(clickData, patchsize, contents, HMn_clicks):
    ctx = dash.callback_context
    ps = int(patchsize)
    # print(fig)
    print(ctx.triggered)
    im = data.chelsea()
    if 'upload-data.contents' == ctx.triggered[0]['prop_id']:
        im = parse_contents(contents)
        fig.update({'data' : [{'source':  'data:image/png;base64,'+drc.numpy_to_b64(im, scalar = False)}]})
        print('imshape', im.shape)
    if 'heatmap-but.n_clicks' == ctx.triggered[0]['prop_id']:
        print('heatmap clicked')
        img = Image.open(BytesIO(base64.b64decode(fig.data[0]['source'].split(";base64,")[-1])))
        img = img.convert('RGB')
        print('img type', type(img))
        transform = pth_transforms.ToTensor()
        img = transform(img)
        w, h = img.shape[1] - img.shape[1] % patch_size, img.shape[2] - img.shape[2] % patch_size
        img = img[:, :w, :h].unsqueeze(0)
        attentions = model.get_last_selfattention(img.to(device))
        nh = attentions.shape[1] # number of head

        w_featmap = img.shape[-2] // patch_size
        h_featmap = img.shape[-1] // patch_size
        # we keep only the output patch attention
        output_patch = 0
        my_patch = 0
        heatmap = attentions[0, :, my_patch, 1:].reshape(nh, -1)
        print('after reshape',heatmap.shape)
        heatmap = heatmap.reshape(nh, w_featmap, h_featmap)
        print('after reshape',heatmap.shape)
        heatmap = nn.functional.interpolate(heatmap.unsqueeze(0), scale_factor=patch_size, mode="nearest")[0].cpu().numpy()
        print('after interpolate',heatmap.shape)
        heatmap = np.asarray(heatmap)
        fig.add_layout_image(
                    x=0,
                    sizex=im.shape[1],
                    y=0,
                    sizey=im.shape[0],
                    xref="x",
                    yref="y",
                    opacity=.5,
                    layer="above",
                    source='data:image/png;base64,'+drc.numpy_to_b64(heatmap[0], scalar = True)
                    )
    if clickData is not None:
        # print(mini)
        points = clickData['points'][0]
        # print(points)
        x = points['x'] # 17 - ?  33 - ? 8 - ? 0, 16, 32, fuc you
        y = points['y']
        xa = x - x % (ps+0) # xanchor is x - mod 16
        ya = y - y % (ps+0)
        print('ancs',xa, ya)
        fig.update_shapes(x0 = xa, x1 =xa + ps, y0 = ya, y1 = ya + ps, name= 'sel')
        # mini.update_traces({'source': px.imshow(img[ya:ya+ps,xa:xa+ps,:])})
        # print(ya, ya+ps, xa, xa+ps)
        #mini = px.imshow(img[ya:ya+ps,xa:xa+ps,:])
    return fig, update_mini(ya, xa, ps)

if __name__ == "__main__":
    app.run_server(debug=True)