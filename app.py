from flask import Flask, request,render_template

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from lmfit.models import PseudoVoigtModel, GaussianModel, LorentzianModel
import os
from io import BytesIO
import numpy as np  
import base64
import plotly.express as px
from werkzeug.utils  import secure_filename
import plotly.graph_objects as go
import json
import plotly
from scipy import optimize, signal
from lmfit import models
import random




app = Flask(__name__)

def create_plot():
    df = pd.read_csv("files/peak.csv", header=None, names=['x', 'y'])
    data = [
                go.Scatter(
                    x=df['x'], # assign x as the dataframe column 'x'
                    y=df['y'],
                    mode = 'markers'
                )

            ]

    graphJSON = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON



def create_model(data):
    curveModel = None
    parameter = None
    x = data['x']
    y = data['y']
    minimum_x = np.min(x)
    maximum_x = np.max(x)
    range_x = maximum_x - minimum_x
    maximum_y = np.max(y)
    for i, modelType in enumerate(data['model']):
        prefix = f'k{i}_'
        curveFit = getattr(models, modelType['type'])(prefix=prefix)
        if modelType['type'] in ['PseudoVoigtModel']: 
            curveFit.set_param_hint('sigma', min=1e-6, max=range_x)
            curveFit.set_param_hint('center', min=minimum_x, max=maximum_x)
            curveFit.set_param_hint('height', min=1e-6, max=1.1*maximum_y)
            curveFit.set_param_hint('amplitude', min=1e-6)
            parameters = {
                prefix+'center': minimum_x + range_x * random.random(),
                prefix+'height': maximum_y * random.random(),
                prefix+'sigma': range_x * random.random()
            }
        else:
            raise NotImplemented(f'curveFit {modelType["type"]} not implemented yet')
        
        modelparameters = curveFit.make_params(**parameters, **modelType.get('parameter', {}))
        if parameter is None:
            parameter = modelparameters
        else:
            parameter.update(modelparameters)
        if curveModel is None:
            curveModel = curveFit
        else:
            curveModel = curveModel + curveFit
    return curveModel, parameter





@app.route("/")
def home():
    return render_template('index.html' )
    
 

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
        f = request.files['file']
        f.filename = "peak.csv"  #some custom file name that you want
        f.save("files/"+f.filename)
        bar = create_plot()
        return render_template('Hello.html', plot=bar)



@app.route('/response', methods=['POST'])
def response():
    s = request.form.get("fname")
    t = request.form.get("note")
    p = int(s)
    q = int(t)
    df = pd.read_csv("files/peak.csv")
    
    mask = (df['x'] >= p) & (df['x'] <= q)
    print(df.loc[mask])
    df_filtered = df.loc[mask]

    df_filtered.to_csv("outfile.csv")
    dframe = pd.read_csv('outfile.csv')
    data = {
        'x': dframe['x'].values,
        'y': dframe['y'].values,
        'model': [
            {'type': 'PseudoVoigtModel'},
            {'type': 'PseudoVoigtModel'},
            {'type': 'PseudoVoigtModel'},
            {'type': 'PseudoVoigtModel'},
            {'type': 'PseudoVoigtModel'},
        ]
    }


    model, params = create_model(data)
    output = model.fit(data['y'], params, x=data['x'])
    fig, gridspec = output.plot(data_kws={'markersize':  1})
    figfile = BytesIO()
    plt.savefig(figfile, format='png')
    figfile.seek(0)
    figdata_png = base64.b64encode(figfile.getvalue()).decode('ascii')



    fig, ax = plt.subplots()
    ax.scatter(data['x'], data['y'], s=4)
    components = output.eval_components(x=data['x'])
    print(len(data['model']))
    for i, model in enumerate(data['model']):
        ax.plot(data['x'], components[f'k{i}_'])

    figfile1= BytesIO()
    plt.savefig(figfile1, format='png')
    figfile1.seek(0)
    figdata_png1 = base64.b64encode(figfile1.getvalue()).decode('ascii')
    
    return render_template('XRayFitting.html', result=figdata_png, result1 = figdata_png1 )
  


if __name__ == "__main__":
    app.run(debug=True)