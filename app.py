import numpy as np
import pandas as pd
import os
import pickle
from flask import Flask, request, url_for, session, flash, Response
from flask import Flask, request, jsonify, render_template
import json
import plotly
import plotly.express as px
from datetime import datetime
import io
import statsmodels.api as sm
from scipy import stats

dfi = pd.read_csv("bitcoin.csv")
dfi.Timestamp = pd.to_datetime(dfi.Timestamp, unit='s')
dfi.index = dfi.Timestamp
df_day= dfi.resample('D').mean()
df_day = df_day.iloc[1200:,:]
sm.tsa.seasonal_decompose(df_day.Weighted_Price)
df_day['Weighted_Price_box'], lmbda = stats.boxcox(df_day.Weighted_Price)


bestmodel = pickle.load( open( "ARIMA_best_model.pkl", "rb" ) )

def prepare_input(start_date, end_date):
    core_date = '2011-12-31'
    startdate = datetime.strptime(start_date, '%Y-%m-%d').date()
    enddate = datetime.strptime(end_date, '%Y-%m-%d').date()
    coredate = datetime.strptime(core_date, '%Y-%m-%d').date()
    startrange = startdate-coredate
    endrange = enddate-coredate
    starter = startrange.days
    ender = endrange.days
    
    return starter, ender

def invboxcox(y,lmbda):
    if lmbda == 0:
        return(np.exp(y))
    else:
        return(np.exp(np.log(lmbda*y+1)/lmbda))

app=Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    dates = [x for x in request.form.values()]
    start_date = dates[0]
    end_date = dates[1]
    a, z = prepare_input(start_date, end_date)
    predict = invboxcox(bestmodel.predict(start=a, end=z), lmbda)
    set = predict.to_frame()
    df = pd.read_csv(io.StringIO(set.to_csv(index=True)))
    df.rename( columns={'Unnamed: 0':'Date'}, inplace=True )    
    fig = px.line(df, x='Date', y='predicted_mean', title='Price Graph')
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
  
    return render_template('index.html', graphJSON=graphJSON)


if __name__=='__main__':
    app.run()
