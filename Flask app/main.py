from flask import Flask
from flask import render_template
from flask import request,redirect,Response,url_for
import random, json
import os
import numpy as np
import tensorflow as tf 
import cv2 

app = Flask(__name__)

global pred
@app.route('/')
def main_page():
	return render_template('main.html',__data__="")
@app.route('/result')
def page():
    global pred
    return render_template('index.html',__data__=str(pred))

@app.route('/postmethod', methods = ['POST'])
def post_javascript_data():
    global pred
    jsdata = request.form['canvas_data']
    nmodel=tf.keras.models.load_model("path to saved model")
    nmodel._make_predict_function()
    jsdata=np.array([float(x) for x in jsdata.split(',')]).reshape(320,320)
    jsdata=cv2.resize(jsdata,(160,160))
    pred=nmodel.predict(jsdata.reshape(1,160,160,1))
    return redirect("/result")
if __name__== '__main__':
	app.run(debug=True)