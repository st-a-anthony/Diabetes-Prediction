from flask import Flask, request,render_template,url_for
import pickle
import pandas as pd
import numpy as np
import pickle

import main as m
app = Flask(__name__)
pkl_filename = "pkl.pkl"
loaded_model = pickle.load(open(pkl_filename, 'rb'))
@app.route("/")
def index():
    return render_template('index.html')

@app.route("/predict", methods = ["GET","POST"])
def predict():
    

    Pregnancies = request.form.get('Pregnancies')
    Glucose = request.form.get('Glucose')
    BloodPressure = request.form.get('BloodPressure')
    SkinThickness = request.form.get('SkinThickness')
    Insulin = request.form.get('Insulin')
    BMI = request.form.get('BMI')
    DiabetesPedigreeFunction = request.form.get('DiabetesPedigreeFunction')
    Age= request.form.get('Age')
    input_data=(Pregnancies,Glucose,BloodPressure, SkinThickness,Insulin,BMI, DiabetesPedigreeFunction,Age)
    # input_data=(6,148,72,35,0,33.6,0.627,50)
    output=m.prediction(input_data)
    print(output)
    if (output == 0):
        a="no"
    else:
        a="yes"

    
    return render_template("index.html", prediction_text = "Diabetics {}".format(a))  
    


    
if __name__ == "__main__":
   app.run(debug=True)