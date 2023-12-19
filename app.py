import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle,os,sys
from src.components.test_transformtion import input_transformer
from src.components.inverse_trnfm_pred import inverse_trnsform
from src.components.predictor import predict


from src.utils import load_object

app = Flask(__name__, template_folder="templates")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
          
    GrLivArea = int(request.form['GrLivArea'])
    BsmtFinSF1 = int(request.form['BsmtFinSF1'])
    CentralAir = request.form['CentralAir']
    TotalBsmtSF = int(request.form['TotalBsmtSF'])
    GarageArea = int(request.form['GarageArea'])
    firstFlrSF = int(request.form['firstFlrSF'])
    MSZoning = request.form['MSZoning']
    GarageCars = int(request.form['GarageCars'])
    OverallQual = int(request.form['OverallQual'])
    OverallCond = int(request.form['OverallCond'])
    data = {
    'OverallQual': [OverallQual],
    'GrLivArea': [GrLivArea],
    'TotalBsmtSF': [TotalBsmtSF],
    'GarageCars': [GarageCars],
    'GarageArea': [GarageArea],
    'firstFlrSF': [firstFlrSF],
    'BsmtFinSF1': [BsmtFinSF1],
    'MSZoning': [MSZoning],
    'CentralAir': [CentralAir],
    'OverallCond': [OverallCond]
    }

    # data = request.form.to_dict()
    df = pd.DataFrame(data)

    feature_file_path = os.path.join('artifacts','features.pkl')
    features  = load_object(feature_file_path)
    # input transformation
    input_tranform = input_transformer()
    trnsfm_df = input_tranform.input_transform_generator(X_test=df, features=features)
    #predition
    model_file_path =  os.path.join('artifacts','model.pkl')
    model  = load_object(model_file_path)
    prediction = model.predict(trnsfm_df)
    #inverse transformation
    prediction = np.round(inverse_trnsform.inverse_transform_generator(prediction[0]))

    return render_template('index.html', prediction_text=f"House Price is {prediction}")

if __name__ == "__main__":
    app.run(debug=True)
