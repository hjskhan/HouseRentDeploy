import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask(__name__, template_folder="templates")
model = pickle.load(open('model.pkl', 'rb'))

# Define the log transformation function
def log_transform(x):
    return np.log(int(x)+1)

# Define the min-max scaling function
def min_max_scale(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)

# Preprocess the input data
def preprocess_input(input_data):
    
     # Convert relevant columns to integers
    input_data['OverallQual'] = input_data['OverallQual'].astype(int)
    input_data['GarageCars'] = input_data['GarageCars'].astype(int)
    input_data['YearBuilt'] = input_data['YearBuilt'].astype(int)
    input_data['GarageType'] = input_data['GarageType'].astype(int)

    # Log transform the continuous variables
    continuous_columns = ['GrLivArea', 'firstFlrSF', 'TotalBsmtSF', 'GarageArea', 'BsmtFinSF1', 'LotArea']
    
    input_data[continuous_columns] = input_data[continuous_columns].apply(log_transform)

    # Manual min-max scaling
    min_vals = {'OverallQual': 1, 'GrLivArea': 5.814131, 'GarageCars': 0, 'firstFlrSF': 5.814131, 'TotalBsmtSF': 0,
                'GarageArea': 0, 'YearBuilt': 1872, 'BsmtFinSF1': 0, 'LotArea': 7.170888, 'GarageType': 0}
    max_vals = {'OverallQual': 10, 'GrLivArea': 8.638171, 'GarageCars': 4, 'firstFlrSF': 8.453827, 'TotalBsmtSF': 8.717846,
                'GarageArea': 7.257708,  'YearBuilt': 2010,'BsmtFinSF1': 8.638525, 'LotArea': 12.279537, 'GarageType': 6}
    
    input_data = input_data.apply(lambda x: min_max_scale(x, min_vals[x.name], max_vals[x.name]))

    return input_data


# Define the inverse log transformation function
def inverse_log_transform(x):
    return np.exp(x) - 1

# Define the inverse min-max scaling function
def inverse_min_max_scale(x, min_val, max_val):
    return ((x * (max_val - min_val)) + min_val)

# Reverse the preprocessing transformations for the predicted value
def reverse_transform(prediction):
    min_val = 10.460270761075149
    max_val = 13.534474352733596
    prediction = inverse_min_max_scale(prediction, min_val, max_val)
    prediction = round(inverse_log_transform(prediction))
    return prediction

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.to_dict()
    df = pd.DataFrame(data, index=[0])
    preprocessed_data = preprocess_input(df)
    prediction = model.predict(preprocessed_data[['OverallQual', 'GrLivArea', 'GarageCars', 'firstFlrSF', 'TotalBsmtSF',
                                                  'GarageArea', 'YearBuilt', 'BsmtFinSF1', 'LotArea', 'GarageType']])
    prediction = reverse_transform(prediction[0])
    return render_template('index.html', prediction_text=f"House Price is {prediction}")

if __name__ == "__main__":
    app.run(debug=True)
