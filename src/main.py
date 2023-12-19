from src.components.data_ingestion import DataIngestion
from src.components.train_transformation import DataTransformer
from src.components.feature_selection import feature_selection
from src.components.model_trainer import model_fitting
from src.components.hyperparameter_tuner import hyp_tune
from src.components.test_transformtion import input_transformer
from src.components.inverse_trnfm_pred import inverse_trnsform
from src.components.predictor import predict

from src.utils import load_object

import pandas as pd

if __name__=='__main__':
    obj = DataIngestion()
    train_data, test_data = obj.initiate()# these are paths to data

    train_transf = DataTransformer()
    X_train,y_train = train_transf.initiate_transf(train_data)

    feature_select = feature_selection()
    features = feature_select.feature_selection_generator(data=X_train, y_target=y_train)
    print(features)


    test_transf = input_transformer()
    X_test = test_transf.read_test_data(test_data)
    X_test , Y_test= test_transf.input_transform_generator(X_test, features)
    
    model_fitter = model_fitting()
    X_train = X_train[features]
    best_model =  model_fitter.model_fit_generator(data=X_train, Y=y_train, feature=features)
    print(best_model)
        
    hyp_tuner = hyp_tune()
    best_params = hyp_tuner.hyp_tune_generator(best_model, X_train, y_train)
    print(best_params)

    pred = predict()
    RMSE = pred.predictor(X_test,Y_test)

    print("test data", RMSE)




    



