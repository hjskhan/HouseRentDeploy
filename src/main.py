from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformer
from src.components.feature_selection import feature_selection
from src.components.model_trainer import model_fitting

import pandas as pd

if __name__=='__main__':
    obj = DataIngestion()
    train_data, test_data = obj.initiate()

    data_transf = DataTransformer()
    X_train,y_train, X_test, y_test = data_transf.initiate_transf(train_data, test_data)

    feature_select = feature_selection()
    features = feature_select.feature_selection_generator(data=X_train, y_target=y_train)

    model_fitter = model_fitting()
    model_score =  model_fitter.model_fit_generator(data=X_train, Y=y_train, feature=features)
    print(model_score)


    



