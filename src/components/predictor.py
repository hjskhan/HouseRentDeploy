import os, sys

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

from dataclasses import dataclass

from src.exception import CustomException
from src.utils import load_object
from src.components.inverse_trnfm_pred import inverse_trnsform
from src.components.test_transformtion import input_transformer

@dataclass
class predictConfg:
    file = os.path.join('artifacts', 'model.pkl')

class predict:
    try:
            
        def __init__(self) -> None:
            try:
                self.model_confg = predictConfg()
            except Exception as e:
                raise CustomException(e, sys)
        
        def predictor(self, X_test, Y_test):
            try:
                model = load_object(self.model_confg.file)
                preds = model.predict(X_test)

                inv_y_pred = inverse_trnsform.inverse_transform_generator(y_predicted=preds)
                
                rmse = np.sqrt(mean_squared_error(Y_test, inv_y_pred))
                rmse = rmse/(755000 - 34900)

                return rmse
            
            except Exception as e:
                raise CustomException(e, sys)

    except Exception as e:
        raise CustomException(e, sys)