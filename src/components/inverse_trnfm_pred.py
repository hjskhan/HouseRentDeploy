import os, sys

import pandas as pd
import numpy as np

from dataclasses import dataclass

from src.exception import CustomException
from src.utils import load_object

@dataclass
class inverse_trnsformConfg:
    file = os.path.join('artifacts', 'min_max_data.pkl')

class inverse_trnsform:
    def __init__(self) -> None:
        self.inverse_trnsform_confg = inverse_trnsformConfg()
    
    def inverse_transform_generator(y_predicted):
        try:
            file_path = inverse_trnsformConfg.file
            min_max_data = load_object(file_path)
            min_SP_train = min_max_data.loc['min','SalePrice']
            max_SP_train = min_max_data.loc['max','SalePrice']

            y_pred_inv_trnfm = np.expm1((y_predicted*(max_SP_train-min_SP_train))+min_SP_train)

            return y_pred_inv_trnfm 

        except Exception as e:
            raise CustomException(e, sys)
