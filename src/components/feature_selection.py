import os, sys
from dataclasses import dataclass

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

@dataclass
class feature_config:
    fs_file_path = os.path.join('artifacts', 'feature_selector.pkl')

class feature_selection:
    def __init__(self) -> None:
        try:
            self.fs_confg = feature_config()
        except Exception as e:
            raise CustomException(e, sys)
        
    def feature_selection_generator(self,y_target, data):
        try:

            model = RandomForestRegressor(random_state=1,max_depth= 7, min_samples_split=5,
                                        min_samples_leaf= 2, n_estimators=200)
            model.fit(data,y_target)
            importances = model.feature_importances_
            feature = pd.Series(importances, index=data.columns).sort_values(ascending=False).head(10).index.tolist()

            save_object(
                file_path=self.fs_confg.fs_file_path,
                obj=model
            )
            return feature
        except Exception as e:
            raise CustomException(e, sys)
