import os, sys

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, cross_val_score, KFold
from sklearn.metrics import mean_squared_error,r2_score, make_scorer
from sklearn.linear_model import LinearRegression,ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import catboost as cb


from dataclasses import dataclass

from src.exception import CustomException
from src.utils import save_object


@dataclass
class hyperparameter_Confg:
    hyp_file_path = os.path.join('artifacts', 'model.pkl')

class hyp_tune:
    def __init__(self) -> None:
      self.hyp_confg = hyperparameter_Confg

    def hyp_tune_generator(self, model_name, X, Y):
       try:
        scorer = make_scorer(mean_squared_error, greater_is_better=False)

        models = {
                'Linear Regression': (LinearRegression(), {}),

                'Elastic Net': (ElasticNet(), {'alpha': [0.1, 1, 10], 'l1_ratio': [0.25, 0.5, 0.75]}),

                'Decision Tree': (DecisionTreeRegressor(), {'criterion': ['mse', 'mae'],
                                                            'max_depth': [None, 5, 10, 15, 20],
                                                            'min_samples_split': [2, 5, 10],
                                                            'min_samples_leaf': [1, 2, 4]}),

                'Random Forest': (RandomForestRegressor(), {'n_estimators': [100, 200, 300],
                                                            'max_depth': [None, 5, 10, 15, 20],
                                                            'min_samples_split': [2, 5, 10],
                                                            'min_samples_leaf': [1, 2, 4]}),
                'SVR': (SVR(), {'C': [0.1, 1, 10, 100],
                                'gamma': [0.1, 0.01, 0.001],
                                'kernel': ['linear', 'rbf']}), 

                'GBR': (GradientBoostingRegressor(), {'n_estimators': [100, 200, 300],
                                                    'learning_rate': [0.05, 0.1, 0.2],
                                                    'max_depth': [3, 4, 5]})
            }
            
        # Check if the model_name is supported
        if model_name not in models:
            print("Model not supported.")
            return
        
        model, params = models[model_name]
        
        # Create GridSearchCV object
        grid_search = GridSearchCV(model, params, cv=KFold(n_splits=5, shuffle=True, random_state=42),
                                scoring=scorer, verbose=1, n_jobs=-1)
        
        # Perform grid search
        grid_search.fit(X, Y)
        
        save_object(file_path=self.hyp_confg.hyp_file_path, 
                    obj=grid_search)

        # Print the best hyperparameters and the corresponding MSE value
        print("Best Hyperparameters:", grid_search.best_params_)
        print("Best RMSE:", np.sqrt(-grid_search.best_score_))
        
        return grid_search.best_params_, np.sqrt(-grid_search.best_score_)
            
       except Exception as e:
          raise CustomException