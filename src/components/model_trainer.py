import os,sys

from src.exception import CustomException
from src.logger import logging

from dataclasses import dataclass

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.linear_model import LinearRegression,ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import catboost as cb


@dataclass
class model_fit_Confg:
    model_fit_file = os.path.join('artifacts', 'model_trainer')

class model_fitting:
    def __init__(self) -> None:
        self.model_fit_confg = model_fit_Confg()

    def model_fit_generator(self, feature, data, Y):
        try:
            X_data = data[feature]
            
            X_train,X_vald,y_train,y_vald = train_test_split(X_data,Y,test_size=0.2,random_state=0)

            svr = SVR(kernel='rbf')
            rndReg = RandomForestRegressor(n_estimators=10,random_state=1)
            elNet = ElasticNet(alpha=0.0001,l1_ratio=0.5)
            dTree = DecisionTreeRegressor(max_depth=5,min_samples_split=4,min_samples_leaf=4)
            lin_reg = LinearRegression()
            model_GBR =  GradientBoostingRegressor(random_state=1)

            model_list,rmse,acc,R2Score,AvgCVScore,AvgCVRMSE = [],[],[],[],[],[]

            models =[lin_reg,dTree,elNet,rndReg,svr,model_GBR]

            for model in models:
                model_list.append(model.fit(X_train,y_train))
                model1 = model.fit(X_train,y_train)
                y_predict = model.predict(X_vald)
                rmse.append(np.sqrt(mean_squared_error(y_vald,y_predict)))
                acc.append(lin_reg.score(X_vald,y_vald))
                R2Score.append(r2_score(y_predict,y_vald))
                #10-fold Cross Validation
                scores = cross_val_score(model1,X_train, y_train, scoring='r2', cv = 10)
                AvgCVScore.append(scores.mean())
                AvgCVRMSE.append(np.sqrt(-cross_val_score(model1,X_train, y_train,
                                                scoring='neg_mean_squared_error', cv = 10)).mean())

            model_score = pd.DataFrame({'Model':['Linear Regression','Descision Tree','Elastic Net',
                                                'Random Forest','SVM','GBR'],
                                        'RMSE':rmse,'Accuracy':acc,'Avg CV RMSE':AvgCVRMSE,
                                        'Avg CV Score': AvgCVScore,'R2 Score':R2Score})
            best_model = model_score[model_score['RMSE'] == model_score['RMSE'].min()]['Model'].values[0]
            return best_model

        except Exception as e:
            raise CustomException(e, sys)
