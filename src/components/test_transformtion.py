import os, sys

import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder

from dataclasses import dataclass

from src.exception import CustomException
from src.utils import save_object, load_object

@dataclass
class input_transformerConfg:
    file = os.path.join('artifacts', 'min_max_data.pkl')

class input_transformer:
    def __init__(self) -> None:
        self.input_transf_Confg = input_transformerConfg()

    def encode(self, data, features):
        try:
            enc = ['LotShape','LandSlope','ExterQual','ExterCond',
                        'HeatingQC','KitchenQual',
                        'BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2',
                        'CentralAir','Functional','GarageFinish','PavedDrive','PoolQC','Fence',
                       'FireplaceQu','GarageQual','GarageCond','MSZoning','LandContour',
                       'Alley','LotConfig','Condition1','Condition2','Foundation','Electrical']
            enc = [i for i in enc if i in features]
            
            data_ord_enc = data[enc]

            ordinal_mapping = {
                'LotShape': ['IR3', 'IR2', 'IR1', 'Reg'],
                'LandSlope': ['Gtl', 'Mod', 'Sev'],
                'ExterQual': ['Po', 'Fa', 'TA', 'Gd', 'Ex'],
                'ExterCond': ['Po', 'Fa', 'TA', 'Gd', 'Ex'],
                'HeatingQC': ['Po', 'Fa', 'TA', 'Gd', 'Ex'],
                'KitchenQual': ['Po', 'Fa', 'TA', 'Gd', 'Ex'],
                'BsmtQual': ['No Basement', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
                'BsmtCond': ['No Basement', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
                'BsmtExposure': ['No Basement', 'No', 'Mn', 'Av', 'Gd'],
                'BsmtFinType1': ['No Basement', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'],
                'BsmtFinType2': ['No Basement', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'],
                'CentralAir': ['N', 'Y'],
                'Functional': ['Sal', 'Sev', 'Maj2', 'Maj1', 'Mod', 'Min2', 'Min1', 'Typ'],
                'GarageFinish': ['No Garage', 'Unf', 'RFn', 'Fin'],
                'PavedDrive': ['N', 'P', 'Y'],
                'PoolQC': ['No Pool', 'Fa', 'TA', 'Gd', 'Ex'],
                'Fence': ['No Fence', 'MnWw', 'GdWo', 'MnPrv', 'GdPrv'],
                'FireplaceQu': ['No Fireplace', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
                'GarageQual': ['No Garage', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
                'GarageCond': ['No Garage', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
                'MSZoning': ['FV', 'A', 'RL', 'RP', 'RM', 'RH', 'I', 'C (all)'],
                'LandContour': ['Low', 'HLS', 'Bnk', 'Lvl'],
                'Alley': ['None', 'Grvl', 'Pave'],
                'LotConfig': ['CulDSac', 'Corner', 'Inside', 'FR2', 'FR3'],
                'Condition1': ['Norm', 'Artery', 'Feedr', 'PosN', 'PosA', 'RRAn', 'RRAe', 'RRNe', 'RRNn'],
                'Condition2': ['Norm', 'Artery', 'Feedr', 'PosN', 'PosA', 'RRAn', 'RRAe', 'RRNe', 'RRNn'],
                'Foundation': ['Slab', 'Wood', 'BrkTil', 'CBlock', 'PConc', 'Stone'],
                'Electrical': ['FuseP', 'FuseA', 'FuseF', 'Mix', 'SBrkr']
            }

            categories = [ordinal_mapping[i] for i in ordinal_mapping if i in enc]

            # Setting up OrdinalEncoder
            ordenc = OrdinalEncoder(categories=categories)
            ord_enc = ordenc.fit_transform(data_ord_enc)    
            data_ord_enc = pd.DataFrame(ord_enc,columns=data_ord_enc.columns)
            for i in data_ord_enc.columns:
                data[i] = data_ord_enc[i].values
            # We have fixed the ordinal data

            # Now going for the nominal data

            # Creating DF of Nominal Data:
            nom = ['Street', 'Neighborhood', 'BldgType', 'HouseStyle', 'RoofStyle',
                                'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Heating',
                                'GarageType', 'MiscFeature', 'SaleType', 'SaleCondition']
            nom = [i for i in nom if i in features]
            
            if nom is None:
                data_nom_enc = data[nom]
                # Encoding the Nominal Data
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                data_nom_enc = data_nom_enc.apply(le.fit_transform)
                pd.DataFrame(data_nom_enc)
                for i in data_nom_enc.columns:
                    data[i] = data_nom_enc[i].values
                # We have encoded the ordinal and nominal data    
            return data    
        
        except Exception as e:
            raise CustomException(e, sys)
        
    def input_transform_generator(self, test_data_path, features):
        try:      
            X_test = pd.read_csv(test_data_path)
            X_test = X_test.rename(columns={'1stFlrSF':'firstFlrSF','2ndFlrSF':'scndFlrSF'})
            X_test.drop(['Id'],axis=1,inplace=True)
            #encoding ordinal and nominal data
            X_test = self.encode(X_test, features)
            
            #handling cont(numerical) data
            file_path = self.input_transf_Confg.file
            min_max_data = load_object(file_path)

            min_SP_train = min_max_data.loc['min','SalePrice']
            max_SP_train = min_max_data.loc['max','SalePrice']

            if 'SalePrice' in X_test.columns:
                X_test['SalePrice'] = np.log1p(X_test['SalePrice'])
                X_test['SalePrice'] = ((X_test['SalePrice'] - min_SP_train) / (max_SP_train - min_SP_train))
                Y_test = X_test['SalePrice']

                cont = ['LotFrontage', 'LotArea','MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF',
                        'firstFlrSF','scndFlrSF','GrLivArea','GarageArea','WoodDeckSF','OpenPorchSF',
                        'EnclosedPorch','ScreenPorch']
                
                # log Tranformation
                log_tr = [i for i in cont if i in features]
                X_test[log_tr] = np.log1p(X_test[log_tr])
                
                X_test = X_test[features]
                #MinMax Scaling
                for column in X_test.columns:
                    X_test[column] =  (X_test[column] - min(min_max_data[column])) / (max(min_max_data[column]) - min(min_max_data[column]))
                return X_test, Y_test
            else:
                cont = ['LotFrontage', 'LotArea','MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF',
                        'firstFlrSF','scndFlrSF','GrLivArea','GarageArea','WoodDeckSF','OpenPorchSF',
                        'EnclosedPorch','ScreenPorch']
                
                # log Tranformation
                log_tr = [i for i in cont if i in features]
                X_test[log_tr] = np.log1p(X_test[log_tr])
                
                X_test = X_test[features]
                #MinMax Scaling
                for column in X_test.columns:
                    X_test[column] =  (X_test[column] - min(min_max_data[column])) / (max(min_max_data[column]) - min(min_max_data[column]))
                return X_test

        except Exception as e:
            raise CustomException(e, sys)        