import os
import sys

import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, FunctionTransformer
from sklearn.pipeline import Pipeline


from dataclasses import dataclass

from src.exception import CustomException
from src.utils import save_object

@dataclass
class DataTransformerConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformer:
    def __init__(self) -> None:
        self.data_transf_confg = DataTransformerConfig()

    # first we define a function to encode the ordinal and nominal data
    def encode(self, data):
        try:
            data_ord_enc = data[['LotShape','LandSlope','ExterQual','ExterCond',
                        'HeatingQC','KitchenQual',
                        'BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2',
                        'CentralAir','Functional','GarageFinish','PavedDrive','PoolQC','Fence',
                       'FireplaceQu','GarageQual','GarageCond','MSZoning','Utilities','LandContour',
                       'Alley','LotConfig','Condition1','Condition2','Foundation','Electrical']]
            ## First creating a map for different ordinal data:
            ord1 = ['IR3','IR2','IR1','Reg'] #LotShape
            ord2 = ['Gtl','Mod','Sev'] #LandSlope
            ord3 = ['Po','Fa','TA','Gd','Ex'] #ExterQual,ExterCond,HeatingQC,KitchenQual
            ord4 = ['No Basement','Po','Fa','TA','Gd','Ex'] #BsmtQual,BsmtCond,
            ord5 = ['No Basement','No','Mn','Av','Gd'] #BsmtExposure
            ord6 = ['No Basement','Unf','LwQ','Rec','BLQ','ALQ','GLQ'] #BsmtFinType1,BsmtFinType2
            ord7 = ['N','Y'] #CentralAir
            ord8 = ['Sal','Sev','Maj2','Maj1','Mod','Min2','Min1','Typ'] #Functional
            ord9 = ['No Garage','Unf','RFn','Fin'] #GarageFinish
            ord10 = ['N','P','Y'] #PavedDrive
            ord11 = ['No Pool','Fa','TA','Gd','Ex'] #PoolQC
            ord12 = ['No Fence','MnWw','GdWo','MnPrv','GdPrv'] #Fence
            ord13 = ['No Fireplace','Po','Fa','TA','Gd','Ex'] #FireplaceQu
            ord14 = ['No Garage','Po','Fa','TA','Gd','Ex'] #GarageQual,GarageCond
            ord15 = ['FV','A','RL','RP','RM','RH','I','C (all)']
            ord16 = ['ELO','NoSeWa','NoSewr','AllPub']
            ord17 = ['Low','HLS','Bnk','Lvl']
            ord18 = ['None','Grvl','Pave']
            ord19 = ['CulDSac','Corner','Inside','FR2','FR3']
            ord20 = ['Norm','Artery','Feedr','PosN','PosA','RRAn','RRAe','RRNe','RRNn']
            ord21 = ['Slab','Wood','BrkTil','CBlock','PConc','Stone']
            ord22 = ['FuseP','FuseA','FuseF','Mix','SBrkr']
            # Creating Categories
            categories = [ord1,ord2,ord3,ord3,ord3,ord3,ord4,ord4,ord5,ord6,ord6,ord7,ord8,ord9,
                        ord10,ord11,ord12,ord13,ord14,ord14,ord15,ord16,ord17,ord18,ord19,
                        ord20,ord20,ord21,ord22]
            # Setting up OrdinalEncoder
            ordenc = OrdinalEncoder(categories=categories)
            ord_enc = ordenc.fit_transform(data_ord_enc)    
            data_ord_enc = pd.DataFrame(ord_enc,columns=data_ord_enc.columns)

            for i in data_ord_enc.columns:
                data[i] = data_ord_enc[i].values
            # We have fixed the ordinal data

            # Now going for the nominal data

            # Creating DF of Nominal Data:
            data_nom_enc = data[['Street', 'Neighborhood', 'BldgType', 'HouseStyle', 'RoofStyle',
                                'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Heating',
                                'GarageType', 'MiscFeature', 'SaleType', 'SaleCondition']]
            # data_nom_enc = data[data_nom_enc]
            # Encoding the Nominal Data
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            data_nom_enc = data_nom_enc.apply(le.fit_transform)
            for i in data_nom_enc.columns:
                data[i] = data_nom_enc[i].values
            # We have encoded the ordinal and nominal data    
            return data    
        except Exception as e:
            raise CustomException(e, sys)

    def fillNANs(self, data):
        try:
            # filling Nans for Categorical Columns
            for i in ['Alley', 'MasVnrType', 'MiscFeature']:
                data[i] = data[i].fillna('None')
            for i in ['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2']:
                data[i]= data[i].fillna('No Basement')
            data['Electrical'] = data['Electrical'].fillna(data['Electrical'].mode()[0])
            data['FireplaceQu'] = data['FireplaceQu'].fillna('No Fireplace')
            for i in ['GarageType','GarageFinish','GarageQual','GarageCond']:
                data[i] = data[i].fillna('No Garage')
            data['PoolQC'] = data['PoolQC'].fillna('No Pool') 
            data['Fence'] = data['Fence'].fillna('No Fence')
            # till now we fixed Nans for Categorical Variables.
    
            # Fill NaN values with the mode in continuous columns
            continuous_cols = data.select_dtypes(include=['float64', 'int64']).columns
            for i in continuous_cols:
                mode = data[i].mode()[0]
                data[i] = data[i].fillna(value=mode)

            return data
        except Exception as e:
            raise CustomException(e, sys)
        
    def Standardize(self, data):
        try:
            cont = ['LotFrontage', 'LotArea','MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF',
                    'firstFlrSF','scndFlrSF','GrLivArea','GarageArea','WoodDeckSF','OpenPorchSF',
                    'EnclosedPorch','ScreenPorch']
            # First we drop ID and SalePrice from data
            data.drop(['Id'],axis=1,inplace=True)
            ##### Log Transformation
            # max_target = data['SalePrice'].max()
            # min_target = data['SalePrice'].min()
            data[cont] = np.log1p(data[cont])
            data['SalePrice'] = np.log1p(data['SalePrice'])
            # min_SP_train = data['SalePrice'].min()
            # max_SP_train = data['SalePrice'].max()
            ##### Standard Scaler Transformation:
            min_val_train,max_val_train = [],[]
            # MinMaxScaler:
            for column in data.columns:
                min_val = data[column].min()
                min_val_train.append(min_val)
                max_val = data[column].max()
                max_val_train.append(max_val)
                data[column] = ((data[column] - min_val) / (max_val - min_val))
            return data


        except Exception as e:
            raise CustomException(e, sys)

    def custom_pipe(self):# creating a custom pipe by combining all above transformations
        try:
            encoding = FunctionTransformer(self.encode)
            filling_Nans = FunctionTransformer(self.fillNANs)
            Standardizing = FunctionTransformer(self.Standardize)

            pipe = Pipeline([
                ('fillNAns',filling_Nans),
                ('encode',encoding),
                ('standardize',Standardizing)
            ])  
            return pipe
                
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_transf(self,train_data_path, test_data_path):
        try:
            train = pd.read_csv(train_data_path)
            test = pd.read_csv(test_data_path)

            train = train.rename(columns={'1stFlrSF':'firstFlrSF','2ndFlrSF':'scndFlrSF'})
            test = test.rename(columns={'1stFlrSF':'firstFlrSF','2ndFlrSF':'scndFlrSF'})

            preprocessor = self.custom_pipe()

            train_arr = preprocessor.fit_transform(train)
            test_arr = preprocessor.fit_transform(test)

            y_train = train['SalePrice'] #target Variable
            y_test = test['SalePrice']
            train.drop('SalePrice',axis=1,inplace=True)
            test.drop('SalePrice',axis=1,inplace=True)

            save_object(
                file_path=self.data_transf_confg.preprocessor_obj_file_path,
                obj=preprocessor
            )

            return (train_arr, test_arr, self.data_transf_confg.preprocessor_obj_file_path)

        except Exception as e:
            raise CustomException(e, sys)
