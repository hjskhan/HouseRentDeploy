import os, sys
from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformer, DataTransformerConfig

import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

@dataclass
class DataIngestionConfig:
    train_data_path:str = os.path.join('artifacts', 'train.csv')
    test_data_path:str = os.path.join('artifacts', 'test.csv')
    raw_data_path:str = os.path.join('artifacts', 'raw_data.csv')

class DataIngestion:
    def __init__(self):
        self.ingest_config = DataIngestionConfig()
    
    def initiate(self):
        try:
            df = pd.read_csv('notebook\data.csv')

            os.makedirs(os.path.dirname(self.ingest_config.train_data_path), exist_ok=True)
            df.to_csv(self.ingest_config.raw_data_path, index=False)

            df_train, df_test = train_test_split(df, random_state=0, test_size=0.2)
            df_train.to_csv(self.ingest_config.train_data_path,index = False, header= True)
            df_test.to_csv(self.ingest_config.test_data_path, index=False, header=True)

            return(
                self.ingest_config.train_data_path, 
                self.ingest_config.test_data_path
            )
            
        except Exception as e:
            raise CustomException(e, sys)
    
if __name__=='__main__':
    obj = DataIngestion()
    train_data, test_data = obj.initiate()

    data_transf = DataTransformer()
    train_data_arr, test_data_arr, _ = data_transf.initiate_transf(train_data, test_data)


