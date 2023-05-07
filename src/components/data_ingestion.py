import os
import sys
sys.path.append('E:/Machine Learning End to End Project/Old_Car_Price_Prediction/')
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass


@dataclass
class DataIngestionconfig:
    train_data_path:str = os.path.join('artifacts','train.csv')
    test_data_path:str = os.path.join('artifacts','test.csv')
    raw_data_path:str =  os.path.join('artifacts','raw.csv')


class DataIngestion:
    def __init__(self) -> None:
        self.ingestion_config = DataIngestionconfig()
    
    def initiate_data_ingestion(self):
        logging.info("Data Ingestion methods starts ")
        try:
            df = pd.read_csv(os.path.join('notebooks/data','CAR DETAILS FROM CAR DEKHO.csv'))
            logging.info('Dataset read a pandas dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False)

            logging.info("Train test split")
            train_set , test_set = train_test_split(df,test_size=0.30,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info('Ingestion of Data is completed ')

            return(
                self.ingestion_config.train_data_path,self.ingestion_config.test_data_path
            )



        except Exception as e:
            logging.info('Exception occured at Data Ingestion stage')
            raise CustomException(e,sys)
        

if __name__=='__main__':
    obj = DataIngestion()
    train_data_path , test_data_path = obj.initiate_data_ingestion()
    print(" Data Ingestion Done Sucessfully ")



