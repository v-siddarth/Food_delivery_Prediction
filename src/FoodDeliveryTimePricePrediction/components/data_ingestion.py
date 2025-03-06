import os
import sys
import numpy as np
import pandas as pd

from src.FoodDeliveryTimePricePrediction.logger import logging
from src.FoodDeliveryTimePricePrediction.exception import customexception
from pathlib import Path
from sklearn.model_selection import train_test_split

class DataIngestionConfig:
    raw_data_path:str=os.path.join("artifacts","raw_data.csv")
    train_data_path:str=os.path.join("artifacts","train_data.csv")
    test_data_path:str=os.path.join("artifacts","test_data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
    
    def initate_data_ingestion(self):
        
        try:
            logging.info("start data ingestion")
            
            data=pd.read_csv(Path(os.path.join("notebooks/data","Food_Delivery_Times.csv")))
            logging.info("data read sucessfully")
            
            os.makedirs(os.path.dirname(os.path.join(self.ingestion_config.raw_data_path)),exist_ok=True)
            logging.info("create artifacts directory sucessfully")
            
            data.to_csv(self.ingestion_config.raw_data_path,index=False)
            logging.info("raw data save in artifacts sucessfully")
            
            train_data,test_data=train_test_split(data,test_size=0.2,random_state=234)
            logging.info("train test split completed")
            
            train_data.to_csv(self.ingestion_config.train_data_path,index=False)
            test_data.to_csv(self.ingestion_config.test_data_path,index=False)
            logging.info("train and test data save in artifacts sucessfully")
            
            logging.info("data ingestion completed successfully")
            
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )
        
        except Exception as e:
            logging.info("error in data ingestion",e)
            raise customexception(e,sys)