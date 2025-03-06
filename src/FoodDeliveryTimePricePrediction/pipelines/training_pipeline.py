import os
import sys
import pandas as pd
import numpy as np

from src.FoodDeliveryTimePricePrediction.logger import logging
from src.FoodDeliveryTimePricePrediction.exception import customexception

from src.FoodDeliveryTimePricePrediction.components.data_ingestion import DataIngestion
from src.FoodDeliveryTimePricePrediction.components.data_transformation import DataTransformation
from src.FoodDeliveryTimePricePrediction.components.model_trainer import ModelTrainer


class TrainingPipeline:
    def start_data_ingestion(self):
        try:
            data_ingestion=DataIngestion()
            train_data_path,test_data_path=data_ingestion.initate_data_ingestion()
            return train_data_path,test_data_path
        except Exception as e:
            logging.info("error in data ingestion",e)
            raise customexception(e,sys)
                
    def start_data_transformation(self,train_data_path,test_data_path):
        try:
            data_transformation=DataTransformation()
            train_arr,test_arr=data_transformation.initate_data_transformation(train_data_path,test_data_path)
            return train_arr,test_arr
        except Exception as e:
            logging.info("error in data transformation",e)
            raise customexception(e,sys)        
    
    def start_training(self):
        try:
            logging.info("Training pipeline started")
            train_data_path,test_data_path=self.start_data_ingestion()
            train_arr,test_arr=self.start_data_transformation(train_data_path,test_data_path)
        except Exception as e:
            logging.info("error in training pipeline",e)
            raise customexception(e,sys)

trainer=TrainingPipeline()
trainer.start_training()
print("training pipeline completed successfully")