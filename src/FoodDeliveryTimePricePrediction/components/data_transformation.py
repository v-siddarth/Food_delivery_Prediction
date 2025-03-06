import os
import sys
import numpy as np
import pandas as pd

from src.FoodDeliveryTimePricePrediction.logger import logging
from src.FoodDeliveryTimePricePrediction.exception  import customexception
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OrdinalEncoder
from src.FoodDeliveryTimePricePrediction.utils.utils import save_object

class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join("artifacts","preprocessor.pkl")
class DataTransformation:
    def __init__(self):
        self.config=DataTransformationConfig()
    def get_data_transformation(self):
        try:
            logging.info("getting data transformation(pipelines)")
            cat_cols=['Weather','Traffic_Level','Time_of_Day','Vehicle_Type']
            num_cols=['Distance_km','Preparation_Time_min','Courier_Experience_yrs']
            logging.info("separate cat and num columns")
            
            Weather_categories= ["Windy","Snowy","Foggy","Rainy","Clear"]
            Traffic_Level_categories=["High","Low","Medium"]
            Time_of_Day_categories= ["Night","Afternoon","Evening","Morning"]
            Vehicle_Type_categories= ["Car","Scooter","Bike"]
            logging.info("ranked categories")
            
            logging.info("start pipelines")
            
            num_pipeline=Pipeline(
                [
                    ("impute",SimpleImputer()),
                    ("scaler",StandardScaler())
                ]
            )
            
            cat_pipeline=Pipeline(
                [
                    ("impute",SimpleImputer(strategy="most_frequent")),
                    ("encoder",OrdinalEncoder(categories=[Weather_categories,Traffic_Level_categories,Time_of_Day_categories,Vehicle_Type_categories]))
                ]
            )
            
            logging.info("create preprocessor")
            
            preprocessor=ColumnTransformer(
                [
                    ("num",num_pipeline,num_cols),
                    ("cat",cat_pipeline,cat_cols)
                ]
            )
            return preprocessor
        except Exception as e:
            logging.info("error in getting data transformation",e)
            raise customexception(e,sys)
    def initate_data_transformation(self,train_data_path,test_data_path):
        try:
            logging.info("Starting data transformation")
            train_data=pd.read_csv(train_data_path)
            test_data=pd.read_csv(test_data_path)
            logging.info("read train and test data from artifacts")
            logging.info(f"this is train data:{train_data.head(2).to_string()}")
            logging.info(f"this is test data:{test_data.head(2).to_string()}")
            
            preprocessor=self.get_data_transformation()
            
            target_features="Delivery_Time_min"
            drop_features=[target_features,"Order_ID"]
            
            input_feature_train_data=train_data.drop(drop_features,axis=1)
            target_feature_train_data=train_data[target_features]
            
            input_feature_test_data=test_data.drop(drop_features,axis=1)
            target_feature_test_data=test_data[target_features]
            logging.info("splitting data into independent and dependent features")
            
            input_feature_train_arr=preprocessor.fit_transform(input_feature_train_data)
            input_feature_test_arr=preprocessor.transform(input_feature_test_data)
            logging.info("applying preprocessing on training and testing datasets")
            
            train_arr=np.c_[input_feature_train_arr,np.array(target_feature_train_data)]
            test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test_data)]
            logging.info("created train and test arrays")
            
            save_object(
                file_path=self.config.preprocessor_obj_file_path,
                obj=preprocessor
            )
            
            return train_arr,test_arr
         
        except Exception as e:
            logging.info("error in data transformation",e)
            raise customexception(e,sys)
        