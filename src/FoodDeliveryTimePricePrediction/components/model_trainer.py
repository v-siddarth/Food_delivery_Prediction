import os
import sys
import pandas as pd
import numpy as np

from src.FoodDeliveryTimePricePrediction.logger import logging
from src.FoodDeliveryTimePricePrediction.exception import customexception

class ModelTrainer:
    def initiate_model_training(self):
        try:
            logging.info("start model Training ")
        except Exception as e:
            logging.info("error in model Training",e)
            raise customexception(e,sys)
     