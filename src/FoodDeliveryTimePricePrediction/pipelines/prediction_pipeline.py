import os
import sys
import pandas as pd
import numpy as np

from src.FoodDeliveryTimePricePrediction.logger import logging
from src.FoodDeliveryTimePricePrediction.exception import customexception

class PredictionPipeline:
    def start_prediction(self):
        try:
            logging.info("Starting prediction pipeline")
        except Exception as e:
            logging.info("error in prediction pipeline",e)
            raise customexception(e,sys)