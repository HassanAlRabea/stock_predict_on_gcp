import configparser
import logging

import joblib

from src.IO.fetch_stock_data import get_last_stock_price
from src.IO.storage_tools import create_bucket, get_model_from_bucket, upload_file_to_bucket
from src.algo.stock_model import get_transformed_data
from src.algo.stock_model import Stock_model


def create_business_logic(ticker):
    # This line here will return a dataframe for the ticker and save
    # in data_fetcher
    data_fetcher = get_transformed_data(ticker)
    #Returns the creation of the businesslogic object containing the model in question
    #which was trained on a specific ticker
    return BusinessLogic(Stock_model(data_fetcher))


class BusinessLogic:
    #Creates the business logic object
    def __init__(self, model_creator):
        self._root_bucket = 'ticker_model_bucket'
        self._config = configparser.ConfigParser()
        self._config.read('application.conf')
        # Seems like model_creator is simply the stock model in question
        self._model_creator = model_creator
        self._create_bucket()

    def get_version(self):
        #Reads the version number from the conf file
        return self._config['DEFAULT']['version']

    def get_bucket_name(self):
        #Obtains the bucket name and version
        return f'{self._root_bucket}_{self.get_version().replace(".", "")}'

    def _get_or_create_model(self, ticker):
        log = logging.getLogger()
        #Sets the filename of the model as ticker.pkl
        model_filename = self.get_model_filename_from_ticker(ticker)
        #Sets model to the saved model or none if it doesn't exist
        model = get_model_from_bucket(model_filename, self.get_bucket_name())
        if model is None:
            log.warning(f'training model for {ticker}')
            #Here is where we fit the model if it doesn't already exist
            model = self._model_creator.fit()
            with open(model_filename, 'wb') as f:
                joblib.dump(model, f)
            #Where we upload and save the new model
            upload_file_to_bucket(model_filename, self.get_bucket_name())
        return model

    def get_model_filename_from_ticker(self, ticker):
        return f'{ticker}.pkl'

    def _create_bucket(self):
        #Creates the bucket if doesn't already exist with the version number
        create_bucket(self.get_bucket_name())

    def do_predictions_for(self, ticker):
        model = self._get_or_create_model(ticker)
        predictions = model.predict()
        return predictions
