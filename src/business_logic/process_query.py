
import numpy as np
import pandas as pd

from sklearn.metrics import balanced_accuracy_score
from datetime import date, datetime, time, timedelta
from src.IO.fetch_stock_data import get_last_stock_price
from src.IO.fetch_stock_data import get_sp500_tickers
from src.algo.stock_model import get_preds_mov_avg

test_size = 0.1                 # proportion of dataset to be used as test set
cv_size = 0.1                   # proportion of dataset to be used as cross-validation set
N_opt_ma = 2

def process_data(ticker):
    ## Moving Average Final Model
    results_ma = pd.DataFrame()

    # for ticker in sp_500:
    df = get_last_stock_price(ticker, True)

    #Extending df to get tomorrow's date
    nextdf = pd.DataFrame(np.zeros(len(df.columns)).reshape(1,len(df.columns)), columns = df.columns)
    as_list = nextdf.index.tolist()
    idx = as_list.index(0)
    as_list[idx] =  pd.to_datetime(pd.datetime.today()+ timedelta(days=1)).strftime('%Y-%m-%d')
    nextdf.index = as_list
    df = df.append(nextdf)

    # Convert Date column to datetime
    df.loc[:, 'date'] = pd.to_datetime(df.index,format='%Y-%m-%d')
    # Change all column headings to be lower case, and remove spacing
    df.columns = [str(x).lower().replace(' ', '_') for x in df.columns]
    # # # Get month of each sample
    df['month'] = df['date'].dt.month
    # # Sort by datetime
    df.sort_values(by='date', inplace=True, ascending=True)

    num_cv = int(cv_size*len(df))
    num_test = 1
    num_train = len(df) - num_cv - num_test

    #Train and CV DF - for simpler models that do not have hyperparameters it becomes and 80/20 split
    train_cv = df[:num_train+num_cv].copy()
    test = df[num_train+num_cv:].copy()

    ## Get the prediction
    est_list = get_preds_mov_avg(df, 'close', N_opt_ma, 0, num_train+num_cv)
    test.loc[:, 'predicted_price'] = est_list
    test.loc[:, 'previous_price'] = train_cv.iloc[-1,3]
    results_ma = results_ma.append(test)
    results_ma['recommendation'] = np.where(results_ma['predicted_price'] > results_ma['previous_price'], "Buy", "Sell")
    results_ma['actual'] = np.where(results_ma['close'] > results_ma['previous_price'], "Buy", "Sell")
    results_ma['accuracy'] = np.where(results_ma['recommendation'] == results_ma['actual'], 1, 0)

    return results_ma

def balanced_accuracy_result():
    ## MA Final Model
    results_ma = pd.DataFrame()
    sp_500 = get_sp500_tickers()
    for ticker in sp_500:
        df = get_last_stock_price(ticker, True)
        # Convert Date column to datetime
        df.loc[:, 'date'] = pd.to_datetime(df.index,format='%Y-%m-%d')
        # Change all column headings to be lower case, and remove spacing
        df.columns = [str(x).lower().replace(' ', '_') for x in df.columns]
        # # # Get month of each sample
        df['month'] = df['date'].dt.month
        # # Sort by datetime
        df.sort_values(by='date', inplace=True, ascending=True)

        num_cv = int(cv_size*len(df))
        num_test = 1
        num_train = len(df) - num_cv - num_test

        #Train and CV DF - for simpler models that do not have hyperparameters it becomes and 80/20 split
        train_cv = df[:num_train+num_cv].copy()
        test = df[num_train+num_cv:].copy()

        ## Get the prediction
        est_list = get_preds_mov_avg(df, 'close', N_opt_ma, 0, num_train+num_cv)
        test.loc[:, 'predicted_price'] = est_list
        test.loc[:, 'previous_price'] = train_cv.iloc[-1,3]
        results_ma = results_ma.append(test)

    results_ma['recommendation'] = np.where(results_ma['predicted_price'] > results_ma['previous_price'], "Buy", "Sell")
    results_ma['actual'] = np.where(results_ma['close'] > results_ma['previous_price'], "Buy", "Sell")
    results_ma['accuracy'] = np.where(results_ma['recommendation'] == results_ma['actual'], 1, 0)
    ba = balanced_accuracy_score(np.array(results_ma['actual']), np.array(results_ma['recommendation']))
    return ba
