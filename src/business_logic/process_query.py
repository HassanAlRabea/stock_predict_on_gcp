#%%
import numpy as np
import pandas as pd

from src.IO.fetch_stock_data import get_last_stock_price
from src.model.moving_avg import get_preds_mov_avg

test_size = 0.1                 # proportion of dataset to be used as test set
cv_size = 0.1                   # proportion of dataset to be used as cross-validation set
N_opt_ma = 2

def process_data(ticker):
    ## MA Final Model
    results_ma = pd.DataFrame()

    # for ticker in sp_500:
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

    # Split into train, cv, and test
    train = df[:num_train].copy()
    cv = df[num_train:num_train+num_cv].copy()
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

# %%
process_data('nflx')