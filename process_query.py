
import numpy as np
import pandas as pd

from fetch_stock_data import get_last_stock_price

test_size = 0.1                 # proportion of dataset to be used as test set
cv_size = 0.1                   # proportion of dataset to be used as cross-validation set
N_opt_ma = 2

def get_preds_mov_avg(df, target_col, N, pred_min, offset):
    """
    Given a dataframe, get prediction at timestep t using values from t-1, t-2, ..., t-N.
    Using simple moving average.
    Inputs
        df         : dataframe with the values you want to predict. Can be of any length.
        target_col : name of the column you want to predict e.g. 'adj_close'
        N          : get prediction at timestep t using values from t-1, t-2, ..., t-N
        pred_min   : all predictions should be >= pred_min
        offset     : for df we only do predictions for df[offset:]. e.g. offset can be size of training set
    Outputs
        pred_list  : list. The predictions for target_col. np.array of length len(df)-offset.
    """
    pred_list = df[target_col].rolling(window = N, min_periods=1).mean() # len(pred_list) = len(df)
    
    # Add one timestep to the predictions
    pred_list = np.concatenate((np.array([np.nan]), np.array(pred_list[:-1])))
    
    # If the values are < pred_min, set it to be pred_min
    pred_list = np.array(pred_list)
    pred_list[pred_list < pred_min] = pred_min
    
    return pred_list[offset:]

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
