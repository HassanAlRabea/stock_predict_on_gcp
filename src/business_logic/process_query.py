
import numpy as np
import pandas as pd

from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import balanced_accuracy_score
from datetime import date, datetime, time, timedelta
from src.IO.fetch_stock_data import get_last_stock_price, get_sp500_tickers


def process_data_clf(ticker):

    # for ticker in sp_500:
    df = get_last_stock_price(ticker, True)

    ## Append tomorrow's date to the frame
    nextdf = pd.DataFrame(np.zeros(len(df.columns)).reshape(1,len(df.columns)), columns = df.columns)
    as_list = nextdf.index.tolist()
    idx = as_list.index(0)
    as_list[idx] =  pd.to_datetime(pd.datetime.today()+ timedelta(days=1)).strftime('%Y-%m-%d')
    nextdf.index = as_list
    df = df.append(nextdf)



    ## Transformations: add previous close and signal
    df['previous_close'] = df['close'].shift()
    df['signal'] = np.where(df['close'] > df['previous_close'], 0, 1)
    df = df.dropna()      


    # Convert Date column to datetime
    df.loc[:, 'date'] = pd.to_datetime(df.index,format='%Y-%m-%d')
    # Change all column headings to be lower case, and remove spacing
    df.columns = [str(x).lower().replace(' ', '_') for x in df.columns]
    # # # Get month of each sample
    df['month'] = df['date'].dt.month
    # # Sort by datetime
    df.sort_values(by='date', inplace=True, ascending=True)

    #### Input params ##################
    test_size = 0.1                 # proportion of dataset to be used as test set
    cv_size = 0.3                   # proportion of dataset to be used as cross-validation set
    num_cv = int(cv_size*len(df))
    num_test = int(test_size*len(df))
    #     num_test = 1
    num_train = len(df) - num_cv - num_test

    # Split into train, cv, and test
    # train = df[:num_train].copy()
    # cv = df[num_train:num_train+num_cv].copy()
    #Train and CV DF - for simpler models that do not have hyperparameters it becomes and 80/20 split
    train_cv = df[:num_train+num_cv].copy()
    test = df[num_train+num_cv:].copy()

    # Setting up and training the model
    rfr = RandomForestClassifier(n_estimators=100, max_depth=30, bootstrap=True)
    param_dist = dict(n_estimators=list(range(1,30)), max_depth=list(range(1,10)))
    y_traincv = train_cv['signal']
    X_traincv = train_cv[['previous_close', 'month']]
    y_test = test['signal']
    X_test = test[['previous_close', 'month']]

    # Training and doing CV to find the best parameters
    rand = RandomizedSearchCV(rfr, param_dist, cv=10, n_iter=20, random_state=0, verbose = 1)
    rand.fit(X_traincv, y_traincv)
    rand.cv_results_
    n_estimators = rand.best_params_['n_estimators']
    max_depth = rand.best_params_['max_depth']

    rfr_final = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, bootstrap=True)
    rfr_final.fit(X_traincv, y_traincv)
    y_pred = rfr_final.predict(X_test)
    ba = str(round(balanced_accuracy_score(np.array(y_test), y_pred)*100,2)) + " %"

    y_pred_text = np.where(y_pred == 0, "Buy", "Sell")
    tomorrow_recommend = y_pred_text[-1]
    results_clf = np.array([ba,tomorrow_recommend])
    return results_clf


def balanced_accuracy_result_clf():
    ## MA Final Model
    results_ma = pd.DataFrame()
    sp_500 = get_sp500_tickers()
    for ticker in sp_500:
        df = get_last_stock_price(ticker, True)

        ## Transformations: add previous close and signal
        df['previous_close'] = df['close'].shift()
        df['signal'] = np.where(df['close'] > df['previous_close'], 0, 1)
        df = df.dropna()      


        # Convert Date column to datetime
        df.loc[:, 'date'] = pd.to_datetime(df.index,format='%Y-%m-%d')
        # Change all column headings to be lower case, and remove spacing
        df.columns = [str(x).lower().replace(' ', '_') for x in df.columns]
        # # # Get month of each sample
        df['month'] = df['date'].dt.month
        # # Sort by datetime
        df.sort_values(by='date', inplace=True, ascending=True)

        #### Input params ##################
        test_size = 0.1                 # proportion of dataset to be used as test set
        cv_size = 0.3                   # proportion of dataset to be used as cross-validation set
        num_cv = int(cv_size*len(df))
        num_test = int(test_size*len(df))
        num_train = len(df) - num_cv - num_test

        # Split into train, cv, and test
        # train = df[:num_train].copy()
        # cv = df[num_train:num_train+num_cv].copy()
        #Train and CV DF - for simpler models that do not have hyperparameters it becomes and 80/20 split
        train_cv = df[:num_train+num_cv].copy()
        test = df[num_train+num_cv:].copy()

        # Setting up and training the model
        rfr = RandomForestClassifier(n_estimators=100, max_depth=30, bootstrap=True)
        param_dist = dict(n_estimators=list(range(1,30)), max_depth=list(range(1,10)))
        y_traincv = train_cv['signal']
        X_traincv = train_cv[['previous_close', 'month']]
        y_test = test['signal']
        X_test = test[['previous_close', 'month']]

        # Training and doing CV to find the best parameters
        rand = RandomizedSearchCV(rfr, param_dist, cv=10, n_iter=20, random_state=0, verbose = 1)
        rand.fit(X_traincv, y_traincv)
        rand.cv_results_
        n_estimators = rand.best_params_['n_estimators']
        max_depth = rand.best_params_['max_depth']

        rfr_final = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, bootstrap=True)
        rfr_final.fit(X_traincv, y_traincv)
        y_pred = rfr_final.predict(X_test)
        results_ma = results_ma.append([[ticker,y_test[-1], y_pred[-1]]], ignore_index = True)

    ba = balanced_accuracy_score(np.array(results_ma[1]), np.array(results_ma[2]))
    
    return ba
