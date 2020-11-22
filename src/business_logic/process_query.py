import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
from src.algo.stock_model import get_transformed_data, get_train_test_split, get_best_parameters
from src.IO.fetch_stock_data import get_sp500_tickers


def balanced_accuracy_result_clf():
    ## MA Final Model
    results_ma = pd.DataFrame()
    sp_500 = get_sp500_tickers()
    for ticker in sp_500:
        df = get_transformed_data(ticker, True)
        y_traincv, X_traincv, y_test, X_test = get_train_test_split(df)
        n_estimators, max_depth = get_best_parameters(df)
        rfr_final = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, bootstrap=True)
        rfr_final.fit(X_traincv, y_traincv)
        y_pred = rfr_final.predict(X_test)
        results_ma = results_ma.append([[ticker,y_test[-1], y_pred[-1]]], ignore_index = True)

    ba = balanced_accuracy_score(np.array(results_ma[1]), np.array(results_ma[2]))
    
    return ba
