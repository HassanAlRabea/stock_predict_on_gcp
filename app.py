from flask import Flask
import pandas as pd

app = Flask(__name__)

from src.business_logic.process_query import process_data
from src.business_logic.process_query import balanced_accuracy_result

@app.route('/', methods=['GET'])
def hello():
    return f'Hello! Please add the following to your URL with your desired ticker:!\nEX: /get_stock_val/<ticker>\n'

@app.route('/get_stock_val/<ticker>', methods=['GET'])
def get_stock_value(ticker):
    prediction = process_data(ticker)
    # .to_string(header = True, index = False)
    pred_price = prediction['predicted_price'].to_string(header = True, index = False)
    recommendation = prediction['recommendation'].to_string(header = True, index = False)
    ba_score = str(round(balanced_accuracy_result() * 100,2)) + " %"
    answer = "Tomorrow's predicted price for " + str(ticker) + " is " + str(pred_price) \
        + ". We recommend you to " + recommendation + " it." \
        +" Note that our latest SP 500 balanced accuracy score is: " + ba_score
    return answer

def test():
    return f'Hello! Please edit the URL with your desired ticker:!\nEX: /get_stock_val/<ticker>\n'

if __name__ == '__main__':
    # Used when running locally only. When deploying to Cloud Run,
    # a webserver process such as Gunicorn will serve the app.
    app.run(host='localhost', port=8080, debug=True)
