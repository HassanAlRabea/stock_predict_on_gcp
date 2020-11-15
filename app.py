from flask import Flask

app = Flask(__name__)

from src.business_logic.process_query import balanced_accuracy_result_clf
from src.business_logic.process_query_improved import create_business_logic

@app.route('/', methods=['GET'])
def hello():
    return f'Hello! Please add the following to your URL with your desired ticker:!\nEX: /get_stock_val/<ticker>\n'

@app.route('/evaluate', methods=['GET'])
def evaluate():
    ba_score_clf = str(round(balanced_accuracy_result_clf() * 100,2)) + " %"
    answer = "Our latest SP 500 balanced accuracy score is: " + ba_score_clf
    return answer

@app.route('/get_stock_val/<ticker>', methods=['GET'])
def get_stock_value(ticker):
    bl = create_business_logic(ticker)
    prediction = bl.do_predictions_for(ticker)
    recommendation = prediction[1]
    ba_score_test = prediction[0]
    answer = "Tomorrow's prediction for " + str(ticker) + ": "  \
        + "We recommend you to " + recommendation + " it." \
        +" Note that our latest balanced accuracy score for this ticker is: " + ba_score_test 
    return answer

if __name__ == '__main__':
    # Used when running locally only. When deploying to Cloud Run,
    # a webserver process such as Gunicorn will serve the app.
    app.run(host='localhost', port=8080, debug=True)
