from flask import Flask

app = Flask(__name__)

from src.business_logic.process_query import process_data

@app.route('/', methods=['GET'])
def hello():
    return f'Hello! Please edit the URL with your desired ticker:!\nEX: /get_stock_val/<ticker>\n'

@app.route('/get_stock_val/<ticker>', methods=['GET'])
def get_stock_value(ticker):
    prediction = process_data(ticker).to_string(header = True, index = False)
    
    return prediction


if __name__ == '__main__':
    # Used when running locally only. When deploying to Cloud Run,
    # a webserver process such as Gunicorn will serve the app.
    app.run(host='localhost', port=8080, debug=True)
