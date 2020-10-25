from flask import Flask
from src.business_logic.process_query import process_data
app = Flask(__name__)


@app.route('/')
def hello_world():
    prediction = process_data('nflx')
    print('Hello World! Testing auto run deployment - Trial number 3')
    return print(prediction)

if __name__ == '__main__':
    # Used when running locally only. When deploying to Cloud Run,
    # a webserver process such as Gunicorn will serve the app.
    app.run(host='localhost', port=8080, debug=True)
