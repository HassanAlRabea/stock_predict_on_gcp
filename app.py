from flask import Flask

app = Flask(__name__)

from process_query import process_data

@app.route('/')
def hello_world():
    prediction = process_data('aapl').to_string(header = True, index = False)
    return prediction

if __name__ == '__main__':
    # Used when running locally only. When deploying to Cloud Run,
    # a webserver process such as Gunicorn will serve the app.
    app.run(host='localhost', port=8080, debug=True)
