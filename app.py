from flask import Flask

app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello World! Fuck you Seb. Here it is: 8===D'


if __name__ == '__main__':
    # Used when running locally only. When deploying to Cloud Run,
    # a webserver process such as Gunicorn will serve the app.
    app.run(host='localhost', port=8080, debug=True)
