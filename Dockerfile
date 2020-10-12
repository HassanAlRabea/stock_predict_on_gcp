FROM continuumio/miniconda3

# Allow statements and log messages to immediately appear in the Knative logs
ENV PYTHONUNBUFFERED True

#Sets our environment variable ENV to APP_HOME / app.
ENV APP_HOME /app

#The WORKDIR line sets our working directory to /app. Then, the Copy line makes local files available in the docker container.
WORKDIR $APP_HOME
COPY . ./

#The next three lines involve setting up the environment and executing it on the server.
RUN conda env create -f scripts/environment.yml
SHELL ["conda", "run", "-n", "stock_predict_on_gcp", "/bin/bash", "-c"]

#Instaling flask server
RUN pip install Flask gunicorn

# Service must listen to $PORT environment variable.
# This default value facilitates local development.
ENV PORT 8080
# Run the web service on container startup. Here we use the gunicorn
# webserver, with one worker process and 8 threads.
# For environments with multiple CPU cores, increase the number of workers
# to be equal to the cores available.
CMD exec gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 8 --timeout 0 app:app
