# stock_predict_on_gcp
## Creating a Stock Prediction application on GCP

### Setting up the project
- Setup Github and Repo
- Setup VSC and extensions
- Setup conda and and create environment
- Make sure that Interpreter is using the python from the relevant conda env
- Make sure to have pylint and python interactive set up
- Set up Google Cloud Build
  - Create new project for this
  - Add a push trigger 
  - Add docker file into git
  - Test docker
- Setting up a google run instance
  - Requires a server so that it can listen and answer queries
  - Add a flask server
  - Setup continuous build

I was hoping to use it 
### Backlog for the next sprint
  - Setup project repo with the required initial dependencies
  - Define the business logic of the application
  - Analyze and understand the yahoo_fin package
  - Set up dummy server and python application file 
  - Build out code to accept inputs from flask server
  - Code the data import and transformation process of the application from yahoo_fin
  - Code model training and storing aspect
  - Test out model creation and prediction with simple models
  - Test out application workflow using simple trained models
  - Test out various models for the best one
