# Driver Acceptance Prediction Model



## Description

The Driver Acceptance Prediction App is a web application that utilizes machine learning models to predict whether the driver will accept or not accept the ride request given different data. The model is trained on ride-sharing record data and it has preprocessed to ensure better predictions. The app has been deployed on Streamlite.

Note: Deployed version of the web pages [Here](https://driver-acceptance-prediction.streamlit.app/)

## Notebooks and dataset

- [Dataset](https://www.kaggle.com/competitions/nyc-taxi-trip-duration/data)

- [Regression model](https://colab.research.google.com/drive/1_s6xxGU-0p4i6QhnpI7n1ZAlJYUptgHW?usp=sharing)


## Features
- Data input: Users can input the data for prediction as there fields on the page indicates.

- Prediction: After input all necessary data, users can click the "Predict" button to get the model's prediction regarding the acceptance status.

## Packages Used

This project has used the some packages such as numpy, tensorflow, which have to be installed to run this web app locally present in `requirements.txt` file. 

## Installation

To run the project locally, there is a need to have Visual Studio Code (vs code) installed on your PC:

- **[VS Code](https://code.visualstudio.com/download)**: It is a source-code editor made by Microsoft with the Electron Framework, for Windows, Linux, and macOS.

## Usage

1. Clone the project 

``` bash
git clone https://github.com/UmuhireJessie/acceptance-prediction.git

```

2. Open the project with vs code

``` bash
cd acceptance-prediction
code .
```

3. Install the required dependencies

``` bash
pip install -r requirements.txt
```


4. Run the project

``` bash
streamlit run app.py
```

5. Use the link printed in the terminal to visualise the app. (Usually `http://127.0.0.1:8501/`)

## Model Files

- driver_acceptance_model.h5: The main driver acceptance prediction model trained on ride-sharing records data.
- scaler.pkl: The scaler used for standardizing features during inference.

## Authors and Acknowledgment

- Jessie Umuhire Umutesi

## License
[MIT](https://choosealicense.com/licenses/mit/)
