##  Prediction of the Client's Response to the Bank's Offer

This repository contains an analysis of a bank's customer response data and a machine learning model that predicts whether a client will respond to a bank's offer.
The application is developed as a Applied Python course assignment during study at HSE University Master’s Programme Machine Learning and Data-Intensive Systems.

### Application

The application is hosted on Streamlit and can be accessed here: https://responcetobankoffer.streamlit.app/

### Files

- `EDA_DP_ML.ipynb`: Jupyter notebook containing exploratory data analysis, data preprocessing, and machine learning model training.
- `data.csv`: The dataset used for analysis and visualization.
- `encoder.pickle`: The saved ordinal encoder used for preprocessing categorical features.
- `main.py`: The main Python script running the Streamlit application.
- `model.pickle`: The saved machine learning model for predicting client responses.
- `requirements.txt`: The required packages for reproducing the analysis environment.
- `scaler.pickle`: The saved scaler used for preprocessing numerical features.
- `training_data.csv`: The preprocessed training data used for model training.

### Usage

To run the Streamlit application on your local machine, follow these steps:

1. Clone the repository.
2. Install the required packages using `pip install -r requirements.txt`.
3. Run the command `streamlit run main.py`.

### License

This project is licensed under the terms of the MIT License.

### Author
Albina Burlova
