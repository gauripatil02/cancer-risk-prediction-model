# Cancer Risk Prediction System
This project implements a machine learning-based web application to predict the risk of cancer based on various health metrics. The application uses a `RandomForestClassifier` model and provides a user-friendly interface built with [Streamlit](https://streamlit.io/).

## Project Structure

* `app.py`: The main Streamlit application file that manages the user interface and model inference.
* `train_model.py`: The script used to train the machine learning model using the provided dataset.
* `The_Cancer_data_1500_V2.csv`: The dataset containing health metrics used for training the model.
* `cancer_model.pkl`: The serialized `RandomForestClassifier` model.
* `scaler.pkl`: The serialized `StandardScaler` used to normalize input features.
* `style.css`: Custom CSS file to style the Streamlit interface.

## Prerequisites
To run this project, you will need to have Python installed. It is recommended to use a virtual environment. Install the necessary dependencies:

bash
pip install streamlit pandas numpy scikit-learn

## How It Works

1. **Data Processing**: The `train_model.py` script loads data from `The_Cancer_data_1500_V2.csv`, scales the features, and trains a `RandomForestClassifier`.
2. **Deployment**: The `app.py` script loads the trained model and scaler to make real-time predictions based on user input provided via the Streamlit web form.

## Running the Application

To launch the web interface, execute the following command in your terminal:

bash
streamlit run app.py

This will start a local server, and you can interact with the cancer risk prediction tool directly through your web browser.
