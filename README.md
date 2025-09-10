# Credit Risk Web Application

This repository contains a Streamlit web application for exploring a credit‑risk dataset and making predictions on whether a loan should be approved or not.

## Features

1. **Statistical Exploration**
   - Displays descriptive statistics (mean, standard deviation, etc.) for the dataset.
   - Allows users to draw histograms for any variable in the dataset.
   - Shows a correlation matrix with an interactive heatmap.

2. **Prediction System**
   - Users can manually enter values for 20 features related to a new customer (age, income, status, etc.).
   - A pre‑trained model (loaded from `models/model.h5`) predicts whether the credit should be granted or not.
   - The three most influential features for each prediction are displayed using SHAP values.

## Getting Started

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare the data and models**

   The necessary files are already included in this repository:

   - `data/clean_data.csv`: cleaned dataset used for exploration.
   - `models/model.h5`: pre‑trained Keras model.
   - `models/scaler.pkl`: scikit‑learn scaler used to preprocess input features.
   - `models/feature_names.pkl`: list of feature names expected by the model.

3. **Run the app**

   ```bash
   streamlit run app.py
   ```

   After running the command, Streamlit will provide a local URL. Open it in your browser to use the web application.

## Notes

The SHAP analysis may take a few seconds to compute on each prediction. Ensure that the `shap` library is installed (it is included in `requirements.txt`).