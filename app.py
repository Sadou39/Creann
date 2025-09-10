import streamlit as st
import pandas as pd
import numpy as np
import shap
import joblib
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Chargement
model = load_model("model.h5")  # Updated path
scaler = joblib.load("scaler.pkl")  # Updated path
feature_names = joblib.load("feature_names.pkl")  # Updated path

# Titre
st.set_page_config(page_title="Prédiction du Risque de Crédit")
st.title(" Application : Risque de Crédit")

# Menu
menu = st.sidebar.radio("Menu", ["Exploration statistique", "Système de prédiction"])

# Exploration statistique
if menu == "Exploration statistique":
    st.header(" Exploration statistique")

    try:
        df = pd.read_csv("clean_data.csv")  # Updated path
        st.subheader("Statistiques descriptives")
        st.write(df.describe())

        st.subheader("Histogramme")
        variable = st.selectbox("Choisir une variable", df.columns[:-1])
        st.bar_chart(df[variable])

        st.subheader("Matrice de corrélation")
        st.dataframe(df.corr().style.background_gradient(cmap='coolwarm'))

    except FileNotFoundError:
        st.error("Le fichier 'clean_data.csv' est manquant.")

# Prédiction
elif menu == "Système de prédiction":
    st.header(" Système de prédiction")

    st.markdown(" Veuillez remplir les caractéristiques du client : ")
    user_input = {}
    for feat in feature_names:
        user_input[feat] = st.number_input(f"{feat}", value=0.0)

    if st.button("Prédire"):
        input_df = pd.DataFrame([user_input])
        
        # Proper scaling
        input_scaled = scaler.transform(input_df)
        
        # Model prediction
        prediction = model.predict(input_scaled)[0][0]

        result = " Crédit refusé (risque élevé)" if prediction > 0.5 else " Crédit accordé (risque faible)"
        st.subheader(f"Résultat : {result}")
        st.write(f"Score : `{prediction:.2f}`")

        # Background data for SHAP
        try:
            df_bg = pd.read_csv("clean_data.csv")
            background = scaler.transform(df_bg[feature_names].sample(50, random_state=0))
        except Exception:
            background = np.zeros((50, len(feature_names)))

        # Compute SHAP values
        try:
            explainer = shap.DeepExplainer(model, background)
            shap_values = explainer.shap_values(input_scaled)
        except Exception:
            explainer = shap.KernelExplainer(model.predict, background)
            shap_values = explainer.shap_values(input_scaled)

        # Extract and flatten SHAP values for the first sample
        shap_values_sample = shap_values[0][0]  # shape (n_features,)

        # Build DataFrame for top features
        shap_df = pd.DataFrame({
            "feature": feature_names,
            "shap_value": shap_values_sample
        }).set_index("feature").abs().sort_values(by="shap_value", ascending=False).head(3)

        st.markdown(" Principales variables contributives :")
        for feat, row in shap_df.iterrows():
            st.write(f"- **{feat}** → impact absolu : `{row['shap_value']:.4f}`")

        # Force plot
        st.markdown("Visualisation SHAP")
        shap.initjs()
        fig = plt.figure(figsize=(20, 3))
        shap.plots.force(shap_values_sample, matplotlib=True)
        st.pyplot(fig)
