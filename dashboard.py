import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay


background_css = """
<style>
body {
    background-image: url('https://fr.wikipedia.org/wiki/Seattle#/media/Fichier:Drapeau_de_Seattle.png');
    background-size: cover;
}
</style>
"""

@st.cache_resource
def load_model():
    # Remplacez ceci par le chargement de votre modèle entraîné
    model = joblib.load(os.path.join(os.path.abspath(os.getcwd()), "pipeline_prediction_ghge_mlp.joblib"))
    return model

model = load_model()

X = pd.read_csv(filepath_or_buffer=os.path.join(os.path.abspath(os.getcwd()), "donnees_avant_preprocessing.csv"))

# Interface utilisateur pour la sélection des paramètres
st.title("Prédictions MLPRegressor et Importance des Features")

# Ajoutez des sliders/inputs pour chaque feature
selected_building_type = st.selectbox(
    "What type of building do you own ?",
    X['BuildingType'].unique().tolist(),
)

selected_primary_property_type = st.selectbox(
    "What is the Primary property type in the selection below ?",
    X['PrimaryPropertyType'].unique().tolist(),
)

selected_year_built = st.number_input(
    label="What is the year of construction for the primary building ?",
    step=1,
    value=1950)

selected_building_number = st.number_input(
    label="How many buildings is there on the property ?",
    step=1,
    value=1)

selected_floor_number = st.number_input(
    label="How many floor is there in all the buildings of the property ?",
    step=1,
    value=1)

selected_gfa_all_building = st.number_input(
    label="GFA of all the building",
    step=0.1,
    value=100.0)

selected_largest_property_type = st.selectbox(
    "What is the largest property type in the selection below ?",
    X['LargestPropertyUseType'].unique().tolist(),
)

selected_gfa_largest_building = st.number_input(
    label="GFA of the largest building",
    step=0.1,
    value=100.0)

selected_secondary_property_type = st.selectbox(
    "What is the secondary property type in the selection below ?",
    X['SecondLargestPropertyUseType'].unique().tolist(),
)

selected_gfa_secondary_building = st.number_input(
    label="GFA of the secondary building",
    step=0.1,
    value=100.0)

selected_site_energy_use = st.number_input(
    label="Total Site Energy Use Comsuption (kBtu)",
    step=0.1,
    value=100.0)

selected_site_energy_use_wn = st.number_input(
    label="Total Site Energy Use Consumption WN (kBtu)",
    step=0.1,
    value=100.0)

selected_electricity = st.number_input(
    label="Total Electricty Consumption (kBtu)",
    step=0.1,
    value=100.0)

selected_natural_gaz = st.number_input(
    label="Total Natural Gaz Consumption (kBtu)",
    step=0.1,
    value=100.0)

# Création du vecteur d'entrée
X = pd.DataFrame.from_dict(
    {
        'BuildingType':[selected_building_type],
        'PrimaryPropertyType': [selected_primary_property_type],
        'YearBuilt': [selected_year_built],
        'NumberofBuildings': [selected_building_number],
        'NumberofFloors': [selected_floor_number],
        'PropertyGFABuilding(s)': [selected_gfa_all_building],
        'LargestPropertyUseType': [selected_largest_property_type],
        'LargestPropertyUseTypeGFA': [selected_gfa_largest_building],
        'SecondLargestPropertyUseType': [selected_secondary_property_type],
        'SecondLargestPropertyUseTypeGFA': [selected_gfa_secondary_building],
        'SiteEnergyUse(kBtu)': [selected_site_energy_use],
        'SiteEnergyUseWN(kBtu)': [selected_site_energy_use_wn],
        'Electricity(kBtu)': [selected_electricity],
        'NaturalGas(kBtu)': [selected_natural_gaz]
    }
    )  # Ajustez selon le nombre de features

# Prédiction
prediction = model.predict(X)[0]

st.write(f"Prédiction : {prediction:.2f}")

# Calcul et affichage de l'importance des features
st.subheader("Importance des Features")

fig, ax = plt.subplots(figsize=(10, 6))
PartialDependenceDisplay.from_estimator(model, X, features=X.columns.tolist(), ax=ax)
st.pyplot(fig)

# Explication de l'importance des features
st.write("""
Le graphique ci-dessus montre l'importance partielle de chaque feature. 
Les lignes représentent comment la prédiction change en fonction de la valeur de chaque feature, 
en maintenant les autres features constantes.
""")