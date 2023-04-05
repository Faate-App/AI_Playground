import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder

df = pd.read_stata("stanford data set/HCMST 2017 fresh sample for public sharing draft v1.1.dta")

satisfaction_mapping = {
    'Refused': -1,
    'Excellent': 5,
    'Good': 4,
    'Fair': 3,
    'Poor': 2,
    'Very Poor': 1
}

# Replace satisfaction categories with numerical values
df['relationship_satisfaction_num'] = df['Q34'].map(satisfaction_mapping)

# Normalize relationship satisfaction (excluding 'Refused' responses)
df['relationship_satisfaction_norm'] = df[df['relationship_satisfaction_num'] != -1].apply(
    lambda row: (row['relationship_satisfaction_num'] - 1) / 4, axis=1
)

# Convert 'Q21B_Year' to numerical values
df['Q21B_Year_num'] = pd.to_numeric(df['Q21B_Year'], errors='coerce')

# Calculate relationship duration in years
study_year = 2017
df['relationship_duration'] = study_year - df['Q21B_Year_num']

# Normalize relationship duration
duration_min = df['relationship_duration'].min()
duration_max = df['relationship_duration'].max()
df['relationship_duration_norm'] = (df['relationship_duration'] - duration_min) / (duration_max - duration_min)

# Map marital status to numerical values
marital_status_mapping = {
    'No, I am not Married': 0,
    'Yes, I am Married': 1
}

df['marital_status_num'] = df['S1'].map(marital_status_mapping)

df['compatibility_score'] = 0.7 * df['relationship_satisfaction_norm'] + 0.2 * df['relationship_duration_norm'] + 0.1 * df['marital_status_num'].astype(int)

# Sélection des caractéristiques d'intérêt
features = [
    'Q6B', 'Q9', 'Q10',
    'ppeduc', 'ppage', 'ppgender', 'ppethm',
]

# For each feature, display number of NA rows
for feature in features:
    print(feature, df[feature].isna().sum())

target = 'compatibility_score'

dfClean = df[features + [target]]

# Convert Q9 to int
dfClean['Q9'] = pd.to_numeric(dfClean['Q9'], errors='coerce')

# Gestion des valeurs manquantes
dfClean = dfClean.dropna()

# Encodage des variables catégorielles
cat_features = ['Q6B', 'Q10', 'ppeduc', 'ppgender', 'ppethm']
enc = OneHotEncoder()
encoded_features = enc.fit_transform(dfClean[cat_features]).toarray()
encoded_feature_names = enc.get_feature_names_out(cat_features)

df_encoded = pd.DataFrame(encoded_features, columns=encoded_feature_names, index=dfClean.index)
dfClean = pd.concat([dfClean.drop(cat_features, axis=1), df_encoded], axis=1)

# Division des données en ensembles d'entraînement et de test
X = dfClean.drop(target, axis=1)
y = dfClean[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraînement du modèle
model = LinearRegression()
model.fit(X_train, y_train)

# Prédiction sur l'ensemble de test
y_pred = model.predict(X_test)

# Calcul de l'erreur quadratique moyenne
mse = mean_squared_error(y_test, y_pred)
print("MSE:", mse)

# Print root mean squared error
rmse = np.sqrt(mse)
print("RMSE:", rmse)

print(len(dfClean))
