"""client_id = "26zRotW1NrOW6JnDvdKwZqZCz90a"
client_secret = "0Wh3Ev1OZH8t_hSkncUgeVGXUHMa"
token = "bb53514d-0c56-38e7-be3e-e566162a01a2"

import requests
from requests.auth import HTTPBasicAuth

url = "https://api.insee.fr/token"

response = requests.post(
    url,
    data={"grant_type": "client_credentials"},
    auth=HTTPBasicAuth(client_id, client_secret),
    verify=False  # Correspond à l'option '-k' dans curl
)

if response.status_code == 200:
    token_data = response.json()
    access_token = token_data.get("access_token")
    print("Token d'accès:", access_token)
else:
    print("Erreur lors de l'obtention du token:", response.text)

def get_siret_info(siret, access_token):
    url = f"https://api.insee.fr/entreprises/sirene3/V3.11/siret/{siret}"
    headers = {
        'Accept': 'application/json',
        'Authorization': f'Bearer {access_token}'
    }
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        return response.json()  # Retourne les données au format JSON
    else:
        return f"Erreur lors de la requête: {response.status_code}"

# Exemple d'utilisation
siret = "919133819"
access_token = "026f40fa-2b0c-3934-a3b3-0e6ee788310b"  # Remplacez par votre token d'accès valide
print(get_siret_info(siret, access_token))
"""

import pandas as pd


file_path = "data//StockEtablissement_utf8.csv"

# Colonnes à importer et leurs types de données (si connus)
columns = ["activitePrincipaleEtablissement", "siren", "siret", "activitePrincipaleRegistreMetiersEtablissement", "enseigne1Etablissement"]
dtype = {"siren": int, "siret": int,'activitePrincipaleRegistreMetiersEtablissement':str,'enseigne1Etablissement':str,'activitePrincipaleEtablissement':str}  # Exemple, ajustez selon vos données

# Taille de chaque bloc (nombre de lignes à lire à la fois)
chunksize = 10000
codes_to_include = ["56.30Z", "56.30A", "56.30B", "56.30C", "56.30D", "56.30E", "56.30F", "56.30G", "56.30H", "56.30I", "56.30J", "56.30K", "56.30L", "56.30M", "56.30N", "56.30O", "56.30P", "56.30Q", "56.30R", "56.30L", "56.30S", "56.30T", "56.30U", "56.30V", "56.30W", "56.30X", "56.30Y"]
# Préparation d'un DataFrame vide pour contenir les résultats
filtered_data = pd.DataFrame(columns=columns)

# Lecture du fichier par blocs
for chunk in pd.read_csv(file_path, usecols=columns, dtype=dtype, chunksize=chunksize):
    # Filtrage du bloc
    filtered_chunk = chunk[chunk['activitePrincipaleEtablissement'].isin(codes_to_include)].dropna()
    # Ajout au DataFrame final
    filtered_data = pd.concat([filtered_data, filtered_chunk], ignore_index=True)


# Affichage des données
print(filtered_data)
filtered_data.to_excel('data//filtered_data.xlsx', index=False)