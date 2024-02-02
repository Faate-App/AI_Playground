import pandas as pd
import numpy as np
from deepface import DeepFace
import re
import requests
import time
from geopy.geocoders import Nominatim
import geopandas as gpd
from shapely.geometry import Point
import random
from faker import Faker
from datetime import datetime
from textblob import TextBlob
import pgeocode
import os
from transformers import pipeline

# %%
def get_iris_code_from_coordinates(latitude, longitude):
    # URL de l'API avec les coordonnées passées en paramètres
    api_url = f"https://pyris.datajazz.io/api/coords?lat={latitude}&lon={longitude}"
    try:
        # Effectuer la requête GET vers l'API
        response = requests.get(api_url)
        
        # Vérifier si la requête a réussi (code de statut 200)
        if response.status_code == 200:
            # Extraire les données JSON de la réponse
            data = response.json()
            
            # Extraire le code IRIS
            iris_code = data.get("complete_code")
            
            if iris_code:
                return iris_code
            else:
                return "Code IRIS non trouvé dans la réponse API."
        else:
            return f"Erreur de requête : {response.status_code}"
    except Exception as e:
        return f"Erreur lors de la requête API : {str(e)}"

# %%
def estimer_poids(taille_cm, imc_cible=22):
    """
    Estime le poids en kilogrammes basé sur la taille en centimètres et un IMC cible.

    :param taille_cm: Taille en centimètres
    :param imc_cible: Indice de masse corporelle cible (par défaut 22)
    :return: Poids estimé en kilogrammes
    """
    taille_m = taille_cm / 100  # Convertir la taille en mètres
    poids_estime = imc_cible * (taille_m ** 2)
    return poids_estime

# %%
age_data = {
    "25-29 ans": {"Femme": 1897867, "Homme": 1887312},
    "30-34 ans": {"Femme": 2045742, "Homme": 1980053},
    "35-39 ans": {"Femme": 2180469, "Homme": 2065743},
    "40-44 ans": {"Femme": 2220320, "Homme": 2112390},
    "45-49 ans": {"Femme": 2097799, "Homme": 2043531},
    "50-54 ans": {"Femme": 2297423, "Homme": 2241191},
    "55-59 ans": {"Femme": 2267692, "Homme": 2159384}
}

def extract_age_range(age_group):
    # Extrait les bornes de l'intervalle d'âge
    start_age, end_age = age_group.split('-')
    start_age = int(start_age)
    end_age = int(end_age.split(' ')[0])
    return start_age, end_age

def find_random_age_by_gender(gender):
    # Vérification de la validité du sexe
    if gender not in ["Femme", "Homme"]:
        return "Sexe non valide"

    # Calculer les poids pour chaque tranche d'âge
    weights = [data[gender] for data in age_data.values()]
    
    # Choisir une tranche d'âge aléatoire en fonction des poids
    chosen_age_group = random.choices(list(age_data.keys()), weights=weights, k=1)[0]
    
    # Extraire l'intervalle d'âge et générer un âge aléatoire dans cet intervalle
    start_age, end_age = extract_age_range(chosen_age_group)
    random_age = random.randint(start_age, end_age)

    return random_age, chosen_age_group

# %%
basecouple = pd.read_excel("data//base-ic-couples-familles-menages-2020.xlsx", sheet_name="IRIS")
basecouple = basecouple.drop(index=0)

# %%
datacouple = pd.read_excel("data//ip1774.xls", sheet_name="Figure 1")
datacouple = datacouple.dropna()

# %%
personnevivantseul = pd.read_excel("data//demo-couple-pers-seul-log-age.xlsx")

# %%
def choose_iris_weighted(iris_data):
    iris_list = iris_data['IRIS'].tolist()
    weights = iris_data['Pop Ménages en 2020 (compl)'].tolist()
    chosen_iris = random.choices(iris_list, weights=weights, k=1)
    return chosen_iris[0]

# Sélectionner un IRIS de manière aléatoire pondérée


chosen_iris = choose_iris_weighted(basecouple)
print(chosen_iris)

# %%
def age_to_interval(age_str):
    try:
        # Utiliser des expressions régulières pour extraire les nombres
        numbers = re.findall(r'\d+', age_str)
        if age_str == "Ensemble" or "en millions" in age_str:
            return None
        if "plus" in age_str:
            # Gérer les cas comme "65 ans ou plus"
            return (int(numbers[0]), float(150))
        elif len(numbers) >= 2:
            # Gérer les cas avec deux nombres, comme "15 à 19 ans"
            return (int(numbers[0]), int(numbers[1]))
        else:
            raise ValueError("Format d'âge non reconnu")
    except ValueError as e:
        print(f"Erreur avec l'entrée : '{age_str}' - {e}")
        raise
if not isinstance(personnevivantseul['Age'].iloc[0], pd.Interval):
    personnevivantseul['Age Range'] = personnevivantseul['Age'].apply(age_to_interval)

if not isinstance(datacouple['Age'].iloc[0], pd.Interval):
    datacouple['Age Range'] = datacouple['Age'].apply(age_to_interval)


# %%

def find_age_interval(age, df):
    for interval in df['Age Range']:
        if interval != None and interval[0] <= age and age <= interval[1]:
            return interval
    return "Âge non trouvé dans les intervalles"

# %%
def get_proportion(age, gender, df):
    # Trouver l'intervalle d'âge
    interval = find_age_interval(age, df)
    if interval == "Âge non trouvé dans les intervalles":
        return interval
    
    # Sélectionner la ligne correspondante à l'intervalle d'âge
    row = df[df['Age Range'] == interval]
    
    # Sélectionner la colonne en fonction du sexe
    if gender.lower() == 'femme':
        proportion = row['Femmes'].values[0]
    elif gender.lower() == 'homme':
        proportion = row['Hommes'].values[0]
    else:
        return "Sexe non reconnu"
    
    # Construire la phrase récapitulative
    return proportion

# %%
def statcouple(IRIS):
    populationtotal = basecouple[basecouple["IRIS"] == IRIS]["Pop Ménages en 2020 (compl)"]
    populationtotal = populationtotal[populationtotal.index[0]]
    proba_statuts = {}
    unionlibre = basecouple[basecouple["IRIS"] == IRIS]["Pop 15 ans ou plus en concubinage ou union libre en 2020 (princ)"]
    unionlibre = unionlibre[unionlibre.index[0]]/populationtotal
    proba_statuts["unionlibre"] = unionlibre
    pacsée = basecouple[basecouple["IRIS"] == IRIS]["Pop 15 ans ou plus pacsée en 2020 (princ)"]
    pacsée = pacsée[pacsée.index[0]]/populationtotal
    proba_statuts["pacsée"] = pacsée
    marier = basecouple[basecouple["IRIS"] == IRIS]["Pop 15 ans ou plus mariée en 2020 (princ)"]
    marier = marier[marier.index[0]]/populationtotal
    proba_statuts["marier"] = marier
    popu_veuf = basecouple[basecouple["IRIS"] == IRIS]["Pop 15 ans ou plus veuves ou veufs en 2020 (princ)"]
    popu_veuf = popu_veuf[popu_veuf.index[0]]/populationtotal
    proba_statuts["veuf"] = popu_veuf
    popu_divorce = basecouple[basecouple["IRIS"] == IRIS]["Pop 15 ans ou plus divorcée en 2020 (princ)"]
    popu_divorce = popu_divorce[popu_divorce.index[0]]/populationtotal
    proba_statuts["divorce"] = popu_divorce
    popu_celib = basecouple[basecouple["IRIS"] == IRIS]["Pop 15 ans ou plus célibataire en 2020 (princ)"]
    popu_celib = popu_celib[popu_celib.index[0]]/populationtotal
    proba_statuts["celib"] = popu_celib
    inconnu = 1 - unionlibre - pacsée - marier - popu_veuf - popu_divorce - popu_celib
    proba_statuts["inconnu"] = inconnu
    return proba_statuts

def random_status(proba_statuts):
    statuts = list(proba_statuts.keys())
    probabilités = list(proba_statuts.values())
    statut_choisi = random.choices(statuts, weights=probabilités, k=1)[0]
    return statut_choisi

def random_with_proba(probability):
    """Renvoie True avec la probabilité indiquée, sinon False."""
    return random.random() < (round(float(probability)) / 100)

def education_famille(data_for_gender):
    output = {}
    # Déterminons si la personne fait partie d'un couple de même sexe
    
    output["couple_meme_sexe"] = random_with_proba(data_for_gender[data_for_gender.index[0]])

    # Déterminons le niveau d'éducation de la personne
    # On suppose que l'éducation est indépendante du fait d'être en couple de même sexe ou non
    output["education_bac_plus_3"] = False
    output["education_bac"] = random_with_proba(data_for_gender[data_for_gender.index[2]]) if output["couple_meme_sexe"] \
                    else random_with_proba(data_for_gender[data_for_gender.index[3]])
    if output["education_bac"]:
        output["education_bac_plus_3"] = random_with_proba(data_for_gender[data_for_gender.index[4]]) if output["couple_meme_sexe"] \
                            else random_with_proba(data_for_gender[data_for_gender.index[5]])

    # On suppose ici que le fait d'avoir un enfant est indépendant du niveau d'éducation et du type de couple
    output["enfant"] = random_with_proba(data_for_gender[data_for_gender.index[-1]]) if output["couple_meme_sexe"] \
            else False  # Nous n'avons pas de données pour les couples de sexe différent avec enfants
    return output

def get_data_couple_age(age, gender, df):
    # Adjust the filter to account for both singular and plural forms of "Homme/Hommes"
    gender_str = 'Homme' if gender.lower() == 'homme' else 'Femme'
    
    # Find the age interval
    interval = find_age_interval(age, df)
    if interval == "Âge non trouvé dans les intervalles":
        return interval
    
    # Select the row corresponding to the age interval
    row = df[df['Age Range'] == interval]
    
    # Filter the columns based on gender, allowing for both 'Homme' and 'Hommes'
    gender_columns = [col for col in df.columns if gender_str in col]
    
    # Select only the columns that match the gender
    gender_data = row[gender_columns].iloc[0]  # Use iloc[0] to select the first (and only) row as a Series
    
    return gender_data  # Convert to dictionary for easier readability


# %% [markdown]
# Analyse Social/Bar
# 

# %%
datasetbar = pd.read_excel("data//fichier_bars-cafes-discotheques.xlsx")
datasetbar

# %%
datasetbar = datasetbar.replace("[ND]", np.nan)
datasetbar = datasetbar.dropna(subset=['codePostalEtablissement'])


# %%
datasetbar[datasetbar['codePostalEtablissement']=="[ND]"]

# %%
file_path = "data//StockEtablissement_utf8.csv"
data = pd.read_csv(file_path,nrows=1000)
data

# %%
# Analyse de sentiment simple avec TextBlob
def analyze_sentiment_textblob(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment
    return {
        "polarity": sentiment.polarity,
        "subjectivity": sentiment.subjectivity
    }

# Analyse de sentiment avancée avec Hugging Face Transformers
def analyze_sentiment_transformers(text):
    classifier = pipeline('sentiment-analysis')
    return classifier(text)

# Analyse de thèmes/domaines avec Hugging Face Transformers
def analyze_domain(text):
    # Utiliser un modèle spécifique pour la classification de thème
    classifier = pipeline('zero-shot-classification', model='facebook/bart-large-mnli')

    # Définissez une liste étendue des thèmes/domaines potentiels
    candidate_labels = [
        "santé", "économie", "sport", "technologie", "politique", "éducation", "environnement",
        "art et culture", "science", "société", "histoire", "loisirs", "psychologie",
        "droit et législation", "mode et beauté", "religion et spiritualité", "actualités et médias",
        "littérature et écriture", "entreprise et management", "voyage et tourisme"
    ]

    # Appliquer le modèle de classification
    return classifier(text, candidate_labels)

def trouver_lieux(lat, lng, type_lieu, api_key="AIzaSyAuz6HCelEoLHELYGFA6HeNC9CAMB3XKmE"):
    """
    Trouve des lieux d'un type spécifique dans une zone donnée en utilisant l'API Google Places.

    Args:
    - lat (float): Latitude du centre de la zone de recherche.
    - lng (float): Longitude du centre de la zone de recherche.
    - type_lieu (str): Type de lieu à rechercher (par exemple, "bar" ou "night_club").
    - api_key (str): Clé API pour Google Places.

    Returns:
    - list: Liste des lieux trouvés.
    """

    url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    params = {
        'location': f'{lat},{lng}',
        'radius': 2000,  # Rayon de recherche en mètres
        'type': type_lieu,
        'key': api_key
    }
    reponse = requests.get(url, params=params)
    resultats = reponse.json()

    if resultats.get("status") == "OK":
        lieux = resultats.get("results")
        noms_lieux = [lieu.get("name") for lieu in lieux]
        return noms_lieux
    else:
        return []

# %%
with open('Biotinder.txt', encoding='utf-8') as f:
    text = [line.strip() for line in f.readlines()]
text

# %%
# Configuration de Faker pour la génération de données
fake = Faker('fr_FR')


if not isinstance(personnevivantseul['Age'].iloc[0], pd.Interval):
            personnevivantseul['Age Range'] = personnevivantseul['Age'].apply(age_to_interval)
else:
            personnevivantseul['Age Range'] = personnevivantseul['Age']
personnevivantseul = personnevivantseul.dropna()

# %%
nomi = pgeocode.Nominatim('fr')
# Nombre de défunt à générer
num_defunts = 50

# Création d'une liste pour stocker les données
defunts_data = []
#JOB DE LA PERSONNE
#LOCALISATION DES BARS ET BOITE A PROXIMITER DE CHEZ ELLE


# Définition de l'URL de base pour les chemins fictifs des photos et vidéos
base_path = ""
class Person:
    def __init__(self,id, nom, prenom, age, sexe, taille, photo,long,lat,bio):
        self.id = id
        self.nom = nom
        self.prenom = prenom
        self.bio = bio
        self.age = age
        self.sexe = sexe
        
        #sentiment_textblob = analyze_sentiment_textblob(bio)
        #domain_analysis = analyze_domain(bio)
        #domaine_analysis_bio = domain_analysis['labels'][:3]
        #subjectivity_bio = sentiment_textblob['subjectivity']
        #self.subjectivity = subjectivity_bio
        #self.domaine_analysis_bio = domaine_analysis_bio
        self.taille = taille
        self.photo = photo
        emotions = {
            'angry': 0.10,   # 10% de chances d'être en colère
            'disgust': 0.05,  # 5% de chances d'être dégoûté
            'fear': 0.10,    # 10% de chances d'avoir peur
            'happy': 0.40,   # 40% de chances d'être heureux
            'sad': 0.20,     # 20% de chances d'être triste
            'surprise': 0.10, # 10% de chances d'être surpris
            'neutral': 0.05   # 5% de chances d'être neutre
        }
        
        
        pourcentage_seul = get_proportion(age, sexe, personnevivantseul)
        
        if not isinstance(datacouple['Age'].iloc[0], pd.Interval):
            datacouple['Age Range'] = datacouple['Age'].apply(age_to_interval)
        else:
            # Si "Age" est déjà une colonne d'intervalles, utilisez-la directement
            datacouple['Age Range'] = datacouple['Age']
        self.seul = random.random() < ((pourcentage_seul+20) / 100)
        self.emotion = random.choices(list(emotions.keys()), weights=emotions.values(), k=1)[0]
        races = ['asian', 'indian', 'black', 'white', 'middle eastern', 'latino hispanic']
        # Poids basés sur une estimation de la répartition démographique en France
        weights = [0.05, 0.02, 0.03, 0.85, 0.03, 0.02]  # Ces valeurs sont des exemples
        self.race = random.choices(races, weights=weights, k=1)[0]
        taille_m = taille / 100  # Convertir la taille en mètres
        self.poids = estimer_poids(taille)
        data_for_gender = get_data_couple_age(age,  sexe, datacouple)

        result = education_famille(data_for_gender)
        self.couple_meme_sexe = result["couple_meme_sexe"]
        self.enfant = result["enfant"]
        self.education_bac = result["education_bac"]
        self.education_bac_plus_3 = result["education_bac_plus_3"]
    
        self.IRIScode = choose_iris_weighted(basecouple)
        self.address = basecouple[basecouple["IRIS"] == self.IRIScode]["Libellé commune ou ARM"][basecouple[basecouple["IRIS"] == self.IRIScode]["Libellé commune ou ARM"].index[0]]
        if(long == 0 and lat==0):
            try:
                data = nomi.query_location(self.address,top_k=1)
                
                # Extraction des coordonnées de base
                lat_base = data['latitude'].values[0]  # Pour obtenir la première valeur de la Series latitude
                lng_base = data['longitude'].values[0]  # Pour obtenir la première valeur de la Series longitude
                # Ajout d'une petite variation aléatoire pour rendre les coordonnées uniques
                lat = lat_base + random.uniform(-0.0005, 0.0005)
                lng = lng_base + random.uniform(-0.0005, 0.0005)
                self.lat = round(lat, 7)
                self.long = round(lng, 7)
            except:
                self.lat = 0
                self.long = 0
        else:
            self.lat = lat
            self.long = long    
        proba_statuts = statcouple(self.IRIScode) 
        self.bars = trouver_lieux(self.lat, self.long, "bar")
        time.sleep(0.001)
        self.boites_de_nuit = trouver_lieux(self.lat, self.long, "night_club")
        self.statut_relationnel = random_status(proba_statuts)
    def __str__(self):
        return pd.DataFrame({'Id':[self.id],'Nom': [self.nom], 'Prénom': [self.prenom], 'Age': [self.age], 'Sexe': [self.sexe], 'Taille': [self.taille], 'Photo': [self.photo], 
                             'Emotion': [self.emotion], 'Race': [self.race], 'Poids': [self.poids],'Vivre seul' : [self.seul],
                             'IRIS:':[self.IRIScode],"statut relationnelle : ":[self.statut_relationnel], "couple_meme_sexe":[self.couple_meme_sexe],
                             "enfant":[self.enfant],"education_bac":[self.education_bac],"Bars a proximité : ":[self.bars],"Boite de nuit : ":[self.boites_de_nuit],"adress : ":[self.address]
                             ,"Bac + 3":[self.education_bac_plus_3],"bio : ":[self.bio],"longitude : ":[self.long],"Latitude : ":[self.lat]
                             }).to_string(index=False)
    def to_dict(self):
        return vars(self)

# Génération des données fictives pour les défunts
personne = []
compteur_erreur = 0
id = 100000
for _ in range(num_defunts):
    prenom = fake.first_name()
    nom = fake.last_name()
    nationalite = "Française"
    id = id + 1
    sexe = random.choice(["Homme", "Femme"])
    if sexe == "Homme":
        taille = np.random.normal(175, 7)
    else:
        taille = np.random.normal(162, 7)
    try:
        age, age_group = find_random_age_by_gender(sexe)
        age = int(round(age,0))
    except:
        age = fake.random_int(25, 55)
        
    
    photo = "" 
    
    latitude = 0
    longitude = 0
    personne.append(Person(id,nom, prenom, age, sexe, taille, "",longitude,latitude,str(text[_])))


defunts_data = [p.to_dict() for p in personne]

# Création du DataFrame
df_defunts = pd.DataFrame(defunts_data)

# Affichage du DataFrame
print(df_defunts)




# %%
print(compteur_erreur)
for i in personne[:4]:
    print(i)

# %% [markdown]
# Export data

# %%
# Exportation en Excel
df_defunts.to_excel("France_Data_gouv.xlsx", index=False)


