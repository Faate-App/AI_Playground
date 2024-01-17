import pandas as pd
import numpy as np
from deepface import DeepFace
import re
import requests
from geopy.geocoders import Nominatim
import geopandas as gpd
from shapely.geometry import Point

def age_to_interval(age_str):
    try:
        # Utiliser des expressions régulières pour extraire les nombres
        numbers = re.findall(r'\d+', age_str)
        if age_str == "Ensemble" or "en millions" in age_str:
            return None
        if "plus" in age_str:
            # Gérer les cas comme "65 ans ou plus"
            return pd.Interval(int(numbers[0]), float('inf'), closed='left')
        elif len(numbers) >= 2:
            # Gérer les cas avec deux nombres, comme "15 à 19 ans"
            return pd.Interval(int(numbers[0]), int(numbers[1]), closed='left')
        else:
            raise ValueError("Format d'âge non reconnu")
    except ValueError as e:
        print(f"Erreur avec l'entrée : '{age_str}' - {e}")
        raise


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

def find_age_interval(age, df):
    for interval in df['Age Range']:
        if interval.left <= age < interval.right:
            return interval
    return "Âge non trouvé dans les intervalles"

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

class Person:
    def __init__(self, nom, prenom, age, sexe, taille, photo,long,lat):
        self.nom = nom
        self.prenom = prenom
        self.age = age
        self.sexe = sexe
        self.taille = taille
        self.photo = photo
        self.emotion = ""
        self.race = ""  
        taille_m = taille / 100  # Convertir la taille en mètres
        self.poids = estimer_poids(taille)
        self.vivreseul = 0
        self.address = ""
        self.long = long
        self.lat = lat
        self.IRISdata = []
        
    def __str__(self):
        return pd.DataFrame({'Nom': [self.nom], 'Prénom': [self.prenom], 'Age': [self.age], 'Sexe': [self.sexe], 'Taille': [self.taille], 'Photo': [self.photo], 'Emotion': [self.emotion], 'Race': [self.race], 'Poids': [self.poids],'Vivre seul' : [self.vivreseul]}).to_string(index=False)
    


#IMPORT DATASET
datacouple = pd.read_excel("data//ip1774.xls", sheet_name="Figure 1")
datacouple = datacouple.dropna()
personnevivantseul = pd.read_excel("data//demo-couple-pers-seul-log-age.xlsx")
basecouple = pd.read_excel("data//base-ic-couples-familles-menages-2020.xlsx", sheet_name="IRIS")  

Taux_de_Chômage = 7,4
habitants = 67162154
salaries = 19770837
tranche_cible_pop = 20787631 #25-49 ans
tranche_cible_emploi = 22862357 #25-54 ans

#INPUT DATA
Nom = "Seidlitz"
Prenom = "Eloi"
Age = 22
Sexe = "Homme"
Taille = 193 #En cm
photo = "data//DSC_0434.JPG"
latitude = 48.86151123046875
longitude = 2.1342475414276123
Person_to_study = Person(Nom, Prenom, Age, Sexe, Taille, photo,longitude,latitude)

if not isinstance(personnevivantseul['Age'].iloc[0], pd.Interval):
    personnevivantseul['Age Range'] = personnevivantseul['Age'].apply(age_to_interval)
else:
    personnevivantseul['Age Range'] = personnevivantseul['Age']
pourcentage_seul = get_proportion(Person_to_study.age, Person_to_study.sexe, personnevivantseul)
Person_to_study.vivreseul = pourcentage_seul

def FaceAnalysis(Person_to_study):
    """Function who detect the face of the person and analyse it

    Args:
        Person_to_study (Person): Information concerning the person

    Returns:
        _type_: _description_
    """
    try:
        analysis = DeepFace.analyze(Person_to_study.photo, actions=['age', 'gender', 'race', 'emotion'])
        analysis = analysis[0]

        age_estime = analysis.get("age", "Non disponible")
        
        Sexe_dict = analysis.get("gender", "Non disponible")
        if Sexe_dict['Man'] > Sexe_dict['Woman']:
            Sexe_estime = "Homme"
        else:
            Sexe_estime = "Femme"
        if Sexe != Sexe_estime:
            print("Le sexe estimé est différent du sexe réel.")
        if np.abs(Age - age_estime) > 5:
            print("L'âge estimé est très différent de l'âge réel.")
        Person_to_study.race = analysis.get("dominant_race", "Non disponible")
        Person_to_study.emotion = analysis.get("dominant_emotion", "Non disponible")
        return Person_to_study
    except Exception as e:
        print("Une erreur s'est produite lors de l'analyse de l'image :", e)


def get_address_from_coordinates(latitude, longitude):
    # Initialize the Nominatim geocoder
    geolocator = Nominatim(user_agent="geoapiExercises")
    
    # Combine latitude and longitude into a single string
    location = geolocator.reverse(f"{latitude}, {longitude}")
    
    return location.raw["address"]

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

def pop_age_seul(age,data):
    column_names = [
        "Pop 15-24 ans vivant seule en 2020 (princ)",
        "Pop 25-54 ans vivant seule en 2020 (princ)",
        "Pop 55-79 ans vivant seule en 2020 (princ)",
        "Pop 80 ans ou plus vivant seule en 2020 (princ)"
    ]
    if age >= 15 and age < 25:
        return data[column_names[0]]
    elif age >= 25 and age < 55:
        return data[column_names[1]]
    elif age >= 55 and age < 80:
         return data[column_names[2]]
    elif age >= 80:
        return data[column_names[3]]




def detect_gender_by_age(df):
    # Dictionnaire pour mapper les tranches d'âge aux colonnes correspondantes
    age_columns_mapping = {
        "Homme": ["Hommes"],
        "Femme": ["Femmes"]
    }

    # Initialisez des listes pour stocker les colonnes correspondantes à l'âge spécifié
    homme_columns = []
    femme_columns = []

    # Parcourez toutes les colonnes du DataFrame
    for column in df.columns:
        for gender, age_keywords in age_columns_mapping.items():
            # Vérifiez si la colonne contient des mots-clés d'âge correspondant à l'âge spécifié
            if any(keyword in column for keyword in age_keywords):
                # Ajoutez la colonne correspondante à la liste appropriée (homme ou femme)
                if gender == "Homme":
                    homme_columns.append(column)
                elif gender == "Femme":
                    femme_columns.append(column)

    # Retournez les listes de colonnes correspondantes à l'âge spécifié pour homme et femme
    return homme_columns, femme_columns



def get_person_info(iris, gender, age, df):
    # Determine the gender-specific columns we're interested in
    if gender.lower() == 'homme':
        gender_cols = [col for col in df.columns if 'Hommes' in col]
    elif gender.lower() == 'femme':
        gender_cols = [col for col in df.columns if 'Femmes' in col]
    else:
        raise ValueError("Gender must be either 'homme' or 'femme'")
    
    # Determine the age range column
    if age < 25:
        age_col = 'P20_POP1524'
    elif 25 <= age < 55:
        age_col = 'P20_POP2554'
    elif 55 <= age < 80:
        age_col = 'P20_POP5579'
    else:
        age_col = 'P20_POP80P'
    
    # Filter the DataFrame for the given IRIS code
    person_df = df[df['IRIS'] == iris]
    
    # Extract the relevant information
    info = person_df[gender_cols + [age_col]].iloc[0].to_dict()
    return info



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








Person_to_study = FaceAnalysis(Person_to_study)

if not isinstance(datacouple['Age'].iloc[0], pd.Interval):
    datacouple['Age Range'] = datacouple['Age'].apply(age_to_interval)
else:
    # Si "Age" est déjà une colonne d'intervalles, utilisez-la directement
    datacouple['Age Range'] = datacouple['Age']
print(f"Pour les {Person_to_study.sexe}s âgés de {Person_to_study.age} ans, la proportion vivant seuls est de {pourcentage_seul}%.")
data_for_gender = get_data_couple_age(Person_to_study.age,  Person_to_study.sexe, datacouple)
print(data_for_gender)

iris_code = get_iris_code_from_coordinates(latitude, longitude)
print(pop_age_seul(Person_to_study.age,basecouple[basecouple["IRIS"] == iris_code]))
populationtotal = basecouple[basecouple["IRIS"] == iris_code]["Pop Ménages en 2020 (compl)"]
print(populationtotal)
popu_veuf = basecouple[basecouple["IRIS"] == iris_code]["Pop 15 ans ou plus veuves ou veufs en 2020 (princ)"]
print(popu_veuf)
popu_divorce = basecouple[basecouple["IRIS"] == iris_code]["Pop 15 ans ou plus divorcée en 2020 (princ)"]
print(popu_divorce)
popu_celib = basecouple[basecouple["IRIS"] == iris_code]["Pop 15 ans ou plus célibataire en 2020 (princ)"]
print(popu_celib)

hommes_correspondants, femmes_correspondantes = detect_gender_by_age(basecouple[basecouple["IRIS"] == iris_code])

for i in hommes_correspondants:
    print(i, basecouple[basecouple["IRIS"] == iris_code][i])
    break

for i in femmes_correspondantes:
    print(i, basecouple[basecouple["IRIS"] == iris_code][i])
    break

print(Person_to_study)



