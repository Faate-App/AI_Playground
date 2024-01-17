import requests
from urllib.parse import urlencode

# Paramètres de l'application LinkedIn
client_id = 'dddd'
client_secret = 'dddddd'
redirect_uri = 'https://www.faate.app/'
authorization_base_url = 'https://www.linkedin.com/oauth/v2/authorization'
token_url = 'https://www.linkedin.com/oauth/v2/accessToken'

# Etape 1: Obtenir l'URL d'autorisation
params = {
    'response_type': 'code',
    'client_id': client_id,
    'redirect_uri': redirect_uri,
    'scope': 'r_liteprofile r_emailaddress',  # Modifiez les autorisations au besoin
}
authorization_url = f"{authorization_base_url}?{urlencode(params)}"
print("Veuillez visiter cette URL pour autoriser:", authorization_url)

# Etape 2: Obtenez le code de redirection (ceci est normalement géré par votre serveur Web)
# Vous devez manuellement copier le code de 'code=XXXX' dans l'URL redirigée
code = input("Entrez le code que vous avez reçu: ")

# Etape 3: Echangez le code contre un token d'accès
access_token_response = requests.post(token_url, data={
    'grant_type': 'authorization_code',
    'code': code,
    'redirect_uri': redirect_uri,
    'client_id': client_id,
    'client_secret': client_secret,
})

# Vérifiez ici la réponse
print(access_token_response.json())
