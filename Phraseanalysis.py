from textblob import TextBlob
from transformers import pipeline

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
    # Ici, 'bert-base-uncased' est un exemple, et il peut ne pas être le mieux adapté pour la classification de thème
    # Vous devrez trouver et utiliser un modèle pré-entraîné adapté à vos besoins
    classifier = pipeline('zero-shot-classification', model='facebook/bart-large-mnli')
    # Définissez les thèmes/domaines potentiels
    candidate_labels = ["santé", "économie", "sport", "technologie", "politique", "éducation", "environnement"]
    return classifier(text, candidate_labels)

# Exemple de texte
text = "Je suis très heureux aujourd'hui car j'ai appris beaucoup de nouvelles choses sur la technologie."

# Analyse de sentiment
sentiment_textblob = analyze_sentiment_textblob(text)
sentiment_transformers = analyze_sentiment_transformers(text)

# Analyse de domaine
domain_analysis = analyze_domain(text)

# Affichage des résultats
print("Analyse de Sentiment (TextBlob):", sentiment_textblob)
print("Analyse de Sentiment (Transformers):", sentiment_transformers)
print("Analyse de Domaine:", domain_analysis)
