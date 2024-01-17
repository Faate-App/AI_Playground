import json
import pandas as pd
from googletrans import Translator, LANGUAGES

# Function to translate text
def translate_text(text, dest_language='fr'):
    translator = Translator()
    translation = translator.translate(text, dest=dest_language)
    return translation.text

# Initialize a list to store all JSON objects
all_data = []

# Read JSON data
with open('data.jsonl', 'r', encoding='utf-8') as file:
    for line in file:
        try:
            data = json.loads(line)
            all_data.append(data)  # Append each JSON object to the list
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON in line: {line}")
            print(str(e))

# Translate content
translated_data = []
for item in all_data:  # Process each JSON object in the list
    translated_item = {}
    for key, value in item.items():
        # Translate and process each item
        translated_item[key] = translate_text(value) if isinstance(value, str) else value
    translated_data.append(translated_item)

# Write translated data to a new JSON file
with open('translated_file.json', 'w', encoding='utf-8') as file:
    json.dump(translated_data, file, ensure_ascii=False, indent=4)

# Convert to DataFrame for Excel export
df = pd.DataFrame(translated_data)  # Create a DataFrame from the list of dictionaries

# Write DataFrame to Excel
df.to_excel('translated_data.xlsx', index=False)
