import pandas as pd

# Load the data
df = pd.read_stata("stanford data set/HCMST 2017 fresh sample for public sharing draft v1.1.dta")

# Map relationship satisfaction categories to numerical values
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

# Calculate compatibility score (average of normalized satisfaction and duration)
df['compatibility_score'] = (df['relationship_satisfaction_norm'] + df['relationship_duration_norm']) / 2

# Display the DataFrame with new columns
print(df[['Q34', 'relationship_satisfaction_norm', 'Q21B_Year', 'relationship_duration', 'relationship_duration_norm', 'compatibility_score']].head())