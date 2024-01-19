import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load the data
data = pd.read_stata("stanford data set/HCMST 2017 fresh sample for public sharing draft v1.1.dta")

# Map categorical values to numerical values
relationship_quality_map = {"very poor": 1, "poor": 2, "fair": 3, "good": 4, "excellent": 5}
breakup_nonmar_map = {"I wanted to break up more": 0, "[Partner Name] wanted to break up more": 0, "We both equally wanted to break up": 0, "Refused": 1}
sex_frequency_map = {"Once a month or less": 1, "2 to 3 times a month": 2, "Once or twice a week": 3, "3 to 6 times a week": 4, "Once a day or more": 5, "Refused": 6}

# Apply the mapping to the columns
data["w6_relationship_quality_num"] = data["w6_relationship_quality"].map(relationship_quality_map)
data["w6_breakup_nonmar_num"] = data["w6_breakup_nonmar"].map(breakup_nonmar_map)
data["w6_sex_frequency_num"] = data["w6_sex_frequency"].map(sex_frequency_map)

# Impute missing values
imputer_median = SimpleImputer(strategy="median")
imputer_most_frequent = SimpleImputer(strategy="most_frequent")

data["w6_relationship_quality_num"] = imputer_median.fit_transform(data[["w6_relationship_quality_num"]])
data["relate_duration_at_w6_years"] = imputer_median.fit_transform(data[["relate_duration_at_w6_years"]])
data["w6_sex_frequency_num"] = imputer_median.fit_transform(data[["w6_sex_frequency_num"]])
data["w6_breakup_nonmar_num"] = imputer_most_frequent.fit_transform(data[["w6_breakup_nonmar_num"]])

# Create the compatibility score
data["compatibility_score"] = 0.5 * data["w6_relationship_quality_num"] + 0.2 * data["relate_duration_at_w6_years"] + 0.2 * data["w6_breakup_nonmar_num"] + 0.1 * data["w6_sex_frequency_num"]
data.loc[data["w6_breakup_nonmar_num"] == 0, "compatibility_score"] *= 0.5
scaler = MinMaxScaler()
data["compatibility_score"] = scaler.fit_transform(data[["compatibility_score"]])

# Prepare the features for the model
excluded_columns = ["w6_relationship_quality", "w6_relationship_quality_num", "relate_duration_at_w6_years", "w6_breakup_nonmar", "w6_breakup_nonmar_num", "w6_sex_frequency", "w6_sex_frequency_num", "compatibility_score"]
features = data.drop(columns=excluded_columns)
features_categorical = features.select_dtypes(include=["object", "category"]).astype(str)
encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
features_encoded = pd.DataFrame(encoder.fit_transform(features_categorical))
features_encoded.columns = encoder.get_feature_names(features_categorical.columns)
features.drop(columns=features_categorical.columns, inplace=True)
features = pd.concat([features, features_encoded], axis=1)

# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(features, data["compatibility_score"], test_size=0.2, random_state=42)

# Select the best features
selector = SelectKBest(score_func=f_regression, k=50)
X_train_selected = selector.fit_transform(X_train, y_train)

# Get the names of the selected features
selected_features = X_train.columns[selector.get_support()]

# Train the model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train[selected_features], y_train)

# Evaluate the model
y_pred_train = rf_model.predict(X_train[selected_features])
y_pred_test = rf_model.predict(X_test[selected_features])
mse_train = mean_squared_error(y_train, y_pred_train)
mse_test = mean_squared_error(y_test, y_pred_test)
