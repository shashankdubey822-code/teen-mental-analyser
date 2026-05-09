import json

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from feature_engineering import FeatureEngineer, SOCIAL_INTERACTION_MAP

# 1. Load Data
df = pd.read_csv('Teen_Mental_Health_Dataset.csv')
print("--- Initial Data Info ---")
print(df.info())

print("\n--- Missing Values ---")
print(df.isnull().sum())

print("\n--- Target Class Distribution ---")
print(df['depression_label'].value_counts())

# 2. Handle missing values if any
# (assuming none for now, but good to fill just in case)
if df.isnull().sum().sum() > 0:
    df.fillna(df.median(numeric_only=True), inplace=True)
    df.fillna(df.mode().iloc[0], inplace=True)

# 3. Normalize and encode categorical variables

import joblib

# Clean raw strings to avoid inconsistent labels
df['gender'] = df['gender'].astype(str).str.lower().str.strip()
df['platform_usage'] = df['platform_usage'].astype(str).str.strip().str.title()
df['social_interaction_level'] = df['social_interaction_level'].astype(str).str.lower().str.strip()

# For models like XGBoost/RandomForest, label encoding or ordinal encoding is needed.
label_encoders = {}

gender_le = LabelEncoder()
df['gender'] = gender_le.fit_transform(df['gender'])
label_encoders['gender'] = gender_le
print(f"\nEncoded gender classes: {gender_le.classes_}")

platform_le = LabelEncoder()
df['platform_usage'] = platform_le.fit_transform(df['platform_usage'])
label_encoders['platform_usage'] = platform_le
print(f"\nEncoded platform_usage classes: {platform_le.classes_}")

social_map = SOCIAL_INTERACTION_MAP
df['social_interaction_level'] = df['social_interaction_level'].where(
    df['social_interaction_level'].isin(social_map), 'medium')
df['social_interaction_level'] = df['social_interaction_level'].map(social_map)
label_encoders['social_interaction_level'] = social_map
print(f"\nEncoded social_interaction_level mapping: {social_map}")

# Save the encoders for manual testing later
joblib.dump(label_encoders, 'label_encoders.pkl')
print("\nSaved Label Encoders to 'label_encoders.pkl'")

# 4. Split Features and Target (raw columns)
X = df.drop('depression_label', axis=1)
y = df['depression_label']

# 5. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)

# 6. Feature engineering (fit on train only to avoid leakage)
feature_engineer = FeatureEngineer()
feature_engineer.fit(X_train)
X_train = feature_engineer.transform(X_train)
X_test = feature_engineer.transform(X_test)

with open('feature_fill_values.json', 'w', encoding='utf-8') as f:
    json.dump(feature_engineer.fill_values, f, indent=2)
print("\nSaved feature fill values to 'feature_fill_values.json'")

with open('feature_columns.json', 'w', encoding='utf-8') as f:
    json.dump(X_train.columns.tolist(), f, indent=2)
print("\nSaved feature columns to 'feature_columns.json'")

print("\n--- Train set shape BEFORE SMOTE ---")
print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(y_train.value_counts())

# 7. Handle Class Imbalance using SMOTE
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print("\n--- Train set shape AFTER SMOTE ---")
print(f"X_train_smote: {X_train_smote.shape}, y_train_smote: {y_train_smote.shape}")
print(y_train_smote.value_counts())

# Save the preprocessed splits for later use
X_train_smote.to_csv('X_train.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
y_train_smote.to_csv('y_train.csv', index=False)
y_test.to_csv('y_test.csv', index=False)

print("\nData preprocessing complete. Preprocessed files saved.")
