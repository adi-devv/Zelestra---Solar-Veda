import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

# Load data
train_data = pd.read_csv('dataset/train.csv')
test_data = pd.read_csv('dataset/test.csv')

# Separate features and target
X = train_data.drop(columns=['id', 'efficiency'])
y = train_data['efficiency']
test_ids = test_data['id']
X_test = test_data.drop(columns=['id'])

# Identify numerical and categorical columns
numeric_features = ['temperature', 'irradiance', 'humidity', 'panel_age',
                   'maintenance_count', 'soiling_ratio', 'voltage', 'current',
                   'module_temperature', 'cloud_coverage', 'wind_speed', 'pressure']

categorical_features = ['string_id', 'error_code', 'installation_type']

# Preprocessing pipeline
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# Split training data for validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)