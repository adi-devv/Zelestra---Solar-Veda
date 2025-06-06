# Import required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

# 1. Load and prepare data
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Separate features and target
X = train_data.drop(columns=['id', 'efficiency'])
y = train_data['efficiency']
test_ids = test_data['id']
X_test = test_data.drop(columns=['id'])

# 2. Define feature types
numeric_features = ['temperature', 'irradiance', 'humidity', 'panel_age', 
                  'maintenance_count', 'soiling_ratio', 'voltage', 'current',
                  'module_temperature', 'cloud_coverage', 'wind_speed', 'pressure']

categorical_features = ['string_id', 'error_code', 'installation_type']

# 3. Create preprocessing pipelines
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# 4. Split data for validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Create and train model with early stopping
# First preprocess the data manually for proper validation
preprocessor.fit(X_train)
X_train_processed = preprocessor.transform(X_train)
X_val_processed = preprocessor.transform(X_val)

model = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    early_stopping_rounds=50,
    eval_metric='rmse',
    random_state=42
)

model.fit(
    X_train_processed, y_train,
    eval_set=[(X_val_processed, y_val)],
    verbose=True
)

# 6. Evaluate model
val_preds = model.predict(X_val_processed)
score = 100 * (1 - np.sqrt(mean_squared_error(y_val, val_preds)))
print(f"Validation Score: {score:.2f}")

# 7. Create submission file
# Create a pipeline that combines preprocessing and model for final predictions
final_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', model)
])

# Retrain on full dataset
final_model.fit(X, y)

# Predict on test set
test_preds = final_model.predict(X_test)

# Create submission
submission = pd.DataFrame({'id': test_ids, 'efficiency': test_preds})
submission.to_csv('submission.csv', index=False)
print("Submission file created successfully!")