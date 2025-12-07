"""
Train the best model and save it using pickle for deployment.
This script trains multiple models, selects the best one, and saves it along with the scaler.
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer

print("\n" + "="*70)
print("TRAINING AND SAVING BEST MODEL FOR DEPLOYMENT")
print("="*70)

# Load and preprocess data
print("\n[1/5] Loading and preprocessing data...")
dataset = pd.read_csv('supervised/world_data.csv')

# Remove sparsely populated features
dataset = dataset.drop(['murder', 'urbanpopulation', 'unemployment'], axis=1)

# Impute with median (more robust than mean)
numeric_cols = dataset.select_dtypes(include=[np.number]).columns
imputer = SimpleImputer(strategy='median')
dataset[numeric_cols] = imputer.fit_transform(dataset[numeric_cols])

# Prepare features and target
y = dataset['lifeexp']
X = dataset[['happiness', 'income', 'sanitation', 'water', 'literacy',
             'inequality', 'energy', 'childmortality', 'fertility',
             'hiv', 'foodsupply', 'population']]

# Scale features
scaler = MinMaxScaler(feature_range=(0, 1))
rescaledX = scaler.fit_transform(X)
X_scaled = pd.DataFrame(rescaledX, index=X.index, columns=X.columns)

print(f"   ✓ Dataset shape: {dataset.shape}")
print(f"   ✓ Features: {len(X.columns)}")

# Split data
test_size = 0.33
seed = 1
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=seed)

# Define all models
print("\n[2/5] Training all models...")
models = [
    ('Linear Regression', LinearRegression()),
    ('KNN', KNeighborsRegressor(n_neighbors=7, weights='distance')),
    ('SVR', SVR(kernel='rbf', C=10, gamma='scale', epsilon=0.1)),
    ('Decision Tree', DecisionTreeRegressor(max_depth=10, min_samples_split=5, random_state=seed)),
    ('Random Forest', RandomForestRegressor(n_estimators=100, max_depth=15, min_samples_split=5, random_state=seed)),
    ('Gradient Boosting', GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=seed))
]

# Train and evaluate all models
results = []
for name, model in models:
    model.fit(X_train, Y_train)
    predictions = model.predict(X_test)
    mae = mean_absolute_error(Y_test, predictions)
    r2 = r2_score(Y_test, predictions)
    results.append((name, model, mae, r2))
    print(f"   ✓ {name}: MAE={mae:.4f}, R²={r2:.4f}")

# Select best model (lowest MAE)
print("\n[3/5] Selecting best model...")
best_result = min(results, key=lambda x: x[2])
best_name = best_result[0]
best_model = best_result[1]
best_mae = best_result[2]
best_r2 = best_result[3]

print(f"   ✓ Best Model: {best_name}")
print(f"   ✓ MAE: {best_mae:.4f} years")
print(f"   ✓ R² Score: {best_r2:.4f}")

# Retrain best model on full dataset for better performance
print("\n[4/5] Retraining best model on full dataset...")
best_model.fit(X_scaled, y)
print(f"   ✓ Model retrained on {len(X_scaled)} samples")

# Save model and scaler
print("\n[5/5] Saving model and preprocessing objects...")

# Create a dictionary with all necessary objects
model_package = {
    'model': best_model,
    'scaler': scaler,
    'imputer': imputer,
    'feature_names': X.columns.tolist(),
    'model_name': best_name,
    'performance': {
        'mae': best_mae,
        'r2': best_r2
    },
    'feature_stats': {
        'min': X.min().to_dict(),
        'max': X.max().to_dict(),
        'mean': X.mean().to_dict()
    }
}

# Save using pickle
with open('best_model.pkl', 'wb') as f:
    pickle.dump(model_package, f)

print(f"   ✓ Model saved as 'best_model.pkl'")

# Verify the saved model works
print("\n" + "="*70)
print("VERIFICATION - Testing saved model...")
print("="*70)

# Load the saved model
with open('best_model.pkl', 'rb') as f:
    loaded_package = pickle.load(f)

# Test prediction with a sample
sample_data = X.iloc[0:1].values
scaled_sample = loaded_package['scaler'].transform(sample_data)
prediction = loaded_package['model'].predict(scaled_sample)[0]
actual = y.iloc[0]

print(f"\nTest Prediction:")
print(f"   Input: {dataset.iloc[0]['country']}")
print(f"   Predicted Life Expectancy: {prediction:.2f} years")
print(f"   Actual Life Expectancy: {actual:.2f} years")
print(f"   Error: {abs(prediction - actual):.2f} years")

print("\n" + "="*70)
print("✓ MODEL SUCCESSFULLY SAVED AND VERIFIED!")
print("="*70)
print(f"\nModel Package Contents:")
print(f"   • Model Type: {loaded_package['model_name']}")
print(f"   • Number of Features: {len(loaded_package['feature_names'])}")
print(f"   • Performance (MAE): {loaded_package['performance']['mae']:.4f} years")
print(f"   • Performance (R²): {loaded_package['performance']['r2']:.4f}")
print(f"\nYou can now use 'best_model.pkl' in your Streamlit app!")
print("="*70 + "\n")
