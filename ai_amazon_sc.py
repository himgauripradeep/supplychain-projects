import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Sample dataset simulating Amazon's inventory and demand
data = {
    'Warehouse': ['Seattle', 'New York', 'Los Angeles', 'Chicago', 'Houston'],
    'Past Demand (Units)': [5000, 7000, 8000, 6500, 7200],
    'Current Inventory (Units)': [5500, 6900, 8200, 6400, 7000],
    'Lead Time (Days)': [2, 3, 1, 2, 4],
    'Reorder Point (Units)': [5200, 6800, 8100, 6300, 7100]
}

df = pd.DataFrame(data)

# Feature Selection
X = df[['Past Demand (Units)', 'Current Inventory (Units)', 'Lead Time (Days)']]
y = df['Reorder Point (Units)']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Model Evaluation
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Absolute Error: {mae}')
print(f'R2 Score: {r2}')

# Visualizing Feature Importance
importances = model.feature_importances_
features = X.columns
plt.figure(figsize=(8, 5))
plt.bar(features, importances, color='skyblue')
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.title("Feature Importance in Inventory Prediction")
plt.show()

# Display dataset
df
