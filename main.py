import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
# --- 1. Synthetic Data Generation ---
np.random.seed(42) # for reproducibility
num_deliveries = 500
data = {
    'Delivery_ID': range(1, num_deliveries + 1),
    'Distance_km': np.random.uniform(1, 20, num_deliveries),
    'Time_minutes': np.random.uniform(10, 120, num_deliveries),
    'Traffic_index': np.random.uniform(1, 5, num_deliveries), # 1=low, 5=high
    'Weather_index': np.random.uniform(1, 5, num_deliveries), # 1=good, 5=bad
    'Demand_index': np.random.uniform(1, 5, num_deliveries), # 1=low, 5=high
    'Delivery_Cost': np.random.uniform(5, 50, num_deliveries)
}
df = pd.DataFrame(data)
#Adding some realistic correlation
df['Delivery_Cost'] = 2 + 1.5 * df['Distance_km'] + 0.5 * df['Traffic_index'] + 0.8 * df['Demand_index'] + np.random.normal(0, 2, num_deliveries)
df['Time_minutes'] = 5 + 2 * df['Distance_km'] + 0.3 * df['Traffic_index'] + np.random.normal(0, 5, num_deliveries)
# --- 2. Data Cleaning and Feature Engineering (Minimal for this example) ---
# Ensure no negative values (though unlikely with this data generation)
df = df[df['Delivery_Cost'] >= 0]
df = df[df['Time_minutes'] >= 0]
# --- 3. Analysis: Predictive Modeling ---
# Simple linear regression to predict delivery cost based on features
X = df[['Distance_km', 'Traffic_index', 'Weather_index', 'Demand_index']]
y = df['Delivery_Cost']
model = LinearRegression()
model.fit(X, y)
r_sq = model.score(X, y)
print(f"Coefficient of determination (R^2): {r_sq}")
print(f"Intercept: {model.intercept_}")
print(f"Coefficients: {model.coef_}")
# --- 4. Visualization ---
plt.figure(figsize=(10, 6))
sns.regplot(x='Distance_km', y='Delivery_Cost', data=df)
plt.title('Delivery Cost vs. Distance')
plt.xlabel('Distance (km)')
plt.ylabel('Delivery Cost')
plt.savefig('delivery_cost_vs_distance.png')
print("Plot saved to delivery_cost_vs_distance.png")
plt.figure(figsize=(10,6))
sns.heatmap(df[['Distance_km', 'Time_minutes', 'Traffic_index', 'Weather_index', 'Demand_index', 'Delivery_Cost']].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.savefig('correlation_matrix.png')
print("Plot saved to correlation_matrix.png")
#Further analysis and model optimization would be done here in a real-world scenario.  This is a simplified example.