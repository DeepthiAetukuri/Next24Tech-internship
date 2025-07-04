import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')
data = {
    'square_footage': [1200, 1800, 1500, 2200, 1600, 1400, 2500, 1300, 2000, 1700],
    'bedrooms': [2, 3, 2, 4, 3, 2, 4, 2, 3, 3],
    'bathrooms': [2, 3, 2, 3, 2, 2, 4, 1, 3, 2],
    'age': [5, 10, 8, 2, 15, 12, 1, 20, 3, 7],
    'city': ['Mumbai', 'Delhi', 'Bangalore', 'Mumbai', 'Pune', 'Delhi', 'Bangalore', 'Pune', 'Mumbai', 'Delhi'],
    'furnished': ['Furnished', 'Unfurnished', 'Semi-furnished', 'Furnished', 'Unfurnished', 'Semi-furnished', 'Furnished', 'Unfurnished', 'Furnished', 'Semi-furnished'],
    'price_lakhs': [85, 60, 45, 120, 40, 55, 90, 35, 100, 65]
}
df = pd.DataFrame(data)
plt.figure(figsize=(10, 6))
sns.boxplot(x='city', y='price_lakhs', data=df, palette='Set2')
plt.title('House Price Distribution by City (₹ Lakhs)')
plt.xlabel('City')
plt.ylabel('Price (₹ Lakhs)')
plt.tight_layout()
plt.savefig('price_by_city.png')
plt.show()
plt.close()
numerical_cols = ['square_footage', 'bedrooms', 'bathrooms', 'age', 'price_lakhs']
plt.figure(figsize=(8, 6))
sns.heatmap(df[numerical_cols].corr(), annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap of Numerical Features')
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
plt.show()
plt.close()
X = df.drop('price_lakhs', axis=1)
y = df['price_lakhs']
numerical_cols = ['square_footage', 'bedrooms', 'bathrooms', 'age']
categorical_cols = ['city', 'furnished']
X['price_per_sqft'] = y / X['square_footage'] * 100000  # INR per sqft
numerical_cols.append('price_per_sqft')
city_index = {'Mumbai': 1.5, 'Delhi': 1.2, 'Bangalore': 1.1, 'Pune': 0.9}
X['city_index'] = X['city'].map(city_index)
numerical_cols.append('city_index')
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_cols)
    ])
model = RandomForestRegressor(n_estimators=100, random_state=42)
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', model)
])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"\nModel Performance:")
print(f"Mean Absolute Error: ₹{mae:.2f} lakhs")
print(f"R² Score: {r2:.4f}")
cat_transformer = pipeline.named_steps['preprocessor'].named_transformers_['cat']
cat_feature_names = cat_transformer.get_feature_names_out(categorical_cols)
feature_names = numerical_cols + list(cat_feature_names)
importances = pipeline.named_steps['model'].feature_importances_
plt.figure(figsize=(10, 6))
sns.barplot(x=importances, y=feature_names, palette='viridis')
plt.title('Feature Importance in Random Forest Model')
plt.xlabel('Importance')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.show()
plt.close()
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Price (₹ Lakhs)')
plt.ylabel('Predicted Price (₹ Lakhs)')
plt.title('Actual vs Predicted House Prices')
plt.tight_layout()
plt.savefig('actual_vs_predicted.png')
plt.show()
plt.close()
sample_house = pd.DataFrame({
    'square_footage': [1500],
    'bedrooms': [2],
    'bathrooms': [2],
    'age': [5],
    'city': ['Mumbai'],
    'furnished': ['Semi-furnished'],
    'price_per_sqft': [0.006],  
    'city_index': [1.5]
})
predicted_price = pipeline.predict(sample_house)
print(f"\nPrice Prediction:")
print(f"Predicted price for sample house (1500 sqft, 2 bedrooms, 2 bathrooms, 5 years old, Mumbai, Semi-furnished): ₹{predicted_price[0]:.2f} lakhs")