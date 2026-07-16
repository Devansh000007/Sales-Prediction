import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from google.colab import files
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Upload dataset
print("Upload Advertising.csv")
uploaded = files.upload()

# Read uploaded file automatically
filename = list(uploaded.keys())[0]
df = pd.read_csv(filename)

print("\nFirst 5 Rows:")
print(df.head())

print("\nColumn Names:")
print(df.columns)

# Remove extra column if present
if "Unnamed: 0" in df.columns:
    df.drop(columns=["Unnamed: 0"], inplace=True)

# Convert all column names to lowercase
df.columns = df.columns.str.lower()

print("\nUpdated Columns:")
print(df.columns)

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Features and Target
X = df[['tv', 'radio', 'newspaper']]
y = df['sales']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
print("\n========== RESULTS ==========")
print("R² Score :", round(r2_score(y_test, y_pred), 3))
print("MAE      :", round(mean_absolute_error(y_test, y_pred), 3))
print("RMSE     :", round(mean_squared_error(y_test, y_pred) ** 0.5, 3))

# Actual vs Predicted Plot
plt.figure(figsize=(6,4))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.grid(True)
plt.show()

# Feature Importance
importance = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
}).sort_values(by="Coefficient", ascending=False)

print("\nFeature Importance:")
print(importance)

plt.figure(figsize=(6,4))
sns.barplot(data=importance, x="Coefficient", y="Feature")
plt.title("Feature Importance")
plt.show()
