import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("train.csv", low_memory=False)

# Drop unnecessary columns
df = df[["Annual_Income", "Num_of_Loan", "Num_of_Delayed_Payment", "Credit_History_Age", "Credit_Score"]]

# Convert 'Credit_Score' to numeric values
credit_score_mapping = {"Poor": 0, "Standard": 1, "Good": 2}
df["Credit_Score"] = df["Credit_Score"].map(credit_score_mapping)

# Convert numerical columns properly
df["Annual_Income"] = pd.to_numeric(df["Annual_Income"], errors="coerce")
df["Num_of_Loan"] = pd.to_numeric(df["Num_of_Loan"], errors="coerce")
df["Num_of_Delayed_Payment"] = pd.to_numeric(df["Num_of_Delayed_Payment"], errors="coerce")
df["Credit_History_Age"] = df["Credit_History_Age"].str.extract(r"(\d+)").astype(float)  # Extracts numeric years

# Drop rows with missing values after conversion
df.dropna(inplace=True)

# Define features (X) and target (y)
X = df.drop(columns=["Credit_Score"])  # Features
y = df["Credit_Score"]  # Target

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train RandomForest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save trained model
joblib.dump(model, "credit_score_model.pkl")

print("Model training complete! Saved as 'credit_score_model.pkl'.")
