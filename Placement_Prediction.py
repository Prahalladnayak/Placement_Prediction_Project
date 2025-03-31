import pandas as pd
import numpy as np
import pickle

# Load dataset
dataset = pd.read_csv(r"C:\Users\praha\OneDrive\Desktop\Data Analyst Material\placment\placementdata.csv")

# Drop unnecessary columns
dataset = dataset.drop(columns=["StudentID", "SSC_Marks", "HSC_Marks"])

# Convert categorical variables
dataset = pd.get_dummies(data=dataset, columns=["ExtracurricularActivities", "PlacementTraining", "PlacementStatus"], drop_first=True)

# Define features and target
x = dataset.iloc[:, :-1]  # ✅ Ensure all 8 features are included
y = dataset.iloc[:, -1]

# Split dataset
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=45)

# Scale features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Train model
from sklearn.svm import SVC
model = SVC(kernel="rbf", C=1, gamma=0.1)  # ✅ Best tuned hyperparameters
model.fit(x_train, y_train)

# Save model and scaler
with open("placement_model.pkl", "wb") as file:
    pickle.dump(model, file)

with open("scaler.pkl", "wb") as file:
    pickle.dump(scaler, file)

print("✅ Model and scaler saved successfully!")
