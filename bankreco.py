import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

# Sample dataset
data = {
    'Age': [25, 32, 47, 51],
    'Salary': [50000, 60000, 100000, 110000],
    'Profession': [1, 2, 3, 4]  # 1: Engineer, 2: Doctor, 3: Lawyer, 4: Student
}

# Target variable (Products)
# 0: Fixed Deposits, 1: Investment, 2: Mutual Funds, 3: Loans
target = [0, 1, 2, 3]

# Create a DataFrame
df = pd.DataFrame(data)
df['Product'] = target

# Define features and target
X = df[['Age', 'Salary', 'Profession']]
y = df['Product']

# Initialize the model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X, y)

# Save the model as a .pkl file
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Model has been saved to model.pkl")
