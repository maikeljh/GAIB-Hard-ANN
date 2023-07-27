# Michael Jonathan Halim 13521124
# GAIB - Hard - ANN

import numpy as np
import pandas as pd
from ann import Sequential, Dense, relu, sigmoid
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

# Load data
df = pd.read_csv('breast_cancer_data.csv')

# Handling missing data for numeric columns with mean imputation
numeric_columns = df.select_dtypes(include=[np.number]).columns
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())

# Preprocessing the data
# Map M to 1 and B to 0
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

# Extract features and target
X = df.drop(['id', 'diagnosis'], axis=1)
y = df['diagnosis']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Count class distribution before oversampling
class_distribution = pd.Series(y_train).value_counts()

# Print the class distribution
print("Before Oversampling\nClass Distribution:")
print(class_distribution)
print()

# Apply SMOTE to balance the classes
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Count class distribution after oversampling
class_distribution = pd.Series(y_train).value_counts()

# Print the class distribution
print("After Oversampling\nClass Distribution:")
print(class_distribution)
print()

# Create the StandardScaler object
scaler = StandardScaler()

# Feature scaling
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create the model
model = Sequential()
model.add(Dense(input_size=X_train.shape[1], output_size=2, activation_function=relu))
model.add(Dense(input_size=2, output_size=1, activation_function=sigmoid))

# Train the model
model.fit(X_train, y_train, epochs=100, learning_rate=0.1)

# Test the model
predictions = model.predict(X_test)

# Evaluate
classification_rep = classification_report(y_test, predictions)

print("\nClassification Report:")
print(classification_rep)