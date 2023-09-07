# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
# Accuracy: 68.83%
#               precision    recall  f1-score   support

#            0       0.79      0.71      0.74        99
#            1       0.55      0.65      0.60        55

#     accuracy                           0.69       154
#    macro avg       0.67      0.68      0.67       154
# weighted avg       0.70      0.69      0.69       154

import pandas as pd
# Load the Iris dataset

df=pd.read_csv(r'C:\Users\mrsoh\Downloads\Multiple-Disease-Prediction-Webapp\Datasets\diabetes.csv')
X=df.drop(['Outcome'],axis=1)
y=df['Outcome']


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the feature values
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a machine learning model (Random Forest or XGBoost)
# You can choose one of the following models and tune hyperparameters:
# model = RandomForestClassifier(n_estimators=100, random_state=42)
model = XGBClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Print classification report for more detailed evaluation
print(classification_report(y_test, y_pred))
