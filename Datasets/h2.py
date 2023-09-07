import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
import matplotlib.pyplot as plt
import seaborn as sns
# Random Oversampling:
# Accuracy: 0.8360655737704918
# Classification Report:
#               precision    recall  f1-score   support    

#            0       0.83      0.83      0.83        29    
#            1       0.84      0.84      0.84        32    

#     accuracy                           0.84        61    
#    macro avg       0.84      0.84      0.84        61    
# weighted avg       0.84      0.84      0.84        61    

# Confusion Matrix:
# [[24  5]
#  [ 5 27]]

# Random Undersampling:
# Accuracy: 0.8524590163934426
# Classification Report:
#               precision    recall  f1-score   support    

#            0       0.83      0.86      0.85        29    
#            1       0.87      0.84      0.86        32    

#     accuracy                           0.85        61    
#    macro avg       0.85      0.85      0.85        61    
# weighted avg       0.85      0.85      0.85        61    

# Confusion Matrix:
# [[25  4]
#  [ 5 27]]

# SMOTE-ENN:
# Accuracy: 0.8032786885245902
# Classification Report:
#               precision    recall  f1-score   support    

#            0       0.77      0.83      0.80        29    
#            1       0.83      0.78      0.81        32    

#     accuracy                           0.80        61    
#    macro avg       0.80      0.80      0.80        61    
# weighted avg       0.81      0.80      0.80        61    

# Confusion Matrix:
# [[24  5]
#  [ 7 25]]
import pandas as pd
data=pd.read_csv(r'C:\Users\mrsoh\Downloads\Multiple-Disease-Prediction-Webapp\Datasets\heart.csv')

# Split your data into features (X) and the target variable (y)
X = data.drop('target', axis=1)
y = data['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1. Random Oversampling (increase minority class)
oversampler = RandomOverSampler(sampling_strategy='auto', random_state=42)
X_resampled, y_resampled = oversampler.fit_resample(X_train, y_train)

# Train a machine learning model with the resampled data
model = RandomForestClassifier(random_state=42)
model.fit(X_resampled, y_resampled)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("Random Oversampling:")
print(f"Accuracy: {accuracy}")
print(f"Classification Report:\n{report}")
print(f"Confusion Matrix:\n{conf_matrix}")

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, square=True)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix (Random Oversampling)')
plt.show()

# 2. Random Undersampling (reduce majority class)
undersampler = RandomUnderSampler(sampling_strategy='auto', random_state=42)
X_resampled, y_resampled = undersampler.fit_resample(X_train, y_train)

# Train a machine learning model with the resampled data
model = RandomForestClassifier(random_state=42)
model.fit(X_resampled, y_resampled)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("\nRandom Undersampling:")
print(f"Accuracy: {accuracy}")
print(f"Classification Report:\n{report}")
print(f"Confusion Matrix:\n{conf_matrix}")

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, square=True)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix (Random Undersampling)')
plt.show()

# 3. SMOTE-ENN (combination of oversampling and undersampling)
smoteenn = SMOTEENN(sampling_strategy='auto', random_state=42)
X_resampled, y_resampled = smoteenn.fit_resample(X_train, y_train)

# Train a machine learning model with the resampled data
model = RandomForestClassifier(random_state=42)
model.fit(X_resampled, y_resampled)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("\nSMOTE-ENN:")
print(f"Accuracy: {accuracy}")
print(f"Classification Report:\n{report}")
print(f"Confusion Matrix:\n{conf_matrix}")

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, square=True)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix (SMOTE-ENN)')
plt.show()
