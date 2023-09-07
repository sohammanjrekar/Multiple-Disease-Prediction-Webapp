import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt
import seaborn as sns

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

import pandas as pd
data=pd.read_csv(r'C:\Users\mrsoh\Downloads\Multiple-Disease-Prediction-Webapp\Datasets\heart.csv')

# Split your data into features (X) and the target variable (y)
X = data.drop('target', axis=1)
y = data['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Undersampling (reduce majority class)
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

print("Random Undersampling:")
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
