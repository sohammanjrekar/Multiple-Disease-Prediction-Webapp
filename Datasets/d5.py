import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier, BaggingClassifier, AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score
from sklearn.utils import resample, shuffle
from imblearn.over_sampling import SMOTE
import pandas as pd

# Class Distribution in Training Data:
# Class 0: 401 samples
# Class 1: 213 samples
# Cross-Validation Scores: [0.79503106 0.7515528  0.76875    0.75       0.76875   ]
# Mean CV Accuracy: 0.7668167701863353
# Test Set Accuracy: 0.7077922077922078
# Test Set Precision: 0.6912878787878788


df=pd.read_csv(r'C:\Users\mrsoh\Downloads\Multiple-Disease-Prediction-Webapp\Datasets\diabetes.csv')
X=df.drop(['Outcome'],axis=1)
y=df['Outcome']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check class distribution in the training data
class_counts = np.bincount(y_train)
print("Class Distribution in Training Data:")
for i, count in enumerate(class_counts):
    print(f"Class {i}: {count} samples")

# Check if oversampling is needed and perform oversampling with SMOTE
if class_counts.min() < class_counts.max():
    oversampler = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = oversampler.fit_resample(X_train, y_train)
    # Shuffle the resampled data
    X_train_resampled, y_train_resampled = shuffle(X_train_resampled, y_train_resampled, random_state=42)
else:
    X_train_resampled, y_train_resampled = X_train, y_train

# Create individual classifiers
svm_classifier = SVC(kernel='linear', C=1)
decision_tree_classifier = DecisionTreeClassifier()
knn_classifier = KNeighborsClassifier(n_neighbors=5)
random_forest_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Bagging with SVM
bagging_svm = BaggingClassifier(estimator=svm_classifier, n_estimators=10, random_state=42)

# AdaBoost with SVM (using algorithm='SAMME' instead of 'SAMME.R')
adaboost_svm = AdaBoostClassifier(estimator=svm_classifier, n_estimators=50, random_state=42, algorithm='SAMME')

# Create a VotingClassifier ensemble combining selected classifiers
ensemble = VotingClassifier(estimators=[
    ('svm', svm_classifier),
    ('decision_tree', decision_tree_classifier),
    ('knn', knn_classifier),
    ('random_forest', random_forest_classifier),
    ('bagging_svm', bagging_svm),
    ('adaboost_svm', adaboost_svm)
], voting='hard')

# Perform 5-fold cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
ensemble_cv_scores = cross_val_score(ensemble, X_train_resampled, y_train_resampled, cv=cv, scoring='accuracy')

print("Cross-Validation Scores:", ensemble_cv_scores)
print("Mean CV Accuracy:", ensemble_cv_scores.mean())

# Fit the ensemble on the entire training set
ensemble.fit(X_train_resampled, y_train_resampled)

# Make predictions on the test set
ensemble_predictions = ensemble.predict(X_test)
ensemble_accuracy = accuracy_score(y_test, ensemble_predictions)
ensemble_precision = precision_score(y_test, ensemble_predictions, average='macro')

print("Test Set Accuracy:", ensemble_accuracy)
print("Test Set Precision:", ensemble_precision)
