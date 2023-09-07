import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier, BaggingClassifier, AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import pandas as pd



# Bagging SVM Accuracy: 0.7662337662337663
# AdaBoost SVM Accuracy: 0.7337662337662337
# Ensemble Accuracy: 0.7532467532467533



# Load the Iris dataset
df=pd.read_csv(r'C:\Users\mrsoh\Downloads\Multiple-Disease-Prediction-Webapp\Datasets\diabetes.csv')
X=df.drop(['Outcome'],axis=1)
y=df['Outcome']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create individual classifiers
svm_classifier = SVC(kernel='linear', C=1)
decision_tree_classifier = DecisionTreeClassifier()
knn_classifier = KNeighborsClassifier(n_neighbors=5)
random_forest_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
gradient_boosting_classifier = GradientBoostingClassifier(n_estimators=100, random_state=42)

# Bagging with SVM
bagging_svm = BaggingClassifier(base_estimator=svm_classifier, n_estimators=10, random_state=42)
bagging_svm.fit(X_train, y_train)
bagging_svm_predictions = bagging_svm.predict(X_test)
bagging_svm_accuracy = accuracy_score(y_test, bagging_svm_predictions)

print("Bagging SVM Accuracy:", bagging_svm_accuracy)

# AdaBoost with SVM (using algorithm='SAMME' instead of 'SAMME.R')
adaboost_svm = AdaBoostClassifier(base_estimator=svm_classifier, n_estimators=50, random_state=42, algorithm='SAMME')
adaboost_svm.fit(X_train, y_train)
adaboost_svm_predictions = adaboost_svm.predict(X_test)
adaboost_svm_accuracy = accuracy_score(y_test, adaboost_svm_predictions)

print("AdaBoost SVM Accuracy:", adaboost_svm_accuracy)

# Create a VotingClassifier ensemble combining all classifiers
ensemble = VotingClassifier(estimators=[
    ('svm', svm_classifier),
    ('decision_tree', decision_tree_classifier),
    ('knn', knn_classifier),
    ('random_forest', random_forest_classifier),
    ('gradient_boosting', gradient_boosting_classifier),
    ('bagging_svm', bagging_svm),
    ('adaboost_svm', adaboost_svm)
], voting='soft')

ensemble.fit(X_train, y_train)

# Make predictions and evaluate accuracy
ensemble_predictions = ensemble.predict(X_test)
ensemble_accuracy = accuracy_score(y_test, ensemble_predictions)

print("Ensemble Accuracy:", ensemble_accuracy)
