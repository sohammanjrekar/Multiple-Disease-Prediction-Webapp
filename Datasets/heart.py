import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
# AdaBoost Cross-Validation Scores: [0.79591837 0.85714286 0.79166667 0.79166667 0.79166667]
# Mean CV Score (AdaBoost): 0.8056122448979591
# Bagging Cross-Validation Scores: [0.81632653 0.81632653 0.8125     0.83333333 0.79166667]
# Mean CV Score (Bagging): 0.8140306122448979
# Soft Voting Cross-Validation Scores: [0.79591837 0.79591837 0.79166667 0.79166667 0.85416667]
# Mean CV Score (Soft Voting): 0.8058673469387754
# Random Forest Best Parameters: {'bootstrap': False, 'max_depth': 10, 'min_samples_leaf': 2, 'min_samples_split': 5, 'n_estimators': 100}
# Random Forest Best Cross-Validation Score: 0.8140306122448979
# Random Forest Accuracy: 80.33%
# Random Forest Precision: 0.83
# Random Forest Recall: 0.78
# Random Forest F1-Score: 0.81
import pandas as pd
df=pd.read_csv(r'C:\Users\mrsoh\Downloads\Multiple-Disease-Prediction-Webapp\Datasets\heart.csv')
X=df.drop(['target'],axis=1)
y=df['target']
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling/Normalization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Hyperparameter Tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define individual classifiers
rf_classifier = RandomForestClassifier(random_state=42)
xgb_classifier = XGBClassifier(random_state=42)

# Create AdaBoost classifier using the Random Forest as the base estimator
adaboost_classifier = AdaBoostClassifier(estimator=rf_classifier, random_state=42)

# Fit the AdaBoost classifier
adaboost_classifier.fit(X_train, y_train)

# Cross-Validation for AdaBoost classifier
adaboost_cv_scores = cross_val_score(adaboost_classifier, X_train, y_train, cv=5, scoring='accuracy')
print("AdaBoost Cross-Validation Scores:", adaboost_cv_scores)
print("Mean CV Score (AdaBoost):", adaboost_cv_scores.mean())

# Create Bagging classifier using the Random Forest as the base estimator
bagging_classifier = BaggingClassifier(estimator=rf_classifier, random_state=42)

# Fit the Bagging classifier
bagging_classifier.fit(X_train, y_train)

# Cross-Validation for Bagging classifier
bagging_cv_scores = cross_val_score(bagging_classifier, X_train, y_train, cv=5, scoring='accuracy')
print("Bagging Cross-Validation Scores:", bagging_cv_scores)
print("Mean CV Score (Bagging):", bagging_cv_scores.mean())

# Create a list of classifiers for soft voting
soft_voting_estimators = [('Random Forest', rf_classifier), ('XGBoost', xgb_classifier)]

# Create a soft voting classifier
soft_voting_classifier = VotingClassifier(estimators=soft_voting_estimators, voting='soft')

# Fit the soft voting classifier
soft_voting_classifier.fit(X_train, y_train)

# Cross-Validation for soft voting classifier
soft_voting_cv_scores = cross_val_score(soft_voting_classifier, X_train, y_train, cv=5, scoring='accuracy')
print("Soft Voting Cross-Validation Scores:", soft_voting_cv_scores)
print("Mean CV Score (Soft Voting):", soft_voting_cv_scores.mean())

# Hyperparameter tuning using GridSearchCV for Random Forest
rf_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

rf_grid_search = GridSearchCV(estimator=rf_classifier, param_grid=rf_param_grid, 
                               cv=5, scoring='accuracy', n_jobs=-1)
rf_grid_search.fit(X_train, y_train)

print("Random Forest Best Parameters:", rf_grid_search.best_params_)
print("Random Forest Best Cross-Validation Score:", rf_grid_search.best_score_)

# Evaluate the best Random Forest model
best_rf_model = rf_grid_search.best_estimator_
y_pred_rf = best_rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, y_pred_rf)
rf_precision = precision_score(y_test, y_pred_rf)
rf_recall = recall_score(y_test, y_pred_rf)
rf_f1 = f1_score(y_test, y_pred_rf)

print(f'Random Forest Accuracy: {rf_accuracy * 100:.2f}%')
print(f'Random Forest Precision: {rf_precision:.2f}')
print(f'Random Forest Recall: {rf_recall:.2f}')
print(f'Random Forest F1-Score: {rf_f1:.2f}')


