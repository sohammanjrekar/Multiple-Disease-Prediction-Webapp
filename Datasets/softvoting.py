import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
# Hard Voting Cross-Validation Scores: [0.79674797 0.7804878  0.69918699 0.74796748 0.77868852]
# Mean CV Score (Hard Voting): 0.7606157536985206
# Soft Voting Cross-Validation Scores: [0.74796748 0.7804878  0.7398374  0.75609756 0.76229508]
# Mean CV Score (Soft Voting): 0.7573370651739304
# Random Forest Best Parameters: {'bootstrap': True, 'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 100}
# Random Forest Best Cross-Validation Score: 0.7834199653471945
# Random Forest Accuracy: 73.38%
# Random Forest Precision: 0.62
# Random Forest Recall: 0.65
# Random Forest F1-Score: 0.64
import pandas as pd
df=pd.read_csv(r'C:\Users\mrsoh\Downloads\Multiple-Disease-Prediction-Webapp\Datasets\diabetes.csv')
X=df.drop(['Outcome'],axis=1)
y=df['Outcome']
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

# Create a list of classifiers for hard voting
hard_voting_estimators = [('Random Forest', rf_classifier), ('XGBoost', xgb_classifier)]

# Create a hard voting classifier
hard_voting_classifier = VotingClassifier(estimators=hard_voting_estimators, voting='hard')

# Fit the hard voting classifier
hard_voting_classifier.fit(X_train, y_train)

# Cross-Validation for hard voting classifier
hard_voting_cv_scores = cross_val_score(hard_voting_classifier, X_train, y_train, cv=5, scoring='accuracy')
print("Hard Voting Cross-Validation Scores:", hard_voting_cv_scores)
print("Mean CV Score (Hard Voting):", hard_voting_cv_scores.mean())

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