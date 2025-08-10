from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

def model_factory(use_gpu=False, random_state=42):
    models = {}
    models['RandomForest'] = RandomForestClassifier(n_estimators=300, random_state=random_state, n_jobs=-1)
    models['ExtraTrees'] = ExtraTreesClassifier(n_estimators=300, random_state=random_state, n_jobs=-1)
    models['HistGB'] = HistGradientBoostingClassifier(max_iter=300, random_state=random_state)
    models['GradientBoosting'] = GradientBoostingClassifier(n_estimators=300, random_state=random_state)
    models['SVC'] = SVC(probability=True, kernel='rbf', random_state=random_state)
    models['KNN'] = KNeighborsClassifier(n_neighbors=7)
    models['MLP'] = MLPClassifier(hidden_layer_sizes=(64,32), max_iter=500, random_state=random_state)
    models['GaussianNB'] = GaussianNB()
    models['DecisionTree'] = DecisionTreeClassifier(random_state=random_state)
    # boosted
    xgb_params = {'n_estimators':300, 'learning_rate':0.05, 'max_depth':5, 'random_state':random_state}
    if use_gpu:
        xgb_params.update({'tree_method':'gpu_hist'})
    models['XGBoost'] = XGBClassifier(**xgb_params, use_label_encoder=False, eval_metric='logloss', n_jobs=1)
    cb_params = {'iterations':300, 'learning_rate':0.05, 'depth':6, 'random_state':random_state, 'verbose':0}
    models['CatBoost'] = CatBoostClassifier(**cb_params)
    return models
