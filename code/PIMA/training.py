import os
import json
import joblib
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from scipy.optimize import minimize
from imblearn.combine import SMOTETomek
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config_utils import load_config
from feature_engineer import feature_engineer
from data_prep import KFoldTargetEncoder, preprocess_numeric
from models import model_factory
from evaluation import generate_report
import logging

# suppress overly chatty warnings (still let errors through)
warnings.filterwarnings("ignore", category=Warning)

def run_full_pipeline(cfg):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('run')
    logger.info("Loading data...")
    df = pd.read_csv(cfg['data_path'])
    if cfg['target'] not in df.columns:
        raise KeyError("Target missing")
    df = feature_engineer(df)
    X = df.drop(columns=[cfg['target']])
    y = df[cfg['target']].astype(int)

    # split
    X_train_raw, X_hold_raw, y_train, y_hold = train_test_split(
        X, y, test_size=cfg['test_size'], stratify=y, random_state=cfg['random_state']
    )

    # categorical detection & CV-safe target encoding if needed
    cat_cols = X_train_raw.select_dtypes(include=['object','category']).columns.tolist()
    te = None
    if cat_cols:
        te = KFoldTargetEncoder(cols=cat_cols, n_splits=cfg['cv_folds'], random_state=cfg['random_state'])
        # OOF encoding on training set
        X_train_enc = te.transform(X_train_raw, y_train)
        # drop original categorical columns and keep encoded columns (col_te)
        for c in cat_cols:
            if c in X_train_enc.columns:
                X_train_enc = X_train_enc.drop(columns=[c])
        # fit full-train maps for holdout transform
        te.fit(X_train_raw, y_train)
        X_hold_enc = te.transform(X_hold_raw, None)
        for c in cat_cols:
            if c in X_hold_enc.columns:
                X_hold_enc = X_hold_enc.drop(columns=[c])
        X_train_final = X_train_enc
        X_hold_final = X_hold_enc
    else:
        X_train_final = X_train_raw.copy()
        X_hold_final = X_hold_raw.copy()

    # numeric preprocess (impute + scale)
    X_train_num, X_hold_num, imputer, scaler, num_cols = preprocess_numeric(X_train_final, X_hold_final)

    # feature selection
    selector = SelectKBest(f_classif, k=min(20, X_train_num.shape[1]))
    X_train_sel = selector.fit_transform(X_train_num, y_train)
    X_hold_sel = selector.transform(X_hold_num)
    selected_cols = list(np.array(num_cols)[selector.get_support()])

    # resample to handle imbalance
    smt = SMOTETomek(random_state=cfg['random_state'])
    X_res, y_res = smt.fit_resample(X_train_sel, y_train)
    logger.info("After resample class counts: %s", np.bincount(y_res))

    # build models (from your models.py factory)
    models = model_factory(use_gpu=cfg.get('use_gpu', False), random_state=cfg['random_state'])

    # optional: fix XGBoost params to avoid deprecated param warning
    if 'XGBoost' in models:
        try:
            models['XGBoost'].set_params(eval_metric='logloss')
            # ensure we don't pass deprecated use_label_encoder if present
            if 'use_label_encoder' in models['XGBoost'].get_params():
                models['XGBoost'].set_params(use_label_encoder=False)
        except Exception:
            pass

    # increase MLP iterations if present
    if 'MLP' in models:
        try:
            models['MLP'].set_params(max_iter=1000)
        except Exception:
            pass

    # wrap certain models with StandardScaler pipeline where scaling helps (SVC, KNN, MLP)
    need_scale = {'SVC', 'KNN', 'MLP'}
    wrapped_models = {}
    for name, est in models.items():
        if name in need_scale:
            # pipeline: scaler then estimator (use clone in CV later)
            wrapped_models[name] = Pipeline([('scaler', StandardScaler()), ('clf', est)])
        else:
            wrapped_models[name] = est

    # keep configured models if provided, else all
    keep = cfg.get('models', list(wrapped_models.keys()))
    models = {k: wrapped_models[k] for k in keep if k in wrapped_models}

    # quick CV ranking on resampled data
    cv = StratifiedKFold(n_splits=cfg['cv_folds'], shuffle=True, random_state=cfg['random_state'])
    cv_scores = {}
    for name, m in models.items():
        try:
            sc = cross_val_score(m, X_res, y_res, cv=cv, scoring='accuracy', n_jobs=1).mean()
            cv_scores[name] = float(sc)
            logger.info("CV %s = %.4f", name, sc)
        except Exception as e:
            logger.warning("Model %s failed CV: %s", name, e)

    if not cv_scores:
        raise RuntimeError("No models successfully evaluated in CV. Check model configs and data.")

    # select top_n
    top_sorted = sorted(cv_scores.items(), key=lambda x: x[1], reverse=True)[:cfg['top_n']]
    top_names = [t[0] for t in top_sorted]
    logger.info("Top models: %s", top_names)

    # prepare tuned_models placeholder (insert Optuna tuning here if desired)
    tuned_models = {n: models[n] for n in top_names}

    # calibrate tuned models (wrap with CalibratedClassifierCV for better probability estimates)
    calibrated = {}
    from sklearn.calibration import CalibratedClassifierCV
    for n, clf in tuned_models.items():
        try:
            # Fit on resampled training set
            clf.fit(X_res, y_res)
            cal = CalibratedClassifierCV(clf, method='sigmoid', cv=3)
            cal.fit(X_res, y_res)
            calibrated[n] = cal
        except Exception as e:
            logger.warning("Calibration failed for %s: %s. Using raw estimator.", n, e)
            # fallback: keep original classifier fitted
            calibrated[n] = clf

    estimators = [(n, calibrated[n]) for n in top_names]

    # optimize soft voting weights (numerical minimization)
    def ensemble_obj(w):
        w = np.array(w)
        if np.any(w < 0):
            return 1.0
        w = w / (w.sum() + 1e-9)
        vc = VotingClassifier(estimators=estimators, voting='soft', weights=list(w))
        try:
            sc = cross_val_score(vc, X_res, y_res, cv=3, scoring='accuracy', n_jobs=1).mean()
            return -sc
        except Exception as e:
            logger.warning("Voting CV failed during weight optimization: %s", e)
            return 1.0

    init = np.ones(len(estimators)) / len(estimators)
    bounds = [(0, 1)] * len(estimators)
    cons = ({'type': 'eq', 'fun': lambda w: w.sum() - 1.0})
    res = minimize(ensemble_obj, x0=init, bounds=bounds, constraints=cons, method='SLSQP', options={'maxiter': 200})
    w_opt = res.x if res.success else init
    w_opt = np.clip(w_opt, 0, 1)
    w_opt = w_opt / (w_opt.sum() + 1e-9)

    final_blend = VotingClassifier(estimators=estimators, voting='soft', weights=list(w_opt))
    final_blend.fit(X_res, y_res)

    # stacking (meta-learner)
    meta = LogisticRegression(max_iter=2000)
    stacking = StackingClassifier(estimators=estimators, final_estimator=meta, cv=5, stack_method='predict_proba')
    stacking.fit(X_res, y_res)

    # Evaluate on holdout (use selected features)
    y_pred_blend = final_blend.predict(X_hold_sel)
    y_pred_stack = stacking.predict(X_hold_sel)
    acc_blend = accuracy_score(y_hold, y_pred_blend)
    acc_stack = accuracy_score(y_hold, y_pred_stack)
    logger.info("Holdout blend acc=%.4f stack acc=%.4f", acc_blend, acc_stack)

    chosen = 'stack' if acc_stack >= acc_blend else 'blend'
    final_model = stacking if chosen == 'stack' else final_blend
    logger.info("Chosen final model: %s", chosen)

    # Save artifacts (models + preprocessors + metadata)
    artifact_dir = cfg['artifact_dir']
    os.makedirs(artifact_dir, exist_ok=True)

    joblib.dump(final_model, os.path.join(artifact_dir, 'final_model.joblib'))
    if cfg.get('save_pickle', True):
        import pickle
        with open(os.path.join(artifact_dir, 'final_model.sav'), 'wb') as f:
            pickle.dump(final_model, f)

    preproc_obj = {
        'imputer': imputer,
        'scaler': scaler,
        'selector': selector,
        'num_cols': num_cols,
        'selected_cols': selected_cols,
        'target_encoder': te  # may be None if no categoricals
    }
    joblib.dump(preproc_obj, os.path.join(artifact_dir, 'preproc.joblib'))

    with open(os.path.join(artifact_dir, 'cv_scores.json'), 'w') as fh:
        json.dump({'cv_scores': cv_scores, 'top': top_names, 'weights': list(map(float, w_opt))}, fh, indent=2)

    # generate PDF report (plots + metrics)
    metrics = {'blend_acc': acc_blend, 'stack_acc': acc_stack}
    generate_report(
        os.path.join(artifact_dir, cfg.get('report_pdf_name', 'report.pdf')),
        cv_scores, top_names, X_hold_sel, y_hold, final_blend, stacking, metrics
    )

    print("Artifacts saved in", artifact_dir)
    return artifact_dir
