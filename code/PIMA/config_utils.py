import yaml
from pathlib import Path

DEFAULTS = {
    'data_path':'pima_diabetes.csv',
    'target':'Outcome',
    'test_size':0.2,
    'random_state':42,
    'cv_folds':5,
    'nested_cv_folds':3,
    'top_n':3,
    'n_trials':20,
    'artifact_dir':'artifacts',
    'use_gpu':False,
    'save_pickle': True,
    'report_pdf_name': 'pima_diabetes_report.pdf'
}

def load_config(path):
    with open(path,'r') as f:
        user = yaml.safe_load(f) or {}
    cfg = DEFAULTS.copy()
    cfg.update(user)
    Path(cfg['artifact_dir']).mkdir(parents=True, exist_ok=True)
    return cfg
