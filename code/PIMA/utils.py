import os
import joblib
def save_joblib(obj, path):
    joblib.dump(obj, path)
def load_joblib(path):
    return joblib.load(path)
