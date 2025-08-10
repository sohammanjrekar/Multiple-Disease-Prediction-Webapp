import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

class KFoldTargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, cols=None, n_splits=5, smoothing=1.0, random_state=42):
        self.cols = cols or []
        self.n_splits = n_splits
        self.smoothing = smoothing
        self.random_state = random_state
        self.global_mean_ = None
        self.maps_ = {}

    def fit(self, X, y):
        X = X.reset_index(drop=True)
        y = pd.Series(y).reset_index(drop=True)
        self.global_mean_ = y.mean()
        # store full-train stats for transform
        for c in self.cols:
            tmp = pd.concat([X[c].astype(str), y], axis=1)
            grp = tmp.groupby(c)[y.name if hasattr(y,'name') else 0].agg(['mean','count'])
            self.maps_[c] = grp.to_dict(orient='index')
        return self

    def transform(self, X, y=None):
        X = X.copy().reset_index(drop=True)
        if y is not None:
            # compute out-of-fold encoding
            y = pd.Series(y).reset_index(drop=True)
            oof = pd.DataFrame(index=X.index)
            skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
            for tr_idx, val_idx in skf.split(X, y):
                X_tr = X.loc[tr_idx]
                y_tr = y.loc[tr_idx]
                for c in self.cols:
                    stats = pd.concat([X_tr[c].astype(str), y_tr], axis=1).groupby(c)[y_tr.name if hasattr(y_tr,'name') else 0].agg(['mean','count'])
                    mapping = stats.to_dict(orient='index')
                    vals = X.loc[val_idx, c].astype(str).map(lambda k: mapping[k]['mean'] if k in mapping else self.global_mean_)
                    oof.loc[val_idx, c + '_te'] = vals.fillna(self.global_mean_)
            for c in self.cols:
                X[c + '_te'] = oof[c + '_te'].values
            return X
        else:
            # use maps_ computed on fit
            for c in self.cols:
                def getval(k):
                    if str(k) in self.maps_[c]:
                        entry = self.maps_[c][str(k)]
                        mean = entry['mean']; cnt = entry['count']
                        sm = (mean * cnt + self.global_mean_ * self.smoothing) / (cnt + self.smoothing)
                        return sm
                    else:
                        return self.global_mean_
                X[c + '_te'] = X[c].map(getval).fillna(self.global_mean_)
            return X

def preprocess_numeric(X_train_df, X_hold_df):
    num_cols = X_train_df.select_dtypes(include=[np.number]).columns.tolist()
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    Xtr = pd.DataFrame(imputer.fit_transform(X_train_df[num_cols]), columns=num_cols, index=X_train_df.index)
    Xtr = pd.DataFrame(scaler.fit_transform(Xtr), columns=num_cols, index=X_train_df.index)
    Xh = pd.DataFrame(imputer.transform(X_hold_df[num_cols]), columns=num_cols, index=X_hold_df.index)
    Xh = pd.DataFrame(scaler.transform(Xh), columns=num_cols, index=X_hold_df.index)
    return Xtr, Xh, imputer, scaler, num_cols
