def warn(*args, **kwargs):
    pass
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, RobustScaler, MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin
import matplotlib.pyplot as plt
df=pd.read_csv("/home/labs/nyosef/elibe/ML_healthcare/data/dataset_diabetes/diabetic_data.csv")
print(df.shape)
import custom_classes_and_functions as ccf
X_train, X_test, y_train, y_test= ccf.complete_pp(df,ohe=False)
X_train.shape, X_test.shape, y_train.value_counts(), y_test.value_counts()

from sklearn.metrics import roc_auc_score as auc

from lightgbm import LGBMClassifier

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

import optuna

def objective001(trial,data,target):


    param = {
        "learning_rate":[0.01],
        "n_estimators":[440],
        "num_leaves":[22],

        "max_depth":[trial.suggest_int("max_depth",8,11)],

        "feature_fraction":[0.5],
        "max_bins":[trial.suggest_int("max_bins",63,255,)],


        "lambda_l1": [trial.suggest_float("lambda_l1", 1e-6, 100, log=True)],
        "lambda_l2": [trial.suggest_float("lambda_l2", 1e-6, 100, log=True)],
        "min_data_in_leaf":[trial.suggest_int("min_data_in_leaf", 2, 100,)],

    }
    model_lgbm = LGBMClassifier(random_state=0,metric="auc",objective="binary",
                                bagging_freq=5,max_cat_to_onehot=10,data_sample_strategy="goss",
                                is_unbalance=True,verbose=-1,n_jobs=1)
    model = GridSearchCV(estimator = model_lgbm,param_grid = param,cv=5,scoring="roc_auc",n_jobs=-1,)



    model.fit(data,target)

    return model.best_score_ - model.cv_results_["std_test_score"][model.best_index_]



study = optuna.create_study(direction='maximize')
study.optimize(lambda trial: objective001(trial, data=X_train, target=y_train), n_trials=500,)
print("lr=0.01",study.best_params, study.best_value)
"""
def objective01(trial,data,target):


    param = {
        "learning_rate":[0.1],
        "n_estimators":[24],
        "num_leaves":[20],

        "max_depth":[trial.suggest_categorical("max_depth",[-1,8])],

        "feature_fraction":[0.4],
        "max_bins":[trial.suggest_int("max_bins",63,255,)],


        "lambda_l1": [trial.suggest_float("lambda_l1", 1e-6, 100, log=True)],
        "lambda_l2": [trial.suggest_float("lambda_l2", 1e-6, 100, log=True)],
        "min_data_in_leaf":[trial.suggest_int("min_data_in_leaf", 2, 100,)],

    }
    model_lgbm = LGBMClassifier(random_state=0,metric="auc",objective="binary",
                                bagging_freq=5,max_cat_to_onehot=10,data_sample_strategy="goss",
                                is_unbalance=True,verbose=-1,n_jobs=1)
    model = GridSearchCV(estimator = model_lgbm,param_grid = param,cv=5,scoring="roc_auc",n_jobs=-1,)



    model.fit(data,target)

    return model.best_score_ - model.cv_results_["std_test_score"][model.best_index_]



study = optuna.create_study(direction='maximize')
study.optimize(lambda trial: objective01(trial, data=X_train, target=y_train), n_trials=1e3,)
print("lr=0.1",study.best_params, study.best_value)

"""