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
from imblearn.ensemble import BalancedRandomForestClassifier
import optuna
from sklearn.model_selection import GridSearchCV
df=pd.read_csv("/home/labs/nyosef/elibe/ML_healthcare/data/dataset_diabetes/diabetic_data.csv")
print(df.shape)
import custom_classes_and_functions as ccf
X_train, X_test, y_train, y_test= ccf.complete_pp(df,ohe=True)
X_train.shape, X_test.shape, y_train.value_counts(), y_test.value_counts()

def objective(trial,data,target):
    param={
    "n_estimators":[trial.suggest_int("n_estimators",1500,2000)],
    "max_samples":[0.1,],
        "max_depth":[11],
        "min_samples_split":[trial.suggest_int("min_samples_split",5,100)],
        "min_samples_leaf":[trial.suggest_int("min_samples_leaf",5,100)],
        "max_features":[trial.suggest_categorical("max_features",["sqrt","log2"])],
        "ccp_alpha":[trial.suggest_float("ccp_alpha",1e-6,1,log=True,)],
        
    }
    brf = BalancedRandomForestClassifier(random_state=0,verbose=0,
                                     oob_score=True,
                                      sampling_strategy=1.0, 
                                      replacement=False,bootstrap=True,
                                      
                                      )

    model = GridSearchCV(estimator = brf,param_grid = param,cv=5,scoring="roc_auc",n_jobs=-1,verbose=0,)



    model.fit(data,target)

    return model.best_score_ - model.cv_results_["std_test_score"][model.best_index_]
study = optuna.create_study(direction='maximize')
study.optimize(lambda trial: objective(trial, data=X_train, target=y_train), n_trials=500,)
print("***",study.best_params, study.best_value)



