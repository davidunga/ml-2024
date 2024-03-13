
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
model_cat = CatBoostClassifier(random_seed=42,auto_class_weights=None,verbose=0)
model_lgbm = LGBMClassifier(random_state=42,verbose=-1)
model_xgb = XGBClassifier(random_state=42)
model_rf=RandomForestClassifier(random_state=42)
model_imrf=BalancedRandomForestClassifier(random_state=42)


for x in [model_lgbm,model_xgb,model_rf,model_cat,model_imrf]:
    ohe=OneHotEncoder()
    test_params = {
    'max_depth':[2,3]
    }

    model = GridSearchCV(estimator = x,param_grid = test_params,cv=3)
    model.fit(ohe.fit_transform(dt),y)
    print("*************************",model.best_params_,model.best_score_,model)
