from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

model = CatBoostClassifier(random_seed=42,auto_class_weights=None)
model = LGBMClassifier(random_state=42)
model = XGBClassifier(random_state=42)
