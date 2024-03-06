import pandas as pd
import numpy as np
from paths import DATA_PATH
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin


class ImbalanceTreater(BaseEstimator, TransformerMixin):
    '''deals with the imbalance in the target variable'''
    def __init__(self,random_under=False,random_over=False,smotenc=False):
        self.random_under=random_under
        self.random_over=random_over
        self.smotenc=smotenc


    def fit(self, X, y=None):
        
        return self

    def transform(self, X):


        if self.random_under:
            from imblearn.under_sampling import RandomUnderSampler 
            y=X['readmitted']
            X=X.drop('readmitted',axis=1)
            sam = RandomUnderSampler(random_state=42)
            X,y=sam.fit_resample(X,y)
            sampled_df = pd.DataFrame(X, columns=X.columns)
            sampled_df['readmitted'] = y
            return sampled_df
        elif self.random_over:
            from imblearn.over_sampling import RandomOverSampler
            y=X['readmitted']
            X=X.drop('readmitted',axis=1)
            sam = RandomOverSampler(random_state=42)
            X,y=sam.fit_resample(X,y)
            sampled_df = pd.DataFrame(X, columns=X.columns)
            sampled_df['readmitted'] = y
            return sampled_df
        elif self.smotenc:
            from imblearn.over_sampling import SMOTENC
            y=X['readmitted']
            X=X.drop('readmitted',axis=1)
            sam = SMOTENC(random_state=42)
            X,y=sam.fit_resample(X,y)
            sampled_df = pd.DataFrame(X, columns=X.columns)
            sampled_df['readmitted'] = y
            return sampled_df
        else:
            return X
