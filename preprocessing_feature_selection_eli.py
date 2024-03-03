
import pandas as pd
import numpy as np
from paths import DATA_PATH
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin



class RowRemoverDuplicatePatient(BaseEstimator, TransformerMixin):
    '''Drops rows based on duplicate patient ID'''
    def __init__(self,trans=True):
        self.trans=trans


    def fit(self, X, y=None):
        
        return self

    def transform(self, X):
        if self.trans: 
            return X.data.drop_duplicates("patient_nbr",inplace=True)

class Age2Int(BaseEstimator, TransformerMixin):
    '''age group to int of avg.'''
    def __init__(self, trans=True):
        self.trans=trans


    def fit(self, X, y=None):
        
        return self

    def transform(self, X):
        if self.trans:
            X["age_mean"]=X["age_mean"]=X.apply(lambda x:np.mean([int(el) for el in x.age[1:-1].split('-')]), axis=1)
            return X
    
class DiabetesDiag(BaseEstimator, TransformerMixin):
    '''makes feature for diabetes diagnosis'''
    def __init__(self, trans=True):
        self.trans=trans


    def fit(self, X, y=None):
        
        return self

    def transform(self, X):
        if self.trans:
            X["diabetes_diag"]= X[["diag_1", "diag_2", "diag_3"]].apply(lambda row: ("diabetes" or "endocrine/nutritional/metabolic diseases and immunity disorders") in row.values, axis=1)

            return X

class MedInterventionAdder(BaseEstimator, TransformerMixin):
    '''makes feature for total number of interventions'''
    def __init__(self,trans=True ):
        self.trans=trans


    def fit(self, X, y=None):
        
        return self

    def transform(self, X):
        if self.trans:
            X["number_interventions"]= X['num_lab_procedures']+X['num_procedures']+X['num_medications']

            return X
    
class MedHelpAdder(BaseEstimator, TransformerMixin):
    '''makes feature for total number of interaction with medical staff'''
    def __init__(self, trans=True):
        self.trans=trans


    def fit(self, X, y=None):
        
        return self

    def transform(self, X):
        if self.trans:
            X["number_med_encounters"]= X['time_in_hospital']+X['number_outpatient']+X['number_emergency']+X['number_inpatient']

            return X

class RobustNumFeatScaler(BaseEstimator, TransformerMixin):

    def __init__(self, trans=True):
        self.trans=trans


    def fit(self, X, y=None):
        
        return self

    def transform(self, X):
        if self.trans:
            num_features = ['time_in_hospital', 'num_lab_procedures',
                        'num_procedures', 'num_medications', 'number_outpatient',
                        'number_emergency', 'number_inpatient', 'number_diagnoses']
            X[num_features] = RobustScaler().fit_transform(X[num_features])

            return X
    