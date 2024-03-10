
import pandas as pd
import numpy as np
from paths import DATA_PATH
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin

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
        #make sure icd9 is correct!!
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
            X["num_lab_procedures_p_day"]= X['num_lab_procedures']/X["time_in_hospital"]
            X["num_procedures_p_day"]=X['num_procedures']/X["time_in_hospital"]
            X["num_medications_p_day"]=X['num_medications']/X["time_in_hospital"]
            X["num_interventions_p_day"]=X["num_medications_p_day"]+X["num_procedures_p_day"]+X["num_lab_procedures_p_day"]

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


class RemovePregenant(BaseEstimator, TransformerMixin):
    '''remove pregenancy related cases'''
    def __init__(self, trans=True):
        self.trans=trans


    def fit(self, X, y=None):
        
        return self

    def transform(self, X):
        if self.trans:
            #make sure icd9 is correct!!
            X["preg_diag"]= X[["diag_1", "diag_2", "diag_3"]].apply(lambda row: (11) in row.values, axis=1)

            return X[X.preg_diag!=True]

class RemoveKids(BaseEstimator, TransformerMixin):
    '''Removes kids from the dataset'''
    def __init__(self, trans=True,min_age_thresh=18):
        self.trans=trans
        self.min_age_thresh=min_age_thresh
        


    def fit(self, X, y=None):
        
        return self

    def transform(self, X):
        if self.trans:
            return X[X.age_mean>min_age_thresh]
