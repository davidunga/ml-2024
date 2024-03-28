from copy import deepcopy
from hashlib import md5
import json
import numpy as np
from typing import Dict, List, Tuple

FROM_CONFIG = '_from_config_'

_default_config = {

    'target_col': 'readmitted',
    'diagnosis_cols': ['diag_1', 'diag_2', 'diag_3'],
    'numeric_cols': ['time_in_hospital', 'num_lab_procedures', 'num_procedures',
                     'num_medications', 'number_outpatient', 'number_emergency',
                     'number_inpatient', 'number_diagnoses', 'weight'],

    'random_state': 1337,
    'finetune': False,

    'data.sanity_mode': 'none',
    'data.test_size': .2,

    'cv.n_folds': 5,
    'cv.scores': ['balanced_accuracy', 'roc_auc', 'f1', 'precision', 'recall'],
    'cv.main_score': 'roc_auc',

    'cv.base.val_size': .2,
    'cv.base.early_stopping_rounds': 5,

    'balance': {
        'method': 'RandomUnderSampler',
        'params': {'random_state': FROM_CONFIG}
    },

    'data.standardize': {
        'default_transform': 'sqrt',
        'feature_transforms': {},
        'outlier_p': 0,
        'offset': .5
    },

    'data.add_features.by_sum': {
        'num_visits': ['number_outpatient', 'number_inpatient', 'number_emergency'],
        'num_nonEm_visits': ['number_outpatient', 'number_inpatient']
    },

    'data.add_features.by_count': [
        {
            "values_to_count": ["No"],
            "invert": True,
            "drop_originals": True,
            "mapping": {
                "biguanides_and_related": [
                    "metformin", "glyburide-metformin", "glimepiride-pioglitazone",
                    "metformin-rosiglitazone"],
                "sulfonylureas_and_meglitinides": [
                    "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
                    "acetohexamide", "glipizide", "glyburide", "tolbutamide"],
                "thiazolidinediones_and_miscellaneous": [
                    "pioglitazone", "rosiglitazone", "troglitazone", "acarbose",
                    "miglitol", "tolazamide", "examide", "citoglipton",
                    "glipizide-metformin", "metformin-pioglitazone"]
            }
        }
    ],

    'data.add_features.by_normalize': {
        'time_in_hospital': (['num_lab_procedures', 'num_procedures', 'num_medications'], 'perDay')
    },

    'data.add_features.construct': {
        'average_age': True,
        'encounter': True
    },

    'data.exclude_features.by_name': ['weight', 'payer_code', 'encounter_id',
                                      'patient_nbr', 'A1Cresult', 'change'],

    'data.exclude_rows.pregnancy_diabetes': False,
    'data.exclude_rows.duplicate': ['patient_nbr'],
    'data.exclude_rows.where': {
        'discharge_disposition_id': [11, 13, 14, 19, 20, 21],
        'gender': ['Unknown/Invalid'],
    },

    'data.categories.group_others': {
        'medical_specialty': ['InternalMedicine', 'Emergency/Trauma', 'Family/GeneralPractice',
                              'Cardiology', 'Surgery-General', 'Nephrology', 'Orthopedics'],
        'race': ['AfricanAmerican', 'Caucasian']
    },

    'data.categories.reduce': {
        'readmitted': {
            'YES': ['<30'],
            'NO': ['>30', 'NO']
        },
        'age': {
            '<30': ['[0-10)', '[10-20)', '[20-30)'],
            '30-60': ['[30-40)', '[40-50)', '[50-60)'],
            '>60': ['[60-70)', '[70-80)', '[80-90)', '[90-100)']
        },
        'admission_type_id': {
            'HighPriority': [1, 2],
            'ClinicReferral': [3, 4, 7]
        },
        'discharge_disposition_id': {
            'Home': [1]
        }
    }
}


def get_default_config() -> Dict:
    return deepcopy(_default_config)


def get_config_name(config: Dict) -> str:
    hash_ = md5(json.dumps(config).encode()).hexdigest()[:4]
    name = f"{config['balance']['method']} seed{config['random_state']} {hash_}"
    if config['finetune']:
        name += " TUNED"
    return name


def inherit_from_config(d: Dict, config: Dict) -> Dict:
    """ recursively replace <FROM_CONFIG> entries with corresponding config entries """
    def _get(v):
        if isinstance(v, Dict): return inherit_from_config(v, config)
        if isinstance(v, List): return [_get(vv) for vv in v]
        return v
    return {k: config[k] if type(v) == type(FROM_CONFIG) and v == FROM_CONFIG else _get(v)
            for k, v in d.items()}
