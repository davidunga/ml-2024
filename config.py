from copy import deepcopy
from hashlib import md5
import json
import numpy as np
from typing import Dict, List, Tuple

_config = {

    'target_col': 'readmitted',
    'diagnosis_cols': ['diag_1', 'diag_2', 'diag_3'],

    'random_state': 1337,

    'estimator.name': '',
    'estimator.params': {},

    'data.sanity_mode': 'none',
    'data.test_size': .2,

    'cv.n_folds': 5,
    'cv.scores': ['balanced_accuracy', 'roc_auc', 'f1', 'precision', 'recall'],
    'cv.main_score': 'roc_auc',

    'cv.base.val_size': .2,
    'cv.base.early_stopping_rounds': 5,

    'balance.method': 'RandomUnderSampler',
    'balance.params': {'random_state': 1337},

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

    'data.bias_thresh': 0.95,

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


def get_config() -> Dict:
    return deepcopy(_config)


def update_estimator_in_config(config: Dict, name: str, params: Dict) -> Dict:
    config = deepcopy(config)
    config['estimator.name'] = name
    config['estimator.params'] = {k: int(v) if isinstance(v, np.integer) else v
                                  for k, v in params.items()}
    return config


def get_config_name(config: Dict) -> str:
    config = deepcopy(config)

    def _get_hash(obj) -> str:
        return md5(json.dumps(obj).encode()).hexdigest()[:4]

    params_hash = _get_hash(config['estimator.params'])
    total_hash = _get_hash(config)

    estimator_name = 'x' if not config['estimator.name'] else config['estimator.name']

    name = f"{estimator_name} {config['balance.method']} seed{config['random_state']}"
    name += f" params{params_hash} {total_hash}"

    return name
