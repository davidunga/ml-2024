from copy import deepcopy
from hashlib import md5
import json

_config = {

    'cv.n_seeds': 5,
    'cv.n_folds': 5,
    'cv.scores': ['balanced_accuracy', 'roc_auc'],

    'data.exclude_cols': ['weight', 'payer_code', 'encounter_id', 'patient_nbr'],
    'data.exclude_rows_where': {
        'discharge_disposition_id': [11, 13, 14, 19, 20, 21],
        'gender': ['Unknown/Invalid'],
    },
    'data.bias_thresh': 0.95,
    'data.readmitted_groups': {'YES': ['<30'], 'NO': ['>30', 'NO']},
    'data.recategorize': {
        'age': {
            '<30': ['[0-10)', '[10-20)', '[20-30)'],
            '30-60': ['[30-40)', '[40-50)', '[50-60)'],
            '>60': ['[60-70)', '[70-80)', '[80-90)', '[90-100)']
        },
        'race': {
            'AfricanAmerican': ['AfricanAmerican'],
            'Caucasian': ['Caucasian']
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


def get_config() -> dict:
    return deepcopy(_config)


def get_config_id(config: dict) -> str:
    id_size = 6
    config_str = json.dumps(config)
    return md5(config_str.encode()).hexdigest()[:id_size]
