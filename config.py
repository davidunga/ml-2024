from copy import deepcopy
from hashlib import md5
import json

_config = {

    'data.sanity_mode': 'none',

    'cv.n_seeds': 5,
    'cv.n_folds': 5,
    'cv.scores': ['balanced_accuracy', 'roc_auc'],

    'balance.method': 'RandomUnderSampler',
    'balance.params': {'random_state': 1},

    'data.standardize': {'outlier_p': .01, 'offset': .5},

    'data.add_by_sum': {
        'num_visits': ['number_outpatient', 'number_inpatient', 'number_emergency'],
        'num_nonEm_visits': ['number_outpatient', 'number_inpatient']
    },

    'data.add_by_normalize': {
        'time_in_hospital': (['num_lab_procedures', 'num_procedures', 'num_medications'], 'perDay')
    },

    'data.exclude_rows_by_duplicates': ['patient_nbr'],

    'data.exclude_cols': ['weight', 'payer_code', 'encounter_id', 'patient_nbr'],
    'data.exclude_pregnancy_diabetes': False,
    'data.exclude_rows_where': {
        'discharge_disposition_id': [11, 13, 14, 19, 20, 21],
        'gender': ['Unknown/Invalid'],
    },

    'data.bias_thresh': 0.95,
    'data.rare_to_other_features': ['medical_specialty'],
    'data.rare_to_other_thresh': 0.01,

    'data.recategorize': {
        'readmitted': {
            'YES': ['<30'],
            'NO': ['>30', 'NO']
        },
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
