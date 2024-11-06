import utils
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE

params_dir = 'config/config.yaml'

def load_dataset(config_file: dict):
    X_train = utils.pickle_load(config_file['train_set_path'][0])
    y_train = utils.pickle_load(config_file['train_set_path'][1])

    X_valid = utils.pickle_load(config_file['valid_set_path'][0])
    y_valid = utils.pickle_load(config_file['valid_set_path'][1])

    X_test = utils.pickle_load(config_file['test_set_path'][0])
    y_test = utils.pickle_load(config_file['test_set_path'][1])

    train_set = pd.concat([X_train, y_train], axis=1)
    valid_set = pd.concat([X_valid, y_valid], axis=1)
    test_set = pd.concat([X_test, y_test], axis=1)

    return train_set, valid_set, test_set

def ohe_fit(config_file):
    for col in config_file['predictors_categorical']:
        ohe = OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)
        ohe.fit(np.array(config_file['range_' + col]).reshape(-1,1))
        utils.pickle_dump(ohe, config_file['ohe_' + col + '_path'])

def ohe_transform(data, config_file):
    data_copy = data.copy()

    for col in config_file['predictors_categorical']:
        ohe = utils.pickle_load(config_file['ohe_' + col + '_path'])
        ohe_features = ohe.transform(np.array(data_copy[col].to_list()).reshape(-1,1))

        column_name = ohe.get_feature_names_out([col])
        ohe_features = pd.DataFrame(ohe_features, columns=column_name)

        ohe_features.set_index(data_copy.index, inplace = True)
        data_copy = pd.concat([ohe_features, data_copy], axis=1)
        data_copy.drop(columns=col, inplace=True)

    return data_copy

def rus_resample(data, col_to_drop):
    data_copy = data.copy()

    rus = RandomUnderSampler(random_state=42)
    X_rus, y_rus = rus.fit_resample(data_copy.drop(col_to_drop, axis=1), data_copy[col_to_drop])
    data_rus = pd.concat([X_rus, y_rus], axis=1)

    return data_rus

def ros_resample(data, col_to_drop):
    data_copy = data.copy()

    ros = RandomOverSampler(random_state=11)
    X_ros, y_ros = ros.fit_resample(data_copy.drop(col_to_drop, axis=1), data_copy[col_to_drop])
    data_ros = pd.concat([X_ros, y_ros], axis=1)

    return data_ros

def sm_resample(data, col_to_drop):
    data_copy = data.copy()

    sm = SMOTE(random_state=42)
    X_sm, y_sm = sm.fit_resample(data_copy.drop(col_to_drop, axis=1), data_copy[col_to_drop])
    data_sm = pd.concat([X_sm, y_sm], axis=1)

    return data_sm

def le_fit(data, le_path):
    le_encoder = LabelEncoder()
    le_encoder.fit(data)
    utils.pickle_dump(le_encoder, le_path)

    return le_encoder

def le_transform(data, config_file):
    data_copy = data.copy()
    le_encoder = utils.pickle_load(config_file['le_path'])
    
    if len(set(data_copy.unique()) - set(le_encoder.classes_) | set(le_encoder.classes_) - set(data_copy.unique())) == 0:
        data_copy = le_encoder.transform(data_copy)
    else:
        raise RuntimeError('Check category in label data and label encoder')

    return data_copy

if __name__ == '__main__':
    params = utils.load_params(params_dir)
    train_set, valid_set, test_set = load_dataset(config_file=params)

    ohe_fit(config_file=params)
    train_set = ohe_transform(data=train_set, config_file=params)
    valid_set = ohe_transform(data=valid_set, config_file=params)
    test_set = ohe_transform(data=test_set, config_file=params)

    data_rus = rus_resample(data=train_set, col_to_drop='churn')
    data_ros = ros_resample(data=train_set, col_to_drop='churn')
    data_sm = sm_resample(data=train_set, col_to_drop='churn')

    le_encoder = le_fit(data=params['range_churn'], le_path=params['le_path'])
    data_rus['churn'] = le_transform(data=data_rus['churn'], config_file=params)
    data_ros['churn'] = le_transform(data=data_ros['churn'], config_file=params)
    data_sm['churn'] = le_transform(data=data_sm['churn'], config_file=params)
    valid_set['churn'] = le_transform(data=valid_set['churn'], config_file=params)
    test_set['churn'] = le_transform(data=test_set['churn'], config_file=params)

    utils.pickle_dump(data_rus.drop(columns='churn'), params['data_rus_path'][0])
    utils.pickle_dump(data_rus['churn'], params['data_rus_path'][1])

    utils.pickle_dump(data_ros.drop(columns='churn'), params['data_ros_path'][0])
    utils.pickle_dump(data_ros['churn'], params['data_ros_path'][1])

    utils.pickle_dump(data_sm.drop(columns='churn'), params['data_sm_path'][0])
    utils.pickle_dump(data_sm['churn'], params['data_sm_path'][1])

    utils.pickle_dump(valid_set.drop(columns='churn'), params['valid_feng_set_path'][0])
    utils.pickle_dump(valid_set['churn'], params['valid_feng_set_path'][1])

    utils.pickle_dump(test_set.drop(columns='churn'), params['test_feng_set_path'][0])
    utils.pickle_dump(test_set['churn'], params['test_feng_set_path'][1])