from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import utils
import json
from tqdm import tqdm

import copy
import hashlib
import pandas as pd

params_dir = 'config/config.yaml'

def load_rus_data(params: dict) -> pd.DataFrame:
    X_rus = utils.pickle_load(params['data_rus_path'][0])
    y_rus = utils.pickle_load(params['data_rus_path'][1])

    return X_rus, y_rus

def load_ros_data(params: dict) -> pd.DataFrame:
    X_ros = utils.pickle_load(params['data_ros_path'][0])
    y_ros = utils.pickle_load(params['data_ros_path'][1])

    return X_ros, y_ros

def load_sm_data(params: dict) -> pd.DataFrame:
    X_sm = utils.pickle_load(params['data_sm_path'][0])
    y_sm = utils.pickle_load(params['data_sm_path'][1])

    return X_sm, y_sm

def load_valid_set(params: dict) -> pd.DataFrame:
    X_valid = utils.pickle_load(params['valid_feng_set_path'][0])
    y_valid = utils.pickle_load(params['valid_feng_set_path'][1])

    return X_valid, y_valid

def load_test_set(params: dict) -> pd.DataFrame:
    X_test = utils.pickle_load(params['test_feng_set_path'][0])
    y_test = utils.pickle_load(params['test_feng_set_path'][1])

    return X_test, y_test

def load_dataset(config_file):
    utils.print_debug('Load data')

    X_rus, y_rus = load_rus_data(config_file)
    X_ros, y_ros = load_ros_data(config_file)
    X_sm, y_sm = load_sm_data(config_file)
    X_valid, y_valid = load_valid_set(config_file)
    X_test, y_test = load_test_set(config_file)

    utils.print_debug('All datasets successfully loaded')

    return X_rus, y_rus, X_ros, y_ros, X_sm, y_sm, X_valid, y_valid, X_test, y_test

def create_log_template():
    utils.print_debug('Create log template')

    logger = {
        'model_name' : [],
        'model_uid' : [],
        'training_time' : [],
        'training_date' : [],
        'performance' : [],
        'f1_score_avg' : [],
        'data_configuration' : []
    }

    utils.print_debug('Log template created')

    return logger

def training_log_updater(current_log, config_file):
    current_log = copy.deepcopy(current_log)
    log_path = config_file['training_log_path']

    try:
        with open(log_path, 'r') as file:
            last_log = json.load(file)
            
    except FileNotFoundError as ffe:
        with open(log_path, 'w') as file:
            file.write('[]')

        with open(log_path, 'r') as file:
            last_log = json.load(file)

    last_log.append(current_log)

    with open(log_path, 'w') as file:
        json.dump(last_log, file)

    return last_log

def create_model_object(config_file: dict) -> list:
    utils.print_debug('Create model object')

    logreg_baseline = LogisticRegression()
    rf_baseline = RandomForestClassifier()
    dc_baseline = DecisionTreeClassifier()
    knn_baseline = KNeighborsClassifier()
    xgb_baseline = XGBClassifier()

    list_of_model = {
        'undersampling' : [
            {'model_name' : logreg_baseline.__class__.__name__, 'model_object' : logreg_baseline, 'model_uid' : ''},
            {'model_name' : rf_baseline.__class__.__name__, 'model_object' : rf_baseline, 'model_uid' : ''},
            {'model_name' : dc_baseline.__class__.__name__, 'model_object' : dc_baseline, 'model_uid' : ''},
            {'model_name' : knn_baseline.__class__.__name__, 'model_object' : knn_baseline, 'model_uid' : ''},
            {'model_name' : xgb_baseline.__class__.__name__, 'model_object' : xgb_baseline, 'model_uid' : ''}
        ],
        'oversampling' : [
            {'model_name' : logreg_baseline.__class__.__name__, 'model_object' : logreg_baseline, 'model_uid' : ''},
            {'model_name' : rf_baseline.__class__.__name__, 'model_object' : rf_baseline, 'model_uid' : ''},
            {'model_name' : dc_baseline.__class__.__name__, 'model_object' : dc_baseline, 'model_uid' : ''},
            {'model_name' : knn_baseline.__class__.__name__, 'model_object' : knn_baseline, 'model_uid' : ''},
            {'model_name' : xgb_baseline.__class__.__name__, 'model_object' : xgb_baseline, 'model_uid' : ''}
        ],
        'smote' : [
            {'model_name' : logreg_baseline.__class__.__name__, 'model_object' : logreg_baseline, 'model_uid' : ''},
            {'model_name' : rf_baseline.__class__.__name__, 'model_object' : rf_baseline, 'model_uid' : ''},
            {'model_name' : dc_baseline.__class__.__name__, 'model_object' : dc_baseline, 'model_uid' : ''},
            {'model_name' : knn_baseline.__class__.__name__, 'model_object' : knn_baseline, 'model_uid' : ''},
            {'model_name' : xgb_baseline.__class__.__name__, 'model_object' : xgb_baseline, 'model_uid' : ''}
        ]
    }

    utils.print_debug('Model object created')

    return list_of_model

def train_eval_model(config_file: dict, model_prefix):
    X_rus, y_rus, \
    X_ros, y_ros, \
    X_sm, y_sm, \
    X_valid, y_valid, \
    X_test, y_test = load_dataset(config_file)

    training_data_config = {
        'undersampling': (X_rus, y_rus),
        'oversampling': (X_ros, y_ros),
        'smote': (X_sm, y_sm)
    }

    list_of_model = create_model_object(config_file)
    list_of_model = copy.deepcopy(list_of_model)
    training_log = create_log_template()

    for config_name, (X_train, y_train) in training_data_config.items():
        utils.print_debug(f'Training with configuration: {config_name}')

        models_for_config = list_of_model[config_name]

        for model in tqdm(models_for_config, desc=f'Training Models ({config_name})'):
            model_name = model_prefix + '-' + model['model_name']
            utils.print_debug(f'Starting training for model: {model_name}')
            
            start_time = utils.time_stamp()
            model['model_object'].fit(X_train, y_train)
            finished_time = utils.time_stamp()

            elapsed_time = finished_time - start_time
            elapsed_time = elapsed_time.total_seconds()
            utils.print_debug(f'Training completed in {elapsed_time:.2f} seconds for model: {model_name}')

            y_pred = model['model_object'].predict(X_valid)
            performance = classification_report(y_valid, y_pred, output_dict=True)

            plain_id = str(start_time) + str(finished_time)
            chiper_id = hashlib.md5(plain_id.encode()).hexdigest()

            model['model_uid'] = chiper_id

            training_log['model_name'].append(model_name)
            training_log['model_uid'].append(chiper_id)
            training_log['training_time'].append(elapsed_time)
            training_log['training_date'].append(str(start_time))
            training_log['performance'].append(performance)
            training_log['f1_score_avg'].append(performance['macro avg']['f1-score'])
            training_log['data_configuration'].append(config_name)

            utils.print_debug(f"Model: {model_name} | Configuration: {config_name} | F1 Score: {performance['macro avg']['f1-score']:.4f}")

    training_log = training_log_updater(training_log, config_file)
    utils.print_debug(f'Training and evaluation completed for all models')

    return training_log, list_of_model

def training_log_to_df(training_log):
    utils.print_debug('Convert training logs into DataFrame...')

    training_res = pd.DataFrame()

    for log in tqdm(training_log, desc="Converting log to DataFrame"):
        df_log = pd.DataFrame(log)

        if not df_log.empty:
            training_res = pd.concat([training_res, df_log], ignore_index=True)

    training_res = training_res.sort_values(['f1_score_avg', 'training_time'], ascending=[False, True])
    training_res = training_res.reset_index(drop=True)

    utils.print_debug('Log successfully converted to DataFrame')

    return training_res

def get_best_model(training_log_df, list_of_model, config_file):
    try:
        if training_log_df is None or len(training_log_df) == 0:
            raise ValueError('DataFrame training log is empty or invalid')
        
        utils.print_debug("Searching the model based on metrics...")

        best_model_info = training_log_df.sort_values([
            'f1_score_avg', 'training_time'], ascending=[False, True]
        ).iloc[0]
        utils.print_debug(f"The best model: {best_model_info['model_name']} with avg f1 score {best_model_info['f1_score_avg']}")

        model_object = None

        for models in list_of_model.values():
            for model_data in models:
                if model_data["model_uid"] == best_model_info["model_uid"]:
                    model_object = model_data["model_object"]
                    utils.print_debug(f"Model object found for {best_model_info['model_uid']}.")
                    break

            if model_object is not None:
                break

        if model_object is None:
            raise RuntimeError("Best model not found in the list of models.")

        result = {
            "model_data": {
                "model_name": model_object.__class__.__name__,
                "model_object": model_object,
                "model_uid": best_model_info["model_uid"],
            },
            "model_log": {
                "model_name": best_model_info["model_name"],
                "model_uid": best_model_info["model_uid"],
                "training_time": best_model_info["training_time"],
                "training_date": best_model_info["training_date"],
                "performance": best_model_info["performance"],
                "f1_score_avg": best_model_info["f1_score_avg"],
                "data_configuration": best_model_info["data_configuration"],
            }
        }
        
        utils.print_debug("Best model successfully retrieved.")

        # Simpan model terbaik menggunakan joblib
        utils.pickle_dump(model_object, config_file['production_model_path'])
        utils.print_debug(f"Best model successfully saved at {config_file['production_model_path']}")
        
        return result
    
    except Exception as e:
        print(f"Error occurred while retrieving the best model: {e}")
        return None

def get_hyperparameters(model_name):
    dist_params = {
        'LogisticRegression': {
            'penalty': ['l1', 'l2', 'none'],
            'C': [0.1, 1, 10],
            'solver': ['lbfgs', 'liblinear']
        },
        'KNeighborsClassifier': {
            'n_neighbors': [3, 5, 7],
            'leaf_size': [10, 20]
        },
        'DecisionTreeClassifier': {
            'criterion': ['gini', 'entropy'],
            'splitter': ['best', 'random'],
            'max_depth': [10, 20, 30]
        },
        'RandomForestClassifier': {
            'n_estimators' : [50, 100],
            'max_depth' : [10, 20],
            'min_samples_split' : [5, 10]
        },
        'XGBClassifier': {
            'n_estimators': [50, 100],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5]
        }
    }

    if model_name in dist_params:
        return dist_params[model_name]
    else:
        raise ValueError(f'Parameter distribution not defined for model {model_name}')
    
def tune_best_model(best_model, config_file):
    utils.print_debug('Start tuning best model')

    model_name = best_model['model_data']['model_name']
    param_grid = get_hyperparameters(model_name)

    data_config = best_model['model_log']['data_configuration']
    if data_config == 'undersampling':
        X_train = utils.pickle_load(config_file['data_rus_path'][0])
        y_train = utils.pickle_load(config_file['data_rus_path'][1])
    elif data_config == 'oversampling':
        X_train = utils.pickle_load(config_file['data_ros_path'][0])
        y_train = utils.pickle_load(config_file['data_ros_path'][1])
    elif data_config == 'smote':
        X_train = utils.pickle_load(config_file['data_sm_path'][0])
        y_train = utils.pickle_load(config_file['data_sm_path'][1])
    else:
        raise ValueError('Data configuration not found')
    
    grid_search = GridSearchCV(
        best_model['model_data']['model_object'],
        param_grid = param_grid,
        cv = 5,
        n_jobs = -1,
        verbose = 420
    )

    start_time = utils.time_stamp()
    utils.print_debug('Fitting grid search...')
    grid_search.fit(X_train, y_train)
    finished_time = utils.time_stamp()
    
    elapsed_time = (finished_time - start_time).total_seconds()
    chiper_id = hashlib.md5((str(start_time) + str(finished_time)).encode()).hexdigest()

    best_estimator = grid_search.best_estimator_
    best_f1_score = grid_search.best_score_
    performance = best_estimator.score(X_train, y_train)

    utils.print_debug(f'Best F1 score: {best_f1_score}, Model UID: {chiper_id}')

    tuned_log_entry = {
        'model_name': model_name,
        'model_uid': chiper_id,
        'training_time': elapsed_time,
        'training_date': str(start_time),
        'performance': grid_search.best_params_,
        'f1_score_avg': best_f1_score,
        'data_configuration': data_config
    }

    training_log_updater(tuned_log_entry, config_file)
    utils.print_debug(f'Best tuned model updated in training log with UID {chiper_id}')

    return best_estimator, performance

if __name__ == '__main__':
    params = utils.load_params(params_dir)

    X_rus, y_rus, \
    X_ros, y_ros, \
    X_sm, y_sm, \
    X_valid, y_valid, \
    X_test, y_test = load_dataset(params)

    training_log, list_of_model = train_eval_model(config_file= params, model_prefix='Baseline')

    training_res = training_log_to_df(training_log)

    model = get_best_model(training_res, list_of_model, params)

    best_estimator, performance = tune_best_model(model, params)

    y_valid_pred = best_estimator.predict(X_valid)
    print('Classification Report Valid Set')
    print(classification_report(y_true = y_valid, y_pred = y_valid_pred, labels = [1,0]))

    y_test_pred = best_estimator.predict(X_test)
    print('Classification Report Test Set')
    print(classification_report(y_true = y_test, y_pred = y_test_pred, labels = [1,0]))

    utils.pickle_dump(best_estimator, params['production_model_path'])