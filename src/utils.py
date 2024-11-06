import yaml
import joblib
from datetime import datetime

params_dir = 'config/config.yaml'
def load_params(params_dir) -> dict:
    print(f"Open configuration file in: {params_dir}")
    try:
        with open(params_dir, 'r') as file:
            params = yaml.safe_load(file)
    except FileNotFoundError:
        raise RuntimeError(f'Parameters file not found in path: {params_dir}')
    except Exception as e:
        raise RuntimeError(f'Error opening file: {e}')

    return params

def pickle_load(file_path: str):
    return joblib.load(file_path)

def pickle_dump(data, file_path: str):
    return joblib.dump(data, file_path)

def time_stamp():
    return datetime.now()

params = load_params(params_dir)
PRINT_DEBUG = params['print_debug']

def print_debug(message: str):
    if PRINT_DEBUG == True:
        print(time_stamp(), message)