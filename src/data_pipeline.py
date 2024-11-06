import utils
import pandas as pd
from sklearn.model_selection import train_test_split

params_dir = 'config/config.yaml'

def load_data(dataset_dir):
    try:
        dataset = pd.read_csv(dataset_dir)
        return dataset
    
    except FileNotFoundError:
        print(f"File {dataset_dir} tidak ditemukan.")

    except pd.errors.EmptyDataError:
        print(f"File {dataset_dir} kosong.")

    except pd.errors.ParserError:
        print(f"Terdapat error saat parsing file {dataset_dir}.")
        
    except Exception as e:
        print(f"Terjadi error: {e}")

def drop_unnecessary_col(df, col_to_drop):
    df = df.drop(columns = col_to_drop)

    return df

def check_data(input_data, params, is_api_call=False):
    # measure the range of input data
    len_input_data = len(input_data)
    
    if is_api_call:
        object_columns = [col for col in params['object_columns'] if col != 'churn']
    else:
        object_columns = params['object_columns']

    # check data types
    assert input_data.select_dtypes('object').columns.to_list() == object_columns, 'an error occurs in object column(s)'
    assert input_data.select_dtypes('float').columns.to_list() == params['float_columns'], 'an error occurs in float column(s)'
    assert input_data.select_dtypes('int').columns.to_list() == params['int_columns'], 'an error occurs in integer column(s)'

    # check range of data
    assert input_data[params['float_columns'][0]].between(params['range_monthly_charges'][0], params['range_monthly_charges'][1]).sum() == len_input_data, 'an error occurs in monthly charges range'
    assert input_data[params['float_columns'][1]].between(params['range_total_charges'][0], params['range_total_charges'][1]).sum() == len_input_data, 'an error occurs in total charges range'
    assert input_data[params['int_columns'][0]].between(params['range_tenure_months'][0], params['range_tenure_months'][1]).sum() == len_input_data, 'an error occurs in tenure months range'
    assert set(input_data[params['object_columns'][0]]).issubset(set(params['range_gender'])), 'an error occurs in gender range'
    assert set(input_data[params['object_columns'][1]]).issubset(set(params['range_senior_citizen'])), 'an error occurs in senior citizen range'
    assert set(input_data[params['object_columns'][2]]).issubset(set(params['range_partner'])), 'an error occurs in partner range'
    assert set(input_data[params['object_columns'][3]]).issubset(set(params['range_dependents'])), 'an error occurs in dependents range'
    assert set(input_data[params['object_columns'][4]]).issubset(set(params['range_phone_service'])), 'an error occurs in phone service range'
    assert set(input_data[params['object_columns'][5]]).issubset(set(params['range_multiple_lines'])), 'an error occurs in multiple lines range'
    assert set(input_data[params['object_columns'][6]]).issubset(set(params['range_internet_service'])), 'an error occurs in internet service range'
    assert set(input_data[params['object_columns'][7]]).issubset(set(params['range_online_security'])), 'an error occurs in online security range'
    assert set(input_data[params['object_columns'][8]]).issubset(set(params['range_online_backup'])), 'an error occurs in online backup range'
    assert set(input_data[params['object_columns'][9]]).issubset(set(params['range_device_protection'])), 'an error occurs in device protection range'
    assert set(input_data[params['object_columns'][10]]).issubset(set(params['range_tech_support'])), 'an error occurs in tech support range'
    assert set(input_data[params['object_columns'][11]]).issubset(set(params['range_streaming_tv'])), 'an error occurs in streaming tv range'
    assert set(input_data[params['object_columns'][12]]).issubset(set(params['range_streaming_movies'])), 'an error occurs in streaming movies range'
    assert set(input_data[params['object_columns'][13]]).issubset(set(params['range_contract'])), 'an error occurs in contract range'
    assert set(input_data[params['object_columns'][14]]).issubset(set(params['range_paperless_billing'])), 'an error occurs in paperless billing range'
    assert set(input_data[params['object_columns'][15]]).issubset(set(params['range_payment_method'])), 'an error occurs in payment method range'

def split_input_output(data, target_column):
    X = data.drop(columns = target_column)
    y = data[target_column]

    return X, y

if __name__ == '__main__':
    params = utils.load_params(params_dir)

    df = load_data(params['dataset_path'])
    df = drop_unnecessary_col(df, 'customerID')

    mapping_citizen = {0 : 'No', 1 : 'Yes'}
    df['SeniorCitizen'] = df['SeniorCitizen'].map(mapping_citizen)

    mapping_payment = {
        'Electronic check' : 'E-Check',
        'Mailed check' : 'M-Check',
        'Bank transfer (automatic)' : 'Auto Bank',
        'Credit card (automatic)' : 'Auto Card'
    }
    df['PaymentMethod'] = df['PaymentMethod'].map(mapping_payment)

    df['TotalCharges'] = df['TotalCharges'].replace(' ', 0).astype('float')

    mapping_column_name = {
    'SeniorCitizen' : 'senior_citizen',
    'Partner' : 'partner',
    'Dependents' : 'dependents',
    'tenure' : 'tenure_months',
    'PhoneService' : 'phone_service',
    'MultipleLines' : 'multiple_lines',
    'InternetService' : 'internet_service',
    'OnlineSecurity' : 'online_security',
    'OnlineBackup' : 'online_backup',
    'DeviceProtection' : 'device_protection',
    'TechSupport' : 'tech_support',
    'StreamingTV' : 'streaming_tv',
    'StreamingMovies' : 'streaming_movies',
    'Contract' : 'contract',
    'PaperlessBilling' : 'paperless_billing',
    'PaymentMethod' : 'payment_method',
    'MonthlyCharges' : 'monthly_charges',
    'TotalCharges' : 'total_charges',
    'Churn' : 'churn'
    }
    df = df.rename(columns = mapping_column_name)

    check_data(input_data=df, params=params)

    X, y = split_input_output(data = df, target_column = 'churn')

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size = params['test_size'],
        random_state = 42,
        stratify = y
    )

    X_valid, X_test, y_valid, y_test = train_test_split(
        X_test,
        y_test,
        test_size = params['valid_size'],
        random_state = 42,
        stratify = y_test
    )

    utils.pickle_dump(df, params['dataset_cleaned_path'])

    utils.pickle_dump(X_train, params['train_set_path'][0])
    utils.pickle_dump(y_train, params['train_set_path'][1])

    utils.pickle_dump(X_test, params['test_set_path'][0])
    utils.pickle_dump(y_test, params['test_set_path'][1])

    utils.pickle_dump(X_valid, params['valid_set_path'][0])
    utils.pickle_dump(y_valid, params['valid_set_path'][1])