from unittest.mock import patch
import pandas as pd
import numpy as np
import utils
import preprocessing
from sklearn.preprocessing import LabelEncoder


def test_load_dataset():
    # Arrange
    params = utils.load_params('config/config.yaml')
    expected_columns = len(params['predictors']) + len(params['object_columns'])

    # Act
    train_set, valid_set, test_set = preprocessing.load_dataset(params)

    # Assert
    assert len(train_set) > 0, "Train set should not be empty"
    assert len(valid_set) > 0, "Validation set should not be empty"
    assert len(test_set) > 0, "Test set should not be empty"
    assert train_set.shape[1] == expected_columns, f"Train set should have {expected_columns} columns"
    assert valid_set.shape[1] == expected_columns, f"Validation set should have {expected_columns} columns"
    assert test_set.shape[1] == expected_columns, f"Test set should have {expected_columns} columns"

def test_ohe_transform():
    # Arrange
    params = utils.load_params('config/config.yaml')
    params['predictors_categorical'] = ['gender']

    mock_data = pd.DataFrame({'gender' : ['Female', 'Male', 'Female', 'Male']})
    expected_columns = ['gender_Male']
    expected_values = pd.DataFrame({'gender_Male': [0, 1, 0, 1]})

    # Act
    preprocessing.ohe_fit(params)
    transformed_data = preprocessing.ohe_transform(mock_data, params)

    # Assert
    for col in expected_columns:
        assert col in transformed_data, f"{col} not found in transformed data columns"

    pd.testing.assert_frame_equal(
        transformed_data[expected_columns].reset_index(drop=True), 
        expected_values, 
        check_dtype=False
    )

def test_resampling_data():
    # Arrange
    mock_data = pd.DataFrame({
        'feature' : np.random.randn(100),
        'churn' : [1] * 30 + [0] * 70
    })

    # Act and Assert
    resampled_data_rus = preprocessing.rus_resample(mock_data, 'churn')
    assert resampled_data_rus['churn'].value_counts().to_list() == [30, 30], "RandomUnderSampler did not balance classes as expected"

    resampled_data_ros = preprocessing.ros_resample(mock_data, 'churn')
    assert resampled_data_ros['churn'].value_counts().to_list() == [70, 70], "RandomOverSampler did not balance classes as expected"

    resampled_data_sm = preprocessing.sm_resample(mock_data, 'churn')
    assert resampled_data_sm['churn'].value_counts().to_list() == [70, 70], "SMOTE did not balance classes as expected"

def test_le_transform():
    # Arrange
    mock_data = pd.Series(['Yes', 'No', 'Yes', 'No', 'Yes'])
    params = {
        'range_churn': ['Yes', 'No'],
        'le_path' : 'le_encoder.pkl'
    }

    mock_le_encoder = LabelEncoder()
    mock_le_encoder.fit(mock_data)

    # Act
    with patch('utils.pickle_dump'), patch('utils.pickle_load') as mock_pickle_load:
        mock_pickle_load.return_value = mock_le_encoder
        preprocessing.le_fit(mock_data, None)
        transformed_data = preprocessing.le_transform(mock_data, params)

    # Assert
    assert np.issubdtype(transformed_data.dtype, np.integer), "Transformed data should be of integer type"
    assert np.array_equal(mock_le_encoder.inverse_transform([0, 1]), ['No', 'Yes']), "Label encoding does not match expected values"