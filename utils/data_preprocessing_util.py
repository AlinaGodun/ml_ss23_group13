from sklearn.preprocessing import (
    Normalizer,
    MinMaxScaler,
    MaxAbsScaler
)
import pandas as pd

def normalize_features(df, cols, scaler_name='min-max', scaler=None, scaler_params={}):
    """
    Normalize selected features in a pandas DataFrame using a specified scaler.

    Args:
        df (pandas.DataFrame): The DataFrame containing the features to be normalized.
        cols (list): A list of column names to be normalized.
        scaler_name (str, optional): The name of the scaler to be used. Default is 'min-max' (for scikit MinMaxScaler). Other available options: 'max-abs' (for MaxAbsScaler), 'norm' (for Nornalizer)
        scaler (sklearn.preprocessing object, optional): An instance of a scaler object to be used for normalization. 
            If not provided, a scaler object will be created based on the scaler_name and scaler_params arguments.
        scaler_params (dict, optional): A dictionary of parameters to be passed to the scaler object when created. 
            Default is an empty dictionary.

    Returns:
        pandas.DataFrame: A copy of the input DataFrame with the specified columns normalized using the specified scaler.
        sklearn.preprocessing object: The scaler object used for normalization.

    Raises:
        KeyError: If an incorrect scaler name is provided. Possible scaler name options: 'norm', 'min-max', 'max-abs'.
    """
    scalers = {
        'norm': Normalizer,
        'min-max': MinMaxScaler,
        'max-abs': MaxAbsScaler
    }
    
    df = df.copy()

    if scaler is None:
        if scaler_name not in scalers.keys():
            raise KeyError(f'Incorrect scaler name provided. Possible scaler name options: norm, min-max, max-abs')
        scaler = scalers[scaler_name](*scaler_params)

    df.loc[:, cols] = scaler.fit_transform(df[cols])
    return df, scaler


def encode_label(df):
    df = df.copy()
    df = df.drop('season', axis=1).join(pd.get_dummies(df.season))
    df['high_fever'].replace({'<3month': 2, '>3month': 1, 'no': 0}, inplace=True)
    df['smoking'].replace({'never': 0, 'occasional': 1, 'daily': 2}, inplace=True)
    df['alcohol_consumption'].replace({'several times/day': 4,
                                    'daily': 3,
                                    'several times/week': 2,
                                    'once/week': 1,
                                    'hardly ever/never': 0}, inplace=True)
    return df


def encode_one_hot(df):
    df = df.copy()
    one_hot_cols = ['season', 'high_fever', 'smoking', 'alcohol_consumption']

    for ohc in one_hot_cols:
        df = df.drop(ohc, axis=1).join(pd.get_dummies(df[ohc], prefix=ohc))
    
    return df


def encode(df, encoding='label'):
    available_encodings = ['label', 'one-hot']

    if encoding == 'label':
        return encode_label(df)
    elif encoding == 'one-hot':
        return encode_one_hot(df)
    else:
        raise KeyError(f'Wrong encoding param provided. Available options: {available_encodings}')