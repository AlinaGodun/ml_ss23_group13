from sklearn.preprocessing import (
    Normalizer,
    MinMaxScaler,
    MaxAbsScaler
)

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