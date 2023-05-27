from MLP import MLP 
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

class NNFramework:
    def __init__(self):
        self.encoder = OneHotEncoder(sparse_output=False, drop='if_binary')
        self.cols_to_encode = []
        self.encoder_fitted = False
    

    def fit_encoder(self, df, cols_to_encode):
        """
        Fits the encoder on the specified columns of the input dataframe.

        Parameters:
            df (pandas.DataFrame): The input dataframe to fit the encoder on.
            cols_to_encode (list or pandas.Index): The subset of columns to be encoded.

        Returns:
            None

        Note:
            - If cols_to_encode is empty or None, the encoder will be fitted on the entire input dataframe.
        """
        self.cols_to_encode = cols_to_encode.tolist() if isinstance(cols_to_encode, pd.Index) else cols_to_encode

        df_train = df.loc[:, self.cols_to_encode].copy() if self.cols_to_encode else df.copy()
        self.encoder = self.encoder.fit(df_train)

        self.encoder_fitted = True


    def encode_dataset(self, df):
        """
        Encodes the provided dataframe using the encoder fitted with fit_encoder().

        Parameters:
            df (pandas.DataFrame): The dataframe to be encoded.

        Returns:
            pandas.DataFrame: The encoded dataframe.

        Raises:
            RuntimeError: If the encoder was not fitted. Call fit_encoder() method before using encode_dataset().
        """
        if not self.encoder_fitted:
            raise RuntimeError('Cannot encode dataset because encoder was not fitted. ' +
                               'Call fit_encoder() method before using encode_dataset()')
        
        df_train = df.loc[:, self.cols_to_encode].copy() if self.cols_to_encode else df.copy()
        df_encoded = pd.DataFrame(self.encoder.transform(df_train),
                                    columns=self.encoder.get_feature_names_out(df_train.columns))
        df_encoded = pd.concat([df.drop(self.cols_to_encode, axis=1), df_encoded], axis=1)
        df_encoded.columns = [c.replace('_1', '') for c in df_encoded.columns]

        return df_encoded
    

    def find_optimal_params(self, model):
        raise NotImplemented
        return model
    