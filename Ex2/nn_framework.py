import time
import numpy as np
from MLP import MLP 
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import itertools
from sklearn.model_selection import cross_val_score
import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

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
    

    def cartesian_product_dict(self, **kwargs):
        """Generate a Cartesian product of parameter combinations.

        Args:
            kwargs: Dictionary of parameters and their possible values.

        Yields:
            dict: Dictionary representing a parameter combination.
        """
        param_names = kwargs.keys()
        for parameter_values in itertools.product(*kwargs.values()):
            yield dict(zip(param_names, parameter_values))

    def gridSearchIteration(self, parameters, X, y):
        """Perform a grid search iteration.

        Args:
            parameters (dict): Parameters for the MLP classifier.
            X (array-like): Input features.
            y (array-like): Target labels.

        Returns:
            float: Mean F1 score from cross-validation.
        """
        mlp = MLP()
        mlp.set_params(**parameters)
        scores = cross_val_score(mlp, X, y, cv=5, scoring='f1_macro', n_jobs=-1)
        return scores.mean()

    def gridSearchCV(self, X, y, param_grid):
        """Perform a grid search for hyperparameter tuning.

        Args:
            X (array-like): Input features.
            y (array-like): Target labels.
            param_grid (dict): Grid of hyperparameters to search.

        Returns:
            MLP: Fitted MLP classifier with the best parameters found.
        """
        param_combinations = list(self.cartesian_product_dict(**param_grid))
        max_score = -1
        for params in tqdm.tqdm(param_combinations):
            score = self.gridSearchIteration(params, X, y)
            if score > max_score:
                max_score = score
                best_params = params
        print(f"Best params found: {best_params}")

        mlp = MLP()
        mlp.set_params(**best_params)
        mlp.fit(X, y)
        return mlp


if __name__ == '__main__':
     
    start = time.time()
    X_main = np.random.random_sample((1000, 10))
    sum_X = X_main.sum(axis=1).astype(int)
    y_main = np.clip(sum_X - np.amin(sum_X), 0, 4)
    X_train, X_test, y_train, y_test = train_test_split(X_main, y_main)

    params = {"n_iter": [100, 200, 300],
            "learning_rate": [0.01, 0.001, 0.0001],
            "hidden_layer_sizes": [[15, 10], [30, 20], [50, 30, 10]],
            "activation_function": ['relu']}
    params = {"n_iter": [100],
            "learning_rate": [0.001, 0.0001],
            "hidden_layer_sizes": [[5, 10]],
            "activation_function": ['relu', 'sigmoid']}
    nnf = NNFramework()
    best_model = nnf.gridSearchCV(X_train, y_train, params)
    y_pred = best_model.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    print(time.time()-start)
