from math import log
from abc import ABC, abstractmethod
import itertools
import numpy as np
import time
from sklearn.utils import check_array, check_random_state, check_X_y
from sklearn.utils.estimator_checks import check_estimator
from sklearn.exceptions import NotFittedError
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neural_network import MLPClassifier as skMLP
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from tqdm import tqdm

class ActivationFunction(ABC):
    @abstractmethod
    def function(self, X):
        return NotImplemented
    
    @abstractmethod
    def derivative(self, X):
        return NotImplemented
    
class ReluActivation(ActivationFunction):
    def function(self, X):
        return np.maximum(0, X)
    
    def derivative(self, X):
        return (X > 0).astype(int)

class Sigmoid(ActivationFunction):
    def function(self, X):
        return 1 / (1+np.exp(-X))
    
    def derivative(self, X):
        sigmoid = self.function(X)
        return sigmoid * (1 - sigmoid)

def softmax(X):
    X = X - np.amax(X)
    return (np.exp(X)/np.sum(np.exp(X), axis=1)[:, None])

class MLP:
    afs = {
        'relu': ReluActivation,
        'sigmoid': Sigmoid}
    
    _estimator_type = "classifier"
    _is_fitted = False
 
    def __init__(self, hidden_layer_sizes=(100,), activation_function='relu', learning_rate=0.01, n_iter=100, seed=1111):
        self.check_params(activation_function, learning_rate, n_iter)
        
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation_function = activation_function
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.seed = seed


    def check_params(self, activation_function, learning_rate, n_iter):
        if activation_function not in self.afs:
            raise KeyError(f'Invalid activation function provided: {self.activation_function}. ' +
                           f'Available activation functions: {self.afs}')
        if learning_rate < 0.0:
            raise ValueError(f'Invalid learning rate provided: {self.learning_rate}. ' +
                              'Learning rate must be positive.')
        
        if n_iter < 0:
            raise ValueError(f'Invalid number of iterations provided: {self.n_iter}. ' +
                              'Number of iterations must be positive.')

    def cross_entropy(self, y, y_pred, weights, bias):
        #TODO regularisation with weights + bias
        loss = 0
        for idx, class_label in zip(range(y_pred.shape[1]), self.classes_):
            y_prob = y_pred[:, idx]
            loss -= (class_label == y)*log(y_prob)
        return loss

    def check_regression_task(self, y):
        if np.issubdtype(y.dtype, np.floating) and np.unique(y).shape[0] > 2:
            # The target variable is a float indicating a regression task
            raise ValueError(f'Unknown label type: Estimator only supports classification')


    def fit(self, X, y):
        if y is None:
            raise ValueError("MLP requires y to be passed, but the target y is None")
        X, y = check_X_y(X,y)
        self.check_regression_task(y)
        random_state = check_random_state(self.seed)
        self._is_fitted = True
        self.classes_, y = np.unique(y, return_inverse=True)
        self.n_features_in_ = X.shape[1]

        activation_function = self.afs[self.activation_function]()
        
        self.weights_, self.bias_ = self.initialize_weights(X, random_state, self.classes_.shape[0])
        for iteration in range(self.n_iter):
            for row_idx in range(X.shape[0]):
                X_sample = X[row_idx, :].reshape(1, -1)
                y_sample = y[row_idx]
                activation_values, z_values = self.feed_forward(X_sample, activation_function)
                #TODO probably remove again
                if np.isnan(activation_values[-1]).any():
                    self.learning_rate = self.learning_rate/10
                    self.weights_, self.bias_ = self.initialize_weights(X, random_state, self.classes_.shape[0])
                    continue
                #TODO debug gradient shit
                self.perform_backpropagation(activation_function, activation_values, z_values, y_sample, X_sample)
        return self

    def perform_backpropagation(self, activation_function, activation_values, z_values, y, X):
        for layer_idx in range(len(activation_values)-1, -1, -1):
            if layer_idx == len(activation_values)-1:
                y_indicator = np.zeros((self.classes_.shape[0]))
                y_indicator[y] = 1.0
                delta = activation_values[-1] - y_indicator
            else:
                #TODO h' shit
                # delta = <derivative> * self.weights_ * delta
                deriv_relu = activation_function.derivative(z_values[layer_idx])
                delta = deriv_relu * (prev_delta @ self.weights_[layer_idx + 1].T)

            if layer_idx == 0:
                gradient = X.T @ delta
            else:
                gradient = activation_values[layer_idx-1].T @ delta
            self.weights_[layer_idx] -= self.learning_rate * gradient
            self.bias_[layer_idx] -= self.learning_rate * delta.T
            prev_delta = delta

    
    def initialize_weights(self, X, random_state, num_classes):
        weights = []
        bias = []
        input_dim = X.shape[1]
        for layer_size in self.hidden_layer_sizes:
            new_layer = random_state.rand(input_dim, layer_size)
            new_bias = random_state.rand(layer_size, 1)
            weights.append(new_layer)
            bias.append(new_bias)
            input_dim = layer_size
        
        output_layer = random_state.rand(input_dim, num_classes)
        output_bias = random_state.rand(num_classes, 1)
        weights.append(output_layer)
        bias.append(output_bias)
        return weights, bias


    def predict(self, X):
        if not self._is_fitted:
            raise NotFittedError("Tried to call predict on model that was not fitted.")
        X = check_array(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("Number of features from X doesn't match with number of features that were trained on")
        activation_function = self.afs[self.activation_function]()
        
        activation_values, _ = self.feed_forward(X, activation_function)
        class_probabilities = activation_values[-1]
        return self.classes_[np.argmax(class_probabilities, axis=1)]
    

    def feed_forward(self, X, activation_function):
        #TODO maybe implement dropout for regularisation?
        activation_values = []
        z_values = []
        for weights, bias in zip(self.weights_, self.bias_):
            z = X @ weights + bias.T
            X = activation_function.function(z)
            z_values.append(z)
            # check if in last layer
            if len(activation_values) == len(self.weights_) - 1:
                activation_values.append(softmax(z))
            else:
                activation_values.append(X)
        return activation_values, z_values

    def get_params(self, deep=False):
        return {"hidden_layer_sizes": self.hidden_layer_sizes,
                "activation_function": self.activation_function,
                "learning_rate": self.learning_rate,
                "n_iter": self.n_iter,
                "seed": self.seed}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            if not hasattr(self, parameter):
                raise AttributeError(f"Class MLP has no attribute '{parameter}'")
            setattr(self, parameter, value)
        #self.check_params()
        return self


    def _more_tags(self):
        return {"requires_y": True, "poor_score": True}
    
def cartesian_product_dict(**kwargs):
    param_names = kwargs.keys()
    for parameter_values in itertools.product(*kwargs.values()):
        yield dict(zip(param_names, parameter_values))

def gridSearchIteration(parameters, X, y):
    mlp = MLP()
    mlp.set_params(**parameters)
    scores = cross_val_score(mlp, X, y, cv=5, scoring='f1_macro', n_jobs=-1)
    return scores.mean()

def gridSearchCV(X, y, param_grid, n_workers = 4):
    param_combinations = list(cartesian_product_dict(**param_grid))
    max_score = -1
    for params in tqdm(param_combinations):
        score = gridSearchIteration(params, X, y)
        if score > max_score:
            max_score = score
            best_params = params
    print(f"Best params found: {best_params}")

    mlp = MLP()
    mlp.set_params(**best_params)
    mlp.fit(X_train, y_train)
    return mlp



if __name__ == "__main__":

    perform_gridsearch = True

    start = time.time()
    X_main = np.random.random_sample((10000, 10))
    sum_X = X_main.sum(axis=1).astype(int)
    y_main = np.clip(sum_X - np.amin(sum_X), 0, 4)
    X_train, X_test, y_train, y_test = train_test_split(X_main, y_main)

    if perform_gridsearch:
        params = {"n_iter": [100, 200, 300],
                "learning_rate": [0.01, 0.001, 0.0001],
                "hidden_layer_sizes": [[15, 10], [30, 20], [50, 30, 10]],
                "activation_function": ['relu']}
        params = {"n_iter": [100],
                "learning_rate": [0.001, 0.0001],
                "hidden_layer_sizes": [[5, 10]],
                "activation_function": ['relu', 'sigmoid']}
        best_model = gridSearchCV(X_train, y_train, params)
        y_pred = best_model.predict(X_test)
        print(accuracy_score(y_test, y_pred))
        print(confusion_matrix(y_test, y_pred))

    else:
        mlp = MLP((15, 10), n_iter=200, learning_rate=0.01)
        mlp.fit(X_train, y_train)
        y_pred = mlp.predict(X_test)
        print(accuracy_score(y_test, y_pred))
        print(confusion_matrix(y_test, y_pred))
        
        skmlp = skMLP((15, 10), solver='sgd', alpha=0.0001, max_iter=200)
        skmlp.fit(X_train, y_train)
        y_pred = skmlp.predict(X_test)
        print(accuracy_score(y_test, y_pred))
        print(confusion_matrix(y_test, y_pred))

    print(time.time()-start)
    