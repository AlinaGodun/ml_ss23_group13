from math import log
import numpy as np
from sklearn.utils import check_array, check_random_state, check_X_y
from sklearn.utils.estimator_checks import check_estimator

def relu(X):
    return np.maximum(0, X)

def second_activation_func(X):
    pass

def softmax(X):
    return (np.exp(X)/np.sum(np.exp(X), axis=1)[:, None])

class MLP:
    available_activation_functions = {
        'relu': relu,
        'second_activation_func': second_activation_func}
    
    _estimator_type = "classifier"
    is_fitted = False
 
    def __init__(self, hidden_layer_sizes=(100,), activation_function='relu', learning_rate=0.01, n_iter=100, seed=1111):
        self.check_params(activation_function, learning_rate, n_iter)
        
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation_function = activation_function
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.seed = seed


    def check_params(self, activation_function, learning_rate, n_iter):
        if activation_function not in self.available_activation_functions:
            raise KeyError(f'Invalid activation function provided: {self.activation_function}. ' +
                           f'Available activation functions: {self.available_activation_functions}')
        
        if learning_rate < 0.0:
            raise ValueError(f'Invalid learning rate provided: {self.activation_function}. ' +
                              'Learning rate must be positive.')
        
        if n_iter < 0:
            raise ValueError(f'Invalid number of iterations provided: {self.activation_function}. ' +
                              'Number of iterations must be positive.')

    def cross_entropy(self, y, y_pred, weights, bias):
        #TODO regularisation with weights + bias
        loss = 0
        print(y_pred.shape)
        print(y_pred)
        for y_pred in y:
            loss += -(y_pred == y)*log(y_pred)
        return loss

    def fit(self, X, y):
        X, y = check_X_y(X,y)
        random_state = check_random_state(self.seed)
        self.is_fitted = True
        self.classes_, y = np.unique(y, return_inverse=True)
        self.features_in_ = X.shape[0]

        activation_function = self.available_activation_functions[self.activation_function]
        
        self.weights_, self.bias_ = self.initialize_weights(X, random_state, self.classes_.shape[0])

        for row_idx in range(X.shape[0]):
            y_pred = self.predict(X[row_idx, :].reshape(1, -1))
            print(f'y_pred: {y_pred}')
            loss = self.cross_entropy(y[row_idx], y_pred, self.weights_, self.bias_)
            print(loss)
    
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
        if not self.is_fitted:
            raise AttributeError("Tried to call predict on model that was not fitted.")
        X = check_array(X)
        activation_function = self.available_activation_functions[self.activation_function]

        D = self.feed_forward(X, activation_function)
        print(D)
        return self.classes_[np.argmax(D, axis=1)]
    

    def feed_forward(self, X, activation_function):
        for weights, bias in zip(self.weights_, self.bias_):
            z = X @ weights + bias.T
            X = activation_function(z)
        y = softmax(X)
        return y
    

    def get_params(self, deep=False):
        return {"hidden_layer_sizes": self.hidden_layer_sizes,
                "activation_function": self.activation_function,
                "learning_rate": self.learning_rate,
                "n_iter": self.n_iter}


    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        #self.check_params()
        return self


    def _more_tags(self):
        return {"requires_y": True, "poor_score": True}
    

if __name__ == "__main__":
    mlp = MLP((10,))
    X = np.random.random_sample((100, 10))
    y = np.random.randint(0, 2, size = (100))
    print(X.shape)
    print(y.shape)
    mlp.fit(X, y)
    y_pred = mlp.predict(X)
    print(y)
    print(y_pred)
    #check_estimator(mlp)
    