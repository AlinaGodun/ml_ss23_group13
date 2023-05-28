from math import log
from abc import ABC, abstractmethod
import numpy as np
import time
from sklearn.utils import check_array, check_random_state, check_X_y
from sklearn.utils.estimator_checks import check_estimator
from sklearn.exceptions import NotFittedError
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.neural_network import MLPClassifier as skMLP
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from math import sqrt


class ActivationFunction(ABC):
    """Abstract base class for activation functions in a neural network.
    
    Subclasses of ActivationFunction must implement the function and derivative methods.
    """
    @abstractmethod
    def function(self, X):
        """Compute the activation function.
        
        Args:
            X (array-like): The input values.
            
        Returns:
            array-like: The computed activation values.
        """
        return NotImplemented
    
    @abstractmethod
    def derivative(self, X):
        """Compute the derivative of the activation function.
        
        Args:
            X (array-like): The input values.
            
        Returns:
            array-like: The computed derivative values.
        """
        return NotImplemented
    
class ReluActivation(ActivationFunction):
    """Rectified Linear Unit (ReLU) activation function.
    
    The ReLU activation function returns the maximum of zero and the input value.
    """
    def function(self, X):
        """Compute the ReLU activation function.
        
        Args:
            X (array-like): The input values.
            
        Returns:
            array-like: The computed ReLU activation values.
        """
        return np.maximum(0, X)
    
    def derivative(self, X):
        """Compute the derivative of the ReLU activation function.
        
        Args:
            X (array-like): The input values.
            
        Returns:
            array-like: The computed derivative values.
        """
        return (X > 0).astype(int)

class Sigmoid(ActivationFunction):
    """Sigmoid activation function.
    
    The sigmoid activation function computes the sigmoid (logistic) function.
    """
    def function(self, X):
        """Compute the sigmoid activation function.
        
        Args:
            X (array-like): The input values.
            
        Returns:
            array-like: The computed sigmoid activation values.
        """
        return 1 / (1+np.exp(-X))
    
    def derivative(self, X):
        """Compute the derivative of the sigmoid activation function.
        
        Args:
            X (array-like): The input values.
            
        Returns:
            array-like: The computed derivative values.
        """
        sigmoid = self.function(X)
        return sigmoid * (1 - sigmoid)

def softmax(X):
    """Compute the softmax function for a matrix of input values.
    
    Args:
        X (array-like): The input values.
        
    Returns:
        array-like: The computed softmax values.
    """
    return (np.exp(X)/np.sum(np.exp(X), axis=1)[:, None])

class MLP:
    """Multi-Layer Perceptron (MLP) classifier.

    MLP is a feedforward artificial neural network model that consists of multiple layers of nodes.
    This implementation supports classification tasks using various activation functions.

    Parameters:
        hidden_layer_sizes (tuple): The number of nodes in each hidden layer. Default is (100,).
        activation_function (str): The activation function to use in the hidden layers. Options: 'relu', 'sigmoid'.
            Default is 'relu'.
        learning_rate (float): The learning rate for gradient descent optimization. Default is 0.01.
        n_iter (int): The number of iterations for training the model. Default is 100.
        seed (int): The random seed for weight initialization. Default is 1111.
    """
    afs = {
        'relu': ReluActivation,
        'sigmoid': Sigmoid}
    
    _estimator_type = "classifier"
    _is_fitted = False
 
    def __init__(self, hidden_layer_sizes=(100,), activation_function='relu', learning_rate=0.01, n_iter=100, seed=1111):
        """Initialize the MLP classifier with the specified parameters.

        Args:
            hidden_layer_sizes (tuple): The number of nodes in each hidden layer.
            activation_function (str): The activation function to use in the hidden layers.
            learning_rate (float): The learning rate for gradient descent optimization.
            n_iter (int): The number of iterations for training the model.
            seed (int): The random seed for weight initialization.
        """
        self.check_params(activation_function, learning_rate, n_iter)
        
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation_function = activation_function
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.seed = seed


    def check_params(self, activation_function, learning_rate, n_iter):
        """Check if the provided parameters are valid.

        Args:
            activation_function (str): The activation function to use in the hidden layers.
            learning_rate (float): The learning rate for gradient descent optimization.
            n_iter (int): The number of iterations for training the model.

        Raises:
            KeyError: If the activation function is not valid.
            ValueError: If the learning rate or number of iterations is not positive.
        """
        if activation_function not in self.afs:
            raise KeyError(f'Invalid activation function provided: {self.activation_function}. ' +
                           f'Available activation functions: {self.afs}')
        if learning_rate < 0.0:
            raise ValueError(f'Invalid learning rate provided: {self.learning_rate}. ' +
                              'Learning rate must be positive.')
        
        if n_iter < 0:
            raise ValueError(f'Invalid number of iterations provided: {self.n_iter}. ' +
                              'Number of iterations must be positive.')

    def cross_entropy(self, y, y_pred):
        """Compute the cross-entropy loss.

        Args:
            y (array-like): The true labels.
            y_pred (array-like): The predicted probabilities.
            weights (list): The weight matrices of the neural network.
            bias (list): The bias vectors of the neural network.

        Returns:
            float: The computed cross-entropy loss.
        """
        #TODO regularisation with weights + bias
        loss = np.zeros(y.shape)
        for idx, class_label in zip(range(y_pred.shape[1]), self.classes_):
            y_prob = y_pred[:, idx]
            loss -= (idx == y)*np.log(y_prob)
        return loss

    def check_regression_task(self, y):
        """Check if the target variable indicates a regression task.

        If the target variable is a float indicating a regression task, raise a ValueError.

        Args:
            y (array-like): The target variable.

        Raises:
            ValueError: If the target variable indicates a regression task.
        """
        if np.issubdtype(y.dtype, np.floating) and np.unique(y).shape[0] > 2:
            # The target variable is a float indicating a regression task
            raise ValueError(f'Unknown label type: Estimator only supports classification')


    def fit(self, X, y, patience = 10):
        """Fit the MLP classifier to the training data.

        Args:
            X (array-like): The training input samples.
            y (array-like): The target values.
            patience (int): number of iterations without improvement after which training is stopped.
        Returns:
            self (object): Returns the instance itself.
        """
        if y is None:
            raise ValueError("MLP requires y to be passed, but the target y is None")
        X, y = check_X_y(X,y)
        delta = 0.001
        self.check_regression_task(y)
        random_state = check_random_state(self.seed)
        self._is_fitted = True
        self.converged_ = False
        self.validation_losses_ = []
        self.training_losses_ = []
        self.classes_, y = np.unique(y, return_inverse=True)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=random_state)

        self.n_features_in_ = X_train.shape[1]

        activation_function = self.afs[self.activation_function]()

        min_loss = float('inf')
        early_stopping_counter = 0
        
        self.weights_, self.bias_ = self.initialize_weights(X_train, random_state, self.classes_.shape[0])
        for iteration in range(self.n_iter):
            for row_idx in range(X_train.shape[0]):
                X_sample = X_train[row_idx, :].reshape(1, -1)
                y_sample = y_train[row_idx]
                activation_values, z_values = self.feed_forward(X_sample, activation_function)
                # print(activation_values)
                #TODO probably remove again
                if np.isnan(activation_values[-1]).any():
                    print("WARNING: divergence detected, please set smaller learning rate.")
                    print("As a punishment we return a model with randomly initialized weights.")
                    self.weights_, self.bias_ = self.initialize_weights(X_train, random_state, self.classes_.shape[0])
                    return self
                #TODO debug gradient shit
                self.perform_backpropagation(activation_function, activation_values, z_values, y_sample, X_sample)
                

            # compute train losses
            activation_values, _ = self.feed_forward(X_train, activation_function)
            class_probabilities = activation_values[-1]
            current_loss = self.cross_entropy(y_train, class_probabilities).mean()
            self.training_losses_.append(current_loss)

            # compute validation losses
            activation_values, _ = self.feed_forward(X_val, activation_function)
            class_probabilities = activation_values[-1]
            current_loss = self.cross_entropy(y_val, class_probabilities).mean()
            self.validation_losses_.append(current_loss)

            

            if current_loss - min_loss <= -delta:
                min_loss = current_loss
                early_stopping_counter = 0
                
            else:
                # print(f'Loss worse: {current_loss = }')
                early_stopping_counter += 1

            if early_stopping_counter == patience:
                print(f'Loss did not go down for {patience} iterations. Stopping training at iteration {iteration}...')
                break
    
        self.converged_ = True
        
        return self

    def perform_backpropagation(self, activation_function, activation_values, z_values, y, X):
        """Perform the backpropagation algorithm to update weights and biases.

        Args:
            activation_function (object): The activation function object.
            activation_values (list): The activation values for each layer.
            z_values (list): The z values for each layer.
            y (int): The target label.
            X (array-like): The input sample.
        """
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
            # print(f'{gradient = }')
            self.weights_[layer_idx] -= self.learning_rate * gradient
            self.bias_[layer_idx] -= self.learning_rate * delta.T
            prev_delta = delta

    def xavier(self, input_dim, output_dim, layer_size, random_state):
        lower = -(sqrt(6.0)/sqrt(input_dim+output_dim))
        upper = -lower  
        new_layer = random_state.rand(input_dim, layer_size)
        new_layer = lower + new_layer * (upper - lower)
        return new_layer

    def he(self, input_dim, output_dim, layer_size, random_state):
        std = (sqrt(2.0)/sqrt(input_dim))
        new_layer = random_state.rand(input_dim, layer_size)
        new_layer = new_layer * std
        return new_layer


    def initialize_weights(self, X, random_state, num_classes):
        """Initialize the weights and biases of the neural network.

        Args:
            X (array-like): The input samples.
            random_state (object): The random state object.
            num_classes (int): The number of classes in the target variable.

        Returns:
            tuple: The initialized weight matrices and bias vectors.
        """
        weights = []
        bias = []

        if self.activation_function == 'relu':
            weight_init_func = self.he
        elif self.activation_function == 'sigmoid':
            weight_init_func = self.xavier

        input_dim = X.shape[1]
        for layer, layer_size in enumerate(self.hidden_layer_sizes):
            if layer == len(self.hidden_layer_sizes)-1:
                output_dim = num_classes
            else:
                output_dim = self.hidden_layer_sizes[layer+1]

            new_layer = weight_init_func(input_dim, output_dim, layer_size, random_state)
            new_bias = np.zeros((layer_size, 1))
            input_dim = layer_size 

            weights.append(new_layer)
            bias.append(new_bias)
            

        output_layer = weight_init_func(input_dim, 0, num_classes, random_state)
        output_bias = np.zeros((num_classes, 1))
        weights.append(output_layer)
        bias.append(output_bias)

        # compute number of params and save it to the model
        number_of_params = 0

        for w, b in zip(weights, bias):
            number_of_params += w.shape[0] * w.shape[1]
            number_of_params += b.shape[0] * b.shape[1]

        self.number_of_params_ = number_of_params

        return weights, bias


    def predict(self, X):
        """Predict class labels for the input samples.

        Args:
            X (array-like): The input samples.

        Returns:
            array-like: The predicted class labels.
        """
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
        """Perform the feedforward algorithm.

        Args:
            X (array-like): The input samples.
            activation_function (object): The activation function object.

        Returns:
            tuple: The activation values and z values for each layer.
        """
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
        """Get the parameters of the MLP classifier.

        Args:
            deep (bool): Whether to recursively retrieve the parameters.

        Returns:
            dict: The parameters of the MLP classifier.
        """
        return {"hidden_layer_sizes": self.hidden_layer_sizes,
                "activation_function": self.activation_function,
                "learning_rate": self.learning_rate,
                "n_iter": self.n_iter,
                "seed": self.seed}

    def set_params(self, **parameters):
        """Set the parameters of the MLP classifier.

        Args:
            **parameters: The parameters to be set.

        Raises:
            AttributeError: If an invalid parameter is provided.

        Returns:
            self (object): Returns the instance itself.
        """
        for parameter, value in parameters.items():
            if not hasattr(self, parameter):
                raise AttributeError(f"Class MLP has no attribute '{parameter}'")
            setattr(self, parameter, value)
        #self.check_params()
        return self


    def _more_tags(self):
        """Provide additional tags for the MLP classifier.

        Returns:
            dict: Additional tags for the MLP classifier.
        """
        return {"requires_y": True, "poor_score": True}


if __name__ == "__main__":

    perform_gridsearch = False

    start = time.time()
    X_main = np.random.random_sample((1000, 10))
    sum_X = X_main.sum(axis=1).astype(int)
    y_main = np.clip(sum_X - np.amin(sum_X), 0, 4)
    X_train, X_test, y_train, y_test = train_test_split(X_main, y_main)

    mlp = MLP(learning_rate=0.001, n_iter=10000)
    mlp.fit(X_main, y_main)
    y_pred = mlp.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    exit()

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
    