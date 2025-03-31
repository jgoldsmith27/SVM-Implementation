import numpy as np
from scipy.optimize import minimize

class SVM:
    """
    Support Vector Machine implementation from scratch.
    This implementation follows the dual formulation of SVM and provides
    different kernel options.
    """
    
    def __init__(self, C=100, kernel='linear', degree=2, gamma=1):
        """
        Initialize SVM with parameters.
        
        Parameters:
        -----------
        C : float, default=100
            Regularization parameter. The strength of the regularization is
            inversely proportional to C. Must be strictly positive.
        kernel : {'linear', 'poly', 'rbf'}, default='linear'
            Specifies the kernel type to be used in the algorithm.
        degree : int, default=2
            Degree of the polynomial kernel function ('poly').
            Ignored by other kernels.
        gamma : float, default=1
            Kernel coefficient for 'rbf' and 'poly' kernels.
        """
        self.C = C
        self.kernel_type = kernel
        self.degree = degree
        self.gamma = gamma
        self.alpha = None
        self.b = None
        self.non_zero_vals = None
        self.P = None
        
        # Set the kernel function based on the type
        if kernel == 'linear':
            self.kernel = self.linearKernel
        elif kernel == 'poly':
            self.kernel = lambda x, y: self.polyKernel(x, y, self.degree)
        elif kernel == 'rbf':
            self.kernel = lambda x, y: self.rbfKernel(x, y, self.gamma)
        else:
            raise ValueError(f"Kernel '{kernel}' not recognized. Options are 'linear', 'poly', or 'rbf'")
    
    def linearKernel(self, x, y):
        """Linear kernel: K(x, y) = x^T * y"""
        return np.dot(x, y)
    
    def polyKernel(self, x, y, degree, r=1):
        """Polynomial kernel: K(x, y) = (x^T * y + r)^degree"""
        return (np.dot(x, y) + r) ** degree
    
    def rbfKernel(self, x, y, gamma):
        """Radial basis function kernel: K(x, y) = exp(-gamma * ||x - y||^2)"""
        return np.exp(-gamma * np.sum((x - y) ** 2))
    
    def __compute_P_mat(self, t, x):
        """
        Compute the P matrix where P_ij = t_i * t_j * K(x_i, x_j)
        """
        n = len(x)
        P = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                P[i][j] = t[i] * t[j] * self.kernel(x[i], x[j])
        return P
    
    def __objective(self, alpha):
        """
        Objective function to minimize in the dual formulation:
        0.5 * alpha^T * P * alpha - sum(alpha)
        """
        return 0.5 * np.dot(alpha, np.dot(self.P, alpha)) - np.sum(alpha)
    
    def __zerofun(self, alpha):
        """
        Equality constraint for the dual formulation:
        sum(alpha_i * t_i) = 0
        """
        return np.dot(np.transpose(alpha), self.targets)
    
    ##### 3.2.4 & 3.2.5 #####
    def fit(self, X, y):
        """
        Fit the SVM model according to the given training data.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like, shape (n_samples,)
            Target values (class labels in classification, real numbers in
            regression). Should be +1 or -1.
            
        Returns:
        --------
        self : object
            Returns self.
        """
        # Save training data
        self.inputs = X
        self.targets = y
        n_train = len(X)
        
        # Compute the P matrix
        self.P = self.__compute_P_mat(self.targets, self.inputs)
        
        # Initial values for alpha
        alpha_0 = np.zeros(n_train)
        
        # Set up bounds and constraints
        bounds = [(0, self.C) for _ in range(n_train)]
        constraint = {'type':'eq', 'fun': self.__zerofun}
        
        # Call the minimize function to find the optimal alphas
        min_ = minimize(self.__objective, alpha_0, bounds=bounds, constraints=constraint)
        
        # Extract the alpha values
        self.alpha = min_.x
        
        # Check if the optimization was successful
        if min_.success:
            print("Optimization successful")
        else:
            print("ERROR: Optimization didn't converge")
        
        # Extract the non-zero alpha values and their corresponding data points
        epsilon = 1e-5
        self.non_zero_vals = []
        for i in range(len(self.alpha)):
            if self.alpha[i] > epsilon:
                self.non_zero_vals.append((self.alpha[i], self.inputs[i], self.targets[i]))
        
        # Compute the bias term b
        self.b = self.__compute_b()
        
        return self
    
    def __compute_b(self):
        """
        Compute the bias term b. We need to use a point on the margin,
        which corresponds to a point with 0 < alpha < C.
        """
        if not self.non_zero_vals:
            return 0
        
        # Default to the first support vector
        support_vector = self.non_zero_vals[0][1]
        t_support = self.non_zero_vals[0][2]
        
        # Find a support vector on the margin (0 < alpha < C)
        for alpha, x, t in self.non_zero_vals:
            if alpha < self.C:
                t_support = t
                support_vector = x
                break

        # Error check - in case we couldn't find an alpha < C
        if t_support == 0:
            print("ERROR: Couldn't find alpha < C")
            return 0
        
        # Calculate b using the support vector on the margin
        b_result = 0
        for alpha, x, t in self.non_zero_vals:
            b_result += alpha * t * self.kernel(support_vector, x)
        
        return b_result - t_support
    
    def indicator(self, point):
        """
        Evaluate the decision function for a given sample.
        
        Parameters:
        -----------
        point : array-like, shape (n_features,)
            Sample to evaluate.
            
        Returns:
        --------
        decision : float
            Decision value for the sample.
        """
        summation = 0
        for alpha, support_vector, target in self.non_zero_vals:
            summation += alpha * target * self.kernel(support_vector, point)
        
        return summation - self.b 
        
    def predict(self, X):
        """
        Predict class labels for samples in X.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features) or single point
            Samples to predict class labels for.
            
        Returns:
        --------
        y_pred : array, shape (n_samples,) or single float
            Predicted class labels (+1 or -1).
        """
        # Handle both single points and arrays
        if len(np.array(X).shape) == 1:
            # Single point
            indicator_val = self.indicator(X)
            return 1 if indicator_val >= 0 else -1
        else:
            # Multiple points
            return np.array([1 if self.indicator(x) >= 0 else -1 for x in X]) 