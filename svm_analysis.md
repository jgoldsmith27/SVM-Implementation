# SVM Analysis

## 5.1 Linear Kernel Exploration

The linear kernel experiments demonstrate the basic capabilities and limitations of linear SVMs:

### Well-Separated Case
In the first experiment (plot `5_1_1_linear_base.png`), we see how the linear kernel performs on well-separated data. The decision boundary is clean and effectively separates the two classes with good margins. This represents an ideal case for linear SVMs where the data is linearly separable.

However, even in cases that are nearly linearly separable, there may still be a few points—often considered outliers—that lie on the wrong side of the margin or even the decision boundary. These points are still referred to as support vectors because they influence the optimal hyperplane. In the soft-margin SVM formulation, such misclassified or marginally placed points are permitted to ensure better generalization, especially in the presence of noise or outliers.

### Overlapping Case 
The second experiment (plot `5_1_2_linear_overlap.png`) shows how the linear kernel handles overlapping classes. The decision boundary attempts to find the best linear separation, but some misclassifications are inevitable due to the non-linear nature of the class distributions.

### C Parameter Exploration
The C parameter experiments (plots `5_1_3_linear_C_0.1.png`, `5_1_3_linear_C_100.png`, `5_1_3_linear_C_1e6.png`) demonstrate how the slack parameter affects the decision boundary:
- Small C (0.1): The model is more tolerant of misclassifications, resulting in a wider margin and a more relaxed decision boundary. It prioritizes generalization over perfectly fitting the training data. The optimization problem is smoother and well-conditioned, so the solver typically converges quickly and reliably.

- Medium C (100): This setting strikes a balance between maximizing the margin and minimizing classification errors. The boundary becomes tighter but still maintains robustness to outliers. The optimization remains tractable and generally converges without issue.

- Large C (1e6): The model becomes very strict, heavily penalizing misclassifications. It tries to classify every point correctly, including outliers, which leads to a narrow margin. This results in a decision boundary that may be strongly distorted by even a few difficult or misclassified points. In datasets where perfect separation is not possible, this setting poses a challenge for the optimizer. It is forced to attempt a separation that doesn’t exist, trying to minimize constraint violations that can’t be fully eliminated. As a result, the optimizer may fail to converge or hit iteration limits.

## 5.2 Non-linear Kernels

The non-linear kernel experiments show how SVMs can handle more complex decision boundaries:

### Data Visualization
Plot `5_2_0_nonlinear_data.png` shows our non-linearly separable dataset with two clusters of class A surrounding one cluster of class B.

### Polynomial Kernel
The polynomial kernel experiments (plots `5_2_1_poly_degree_1.png`, `5_2_1_poly_degree_2.png`, `5_2_1_poly_degree_3.png`) show:
- Degree 1: Equivalent to linear kernel, cannot capture the non-linear pattern
- Degree 2: Creates a curved decision boundary that better fits the data
- Degree 3: More complex boundary that may start to overfit

### RBF Kernel
The RBF kernel experiments (plots `5_2_2_rbf_gamma_0.1.png`, `5_2_2_rbf_gamma_1.0.png`, `5_2_2_rbf_gamma_10.0.png`) demonstrate:
- Low gamma (0.1): Smoother, more general decision boundary
- Medium gamma (1.0): Good balance between flexibility and generalization
- High gamma (10.0): More complex boundary that closely fits the training data

## 5.3 Kernel Parameters

### Polynomial Kernel Degree
The polynomial degree experiments (plots `5_3_1_poly_degree_1.png` through `5_3_1_poly_degree_5.png`) show the bias-variance trade-off:
- Lower degrees (1-2) have high bias but low variance
- Higher degrees (3-5) have lower bias but higher variance, potentially leading to overfitting
- The optimal degree depends on the complexity of the true decision boundary

### RBF Kernel Gamma
The gamma parameter experiments (plots `5_3_2_rbf_gamma_0.1.png` through `5_3_2_rbf_gamma_100.0.png`) illustrate:
- Low gamma: High bias, low variance (smoother boundary)
- High gamma: Low bias, high variance (more complex boundary)
- The gamma parameter effectively controls the "reach" of each support vector

## 5.4 Slack Parameter (C)

The slack parameter experiments across different kernels (plots `5_4_linear_C_*.png`, `5_4_poly_C_*.png`, `5_4_rbf_C_*.png`) show:

### Small C Values (0.01)
- More relaxed boundaries
- Larger margins
- More training errors allowed
- Higher bias, lower variance

### Large C Values (10000.0)
- Stricter boundaries
- Smaller margins
- Fewer training errors allowed
- Lower bias, higher variance

The effect is consistent across kernels, though more pronounced with non-linear kernels.

## 5.5 Sklearn Comparison

The comparison between our implementation and sklearn's SVM (plots `5_5_1_our_svm.png`, `5_5_2_sklearn_svm.png`, and `5_5_3_comparison.png`) shows:

### Similarities
- Both implementations produce similar decision boundaries
- Support vectors are selected in similar locations
- Overall classification performance is comparable

### Differences
- Sklearn's implementation may be more numerically stable
- Minor differences in the exact position of the decision boundary
- Slight variations in the number of support vectors selected (increased when outliers present in data)

The performance metrics (in `5_5_4_metrics.txt`) show that both implementations achieve similar accuracy levels, with any differences likely due to implementation details rather than fundamental algorithmic differences.
