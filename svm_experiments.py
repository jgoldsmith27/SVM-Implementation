import numpy as np
import random
import matplotlib.pyplot as plt
import os
from sklearn import svm
from svm import SVM

# Create output directory
if not os.path.exists('plots'):
    os.makedirs('plots')

# Set random seed for reproducibility
np.random.seed(100)

def prepare_data(classA, classB):
    """Prepare inputs and targets from class data"""
    inputs = np.concatenate((classA, classB))
    targets = np.concatenate((np.ones(classA.shape[0]), -np.ones(classB.shape[0])))
    
    permute = list(range(len(inputs)))
    random.shuffle(permute)
    inputs = inputs[permute, :]
    targets = targets[permute]
    
    return inputs, targets

def plot_decision_boundary(model, classA, classB, title="SVM Decision Boundary", filename="decision_boundary.png", params=None):
    """Plot data points and decision boundary"""
    plt.figure(figsize=(8, 6))
    
    # Plot data points
    plt.scatter([p[0] for p in classA], [p[1] for p in classA], 
                c='blue', edgecolors='black', s=30, alpha=0.7, label='Class A (+1)')
    plt.scatter([p[0] for p in classB], [p[1] for p in classB], 
                c='red', edgecolors='black', s=30, alpha=0.7, label='Class B (-1)')
    
    # Create mesh grid
    x_min = min(min(p[0] for p in classA), min(p[0] for p in classB)) - 1
    x_max = max(max(p[0] for p in classA), max(p[0] for p in classB)) + 1
    y_min = min(min(p[1] for p in classA), min(p[1] for p in classB)) - 1
    y_max = max(max(p[1] for p in classA), max(p[1] for p in classB)) + 1
    
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                        np.linspace(y_min, y_max, 100))
    
    # Get decision values
    if isinstance(model, SVM):
        Z = np.array([model.indicator(point) for point in np.c_[xx.ravel(), yy.ravel()]])
    else:
        Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundary and margins
    plt.contour(xx, yy, Z, 
                levels=[-1.0, 0.0, 1.0],
                colors=('red', 'black', 'blue'),
                linestyles=('--', '-', '--'),
                linewidths=(1, 2, 1))
    
    # Plot support vectors
    if isinstance(model, SVM) and hasattr(model, 'non_zero_vals'):
        sv_points = np.array([x for _, x, _ in model.non_zero_vals])
        if len(sv_points) > 0:
            plt.scatter(sv_points[:, 0], sv_points[:, 1], 
                        s=100, facecolors='none', edgecolors='green', 
                        label='Support Vectors')
    elif hasattr(model, 'support_vectors_'):
        plt.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1],
                    s=100, facecolors='none', edgecolors='green',
                    label='Support Vectors')
    
    plt.grid(True)
    plt.legend()
    
    # Add parameter details to title if provided
    if params:
        param_str = ', '.join(f'{k}={v}' for k, v in params.items())
        title = f"{title}\n({param_str})"
    plt.title(title)
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.axis('equal')
    plt.savefig(f'plots/{filename}')
    plt.close()

def section_5_1():
    """Section 5.1: Linear Kernel Exploration"""
    print("\nSection 5.1: Linear Kernel Exploration")
    
    # Base parameters
    base_params = {'kernel': 'linear', 'C': 100}
    
    # Base case - well separated
    classA = np.random.randn(100, 2) * 0.3 + [1.5, 0.5]
    classB = np.random.randn(200, 2) * 0.3 + [0.0, 0.0]
    inputs, targets = prepare_data(classA, classB)
    
    model = SVM(**base_params)
    model.fit(inputs, targets)
    plot_decision_boundary(model, classA, classB,
                         title="5.1.1 Linear Kernel - Well Separated",
                         filename="5_1_1_linear_base.png",
                         params=base_params)
    
    # Overlapping case - same parameters, different data
    classA = np.random.randn(100, 2) * 0.3 + [0.5, 0.5]
    classB = np.random.randn(200, 2) * 0.3 + [0.0, 0.0]
    inputs, targets = prepare_data(classA, classB)
    
    model = SVM(**base_params)
    model.fit(inputs, targets)
    plot_decision_boundary(model, classA, classB,
                         title="5.1.2 Linear Kernel - Overlapping",
                         filename="5_1_2_linear_overlap.png",
                         params=base_params)
    
    # C parameter exploration - same data, different C
    classA = np.random.randn(100, 2) * 0.6 + [1.5, 0.5]
    classB = np.random.randn(200, 2) * 0.3 + [0.0, 0.0]
    inputs, targets = prepare_data(classA, classB)
    
    for C in [0.1, 100, 1e6]:
        params = {'kernel': 'linear', 'C': C}
        model = SVM(**params)
        model.fit(inputs, targets)
        plot_decision_boundary(model, classA, classB,
                             title=f"5.1.3 Linear Kernel - C Variation",
                             filename=f"5_1_3_linear_C_{C}.png",
                             params=params)

def section_5_2():
    """Section 5.2: Non-linear Kernels"""
    print("\nSection 5.2: Non-linear Kernels")
    
    # Use exactly the dataset from notebook
    classA = np.concatenate((
        np.random.randn(100, 2) * 0.3 + [1.5, 0.5],
        np.random.randn(100, 2) * 0.3 + [-1.5, 0.5]))
    classB = np.random.randn(200, 2) * 0.3 + [0.0, 0.0]
    inputs, targets = prepare_data(classA, classB)
    
    # Plot raw data
    plt.figure(figsize=(8, 6))
    plt.scatter([p[0] for p in classA], [p[1] for p in classA], 
                c='blue', edgecolors='black', s=30, alpha=0.7, label='Class A (+1)')
    plt.scatter([p[0] for p in classB], [p[1] for p in classB], 
                c='red', edgecolors='black', s=30, alpha=0.7, label='Class B (-1)')
    plt.grid(True)
    plt.legend()
    plt.title("5.2.0 Non-linearly Separable Data")
    plt.axis('equal')
    plt.savefig('plots/5_2_0_nonlinear_data.png')
    plt.close()
    
    # Test polynomial kernel - vary only degree
    base_params = {'kernel': 'poly', 'C': 100, 'gamma': 1}
    for degree in [1, 2, 3]:
        params = {**base_params, 'degree': degree}
        model = SVM(**params)
        model.fit(inputs, targets)
        plot_decision_boundary(model, classA, classB,
                             title=f"5.2.1 Polynomial Kernel",
                             filename=f"5_2_1_poly_degree_{degree}.png",
                             params=params)
    
    # Test RBF kernel - vary only gamma
    base_params = {'kernel': 'rbf', 'C': 100}
    for gamma in [0.1, 1.0, 10.0]:
        params = {**base_params, 'gamma': gamma}
        model = SVM(**params)
        model.fit(inputs, targets)
        plot_decision_boundary(model, classA, classB,
                             title=f"5.2.2 RBF Kernel",
                             filename=f"5_2_2_rbf_gamma_{gamma}.png",
                             params=params)

def section_5_3():
    """Section 5.3: Kernel Parameters"""
    print("\nSection 5.3: Kernel Parameters")
    
    # Use the same non-linearly separable dataset
    classA = np.concatenate((
        np.random.randn(100, 2) * 0.3 + [1.5, 0.5],
        np.random.randn(100, 2) * 0.3 + [-1.5, 0.5]))
    classB = np.random.randn(200, 2) * 0.3 + [0.0, 0.0]
    inputs, targets = prepare_data(classA, classB)
    
    # Polynomial kernel - vary only degree
    base_params = {'kernel': 'poly', 'C': 100, 'gamma': 1}
    for degree in [1, 2, 3, 5]:
        params = {**base_params, 'degree': degree}
        model = SVM(**params)
        model.fit(inputs, targets)
        plot_decision_boundary(model, classA, classB,
                             title=f"5.3.1 Polynomial Kernel",
                             filename=f"5_3_1_poly_degree_{degree}.png",
                             params=params)
    
    # RBF kernel - vary only gamma
    base_params = {'kernel': 'rbf', 'C': 100}
    for gamma in [0.1, 1.0, 10.0, 100.0]:
        params = {**base_params, 'gamma': gamma}
        model = SVM(**params)
        model.fit(inputs, targets)
        plot_decision_boundary(model, classA, classB,
                             title=f"5.3.2 RBF Kernel",
                             filename=f"5_3_2_rbf_gamma_{gamma}.png",
                             params=params)

def section_5_4():
    """Section 5.4: Slack Parameter"""
    print("\nSection 5.4: Slack Parameter")
    
    # Use non-linearly separable dataset
    classA = np.concatenate((
        np.random.randn(100, 2) * 0.3 + [1.5, 0.5],
        np.random.randn(100, 2) * 0.3 + [-1.5, 0.5]))
    classB = np.random.randn(200, 2) * 0.3 + [0.0, 0.0]
    inputs, targets = prepare_data(classA, classB)
    
    C_values = [0.01, 1.0, 100.0, 10000.0]
    
    # Test C with different kernels - keep other parameters constant
    kernel_params = {
        'linear': {'kernel': 'linear'},
        'poly': {'kernel': 'poly', 'degree': 2, 'gamma': 1},
        'rbf': {'kernel': 'rbf', 'gamma': 1}
    }
    
    for kernel_name, base_params in kernel_params.items():
        for C in C_values:
            params = {**base_params, 'C': C}
            model = SVM(**params)
            model.fit(inputs, targets)
            plot_decision_boundary(model, classA, classB,
                                 title=f"5.4 {kernel_name.capitalize()} Kernel",
                                 filename=f"5_4_{kernel_name}_C_{C}.png",
                                 params=params)

def section_5_5():
    """Section 5.5: Sklearn Comparison"""
    print("\nSection 5.5: Sklearn Comparison")
    
    # Use non-linearly separable dataset
    classA = np.concatenate((
        np.random.randn(100, 2) * 0.3 + [1.5, 0.5],
        np.random.randn(100, 2) * 0.3 + [-1.5, 0.5]))
    classB = np.random.randn(200, 2) * 0.3 + [0.0, 0.0]
    inputs, targets = prepare_data(classA, classB)
    
    # Parameters for both implementations
    params = {'kernel': 'rbf', 'C': 10, 'gamma': 1}
    
    # Train both models with same parameters
    our_svm = SVM(**params)
    our_svm.fit(inputs, targets)
    
    sklearn_svm = svm.SVC(**params)
    sklearn_svm.fit(inputs, targets)
    
    # Individual plots
    plot_decision_boundary(our_svm, classA, classB,
                         title="5.5.1 Our SVM Implementation",
                         filename="5_5_1_our_svm.png",
                         params=params)
    
    plot_decision_boundary(sklearn_svm, classA, classB,
                         title="5.5.2 Sklearn SVM",
                         filename="5_5_2_sklearn_svm.png",
                         params=params)
    
    # Side-by-side comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    param_str = ', '.join(f'{k}={v}' for k, v in params.items())
    fig.suptitle(f"5.5.3 SVM Implementation Comparison\n({param_str})")
    
    for ax, model, title in [(ax1, our_svm, "Our SVM"), (ax2, sklearn_svm, "Sklearn SVM")]:
        # Plot data
        ax.scatter([p[0] for p in classA], [p[1] for p in classA],
                  c='blue', edgecolors='black', s=30, alpha=0.7, label='Class A (+1)')
        ax.scatter([p[0] for p in classB], [p[1] for p in classB],
                  c='red', edgecolors='black', s=30, alpha=0.7, label='Class B (-1)')
        
        # Create mesh grid
        x_min = min(min(p[0] for p in classA), min(p[0] for p in classB)) - 1
        x_max = max(max(p[0] for p in classA), max(p[0] for p in classB)) + 1
        y_min = min(min(p[1] for p in classA), min(p[1] for p in classB)) - 1
        y_max = max(max(p[1] for p in classA), max(p[1] for p in classB)) + 1
        
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                            np.linspace(y_min, y_max, 100))
        
        # Get decision values
        if isinstance(model, SVM):
            Z = np.array([model.indicator(point) for point in np.c_[xx.ravel(), yy.ravel()]])
        else:
            Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # Plot decision boundary
        ax.contour(xx, yy, Z,
                  levels=[-1.0, 0.0, 1.0],
                  colors=('red', 'black', 'blue'),
                  linestyles=('--', '-', '--'))
        
        ax.set_title(title)
        ax.grid(True)
        ax.legend()
        ax.axis('equal')
    
    plt.tight_layout()
    plt.savefig('plots/5_5_3_comparison.png')
    plt.close()
    
    # Save comparison metrics
    with open('plots/5_5_4_metrics.txt', 'w') as f:
        # Make predictions
        our_pred = np.array([our_svm.predict(x) for x in inputs])
        sklearn_pred = sklearn_svm.predict(inputs)
        
        # Calculate accuracies
        our_acc = np.mean(our_pred == targets)
        sklearn_acc = np.mean(sklearn_pred == targets)
        agreement = np.mean(our_pred == sklearn_pred)
        
        f.write("=== SVM Implementation Comparison ===\n")
        f.write(f"Parameters: {param_str}\n")
        f.write(f"Our SVM Accuracy: {our_acc:.4f}\n")
        f.write(f"Sklearn SVM Accuracy: {sklearn_acc:.4f}\n")
        f.write(f"Prediction Agreement: {agreement:.4f}\n")

def main():
    """Run all sections"""
    section_5_1()  # Linear kernel exploration
    section_5_2()  # Non-linear kernels
    section_5_3()  # Kernel parameters
    section_5_4()  # Slack parameter
    section_5_5()  # Sklearn comparison

if __name__ == "__main__":
    main() 