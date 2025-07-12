from sklearn.model_selection import cross_val_score
import numpy as np

def f_per_particle(mask, clf, X, y, alpha = 0.9):
    """
    Args:
        mask: binary array represented to selected features, passed in from f().
        clf: the classifier used during pso training.
        X: feature matrix.
        y: target labels.
        alpha: weighting factor for feature count penalty, default is 0.9.
    """

    """
    If the model selects for no features at all, the penalty is 'inf' to
    avoid this outcome in some edge cases where the penalty for every
    other possible combination is more than 1.
    
    Then selects the feature set based on the mask, and performs 3 fold
    cross-validation using roc_auc and returns the penalized objective.
    The penalization function 1 - score favours the highest possible roc-auc
    score while also minimizing the features. This is because the goal is to
    return the minimal score possible.
    """
    if np.count_nonzero(mask) == 0:
        return float('inf')
    X_selected = X.iloc[:, mask == 1]
    score = cross_val_score(clf, X_selected, y, cv = 3, scoring = 'roc_auc').mean()
    return 1 - score + alpha * (np.sum(mask) / X.shape[1])

def f(x, clf, X, y):
    """
    Args:
        model: the model to use for evaluation.
        X: feature_matrix.
        y: target labels.
    """
    n_particles = x.shape[0]
    return np.array([f_per_particle(x[i], clf, X, y) for i in range(n_particles)])