import pickle
import numpy as np
import os
from sklearn.model_selection import cross_val_score

def safe_saver(file, path, name):
    """
    This is a simple saving function for .pkl files.
    
    args:
        file: the file you wish to save.
        path: the directory you wish to save to, if it
              does not exist, it will be created.
        name: the name you wish the file to be called.
    """
    os.makedirs(path, exist_ok = True)
    
    with open(f'{path}/{name}.pkl', 'wb') as f:
        pickle.dump(file, f)

def safe_loader(file):
    """
    Simple function for opening .pkl files
    args:
        file: the path to the file you want to open, include
        .pkl ending.
    """
    if os.path.exists(file):
        with open(file, 'rb') as f:
            return pickle.load(f)
        
    print(f"{file} not found! Please check directory.")
    return None

def f_per_particle(mask, clf, X, y, alpha = 0.9):
    if np.count_nonzero(mask) == 0:
        return 1.0
    X_selected = X.iloc[:, mask == 1]
    score = cross_val_score(clf, X_selected, y, cv = 3, scoring = 'f1').mean()
    return 1 - score + alpha * (np.sum(mask) / X.shape[1])

def f(x, clf, X, y):
    n_particles = x.shape[0]
    return np.array([f_per_particle(x[i], clf, X, y) for i in range(n_particles)])

