from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import pandas as pd
from data_utils import safe_saver, safe_loader
from pso_utils import f
import pyswarms as ps
import numpy as np
import argparse

def pso(model, sample):
    """
    Function to find pso feature sets. It loads in a file, pso_options.pkl, that has
    a variety of pso configurations and then performs feature reduction for those sets.
    It returns a dictionary of feature that can be used to reduce columns for further
    testing.
    Args:
        model: 'log', 'rf', or 'xgb'. Selects the model to use for pso reduction.
        sample: scaled or 'smote'. Selects the data to use for pso reduction.
    """
    
    """
    Due to the fact that pso takes a significant amount of time to perform, I selected
    simplex models that will later be optimized.
    """
    if model == 'log':
        model = LogisticRegression(max_iter = 1000, solver = 'liblinear')
        name = 'log'
    elif model == 'rf':
        model = RandomForestClassifier(n_estimators = 10, random_state = 42, n_jobs = 4)
        name = 'rf'
    elif model == 'xgb':
        model = XGBClassifier(n_estimators = 100, max_depth = 5, learning_rate = 0.1,
                              device = 'cuda', random_state = 42)
        name = 'xgb'
    else:
        raise ValueError(f"Unsupported model selected: '{model}'. Please choose from: 'log', 'rf', or 'xgb'.")

    """
    Loads in the scaled training files, finds the number of columns for n_features, and
    loads in the pre-made dictionary of different pso configurations and initializes
    the empty dictionary.
    """
    if sample == 'scaled':
        X = pd.read_csv(f'data/processed/scaled/X_train_scaled.csv')
        y = pd.read_csv('data/processed/y_train.csv')['Churn']
    elif sample == 'smote':
        X = pd.read_csv(f'data/processed/smote/X_train_smote.csv')
        y = pd.read_csv('data/processed/smote/y_train_smote.csv')['Churn']
    else:
        raise ValueError(f"Unsupported sample selected: '{sample}'. Choose 'scaled' or 'smote'.")

    n_features = X.shape[1]
    options = safe_loader('data/processed/pso_options.pkl')
    features_dict = {}

    """
    This loads through all the configurations for the selected model.
    """
    for spec, config in options.items():

        optimizer = ps.discrete.BinaryPSO(n_particles = 50, dimensions  = n_features, options = config)

        cost, best_mask = optimizer.optimize(lambda x : f(x, model, X, y), iters = 100)
        selected_features = np.where(best_mask == 1)[0]

        features_dict[spec] = {'cost' : cost,
                               'best_mask' : best_mask,
                               'features' : selected_features}

    """
    Saves the feature sets.
    """
    safe_saver(features_dict, f"data/processed/{sample}/", f"pso_feature_set_{name}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'just a simple argument parser.')
    parser.add_argument('--model', required = True, help = 'model: log, rf, or xgb')
    parser.add_argument('--sample', required = True, help = 'sample: scaled or smote')

    arg = parser.parse_args()

    pso(arg.model, arg.sample)