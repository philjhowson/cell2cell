from pso_training_utils import pso_train_log_rf, pso_train_xgb, load_data
from general_utils import safe_saver
import os
import json
import argparse

def train_models(model, sample):
    """
    Loads in the selected model and runs a training and validation function.
    GridSearchCV is performed with predetermine params that can be adjusted
    below.
    Args:
        model: log, rf, xgb.
        sample: scaled, smote.
    """

    """
    Ensures valid arguments are selected for the training loop.
    """
    models = set(['log', 'rf', 'xgb'])
    samples = set(['scaled', 'smote'])

    if model not in models:
        raise ValueError(f"Invalid model selected: {model}.")
    if sample not in samples:
        raise ValueError(f"Invalid sample selected: {sample}.")

    """
    selects the correct parameters depending on the model.
    """
    match model:
        case 'log':
            params = {'penalty' : ['l1', 'l2'],
                      'C' : [0.01, 0.1, 1, 10]}

        case 'rf':
            params = {'n_estimators': [100, 200, 300],
                      'max_depth': [2, 4, 5],
                      'min_samples_split': [15, 20],
                      'min_samples_leaf': [5, 10],
                      'max_features': [0.05, 0.1, 0.2]}

        case 'xgb':
            params = {'max_depth': [4, 6, 8],
                      'learning_rate': [0.001, 0.01],
                      'min_child_weight': [1, 5],
                      'gamma': [0.5, 2],
                      'subsample': [0.6, 0.8],
                      'colsample_bytree': [0.6, 0.8],
                      'eval_metric': ['auc'], 'device': ['cuda'],
                      'random_state': [42]}

    """
    Loads in the data and runs the selected training and validation function.
    """
    X_train, y_train, X_test, y_test = load_data(model, sample, 'pso')

    if model != 'xgb':
        classifier, params, results, features, pso_results = pso_train_log_rf(params, X_train, y_train, X_test, y_test, sample, model)

    else:
        classifier, params, results, features, pso_results =  pso_train_xgb(params, X_train, y_train, X_test, y_test, sample)       

    """
    Saves the model, best parameters, results, feature importances, and pso results if pso was selected.
    """
    safe_saver(classifier, f"models/{sample}", f"pso_{model}_model")
    with open(f"models/{sample}/pso_{model}_params.json", 'w') as f:
        json.dump(params, f)
    safe_saver(params, f"models/{sample}/", f"pso_{model}_params")

    os.makedirs(f"metrics/{sample}/", exist_ok = True)
    with open(f"metrics/{sample}/pso_scores_{model}.json", 'w') as f:
        json.dump(results, f)

    features.to_csv(f"models/{sample}/pso_feature_importances_{model}.csv")

    with open(f"metrics/{sample}/pso_grid_scores_{model}.json", 'w') as f:
        json.dump(pso_results, f)        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Just a simple argument parser.')
    parser.add_argument('--model', required = True, help = 'log, rf, or xgb')
    parser.add_argument('--sample', required = True, help = 'scaled or smote')

    arg = parser.parse_args()

    train_models(arg.model, arg.sample)
