from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score
from model_functions import train_log_rf, train_xgb
import pandas as pd
import argparse

def train_baseline(model):
    match model:
        case 'log':
            model = LogisticRegression(max_iter = 1000, solver = 'liblinear')
            name = 'log'
            params = {'penalty' : ['l1', 'l2'],
                      'C' : [0.01, 0.1, 1, 10]}

        case 'rf':
            model = RandomForestClassifier(n_estimators = 10, random_state = 42, n_jobs = 4)
            name = 'rf'
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
                      'colsample_bytree': [0.6, 0.8]}
            name = 'xgb'

    X_train = pd.read_csv('data/processed/scaled/X_train_scaled.csv')
    y_train = pd.read_csv('data/processed/y_train.csv')['Churn']
    X_test = pd.read_csv('data/processed/scaled/X_test_scaled.csv')
    y_test = pd.read_csv('data/processed/y_test.csv')['Churn']

    if name != 'xgb':

        train_log_rf(model, params, X_train, y_train, X_test, y_test, 'scaled', 'baseline', name)

    else:

        train_xgb(params, X_train, y_train, X_test, y_test, 'scaled', 'baseline', name)

def RFECV_reduced(model):
    match model:
        case 'log':
            model = LogisticRegression(max_iter = 1000, solver = 'liblinear')
            name = 'log'
            params = {'penalty' : ['l1', 'l2'],
                      'C' : [0.01, 0.1, 1, 10]}

        case 'rf':
            model = RandomForestClassifier(n_estimators = 10, random_state = 42, n_jobs = 4)
            name = 'rf'
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
                      'colsample_bytree': [0.6, 0.8]}
            name = 'xgb'

    X_train = pd.read_csv('data/processed/scaled/X_train_scaled.csv')
    y_train = pd.read_csv('data/processed/y_train.csv')['Churn']
    X_test = pd.read_csv('data/processed/scaled/X_test_scaled.csv')
    y_test = pd.read_csv('data/processed/y_test.csv')['Churn']

    if name != 'xgb':
        train_log_rf(model, params, X_train, y_train, X_test, y_test, 'scaled', 'RFECV', name, 'RFECV')

    else:
        train_xgb(params, X_train, y_train, X_test, y_test, 'scaled', 'RFECV', name, 'RFECV')

def pso_reduced(model):
    match model:
        case 'log':
            model = LogisticRegression(max_iter = 1000, solver = 'liblinear')
            name = 'log'
            params = {'penalty' : ['l1', 'l2'],
                      'C' : [0.01, 0.1, 1, 10]}

        case 'rf':
            model = RandomForestClassifier(n_estimators = 10, random_state = 42, n_jobs = 4)
            name = 'rf'
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
                      'colsample_bytree': [0.6, 0.8]}
            name = 'xgb'

    if name != 'xgb':
        train_log_rf(model, params, folder = 'scaled', iteration = 'pso', name = name, reduction = 'pso')

    else:
        train_xgb(params, folder = 'scaled', iteration = 'pso', name = name, reduction = 'pso')

def pca_reduced(model):
    match model:
        case 'log':
            model = LogisticRegression(max_iter = 1000, solver = 'liblinear')
            name = 'log'
            params = {'penalty' : ['l1', 'l2'],
                      'C' : [0.01, 0.1, 1, 10]}

        case 'rf':
            model = RandomForestClassifier(n_estimators = 10, random_state = 42, n_jobs = 4)
            name = 'rf'
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
                      'colsample_bytree': [0.6, 0.8]}
            name = 'xgb'

    y_train = pd.read_csv('data/processed/y_train.csv')['Churn']
    y_test = pd.read_csv('data/processed/y_test.csv')['Churn']

    if name != 'xgb':
        train_log_rf(model, params, y_train = y_train, y_test = y_test, folder = 'scaled',
                     iteration = 'pca', name = name, reduction = 'pca')

    else:
        train_xgb(params, y_train = y_train, y_test = y_test, folder = 'scaled',
                  iteration = 'pca', name = name, reduction = 'pca')

def run_script(function, model):

    match function:
        case 'base':
            train_baseline(model)
        case 'rfe':
            RFECV_reduced(model)
        case 'pso':
            pso_reduced(model)
        case 'pca':
            pca_reduced(model)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Just a simple argument parser.')
    parser.add_argument('--function', required = True, help = 'base, rfe, pso, or pca')
    parser.add_argument('--model', required = True, help = 'log, rf, or xgb')

    arg = parser.parse_args()

    run_script(arg.function, arg.model)