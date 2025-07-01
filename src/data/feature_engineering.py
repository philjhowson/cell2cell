import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from data_functions import safe_saver, safe_loader, f
import argparse
import pyswarms as ps
import numpy as np


def feature_engineering():

    X = pd.read_csv('data/processed/X_train.csv')
    X_test = pd.read_csv('data/processed/X_test.csv')
    y = pd.read_csv('data/processed/y_train.csv')

    """
    MinMaxScaler used because many models assume similar scales for features and
    so if a large value is imputed the model will incorrectly overweight it.
    """

    scaler = MinMaxScaler()

    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns = X.columns)

    X_test_scaled = scaler.transform(X_test)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns = X_test.columns)

    X_scaled.to_csv('data/processed/scaled/X_train_scaled.csv', index = False)
    X_test_scaled.to_csv('data/processed/scaled/X_test_scaled.csv', index = False)

    safe_saver(scaler, 'encoders/', 'MinMaxScaler') 

    pca = PCA(n_components = 0.95)
    X_train_pca = X_scaled.copy()
    X_train_pca = pca.fit_transform(X_train_pca)

    X_train_pca = pd.DataFrame(X_train_pca, columns = [f'PC{i+1}' for i in range(X_train_pca.shape[1])])

    X_train_pca.to_csv('data/processed/scaled/X_train_scaled_pca.csv', index = False)
    safe_saver(pca, 'encoders/', 'PCA_transformer_scaled')

    X_test_pca = X_test_scaled.copy()
    X_test_pca = pca.transform(X_test_pca)
    
    X_test_pca = pd.DataFrame(X_test_pca, columns = [f'PC{i+1}' for i in range(X_test_pca.shape[1])])

    X_test_pca.to_csv('data/processed/scaled/X_test_scaled_pca.csv', index = False)

def PSO(model):
    
    match model:
        case 'log':
            model = LogisticRegression(max_iter = 1000, solver = 'liblinear')
            name = 'log'
        case 'rf':
            model = RandomForestClassifier(n_estimators = 10, random_state = 42, n_jobs = 4)
            name = 'rf'
        case 'xgb':
            model = XGBClassifier(n_estimators = 25, max_depth = 2, learning_rate = 0.1,
                                reg_alpha = 0.1, reg_lambda = 5, subsample = 0.2, 
                                colsample_bytree = 0.2, device = 'cuda', random_state = 42)
            name = 'xgb'

    X = pd.read_csv('data/processed/scaled/X_train_scaled.csv')
    y = pd.read_csv('data/processed/y_train.csv')['Churn']

    n_features = X.shape[1]
    options = safe_loader('data/processed/pso_options.pkl')
    features_dict = {}

    for spec, config in options.items():

        optimizer = ps.discrete.BinaryPSO(n_particles = 50, dimensions  = n_features, options = config)

        cost, best_mask = optimizer.optimize(lambda x : f(x, model, X, y), iters = 100)
        selected_features = np.where(best_mask == 1)[0]

        features_dict[spec] = {'cost' : cost,
                               'best_mask' : best_mask,
                               'features' : selected_features}

    safe_saver(features_dict, 'data/processed/scaled/', f"pso_features_{name}")

    print(f"PSO feature selection complete. Summary of findings:\n{features_dict}")

def RFECV_reduction(model):

    match model:
        case 'log':
            model = LogisticRegression(max_iter = 1000, solver = 'liblinear')
            name = 'log'
        case 'rf':
            model = RandomForestClassifier(n_estimators = 10, random_state = 42, n_jobs = 4)
            name = 'rf'
        case 'xgb':
            model = XGBClassifier(n_estimators = 100, max_depth = 5, learning_rate = 0.1,
                                device = 'cuda', random_state = 42)
            name = 'xgb'

    X_train = pd.read_csv('data/processed/scaled/X_train_scaled.csv')
    y_train = pd.read_csv('data/processed/y_train.csv')['Churn']

    selector = RFECV(estimator = model, step = 1, cv = StratifiedKFold(5),
                     scoring = 'roc_auc')

    selector.fit(X_train, y_train)
    features = X_train.columns[selector.support_]
    safe_saver(features, 'data/processed/scaled/', f'RFECV_features_{name}')

def run_functions(function, model):
    
    match function:
        case 'features':
            feature_engineering()
        case 'pso':
            PSO(model)
        case 'rfe':
            RFECV_reduction(model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'feature engineering')
    parser.add_argument('--function', default = 'features', help = 'Default: features. Choose features, for scaling/PCA pipeline, pso for pyswarms feature selection, rfe for RFECV feature selection.')
    parser.add_argument('--model', default = 'log', help = 'Model for PSO and RFECV. Choose from log, rf, or xgb.')
    arg = parser.parse_args()

    run_functions(function = arg.function, model = arg.model)
