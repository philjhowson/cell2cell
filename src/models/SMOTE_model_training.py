from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, roc_auc_score
from model_functions import safe_saver, safe_loader
from xgboost import XGBClassifier
import pandas as pd
import argparse


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
                      'max_features': [0.05, 0.1, 0.2]
                      }
        case 'xgb':
            model = XGBClassifier(device = 'cuda', random_state = 42)
            name = 'xgb'
            params = {'n_estimators': [25, 50, 100],
                      'max_depth': [2, 3, 4],
                      'learning_rate': [0.001, 0.005, 0.01],
                      'subsample': [0.2, 0.5],
                      'colsample_bytree': [0.2, 0.3],
                      'gamma': [0, 1, 5],
                      'reg_alpha': [0, 0.1, 1],
                      'reg_lambda': [1, 5, 10]
                     }

    X_train = pd.read_csv('data/processed/X_train_resampled.csv')
    y_train = pd.read_csv('data/processed/y_train_resampled.csv')['Churn']

    cols_to_keep = safe_loader(f"data/processed/smote_RFECV_features_{name}.pkl")

    X_train = X_train[cols_to_keep]

    grid = GridSearchCV(model, param_grid = params, scoring = 'f1', cv = 4)

    grid.fit(X_train, y_train)

    safe_saver(grid.best_estimator_, 'models/', f"smote_RFECV_best_{name}_model")
    safe_saver(grid.best_params_, 'models/', f"smote_RFECV_best_{name}_params")

    y_pred = grid.predict(X_train)

    train_f1 = f1_score(y_train, y_pred, average = 'weighted')
    train_roc = roc_auc_score(y_train, y_pred)

    X_test = pd.read_csv('data/processed/X_test.csv')
    X_test = X_test[cols_to_keep]
    y_test = pd.read_csv('data/processed/y_test.csv')['Churn']

    y_pred = grid.predict(X_test)
    test_f1 = f1_score(y_test, y_pred, average = 'weighted')
    test_roc = roc_auc_score(y_test, y_pred)

    results = {'train_f1' : train_f1,
               'train_roc' : train_roc,
               'test_f1' : test_f1,
               'test_roc' : test_roc}

    safe_saver(results, 'metrics/', f"smote_RFE_scores_{name}")

    print(f"Training F1-Score: {round(train_f1, 3)}\nTest F1-Score: {round(test_f1, 3)}")
    print(f"Training ROC-AUC Score: {round(train_roc, 3)}\nTest ROC-AUC: {round(test_roc, 3)}")

    match name:
        case 'log':
            importances = grid.best_estimator_.coef_[0]
        case _:
            importances = grid.best_estimator_.feature_importances_

    feature_names = X_train.columns
    feature_importance = pd.DataFrame(zip(feature_names, importances), columns = ['Feature', 'Importance'])
    feature_importance['Absolute'] = abs(importances)
    feature_importance.sort_values(by = 'Absolute', ascending = False, inplace = True)
    feature_importance.to_csv(f"models/smote_RFECV_feature_importance_{name}.csv", index = False)

    print(feature_importance.head(10))

def PSO_reduced(model):

    match model:
        case 'log':
            model = LogisticRegression(max_iter = 1000, solver = 'liblinear')
            pso_values = safe_loader('data/processed/smote_pso_features_log.pkl')
            name = 'log'
            params = {'penalty' : ['l1', 'l2'],
                      'C' : [0.01, 0.1, 1, 10]}
        case 'rf':
            model = RandomForestClassifier(n_estimators = 10, random_state = 42, n_jobs = 4)
            pso_values = safe_loader('data/processed/smote_pso_features_rf.pkl')
            name = 'rf'
            params = {'n_estimators': [100, 200, 300],
                      'max_depth': [2, 4, 5],
                      'min_samples_split': [15, 20],
                      'min_samples_leaf': [5, 10],
                      'max_features': [0.05, 0.1, 0.2]
                      }
        case 'xgb':
            model = XGBClassifier(device = 'cuda', random_state = 42)
            name = 'xgb'
            pso_values = safe_loader('data/processed/smote_pso_features_xgb.pkl')
            params = {'n_estimators': [25, 50, 100],
                      'max_depth': [2, 3, 4],
                      'learning_rate': [0.001, 0.005, 0.01],
                      'subsample': [0.2, 0.5],
                      'colsample_bytree': [0.2, 0.3],
                      'gamma': [0, 1, 5],
                      'reg_alpha': [0, 0.1, 1],
                      'reg_lambda': [1, 5, 10]
                     }  

    X_train = pd.read_csv('data/processed/X_train_resampled.csv')
    y_train = pd.read_csv('data/processed/y_train_resampled.csv')['Churn']
    X_test = pd.read_csv('data/processed/X_test.csv')
    y_test = pd.read_csv('data/processed/y_test.csv')['Churn']

    results = {}

    for item in pso_values:

        feature_indices = pso_values[item]['features']
        cols_to_keep = X_test.columns[feature_indices]
        X = X_train[cols_to_keep]

        grid = GridSearchCV(model, param_grid = params, scoring = 'f1',
                            cv = 4)

        grid.fit(X, y_train)
        y_pred = grid.predict(X)
        train_f1 = f1_score(y_train, y_pred, average = 'weighted')
        train_roc = roc_auc_score(y_train, y_pred)

        X = X_test[cols_to_keep]
        y_pred = grid.predict(X)
        test_f1 = f1_score(y_test, y_pred, average = 'weighted')
        test_roc = roc_auc_score(y_test, y_pred)

        results[item] = {'train_f1' : train_f1,
                         'train_roc' : train_roc,
                         'test_f1' : test_f1,
                         'test_roc' : test_roc,
                         'params' : grid.best_params_,
                         'features' : cols_to_keep}
    
    best_roc = max(results, key = lambda k: results[k]['test_roc'])
    feature_indices = pso_values[best_roc]['features']
    cols_to_keep = X_train.columns[feature_indices]
    X_train = X_train[cols_to_keep]

    model.fit(X_train, y_train)

    safe_saver(model, 'models/', f"smote_best_pso_model_{name}")
    safe_saver(results[best_roc], 'models/', f"smote_best_pso_metrics_{name}")

    print(f"Best PSO model: {best_roc}\n"
          f"Features: {results[best_roc]['features'].tolist()}\n"
          f"Hyperparameters: {results[best_roc]['params']}\n"
          f"F1-Scores: {results[best_roc]['train_f1']}; {results[best_roc]['test_f1']}\n"
          f"ROC-AUCs: {results[best_roc]['train_roc']}; {results[best_roc]['test_roc']}")

    match name:
        case 'log':
            importances = model.coef_[0]
        case _:
            importances = model.feature_importances_

    feature_names = X_train.columns
    feature_importance = pd.DataFrame(zip(feature_names, importances), columns = ['Feature', 'Importance'])
    feature_importance['Absolute'] = abs(importances)
    feature_importance.sort_values(by = 'Absolute', ascending = False, inplace = True)
    feature_importance.to_csv(f"models/smote_pso_{name}_feature_importance.csv", index = False)

    print(feature_importance.head(10))  

def PCA_reduced(model):

    X_train_pca = pd.read_csv('data/processed/X_train_pca.csv')
    y_train = pd.read_csv('data/processed/y_train_resampled.csv')['Churn']

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
                      'max_features': [0.05, 0.1, 0.2]
                      }
        case 'xgb':
            model = XGBClassifier(device = 'cuda', random_state = 42)
            name = 'xgb'
            params = {'n_estimators': [25, 50, 100],
                      'max_depth': [2, 3, 4],
                      'learning_rate': [0.001, 0.005, 0.01],
                      'subsample': [0.2, 0.5],
                      'colsample_bytree': [0.2, 0.3],
                      'gamma': [0, 1, 5],
                      'reg_alpha': [0, 0.1, 1],
                      'reg_lambda': [1, 5, 10]
                     }  

    grid = GridSearchCV(model, param_grid = params, scoring = 'f1',
                           cv = 4)

    grid.fit(X_train_pca, y_train)

    safe_saver(grid.best_estimator_, 'models/', f"smote_best_pca_model_{name}")
    safe_saver(grid.best_params_, 'models/', f"smote_best_pca_params_{name}")

    y_pred = grid.predict(X_train_pca)

    train_f1 = f1_score(y_train, y_pred, average = 'weighted')
    train_roc = roc_auc_score(y_train, y_pred)

    X_test = pd.read_csv('data/processed/X_test_pca.csv')
    y_test = pd.read_csv('data/processed/y_test.csv')['Churn']

    y_pred = grid.predict(X_test)
    test_f1 = f1_score(y_test, y_pred, average = 'weighted')
    test_roc = roc_auc_score(y_test, y_pred)

    results = {'train_f1' : train_f1,
               'train_roc' : train_roc,
               'test_f1' : test_f1,
               'test_roc' : test_roc}

    safe_saver(results, 'metrics/', f"smote_best_pca_metrics_{name}")

    print(f"Training F1-Score: {round(train_f1, 3)}\nTest F1-Score: {round(test_f1, 3)}")
    print(f"Training ROC-AUC Score: {round(train_roc, 3)}\nTest ROC-AUC: {round(test_roc, 3)}")

def run_script(function, model):

    match function:
        case 'rfe':
            RFECV_reduced(model)
        case 'pso':
            PSO_reduced(model)
        case 'pca':
            PCA_reduced(model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Train logistic regression on SMOTENC or PCA reduced data')
    parser.add_argument('--function', default = 'rfe', help = 'Default rfe. Choose rfe, pso, or pca.')
    parser.add_argument('--model', default = 'log', help = 'Default log. Choose either log, rf, or xgb.')
    arg = parser.parse_args()

    run_script(arg.function, arg.model)
