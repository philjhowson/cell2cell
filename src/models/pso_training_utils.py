from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import numpy as np
from itertools import product
from general_utils import safe_loader
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import pandas as pd

def load_data(model, sample, strategy):
    """
    loads in the data for the training and testing. If rfecv is selected, it will
    also select the relevant columns from the dataframes.
        Args:
            model: model being loaded, required to retrieve the correct feature
                sets for reduced data. log, rf, xgb.
            sample: scaled or smote.
            strategy: feature reduction technique. baseline, rfecv, pso, pca.
    """
    if strategy != 'pca':
        if sample == 'scaled':
            X_train = pd.read_csv('data/processed/scaled/X_train_scaled.csv')
            y_train = pd.read_csv('data/processed/y_train.csv')['Churn']
            X_test = pd.read_csv('data/processed/scaled/X_test_scaled.csv')
        elif sample == 'smote':
            X_train = pd.read_csv('data/processed/smote/X_train_smote.csv')
            y_train = pd.read_csv('data/processed/smote/y_train_smote.csv')['Churn']
            X_test = pd.read_csv('data/processed/scaled/X_test_scaled.csv')
        else:
            raise ValueError(f"Invalid sample selected: {sample}")
    else:
        if sample == 'scaled':
            X_train = pd.read_csv('data/processed/scaled/X_train_scaled_pca.csv')
            y_train = pd.read_csv('data/processed/y_train.csv')['Churn']
            X_test = pd.read_csv('data/processed/scaled/X_test_scaled_pca.csv')
        elif sample == 'smote':
            X_train = pd.read_csv('data/processed/smote/X_train_smote_pca.csv')
            y_train = pd.read_csv('data/processed/smote/y_train_smote.csv')['Churn']
            X_test = pd.read_csv('data/processed/scaled/X_test_scaled_pca.csv')

    y_test = pd.read_csv('data/processed/y_test.csv')['Churn']

    if strategy == 'baseline':
        pass
    elif strategy == 'rfecv':
        cols_to_keep = safe_loader(f"data/processed/{sample}/RFECV_features_{model}.pkl")
        X_train = X_train[cols_to_keep]
        X_test = X_test[cols_to_keep]

    return X_train, y_train, X_test, y_test

def focal_loss(alpha = 0.6, gamma = 2.5):
    """
    Focal loss function for xgb models.
    Args:
        alpha: float. Increase alpha to penalize misclassification of minority class.
        gamma: float. Increase gamma to penalize hard-to-classify examples more heavily.
    """
    def fl_obj(preds, dtrain):
        labels = dtrain.get_label()
        preds = 1.0 / (1.0 + np.exp(-preds))
        """
        custom grad and hessian values for the focal loss functions
        that moderates the adjustment and smoothness of the adjustment
        for the xgb model. (labels - preds) determines how wrong the
        predictions are, and alpha and gamma adjust the weights put on
        misclassification of the minority class or hard-to-classify
        examples.
        """
        grad = alpha * (labels - preds) * ((1 - preds) ** gamma)
        """
        preds * (1 - preds) models the sensitivity of the prediction.
        Hessian values essentially determine how large a step size
        should be taken in the direction determined by the gradient.
        """
        hess = alpha * preds * (1 - preds) * ((1 - preds) ** gamma)

        return -grad, hess

    return fl_obj

def pso_train_log_rf(params, X_train, y_train, X_test, y_test, sample, model):
    """
    This performs training and testing for either log or rf, based on the selected
    model argument.
        Args:
            params: a dictionary of hyperparameters for grid search.
            X_train: Training data. Expects a df.
            y_train: Training label. Expects 1d vector of labels (pd.Series or otherwise)
            X_test: Test data. Expects a df.
            y_test: Test labels. Expects 1d vector of labels (pd.Series or otherwise)
            model: 'log' or 'rf' 
    """
    """
    selects the correct classifier based on the model argument and then performs grid search,
    and the produces training and test f1 and roc-auc scores.
    """
    if model == 'log':
        classifier = LogisticRegression(max_iter = 1000, solver = 'liblinear')
    else:
        classifier = RandomForestClassifier(n_estimators = 10, random_state = 42, n_jobs = 4)
    """
    This loads in the pso features sets generated from the pso stage and performs a type of
    grid search to determine which feature set works best with the data. Models are
    evaluated and the results, parameters and features are recorded.
    """
    pso_values = safe_loader(f"data/processed/{sample}/pso_feature_set_{model}.pkl")

    pso_results = {}

    for item in pso_values:

        feature_indices = pso_values[item]['features']
        cols_to_keep = X_train.columns[feature_indices]
        X = X_train[cols_to_keep]

        grid = GridSearchCV(classifier, param_grid = params, scoring = 'roc_auc', cv = 4)

        grid.fit(X, y_train)
        y_pred = grid.predict(X)
        y_proba = grid.predict_proba(X)[:, 1]
        train_f1 = f1_score(y_train, y_pred, average = 'weighted')
        train_roc = roc_auc_score(y_train, y_proba)

        X = X_test[cols_to_keep]
        y_pred = grid.predict(X)
        y_proba = grid.predict_proba(X)[:, 1]
        test_f1 = f1_score(y_test, y_pred, average = 'weighted')
        test_roc = roc_auc_score(y_test, y_proba)

        pso_results[item] = {'train_f1' : train_f1,
                             'train_roc' : train_roc,
                             'test_f1' : test_f1,
                             'test_roc' : test_roc,
                             'params' : grid.best_params_,
                             'features' : list(cols_to_keep)}
        
    """
    finds the key for the highest roc-auc score on the test data, extracts
    the feature indices, then determines the columns to keep and drops
    irrelevant columns from training and test data. Finally, retrieves
    the best parameters.
    """
    best_roc = max(pso_results, key = lambda k: pso_results[k]['test_roc'])
    cols_to_keep = pso_results[best_roc]['features']
    X_train = X_train[cols_to_keep]
    X_test = X_test[cols_to_keep]
    best_params = pso_results[best_roc]['params']

    """
    retrains the classifier with the best parameters and feature set and
    reproduces the training and test scores. Then returns the classifier,
    the best parameters, final results, feature importances, and the
    pso 'grid search' results.
    """
    classifier.set_params(**best_params)
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_train)
    y_pred_proba = classifier.predict_proba(X_train)[:, 1]

    train_f1 = f1_score(y_train, y_pred, average = 'weighted')
    train_roc = roc_auc_score(y_train, y_pred_proba)

    y_pred = classifier.predict(X_test)
    y_pred_proba = classifier.predict_proba(X_test)[:, 1]
    test_f1 = f1_score(y_test, y_pred, average = 'weighted')
    test_roc = roc_auc_score(y_test, y_pred_proba)

    results = {'train_f1' : train_f1,
               'train_roc' : train_roc,
               'test_f1' : test_f1,
               'test_roc' : test_roc}

    print(f"Best PSO model: {best_roc}\n"
          f"Features: {cols_to_keep}\n"
          f"Hyperparameters: {best_params}\n"
          f"F1-Scores: {round(results['train_f1'], 3)}; {round(results['test_f1'], 3)}\n"
          f"ROC-AUCs: {round(results['train_roc'], 3)}; {round(results['test_roc'], 3)}")

    match model:
        case 'log':
            importances = classifier.coef_[0]
        case _:
            importances = classifier.feature_importances_

    feature_names = X_train.columns
    feature_importance = pd.DataFrame(zip(feature_names, importances), columns = ['Feature', 'Importance'])
    if model == 'log':
        feature_importance['Absolute'] = abs(feature_importance['Importance'])
        feature_importance.sort_values(by = 'Absolute', ascending = False, inplace = True)
    else:
        feature_importance.sort_values(by = 'Importance', ascending = False, inplace = True)

    print(feature_importance.head(10))

    best_params = {'parameters' : best_params,
                   'pso_specs' : best_roc,
                   'features' : list(cols_to_keep)}

    return classifier, best_params, results, feature_importance, pso_results

def pso_train_xgb(params, X_train, y_train, X_test, y_test, sample):
    """
    This performs training and testing for xgb, based on the selected
    model argument.
        Args:
            params: a dictionary of hyperparameters for grid search.
            X_train: Training data. Expects a df.
            y_train: Training label. Expects 1d vector of labels (pd.Series or otherwise)
            X_test: Test data. Expects a df.
            y_test: Test labels. Expects 1d vector of labels (pd.Series or otherwise)
    """
    """
    loads in the feature set the from pso stage, splits the data into a training and evaluation
    set, the performs a 'grid search' for each pso set and the parameters passed to this function.
    The best values for each pso feature set are recorded.
    """
    pso_values = safe_loader(f"data/processed/{sample}/pso_feature_set_xgb.pkl")
    
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size = 0.3, stratify = y_train)

    pso_results = {}

    for item in pso_values:

        feature_indices = pso_values[item]['features']
        cols_to_keep = X_train.columns[feature_indices]
        X = X_tr[cols_to_keep]
        X_v = X_val[cols_to_keep]
        X_test_reduced = X_test[cols_to_keep]

        dtrain = xgb.DMatrix(X, label = y_tr)
        dval = xgb.DMatrix(X_v, label = y_val)
        evals = [(dval, 'validation')]
        dtest = xgb.DMatrix(X_test_reduced, label = y_test)

        best_score = 0
        best_params = None

        for values in product(*params.values()):
            model_params = dict(zip(params.keys(), values))
            model_params.update({'eval_metric': 'auc', 'device': 'cuda',
                                'random_state': 42})

            model = xgb.train(model_params, dtrain, num_boost_round = 10000, evals = evals, 
                                early_stopping_rounds = 20, verbose_eval = False,
                                obj = focal_loss(alpha = 0.25, gamma = 2.0))

            y_pred_probs = model.predict(dtrain)
            y_pred = (y_pred_probs >= 0.5).astype(int)
            train_f1 = f1_score(y_tr, y_pred, average = 'weighted')
            train_roc = roc_auc_score(y_tr, y_pred_probs)

            y_pred_probs = model.predict(dtest)
            y_pred = (y_pred_probs >= 0.5).astype(int)
            test_f1 = f1_score(y_test, y_pred, average = 'weighted')
            test_roc = roc_auc_score(y_test, y_pred_probs)

            """
            simply updates the important values to be saved if the test
            roc is higher than whatever the current best roc is.
            """
            if test_roc > best_score:
                best_score = train_roc

                best_train_f1 = train_f1
                best_train_roc = train_roc
                best_test_f1 = test_f1
                best_test_roc = test_roc
                best_params = model_params
                best_iter = model.best_iteration

        pso_results[item] = {'train_f1' : best_train_f1,
                             'train_roc' : best_train_roc,
                             'test_f1' : best_test_f1,
                             'test_roc' : best_test_roc,
                             'params' : best_params,
                             'best_iter' : best_iter, 
                             'features' : list(cols_to_keep)}
    """
    finds the highest roc score among all the stored values and
    extracts the relevant feature set and parameters. Then
    creates the dtrain and dtest data using xgb.DMatrix() for GPU
    compatibility.
    """
    best_roc = max(pso_results, key = lambda k: pso_results[k]['test_roc'])
    cols_to_keep = pso_results[best_roc]['features']
    best_params = pso_results[best_roc]['params']

    X = X_train[cols_to_keep]
    X_test_reduced = X_test[cols_to_keep]
    dtrain = xgb.DMatrix(X, label = y_train, feature_names = list(X.columns))
    dtest = xgb.DMatrix(X_test_reduced, label = y_test)

    best_config = {'params': best_params, 'num_boost_rounds': best_iter}
    
    """
    retrains the classifier with the best parameters and feature set and
    reproduces the training and test scores. Then returns the classifier,
    the best parameters, final results, feature importances, and the
    pso 'grid search' results.
    """
    model = xgb.train(best_params, dtrain, num_boost_round = best_iter, 
                      verbose_eval = False, obj = focal_loss(alpha = 0.25, gamma = 2.0))

    y_pred_probs = model.predict(dtrain)
    y_pred = (y_pred_probs >= 0.5).astype(int)

    train_f1 = f1_score(y_train, y_pred, average = 'weighted')
    train_roc = roc_auc_score(y_train, y_pred_probs)

    y_pred_probs = model.predict(dtest)
    y_pred = (y_pred_probs >= 0.5).astype(int)
    test_f1 = f1_score(y_test, y_pred, average = 'weighted')
    test_roc = roc_auc_score(y_test, y_pred_probs)

    results = {'train_f1' : train_f1,
               'train_roc' : train_roc,
               'test_f1' : test_f1,
               'test_roc' : test_roc,
               'num_rounds' : best_iter}

    print("Best params:", best_params)

    print(f"Training F1-Score: {round(train_f1, 3)}\nTest F1-Score: {round(test_f1, 3)}")
    print(f"Training ROC-AUC Score: {round(train_roc, 3)}\nTest ROC-AUC: {round(test_roc, 3)}")

    importances = model.get_score()
    feature_importance = pd.DataFrame(importances.items(), columns = ['Feature', 'Importance'])
    feature_importance.sort_values(by = 'Importance', ascending = False, inplace = True)

    print('Feature Importances:')
    print(feature_importance.head(10))

    return model, best_config, results, feature_importance, pso_results