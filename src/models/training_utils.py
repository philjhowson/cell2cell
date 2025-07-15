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

def train_log_rf(params, X_train, y_train, X_test, y_test, model):
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

    print(f"Training {model} model...")

    grid = GridSearchCV(classifier, param_grid = params, scoring = 'roc_auc', cv = 4)

    grid.fit(X_train, y_train)

    print(f"Evaluating {model} model...")

    y_pred = grid.predict(X_train)
    y_proba = grid.predict_proba(X_train)[:, 1]

    train_f1 = f1_score(y_train, y_pred, average = 'weighted')
    train_roc = roc_auc_score(y_train, y_proba)

    y_pred = grid.predict(X_test)
    y_proba = grid.predict_proba(X_test)[:, 1]
    test_f1 = f1_score(y_test, y_pred, average = 'weighted')
    test_roc = roc_auc_score(y_test, y_proba)

    results = {'train_f1' : train_f1,
               'train_roc' : train_roc,
               'test_f1' : test_f1,
               'test_roc' : test_roc}

    print("Best params:", grid.best_params_)

    print(f"Training F1-Score: {round(train_f1, 3)}\nTest F1-Score: {round(test_f1, 3)}")
    print(f"Training ROC-AUC Score: {round(train_roc, 3)}\nTest ROC-AUC: {round(test_roc, 3)}")

    """
    selects the correct feature importance bestric from best_estimator_ and creates dataframe with
    feature importance scores and feautre names. If the model is log, a column with absolute values
    is created the the dataframe is sorted based on the absolute values.
    """

    match model:
        case 'log':
            importances = grid.best_estimator_.coef_[0]
        case 'rf':
            importances = grid.best_estimator_.feature_importances_

    feature_names = X_train.columns
    feature_importance = pd.DataFrame(zip(feature_names, importances), columns = ['Feature', 'Importance'])
    if model == 'log':
        feature_importance['Absolute'] = abs(feature_importance['Importance'])
        feature_importance.sort_values(by = 'Absolute', ascending = False, inplace = True)
    else:
        feature_importance.sort_values(by = 'Importance', ascending = False, inplace = True)
    
    print('Feature Importances:')
    print(feature_importance.head(10))

    return grid.best_estimator_, grid.best_params_, results, feature_importance

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

def train_xgb(params, X_train, y_train, X_test, y_test):
    """
    This performs training and testing for either log or rf, based on the selected
    model argument.
        Args:
            params: a dictionary of hyperparameters for grid search.
            X_train: Training data. Expects a df.
            y_train: Training label. Expects 1d vector of labels (pd.Series or otherwise)
            X_test: Test data. Expects a df.
            y_test: Test labels. Expects 1d vector of labels (pd.Series or otherwise)
    """
    """
    Splits the training data into a training and evaluation set for xgb and creates the DMatrix() 
    objects needed to move the data to the GPU along with the model. Also creates the evals object
    so that early_stopping_rounds can be effectively used.
    """
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size = 0.3, stratify = y_train)

    dtrain = xgb.DMatrix(X_tr, label = y_tr)
    dval = xgb.DMatrix(X_val, label = y_val)
    evals = [(dval, 'validation')]

    """
    Essentially a custom GridSearchCV loop. Values are taken from the params dictionary
    and the Cartesian product is generated such that all possible combinations are produced.
    Trains the model and records the best score, the model params used to get that score
    and the best iteration during traning.
    """

    best_score = 0
    best_params = None

    for values in product(*params.values()):
        model_params = dict(zip(params.keys(), values))

        model = xgb.train(model_params, dtrain, num_boost_round = 10000, evals = evals, 
                              early_stopping_rounds = 20, verbose_eval = False,
                              obj = focal_loss(alpha = 0.25, gamma = 2.0))

        score = model.best_score

        if score > best_score:
                best_score = score
                best_params = model_params
                best_iter = model.best_iteration

    """
    records best params and the best iteration, then retrains the model
    with those parameters and evaluates the training and test dataset.
    Results are recorded and then returned with feature importances.
    """

    best_config = {'params': best_params, 'num_boost_rounds': best_iter}

    dtrain = xgb.DMatrix(X_train, label = y_train, feature_names = list(X_train.columns))        

    model = xgb.train(best_params, dtrain, num_boost_round = best_iter, 
                      verbose_eval = False, obj = focal_loss(alpha = 0.25, gamma = 2.0))

    dtrain = xgb.DMatrix(X_train)

    y_pred_probs = model.predict(dtrain)
    y_pred = (y_pred_probs >= 0.5).astype(int)

    train_f1 = f1_score(y_train, y_pred, average = 'weighted')
    train_roc = roc_auc_score(y_train, y_pred_probs)

    dtest = xgb.DMatrix(X_test)

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

    return model, best_config, results, feature_importance