import torch.nn as nn
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from itertools import product
import pandas as pd
from typing import Dict, Any
import xgboost as xgb
import numpy as np
import pickle
import copy
import os

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

def safe_loader(path):
    """
    A simple loading function for .pkl files.

    args:
        path: the path, including the file name and extension
        of the desired file.
    """

    with open(path, 'rb') as f:
        return pickle.load(f)
    
def get_models_and_features(models, path_to_models, path_to_features):

    model_set = {}

    for model in models:
        filename = '_features_'.join(model.split('_')[:2]) + '.pkl'
        name = '_'.join(model.split('_')[:2])

        if os.path.exists(f"{path_to_features}/{filename}"):
            features = safe_loader(f"{path_to_features}/{filename}")
        else:
            features = None

        model_set[name] = {}
        model_set[name]['model'] = safe_loader(f"{path_to_models}/{model}")
        model_set[name]['features'] = features

    return model_set
    
def train_log_rf(model, params, X_train = None, y_train = None, X_test = None, y_test = None,
                 folder = None, iteration = None, name = None, reduction = None):

    if reduction:
        if reduction == 'pca':
            X_train = pd.read_csv(f"data/processed/{folder}/X_train_{folder}_pca.csv")
            X_test = pd.read_csv(f"data/processed/{folder}/X_test_{folder}_pca.csv")
        else:
            cols_to_keep = safe_loader(f"data/processed/{folder}/{reduction}_features_{name}.pkl")
            X_train = X_train[cols_to_keep]
            X_test = X_test[cols_to_keep]

    grid = GridSearchCV(model, param_grid = params, scoring = 'roc_auc', cv = 4)

    grid.fit(X_train, y_train)

    best_params = grid.best_params_

    safe_saver(grid.best_estimator_, f"models/{folder}/", f"{iteration}_{name}_model")
    safe_saver(grid.best_params_, f"models/{folder}/", f"{iteration}_{name}_params")

    y_pred = grid.predict(X_train)
    y_pred_proba = grid.predict_proba(X_train)[:, 1]

    train_f1 = f1_score(y_train, y_pred, average = 'weighted')
    train_roc = roc_auc_score(y_train, y_pred_proba)

    y_pred = grid.predict(X_test)
    y_pred_proba = grid.predict_proba(X_test)[:, 1]
    test_f1 = f1_score(y_test, y_pred, average = 'weighted')
    test_roc = roc_auc_score(y_test, y_pred_proba)

    results = {'train_f1' : train_f1,
               'train_roc' : train_roc,
               'test_f1' : test_f1,
               'test_roc' : test_roc}

    safe_saver(results, f"metrics/{folder}/", f"{iteration}_scores_{name}")

    print("Best params:", best_params)

    print(f"Training F1-Score: {round(train_f1, 3)}\nTest F1-Score: {round(test_f1, 3)}")
    print(f"Training ROC-AUC Score: {round(train_roc, 3)}\nTest ROC-AUC: {round(test_roc, 3)}")

    match name:
        case 'log':
            importances = grid.best_estimator_.coef_[0]
        case _:
            importances = grid.best_estimator_.feature_importances_

    feature_names = X_train.columns
    feature_importance = pd.DataFrame(zip(feature_names, importances), columns = ['Feature', 'Importance'])
    if name == 'log':
        feature_importance['Absolute'] = abs(feature_importance['Importance'])
        feature_importance.sort_values(by = 'Absolute', ascending = False, inplace = True)
    else:
        feature_importance.sort_values(by = 'Importance', ascending = False, inplace = True)
    feature_importance.to_csv(f"models/{folder}/{iteration}_feature_importance_{name}.csv", index = False)
    
    print('Feature Importances:')
    print(feature_importance.head(10))

def pso_train_log_rf(model, params, X_train = None, y_train = None, X_test = None, y_test = None,
                     folder = None, name = None):
    
    pso_values = safe_loader(f"data/processed/{folder}/pso_feature_set_{name}.pkl")

    results = {}

    for item in pso_values:

        feature_indices = pso_values[item]['features']
        cols_to_keep = X_train.columns[feature_indices]
        X = X_train[cols_to_keep]

        grid = GridSearchCV(model, param_grid = params, scoring = 'roc_auc',
                            cv = 4)

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
    X_test = X_test[cols_to_keep]
    best_params = results[best_roc]['params']

    safe_saver(cols_to_keep, f"data/processed/{folder}/", f"pso_features_{name}")

    model.set_params(**best_params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_train)
    y_pred_proba = model.predict_proba(X_train)[:, 1]

    train_f1 = f1_score(y_train, y_pred, average = 'weighted')
    train_roc = roc_auc_score(y_train, y_pred_proba)

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    test_f1 = f1_score(y_test, y_pred, average = 'weighted')
    test_roc = roc_auc_score(y_test, y_pred_proba)

    results = {'train_f1' : train_f1,
               'train_roc' : train_roc,
               'test_f1' : test_f1,
               'test_roc' : test_roc}

    safe_saver(results, f"metrics/{folder}/", f"pso_scores_{name}")

    safe_saver(model, f"models/{folder}/", f"pso_{name}_model")
    safe_saver(best_params, f"models/{folder}/", f"pso_{name}_params")

    print(f"Best PSO model: {best_roc}\n"
          f"Features: {list(X_train.columns)}\n"
          f"Hyperparameters: {best_params}\n"
          f"F1-Scores: {round(results['train_f1'], 3)}; {round(results['test_f1'], 3)}\n"
          f"ROC-AUCs: {round(results['train_roc'], 3)}; {round(results['test_roc'], 3)}")

    match name:
        case 'log':
            importances = model.coef_[0]
        case _:
            importances = model.feature_importances_

    feature_names = X_train.columns
    feature_importance = pd.DataFrame(zip(feature_names, importances), columns = ['Feature', 'Importance'])
    if name == 'log':
        feature_importance['Absolute'] = abs(feature_importance['Importance'])
        feature_importance.sort_values(by = 'Absolute', ascending = False, inplace = True)
    else:
        feature_importance.sort_values(by = 'Importance', ascending = False, inplace = True)
    feature_importance.to_csv(f"models/{folder}/pso_feature_importance_{name}.csv", index = False)

    print(feature_importance.head(10))  

def focal_loss(alpha = 0.25, gamma = 2.0):
    def fl_obj(preds, dtrain):
        labels = dtrain.get_label()
        preds = 1.0 / (1.0 + np.exp(-preds))

        grad = alpha * (labels - preds) * ((1 - preds) ** gamma)
        hess = alpha * preds * (1 - preds) * ((1 - preds) ** gamma)

        return -grad, hess

    return fl_obj

def train_xgb(params, X_train = None, y_train = None, X_test = None, y_test = None,
              folder = None, iteration = None, name = None, reduction = None):

    if reduction:
        if reduction == 'pca':
            X_train = pd.read_csv(f"data/processed/{folder}/X_train_{folder}_pca.csv")
            X_test = pd.read_csv(f"data/processed/{folder}/X_test_{folder}_pca.csv")
        else:
            cols_to_keep = safe_loader(f"data/processed/{folder}/{reduction}_features_{name}.pkl")
            X_train = X_train[cols_to_keep]
            X_test = X_test[cols_to_keep]

    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size = 0.3, stratify = y_train)

    dtrain = xgb.DMatrix(X_tr, label = y_tr)
    dval = xgb.DMatrix(X_val, label = y_val)
    evals = [(dval, 'validation')]

    best_score = 0
    best_params = None

    for values in product(*params.values()):
        model_params = dict(zip(params.keys(), values))
        model_params.update({'eval_metric': 'auc', 'device': 'cuda',
                             'random_state': 42})

        model = xgb.train(model_params, dtrain, num_boost_round = 10000, evals = evals, 
                              early_stopping_rounds = 20, verbose_eval = False,
                              obj = focal_loss(alpha = 0.25, gamma = 2.0))

        score = model.best_score

        if score > best_score:
                best_score = score
                best_params = model_params
                best_iter = model.best_iteration

    best_config = {'params': best_params, 'num_boost_rounds': best_iter}
    safe_saver(best_config, f"models/{folder}/", f"{iteration}_{name}_params")

    dtrain = xgb.DMatrix(X_train, label = y_train, feature_names = list(X_train.columns))        

    model = xgb.train(best_params, dtrain, num_boost_round = best_iter, 
                      verbose_eval = False, obj = focal_loss(alpha = 0.25, gamma = 2.0))

    safe_saver(model, f"models/{folder}/", f"{iteration}_{name}_model") 

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

    safe_saver(results, f"metrics/{folder}", f"{iteration}_scores_{name}")   

    print("Best params:", best_params)

    print(f"Training F1-Score: {round(train_f1, 3)}\nTest F1-Score: {round(test_f1, 3)}")
    print(f"Training ROC-AUC Score: {round(train_roc, 3)}\nTest ROC-AUC: {round(test_roc, 3)}")

    importances = model.get_score()
    feature_importance = pd.DataFrame(importances.items(), columns = ['Feature', 'Importance'])
    feature_importance.sort_values(by = 'Importance', ascending = False, inplace = True)
    feature_importance.to_csv(f"models/{folder}/{iteration}_feature_importance_{name}.csv", index = False)

    print('Feature Importances:')
    print(feature_importance.head(10))

def pso_train_xgb(params, X_train = None, y_train = None, X_test = None, y_test = None,
                     folder = None, name = None):
    
    pso_values = safe_loader(f"data/processed/{folder}/pso_feature_set_{name}.pkl")
    
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size = 0.3, stratify = y_train)

    results = {}

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

            if test_roc > best_score:
                best_score = train_roc

                best_train_f1 = train_f1
                best_train_roc = train_roc
                best_test_f1 = test_f1
                best_test_roc = test_roc
                best_params = model_params
                best_iter = model.best_iteration

        results[item] = {'train_f1' : best_train_f1,
                         'train_roc' : best_train_roc,
                         'test_f1' : best_test_f1,
                         'test_roc' : best_test_roc,
                         'params' : best_params,
                         'best_iter' : best_iter, 
                         'features' : cols_to_keep}

    best_roc = max(results, key = lambda k: results[k]['test_roc'])
    feature_indices = pso_values[best_roc]['features']
    cols_to_keep = X_train.columns[feature_indices]
    best_params = results[best_roc]['params']

    safe_saver(cols_to_keep, f"data/processed/{folder}/", f"pso_features_{name}")

    X = X_train[cols_to_keep]
    X_test_reduced = X_test[cols_to_keep]
    dtrain = xgb.DMatrix(X, label = y_train, feature_names = list(X.columns))
    dtest = xgb.DMatrix(X_test_reduced, label = y_test)

    best_config = {'params': best_params, 'num_boost_rounds': best_iter}
    safe_saver(best_config, f"models/{folder}/", f"pso_{name}_params")
       
    model = xgb.train(best_params, dtrain, num_boost_round = best_iter, 
                      verbose_eval = False, obj = focal_loss(alpha = 0.25, gamma = 2.0))

    safe_saver(model, f"models/{folder}/", f"pso_{name}_model") 

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

    safe_saver(results, f"metrics/{folder}", f"pso_scores_{name}")   

    print("Best params:", best_params)

    print(f"Training F1-Score: {round(train_f1, 3)}\nTest F1-Score: {round(test_f1, 3)}")
    print(f"Training ROC-AUC Score: {round(train_roc, 3)}\nTest ROC-AUC: {round(test_roc, 3)}")

    importances = model.get_score()
    feature_importance = pd.DataFrame(importances.items(), columns = ['Feature', 'Importance'])
    feature_importance.sort_values(by = 'Importance', ascending = False, inplace = True)
    feature_importance.to_csv(f"models/{folder}/pso_feature_importance_{name}.csv", index = False)

    print('Feature Importances:')
    print(feature_importance.head(10))

def model_predictions(train: pd.DataFrame, test: pd.DataFrame, train_pca: pd.DataFrame,
                      test_pca: pd.DataFrame, dictionary: Dict[str, Any], name: str = None):

    training_preds = {}
    training_proba = {}
    test_preds = {}
    test_proba = {}

    for item in dictionary:

        model = dictionary[item].get('model')
        features = dictionary[item].get('features')

        if features is not None and len(features) > 0:
            X_train = train[features].copy()
            X_test = test[features].copy()
        elif item.startswith('pca'):
            X_train = train_pca.copy()
            X_test = test_pca.copy()
        else:
            X_train = train.copy()
            X_test = test.copy()

        if isinstance(model, xgb.Booster):
            X_train = xgb.DMatrix(X_train)
            X_test = xgb.DMatrix(X_test)

            raw_scores = model.predict(X_train)
            train_probs = 1 / (1 + np.exp(-raw_scores))
            train_predictions = (train_probs > 0.5).astype(int)

            raw_scores = model.predict(X_test)
            test_probs = 1 / (1 + np.exp(-raw_scores))
            test_predictions = (test_probs > 0.5).astype(int)

        else:
            train_predictions = model.predict(X_train)
            train_probs = model.predict_proba(X_train)[:, 1]

            test_predictions = model.predict(X_test)
            test_probs = model.predict_proba(X_test)[:, 1]    

        if name == 'smote':
            training_preds[f"smote_{item}"] = train_predictions
            training_proba[f"smote_{item}"] = train_probs

            test_preds[f"smote_{item}"] = test_predictions
            test_proba[f"smote_{item}"] = test_probs

        else:
            training_preds[item] = train_predictions
            training_proba[item] = train_probs

            test_preds[item] = test_predictions
            test_proba[item] = test_probs

    return pd.DataFrame(training_preds), pd.DataFrame(training_proba), pd.DataFrame(test_preds), pd.DataFrame(test_proba)

class ChurnNet(nn.Module):
    def __init__(self, input_size):
        """
        initializes an FNN with hidden layers, and ReLU() activations.
        Uses the default Kaiming weight initializations.
        args:
            input_size: must be set. Dynamic setting is possible with
            df.shape[1].
        """
        super(ChurnNet, self).__init__()
        self.ChurnNet = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1))

    def forward(self, x):
        x = self.ChurnNet(x)
        return x
    
class ShallowNet(nn.Module):
    def __init__(self, input_size):
        """
        initializes an FNN with shallow architecture, and ReLU()
        activations. Uses the default Kaiming weight initializations.
        args:
            input_size: must be set. Dynamic setting is possible with
            df.shape[1].
        """
        super(ShallowNet, self).__init__()
        self.ShallowNet = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1))

    def forward(self, x):
        x = self.ShallowNet(x)
        return x
    
class EarlyStop():
    def __init__(self, patience):
        self.best_f1 = 0
        self.best_roc = 0
        self.counter = 0
        self.best_val = float('inf')
        self.patience = patience
        self.best_model = None

    def early_stop(self, model, val, f1, roc):
        if val < self.best_val:
            self.best_val = val
            self.counter = 0
        else:
            self.counter += 1
        if roc > self.best_roc:
            self.best_model = copy.deepcopy(model)
        if self.counter > self.patience:
            return True
        return False

def train_net(X, y, epochs, learning_rate, model):

    X_train, X_val, y_train, y_val = train_test_split(X, y, stratify = y, test_size = 0.3)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if model == 'ChurnNet':
        model = ChurnNet(X_train.shape[1])
    elif model == 'ShallowNet':
        model = ShallowNet(X_train.shape[1])
    model.to(device)
    optimizer = Adam(model.parameters(), lr = learning_rate)
    BCE_loss = nn.BCELoss()
    scheduler = ReduceLROnPlateau(optimizer, mode = 'min', 
                              factor = 0.1, patience = 3)
    EarlyStopper = EarlyStop(patience = 6)

    X_train = torch.tensor(X_train.values, dtype = torch.float32)
    X_val = torch.tensor(X_val.values, dtype = torch.float32)
    y_train = torch.tensor(y_train.values, dtype = torch.float32)
    y_val = torch.tensor(y_val.values, dtype = torch.float32)

    train = TensorDataset(X_train, y_train)
    val = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train, batch_size = 128, shuffle = True)
    val_loader = DataLoader(val, batch_size = 128, shuffle = False)

    history = {'train_loss' : [], 'train_grad' : [], 'train_f1' : [], 'train_roc' : [],
               'val_loss' : [], 'val_f1' : [], 'val_roc' : [], 'epoch' : []}

    for epoch in range(epochs):
        predictions = []
        labels = []
        all_grads = []
        train_loss = 0
        history['epoch'].append(epoch + 1)

        model.train()

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = torch.sigmoid(model(X_batch).squeeze(dim = 1))
            loss = BCE_loss(outputs, y_batch.float())
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

            predictions.extend(outputs.cpu().tolist())
            labels.extend(y_batch.cpu().tolist())

            for param in model.parameters():
                if param.grad is not None:
                    all_grads.append(param.grad.detach().abs().mean().item())

        mean_grad = sum(all_grads) / len(all_grads) if all_grads else 0.0
        avg_train_loss = train_loss / len(train_loader)
        history['train_grad'].append(mean_grad)
        history['train_loss'].append(avg_train_loss)

        pred_labels = [1 if p >= 0.5 else 0 for p in predictions]
        train_f1 = f1_score(labels, pred_labels, average = 'weighted')
        train_roc = roc_auc_score(labels, pred_labels)
        history['train_f1'].append(train_f1)
        history['train_roc'].append(train_roc)

        model.eval()
        val_predictions = []
        val_labels = []
        val_loss = 0

        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            outputs = torch.sigmoid(model(X_batch).squeeze(dim = 1))
            loss = BCE_loss(outputs, y_batch.float())
            val_loss += loss.item()

            val_predictions.extend(outputs.cpu().tolist())
            val_labels.extend(y_batch.cpu().tolist())

        avg_val_loss = val_loss / len(val_loader)
        history['val_loss'].append(avg_val_loss)

        pred_labels = [1 if p >= 0.5 else 0 for p in val_predictions]
        val_f1 = f1_score(val_labels, pred_labels, average = 'weighted')
        val_roc = roc_auc_score(val_labels, pred_labels)
        history['val_f1'].append(val_f1)
        history['val_roc'].append(val_roc)

        print(f"Training F1-Score: {round(train_f1, 3)}; Training ROC-AUC: {round(train_roc, 3)}; Training Loss: {round(avg_train_loss, 3)}")
        print(f"Training Gradient: {round(mean_grad, 3)}")
        print(f"Validation F1-Score: {round(val_f1, 3)}; Validation ROC-AUC: {round(val_roc, 3)}; Validation Loss: {round(avg_val_loss, 3)}")

        scheduler.step(avg_val_loss)
        if EarlyStopper.early_stop(model, avg_val_loss, val_f1, val_roc):
            print(f"Early stopping at epoch {epoch}.")
            break

    return EarlyStopper.best_model, history

def test_net(X, y, model):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if model == 'ChurnNet':
        model = ChurnNet(X.shape[1])
        model.load_state_dict(torch.load('models/ChurnNet_weights.pth', weights_only = True))
    elif model == 'ShallowNet':
        model = ShallowNet(X.shape[1])
        model.load_state_dict(torch.load('models/ShallowNet_weights.pth', weights_only = True))
    model.to(device)

    X_test = torch.tensor(X.values, dtype = torch.float32)
    y_test = torch.tensor(y.values, dtype = torch.float32)

    test = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test, batch_size = 128, shuffle = False)

    test_preds = []
    test_labels = []

    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        outputs = torch.sigmoid(model(X_batch).squeeze(dim = 1))

        test_preds.extend(outputs.cpu().tolist())
        test_labels.extend(y_batch.cpu().tolist())

    pred_labels = [1 if p >= 0.5 else 0 for p in test_preds]
    test_f1 = f1_score(test_labels, pred_labels, average = 'weighted')
    test_roc = roc_auc_score(test_labels, pred_labels)
    conf_mat = confusion_matrix(test_labels, pred_labels, normalize = 'all')
    class_report = classification_report(test_labels, pred_labels, output_dict = True)

    test_scores = {'F1' : test_f1,
                   'test-roc' : test_roc}

    print(f"Test F1-Score: {round(test_f1, 3)}; Test ROC-AUC: {round(test_roc, 3)}")
    print(f"Confusion Matrix:\n{conf_mat}")
    print(f"Classification Report:\n{pd.DataFrame(class_report).transpose().round(3)}")

    return test_scores, conf_mat, class_report