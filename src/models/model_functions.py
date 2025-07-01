import torch.nn as nn
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from itertools import product
import pandas as pd
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

    safe_saver(grid.best_estimator_, f"models/{folder}/", f"{iteration}_best_{name}_model")
    safe_saver(grid.best_params_, f"models/{folder}/", f"{iteration}_best_{name}_params")

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
    
    pso_values = safe_loader(f"data/processed/{folder}/pso_features_{name}.pkl")

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
    safe_saver(best_params, f"models/{folder}/", f"pso_best_{name}_params")

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
    safe_saver(best_config, f"models/{folder}/", f"{iteration}_best_{name}_params")

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
    
    pso_values = safe_loader(f"data/processed/{folder}/pso_features_{name}.pkl")
    
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

    X = X_train[cols_to_keep]
    X_test_reduced = X_test[cols_to_keep]
    dtrain = xgb.DMatrix(X, label = y_train, feature_names = list(X.columns))
    dtest = xgb.DMatrix(X_test_reduced, label = y_test)

    best_config = {'params': best_params, 'num_boost_rounds': best_iter}
    safe_saver(best_config, f"models/{folder}/", f"pso_best_{name}_params")
       
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

class ChurnNet(nn.Module):
    def __init__(self, input_size = 3):
        """
        initializes an FNN with multiple hidden layers, and ReLU()
        activations. Uses the default Kaiming weight initializations.
        args:
            input_size: default is 3, which refers to how many model
            outputs are stacked in the ensemble model. 1 feature for
            each set of model predictions.
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

def train_ChurnNet(df, epochs, name):

    X = df.drop(columns = 'Churn')
    y = df['Churn']

    X_train, X_val, y_train, y_val = train_test_split(X, y, stratify = y, test_size = 0.3)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ChurnNet(X_train.shape[1])
    model.to(device)
    optimizer = Adam(model.parameters(), lr = 0.001)
    BCE_loss = nn.BCELoss()
    scheduler = ReduceLROnPlateau(optimizer, mode = 'min', 
                              factor = 0.1, patience = 3, 
                              verbose = True)
    EarlyStopper = EarlyStop(patience = 6)

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
        batch = 0
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
        avg_loss = train_loss / len(train_loader)
        history['train_grad'].append(mean_grad)
        history['train_loss'].append(avg_loss)

        pred_labels = [1 if p >= 0.5 else 0 for p in predictions]
        train_f1 = f1_score(labels, pred_labels, average = 'weights')
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

        avg_loss = val_loss / len(val_loader)
        history['val_loss'].append(avg_loss)

        pred_labels = [1 if p >= 0.5 else 0 for p in val_predictions]
        val_f1 = f1_score(val_labels, pred_labels, average = 'weights')
        val_roc = roc_auc_score(val_labels, pred_labels)
        history['val_f1'].append(val_f1)
        history['val_roc'].append(val_roc)

        scheduler.step(avg_loss)
        if EarlyStopper.early_stop(model, avg_loss, val_f1, val_roc):
            print(f"Early stopping at epoch {epoch}")
            break

    return EarlyStopper.best_model, history

