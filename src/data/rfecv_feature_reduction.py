from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from data_utils import safe_saver
import pandas as pd
import argparse

def RFECV_reduction(model, sample):
    """
    This function performs RFECV feature reduction and saves the selected features.
    Args:
        model: 'log', 'rf', or 'xgb'. Selects the model to use for RFECV reduction.
        sample: scaled or 'smote'. Selects the data to use for RFECV reduction.
    """

    """
    Matches the correct model, loads in the data, then performs RFECV() with the
    selected model. I set step = 1, such that only one feature is removed at each step
    because this is more precises, although more time consuming. I also using scoring =
    'roc_auc' because this is a very unbalanced set with binary classification, which
    the roc-auc score is often very robust for.
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
    Loads in the correct data for the selected sample.
    """
    if sample == 'scaled':
        X_train = pd.read_csv('data/processed/scaled/X_train_scaled.csv')
        y_train = pd.read_csv('data/processed/y_train.csv')['Churn']
    elif sample == 'smote':
        X_train = pd.read_csv('data/processed/smote/X_train_smote.csv')
        y_train = pd.read_csv('data/processed/smote/y_train_smote.csv')['Churn']
    else:
        raise ValueError(f"Unsupported sample selected: '{sample}'. Choose 'scaled' or 'smote'.")

    selector = RFECV(estimator = model, step = 1, cv = StratifiedKFold(5),
                     scoring = 'roc_auc')

    selector.fit(X_train, y_train)
    features = X_train.columns[selector.support_]
    safe_saver(features, f"data/processed/{sample}/", f"RFECV_features_{name}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'just a simple argument parser.')
    parser.add_argument('--model', required = True, help = 'model: log, rf, or xgb')
    parser.add_argument('--sample', required = True, help = 'sample: scaled or smote')

    arg = parser.parse_args()

    RFECV_reduction(arg.model, arg.sample)