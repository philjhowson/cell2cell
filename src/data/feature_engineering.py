import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTENC
from sklearn.model_selection import train_test_split
from data_utils import safe_saver, safe_loader
import numpy as np

def feature_engineering():

    X = pd.read_csv('data/processed/X_train.csv')
    X_test = pd.read_csv('data/processed/X_test.csv')

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

    """
    PCA feature reduction. I selected 0.95 in order to maintain the variance as much
    as possible while still reducing the features.
    """ 

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

    """
    I make a list of numerical columns to extract the list of columns which are categorical.
    I estimated there were likely many more categorical columns, especially after
    OneHotEncoding, so it would be much more efficient to make a list of numerical_columns.
    To some degree, without a sheet of what every column represents, I have to guess. But,
    I assume that everywhere that a number represents yes or no, or likely some qualitative
    value (e.g., good, average, bad), that it should be treated as a category, especially
    for SMOTENC where values that are not categorical are treated differently and for
    OneHotEncoded items, I have to mark them as categorical so that SMOTEENC will make sure
    only actual values in the column (e.g., 0 or 1) will be returned.
    """

    numeric_columns = ['MonthlyRevenue', 'MonthlyMinutes', 'TotalRecurringCharge',
                       'DirectorAssistedCalls', 'OverageMinutes', 'RoamingClass',
                       'PercChangeMinutes', 'PercChangeRevenues', 'DroppedCalls',
                       'BlockedCalls', 'UnansweredCalls', 'CuustomerCareCalls',
                       'ThreewayCalls', 'ReceivedCalls', 'OutboundCalls', 'RoamingCalls',
                       'InboundCalls', 'PeakCallsInOut', 'OffPeakCallsInOut',
                       'DroppedBlockedCalls', 'CallForwardingCalls', 'CallWaitingCalls',
                       'MonthsInService', 'UniqueSubs', 'ActiveSubs', 'Handsets',
                       'CurrentEquipmentDays', 'AgeHH1', 'AgeHH2' 'RetentionOffersAccepted']

    categorical_columns = [col for col in X_scaled.columns if col not in numeric_columns]

    cat_features = [X_scaled.columns.get_loc(col) for col in categorical_columns]

    """
    I need to load in the training y data in order to be smote encoded.
    """
    y = pd.read_csv('data/processed/y_train.csv')

    smote = SMOTENC(categorical_features = cat_features, random_state = 42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

    X_resampled.to_csv('data/processed/smote/X_train_smote.csv', index = False)
    y_resampled.to_csv('data/processed/smote/y_train_smote.csv', index = False)

    print('Before Resampling:', len(X_scaled), 'After Resampling:', len(X_resampled)) 

    """
    A separate PCA encoder is fit on the smote data because the smote encoding may
    results in different components.
    """
    X_train_pca = X_resampled.copy()
    
    pca = PCA(n_components = 0.95)
    X_train_pca = pca.fit_transform(X_train_pca)

    X_train_pca = pd.DataFrame(X_train_pca, columns = [f'PC{i+1}' for i in range(X_train_pca.shape[1])])

    X_train_pca.to_csv('data/processed/smote/X_train_smote_pca.csv', index = False)
    safe_saver(pca, 'encoders/', 'PCA_transformer_smote')

    X_test_pca = X_test_scaled.copy()
    X_test_pca = pca.transform(X_test_pca)
    
    X_test_pca = pd.DataFrame(X_test_pca, columns = [f'PC{i+1}' for i in range(X_test_pca.shape[1])])
    X_test_pca.to_csv('data/processed/smote/X_test_smote_pca.csv', index = False)

if __name__ == '__main__':
    feature_engineering()
