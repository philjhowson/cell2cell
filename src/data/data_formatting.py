from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from data_functions import safe_saver
import category_encoders as ce
import pandas as pd
import warnings

warnings.simplefilter(action = 'ignore', category = FutureWarning)

def format_data():

    data = pd.read_csv('data/raw/cell2celltrain.csv')
    
    """
    replaces all 'Yes' with 1 and 'No' with 0 for later
    analyses.
    """
    data.replace({'Yes': 1, 'No' : 0}, inplace = True)

    """
    replaces credit ratings with simple digits, 1 being the
    highest and 7 being the lowest.
    """
    ratings = {'1-Highest' : 1, '2-High' : 2, '3-Good' : 3,
               '4-Medium' : 4, '5-Low' : 5, '6-VeryLow' : 6,
               '7-Lowest': 7}
    
    data['CreditRating'].replace(ratings, inplace = True)

    """
    converts 'Known' into 1 and 'Unknown' into 0. It is unclear
    from this data if 'Known' means yes, or if it just means
    they know the homeownership status.
    """
    data['Homeownership'].replace({'Known' : 1, 'Unknown' : 0},
                                   inplace = True)

    """
    because I'm later replacing NaN values with the median and using
    imputer techniques, I am splitting the data at this step to avoid
    data leakage and I drop the 'CustomerID' column because it provides
    no unique information. It just a unique value for each custom.
    """

    X = data.drop(columns = ['CustomerID', 'Churn'])
    y = data['Churn']

    X_train, X_, y_train, y_ = train_test_split(X, y, test_size = 0.5, stratify = y,
                                                random_state = 42)
    
    X_val, X_test, y_val, y_test = train_test_split(X_, y_, test_size = 0.5, stratify = y_,
                                                    random_state = 42)

    train_index = X_train.index
    val_index = X_val.index
    test_index = X_test.index

    safe_saver(train_index, 'data/processed/', 'train_index')
    safe_saver(val_index, 'data/processed/', 'validation_index')
    safe_saver(test_index, 'data/processed/', 'test_index')

    X_train, X_val, X_test = X_train.reset_index(drop = True), X_val.reset_index(drop = True), X_test.reset_index(drop = True)
    y_train, y_val, y_test = y_train.reset_index(drop = True), y_val.reset_index(drop = True), y_test.reset_index(drop = True)

    """
    here I replace the 'Unknown' data in HandsetPrice with the median
    value. I did explore the difference between mean and median values
    and determined median was more appropriate. This is because the mean
    was dragged up signficantly by high value of some phones. The same is
    done with MaritalStatus, only, the median is used because values only
    are 0 or 1, where a mean would be inappropraite. 
    """

    for dataset in [X_train, X_val, X_test]:

        dataset['HandsetPrice'] = dataset['HandsetPrice'].replace({'Unknown': None})
        dataset['HandsetPrice'] = pd.to_numeric(dataset['HandsetPrice'], errors='coerce')
        median_handset_price = dataset['HandsetPrice'].median()
        dataset['HandsetPrice'].fillna(median_handset_price, inplace = True)
        dataset['HandsetPrice'] = dataset['HandsetPrice'].astype(int)
        
        dataset['MaritalStatus'] = dataset['MaritalStatus'].replace({'Unknown': None})
        dataset['MaritalStatus'] = pd.to_numeric(dataset['MaritalStatus'], errors = 'coerce')
        median_marital_status = dataset['MaritalStatus'].median()
        dataset['MaritalStatus'].fillna(median_marital_status, inplace = True)
        dataset['MaritalStatus'] = dataset['MaritalStatus'].astype(int)

    """
    I am using OneHotEncoder for both 'PrizmCode' and 'Occupation' for
    exploratory data anlysis. If there are sufficient correlations,
    columns may be retained, otherwise they may be dropped.
    """

    encoder = OneHotEncoder(handle_unknown = 'ignore', sparse_output = False)
    encoder.fit(X_train[['PrizmCode', 'Occupation']])
        
    encoded = pd.DataFrame(encoder.transform(X_train[['PrizmCode', 'Occupation']]),
                           columns = encoder.get_feature_names_out(['PrizmCode', 'Occupation']))
    encoded = encoded.astype('int')
    X_train = pd.concat([X_train, encoded], axis = 1).drop(columns = ['PrizmCode', 'Occupation'])

    encoded = pd.DataFrame(encoder.transform(X_val[['PrizmCode', 'Occupation']]),
                           columns = encoder.get_feature_names_out(['PrizmCode', 'Occupation']))
    encoded = encoded.astype('int')
    X_val = pd.concat([X_val, encoded], axis = 1).drop(columns = ['PrizmCode', 'Occupation'])

    encoded = pd.DataFrame(encoder.transform(X_test[['PrizmCode', 'Occupation']]),
                           columns = encoder.get_feature_names_out(['PrizmCode', 'Occupation']))
    encoded = encoded.astype('int')
    X_test = pd.concat([X_test, encoded], axis = 1).drop(columns = ['PrizmCode', 'Occupation'])
    
    safe_saver(encoder, 'encoders/', 'OneHotEncoder')

    """
    Here I replace the majority of NaN values with the median. I
    tended towards the median for two reasons: columns like 'HandsetModels'
    are clearly categorical in nature, although they are numerically encoded.
    As such, having float values with decimal places are inappropriate.
    The values need to be whole numbers. In many other cases, I found
    that there was a tendency for extreme values (outliers) to pull up the
    mean, so median was again selected. 'ServiceArea' is a categorical
    variable and I removed it so I can impute the missing values and target
    encode later.
    """

    na_columns = data.columns[data.isna().any()].tolist()
    na_columns.remove('ServiceArea')

    for dataset in [X_train, X_val, X_test]:
        for column in na_columns:
            median = dataset[column].median()
            dataset[column].fillna(median, inplace = True)

    """
    impute the missing values for service area.
    """

    imputer = SimpleImputer(strategy = 'most_frequent')

    X_train['ServiceArea'] = imputer.fit_transform(X_train[['ServiceArea']]).flatten()
    X_val['ServiceArea'] = imputer.transform(X_val[['ServiceArea']]).flatten()
    X_test['ServiceArea'] = imputer.transform(X_test[['ServiceArea']]).flatten()

    safe_saver(imputer, 'encoders/', 'SimpleImputer')

    """
    now I target encode both 'ServiceArea', because it has over 700 unique values
    which is just too much for OneHotEncoding, and 'HandsetModels' is actually a
    categorical value that is encoded as a number and has 15 unique values.
    """

    encoder = ce.TargetEncoder(cols = ['ServiceArea', 'HandsetModels'])

    encoded = encoder.fit_transform(X_train[['ServiceArea', 'HandsetModels']],
                                    y_train)

    X_train = pd.concat([X_train.drop(columns = ['ServiceArea', 'HandsetModels']), encoded],
                        axis = 1)

    encoded = encoder.transform(X_val[['ServiceArea', 'HandsetModels']],
                                y_val)

    X_val = pd.concat([X_val.drop(columns = ['ServiceArea', 'HandsetModels']), encoded],
                        axis = 1)

    encoded = encoder.transform(X_test[['ServiceArea', 'HandsetModels']],
                                y_test)

    X_test = pd.concat([X_test.drop(columns = ['ServiceArea', 'HandsetModels']), encoded],
                        axis = 1)

    safe_saver(encoder, 'encoders/', 'TargetEncoder')

    X_train.to_csv('data/processed/X_train.csv', index = False)
    X_val.to_csv('data/processed/X_val.csv', index = False)
    X_test.to_csv('data/processed/X_test.csv', index = False)
    y_train.to_csv('data/processed/y_train.csv', index = False)
    y_val.to_csv('data/processed/y_val.csv', index = False)
    y_test.to_csv('data/processed/y_test.csv', index = False)

if __name__ == '__main__':
    format_data()
