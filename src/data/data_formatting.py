from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import warnings

warnings.simplefilter(action = 'ignore', category = FutureWarning)

def exploration():

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
    here I replace the 'Unknown' data in HandsetPrice with the median
    value. I did explore the difference between mean and median values
    and determined median was more appropriate. This is because the mean
    was dragged up signficantly by high value of some phones. The same is
    done with MaritalStatus, only, the median is used because values only
    are 0 or 1, where a mean would be inappropraite. 
    """

    handy = data['HandsetPrice'][data['HandsetPrice'] != 'Unknown'].copy()
    handy = handy.astype('int')
    median = handy.median()

    data['HandsetPrice'].replace({'Unknown' : median}, inplace = True)
    data['HandsetPrice'] = data['HandsetPrice'].astype('int')

    marriage = data['MaritalStatus'][data['MaritalStatus'] != 'Unknown'].copy()
    marriage = marriage.astype('int')
    median = marriage.median()

    data['MaritalStatus'].replace({'Unknown' : median}, inplace = True)
    data['MaritalStatus'] = data['MaritalStatus'].astype('int')

    """
    I am using OneHotEncoder for both 'PrizmCode' and 'Occupation' for
    exploratory data anlysis. If there are sufficient correlations,
    columns may be retained, otherwise they may be dropped.
    """

    encoder = OneHotEncoder(handle_unknown = 'ignore', sparse_output = False)
    encoded = encoder.fit_transform(data[['PrizmCode', 'Occupation']])
    cols = encoder.get_feature_names_out(['PrizmCode', 'Occupation'])
    encoded = pd.DataFrame(encoded, columns = cols).astype('int')

    data = pd.concat([data, encoded], axis = 1)
    data.drop(columns = ['PrizmCode', 'Occupation'], inplace = True)

    data.info()

    """

    """
    print('ServiceArea', len(set(data['ServiceArea'])))
    
if __name__ == '__main__':
    exploration()
