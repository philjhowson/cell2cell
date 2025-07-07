import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from data_functions import safe_saver

def data_exploration():

    features = pd.read_csv('data/processed/X_train.csv')
    target = pd.read_csv('data/processed/y_train.csv')
    
    data = pd.concat([features, target], axis = 1)
    columns = features.columns

    """
    To investigate the distribution of the features and the
    target 'Churn', I generate a series of boxplots.
    """

    os.makedirs('images/eda', exist_ok = True)

    for index, value in enumerate(range(10, 71, 10)):

        if index == 0:
            start = 0

        end = value

        if value > len(columns):
            end = len(columns) - 1

        temp = pd.concat([data.iloc[:, start : end], data['Churn']], axis = 1)

        column = temp.columns

        if end % 10 == 0:
            fig, ax = plt.subplots(2, 5, figsize = (20, 10))
        else:
            fig, ax = plt.subplots(1, 5, figsize = (20, 5))

        for index, axes in enumerate(ax.flatten()):
            
            sns.boxplot(data = temp, x = temp['Churn'],
                             y = temp[column[index]], ax = axes)
            axes.set_title(f"{column[index]} vs. Churn")

        plt.tight_layout()
        plt.savefig(f'images/eda/boxplots_columns_{start}-{value}.png')

        start = value

    """
    Because the data includes multiple variables, I want to investigate
    the degree to which there is multicollinarity. To do this, I am
    producting a heatmap with correlations. The correlation plot is quite
    large so I also want to produce a smaller map that has only those values
    that are high correlation with each other for possible disposal.
    """

    corr_mat = features.corr()
    corr_abs = abs(corr_mat)
    mask = corr_abs >= 0.8
    mask = mask & ~np.eye(mask.shape[0], dtype = bool)
    filtered_cols = mask.any(axis = 0)
    filtered_corr_mat = corr_mat.loc[filtered_cols, filtered_cols]

    plt.figure(figsize = (20, 20))
    sns.heatmap(corr_mat, annot = True, cmap = 'coolwarm',
                linewidths = 0.5, fmt = '.2f')

    plt.tight_layout()
    plt.savefig('images/eda/correlation_matrix.png')

    plt.figure(figsize = (10, 10))
    sns.heatmap(filtered_corr_mat, annot = True, cmap = 'coolwarm',
                linewidths = 0.5, fmt = '.2f')

    plt.tight_layout()
    plt.savefig('images/eda/filtered_correlation_matrix.png')

    """
    Now, I want to retrieve the pairs that are correlated for later
    processed.
    """
    
    rows = filtered_corr_mat.index
    cols = filtered_corr_mat.columns 

    mask = np.triu((filtered_corr_mat > 0.8), k=1)
    pairs = list(zip(*np.where(mask)))
    values = [(rows[i], cols[j]) for i, j in pairs]

    safe_saver(values, 'data/processed/', 'highly_correlated_feature_pairs')

    """
    Now I am plotting the distrubtions of 'Churn' and 'No Churn' as a method
    of identifying how imbalanced the datasets are. The data is relatively
    unbalanced, so during feature engineering, a method like SMOTE will
    more probably be used.
    """

    churn = data['Churn'].sum()
    no_churn = len(data) - churn

    sizes = [churn, no_churn]
    labels = ['Churn', 'No Churn']
    colors = ['purple', 'blue']

    plt.pie(sizes, labels = labels, autopct = '%1.1f%%',
            colors = colors, startangle = 90)
    plt.savefig('images/eda/churn_percentage.png')
    
if __name__ == '__main__':
    data_exploration()
