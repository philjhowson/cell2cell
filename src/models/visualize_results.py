import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn
from vis_utils import scores_dictionary, retrieve_scores, label_bars

def plot_results():

    directory = "metrics/scaled/"
    filenames = ['baseline_log', 'baseline_rf', 'baseline_xgb', 'pca_log', 'pca_rf', 'pca_xgb',
                  'pso_log', 'pso_rf', 'pso_xgb', 'RFECV_log', 'RFECV_rf', 'RFECV_xgb']
    scaled_results = scores_dictionary(directory, filenames)
    scaled_f1, scaled_roc = retrieve_scores(scaled_results)

    directory = "metrics/smote/"
    smote_results = scores_dictionary(directory, filenames)
    smote_f1, smote_roc = retrieve_scores(smote_results)

    titles = ['Baseline F1-Scores', 'Baseline ROC-AUC Scores', 'PCA F1-Scores', 'PCA ROC-AUC Scores',
              'PSO F1-Scores', 'PSO ROC-AUC Scores', 'RFECV F1-Scores', 'RFECV ROC-AUC Scores']
    axes_ticks = ['Logistic Regression', 'Random Forests', 'XGBoost']

    fig, ax = plt.subplots(2, 4, figsize = (20, 5))

    i = 0
    width = 0.35

    for index, axes in enumerate(ax.flat):
        if index % 2 == 0:
            scaled_values = scaled_f1[i]
            smote_values = smote_f1[i]
        else:
            scaled_values = scaled_roc[i]
            smote_values = smote_roc[i]
            i += 1

        x_values = [x - width for x in range(3)]
        bars = axes.bar(x_values, scaled_values, width = width, align = 'edge', color = 'blue', label = 'Scaled')
        label_bars(bars, axes, 0.1)

        bars = axes.bar(range(3), smote_values, width = width, align = 'edge', color = 'purple', label = 'Smote')
        label_bars(bars, axes, 0.1)

        axes.set_xticks(range(3))
        axes.set_xticklabels(axes_ticks)
        axes.set_title(titles[index])
        axes.set_ylim(0, 1)
        axes.legend()

    plt.tight_layout()
    plt.savefig('images/model_results.png')

if __name__ == '__main__':
    plot_results()