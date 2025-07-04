import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from model_functions import safe_loader, safe_saver, get_models_and_features, model_predictions
import argparse
import os

def print_model_results():

    path = 'metrics/scaled/'
    files = sorted([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))], key = str.lower)
    shorthand = ['base_log', 'base_rf', 'base_xgb', 'pca_log', 'pca_rf', 'pca_xgb',
                 'pso_log', 'pso_rf', 'pso_xgb', 'rfecv_log', 'rfecv_rf', 'rfecv_xgb']
    scaled_results = {}

    for index, file in enumerate(files):
        scaled_results[shorthand[index]] = safe_loader(f"{path}{file}")

    scaled_test_f1 = [float(scaled_results[item]['test_f1']) for item in scaled_results.keys()]
    scaled_test_roc = [float(scaled_results[item]['test_roc']) for item in scaled_results.keys()]

    path = 'metrics/smote/'
    files = sorted([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))], key = str.lower)
    smote_results = {}

    for index, file, in enumerate(files):
        smote_results[shorthand[index]] = safe_loader(f"{path}{file}")

    smote_test_f1 = [float(smote_results[item]['test_f1']) for item in smote_results.keys()]
    smote_test_roc = [float(smote_results[item]['test_roc']) for item in smote_results.keys()]

    titles = ['Baseline Test F1', 'Baseline Test ROC-AUC', 'PCA Test F1', 'PCA Test ROC-AUC',
              'PSO Test F1', 'PSO Test ROC-AUC', 'RFECV Test F1', 'RFECV Test ROC-AUC']
    xlabs = ['Logistic Regression', 'Random Forests', 'XGBoost']

    fig, ax = plt.subplots(2, 4, figsize = (30, 10))
    width = 0.4
    x = np.arange(3)
    indices = 0

    for index, axes in enumerate(ax.flat):

        subset = indices * 3

        if index % 2 == 0:
            subset_scaled = scaled_test_f1[subset : subset + 3]
            subset_smote = smote_test_f1[subset : subset + 3]
        else:
            subset_scaled = scaled_test_roc[subset : subset + 3]
            subset_smote = smote_test_roc[subset : subset + 3]
            indices += 1

        scaled_bars = axes.bar(x - width/2, subset_scaled, width, color = 'blue', label = 'Scaled')
        smote_bars = axes.bar(x + width/2, subset_smote, width, color = 'purple', label = 'Smote')

        for bar in scaled_bars:
            height = bar.get_height()
            x_pos = bar.get_x() + bar.get_width() / 2
            axes.text(x_pos, height - 0.03, f'{round(height, 3)}', ha = 'center',
                      va = 'top', color = 'white', fontweight = 'bold', fontsize = 12)
        
        for bar in smote_bars:
            height = bar.get_height()
            x_pos = bar.get_x() + bar.get_width() / 2
            axes.text(x_pos, height - 0.03, f'{round(height, 3)}', ha = 'center',
                      va = 'top', color = 'white', fontweight = 'bold', fontsize = 12)       

        axes.set_title(titles[index])
        axes.title.set_fontsize(14)
        axes.set_ylim(0, 1)
        axes.set_xticks(x)
        axes.set_xticklabels(xlabs)#, rotation = 45, ha = 'right')
        axes.tick_params(axis = 'both', labelsize = 14)
        axes.legend()
    
    plt.tight_layout()
    plt.savefig('images/training_and_test_scores.png')

def get_predictions():

    path_to_models = 'models/scaled'
    scaled_model_names = sorted([f for f in os.listdir(path_to_models) if f.endswith('model.pkl')], key = str.lower)
    path_to_features = 'data/processed/scaled'

    scaled_dictionary = get_models_and_features(scaled_model_names, path_to_models, path_to_features)

    X_train_scaled = pd.read_csv(f"{path_to_features}/X_train_scaled.csv")
    X_test_scaled = pd.read_csv(f"{path_to_features}/X_test_scaled.csv")
    X_train_scaled_pca = pd.read_csv(f"{path_to_features}/X_train_scaled_pca.csv")
    X_test_scaled_pca = pd.read_csv(f"{path_to_features}/X_test_scaled_pca.csv")  

    print('Getting model predictions...')

    scaled_train_preds, scaled_train_probs, scaled_test_preds, scaled_test_probs = model_predictions(X_train_scaled, X_test_scaled,
                                                                                                     X_train_scaled_pca, X_test_scaled_pca,
                                                                                                     scaled_dictionary)

    path_to_models = 'models/smote'
    smote_model_names = sorted([f for f in os.listdir(path_to_models) if f.endswith('model.pkl')])
    path_to_features = 'data/processed/smote'

    smote_dictionary = get_models_and_features(smote_model_names, path_to_models, path_to_features)

    X_train_smote_pca = pd.read_csv(f"{path_to_features}/X_train_scaled_pca.csv")
    X_test_smote_pca = pd.read_csv(f"{path_to_features}/X_test_smote_pca.csv")

    smote_train_preds, smote_train_probs, smote_test_preds, smote_test_probs = model_predictions(X_train_scaled, X_test_scaled,
                                                                                                 X_train_smote_pca, X_test_smote_pca,
                                                                                                 smote_dictionary, 'smote')
    
    train_preds = pd.concat([scaled_train_preds, smote_train_preds], axis = 1)
    train_probs = pd.concat([scaled_train_probs, smote_train_probs], axis = 1)
    test_preds = pd.concat([scaled_test_preds, smote_test_preds], axis = 1)
    test_probs = pd.concat([scaled_test_probs, smote_test_probs], axis = 1)

    train_preds.to_csv('data/processed/training_model_preds.csv', index = False)
    train_probs.to_csv('data/processed/training_model_probs.csv', index = False)
    test_preds.to_csv('data/processed/test_model_preds.csv', index = False)
    test_probs.to_csv('data/processed/test_model_probs.csv', index = False)

    print('All files saved successfully!') 

def get_correlations():

    train_preds = pd.read_csv('data/processed/training_model_preds.csv')
    train_probs = pd.read_csv('data/processed/training_model_probs.csv')

    best_model = train_preds['RFECV_xgb']
    train_preds.drop(columns = ['RFECV_xgb'], inplace = True)
    pred_correlations = {}

    for column in train_preds.columns:

        temp = train_preds[column].copy()
        corr = best_model.corr(temp)

        pred_correlations[column] = corr

    best_model = train_probs['RFECV_xgb']
    train_probs.drop(columns = ['RFECV_xgb'], inplace = True)
    prob_correlations = {}

    for column in train_probs.columns:

        temp = train_probs[column].copy()
        corr = best_model.corr(temp)

        prob_correlations[column] = corr

    sorted_corr = sorted(pred_correlations.items(), key = lambda x: x[1])
    lowest_two = sorted_corr[:2]

    pred_keys = [k for k, v in lowest_two] 

    sorted_corr = sorted(prob_correlations.items(), key = lambda x: x[1])
    lowest_two = sorted_corr[:2]

    prob_keys = [k for k, v in lowest_two]

    safe_saver(pred_keys, 'data/processed/', 'lowest_correlations_preds')
    safe_saver(prob_keys, 'data/processed/', 'lowest_correlations_probs')

    print(f"Lowest Prediction Correlations: {pred_keys}")
    print(f"Lowest Probability Correlations: {prob_keys}")

def run_function(function):
    if function == 'vis':
        print_model_results()
    if function == 'preds':
        get_predictions()
    if function == 'corr':
        get_correlations()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'just a simple argument parser')
    parser.add_argument('--function', default = None, help = 'vis, preds, or corr')

    arg = parser.parse_args()

    run_function(arg.function)
