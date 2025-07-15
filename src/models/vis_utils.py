import os
import json

def scores_dictionary(directory, filenames):
    """
    Retrieves all .json files from a directory and names each key based on a list of filenames given.
    Args:
        directory: path to the .json files.
        filenames: list of filenames to use.
    """

    files = sorted([f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and 'grid' not in f.split('_')])

    results = {}
    
    for index, file in enumerate(files):
        with open(os.path.join(directory, file), 'r') as f:
            results[filenames[index]] = json.load(f)

    return results

def retrieve_scores(dictionary):
    """
    Takes an input dictionary of values and gives back scores in chunks of
    three.
    Args:
        dictionary: the dictionary to be chunked.
    """

    values = [dictionary[key]['test_f1'] for key in dictionary]
    f1 = [values[i:i+3] for i in range(0, len(values), 3)]

    values = [dictionary[key]['test_roc'] for key in dictionary]
    roc = [values[i:i+3] for i in range(0, len(values), 3)]

    return f1, roc

def label_bars(bars, axes, offset):
    for bar in bars:
        height = bar.get_height()
        x = bar.get_x() + bar.get_width() / 2
        axes.text(x, height - offset, f'{height:.3f}', ha = 'center', va = 'bottom', color = 'white',
                  fontweight = 'bold')