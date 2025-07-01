import pandas as pd
import numpy as np
from model_functions import safe_saver, safe_loader
import os


def get_predictions():

    pass

def print_model_results():

    path = 'metrics/scaled/'
    files = sorted([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])

    print(files)

if __name__ == '__main__':
    print_model_results()