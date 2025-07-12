import pickle
import numpy as np
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

def safe_loader(file):
    """
    Simple function for opening .pkl files
    args:
        file: the path to the file you want to open, include
        .pkl ending.
    """
    if os.path.exists(file):
        with open(file, 'rb') as f:
            return pickle.load(f)
        
    print(f"{file} not found! Please check directory.")
    return None

