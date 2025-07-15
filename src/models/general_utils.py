import pickle
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

def safe_loader(path):
    """
    A simple loading function for .pkl files.

    args:
        path: the path, including the file name and extension
        of the desired file.
    """

    with open(path, 'rb') as f:
        return pickle.load(f)