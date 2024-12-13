import pandas as pd
import numpy as np
import pickle

def load_data(file_path):
    data=pd.read_csv(file_path)
    return data


def save_file(name, obj):
    """
    Function to save an object as pickle file
    """
    with open(name, 'wb') as f:
        pickle.dump(obj, f)


def load_file(name):
    """
    Function to load a pickle object
    """
    return pickle.load(open(name, "rb"))
