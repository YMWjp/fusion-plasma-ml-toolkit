import os
import numpy as np
import yaml

CONFIG_DIR = os.path.dirname(__file__)

def get_shot_numbers():
    path = os.path.join(CONFIG_DIR, 'shot_numbers.csv')
    with open(path, 'r') as f:
        return np.genfromtxt(f, delimiter=',', skip_header=1, usecols=0, dtype=int)

def get_parameters():
    path = os.path.join(CONFIG_DIR, 'parameters.yaml')
    with open(path, 'r') as f:
        return yaml.safe_load(f)['parameters']

def get_basic_info_for_header():
    path = os.path.join(CONFIG_DIR, 'parameters.yaml')
    with open(path, 'r') as f:
        return yaml.safe_load(f)['basic_info_for_header']

def load_config():
    path = os.path.join(CONFIG_DIR, 'config.yaml')
    with open(path, 'r') as f:
        return yaml.safe_load(f)
    
