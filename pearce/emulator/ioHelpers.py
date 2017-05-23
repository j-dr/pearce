#!/bin/bash
'''Contains several helper functions for emulator file IO. For most part, these are very general
and useful in multiple cases. More specific, one-use functions are generally left where they are.'''
# TODO FOR GOD SAKES DECIDE ON CAMELCASE V UNDERSCORES YOU MADMAN

from os import path
import cPickle as pickle
import numpy as np
from collections import namedtuple

__all__ = ['parameter', 'DEFAULT_PARAMS', 'GLOBAL_FILENAME','PARAMS_FILENAME', 'TRAINING_FILE_LOC_FILENAME',
           'params_file_reader','training_file_loc_reader', 'obs_file_reader','global_file_reader',
           'config_reader']

parameter = namedtuple('parameter', ['name', 'low', 'high'])

# global object that defines the names and ordering of the parameters, as well as their boundaries.
# TODO reading in bounds/params from config
# TODO add assembias params
DEFAULT_PARAMS = [parameter('logMmin', 11.7, 12.5),
          parameter('sigma_logM', 0.2, 0.7),
          parameter('logM0', 10, 13),
          parameter('logM1', 13.1, 14.3),
          parameter('alpha', 0.75, 1.25),
          parameter('f_c', 0.1, 0.5)]

# I initially had the global_filename be variable. Howerver, I couldn't find a reason one would change it!
GLOBAL_FILENAME = 'global_file.npy'
PARAMS_FILENAME = 'params.pkl'
TRAINING_FILE_LOC_FILENAME = 'training_file_loc.pkl'

def params_file_reader(dirname, fname = PARAMS_FILENAME):
    '''
    Load the parameter file from a given directory.
    :param dir:
            Directory to get the file from.
    :param fname:
            Optional. Custom filename. Default is PARAMS_FILENAME
    :return:
        ordered_params, a list of parameter tuples.
    '''
    with open(path.join(dirname, fname)) as f:
        ordered_params = pickle.load(f)

    return ordered_params


def training_file_loc_reader(dirname, fname=TRAINING_FILE_LOC_FILENAME):
    '''
    Load the training location file from a given directory.
    :param dir:
            Directory to get the file from.
    :param fname:
            Optional. Custom filename. Default is PARAMS_FILENAME
    :return:
        ordered_params, a list of parameter tuples.
    '''
    with open(path.join(dirname, fname)) as f:
        training_file_loc = pickle.load(f)

    return training_file_loc

def obs_file_reader(corr_file, cov_file=None):
    '''
    A helper function to parse the training data files.
    :param corr_file:
        Filename of the file with xi information
    :param cov_file:
        Optional. Filename containing the jackknifed covariance matrix.
    :return:
        HOD parameters, xi, cov (if cov_file is not None)
    '''

    assert path.exists(corr_file)
    obs = np.loadtxt(corr_file)  # not sure if this will work, might nead to transpose
    params = {}
    with open(corr_file) as f:
        for i, line in enumerate(f):
            if line[0] != '#' or i < 3:
                continue  # only looking at comments, and first two lines don't have params. Note: Does have cosmo!
            splitLine = line.strip('# \n').split(':')  # split into key val pair
            params[splitLine[0]] = float(splitLine[1])

    if cov_file is not None:
        assert path.exists(cov_file)
        cov = np.loadtxt(cov_file)

        return params, obs, cov
    return params, obs


def global_file_reader(dirname, fname=GLOBAL_FILENAME):
    '''
    Helper function, useful for reading the information in the global file.
    :param global_filename:
        Path+filename for the global file.
    :return:
        bins, cosmo_params, obs, method
    '''
    global_filename = path.join(dirname, fname)
    bins = np.loadtxt(global_filename)
    # cosmology parameters are stored in the global header
    cosmo_params = {}
    with open(global_filename) as f:
        for i, line in enumerate(f):
            if i == 0:
                splitLine = line.strip('# \n').split(':')  # split into key val pair
                method = splitLine[1].strip()
                continue
            elif i == 1:
                splitLine = line.strip('# \n').split(':')  # split into key val pair
                obs = splitLine[1].strip()
                continue
            elif line[0] != '#' or i < 3:
                continue  # only looking at comments, and first two lines don't have params. Note: Does have cosmo!
            splitLine = line.strip('# \n').split(':')  # split into key val pair
            try:
                cosmo_params[splitLine[0]] = float(splitLine[1])
            except ValueError:
                cosmo_params[splitLine[0]] = splitLine[1].strip()

    return bins, cosmo_params,obs, method


# Could use ConfigParser maybe
def config_reader(filename):
    '''
    General helper module. Turns a file of key:value pairs into a dictionary.
    :param filename:
        Config filename.
    :return:
        A dictionary of key-value pairs.
    '''
    config = {}
    with open(filename) as f:
        for line in f:
            line = line.strip()
            splitline = line.split(':')
            config[splitline[0].strip()] = splitline[1].strip()

    return config
