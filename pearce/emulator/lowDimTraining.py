#!/bin/bash
'''Training in high dimensions is difficult. This module incorporates functions that learn
hyperparamters from lower dimensions of data.'''
from os import path
from glob import glob
from itertools import combinations, product
from collections import defaultdict
import numpy as np
from .emu import OriginalRecipe, ExtraCrispy
from .ioHelpers import obs_file_reader

#TODO does this need to be in a separate file here? Can I attach it to emu?
def low_dim_train(training_dir, ordered_params, independent_variable, n_params = 3, emu_type = 'OriginalRecipe'):
    '''
    Train the hyperparameters in lower dimensions and average them together.
    :param training_dir:
        Directory with the training data, passed into the emu object
    :param ordered_params:
        Ordered parameter list, passed into the emu object
    :param independent_variable:
        The independent variable to emulate and learn params for.
    :param n_params:
        Number of parameters to hold train the emulator with. Default is 3.
    :param emu_type:
        Type of emulator to use. Default is OriginalRecipe
    :return:
    '''

    emu_obj = OriginalRecipe if emu_type == 'OriginalRecipe' else ExtraCrispy

    #TODO change 'r' to something more general
    if ordered_params[-1].name == 'r':
        #we don' do combinations in 'r'
        varied_params = ordered_params[:-1]
    else:
        varied_params = ordered_params[:]
    hyper_params = {p.name: [] for p in ordered_params}
    hyper_params['amp'] = [] #special case
    #unique values in the training data
    unique_values = get_unique_values(training_dir)#{p.name:list(set(emu.x[:, idx])) for idx, p in enumerate(varied_params)}
    #all unique combinations to train
    param_combinations = combinations(varied_params, n_params)

    emu = None
    n_max = 20#

    for pc in param_combinations:
        #for each combination, also train for each combination of unique values
        #we're being very thorough
        #these are unique values of the params were holding fixed, those not in pc
        fixed_pc = [p for p in varied_params if p not in pc]
        unique_values_pc = product(*[list(unique_values[p.name]) for p in fixed_pc])
        n_uv = np.prod([len(list(unique_values[p.name])) for p in fixed_pc])

        if n_uv > n_max:
            #select a subsample
            unique_values_idx = set(np.random.choice(n_uv,size = n_max, replace=False))
        else:
            unique_values_idx = set(xrange(n_uv))

        #now, rebuild and train the emulator.
        for idx, uv in enumerate(unique_values_pc):
            if idx not in unique_values_idx:
                continue
            print uv
            #hold these parameters fixed
            fixed_params = {p.name:uv[idx] for idx, p in enumerate(fixed_pc)}
            if emu is None:
                emu = emu_obj(training_dir, ordered_params, independent_variable, fixed_params)
            else:
                emu.get_training_data(training_dir, independent_variable, fixed_params)
                emu.build_emulator(independent_variable, fixed_params)
            success = emu.train()

            if not success:
                continue

            for p, m in zip(pc, emu.metric[1:]):
                hyper_params[p.name].append(m)
            hyper_params['amp'].append(emu.metric[0])
            if ordered_params[-1].name == 'r': #has 'r'
                hyper_params['r'].append(emu.metric[-1])
        print

    for key in hyper_params:
        hyper_params[key] = np.array(hyper_params[key])

    for key, val in hyper_params.iteritems():
        print key,val.shape,np.median(val), val.mean(), val.std()
    print

    return {key:val.mean() for key, val in hyper_params.iteritems()}

def get_unique_values(training_dir):
    '''
    Returns a dictionary of sets containing the unique training values of each
    parameter in the training_dir. I tried to wrap this feature into the emu
    object. However, that requires loading all data at once, something that isn't
    always possible and led to the creation of this module in the first place.
    :param training_dir:
        Directory holding the training data for the emulator.
    :return:
        unique_values, a dictionary of sets containins unique values of
        the parameters in the training file.
    '''
    unique_values = defaultdict(set)
    obs_files = glob(path.join(training_dir, 'obs*.npy'))
    for obs_file in obs_files:
        params, _ = obs_file_reader(obs_file)
        for key, val in params.iteritems():
            unique_values[key].add(val)

    return unique_values
