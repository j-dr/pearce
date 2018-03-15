#!/bin/bash
'''Training helper. Is called from training data as main, to send to cluster.'''

from os import path
import sys
import warnings
import numpy as np
#from trainingData import PARAMS, GLOBAL_FILENAME
#from ioHelpers import global_file_reader
#from ..mocks.kittens import cat_dict
#Not happy about this, look into a change.


sys.path.append('..')
from pearce.emulator import global_file_reader, params_file_reader
from pearce.mocks import cat_dict


# TODO to ioHelpers?
def load_training_params(param_file):
    '''
    Load the parameters to calculate. Will load bins, parameters, cosmology info, output directory, and a unique id!
    :param param_file:
        The file where the parameters are stored. The output directory is the same directory as the paramfile, and
        the radial bins are stored in the same directory as well.
    :return: hod parameters, bins, obs, cosmo paraemters, job id
    '''
    hod_params = np.loadtxt(param_file)
    if len(hod_params.shape) == 1:
        hod_params = np.array([hod_params]) #needs to have 2 dims
    dirname = path.dirname(param_file)
    bins, cosmo_params, obs,log_obs, _ = global_file_reader(dirname)
    ordered_params = params_file_reader(dirname)
    job_id = int(param_file.split('.')[-2][-3:]) #last 3 digits of paramfile is a unique id.

    return hod_params, bins,obs, log_obs, cosmo_params,ordered_params, dirname, job_id

def calc_training_points(hod_params, bins,obs, cosmo_params,ordered_params, dirname, job_id,log_obs=True):
    '''
    given an array of hod parameters (and a few other things) populate a catalog and calculate an observable at those points.
    :param hod_params:
        An array of hod parameter points. Each row represents a "point" in HOD space. This function iterates over those
        points anc calculates the observable at each of them.
    :param bins:
        Radial (or angular) bins for the observable calculation.
    :param obs:
        The name of the observable to calculate. String. 
    :param log_obs: 
        Bool. Take the log of the observable. Should be true for xi and wp
    :param cosmo_params:
        cosmology information, used for loading a catalog. Required is simname, scale_factor, and anything required to load
        a specific catalog (Lbox, npart, etc.)
    :param dirname:
        The directory to write the outputs of this function.
    :param job_id:
        A unique identifier of this call, so saves from this function will not overwrite parallel calls to this function.
    :return: None
    '''
    cat = cat_dict[cosmo_params['simname']](**cosmo_params)#construct the specified catalog!
    #Could add a **kwargs to load to shorten this.
    #That makes things a little messier though IMO
    # TODO tol?
    hod_kwargs = cosmo_params['hod_kwargs'] if 'hod_kwargs' in cosmo_params else dict()
    # TODO this is a BAD idea delete tomorrow
    for key, val in hod_kwargs.iteritems():
        if type(val) is str and val.split('.')[-1] == 'npy': #may god have mercy on my soul
            #with open(val, 'r') as f:
            #   hod_kwargs[key] = pickle.load(f)
            hod_kwargs[key] = np.loadtxt(val)

    if 'HOD' in cosmo_params:
        cat.load(cosmo_params['scale_factor'], cosmo_params['HOD'], hod_kwargs=hod_kwargs)
    else:
        cat.load(cosmo_params['scale_factor'], hod_kwargs=hod_kwargs)

    #Number of repopulations to do
    n_repops=1
    if 'n_repops' in cosmo_params:
        n_repops = int(cosmo_params['n_repops'])
        assert n_repops > 1

    #check if we should use a non-default observable.
    calc_observable = cat.calc_xi

    if obs!='xi': #if the observable we're after is different than the default.
        try:
            #get the function to calculate the observable
            calc_observable = getattr(cat, 'calc_%s'%obs)
        except AttributeError:
            #TODO just fail, warning has no purpose.
            warnings.warn('WARNING: Observable %s invalid; using default xi' % (obs))
            calc_observable = cat.calc_xi
            obs = 'xi'
    # TODO should be a better way to do this
    # take the log of some observables, but not all
    transform_func = lambda x : x
    if log_obs:#obs in {'xi', 'wp'}: 
        transform_func = np.log10

    #check to see if there are kwargs for calc_observable
    args = calc_observable.args #get function args
    kwargs = {}
    #cosmo_params may have args for our function.
    if any(arg in cosmo_params for arg in args):
        kwargs.update({arg: cosmo_params[arg] for arg in args if arg in cosmo_params})
    if n_repops > 1: #don't do jackknife in this case
        if 'do_jackknife' in cosmo_params and cosmo_params['do_jackknife']:
            warnings.warn('WARNING: Cannot perform jackknife with n_repops>1. Turning off jackknife.')
            kwargs['do_jackknife']=False
    if kwargs: #if there are kwargs to pass in.
        _calc_observable = calc_observable #might not have to do this, but play it safe.
        calc_observable = lambda bins: _calc_observable(bins, **kwargs)#make it so kwargs are default.


    for id, hod in enumerate(hod_params):
        #construct a dictionary for the parameters
        # Could store the param ordering in the file so it doesn't need to be imported.
        hod_dict = {p: val for p, val in zip(ordered_params, hod)}
        #Populating the same cat over and over is faster than making a new one over and over!
        if n_repops ==1:
            cat.populate(hod_dict)
            obs_val, obs_cov = calc_observable(bins)
        else: #do several repopulations
            obs_repops = np.zeros((n_repops, bins.shape[0]-1))

            for repop in xrange(n_repops):
                cat.populate(hod_dict)
                obs_i = calc_observable(bins)
                obs_repops[repop, :] = transform_func(obs_i) #TODO i suppose this may want to be a flag, or rolled into the func?

            #obs_val = np.mean(obs_repops, axis=0)
            #obs_cov = np.cov(obs_repops, rowvar=False)/np.sqrt(n_repops)#error on mean # TODO make sure is right?
            obs_val = np.mean(obs_repops, axis=0)
            obs_cov = np.cov(obs_repops, rowvar=False)


        # Consider storing them all and writing them all at once. May be faster.
        # Writing the hod params as a header. Could possibly recover them from the same file I used to read these in.
        # However, I think storing them in teh same place is easier.
        header_start = ['Cosmology: %s' % cosmo_params['simname'],'Observable: %s' % obs, 'Params for HOD:']
        header_start.extend('%s:%.3f' % (key, val) for key, val in hod_dict.iteritems())
        header = '\n'.join(header_start)
        #TODO change file naming system;
        np.savetxt(path.join(dirname, 'obs_job%03d_HOD%03d.npy'%(job_id, id) ), obs_val, header=header)
        np.savetxt(path.join(dirname, 'cov_job%03d_HOD%03d.npy'%(job_id, id) ), obs_cov, header=header)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Helper function called as main from "trainingData.py." Reads in \
                                                collections of HOD params and calculates their correlation functions.')
    parser.add_argument('param_file', type = str, help='File where the vector of HOD params are stored.')

    args = vars(parser.parse_args())

    hod_params, bins,obs,log_obs, cosmo_params, ordered_params, dirname, job_id = load_training_params(args['param_file'])
    calc_training_points(hod_params, bins,obs, cosmo_params,ordered_params, dirname, job_id,log_obs=log_obs)
