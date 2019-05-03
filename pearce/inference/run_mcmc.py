#!/bin/bash
"""This file hold the function run_mcmc, which  takes a trained emulator and a set of truth data and runs
    and MCMC analysis with a predefined number of steps and walkers."""

from time import time
from multiprocessing import cpu_count, Pool
import warnings

from os import path
from ast import literal_eval

import numpy as np
import emcee as mc
from scipy.linalg import inv
import h5py

from pearce.emulator import OriginalRecipe, ExtraCrispy, SpicyBuffalo, NashvilleHot

# liklihood functions need to be defined here because the emulator will be made global

def lnprior(theta, param_names, *args):
    """
    Prior for an MCMC. Default is to assume flat prior for all parameters defined by the boundaries the
    emulator is built from. Retuns negative infinity if outside bounds or NaN
    :param theta:
        The parameters proposed by the sampler.
    :param param_names
        The names identifying the values in theta, needed to extract their boundaries
    :return:
        Either 0 or -np.inf, depending if the params are allowed or not.
    """
    for p, t in zip(param_names, theta):
        low, high = _emus[0].get_param_bounds(p)

        if np.isnan(t) or t < low or t > high:
            return -np.inf
    return 0

def lnlike(theta, param_names, fixed_params, r_bin_centers, y, combined_inv_cov):
    """
    :param theta:
        Proposed parameters.
    :param param_names:
        The names of the parameters in theta
    :param fixed_params:
        Dictionary of parameters necessary to predict y_bar but are not being sampled over.
    :param r_bin_centers:
        The centers of the r bins y is measured in, angular or radial.
    :param ys:
        The measured values of the observables to compare to the emulators. Must be an interable that contains
        predictions of each observable.
    :param combined_inv_cov:
        The inverse covariance matrices. Explicitly, the inverse of the sum of the mesurement covaraince matrix
        and the matrix from the emulator, both for each observable. Both are independent of emulator parameters,
         so can be precomputed. Must be an iterable with a matrixfor each observable.
    :return:
        The log liklihood of theta given the measurements and the emulator.
    """
    param_dict = dict(zip(param_names, theta))
    param_dict.update(fixed_params)

    emu_preds = []
    for _emu, in zip(_emus):
        y_bar = _emu.emulate_wrt_r(param_dict, r_bin_centers)[0]

        emu_preds.append(10**y_bar)
        #delta = y_bar - y
        #chi2 -= np.dot(delta, np.dot(combined_inv_cov, delta))

    emu_pred = np.hstack(emu_preds)

    delta = emu_pred - y
    return - np.dot(delta, np.dot(combined_inv_cov, delta))

def lnprob(theta, *args):
    """
    The total liklihood for an MCMC. Mostly a generic wrapper for the below functions.
    :param theta:
        Parameters for the proposal
    :param args:
        Arguments to pass into the liklihood
    :return:
        Log Liklihood of theta, a float.
    """
    lp = lnprior(theta, *args)
    if not np.isfinite(lp):
        return -np.inf

    return lp + lnlike(theta, *args)

def _run_tests(y, cov, r_bin_centers, param_names, fixed_params, ncores):
    """
    Run tests to ensure inputs are valid. Params are the same as in run_mcmc.

    :params:
        Same as in run_mcmc. See docstring for details.

    :return: ncores, which may be updated if it is an invalid value.
    """

    assert ncores == 'all' or ncores > 0
    if type(ncores) is not str:
        assert int(ncores) == ncores

    max_cores = cpu_count()
    if ncores == 'all':
        ncores = max_cores
    elif ncores > max_cores:
        warnings.warn('ncores invalid. Changing from %d to maximum %d.' % (ncores, max_cores))
        ncores = max_cores
        # else, we're good!

    #make sure all inputs are of consistent shape
    assert y.shape[0] == cov.shape[0] and cov.shape[1] == cov.shape[0]
    #print y.shape[0]/r_bin_centers.shape[0] ,len(_emus) , y.shape[0]/r_bin_centers.shape[0] 
    assert y.shape[0]/r_bin_centers.shape[0] == len(_emus) and y.shape[0]%r_bin_centers.shape[0] == 0
    # TODO informative error message when the array is jsut of the wrong shape?/

    # check we've defined all necessary params
    assert all([ _emu.emulator_ndim <= len(fixed_params) + len(param_names) + 1 for _emu in _emus])  # for r
    tmp = param_names[:]
    assert not any([key in param_names for key in fixed_params])  # param names can't include the
    tmp.extend(list(fixed_params.keys()))
    print(tmp)
    assert _emus[0].check_param_names(tmp, ignore=['r'])

    return ncores

# TOOD make functions that save/restore a state, not just the chains.
def _resume_from_previous(resume_from_previous, nwalkers, num_params):
    """
    Create initial guess by loading previous chain's last position.
    :param resume_from_previous:
        String giving the file name of the previous chain to use.
    :param nwalkers:
        Number of walkers to initiate. Must be the same as in resume_from_previous
    :param num_params:
        Number of params to initiate, must be the same as in resume_from_previous

    :return: pos0, the initial position for each walker in the chain.
    """
    # load a previous chain
    # TODO add error messages here
    old_chain = np.loadtxt(resume_from_previous)
    if len(old_chain.shape) == 2:
        c = old_chain.reshape((nwalkers, -1, num_params))
        pos0 = c[:, -1, :]
    else:  # 3
        pos0 = old_chain[:, -1, :]

    return pos0

def _random_initial_guess(param_names, nwalkers, num_params):
    """
    Create a random initial guess for the sampler. Creates a 3-sigma gaussian ball around the center of the prior space.
    :param param_names:
        The names of the parameters in the emulator
    :param nwalkers:
        Number of walkers to initiate. Must be the same as in resume_from_previous
    :param num_params:
        Number of params to initiate, must be the same as in resume_from_previous
    :return: pos0, the initial position of each walker for the chain.
    """

    pos0 = np.zeros((nwalkers, num_params))
    for idx, pname in enumerate(param_names):
        low, high = _emus[0].get_param_bounds(pname)
        pos0[:, idx] = np.random.randn(nwalkers) * (np.abs(high - low) / 6.0) + (low + high) / 2.0
        # TODO variable with of the initial guess

    return pos0

def run_mcmc(emus,  param_names, y, cov, r_bin_centers,fixed_params = {}, \
             resume_from_previous=None, nwalkers=1000, nsteps=100, nburn=20, ncores='all', return_lnprob = False):
    """
    Run an MCMC using emcee and the emu. Includes some sanity checks and does some precomputation.
    Also optimized to be more efficient than using emcee naively with the emulator.

    :param emus:
        A trained instance of the Emu object. If there are multiple observables, should be a list. Otherwiese,
        can be a single emu object
    :param param_names:
        Names of the parameters to constrain
    :param ys:
        data to constrain against. either one array of observables, or multiple where each new observable is a column.
    # TODO figure out whether it should be row or column and assign appropriately
    :param covs:
        measured covariance of y for each y. Should have the same iteration properties as ys
    :param r_bin_centers:
        The scale bins corresponding to all y in ys
    :param resume_from_previous:
        String listing filename of a previous chain to resume from. Default is None, which starts a new chain.
    :param fixed_params:
        Any values held fixed during the emulation, default is {}
    :param nwalkers:
        Number of walkers for the mcmc. default is 1000
    :param nsteps:
        Number of steps for the mcmc. Default is 1--
    :param nburn:
        Number of burn in steps, default is 20
    :param ncores:
        Number of cores. Default is 'all', which will use all cores available
    :param return_lnprob:
        Whether or not to return the lnprobs of the samples along with the samples. Default is False, which returns
        just the samples.
    :return:
        chain, collaposed to the shape ((nsteps-nburn)*nwalkers, len(param_names))
    """
    # make emu global so it can be accessed by the liklihood functions
    if type(emus) is not list:
        emus = [emus]
    _emus = emus
    global _emus

    ncores= _run_tests(y, cov, r_bin_centers,param_names, fixed_params, ncores)
    num_params = len(param_names)

    combined_inv_cov = inv(cov)

    sampler = mc.EnsembleSampler(nwalkers, num_params, lnprob,
                                 threads=ncores, args=(param_names, fixed_params, r_bin_centers, y, combined_inv_cov))

    if resume_from_previous is not None:
        try:
            assert nburn == 0
        except AssertionError:
            raise AssertionError("Cannot resume from previous chain with nburn != 0. Please change! ")
        # load a previous chain
        pos0 = _resume_from_previous(resume_from_previous, nwalkers, num_params)
    else:
        pos0 = _random_initial_guess(param_names, nwalkers, num_params)

    # TODO turn this into a generator
    sampler.run_mcmc(pos0, nsteps)

    chain = sampler.chain[:, nburn:, :].reshape((-1, num_params))

    if return_lnprob:
        lnprob_chain = sampler.lnprobability[:, nburn:].reshape((-1, )) # TODO think this will have the right shape
        return chain, lnprob_chain
    return chain

def run_mcmc_iterator(emus, param_names, y, cov, r_bin_centers,fixed_params={},
                      resume_from_previous=None, nwalkers=1000, nsteps=100, nburn=20, ncores='all', return_lnprob=False):
    """
    Run an MCMC using emcee and the emu. Includes some sanity checks and does some precomputation.
    Also optimized to be more efficient than using emcee naively with the emulator.

    This version, as opposed to run_mcmc, "yields" each step of the chain, to write to file or to print.

    :param emus:
        A trained instance of the Emu object. If there are multiple observables, should be a list. Otherwiese,
        can be a single emu object
    :param param_names:
        Names of the parameters to constrain
    :param y:
        data to constrain against. either one array of observables, of size (n_bins*n_obs)
    # TODO figure out whether it should be row or column and assign appropriately
    :param cov:
        measured covariance of y for each y. Should have the same shape as y, but square
    :param r_bin_centers:
        The scale bins corresponding to all y in ys
    :param resume_from_previous:
        String listing filename of a previous chain to resume from. Default is None, which starts a new chain.
    :param fixed_params:
        Any values held fixed during the emulation, default is {}
    :param nwalkers:
        Number of walkers for the mcmc. default is 1000
    :param nsteps:
        Number of steps for the mcmc. Default is 1--
    :param nburn:
        Number of burn in steps, default is 20
    :param ncores:
        Number of cores. Default is 'all', which will use all cores available
    :param return_lnprob:
        Whether to return the evaluation of lnprob on the samples along with the samples. Default is Fasle,
        which only returns samples.
    :yield:
        chain, collaposed to the shape ((nsteps-nburn)*nwalkers, len(param_names))
    """

    if type(emus) is not list:
        emus = [emus]

    _emus = emus
    global _emus

    ncores = _run_tests(y, cov, r_bin_centers, param_names, fixed_params, ncores)
    pool = Pool(processes=ncores)

    num_params = len(param_names)
    combined_inv_cov = inv(cov)

    sampler = mc.EnsembleSampler(nwalkers, num_params, lnprob, pool=pool,
                                 args=(param_names, fixed_params, r_bin_centers, y, combined_inv_cov))

    # TODO this is currently broken with the config option
    if resume_from_previous is not None:
        try:
            assert nburn == 0
        except AssertionError:
            raise AssertionError("Cannot resume from previous chain with nburn != 0. Please change! ")
        # load a previous chain
        pos0 = _resume_from_previous(resume_from_previous, nwalkers, num_params)
    else:
        pos0 = _random_initial_guess(param_names, nwalkers, num_params)

    for result in sampler.sample(pos0, iterations=nsteps, storechain=False):
        if return_lnprob:
            yield result[0], result[1]
        else:
            yield result[0]

def run_mcmc_config(config_fname):
    """
    Run an MCMC from a config file generated from intialize_mcmc.

    Essentially, a re-skin of the above. However, this is the preferred
    method for using this module, because it gurantees the state space
    of the samples is explicitly saved with them.
    :param config_fname:
        An hdf5 filename prepared a la initialize_mcmc. Will have the chain added as a dataset
    """

    assert path.isfile(config_fname), "Invalid config fname for chain"

    print(config_fname)
    f = h5py.File(config_fname, 'r+')
    emu_type_dict = {'OriginalRecipe':OriginalRecipe,
                     'ExtraCrispy': ExtraCrispy,
                     'SpicyBuffalo': SpicyBuffalo,
                     'NashvilleHot': NashvilleHot}
    fixed_params = f.attrs['fixed_params']
    fixed_params = {} if fixed_params is None else literal_eval(fixed_params)
    #metric = f.attrs['metric'] if 'metric' in f.attrs else {}
    emu_hps = f.attrs['emu_hps']
    emu_hps = {} if emu_hps is None else literal_eval(emu_hps)

    seed = f.attrs['seed']
    seed = int(time()) if seed is None else seed

    training_file = f.attrs['training_file']
    emu_type = f.attrs['emu_type']

    if type(training_file) is str:
        training_file = [training_file]

    if type(emu_type) is str:
        emu_type = [emu_type]

    assert len(emu_type) == len(training_file)

    emus = []

    np.random.seed(seed)
    for et, tf in zip(emu_type, training_file): # TODO iterate over the others?
        emu = emu_type_dict[et](tf,
                                 fixed_params = fixed_params,
                                 **emu_hps)
        emus.append(emu)
        # TODO write hps to the file too

    assert 'obs' in list(f.attrs.keys()), "No obs info in config file."
    obs_cfg = literal_eval(f.attrs['obs'])
    rbins = np.array(obs_cfg['rbins'])
    rpoints = (rbins[1:]+rbins[:-1])/2.0
    orig_n_bins = len(rpoints)
    rpoints = rpoints[-emu.n_bins:]

    # un-stack these
    # TODO once i have the covariance terms these will need to be propertly combined

    y = np.hstack([f['data'][(i+1)*orig_n_bins-emu.n_bins:(i+1)*orig_n_bins] \
                   for i, e in enumerate(emus)])
    # not sure if they should be flipped
    _cov = np.hstack([f['cov'][(i+1)*orig_n_bins-emu.n_bins:(i+1)*orig_n_bins, :] for i, e in enumerate(emus)])
    cov = np.vstack([ _cov[:, (i+1)*orig_n_bins-emu.n_bins:(i+1)*orig_n_bins] for i, e in enumerate(emus)])

    #covs = [f['cov'][-e.n_bins:, :][:, -e.n_bins:] for i,e in enumerate(emus)]

    nwalkers, nsteps = f.attrs['nwalkers'], f.attrs['nsteps']

    nburn, seed, fixed_params = f.attrs['nburn'], f.attrs['seed'], f.attrs['chain_fixed_params']

    nburn = 0 if nburn is None else nburn
    seed = int(time()) if seed is None else seed
    fixed_params = {} if fixed_params is None else fixed_params

    if type(fixed_params) is str:
        try:
            fixed_params = literal_eval(fixed_params)
        except ValueError: #malformed string, can't be eval'd
            pass

    if fixed_params and type(fixed_params) is str:
        assert fixed_params in {'HOD', 'cosmo'}, "Invalied fixed parameter value."
        assert 'sim' in list(f.attrs.keys()), "No sim information in config file."
        sim_cfg = literal_eval(f.attrs['sim'])
        if fixed_params == 'HOD':
            fixed_params = sim_cfg['hod_params']
        else:
            assert 'cosmo_params' in sim_cfg, "Fixed cosmology requested, but the values of the cosmological\"" \
                                                     "params were not specified. Please add them to the sim config."
            fixed_params = sim_cfg['cosmo_params']

    elif "HOD" in fixed_params:
        assert 'sim' in list(f.attrs.keys()), "No sim information in config file."
        sim_cfg = literal_eval(f.attrs['sim'])
        del fixed_params['HOD']
        fixed_params.update(sim_cfg['hod_params'])
        if 'logMmin' in fixed_params:
            del fixed_params['logMmin']
    elif "cosmo" in fixed_params:
        assert 'sim' in list(f.attrs.keys()), "No sim information in config file."
        sim_cfg = literal_eval(f.attrs['sim'])
        assert 'cosmo_params' in sim_cfg, "Fixed cosmology requested, but the values of the cosmological\"" \
                                                     "params were not specified. Please add them to the sim config."

        del fixed_params['cosmo']
        fixed_params.update(sim_cfg['cosmo_params'])

    #TODO resume from previous, will need to access the written chain
    param_names = [pname for pname in emu.get_param_names() if pname not in fixed_params]
    f.attrs['param_names'] = param_names

    #chain = np.zeros((nwalkers*nsteps, len(param_names)), dtype={'names':param_names,
    #                                                             'formats':['f8' for _ in param_names]})
    # TODO warning? Overwrite key?
    if 'chain' in list(f.keys()):
        del f['chain']#[:,:] = chain
        # TODO anyway to make sure all shpaes are right?
        #chain_dset = f['chain']

    f.create_dataset('chain', (nwalkers*nsteps, len(param_names)), chunks = True, compression = 'gzip')

    #lnprob = np.zeros((nwalkers*nsteps,))
    if 'lnprob' in list(f.keys()):
        del f['lnprob']#[:] = lnprob 
        # TODO anyway to make sure all shpaes are right?
        #lnprob_dset = f['lnprob']
    f.create_dataset('lnprob', (nwalkers*nsteps, ) , chunks = True, compression = 'gzip')
    f.close()
    np.random.seed(seed)
    print(nwalkers)
    print(nsteps)
    for step, pos in enumerate(run_mcmc_iterator(emus, param_names, y, cov, rpoints,\
                                                 fixed_params=fixed_params, nwalkers=nwalkers,\
                                                 nsteps=nsteps, nburn=nburn, return_lnprob=True, ncores = 16)):

        f = h5py.File(config_fname, 'r+')
        f['chain'][step*nwalkers:(step+1)*nwalkers] = pos[0]
        f['lnprob'][step*nwalkers:(step+1)*nwalkers] = pos[1]
        f.close()


if __name__ == "__main__":
    from sys import argv
    fname = argv[1] 
    suffix = fname.split('.')[-1]

    if suffix == 'hdf5' or suffix == 'h5':
        pass
    elif suffix == 'yaml': # parse yaml file
        import yaml
        with open(fname, 'r') as ymlfile:
                cfg = yaml.load(ymlfile)
                filename = cfg['fname']
        fname = filename

    else:
        raise IOError("Invalid input filetype")

    run_mcmc_config(fname)

