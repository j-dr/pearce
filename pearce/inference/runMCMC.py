#!/bin/bash
"""This file hold the function run_mcmc, which  takes a trained emulator and a set of truth data and runs
    and MCMC analysis with a predefined number of steps and walkers."""

from multiprocessing import cpu_count
import warnings
from itertools import izip

import numpy as np
import emcee as mc
from scipy.linalg import inv


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
    for p, t in izip(param_names, theta):
        low, high = _emu.get_param_bounds(p)

        if np.isnan(t) or t < low or t > high:
            #print p, t, low, high
            return -np.inf
    return 0

def lnlike(theta, param_names, fixed_params, r_bin_centers, ys, combined_inv_covs):
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
    param_dict = dict(izip(param_names, theta))
    param_dict.update(fixed_params)

    chi2 = 0
    for _emu, y , combined_inv_cov in izip(_emus, ys, combined_inv_covs):
        y_bar = _emu.emulate_wrt_r(param_dict, r_bin_centers)[0]

<<<<<<< HEAD
    #if np.random.rand() > 0.999: #print a subsample of the time
    #print y_bar
    #print y
    #print '*'*30
=======
        delta = y_bar - y
>>>>>>> a7cf37e46f9c4880b1d6da74775f097ee441fd39

        chi2 -= np.dot(delta, np.dot(combined_inv_cov, delta))

    return chi2

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

def _run_tests(ys, covs, r_bin_centers, param_names, fixed_params, ncores):
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
    for y, cov in izip(ys, covs):
        assert y.shape[0] == cov.shape[0] and cov.shape[1] == cov.shape[0]
        assert y.shape[0] == r_bin_centers.shape[0]
        # TODO informative error message when the array is jsut of the wrong shape?/

    # check we've defined all necessary params
    assert _emu.emulator_ndim <= len(fixed_params) + len(param_names) + 1  # for r
    tmp = param_names[:]
    assert not any([key in param_names for key in fixed_params])  # param names can't include the
    tmp.extend(fixed_params.keys())
    assert _emu.check_param_names(tmp, ignore=['r'])

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
        low, high = _emu.get_param_bounds(pname)
        pos0[:, idx] = np.random.randn(nwalkers) * (np.abs(high - low) / 6.0) + (low + high) / 2.0
        # TODO variable with of the initial guess

    return pos0

def run_mcmc(emus,  param_names, ys, covs, r_bin_centers,fixed_params = {}, \
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

    if type(ys) is list:
        ys = np.array(ys)
    if type(covs) is list:
        covs = np.array(covs)

    if len(ys.shape) ==1: # only one measurement
        ys = ys.reshape((-1, 1))
        covs = covs.reshape((-1, 1))

    ncores= _run_tests(ys, covs, r_bin_centers,param_names, fixed_params, ncores)
    num_params = len(param_names)

    combined_inv_covs = np.ones_like(covs)
    for i, _emu, cov in enumerate(covs):
        combined_inv_covs[i] = inv(_emu.ycov + cov)

    sampler = mc.EnsembleSampler(nwalkers, num_params, lnprob,
                                 threads=ncores, args=(param_names, fixed_params, r_bin_centers, ys, combined_inv_covs))

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

def run_mcmc_iterator(emus, param_names, ys, covs, r_bin_centers,fixed_params={},
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
        Whether to return the evaluation of lnprob on the samples along with the samples. Default is Fasle,
        which only returns samples.
    :yield:
        chain, collaposed to the shape ((nsteps-nburn)*nwalkers, len(param_names))
    """

    if type(emus) is not list:
        emus = [emus]

    _emu = emus
    global _emu

    if type(ys) is list:
        ys = np.array(ys)
    if type(covs) is list:
        covs = np.array(covs)

    if len(ys.shape) ==1: # only one measurement
        ys = ys.reshape((-1, 1))
        covs = covs.reshape((-1, 1))

    ncores = _run_tests(ys, covs, r_bin_centers, param_names, fixed_params, ncores)
    num_params = len(param_names)

    combined_inv_covs = np.ones_like(covs)
    for i, _emu, cov in enumerate(covs):
        combined_inv_covs[i] = inv(_emu.ycov + cov)

    sampler = mc.EnsembleSampler(nwalkers, num_params, lnprob,
                                 threads=ncores, args=(param_names, fixed_params, r_bin_centers, ys, combined_inv_covs))

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
