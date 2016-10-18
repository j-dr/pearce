#!/bin/bash
'''The Emu object esentially wraps the George gaussian process code. It handles building, training, and predicting.'''

from os import path
from time import time
import warnings
from glob import glob
from itertools import izip
from multiprocessing import cpu_count

import numpy as np
import scipy.optimize as op
from scipy.linalg import inv
from scipy.interpolate import interp1d
import george
from george.kernels import *
import emcee as mc

from .trainingData import PARAMS, GLOBAL_FILENAME
from .ioHelpers import global_file_reader, obs_file_reader

class Emu(object):
    # NOTE I believe that I'll have to subclass for the scale-indepent and original versions.
    # I'm going to write the original up so I can understand how I want this to work in particular, and may
    # have to copy a lot of it to the subclass

    # TODO load initial guesses from file.
    def __init__(self, training_dir, params=PARAMS, independent_variable=None, fixed_params={}, metric={}):

        '''
        # TODO change name of type
        self.type = type
        try:
            assert self.type in {'original_recipe', 'extra_crispy'}
        except AssertionError:
            raise AssertionError('Type %s not valid for Emu.'%self.type)
        '''
        if independent_variable == 'bias':
            raise NotImplementedError("I have to work on how to do xi_mm first.")

        assert independent_variable in {None,'r2'}  # no bias for now.

        self.ordered_params = params

        self.get_training_data(training_dir, independent_variable, fixed_params)
        self.build_emulator(independent_variable, fixed_params, metric)

    # I can get LHC from the training data. If any coordinate equals any other in its column we know!
    def get_training_data(self, training_dir, independent_variable, fixed_params):
        '''Implemented in subclasses. '''
        raise NotImplementedError

    def _iv_transform(self, independent_variable, obs, cov):
        '''
        Independent variable tranform. Helper function that consolidates this operation all in one place.
        :param independent_variable:
            Which iv to transform to. Current optins are None (just take log) and r2.
        :param obs:
            Observable to transform (xi, wprp, etc.)
        :param cov:
            Covariance of obs
        :return:
            y, yerr the transformed iv's for the emulator
        '''
        if independent_variable is None:
            y = np.log10(obs)
            # Approximately true, may need to revisit
            # yerr[idx * NBINS:(idx + 1) * NBINS] = np.sqrt(np.diag(cov)) / (xi * np.log(10))
            y_err = np.sqrt(np.diag(cov)) / (
                obs * np.log(10))  # I think this is right, extrapolating from the above.
        elif independent_variable=='r2':  # r2
            y = obs * self.bin_centers * self.bin_centers
            y_err = np.sqrt(np.diag(cov)) * self.bin_centers  # I think this is right, extrapolating from the above.
        else:
            raise ValueError('Invalid independent variable %s'%independent_variable)

        '''
        if independent_variable == 'bias':
            y[idx * NBINS:(idx + 1) * NBINS] = xi / xi_mm
            ycovs.append(cov / np.outer(xi_mm, xi_mm))
        '''

        return y, y_err

    def _sort_params(self, t, argsort=False):
        '''
        Sort the parameters in a defined away given the orderering.
        :param t:
            Parameter vector to sort. Should have dims (N, N_params) and be in the order
            defined by ordered_params
        :param argsort:
            If true, return indicies that would sort the array rather than the sorted array itself.
            Default is False.
        :return:
            If not argsort, returns the sorted array by column and row. 
            If argsort, return the indicies that would sort the array.
        '''
        if t.shape[0] == 1:
            if argsort:
                return np.array([0])
            return t #a row array is already sorted!
        
        if argsort: #returns indicies that would sort the array
            #weird try structure because this view is very tempermental!
            try:
                idxs = np.argsort(t.view(','.join(['float64' for _ in xrange(min(t.shape))])),
                    order=['f%d' % i for i in xrange(min(t.shape))], axis=0)
            except ValueError: #sort with other side
                idxs = np.argsort(t.view(','.join(['float64' for _ in xrange(max(t.shape))])),
                    order=['f%d' % i for i in xrange(max(t.shape))], axis=0)

            return idxs[:,0]

        try:
            t = np.sort(t.view(','.join(['float64' for _ in xrange(min(t.shape))])),
                order=['f%d' % i for i in xrange(min(t.shape))], axis=0).view(np.float)
        except ValueError: #sort with other side
            t = np.sort(t.view(','.join(['float64' for _ in xrange(max(t.shape))])),
                order=['f%d' % i for i in xrange(max(t.shape))], axis=0).view(np.float)

        return t

    def build_emulator(self, independent_variable, fixed_params, metric={}):
        '''
        Initialization of the emulator from recovered training data.
        :param independent_variable:
            independent_variable to emulate.
        :param fixed_params:
            Parameterst to hold fixed in teh training data
        :param metric:
            Optional. Define a metric instead of the ones that are built in.
        :return: None
        '''
        if not metric:
            ig = self.get_initial_guess(independent_variable, fixed_params)
        else:
            ig = metric  # use the user's initial guesses

        self.metric = [ig['amp']]
        for p in self.ordered_params:
            if p.name in fixed_params:
                continue
            try:
                self.metric.append(ig[p.name])
            except KeyError:
                raise KeyError('Key %s was not in the metric.' % p.name)

        self.metric = np.array(self.metric)

        a = self.metric[0] 
        kernel = a * ExpSquaredKernel(self.metric[1:], ndim=self.emulator_ndim)
        self.gp = george.GP(kernel)
        # gp = george.GP(kernel, solver=george.HODLRSolver, nleaf=x.shape[0]+1,tol=1e-18)
        self.fixed_params = fixed_params  # remember which params were fixed

        self.gp.compute(self.x, self.yerr,sort=False)  # NOTE I'm using a modified version of george!

    # Not sure this will work at all in an LHC scheme.
    def get_plot_data(self, em_params, training_dir, independent_variable=None, fixed_params={}):
        '''
        Similar function to get_training_data. However, returns values for plotting comparisons to the emulator.
        :param em_params:
            Similar to fixed params. A dictionary of values held fixed in the emulator, as opposed to fixed_params
            which are values held fixed in the training data.
        :param training_dir:
            Directory where training data from trainginData is stored.
        :param independent_variable:
            Independant variable to emulate. Options are xi, r2xi, and bias (eventually).
        :param fixed_params:
            Parameters to hold fixed. Only available if data in training_dir is a full hypercube, not a latin hypercube.
        :return: log_r, y, yerr for the independent variable at the points specified by fixed nad em params.
        '''
        assert len(em_params) + len(fixed_params) - len(self.ordered_params) <= 1  # can exclude r

        bins, cosmo_params,obs, method = global_file_reader(path.join(training_dir, GLOBAL_FILENAME))

        if method == 'LHC':
            raise ValueError('The data in training_dir is form a Latin Hypercube. \
                                        It is not possible to get plot data from this data. ')

        obs_files = sorted(glob(path.join(training_dir, 'obs*.npy')))
        cov_files = sorted(
            glob(path.join(training_dir, 'cov*.npy')))  # since they're sorted, they'll be paired up by params.
        bin_centers = (bins[:-1] + bins[1:]) / 2
        nbins = bin_centers.shape[0]

        npoints = len(obs_files)  # each file contains NBINS points in r, and each file is a 6-d point

        log_bin_centers = np.zeros((npoints, nbins))
        y = np.zeros((npoints, nbins))
        yerr = np.zeros((npoints, nbins))

        x = np.zeros((npoints,len(em_params))) 

        warned = False
        num_skipped = 0
        num_used = 0
        for idx, (obs_file, cov_file) in enumerate(izip(obs_files, cov_files)):
            params, obs, cov = obs_file_reader(obs_file, cov_file)

            # skip values that aren't where we've fixed them to be.
            # It'd be nice to do this before the file I/O. Not possible without putting all info in the filename.
            # or, a more nuanced file structure
            # TODO check if a fixed_param is not one of the options
            if any(params[key] != val for key, val in fixed_params.iteritems()):
                continue

            #same as the above for emulation params
            #more complex since em_params can have a float or array
            to_continue = True
            for key, val in em_params.iteritems():
                if type(val) is type(y):
                    if np.all(np.abs(params[key] - val) > 1e-3):
                        break
                elif np.abs(params[key] - val) > 1e-3:
                    break
            else:
                to_continue = False  # no break

            if to_continue:
                continue

            if np.any(np.isnan(cov)) or np.any(np.isnan(obs)):
                if not warned:
                    warnings.warn('WARNING: NaN detected. Skipping point in %s' % cov_file)
                    warned = True
                num_skipped += 1
                continue

            num_used += 1

            log_bin_centers[idx] = np.log10(bin_centers)
            y[idx], yerr[idx] = self._iv_transform(independent_variable,obs, cov)

            x[idx] =  np.array([params[p.name] for p in self.ordered_params if p.name in params])

        # remove rows that were skipped due to the fixed thing
        # NOTE: HACK
        # a reshape may be faster.
        zeros_slice = np.all(y != 0.0, axis=1)

        print x[zeros_slice].shape

        sort_idxs = self._sort_params(x[zeros_slice], argsort=True)

        return log_bin_centers[zeros_slice][sort_idxs], y[zeros_slice][sort_idxs], yerr[zeros_slice][sort_idxs]

    def get_initial_guess(self, independent_variable, fixed_params):
        '''
        Return the initial guess for the emulator, based on what the iv is. Guesses are learned from
        previous experiments.
        :param independent_variable:
            Which variable to return the guesses for.
        :param fixed_params:
            Parameters to hold fixed; only return guess for parameters that are not fixed.
        :return: initial_guesses, a dictionary of the guess for each parameter
        '''

        #default
        ig = {'amp': 1}
        ig.update({p.name: 0.1 for p in self.ordered_params})

        if self.obs == 'xi':
            if independent_variable is None:
                ig = {'amp': 0.481, 'logMmin': 0.1349, 'sigma_logM': 0.089,
                  'logM0': 2.0, 'logM1': 0.204, 'alpha': 0.039,
                  'f_c': 0.041, 'r': 0.040}
            else:
                pass
        elif self.obs == 'wp':
            if independent_variable is None:
                ig = {'logMmin': 1.7348042925, 'f_c': 0.327508062386, 'logM0':15.8416094906,
                        'sigma_logM':5.36288382789, 'alpha': 3.63498762588, 'r': 0.306139450843,
                        'logM1': 1.66509412286, 'amp': 1.18212664544}
        else:
            pass #no other guesses saved yet.

        # remove entries for variables that are being held fixed.
        for key in fixed_params.iterkeys():
            del ig[key]

        return ig

    def train(self, **kwargs):
        raise NotImplementedError

    def emulate(self, em_params, jackknife_errors=False):
        raise NotImplementedError

    def emulate_wrt_r(self, em_params, bin_centers, jackknife_errors=False):
        raise NotImplementedError

    def _jackknife_errors(self, y, t):
        '''
        Calculate the LOO Jackknife error matrix. This is implemented using the analytic LOO procedure,
        which is much faster than re-doing an inversion for each sample. May be useful if the GP's matrix is not
        accurate.
        :param y:
            Values of the independent variable for the training points, used in the prediction.
        :param t:
            Values of the dependant variables to predict at.
        :return:
            jk_cov: a covariance matrix with the dimensions of cov.
        '''
        #from time import time

        # We need to perform one full inverse to start.
        K_inv_full = self.gp.solver.apply_inverse(np.eye(self.gp._alpha.size),
                                                  in_place=True)

        # TODO deepcopy?
        x = self.x[:]

        N = K_inv_full.shape[0]

        mus = np.zeros((N, t.shape[0]))
        #t0 = time()

        # iterate over training points to leave out
        for idx in xrange(N):

            # swap the values of the LOO point and the last point.
            x[[N - 1, idx]] = x[[idx, N - 1]]
            y[[N - 1, idx]] = y[[idx, N - 1]]

            K_inv_full[[idx, N - 1], :] = K_inv_full[[N - 1, idx], :]
            K_inv_full[:, [idx, N - 1]] = K_inv_full[:, [N - 1, idx]]

            # the inverse of the LOO GP
            # formula found via MATH
            K_m_idx_inv = K_inv_full[:N - 1, :][:, :N - 1] \
                          - np.outer(K_inv_full[N - 1, :N - 1], K_inv_full[:N - 1, N - 1]) / K_inv_full[N - 1, N - 1]

            alpha_m_idx = np.dot(K_m_idx_inv, y[:N - 1] - self.gp.mean(x[:N - 1]))

            Kxxs_t = self.gp.kernel.value(t, x[:N - 1])

            # Store the estimate for this LOO GP
            mus[idx, :] = np.dot(Kxxs_t, alpha_m_idx) + self.gp.mean(t)

            # print mus[idx]
            # print

            # restore the original values for the next loop
            x[[N - 1, idx]] = x[[idx, N - 1]]
            y[[N - 1, idx]] = y[[idx, N - 1]]

            K_inv_full[[idx, N - 1], :] = K_inv_full[[N - 1, idx], :]
            K_inv_full[:, [idx, N - 1]] = K_inv_full[:, [N - 1, idx]]

        #print time() - t0, 's Total'
        # return the jackknife cov matrix.
        cov = (N - 1.0) / N * np.cov(mus, rowvar=False)
        if mus.shape[1] ==1 :
            return np.array([[cov]]) #returns float in this case
        else:
            return cov

    def goodness_of_fit(self, truth_dir, N=None, statistic='r2'):
        '''
        Calculate the goodness of fit of an emulator as compared to some validation data.
        :param truth_dir:
            Directory structured similary to the training data, but NOT used for training.
        :param N:
            Number of points to use to calculate G.O.F. measures. "None" tests against all values in truth_dir. If N
            is less than the number of points, N are randomly selected.
        :param statistic:
            What G.O.F. statistic to calculate. Default is R2. Other options are rmsfd, abs(olute), and rel(ative).
        :return: values, a numpy arrray of the calculated statistics at each of the N training opints.
        '''
        assert statistic in {'r2', 'rmsfd','abs','rel'}

        obs_files = sorted(glob(path.join(truth_dir, 'obs*.npy')))
        cov_files = sorted(glob(path.join(truth_dir, 'cov*.npy')))

        np.random.seed(int(time()))

        if N is None:
            idxs = np.arange(len(obs_files))
        else:
            idxs = np.random.choice(len(obs_files), N, replace=False)

        values = []
        bins, _, _, _ = global_file_reader(path.join(truth_dir, GLOBAL_FILENAME))
        bin_centers = (bins[1:] + bins[:-1]) / 2
        for idx in idxs:
            params, true_obs, _ = obs_file_reader(obs_files[idx], cov_files[idx])
            pred_log_obs, _ = self.emulate_wrt_r(params, bin_centers)

            if statistic == 'rmsfd':
                values.append(np.sqrt(np.mean(((pred_log_obs - np.log10(true_obs)) ** 2) / (np.log10(true_obs) ** 2))))
            elif statistic == 'r2':  # r2
                SSR = np.sum((pred_log_obs - np.log10(true_obs)) ** 2)
                SST = np.sum((np.log10(true_obs) - np.log10(true_obs).mean()) ** 2)

                values.append(1 - SSR / SST)

            elif statistic == 'abs':
                values.append((pred_log_obs - np.log10(true_obs)) )
                #values.append((10**pred_log_obs - true_obs) )
            else: #'rel'
                values.append(((pred_log_obs - np.log10(true_obs)) )/np.log10(true_obs) )
                #values.append((10**pred_log_obs - true_obs )/true_obs )

        return np.array(values)

    def run_mcmc(self, y, cov, bin_centers, nwalkers=1000, nsteps=100, nburn=20, n_cores='all'):
        '''
        Run an MCMC sampler, using the emulator. Uses emcee to perform sampling.
        :param y:
            A true y value to recover the parameters of theta. NOTE: The emulator emulates some indepedant variables in 
            log space, others in linear. Make sure y is in the same space!
        :param cov:
            The measurement covariance matrix of y
        :param bin_centers:
            The centers of the bins y is measured in (radial or angular).
        :param nwalkers:
            Optional. Number of walkers for emcee. Default is 1000.
        :param nsteps:
            Optional. Number of steps for emcee. Default is 100.
        :param nburn:
            Optional. Number of burn-in steps for emcee. Default is 20.
        :param n_cores:
            Number of cores, either an iteger or 'all'. Default is 'all'.
        :return:
            chain, a numpy array of the sample chain.
        '''

        assert n_cores == 'all' or n_cores > 0
        if type(n_cores) is not str:
            assert int(n_cores) == n_cores

        max_cores = cpu_count()
        if n_cores == 'all':
            n_cores = max_cores
        elif n_cores > max_cores:
            warnings.warn('n_cores invalid. Changing from %d to maximum %d.' % (n_cores, max_cores))
            n_cores = max_cores
            # else, we're good!

        assert y.shape[0] == cov.shape[0] and cov.shape[1] == cov.shape[0]
        assert y.shape[0] == bin_centers.shape[0]


        sampler = mc.EnsembleSampler(nwalkers, self.sampling_ndim, lnprob,
                                      threads=n_cores, args=(self, y, cov, bin_centers))

        pos0 = np.zeros((nwalkers, self.sampling_ndim))
        # The zip ensures we don't use the params that are only for the emulator
        for idx, (p, _) in enumerate(izip(self.ordered_params, xrange(self.sampling_ndim))):
            #pos0[:, idx] = np.random.uniform(p.low, p.high, size=nwalkers)
            pos0[:, idx] = np.random.normal(loc = (p.high+p.low)/2, scale = (p.high+p.low)/10, size = nwalkers)

        sampler.run_mcmc(pos0, nsteps)

        # Note, still an issue of param label ordering here.
        chain = sampler.chain[:, nburn:, :].reshape((-1, self.sampling_ndim))

        return chain

#These functions cannot be instance methods
#Emcee throws a few when trying to compile the liklihood functions that are attached
# to the object calling it
def lnprob(theta, *args):
    '''
    The total liklihood for an MCMC. Sadly, can't be an instance of the Emu Object.
    :param theta:
        Parameters for the proposal
    :param args:
        Arguments to pass into the liklihood
    :return:
        Log Liklihood of theta, a float.
    '''
    lp = lnprior(theta, *args)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, *args)

def lnprior(theta, emu, *args):
    '''
    Prior for an MCMC. Currently asserts theta is between the boundaries used to make the emulator.
    Could do something more clever later.
    :param theta:
        The parameters proposed by the sampler.
    :param emu:
        The emulator object. Needs to be accessed to get the priors.
    :return:
        Either 0 or -np.inf, depending if the params are allowed or not.
    '''
    return 0 if all(p.low < t < p.high for p, t in izip(emu.ordered_params, theta)) else -np.inf

def lnlike(theta, emu, y, cov, bin_centers):
    '''
    The liklihood of parameters theta given the other parameters and the emulator.
    :param theta:
        Proposed parameters.
    :param emu:
        The emulator object. Used to perform the emulation.
    :param y:
        The measured value of the observable to compare to the emulator.
    :param cov:
        The covariance matrix of the measured values.
    :param bin_centers:
        The centers of the bins y is measured in, angular or radial.
    :return:
        The log liklihood of theta given the measurements and the emulator.
    '''
    em_params = {p.name: t for p,t in zip(emu.ordered_params, theta)}

    # using my own notation
    y_bar, G = emu.emulate_wrt_r(em_params, bin_centers)
    # should chi2 be calculated in log or linear?
    # answer: the user is responsible for taking the log before it comes here.

    D = G + cov
    delta = y_bar - y
    chi2 = -0.5 * np.dot(delta, np.dot(inv(D), delta))
    return chi2

class OriginalRecipe(Emu):
    '''Emulator that emulates with bins as an implicit parameter. '''
    def get_training_data(self, training_dir, independent_variable, fixed_params):
        '''
        Read the training data for the emulator and attach it to the object.
        :param training_dir:
            Directory where training data from trainginData is stored.
        :param independent_variable:
            Independant variable to emulate. Options are xi, r2xi, and bias (eventually).
        :param fixed_params:
            Parameters to hold fixed. Only available if data in training_dir is a full hypercube, not a latin hypercube.
        :return: None
        '''

        bins, cosmo_params, obs, method = global_file_reader(path.join(training_dir, GLOBAL_FILENAME))

        if fixed_params and method == 'LHC':
            raise ValueError('Fixed parameters is not empty, but the data in training_dir is form a Latin Hypercube. \
                                Cannot performs slices on a LHC.')

        self.obs = obs

        obs_files = sorted(glob(path.join(training_dir, 'obs*.npy')))
        cov_files = sorted(
            glob(path.join(training_dir, 'cov*.npy')))  # since they're sorted, they'll be paired up by params.
        self.bin_centers = (bins[:-1] + bins[1:]) / 2
        nbins = self.bin_centers.shape[0]

        npoints = len(obs_files) * nbins  # each file contains NBINS points in r, and each file is a 6-d point

        varied_params = set([p.name for p in self.ordered_params]) - set(fixed_params.keys())
        ndim = len(varied_params)  # lest we forget r

        x = np.zeros((npoints, ndim))
        y = np.zeros((npoints,))
        yerr = np.zeros((npoints,))

        warned = False
        num_skipped = 0
        num_used = 0
        for idx, (obs_file, cov_file) in enumerate(izip(obs_files, cov_files)):
            params, obs, cov = obs_file_reader(obs_file, cov_file)

            # skip values that aren't where we've fixed them to be.
            # It'd be nice to do this before the file I/O. Not possible without putting all info in the filename.
            # or, a more nuanced file structure
            # TODO check if a fixed_param is not one of the options i.e. typo
            if any(params[key] != val for key, val in fixed_params.iteritems()):
                continue

            if np.any(np.isnan(cov)) or np.any(np.isnan(obs)):
                if not warned:
                    warnings.warn('WARNING: NaN detected. Skipping point in %s' % cov_file)
                    warned = True
                num_skipped += 1
                continue

            num_used += 1

            # doing some shuffling and stacking
            file_params = []
            # NOTE could do a param ordering here
            for p in self.ordered_params:
                if p.name in fixed_params:
                    continue
                #TODO change 'r' to something else.
                if p.name == 'r':
                    file_params.append(np.log10(self.bin_centers))
                else:
                    file_params.append(np.ones((nbins,)) * params[p.name])

            x[idx * nbins:(idx + 1) * nbins, :] = np.stack(file_params).T
            # TODO the time has come to do something smarter for bias... I will ignore for now.

            y[idx * nbins:(idx + 1) * nbins], yerr[idx * nbins:(idx + 1) * nbins] = self._iv_transform(independent_variable,obs, cov)

        # ycov = block_diag(*ycovs)
        # ycov = np.sqrt(np.diag(ycov))

        # remove rows that were skipped due to the fixed thing
        # NOTE: HACK
        # a reshape may be faster.
        zeros_slice = np.all(x != 0.0, axis=1)
        # set the results of these calculations.
        self.fixed_ndim = len(fixed_params)
        self.emulator_ndim = ndim  # The number of params for the emulator is different than those in sampling.
        self.sampling_ndim = ndim - 1
        self.x = x[zeros_slice]
        self.y = y[zeros_slice]
        self.yerr = yerr[zeros_slice]

        self.y_hat = 0  # self.y.mean()
        self.y -= self.y_hat  # mean-subtract.

    def train(self, **kwargs):
        '''
            Train the emulator. Has a spotty record of working. Better luck may be had with the NAMEME code.
            :param kwargs:
                Kwargs that will be passed into the scipy.optimize.minimize
            :return: success: True if the training was successful.
            '''

        # move these outside? hm.
        def nll(p):
            # Update the kernel parameters and compute the likelihood.
            # params are log(a) and log(m)
            self.gp.kernel[:] = p
            ll = self.gp.lnlikelihood(self.y, quiet=True)

            # The scipy optimizer doesn't play well with infinities.
            return -ll if np.isfinite(ll) else 1e25

        # And the gradient of the objective function.
        def grad_nll(p):
            # Update the kernel parameters and compute the likelihood.
            self.gp.kernel[:] = p
            return -self.gp.grad_lnlikelihood(self.y, quiet=True)

        p0 = self.gp.kernel.vector
        results = op.minimize(nll, p0, jac=grad_nll, **kwargs)
        # results = op.minimize(nll, p0, jac=grad_nll, method='TNC', bounds =\
        #   [(np.log(0.01), np.log(10)) for i in xrange(ndim+1)],options={'maxiter':50})
        print results

        self.gp.kernel[:] = results.x
        self.gp.recompute()
        self.metric = np.exp(results.x)

        return results.success

    def emulate(self, em_params, jackknife_errors=False):
        '''
        Perform predictions with the emulator.
        :param varied_em_params:
            Dictionary of what values to predict at for each param. Values can be
            an array or a float.
        :return: mu, cov. The predicted value and the covariance matrix for the predictions
        '''
        input_params = {}
        input_params.update(self.fixed_params)
        input_params.update(em_params)
        assert len(input_params) == self.emulator_ndim+self.fixed_ndim  # check dimenstionality
        for i in input_params:
            assert any(i == p.name for p in self.ordered_params)

        # i'd like to remove 'r'. possibly requiring a passed in param?
        t_list = [input_params[p.name] for p in self.ordered_params if p.name in em_params]
        t_grid = np.meshgrid(*t_list)
        t = np.stack(t_grid).T
        # TODO george can sort?
        t = t.reshape((-1, self.emulator_ndim))

        t = self._sort_params(t)

        if jackknife_errors:
            mu = self.gp.predict(self.y, t, mean_only=True)
            cov = self._jackknife_errors(self.y, t)
        else:
            mu, cov = self.gp.predict(self.y, t)

        # mu = mu.reshape((-1, len(self.bin_centers)))
        # errs = errs.reshape((-1, len(self.bin_centers)))
        # Note ordering is unclear if em_params has more than 1 value.
        return mu, cov
    #TODO It's not clear to the user if bin_centers should be log or not!
    def emulate_wrt_r(self, em_params, bin_centers, jackknife_errors=False):
        '''
        Conveniance function. Add's 'r' to the emulation automatically, as this is the
        most common use case.
        :param em_params:
            Dictionary of what values to predict at for each param. Values can be array
            or float.
        :param bin_centers:
            Centers of bins to predict at, for each point in HOD-space.
        :return:
        '''
        vep = dict(em_params)
        # TODO change 'r' to something more general
        vep.update({'r': np.log10(bin_centers)})
        out = self.emulate(vep, jackknife_errors)
        return out
    
class ExtraCrispy(Emu):
    '''Emulator that emulates with bins as an implicit parameter. '''

    def get_training_data(self, training_dir, independent_variable, fixed_params):
        '''
        Read the training data for the emulator and attach it to the object.
        :param training_dir:
            Directory where training data from trainginData is stored.
        :param independent_variable:
            Independant variable to emulate. Options are xi, r2xi, and bias (eventually).
        :param fixed_params:
            Parameters to hold fixed. Only available if data in training_dir is a full hypercube, not a latin hypercube.
        :return: None
        '''

        bins, cosmo_params,obs, method = global_file_reader(path.join(training_dir, GLOBAL_FILENAME))

        if fixed_params and method == 'LHC':
            raise ValueError('Fixed parameters is not empty, but the data in training_dir is form a Latin Hypercube. \
                                Cannot performs slices on a LHC.')

        self.obs = obs

        obs_files = sorted(glob(path.join(training_dir, 'obs*.npy')))
        cov_files = sorted(
            glob(path.join(training_dir, 'cov*.npy')))  # since they're sorted, they'll be paired up by params.
        self.bin_centers = (bins[:-1] + bins[1:]) / 2
        nbins = self.bin_centers.shape[0]
        # HERE
        npoints = len(obs_files)  # each file contains NBINS points in r, and each file is a 6-d point

        # HERE
        varied_params = set(self.ordered_params) - set(fixed_params.keys())
        ndim = len(varied_params)  # lest we forget r

        # not sure about this.

        x = np.zeros((npoints, ndim))
        y = np.zeros((npoints, nbins))
        # yerr = np.zeros((npoints, nbins))

        warned = False
        num_skipped = 0
        num_used = 0
        for idx, (obs_file, cov_file) in enumerate(izip(obs_files, cov_files)):
            #TODO change funciton and return
            params, obs, cov = obs_file_reader(obs_file, cov_file)

            # skip values that aren't where we've fixed them to be.
            # It'd be nice to do this before the file I/O. Not possible without putting all info in the filename.
            # or, a more nuanced file structure
            # TODO check if a fixed_param is not one of the options i.e. typo
            if any(params[key] != val for key, val in fixed_params.iteritems()):
                continue

            if np.any(np.isnan(cov)) or np.any(np.isnan(obs)):
                if not warned:
                    warnings.warn('WARNING: NaN detected. Skipping point in %s' % cov_file)
                    warned = True
                num_skipped += 1
                continue

            num_used += 1

            # doing some shuffling and stacking
            file_params = []
            # NOTE could do a param ordering here
            for p in self.ordered_params:
                if p.name in fixed_params:
                    continue
                file_params.append(params[p.name])

            x[idx, :] = np.stack(file_params).T

            y[idx], _ = self._iv_transform(independent_variable, obs, cov)

        # ycov = block_diag(*ycovs)
        # ycov = np.sqrt(np.diag(ycov))

        # remove rows that were skipped due to the fixed thing
        # NOTE: HACK
        # a reshape may be faster.
        zeros_slice = np.all(x != 0.0, axis=1)
        # set the results of these calculations.
        self.fixed_ndim = len(fixed_params)
        self.emulator_ndim = ndim  # the number of parameters for the emulator different than a sampler, in some cases
        self.sampling_ndim = ndim  # TODO not sure if this is the best design choice.
        self.x = x[zeros_slice]
        self.y = y[zeros_slice, :]
        self.yerr = np.zeros((self.x.shape[0],))  # We don't use errors for extra crispy!
        # self.yerr = yerr[zeros_slice]

        self.y_hat = np.zeros((self.y.shape[1],))  # self.y.mean(axis=0)
        self.y -= self.y_hat  # mean-subtract.

    # TODO train isn't the best word here, since I used "training data" for another purpose
    def train(self, **kwargs):
        '''
        Train the emulator. Has a spotty record of working. Better luck may be had with the NAMEME code.
        :param kwargs:
            Kwargs that will be passed into the scipy.optimize.minimize
        :return: success: True if the training was successful.
        '''

        # move these outside? hm.
        def nll(p):
            # Update the kernel parameters and compute the likelihood.
            # params are log(a) and log(m)
            self.gp.kernel[:] = p
            # check this has the right direction
            ll = np.sum(self.gp.lnlikelihood(y, quiet=True) for y in self.y)

            # The scipy optimizer doesn't play well with infinities.
            return -ll if np.isfinite(ll) else 1e25

        # And the gradient of the objective function.
        def grad_nll(p):
            # Update the kernel parameters and compute the likelihood.
            self.gp.kernel[:] = p
            # mean or sum?
            return -np.mean(self.gp.grad_lnlikelihood(y, quiet=True) for y in self.y)

        p0 = self.gp.kernel.vector
        results = op.minimize(nll, p0, jac=grad_nll, **kwargs)
        # results = op.minimize(nll, p0, jac=grad_nll, method='TNC', bounds =\
        #   [(np.log(0.01), np.log(10)) for i in xrange(ndim+1)],options={'maxiter':50})

        self.gp.kernel[:] = results.x
        self.gp.recompute()
        self.metric = np.exp(results.x)

        return results.success

    def emulate(self, em_params, jackknife_errors =False):
        '''
        Perform predictions with the emulator.
        :param varied_em_params:
            Dictionary of what values to predict at for each param. Values can be
            an array or a float.
        :return: mu, cov. The predicted value and the covariance matrix for the predictions
        '''
        input_params = {}
        input_params.update(self.fixed_params)
        input_params.update(em_params)
        assert len(input_params) == self.emulator_ndim+self.fixed_ndim  # check dimenstionality
        for i in input_params:
            assert any(i == p.name for p in self.ordered_params)

        # i'd like to remove 'r'. possibly requiring a passed in param?
        t_list = [input_params[p.name] for p in self.ordered_params if p.name in em_params]
        t_grid = np.meshgrid(*t_list)
        t = np.stack(t_grid).T
        # should add a fixed_param dim
        t = t.reshape((-1, self.emulator_ndim))
        t = self._sort_params(t)

        all_mu = np.zeros((t.shape[0], self.y.shape[1]))  # t down nbins across
        # all_err = np.zeros((t.shape[0], self.y.shape[1]))
        all_cov = []# np.zeros((t.shape[0], t.shape[0], self.y.shape[1]))

        for idx, (y, y_hat) in enumerate(izip(self.y.T, self.y_hat)):
            if jackknife_errors:
                mu = self.gp.predict(y,t, mean_only=True)
                cov = self._jackknife_errors(y, t)
            else:
                mu, cov = self.gp.predict(y, t)
            # mu and cov come out as (1,) arrays.
            all_mu[:, idx] = mu
            # all_err[:, idx] = np.sqrt(np.diag(cov))
            #all_cov[:, :, idx] = cov
            all_cov.append(cov)

        #Reshape to be consistent with my otehr implementation
        mu = all_mu.reshape((-1,))
        cov = np.zeros((mu.shape[0], mu.shape[0]))
        nbins = self.y.shape[1]
        #This seems pretty inefficient; i'd like a more elegant way to do this.
        for n, c in enumerate(all_cov):
            for i, row in enumerate(c):
                for j, val in enumerate(row):
                    cov[i*nbins+n, j*nbins+n] = val
        return mu, cov

    def emulate_wrt_r(self, em_params, bin_centers,jackknife_errors=False, kind='slinear'):
        '''
        Conveniance function. Add's 'r' to the emulation automatically, as this is the
        most common use case.
        :param em_params:
            Dictionary of what values to predict at for each param. Values can be array
            or float.
        :param bin_centers:
            Centers of scale bins to predict at, for each point in HOD-space.
        :param kind:
            Kind of interpolation to do, is necessary. Default is slinear.
        :return:
            Mu and Cov, the predicted mu and covariance at em_params and bin_centers. If bin_centers
            is not equal to the bin_centers in the training data, the mean is interpolated as is the variance.
            Off diagonal elements are set to 0.
        '''
        # turns out this how it already works!
        mu, cov = self.emulate(em_params, jackknife_errors)
        # don't need to interpolate!
        if np.all(bin_centers == self.bin_centers):
            return mu, cov

        # TODO check bin_centers in bounds!
        # TODO is there any reasonable way to interpolate the covariance?
        all_err= np.sqrt(np.diag(cov))

        all_mu = mu.reshape((-1, len(self.bin_centers)))
        all_err = all_err.reshape((-1, len(self.bin_centers)))

        if len(all_mu.shape) == 1:  # just one calculation
            xi_interpolator = interp1d(self.bin_centers, all_mu, kind=kind)
            new_mu = xi_interpolator(bin_centers)
            err_interp = interp1d(self.bin_centers, all_err, kind=kind)
            new_err = err_interp(bin_centers)
            return new_mu, new_err

        new_mu, new_err = [], []
        for mean, err in izip(all_mu, all_err):
            xi_interpolator = interp1d(self.bin_centers, mean, kind=kind)
            interp_mean = xi_interpolator(bin_centers)
            new_mu.append(interp_mean)
            err_interp = interp1d(self.bin_centers, err, kind=kind)
            interp_err = err_interp(bin_centers)
            new_err.append(interp_err)
        mu = np.array(new_mu).reshape((-1,))
        cov = np.diag(np.array(new_err).reshape((-1,)))
        return mu, cov
