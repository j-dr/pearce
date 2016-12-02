#!/bin/bash
"""The Emu object esentially wraps the George gaussian process code. It handles building, training, and predicting."""

import warnings
from glob import glob
from itertools import izip
from multiprocessing import cpu_count
from os import path
from time import time
from abc import ABCMeta, abstractmethod

import emcee as mc
import george
import numpy as np
import scipy.optimize as op
from george.kernels import *
from scipy.interpolate import interp1d, interp2d
from scipy.linalg import inv
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR

from .ioHelpers import global_file_reader, obs_file_reader
from .trainingData import GLOBAL_FILENAME


# TODO abstract base classes? Depends on what methods need to be subclassed.
# TODO methods I think need to be reordered

class Emu(object):
    # TODO docstrings? here and init.

    __metaclass__ = ABCMeta
    valid_methods = {'gp', 'svr', 'gbdt', 'rf', 'krr'}  # could add more, coud even check if they exist in sklearn

    def __init__(self, training_dir, method='gp', hyperparams={}, params=None, fixed_params={},
                 independent_variable=None):

        assert method in self.valid_methods

        if independent_variable == 'bias':
            raise NotImplementedError("I have to work on how to do xi_mm first.")
        assert independent_variable in {None, 'r2'}  # no bias for now.

        # TODO I hate the assembly bias parameter keys. It'd be nice if the use could pass in something
        # else and I make a change
        # use default if needed
        if params is None:
            from .ioHelpers import DEFAULT_PARAMS as params

        self.method = method
        # TODO store hyperparams?
        # TODO ordered_params needs some checks, cuz i've hardcoded some index stuff in the new version
        # or, be less lazy with those
        # TODO ordered_params should be hidden from the user. It's somethign that exists only for consistancy.
        self.ordered_params = params
        if any(p.name == 'z' for p in self.ordered_params) and any(p.name =='r' for p in self.ordered_params):
            if self.ordered_params[-1].name != 'r' or self.ordered_params[-2].name != 'z':
                warnings.warn('Your ordered params order is possibly problematic! ')
        self.fixed_params = fixed_params
        self.independent_variable = independent_variable

        self.load_training_data(training_dir)
        self.build_emulator(hyperparams)

    ###Data Loading and Manipulation####################################################################################

    def get_data(self, data_dir, em_params, fixed_params, independent_variable):
        """
        Read data in the format compatible with this object and return it
        :param data_dir:
            Directory where data from trainingData is stored
        :param em_params:
            Parameters held fixed by the emulator. Slightly different than fixed_params, used for plotting.
        :param fixed_params:
            Parameters to hold fixed. Only available if data in data_dir is a full hypercube, not a latin hypercube
        :param independent_variable:
            Independant variable to emulate. Options are xi, r2xi, and bias (eventually)..
        :return: None
        """

        input_params = {}
        input_params.update(em_params)
        input_params.update(fixed_params)
        assert len(input_params) - len(self.ordered_params) <= 1  # can exclude r
        # HERE can we exclude z too, somehow?
        sub_dirs = glob(path.join(data_dir, 'a_*'))
        sub_dirs_as = np.array([float(fname[-7:]) for fname in sub_dirs])

        if 'z' in fixed_params:  # don't have to look in dirs that aren't fixed.
            zs = fixed_params['z']
            if type(zs) is float:
                zs = [zs]
            sub_dirs = []
            _sub_dirs_as = []
            for z in zs:
                input_a = 1 / (1 + z)
                idx = np.argmin(np.abs(sub_dirs_as - input_a))
                a = sub_dirs_as[idx]
                if np.abs(a - input_a) > 0.05:  # tolerance
                    raise IOError('No subfolder within tolerance of z=%.3f' % z)
                sub_dirs.append(path.join(data_dir, 'a_%.5f' % a))
                _sub_dirs_as.append(a)
            sub_dirs_as = np.array(_sub_dirs_as)
        self.redshift_bin_centers = 1 / sub_dirs_as - 1


        all_x, all_y, all_yerr = [], [], []

        for sub_dir, a in zip(sub_dirs, sub_dirs_as):

            z = 1.0 / a - 1.0

            bins, cosmo_params, obs, sampling_method = global_file_reader(path.join(sub_dir, GLOBAL_FILENAME))
            # Could add an assert that a = cosmo_params['scale_factor']
            # Not sure if I need a flag here for plot_data; I thnk it's always used with a fixed_params so should be fine.
            if fixed_params and sampling_method == 'LHC':
                if fixed_params.keys() != ['z']:  # allowed:
                    raise ValueError('Fixed parameters is not empty, but the data in data_dir is form a Latin Hypercube. \
                                    Cannot performs slices on a LHC.')

            self.obs = obs

            obs_files = sorted(glob(path.join(sub_dir, 'obs*.npy')))
            cov_files = sorted(
                glob(path.join(sub_dir, 'cov*.npy')))  # since they're sorted, they'll be paired up by params.
            self.scale_bin_centers = (bins[:-1] + bins[1:]) / 2
            scale_nbins = self.scale_bin_centers.shape[0]

            # if self.ordered_params[-1].name != 'r':
            #   scale_nbins=1

            npoints = len(obs_files) * scale_nbins  # each file contains NBINS points in r, and each file is a 6-d point
            # if self.ordered_params[-1].name == 'r':
            #    npoints*=scale_nbins #account for 'r'

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
                # Note is is complex because fixed_params can have floats or arrays of floats.
                # TODO check if a fixed_param is not one of the options i.e. typo
                to_continue = True
                for key, val in input_params.iteritems():
                    if key in {'z', 'r'}:
                        continue #these won't be in params, or have been already screened.
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

                # doing some shuffling and stacking
                file_params = []
                # NOTE could do a param ordering here
                for p in self.ordered_params:
                    if p.name in fixed_params:  # may need to be input_params
                        continue
                    # TODO change 'r' to something else.
                    if p.name == 'r':
                        file_params.append(np.log10(self.scale_bin_centers))
                    elif p.name == 'z':
                        file_params.append(np.ones((scale_nbins,)) * z)
                    else:
                        file_params.append(np.ones((scale_nbins,)) * params[p.name])

                # There is an open question how to handle the fact that x is repeated in the EC case.
                # will punt for now.
                x[idx * scale_nbins:(idx + 1) * scale_nbins, :] = np.stack(file_params).T

                y[idx * scale_nbins:(idx + 1) * scale_nbins], yerr[idx * scale_nbins:(
                                                                                     idx + 1) * scale_nbins] = self._iv_transform(
                    independent_variable, obs, cov)

                # ycov = block_diag(*ycovs)
                # ycov = np.sqrt(np.diag(ycov))

                # remove rows that were skipped due to the fixed thing
                # NOTE: HACK
                # a reshape may be faster.
            zeros_slice = np.any(x != 0.0, axis=1)
                # set the results of these calculations.

                # return x[zeros_slice], y[zeros_slice], yerr[zeros_slice]
            all_x.append(x[zeros_slice])
            all_y.append(y[zeros_slice])
            all_yerr.append(yerr[zeros_slice])

        # TODO sort?
        return np.vstack(all_x), np.hstack(all_y), np.hstack(all_yerr)

    def get_plot_data(self, em_params, training_dir, independent_variable=None, fixed_params={}):
        """
        Similar function to load_training_data. However, returns values for plotting comparisons to the emulator.
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
        """

        x, y, yerr = self.get_data(training_dir, em_params, fixed_params, independent_variable)


        sort_idxs = self._sort_params(x, argsort=True)

        log_bin_centers = np.log10(self.scale_bin_centers)
        # repeat for each row of y
        log_bin_centers = np.tile(log_bin_centers, sort_idxs.shape[0] / len(self.scale_bin_centers))

        return log_bin_centers, y[sort_idxs], yerr[sort_idxs]

    @abstractmethod
    def load_training_data(self, training_dir):
        pass

    def _iv_transform(self, independent_variable, obs, cov):
        """
        Independent variable tranform. Helper function that consolidates this operation all in one place.
        :param independent_variable:
            Which iv to transform to. Current optins are None (just take log) and r2.
        :param obs:
            Observable to transform (xi, wprp, etc.)
        :param cov:
            Covariance of obs
        :return:
            y, yerr the transformed iv's for the emulator
        """
        if independent_variable is None:
            y = np.log10(obs)
            # Approximately true, may need to revisit
            # yerr[idx * NBINS:(idx + 1) * NBINS] = np.sqrt(np.diag(cov)) / (xi * np.log(10))
            y_err = np.sqrt(np.diag(cov)) / (
                obs * np.log(10))  # I think this is right, extrapolating from the above.
        elif independent_variable == 'r2':  # r2
            y = obs * self.scale_bin_centers * self.scale_bin_centers
            y_err = np.sqrt(
                np.diag(cov)) * self.scale_bin_centers  # I think this is right, extrapolating from the above.
        else:
            raise ValueError('Invalid independent variable %s' % independent_variable)

        """
        if independent_variable == 'bias':
            y[idx * NBINS:(idx + 1) * NBINS] = xi / xi_mm
            ycovs.append(cov / np.outer(xi_mm, xi_mm))
        """

        return y, y_err

    def _sort_params(self, t, argsort=False):
        """
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
        """
        if t.shape[0] == 1:
            if argsort:
                return np.array([0])
            return t  # a row array is already sorted!

        if argsort:  # returns indicies that would sort the array
            # weird try structure because this view is very tempermental!
            try:
                idxs = np.argsort(t.view(','.join(['float64' for _ in xrange(min(t.shape))])),
                                  order=['f%d' % i for i in xrange(min(t.shape))], axis=0)
            except ValueError:  # sort with other side
                idxs = np.argsort(t.view(','.join(['float64' for _ in xrange(max(t.shape))])),
                                  order=['f%d' % i for i in xrange(max(t.shape))], axis=0)

            return idxs[:, 0]

        try:
            t = np.sort(t.view(','.join(['float64' for _ in xrange(min(t.shape))])),
                        order=['f%d' % i for i in xrange(min(t.shape))], axis=0).view(np.float)
        except ValueError:  # sort with other side
            t = np.sort(t.view(','.join(['float64' for _ in xrange(max(t.shape))])),
                        order=['f%d' % i for i in xrange(max(t.shape))], axis=0).view(np.float)

        return t

    ###Emulator Building and Training###################################################################################

    def build_emulator(self, hyperparams):
        """
        Initialization of the emulator from recovered training data. Calls submethods depending on "method"
        :param method:
            The machine learning method to use.
.       :param hyperparams
            A dictionary of hyperparameter kwargs for the emulator
        :param fixed_params:
            Parameterst to hold fixed in teh training data
        :return: None
        """

        if self.method == 'gp':
            self._build_gp(hyperparams)
        else:  # an sklearn method
            self._build_skl(hyperparams)

    @abstractmethod
    def _build_gp(self, hyperparams):
        pass

    @abstractmethod
    def _build_skl(self, hyperparams):
        pass

    def _get_initial_guess(self, independent_variable):
        """
        Return the initial guess for the emulator, based on what the iv is. Guesses are learned from
        previous experiments.
        :param independent_variable:
            Which variable to return the guesses for.
        :param fixed_params:
            Parameters to hold fixed; only return guess for parameters that are not fixed.
        :return: initial_guesses, a dictionary of the guess for each parameter
        """

        # default
        ig = {'amp': 1}
        ig.update({p.name: 0.1 for p in self.ordered_params})

        if self.obs == 'xi':
            if independent_variable is None:
                ig = {'amp': 0.481, 'logMmin': 0.1349, 'sigma_logM': 0.089,
                      'logM0': 2.0, 'logM1': 0.204, 'alpha': 0.039,
                      'f_c': 0.041, 'r': 0.040, 'z':1.0}
            else:
                pass
        elif self.obs == 'wp':
            if independent_variable is None:
                ig = {'logMmin': 1.7348042925, 'f_c': 0.327508062386, 'logM0': 15.8416094906,
                      'sigma_logM': 5.36288382789, 'alpha': 3.63498762588, 'r': 0.306139450843,
                      'logM1': 1.66509412286, 'amp': 1.18212664544, 'z':1.0}
        else:
            pass  # no other guesses saved yet.

        # remove entries for variables that are being held fixed.
        for key in self.fixed_params.iterkeys():
            try:
                del ig[key]
            except KeyError:
                pass #can happen for redshift and others. 

        return ig

    def _make_kernel(self, metric):
        """
        Helper method to build a george kernel for GP's and kernel-based regressions.
        :param metric:
            Hyperparams for kernel determining relative length scales and amplitudes
        :return:
            A george ExpSquredKernel object with this metric
        """

        if not metric:
            ig = self._get_initial_guess(self.independent_variable)
        else:
            ig = metric  # use the user's initial guesses

        metric = [ig['amp']]
        for p in self.ordered_params:
            if p.name in self.fixed_params:
                continue
            try:
                metric.append(ig[p.name])
            except KeyError:
                raise KeyError('Key %s was not in the metric.' % p.name)

        metric = np.array(metric)

        a = metric[0]
        # TODO other kernels?
        return a * ExpSquaredKernel(metric[1:], ndim=self.emulator_ndim)

    ###Emulation and methods that Utilize it############################################################################
    def emulate(self, em_params, gp_errs=False):
        """
        Perform predictions with the emulator.
        :param em_params:
            Dictionary of what values to predict at for each param. Values can be
            an array or a float.
        :param gp_errs:
            Boolean, decide whether or not to return the errors from the gp prediction. Default is False.
            Will throw error if method is not gp.
        :return: mu, cov. The predicted value and the covariance matrix for the predictions
        """

        if gp_errs:
            assert self.method == 'gp'  # only has meaning for gp's

        input_params = {}
        input_params.update(self.fixed_params)
        input_params.update(em_params)
        assert len(input_params) - self.emulator_ndim + self.fixed_ndim <=2  # check dimenstionality
        for i in input_params:  # check that the names in input params are all defined in the ordering.
            assert any(i == p.name for p in self.ordered_params)

        # i'd like to remove 'r'. possibly requiring a passed in param?
        t_list = [input_params[p.name] for p in self.ordered_params if p.name in em_params]
        t_grid = np.meshgrid(*t_list)
        t = np.stack(t_grid).T
        # TODO george can sort?
        t = t.reshape((-1, self.emulator_ndim))

        t = self._sort_params(t)

        return self._emulate_helper(t, gp_errs)

    @abstractmethod
    def _emulate_helper(self, t, gp_errs=False):
        pass

    def emulate_wrt_r(self, em_params, r_bin_centers, gp_errs=False):
        """
        Helper function to emulate over r bins.
        :param em_params:
            Parameters to predict at
        :param r_bin_centers:
            Radial bins to predict at
        :param gp_err:
            Boolean, whether or not to use the errors from the GP. Default is False.
            If method is not 'gp', will throw an error
        :return:
            mu, and if gp_err, a cov. matrix
        """
        ep = {}
        ep.update(em_params)
        z_bin_centers = []
        try:
            z_bin_centers = ep['z']
            del ep['z']
        except KeyError:
            pass

        out= self.emulate_wrt_r_z(ep, r_bin_centers, z_bin_centers, gp_errs)
        #Reshape to have ((-1, r_bin_centers)) shape.
        if gp_errs:
            mu, errs  = out
        else:
            mu = out
        #the swapaxes ensures this works correctly
        #mu = mu.swapaxes(1,2).reshape((-1, r_bin_centers.shape[0]))
        mu = mu.reshape((-1, r_bin_centers.shape[0]))
        if not gp_errs:
            return mu
        errs = errs.reshape(mu.shape)
        return mu, errs


    def emulate_wrt_z(self, em_params, z_bin_centers, gp_errs=False):
        """
        Helper function to emulate over z bins.
        :param em_params:
            Parameters to predict at
        :param z_bin_centers:
            Radial bins to predict at
        :param gp_err:
            Boolean, whether or not to use the errors from the GP. Default is False.
            If method is not 'gp', will throw an error
        :return:
            mu, and if gp_err, a cov. matrix
        """
        ep = {}
        ep.update(em_params)
        r_bin_centers = []
        try:
            r_bin_centers = ep['r']
            del ep['r']
        except KeyError:
            pass


        out =  self.emulate_wrt_r_z(ep, r_bin_centers, z_bin_centers, gp_errs)
        if gp_errs:
            mu, errs  = out
        else:
            mu = out
        #TODO not sure this reshape does what I want.
        mu = mu.swapaxes(1,2).reshape((-1, z_bin_centers.shape[0]))
        if not gp_errs:
            return mu
        errs = errs.swapaxes(1,2).reshape(mu.shape)
        return mu, errs

    @abstractmethod
    def emulate_wrt_r_z(self, em_params, r_bin_centers, z_bin_centers, gp_errs=False):
        pass

    # TODO Emulate wrt z

    def estimate_uncertainty(self, truth_dir, N=None):
        """
        Estimate the uncertainty of the emulator by comparing to a "test" box of true values.
        :param truth_dir:
            Name of a directory of true test values, of the same format as the train_dir
        :param N:
            Number of points to compare to. If None (default) will use all points. Else will select random sample.
        :return:
            covariance matrix with dim n_binsxn_bins. Will only have diagonal elemtns of est. uncertainties.
        """
        rms_err = self.goodness_of_fit(truth_dir, N, statistic='rms')

        return np.diag(rms_err ** 2)

    def goodness_of_fit(self, truth_dir, N=None, statistic='r2'):
        """
        Calculate the goodness of fit of an emulator as compared to some validation data.
        :param truth_dir:
            Directory structured similary to the training data, but NOT used for training.
        :param N:
            Number of points to use to calculate G.O.F. measures. "None" tests against all values in truth_dir. If N
            is less than the number of points, N are randomly selected.
        :param statistic:
            What G.O.F. statistic to calculate. Default is R2. Other options are rmsfd, abs(olute), and rel(ative).
        :return: values, a numpy arrray of the calculated statistics at each of the N training opints.
        """
        assert statistic in {'r2', 'rms', 'rmsfd', 'abs', 'rel'}
        if N is not None:
            assert N > 0 and int(N) == N

        x, y, _ = self.get_data(truth_dir, {}, self.fixed_params, self.independent_variable)

        bins, _, _, _ = global_file_reader(path.join(truth_dir, GLOBAL_FILENAME))
        bin_centers = (bins[1:] + bins[:-1]) / 2
        scale_nbins = len(bin_centers)

        # this hack is not a futureproff test!
        # TODO this may need a fix...
        if self.ordered_params[-1].name != 'r':
            x = x[0:-1:scale_nbins, :]

        y = y.reshape((-1, scale_nbins))

        np.random.seed(int(time()))

        if N is not None:  # make a random choice
            idxs = np.random.choice(x.shape[0], N, replace=False)

            x, y = x[idxs], y[idxs]

        pred_y = self._emulate_helper(x, False)
        pred_y = pred_y.reshape((-1, scale_nbins))

        # have to inerpolate...
        if self.ordered_params[-1].name != 'r':

            if not np.all(bin_centers == self.scale_bin_centers):
                bin_centers = bin_centers[self.scale_bin_centers[0] <= bin_centers <= self.scale_bin_centers[-1]]
                new_mu = []
                for mean in pred_y:
                    xi_interpolator = interp1d(self.scale_bin_centers, mean, kind='slinear')
                    interp_mean = xi_interpolator(bin_centers)
                    new_mu.append(interp_mean)
                pred_y = np.array(new_mu)
                y = y[:, self.scale_bin_centers[0] <= bin_centers <= self.scale_bin_centers[-1]]

        if statistic == 'rmsfd':
            return np.sqrt(np.mean((((pred_y - y) ** 2) / (y ** 2)), axis=0))

        elif statistic == 'rms':
            return np.sqrt(np.mean(((pred_y - y) ** 2), axis=0))

        # TODO sklearn methods can do this themselves. But i've already tone the prediction!
        elif statistic == 'r2':  # r2
            SSR = np.sum((pred_y - y) ** 2, axis=0)
            SST = np.sum((y - y.mean(axis=0)) ** 2, axis=0)

            return 1 - SSR / SST

        elif statistic == 'abs':
            return pred_y - y
            # return np.mean((pred_y - y), axis=0)
        else:  # 'rel'
            return (pred_y - y) / y
            # return np.mean((pred_y - y) / y, axis=0)

    @abstractmethod
    def train_metric(self, **kwargs):
        pass

    # TODO this feature is not super useful anymore, and also is poorly defined w.r.t non gp methods.
    # did a lot of work on it tho, maybe i'll leave it around...?
    def _loo_errors(self, y, t):
        """
        Calculate the LOO Jackknife error matrix. This is implemented using the analytic LOO procedure,
        which is much faster than re-doing an inversion for each sample. May be useful if the GP's matrix is not
        accurate.
        :param y:
            Values of the independent variable for the training points, used in the prediction.
        :param t:
            Values of the dependant variables to predict at.
        :return:
            jk_cov: a covariance matrix with the dimensions of cov.
        """
        # from time import time

        assert self.method == 'gp'

        if isinstance(self, ExtraCrispy):
            emulator = self.emulators[0]  # hack for EC, do somethign smarter later
        else:
            emulator = self.emulator

        # We need to perform one full inverse to start.
        K_inv_full = emulator.solver.apply_inverse(np.eye(emulator._alpha.size),
                                                   in_place=True)

        # TODO deepcopy?
        x = self.x[:]

        N = K_inv_full.shape[0]

        mus = np.zeros((N, t.shape[0]))
        # t0 = time()

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

            alpha_m_idx = np.dot(K_m_idx_inv, y[:N - 1] - emulator.mean(x[:N - 1]))

            Kxxs_t = emulator.kernel.value(t, x[:N - 1])

            # Store the estimate for this LOO GP
            mus[idx, :] = np.dot(Kxxs_t, alpha_m_idx) + emulator.mean(t)

            # print mus[idx]
            # print

            # restore the original values for the next loop
            x[[N - 1, idx]] = x[[idx, N - 1]]
            y[[N - 1, idx]] = y[[idx, N - 1]]

            K_inv_full[[idx, N - 1], :] = K_inv_full[[N - 1, idx], :]
            K_inv_full[:, [idx, N - 1]] = K_inv_full[:, [N - 1, idx]]

        # print time() - t0, 's Total'
        # return the jackknife cov matrix.
        cov = (N - 1.0) / N * np.cov(mus, rowvar=False)
        if mus.shape[1] == 1:
            return np.array([[cov]])  # returns float in this case
        else:
            return cov

    def run_mcmc(self, y, cov, bin_centers, nwalkers=1000, nsteps=100, nburn=20, n_cores='all'):
        """
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
        """

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
            # pos0[:, idx] = np.random.uniform(p.low, p.high, size=nwalkers)
            pos0[:, idx] = np.random.normal(loc=(p.high + p.low) / 2, scale=(p.high + p.low) / 10, size=nwalkers)

        sampler.run_mcmc(pos0, nsteps)

        # Note, still an issue of param label ordering here.
        chain = sampler.chain[:, nburn:, :].reshape((-1, self.sampling_ndim))

        return chain


# These functions cannot be instance methods
# Emcee throws a few when trying to compile the liklihood functions that are attached
# to the object calling it
def lnprob(theta, *args):
    """
    The total liklihood for an MCMC. Sadly, can't be an instance of the Emu Object.
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


def lnprior(theta, emu, *args):
    """
    Prior for an MCMC. Currently asserts theta is between the boundaries used to make the emulator.
    Could do something more clever later.
    :param theta:
        The parameters proposed by the sampler.
    :param emu:
        The emulator object. Needs to be accessed to get the priors.
    :return:
        Either 0 or -np.inf, depending if the params are allowed or not.
    """
    return 0 if all(p.low < t < p.high for p, t in izip(emu.ordered_params, theta)) else -np.inf


def lnlike(theta, emu, y, cov, bin_centers):
    """
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
    """
    em_params = {p.name: t for p, t in zip(emu.ordered_params, theta)}

    # using my own notation
    y_bar, G = emu.emulate_wrt_r(em_params, bin_centers)
    # should chi2 be calculated in log or linear?
    # answer: the user is responsible for taking the log before it comes here.

    D = G + cov
    delta = y_bar - y
    chi2 = -0.5 * np.dot(delta, np.dot(inv(D), delta))
    return chi2


class OriginalRecipe(Emu):
    """Emulator that emulates with bins as an implicit parameter. """

    def load_training_data(self, training_dir):
        """
        Read the training data for the emulator and attach it to the object.
        :param training_dir:
            Directory where training data from trainginData is stored. May also be a list of several points.
        :param fixed_params:
            Parameters to hold fixed. Only available if data in training_dir is a full hypercube, not a latin hypercube.
        :return: None
        """
        if type(training_dir) is not list:
            training_dir = [training_dir]

        xs, ys, yerrs = [], [], []
        for td in training_dir:
            x, y, yerr = self.get_data(td, {}, self.fixed_params, self.independent_variable)
            xs.append(x)
            ys.append(y)
            yerrs.append(yerr)

        self.x = np.vstack(xs)
        # hstack for 1-D
        self.y = np.hstack(ys)
        self.yerr = np.hstack(yerrs)

        self.y_hat = np.zeros(self.y.shape[1]) if len(y.shape) > 1 else 0  # self.y.mean(axis = 0)
        self.y -= self.y_hat

        ndim = self.x.shape[1]
        self.fixed_ndim = len(self.fixed_params)
        self.emulator_ndim = ndim  # The number of params for the emulator is different than those in sampling.
        self.sampling_ndim = ndim - 1

    def _build_gp(self, hyperparams):
        """
        Initialize the GP emulator.
        :param hyperparams:
            Key word parameters for the emulator
        :return: None
        """
        # TODO could use more of the hyperparams...
        metric = hyperparams['metric'] if 'metric' in hyperparams else {}
        kernel = self._make_kernel(metric)
        # TODO is it confusing for this to have the same name as the sklearn object with a different API?
        # maybe it should be a property? or private?
        self.emulator = george.GP(kernel)
        # gp = george.GP(kernel, solver=george.HODLRSolver, nleaf=x.shape[0]+1,tol=1e-18)

        self.emulator.compute(self.x, self.yerr, sort=False)  # NOTE I'm using a modified version of george!

    def _build_skl(self, hyperparams):
        """
        Build a scikit learn emulator
        :param hyperparams:
            Key word parameters for the emulator
        :return: None
        """
        skl_methods = {'gbdt': GradientBoostingRegressor, 'rf': RandomForestRegressor, \
                       'svr': SVR, 'krr': KernelRidge}

        if self.method in {'svr', 'krr'}:  # kernel based method
            metric = hyperparams['metric'] if 'metric' in hyperparams else {}
            kernel = self._make_kernel(metric)
            if 'metric' in hyperparams:
                del hyperparams['metric']
            if self.method == 'svr':  # slight difference in these, sadly
                hyperparams['kernel'] = kernel.value
            else:  # krr
                hyperparams['kernel'] = lambda x1, x2: kernel.value(np.array([x1]), np.array([x2]))

        self.emulator = skl_methods[self.method](**hyperparams)
        self.emulator.fit(self.x, self.y)

    def _emulate_helper(self, t, gp_errs):
        """
        Helper function that takes a dependent variable matrix and makes a prediction.
        :param t:
            Dependent variable matrix. Assumed to be in the order defined by ordered_params
        :param gp_errs:
            Whether or not to return errors in the gp case
        :return:
            mu, cov (if gp_errs True). Predicted value for dependetn variable t.
        """
        if self.method == 'gp':
            if gp_errs:
                mu, cov = self.emulator.predict(self.y, t, mean_only=False)
                return mu, np.diag(cov)
            else:
                return self.emulator.predict(self.y, t, mean_only = True)
        else:
            return self.emulator.predict(t)

    # TODO It's not clear to the user if bin_centers should be log or not!
    def emulate_wrt_r_z(self, em_params, r_bin_centers, z_bin_centers, gp_errs=False):
        """
        Conveniance function. Add's 'r' to the emulation automatically, as this is the
        most common use case.
        :param em_params:
            Dictionary of what values to predict at for each param. Values can be array
            or float.
        :param bin_centers:
            Centers of bins to predict at, for each point in HOD-space.
        :return:
        """
        vep = dict(em_params)
        rpc = np.log10(r_bin_centers) if np.any(r_bin_centers)else []  # make sure not to throw an error
        # TODO change 'r' to something more general
        for key, val in zip(['r', 'z'], (rpc, z_bin_centers)):
            if key not in vep and np.any(val):  # key must not already exist and must be nonzero:
                vep[key] = val
        # vep.update({'r': np.log10(r_bin_centers), 'z': z_bin_centers})
        out = self.emulate(vep, gp_errs)
        if gp_errs:
            mu, errs  = out
        else:
            mu = out
        #TODO not sure this reshape does what I want.
        mu = mu.reshape((-1,z_bin_centers.shape[0], r_bin_centers.shape[0]))
        if not gp_errs:
            return mu
        errs = errs.reshape(mu.shape)
        return mu, errs

    def train_metric(self, **kwargs):
        """
        Train the metric parameters of the GP. Has a spotty record of working.
        Best used as used in lowDimTraining.
        If attempted to be used with an emulator that is not GP, will raise an error.
        :param kwargs:
            Kwargs that will be passed into the scipy.optimize.minimize
        :return: success: True if the training was successful.
        """

        # TODO kernel based methods may want to use this...
        assert self.method == 'gp'

        # move these outside? hm.
        def nll(p):
            # Update the kernel parameters and compute the likelihood.
            # params are log(a) and log(m)
            self.emulator.kernel[:] = p
            ll = self.emulator.lnlikelihood(self.y, quiet=True)

            # The scipy optimizer doesn't play well with infinities.
            return -ll if np.isfinite(ll) else 1e25

        # And the gradient of the objective function.
        def grad_nll(p):
            # Update the kernel parameters and compute the likelihood.
            self.emulator.kernel[:] = p
            return -self.emulator.grad_lnlikelihood(self.y, quiet=True)

        p0 = self.emulator.kernel.vector
        results = op.minimize(nll, p0, jac=grad_nll, **kwargs)
        # results = op.minimize(nll, p0, jac=grad_nll, method='TNC', bounds =\
        #   [(np.log(0.01), np.log(10)) for i in xrange(ndim+1)],options={'maxiter':50})
        print results

        self.emulator.kernel[:] = results.x
        self.emulator.recompute()
        # self.metric = np.exp(results.x)

        return results.success


class SpicyBuffalo(Emu):
    """Emulator that emulates wrt one bin (scale or redshift) and carries separate emulators for the other. """

    def __init__(self, training_dir, em_param='r', **kwargs):
        """
        Similar initialization as the superclass with one additional parameter: Em_param
        :param training_dir:
            See above in EMu
        :param em_param:
            Parameter amongst 'r' and 'z' to emulate along. Separate emulators are made along the other parameter.
        :param kwargs:
            As in Emu
        """

        assert em_param in {'r', 'z'}
        self.em_param = em_param  # TODO this name sucks
        super(SpicyBuffalo, self).__init__(training_dir, **kwargs)

    def load_training_data(self, training_dir):
        """
        Read the training data for the emulator and attach it to the object.
        :param training_dir:
            Directory where training data from trainginData is stored.
        :param fixed_params:
            Parameters to hold fixed. Only available if data in training_dir is a full hypercube, not a latin hypercube.
        :return: None
        """
        if type(training_dir) is not list:
            training_dir = [training_dir]

        xs, ys,yerrs = [], [], []
        for td in training_dir:
            x, y, yerr = self.get_data(td, {}, self.fixed_params, self.independent_variable)
            xs.append(x)
            ys.append(y)
            yerrs.append(yerr)

        # this is a bit of a mess. apologies.

        if self.em_param == 'z':
            nbins = len(self.scale_bin_centers)
            self.x = np.vstack(xs)[0:-1:nbins, :]
            #TODO I checked the 'r' case but not this one
            self.y = np.hstack(ys).reshape((-1, nbins))
            self.yerr = np.hstack(yerrs).reshape(self.y.shape)
            #TODO I think this name is confusing
            self.em_bin_centers = self.scale_bin_centers
        else:
            #since z is the second from the end, not so easy as to just skip over them.
            #TODO I feel like this has to be easier than this.
            nbins = len(self.redshift_bin_centers)
            _xs = []
            for x,y,yerr in zip(xs,ys, yerrs):
                n_per_bin = x.shape[0] / nbins
                _xs.append(x[:n_per_bin, :])
            
            self.x = np.vstack(_xs)
            self.y = np.vstack(yy.reshape((-1,nbins), order = 'F') for yy in ys)
            self.yerr = np.vstack(ye.reshape((-1,nbins), order = 'F') for ye in yerrs) 
            '''
            yerr = np.hstack(yerrs)
            self.y = np.vstack([y[i * nbins:(i + 1) * nbins] for i in xrange(y.shape[0]/nbins)])
            
            self.yerr = np.vstack([yerr[i * nbins:(i+1) * nbins] for i in xrange(y.shape[0]/nbins)])
            '''

            self.em_bin_centers = self.redshift_bin_centers


        self.y_hat = np.zeros(self.y.shape[1:]) if len(self.y.shape) > 1 else 0  # self.y.mean(axis = 0)
        self.y -= self.y_hat

        ndim = self.x.shape[1]
        self.fixed_ndim = len(self.fixed_params)
        self.emulator_ndim = ndim  # The number of params for the emulator is different than those in sampling.
        self.sampling_ndim = ndim - 1

    def _build_gp(self, hyperparams):
        """
        Initialize the GP emulator.
        :param hyperparams:
            Key word parameters for the emulator
        :return: None
        """
        # TODO could use more of the hyperparams...
        metric = hyperparams['metric'] if 'metric' in hyperparams else {}
        kernel = self._make_kernel(metric)
        # TODO is it confusing for this to have the same name as the sklearn object with a different API?
        # maybe it should be a property? or private?
        emulator = george.GP(kernel)
        # gp = george.GP(kernel, solver=george.HODLRSolver, nleaf=x.shape[0]+1,tol=1e-18)
        self.emulators = [None for i in xrange(self.yerr.shape[1])]

        for i in xrange(self.yerr.shape[1]):
                emulator = george.GP(kernel)

                emulator.compute(self.x, self.yerr[:, i],sort=False)  # NOTE I'm using a modified version of george!
                self.emulators[i] = emulator

    def _build_skl(self, hyperparams):
        """
        Build a scikit learn emulator
        :param hyperparams:
            Key word parameters for the emulator
        :return: None
        """
        skl_methods = {'gbdt': GradientBoostingRegressor, 'rf': RandomForestRegressor, \
                       'svr': SVR, 'krr': KernelRidge}

        # Same kernel concerns as above.
        if self.method in {'svr', 'krr'}:  # kernel based method
            metric = hyperparams['metric'] if 'metric' in hyperparams else {}
            kernel = self._make_kernel(metric)
            if 'metric' in hyperparams:
                del hyperparams['metric']
            if self.method == 'svr':  # slight difference in these, sadly
                hyperparams['kernel'] = kernel.value
            else:  # krr
                hyperparams['kernel'] = lambda x1, x2: kernel.value(np.array([x1]), np.array([x2]))

        self.emulators = [[skl_methods[self.method](**hyperparams) for i in xrange(self.yerr.shape[1])] \
                          for j in xrange(self.yerr.shape[2])]
        # TODO make sure this works?
        for y, emulator in izip(self.y.T, self.emulators):
            emulator.fit(self.x, y)

    def _emulate_helper(self, t, gp_errs=False):
        """
        Helper function that takes a dependent variable matrix and makes a prediction.
        :param t:
            Dependent variable matrix. Assumed to be in the order defined by ordered_params
        :param gp_errs:
            Whether or not to return errors in the gp case
        :return:
            mu, cov (if gp_errs True). Predicted value for dependetn variable t.
        """
        all_mu = np.zeros((t.shape[0], self.y.shape[1]))  # t down scale_nbins across
        all_err = np.zeros((t.shape[0], self.y.shape[1]))
        #all_cov = []  # np.zeros((t.shape[0], t.shape[0], self.y.shape[1]))

        for idx, (y, y_hat, emulator) in enumerate(izip(self.y.T, self.y_hat, self.emulators)):
            if self.method == 'gp':
                out = emulator.predict(y, t, mean_only=not gp_errs)
                if gp_errs:
                    mu, cov = out
                    all_err[:,idx] = np.diag(cov)
                else:
                    mu = out
            else:
                mu = emulator.predict(t)
        
            # mu and cov come out as (1,) arrays.
            all_mu[:, idx] = mu + y_hat
            # all_err[:, idx] = np.sqrt(np.diag(cov))
            # all_cov[:, :, idx] = cov

        # Reshape to be consistent with my otehr implementation
        mu = all_mu.reshape((-1,))
        err = all_err.reshape(mu.shape)
        if not gp_errs:
            return mu
        return mu, err
        '''
        cov = np.zeros((mu.shape[0], mu.shape[0]))
        scale_nbins = self.y.shape[1]
        redshift_nbins = self.y.shape[2]
        # This seems pretty inefficient; i'd like a more elegant way to do this.
        for n, c in enumerate(all_cov):
            for i, row in enumerate(c):
                for j, val in enumerate(row):
                    cov[i * scale_nbins + n, j * scale_nbins + n] = val
        return mu, cov
        '''

    def emulate_wrt_r_z(self, em_params, r_bin_centers, z_bin_centers, gp_errs=False, kind='linear'):
        """
        Conveniance function. Add's 'r' and 'z' to the emulation automatically, as this is the
        most common use case.
        :param em_params:
            Dictionary of what values to predict at for each param. Values can be array
            or float.
        :param r_bin_centers:
            Centers of scale bins to predict at, for each point in HOD-space.
        :param z_bin_centers:
            Centers of redshift bins to predict at, for each point in HOD-space.
        :param kind:
            Kind of interpolation to do, is necessary. Default is slinear.
        :return:
            Mu and Cov, the predicted mu and covariance at em_params and bin_centers. If bin_centers
            is not equal to the bin_centers in the training data, the mean is interpolated as is the variance.
            Off diagonal elements are set to 0.
        """

        vep = dict(em_params)
        rpc = np.log10(r_bin_centers) if np.any(r_bin_centers)else []  # make sure not to throw an error
        # TODO change 'r' to something more general
        for key, val in zip(['r', 'z'], (rpc, z_bin_centers)):
            if key == self.em_param and key not in vep and np.any(val):  # key must not already exist and must be nonzero:
                vep[key] = val

        out = self.emulate(vep, gp_errs)
        # don't need to interpolate!
        for key, input_bin, owned_bin in zip(['r', 'z'], [r_bin_centers, z_bin_centers],
                                        [self.scale_bin_centers, self.redshift_bin_centers]):
            if key!=self.em_param and np.any(input_bin) and np.any(input_bin != owned_bin):
                break
        else:
            # if any that exist are not equal to the owned ones, keep going. Else, return.
            return out

        if gp_errs:
            mu, err = out
        else:
            mu = out
            # I'm a bad, lazy man
            err = np.zeros_like(mu)

        #remmeber, these are the bins of what were not emulating! 
        if self.em_param == 'z':
            em_bin_centers = r_bin_centers
        else:
            em_bin_centers = z_bin_centers


        # Remember, these are for the parameter we're NOT emulating
        # TODO check bin_centers in bounds!
        # TODO is there any reasonable way to interpolate the covariance?
        all_mu = mu.reshape((-1, len(self.em_bin_centers)))
        all_err = err.reshape(all_mu.shape)

        # TODO can I combine these two?
        if all_mu.shape[0] == 1 or len(all_mu.shape) == 1:  # just one calculation
            xi_interpolator = interp1d(self.em_bin_centers, all_mu, kind=kind)
            new_mu = xi_interpolator(em_bin_centers)
            if not gp_errs:
                return new_mu

            err_interp = interp1d(self.em_bin_centerss, all_err, kind=kind)
            new_err = err_interp(em_bin_centers)
            return new_mu, new_err

        # TODO ... is this rightt? Was that all I had to do?
        new_mu, new_err = [], []
        for mean, err in izip(all_mu, all_err):
            xi_interpolator = interp1d(self.em_bin_centers, mean, kind=kind)
            interp_mean = xi_interpolator(em_bin_centers)
            new_mu.append(interp_mean)
            err_interp = interp1d(self.em_bin_centers, err, kind=kind)
            interp_err = err_interp(em_bin_centers)
            new_err.append(interp_err)

        # TODO no clue if this makes sense; may need a resisze
        mu = np.vstack(new_mu)
        err = np.vstack(new_err)
        if self.em_param == 'r':
            #unfortunate reshape constraint.
            #stops us from rewriting superclasses though!
            mu = mu.T
            err = err.T 
        if gp_errs:
            return mu, err
        return mu

    def train_metric(self, **kwargs):
        """
        Train the emulator. Has a spotty record of working. Better luck may be had with the NAMEME code.
        :param kwargs:
            Kwargs that will be passed into the scipy.optimize.minimize
        :return: success: True if the training was successful.
        """

        assert self.method == 'gp'

        # emulators is a list containing refernces to the same object. this should still work!
        emulator = self.emulators[0]

        # move these outside? hm.
        def nll(p):
            # Update the kernel parameters and compute the likelihood.
            # params are log(a) and log(m)
            emulator.kernel[:] = p
            # check this has the right direction
            ll = np.mean(emulator.lnlikelihood(y, quiet=True) for y in self.y.flatten())

            # The scipy optimizer doesn't play well with infinities.
            return -ll if np.isfinite(ll) else 1e25

        # And the gradient of the objective function.
        def grad_nll(p):
            # Update the kernel parameters and compute the likelihood.
            emulator.kernel[:] = p
            # mean or sum?
            return -np.mean(emulator.grad_lnlikelihood(y, quiet=True) for y in self.y.flatten())

        p0 = emulator.kernel.vector
        results = op.minimize(nll, p0, jac=grad_nll, **kwargs)
        # results = op.minimize(nll, p0, jac=grad_nll, method='TNC', bounds =\
        #   [(np.log(0.01), np.log(10)) for i in xrange(ndim+1)],options={'maxiter':50})

        emulator.kernel[:] = results.x
        emulator.recompute()

        return results.success


class ExtraCrispy(Emu):
    """Emulator that emulates with bins as an implicit parameter. """

    def load_training_data(self, training_dir):
        """
        Read the training data for the emulator and attach it to the object.
        :param training_dir:
            Directory where training data from trainginData is stored.
        :param fixed_params:
            Parameters to hold fixed. Only available if data in training_dir is a full hypercube, not a latin hypercube.
        :return: None
        """
        if type(training_dir) is not list:
            training_dir = [training_dir]

        xs, ys, yerrs = [], [], []
        for td in training_dir:
            x, y, yerr = self.get_data(td, {}, self.fixed_params, self.independent_variable)

            xs.append(x)
            ys.append(y)
            yerrs.append(yerr)

        # now, need to do some shuffling to get these right.
        # NOTE this involves creating a large array and slicing it down, essentially. Not the most efficient.
        # Possibly more efficient would be to load in the format we do here, and then do the transform on that.
        # However, the one in the superclass is the "standard" approach.

        scale_nbins = len(self.scale_bin_centers)
        redshift_nbins = len(self.redshift_bin_centers)
        self.x = np.vstack(xs)[0:-1:scale_nbins*redshift_nbins, :]
        self.y = np.hstack(ys).reshape((-1, scale_nbins, redshift_nbins))
        #self.yerr = np.zeros_like(self.y)
        self.yerr = np.hstack(yerrs).reshape((-1, scale_nbins, redshift_nbins))
        self.y_hat = np.zeros(self.y.shape[1:]) if len(self.y.shape) > 1 else 0  # self.y.mean(axis = 0)
        self.y -= self.y_hat

        ndim = self.x.shape[1]
        self.fixed_ndim = len(self.fixed_params)
        self.emulator_ndim = ndim  # The number of params for the emulator is different than those in sampling.
        self.sampling_ndim = ndim - 1

    def _build_gp(self, hyperparams):
        """
        Initialize the GP emulator.
        :param hyperparams:
            Key word parameters for the emulator
        :return: None
        """
        # TODO could use more of the hyperparams...
        metric = hyperparams['metric'] if 'metric' in hyperparams else {}
        kernel = self._make_kernel(metric)
        # TODO is it confusing for this to have the same name as the sklearn object with a different API?
        # maybe it should be a property? or private?
        self.emulators = [[None for i in xrange(self.yerr.shape[1])] for j in xrange(self.yerr.shape[2])]

        for i in xrange(self.yerr.shape[1]):
            for j in xrange(self.yerr.shape[2]):
                emulator = george.GP(kernel)

                emulator.compute(self.x, self.yerr[:, i,j],sort=False)  # NOTE I'm using a modified version of george!
                self.emulators[j][i] = emulator
                # For EC, i'm storing an emulator per bin.
                # I'll have to thikn about how to differ the hyperparams.
                # For now, it'll replicate the same behavior as before.
                # TODO not happy, in general, EC has "emulators" not "emulator" like the others.
                # Arguement would be this should be all abstracted out from the user.

    def _build_skl(self, hyperparams):
        """
        Build a scikit learn emulator
        :param hyperparams:
            Key word parameters for the emulator
        :return: None
        """
        skl_methods = {'gbdt': GradientBoostingRegressor, 'rf': RandomForestRegressor, \
                       'svr': SVR, 'krr': KernelRidge}

        # Same kernel concerns as above.
        if self.method in {'svr', 'krr'}:  # kernel based method
            metric = hyperparams['metric'] if 'metric' in hyperparams else {}
            kernel = self._make_kernel(metric)
            if 'metric' in hyperparams:
                del hyperparams['metric']
            if self.method == 'svr':  # slight difference in these, sadly
                hyperparams['kernel'] = kernel.value
            else:  # krr
                hyperparams['kernel'] = lambda x1, x2: kernel.value(np.array([x1]), np.array([x2]))

        self.emulators = [[skl_methods[self.method](**hyperparams) for i in xrange(self.yerr.shape[1])] \
                          for j in xrange(self.yerr.shape[2])]
        # TODO make sure this works?
        for scale_ys, scale_emulators in izip(self.y.T, self.emulators):
            for y, emulator in izip(scale_ys, scale_emulators):
                emulator.fit(self.x, y)

    def _emulate_helper(self, t, gp_errs=False):
        """
        Helper function that takes a dependent variable matrix and makes a prediction.
        :param t:
            Dependent variable matrix. Assumed to be in the order defined by ordered_params
        :param gp_errs:
            Whether or not to return errors in the gp case
        :return:
            mu, cov (if gp_errs True). Predicted value for dependetn variable t.
        """
        all_mu = np.zeros((t.shape[0], self.y.shape[1], self.y.shape[2]))  # t down scale_nbins across
        all_err = np.zeros_like(all_mu)
        #all_cov = []  # np.zeros((t.shape[0], t.shape[0], self.y.shape[1]))
        #TODO pythonic iteration. Not happening right now.
        for scale_idx in xrange(self.y.shape[1]):
            for z_idx in xrange(self.y.shape[2]):
                if self.method=='gp':
                    out = self.emulators[z_idx][scale_idx].predict(self.y[:, scale_idx, z_idx], t, mean_only = not gp_errs)
                    if gp_errs:
                        mu, cov = out
                        all_err[:, scale_idx, z_idx] = np.diag(cov)
                    else:
                        mu = out
                else:
                    mu = self.emulators[z_idx][scale_idx].predict(t)
                # mu and cov come out as (1,) arrays.
                all_mu[:, scale_idx, z_idx] = mu + self.y_hat[scale_idx, z_idx]
                # all_err[:, idx] = np.sqrt(np.diag(cov))
                # all_cov[:, :, idx] = cov

        # Reshape to be consistent with my otehr implementation
        mu = all_mu.reshape((-1,))
        if not gp_errs:
            return mu
        return mu, all_err.rehsape((-1,))

        '''
        cov = np.zeros((mu.shape[0], mu.shape[0]))
        scale_nbins = self.y.shape[1]
        redshift_nbins = self.y.shape[2]
        # This seems pretty inefficient; i'd like a more elegant way to do this.
        for n, c in enumerate(all_cov):
            for i, row in enumerate(c):
                for j, rrow in enumerate(row):
                    for k, val in enumerate(rrow):
                        # TODO i am broken pls fix me i don't understand indicies
                        cov[j * scale_nbins + i * redshift_nbins + n, j * scale_nbins + n] = val
        return mu, cov
        '''

    def emulate_wrt_r_z(self, em_params, r_bin_centers, z_bin_centers, gp_errs=False, kind='cubic'):
        """
        Conveniance function. Add's 'r' and 'z' to the emulation automatically, as this is the
        most common use case.
        :param em_params:
            Dictionary of what values to predict at for each param. Values can be array
            or float.
        :param r_bin_centers:
            Centers of scale bins to predict at, for each point in HOD-space.
        :param z_bin_centers:
            Centers of redshift bins to predict at, for each point in HOD-space.
        :param kind:
            Kind of interpolation to do, is necessary. Default is slinear.
        :return:
            Mu and Cov, the predicted mu and covariance at em_params and bin_centers. If bin_centers
            is not equal to the bin_centers in the training data, the mean is interpolated as is the variance.
            Off diagonal elements are set to 0.
        """
        # turns out this how it already works!
        #assert self.em_param not in em_params
        assert 'r' not in em_params
        assert 'z' not in em_params
        out = self.emulate(em_params, gp_errs)
        # don't need to interpolate!
        for input_bin, owned_bin in zip([r_bin_centers, z_bin_centers], [self.scale_bin_centers, self.redshift_bin_centers]):
            if np.any(input_bin) and np.any(input_bin != owned_bin):
                break
        else:
            # if any that exist are not equal to the owned ones, keep going. Else, return.
            return out

        if gp_errs:
            mu, err = out
        else:
            mu = out
            # I'm a bad, lazy man
            err = np.zeros_like(mu)

        if kind == 'cubic' and any(len(bc) < 3 for bc in (self.scale_bin_centers, self.redshift_bin_centers)):
            kind = 'linear' #can only do cubic if there's 3 points


        # TODO check bin_centers in bounds!
        # TODO is there any reasonable way to interpolate the covariance?
        all_mu = mu.reshape((-1, len(self.redshift_bin_centers), len(self.scale_bin_centers)))
        all_err = err.reshape(all_mu.shape)

        # TODO can I combine these two?
        if all_mu.shape[0] == 1:  # just one calculation
            xi_interpolator = interp2d(self.scale_bin_centers, self.redshift_bin_centers, all_mu[0], kind=kind)
            new_mu = xi_interpolator(r_bin_centers, z_bin_centers)
            if not gp_errs:
                return new_mu

            err_interp = interp2d(self.scale_bin_centers, self.redshift_bin_centers, all_err[0], kind=kind)
            new_err = err_interp(r_bin_centers, z_bin_centers)
            return new_mu, new_err

        # TODO ... is this rightt? Was that all I had to do?
        new_mu, new_err = [], []
        for mean, err in izip(all_mu, all_err):

            xi_interpolator = interp2d(self.scale_bin_centers, self.redshift_bin_centers, mean, kind=kind)
            interp_mean = xi_interpolator(r_bin_centers, z_bin_centers)
            new_mu.append(interp_mean)
            err_interp = interp2d(self.scale_bin_centers, self.redshift_bin_centers, err, kind=kind)
            interp_err = err_interp(r_bin_centers, z_bin_centers)
            new_err.append(interp_err)

        #TODO no clue if this makes sense; may need a resisze
        mu = np.vstack(new_mu)
        err = np.vstack(new_err)
        if gp_errs:
            return mu, err
        return mu

    # TODO I could probably make this learn the metric for each individual emulator
    # TODO could make this learn the metric for other kernel based emulators...
    def train_metric(self, **kwargs):
        """
        Train the emulator. Has a spotty record of working. Better luck may be had with the NAMEME code.
        :param kwargs:
            Kwargs that will be passed into the scipy.optimize.minimize
        :return: success: True if the training was successful.
        """

        assert self.method == 'gp'

        # emulators is a list containing refernces to the same object. this should still work!
        emulator = self.emulators[0]

        # move these outside? hm.
        def nll(p):
            # Update the kernel parameters and compute the likelihood.
            # params are log(a) and log(m)
            emulator.kernel[:] = p
            # check this has the right direction
            ll = np.sum(emulator.lnlikelihood(y, quiet=True) for y in self.y.flatten())

            # The scipy optimizer doesn't play well with infinities.
            return -ll if np.isfinite(ll) else 1e25

        # And the gradient of the objective function.
        def grad_nll(p):
            # Update the kernel parameters and compute the likelihood.
            emulator.kernel[:] = p
            # mean or sum?
            return -np.mean(emulator.grad_lnlikelihood(y, quiet=True) for y in self.y.flatten())

        p0 = emulator.kernel.vector
        results = op.minimize(nll, p0, jac=grad_nll, **kwargs)
        # results = op.minimize(nll, p0, jac=grad_nll, method='TNC', bounds =\
        #   [(np.log(0.01), np.log(10)) for i in xrange(ndim+1)],options={'maxiter':50})

        emulator.kernel[:] = results.x
        emulator.recompute()

        return results.success
