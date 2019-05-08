#!bin/bash
"""
New object that generates training data in a sparter way.
Rather than firing off many jobs to do the training, will do everying with MPI in one program.
Will also write to an hdf5 file, to reduce the enormous clutter produced so far.
"""
from os import path
from glob import glob
from shutil import copyfile
from subprocess import call
from time import time
import warnings
from itertools import product
import yaml
import numpy as np
from scipy.optimize import minimize_scalar
import pandas as pd
import h5py

from pearce.mocks import cat_dict


class Trainer(object):

    def __init__(self, config_fname):
        """
        This function creates a trainer object from a YAML config
        and preapres everything necessary to run the training jobs
        :param config_fname:
            A YAML config file containing all the details needed to train
        """

        try:
            assert path.isfile(config_fname)
        except AssertionError:
            raise AssertionError("%s is not a valid filename." % config_fname)

        with open(config_fname, 'r') as ymlfile:
            cfg = yaml.load(ymlfile)

        # TODO could do more user babysitting here

        self.prep_cosmology(cfg['cosmology'])
        # TODO one thing to consider is how to have HOD-independent emu
        # quantities like xi_mm could require this.
        self.prep_hod(cfg['HOD'])
        self.prep_observation(cfg['observation'])
        self.prep_computation(cfg['computation'])

    def prep_cosmology(self, cosmo_cfg):
        """
        Prepare the cosmology catalogs we'll need.
        Identifies the relevant cat objects and stores them in a list
        Also identifies the params needed to load up specific catalogs come compute time.
        :param cosmo_cfg:
            A dictionary (loaded from the YAML file) of all parameters relevant to cosmological emulation.
            To load multiple instances of the same type of box (like, several different cosmologies in the same
            simulation set) use the argument 'boxno', which can be a list, an integer, or a colon separated pair
            like 0:40 which will be turned into a range range(0,40). If loading up one, it doesn't need to be specified.

            'simname' specifies the name of the cached cats. Other args passed in as kwargs to the __init__
            Other optional args are 'particles' and 'downsampling_factor' which determine if and which particle catalog
            to load alongside the halo catalog. Certain observables require particles, so be wary!
        """

        # if we have to load an ensemble of cats, do so. otherwise, just load up one.
        # it's also possible to load up one

        # TODO let the user specify redshifts instead? annoying.
        scale_factors = cosmo_cfg['scale_factors']
        self._scale_factors = [scale_factors] if type(scale_factors) is float else scale_factors
        cosmo_cfg['scale_factors'] = self._scale_factors #apply the same change

        if 'boxno' in cosmo_cfg:
            if ':' in cosmo_cfg['boxno']:  # shorthand to do ranges, like 0:40
                splitstr = cosmo_cfg['boxno'].split(':')
                boxnos = list(range(int(splitstr[0]), int(splitstr[1])))
            else:
                boxnos = cosmo_cfg['boxno']
                boxnos = [boxnos] if type(boxnos) is int else boxnos
            del cosmo_cfg['boxno']
            self.cats = []

            if 'realization' in cosmo_cfg:
                if ':' in cosmo_cfg['realization']:  # shorthand to do ranges, like 0:40
                    splitstr = cosmo_cfg['realization'].split(':')
                    realizations = list(range(int(splitstr[0]), int(splitstr[1])))
                else:
                    realizations = cosmo_cfg['realization']
                    realizations = [realizations] if type(realizations) is int else realizations

                del cosmo_cfg['realization']

                for boxno in boxnos:
                    for realization in realizations:
                        self.cats.append(
                            cat_dict[cosmo_cfg['simname']](boxno=boxno,realization = realization,  **cosmo_cfg))  # construct the specified catalog!

            # if there are multiple cosmos, they need to have a boxno kwarg
            else:
                for boxno in boxnos:
                    self.cats.append(
                        cat_dict[cosmo_cfg['simname']](boxno=boxno, **cosmo_cfg))  # construct the specified catalog!

        else:  # fixed cosmology
            self.cats = [cat_dict[cosmo_cfg['simname']](**cosmo_cfg)]

        self._cosmo_param_names,_ = self.cats[0]._get_cosmo_param_names_vals()
        self._cosmo_param_vals = np.zeros((len(self.cats), len(self._cosmo_param_names)))

        for idx, cat in enumerate(self.cats):
            _, vals = cat._get_cosmo_param_names_vals() 

            self._cosmo_param_vals[idx,:] = vals

        # In the future if there are more annoying params i'll turn this into a loop or something.
        # TODO does yaml parse these to bools automatically?
        self._particles = bool(cosmo_cfg['particles']) if 'particles' in cosmo_cfg else False
        #print cosmo_cfg.keys()
        self._downsample_factor = float(cosmo_cfg['downsample_factor']) if 'downsample_factor' in cosmo_cfg else 1e-3

    def prep_hod(self, hod_cfg):
        """
        Prepare HOD information we'll need.
        Stores  the HOD name and other kwargs.

        Most importantly, loads or generates the HOD parameters. Can either accept a
        'paramfile' which will be read as a pandas dataframe. The column names will be stored
        as the parameter names, and the values as the HODs to use.

        The alternative is to pass in an 'ordered_params' dictionary or ordered_dict, with
        keys as param names and values with tuples defining the lower and upper bounds of the
        parameter. Ex: ordered_params['logMmin'] = (12.5, 14.5). Also required is
        'num_hods', the number of parameters to generate. A random, non-optimized
        LHC will be generated.
        :param hod_cfg:
            A dictionary (loaded from the YAML file) of parameters relevant to HOD emulation.
            Needs to specify the name of the HOD model to be loaded; available ones are in customHODModels or halotools.
            All other args will be passed in as kwargs.
        """
        # TODO possibly better error messaging
        self.hod = hod_cfg['model']  # string defining which model
        # TODO confirm is valid?
        del hod_cfg['model']
        # A little unnerved by the assymetry between this and cosmology
        # the cosmo params are attached to the objects
        if 'paramfile' in hod_cfg:
            assert path.exists(hod_cfg['paramfile'])
            df = pd.read_csv(hod_cfg['paramfile'])
            self._hod_param_names = df.columns
            self._hod_param_vals = df.data

            del hod_cfg['paramfile']
        else:
            assert 'ordered_params' in hod_cfg
            assert 'num_hods' in hod_cfg #TODO better test here

            # make a LHC from scratch
            ordered_params = hod_cfg['ordered_params']

            if 'fixed_nd' in hod_cfg and 'logMmin' in ordered_params:
                self._logMmin_bounds = ordered_params['logMmin']
                del ordered_params['logMmin'] # do I want to keep this for any reason?
            else:
                self._logMmin_bounds = None

            self._hod_param_names = list(ordered_params.keys())

            seed = hod_cfg.get("seed", None)
            self._hod_param_vals = self._make_LHC(ordered_params, hod_cfg['num_hods'], seed = seed)

            del hod_cfg['ordered_params']
            del hod_cfg['num_hods']

        if 'min_ptcl' in hod_cfg:
            self._min_ptcl = hod_cfg['min_ptcl']
            del hod_cfg['min_ptcl']
        else:
            self._min_ptcl = 200 #default

        try:
            self._fixed_nd = float(hod_cfg['fixed_nd'])
            del hod_cfg['fixed_nd']
        except KeyError:
            self._fixed_nd = None

        self._hod_kwargs = hod_cfg
        # need scale factors too, but just use them from cosmology

    def _make_LHC(self, ordered_params, N, seed = None):
        """Return a vector of points in parameter space that defines a latin hypercube.
            :param ordered_params:
                OrderedDict that defines the ordering, name, and ranges of parameters
                used in the trianing data. Keys are the names, value of a tuple of (lower, higher) bounds
            :param N:
                Number of points per dimension in the hypercube. Default is 500.
            :return
                A latin hyper cube sample in HOD space in a numpy array.
        """
        if seed is None:
            seed = int(time())
        np.random.seed(seed)

        points = []
        # by linspacing each parameter and shuffling, I ensure there is only one point in each row, in each dimension.
        for plow, phigh in ordered_params.values():
            point = np.linspace(plow, phigh, num=N)
            np.random.shuffle(point)  # makes the cube random.
            points.append(point)
        return np.stack(points).T

    def prep_observation(self, obs_cfg):
        """
        Prepares variables related to the calculation of the observable (xi, wp, etc).
        Must specify 'obs', the observable to be calculated. Must be a calc_<> for a cat object
        May also specify 'bins', the binning scheme for a scale dependent variable (in Mpc or degrees)
        'n_repops', the number of times to repopulate a given hod+cosmology combination
        'log_obs', whether the log of the observable should be taken
        All other arguments will be passed into the observable function as kwargs
        :param obs_cfg:
            A dictionary (from a YAML file) specifying the arguments
        """
        # required kwargs, the observable being calculated and the binning scheme along with it.
        self.obs = obs_cfg['obs']
        del obs_cfg['obs']
        # Would be nice to have a scheme to specify bins without listing them all
        # TODO warn if you chose an observable that requires bins
        if 'bins' in obs_cfg:
            self.scale_bins = np.array((obs_cfg['bins']))
            del obs_cfg['bins']
            self.n_bins = len(self.scale_bins)-1
        else:
            # For things like number density, don't need bins. this is temporary though.
            self.scale_bins = None
            self.n_bins = 1

        # number of repopulations to average together for each cosmo/hod combination
        # default is 1
        n_repops = 1
        if 'n_repops' in obs_cfg:
            n_repops = int(obs_cfg['n_repops'])
            assert n_repops > 1
            del obs_cfg['n_repops']
        self._n_repops = n_repops

        # whether or not to take the log of the observable after calculation
        # certain things, like xi and wp, are easier to emulate in log space
        # will define a tranform function for either case
        log_obs = bool(obs_cfg['log_obs']) if 'log_obs' in obs_cfg else True
        self._transform_func = np.log10 if log_obs else lambda x: x

        if 'log_obs' in obs_cfg:
            del obs_cfg['log_obs']


        # seed to use for galaxy population
        if 'seed' in obs_cfg:
            self.pop_seed = obs_cfg['seed']
            del obs_cfg['seed']
        else:
            self.pop_seed = None

        # This goes through the motions of getting calc observable.
        # Each cat will have to retrieve their own version though.

        dummy_cat = self.cats[0]
        try:
            # get the function to calculate the observable
            calc_observable = getattr(dummy_cat, 'calc_%s' % self.obs)
        except AttributeError:
            raise AttributeError(
                "Observable %s is not available. Please check if you specified correctly, only\n observables that are calc_<> cat functions are allowed." %self.obs)

        # check to see if there are kwargs for calc_observable
        #print calc_observable
        args = calc_observable.args  # get function args
        kwargs = {}
        # obs_cfg may have args for our function.
        if any(arg in obs_cfg for arg in args):
            kwargs.update({arg: obs_cfg[arg] for arg in args if arg in obs_cfg})
        if n_repops > 1:  # don't do jackknife in this case
            if 'do_jackknife' in obs_cfg and obs_cfg['do_jackknife']:
                warnings.warn('WARNING: Cannot perform jackknife with n_repops>1. Turning off jackknife.')
                if 'do_jackknife' in args:
                    kwargs['do_jackknife'] = False

        self._calc_observable_kwargs = kwargs

    def _get_calc_observable(self, cat):
        """
        Get the calc_observable function for this cat. Ex, will retrieve a reference to calc_xi
        if the observabel is xi. Additionally, if log_obs is true, will wrap in a log10 automatically.
        :param cat:
            The cat object to retrive calc_observabel for.
        :return:
            calc_observable, a function that calculates the observable for cat with no args
        """
        # TODO wanna make sure this works with no args
        _calc_observable = getattr(cat, 'calc_%s' % self.obs)
        if self._calc_observable_kwargs:
            if self.scale_bins is not None:
                return lambda : _calc_observable(self.scale_bins, **self._calc_observable_kwargs)
            else:
                return lambda : _calc_observable(**self._calc_observable_kwargs)
        else:
            if self.scale_bins is not None:
                return lambda : _calc_observable(self.scale_bins)
            else:
                return _calc_observable

    def prep_computation(self, comp_cfg):
        """
        #TODO
        """

        # here i could export enviornment variables that ensure the cores per node is right?
        self.output_fname = comp_cfg['filename']
        overwrite = comp_cfg['overwrite'] if 'overwrite' in comp_cfg else False
        if path.exists(self.output_fname) and not overwrite:
            raise IOError("Overwrite is False or Not Specified, but the file exists. Please change the config.")


        self.system = comp_cfg.get("system", None)
        self.n_jobs = comp_cfg.get("n_jobs", -1)
        self.max_time = comp_cfg.get("max_time", -1)

        if 'queue_skipper' in comp_cfg and comp_cfg['queue_skipper']:
            self._skip_queue = True
            assert self.system is not None, "Please specify the system when using the queue skipper."
            assert self.n_jobs > 0, "Please specify the number of jobs for the queue skipper"
            assert self.max_time > 0, "Please specify a maximum time for the queue skipper."
        else:
            self._skip_queue = False


    def _get_group_name(self, cosmo_idx, scale_factor):
        """
        Helper function to return a standard group/dataset name based on the indicies
        # TODO
        """
        return "cosmo_no_%02d/a_%.3f"%(cosmo_idx, self._scale_factors[scale_factor])

    def _add_logMmin(self, hod_params, cat):
        """
        In the fixed number density case, find the logMmin value that will match the nd given hod_params
        :param: hod_params:
            The other parameters besides logMmin
        :param cat:
            the catalog in question
        :return:
            None. hod_params will have logMmin added to it.
        """
        hod_params['logMmin'] = 13.0 #initial guess
        #cat.populate(hod_params) #may be overkill, but will ensure params are written everywhere
        def func(logMmin, hod_params):
            hod_params.update({'logMmin':logMmin}) 
            return (cat.calc_analytic_nd(hod_params) - self._fixed_nd)**2

        res = minimize_scalar(func, bounds = self._logMmin_bounds, args = (hod_params,), options = {'maxiter':100}, method = 'Bounded')

        # assuming this doens't fail
        print('logMmin', res.x)
        hod_params['logMmin'] = res.x

    def _divide_tasks(self, size):
        """
        divide each unique HOD, Scale Factor, and cosmology task between
        every jobs
        :param size:
            Number of MPI jobs/ submission scripts to divide amongst.
        :return:
            sendbuf, a numpy array of shape (size, ceil(n_tasks/size), 3)
            Each index of the size dimension contains the HOD, SF, and cosmology index of the task
        """
        all_param_idxs = np.array(list(product(range(len(self.cats)), range(len(self._scale_factors)),
                                range(self._hod_param_vals.shape[0]))))

        n_combos = len(all_param_idxs)
        n_per_node = int(np.ceil(float(n_combos)/size))
        remainder = n_combos%size
        zero_remainder = remainder == 0

        sendbuf = np.zeros([size, n_per_node, 3], dtype = 'i')

        past_remainder_counter = 0
        # This could be cleaned up to be more readable
        # I believe this is still broken for weird node #'s
        # current hack is to make sure hte number of nodes evenly divides the number of HODs*cosmos*sfs
        #print size, n_per_node, remainder, n_combos

        for i in range(size):
            if remainder == 0:
                if not zero_remainder: # initially, after we used it as a counter
                    sendbuf[i, :-1, :] =  all_param_idxs[i*n_per_node - past_remainder_counter:(i+1)*n_per_node-past_remainder_counter-1, :]
                    sendbuf[i,-1, :] = np.array([-1,-1,-1])
                    past_remainder_counter+=1
                else: #no extra lines
                    sendbuf[i, :, :] =  all_param_idxs[i*n_per_node:(i+1)*n_per_node, :]

            else:
                sendbuf[i,:,:] = all_param_idxs[i*n_per_node:(i+1)*n_per_node, :]
                remainder-=1

            #print sendbuf[i]
        return sendbuf

    def compute_measurement(self, param_idxs, rank = None):

        param_idxs = param_idxs.astype(int)
        if self.n_bins > 1 :
            output = np.zeros((param_idxs.shape[0], self.n_bins))
            output_cov = np.zeros((param_idxs.shape[0], self.n_bins, self.n_bins))

        else:
            output = np.zeros((param_idxs.shape[0],))
            output_cov = np.zeros((param_idxs.shape[0]))

        last_cosmo_idx, last_scale_factor_idx = -1, -1
        t0 = time()
        for output_idx, (cosmo_idx, scale_factor_idx, hod_idx) in enumerate(param_idxs):
            if rank is not None:
                print('Rank: %d, Cosmo: %d, Scale_Factor: %d, HOD: %d'%(rank, cosmo_idx, scale_factor_idx, hod_idx))
            else:
                print('Cosmo: %d, Scale_Factor: %d, HOD: %d'%(cosmo_idx, scale_factor_idx, hod_idx))

            print('Time: %.2f'%(time()- t0))
            print('*'*30)

            if any(idx == -1 for idx in [cosmo_idx, scale_factor_idx, hod_idx]):
                continue # skip these placeholders
            if last_cosmo_idx != cosmo_idx or last_scale_factor_idx != scale_factor_idx:
                if last_cosmo_idx != -1:
                    # free some memory
                    last_cat = self.cats[last_cosmo_idx]
                    del last_cat.halocat
                    del last_cat.model

                cat = self.cats[cosmo_idx]

                scale_factor = self._scale_factors[scale_factor_idx]

                #print self._scale_factors, scale_factor_idx, scale_factor
                cat.load(scale_factor, HOD= self.hod, particles = self._particles,
                         downsample_factor=self._downsample_factor, hod_kwargs= self._hod_kwargs  )

                calc_observable = self._get_calc_observable(cat)

            hod_params = dict(list(zip(self._hod_param_names, self._hod_param_vals[hod_idx, :])))
            if self._fixed_nd is not None:
                self._add_logMmin(hod_params, cat)
            #continue
            if self.pop_seed is None:
                seed = int(time())
            else:
                seed = self.pop_seed

            np.random.seed(seed)
            if self._n_repops == 1:
                cat.populate(hod_params, min_ptcl = self._min_ptcl)
                # TODO this will fail if you don't jackknife when n_repops is 1
                obs_val, obs_cov = self._transform_func(calc_observable())
            else:  # do several repopulations
                if self.n_bins > 1:
                    obs_repops = np.zeros((self._n_repops, self.n_bins))
                else:
                    obs_repops = np.zeros((self._n_repops,))

                for repop in range(self._n_repops):
                    #print repop
                    cat.populate(hod_params, min_ptcl= self._min_ptcl)
                    try:
                        obs_repops[repop] = self._transform_func(calc_observable())
                    except ValueError: #likely an issue with population. If it comes up again, send it up
                        cat.populate(hod_params, min_ptcl= self._min_ptcl)
                        obs_repops[repop] = self._transform_func(calc_observable())

                obs_val = np.mean(obs_repops, axis=0)
                obs_cov = np.cov(obs_repops, rowvar=False)

            output[output_idx] = obs_val
            output_cov[output_idx] = obs_cov

            last_cosmo_idx = cosmo_idx
            last_scale_factor_idx = scale_factor_idx

        return output, output_cov

    def write_hdf5_file(self, output, output_cov):

        all_param_idxs = np.array(list(product(range(len(self.cats)), range(len(self._scale_factors)),
                                range(self._hod_param_vals.shape[0]))))

        if self.n_bins == 1:
            output = output.reshape((-1,))
            output_cov = output_cov.reshape((-1,))
        else:
            output = output.reshape((-1, self.n_bins))
            output_cov = output_cov.reshape((-1, self.n_bins, self.n_bins))

        all_cosmo_sf_pairs = np.array(list(product(range(len(self.cats)), range(len(self._scale_factors)))))

        f = h5py.File(self.output_fname, 'w')
        try:
            f.attrs['obs'] = self.obs
            f.attrs['min_ptcl'] = self._min_ptcl
            f.attrs['cosmo_param_names'] = self._cosmo_param_names
            f.attrs['cosmo_param_vals'] = self._cosmo_param_vals
            f.attrs['scale_factors'] = self._scale_factors
            f.attrs['hod_param_names'] = self._hod_param_names
            f.attrs['hod_param_vals'] = self._hod_param_vals
            f.attrs['scale_bins'] = self.scale_bins
        except RuntimeError:  # data too large
            attrs = f.create_group('attrs')

            f.attrs['obs'] = self.obs
            f.attrs['min_ptcl'] = self._min_ptcl
            f.attrs['cosmo_param_names'] = self._cosmo_param_names
            f.attrs['scale_factors'] = self._scale_factors
            f.attrs['hod_param_names'] = self._hod_param_names
            f.attrs['scale_bins'] = self.scale_bins

            attrs.create_dataset('cosmo_param_vals', data=self._cosmo_param_vals)
            attrs.create_dataset('hod_param_vals', data=self._hod_param_vals)

        for cosmo_sf_pair in all_cosmo_sf_pairs:
            group_name = self._get_group_name(*cosmo_sf_pair)
            grp = f.create_group(group_name)  # could rename the above to the group name

            # I could compute this, which would be faster, but this is easier to read.
            hod_idxs = np.where(np.all(all_param_idxs[:, :2] == cosmo_sf_pair, axis=1))[0]

            grp.create_dataset("obs", data=output[hod_idxs], chunks=True, compression='gzip')
            grp.create_dataset("cov", data=output_cov[hod_idxs], chunks=True, compression='gzip')

        f.close()

    def run(self, comm):
        """
        #TODO
        """

        rank = comm.Get_rank()
        size = comm.Get_size()

        n_combos = len(self.cats)*len(self._scale_factors)*len(self._hod_param_vals)
        n_per_node = np.ceil(float(n_combos) / size)

        if rank == 0:
            all_param_idxs_send = self._divide_tasks(size)
        else:
            all_param_idxs_send = None

        param_idxs = comm.scatter(all_param_idxs_send, root=0)

        output, output_cov = self.compute_measurement(param_idxs, rank)

        # TODO, someday, glorious parallel hdf5.
        # the "gather" should at max collect 16mb, so it may not be sooo bad.
        #f = h5py.File(self.output_fname, 'w', driver='mpio', comm=comm)

        all_output = None
        all_output_cov = None
        if rank == 0:
            if self.n_bins == 1:
                all_output = np.empty([size, n_per_node], dtype=output.dtype)
                all_output_cov = np.empty([size, n_per_node], dtype = output_cov.dtype)
            else:
                all_output = np.empty([size, n_per_node, self.n_bins], dtype=output.dtype)
                all_output_cov = np.empty([size, n_per_node, self.n_bins, self.n_bins], output_cov.dtype)
        
        comm.Gather(output, all_output, root=0)
        comm.Gather(output_cov, all_output_cov, root = 0)


        if rank == 0:
            self.write_hdf5_file(all_output, all_output_cov)

    def queue_skipper(self, make_command, config_fname, rerun = False):
        """
        A dark twisted thing. The standard run command does the "right" thing, creating a large MPI job
        that does all the training in parallel. However, this can frequently get stuck in the queue of clusters.
        This function creates a series of small jobs that run indepently, which is usually much faster to run,
        though less ethically.
        :param make_command:
            A function that takes a jobname, a maximum time (in hours) and an output directory,
            and returns a tring that executes a submission command. This will be called
            to send jobs to the queue.
        :param config_fname:
            The filename of the config file for the trainer.
        :return:
        """

        output_directory = path.dirname(self.output_fname)
        # Only have to write HOD info. Rest is uniquely specified by the config.
        if not rerun:
            np.savetxt(path.join(output_directory, HOD_FNAME), self._hod_param_vals, header = '\t'.join(self._hod_param_names))
            copyfile(config_fname, path.join(output_directory, CONFIG_FNAME))

            #split into a unique job per scale factor and cosmology
            all_param_idxs = self._divide_tasks(self.n_jobs)

        else:
            assert self.n_jobs == len(glob(path.join(output_directory, 'trainer_*.npy'))), 'n_jobs has changed, cannot rerun'


        for idx in range(self.n_jobs):

            # slice out a portion of the poitns
            jobname = 'trainer_%04d' %idx
            param_filename = path.join(output_directory, jobname + '.npy')
            if not rerun:
                np.savetxt(param_filename, all_param_idxs[idx])
            elif path.exists(path.join(output_directory, 'output_%04d.npy'%idx)):\
                continue # this one ran successfull

            # TODO allow queue changing
            command = make_command(jobname, self.max_time, output_directory)
            # the odd shell call is to deal with minute differences in the systems.
            # TODO make this more general
            call(command, shell=self.system == 'sherlock')


def make_kils_command(jobname, max_time, outputdir, queue= 'bulletmpi'):
    '''
    Return a list of strings that comprise a bash command to call trainingHelper.py on the cluster.
    Designed to work on ki-ls's batch system
    :param jobname:
        Name of the job. Will also be used to make the parameter file and log file.
    :param max_time:
        Time for the job to run, in hours.
    :param outputdir:
        Directory to store output and param files.
    :param queue:
        Optional. Which queue to submit the job to.
    :return:
        Command, a list of strings that can be ' '.join'd to form a bash command.
    '''
    log_file = jobname + '.out'
    param_file = jobname + '.npy'
    command = ['bsub',
               '-q', queue,
               '-n', str(8),
               '-J', jobname,
               '-oo', path.join(outputdir, log_file),
               '-W', '%d:00' % max_time,
               #'-R span[ptile=8]',
               #'--exclusive',
               'python', path.join(path.dirname(__file__), 'trainingHelper.py'),
               path.join(outputdir, param_file)]

    return command


def make_sherlock_command(jobname, max_time, outputdir, queue=None):
    '''
    Return a list of strings that comprise a bash command to call trainingHelper.py on the cluster.
    Designed to work on sherlock's sbatch system. Differnet from the above in that it must write a file
    to disk in order to work. Still returns a callable script.
    :param jobname:
        Name of the job. Will also be used to make the parameter file and log file.
    :param max_time:
        Time for the job to run, in hours.
    :param outputdir:
        Directory to store output and param files.
    :param queue:
        Optional. Which queue to submit the job to.
    :return:
        Command, a string to call to submit the job.
    '''
    log_file = jobname + '.out'
    err_file = jobname + '.err'
    param_file = jobname + '.npy'
    sbatch_header = ['#!/bin/bash',
                     '--job-name=%s' % jobname,
                     '-p iric',  # KIPAC queue
                     '--output=%s' % path.join(outputdir, log_file),
                     '--error=%s' % path.join(outputdir, err_file),
                     '--time=%d:00' % (max_time * 60),  # max_time is in minutes
                     '--ntasks=1',
                     '--cpus-per-task=8',
                     '--mem-per-cpu=MaxMemPerCPU']
                     #'--qos=normal',
                     #'--nodes=%d' % 1,
                     #'--exclusive']#,
                     #'--mem-per-cpu=32000',
                     #'--cpus-per-task=%d' % 16]

    sbatch_header = '\n#SBATCH '.join(sbatch_header)

    call_str = ['python', path.join(path.dirname(__file__), 'trainingHelper.py'),
                path.join(outputdir, param_file)]

    call_str = ' '.join(call_str)
    # have to write to file in order to work.
    with open(path.join(outputdir, 'tmp.sbatch'), 'w') as f:
        f.write(sbatch_header + '\n' + call_str)

    return 'sbatch %s' % (path.join(outputdir, 'tmp.sbatch'))

HOD_FNAME = 'HOD_params.npy'
CONFIG_FNAME = 'config.yaml'


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train an emulator from a config file.')
    parser.add_argument('config_fname', type=str, help='Config YAML File')
    parser.add_argument('--rerun', action='store_true')
    args = vars(parser.parse_args())

    config_fname = args['config_fname'] # could implement full argparse if i want i guess

    trainer = Trainer(config_fname)

    rerun = args['rerun']
    if trainer._skip_queue:
        # unclear if having this here or elsewhere is better.
        # My thinking currently is that this functionality enables other systems more transparently.
        if trainer.system == 'sherlock':
            make_command = make_sherlock_command
        else:
            make_command = make_kils_command

        trainer.queue_skipper(make_command, config_fname, rerun)
    else:

        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        trainer.run(comm)


