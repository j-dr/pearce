#!/bin/bash
'''This file samples points in the parameter space, and sends off jobs to perform the calculation of an observable at those
points in parameter space. '''

from time import time
from os import path, mkdir
from subprocess import call
from itertools import izip
import warnings
import cPickle as pickle

import numpy as np

from .ioHelpers import config_reader
from ..mocks import cat_dict

# I initially had the global_filename be variable. Howerver, I couldn't find a reason one would change it!
GLOBAL_FILENAME = 'global_file.npy'
PARAMS_FILENAME = 'params.pkl'


# I think that it's better to have this param global, as it prevents there from being any conflicts.
def makeLHC(ordered_params, N=500):
    '''Return a vector of points in parameter space that defines a latin hypercube.
    :param ordered_params:
        list of "parameter" named tuple objects that define the ordering, name, and ranges of parameters
        used in the trianing data.
    :param N:
        Number of points per dimension in the hypercube. Default is 500.
    :return
        A latin hyper cube sample in HOD space in a numpy array.
    '''
    np.random.seed(int(time()))

    points = []
    # by linspacing each parameter and shuffling, I ensure there is only one point in each row, in each dimension.
    for p in ordered_params:
        point = np.linspace(p.low, p.high, num=N)
        np.random.shuffle(point)  # makes the cube random.
        points.append(point)
    return np.stack(points).T


def makeFHC(ordered_params, N=4):
    '''
    Return a vector of points in parameter space that defines a afull hyper cube.
    :param ordered_params:
        list of "parameter" named tuple objects that define the ordering, name, and ranges of parameters
        used in the trianing data.
    :param N:
        Number of points per dimension. Can be an integer or list. If it's a number, it will be the same
        across each dimension. If a list, defines points per dimension in the same ordering as ordered_params.
    :return:
        A full hyper cube sample in HOD space in a numpy array.
    '''

    if type(N) is int:
        N = [N for i in xrange(len(ordered_params))]

    assert type(N) is list

    n_total = np.prod(N)
    # TODO check if n_total is 1.

    grid_points = np.meshgrid(*[np.linspace(param.low, param.high, n) \
                                for n, param in izip(N, ordered_params)])
    points = np.stack(grid_points).T
    points = points.reshape((-1, len(ordered_params)))

    # Not sure the change i've made is right yet.
    # points = np.zeros((n_total, len(ordered_params)))
    # n_segment = n_total  # could use the same variable, but this is clearer
    # # For each param, assign the values such that it fills out the cube.
    # for i, (n, param) in enumerate(izip(N, ordered_params)):
    #     values = np.linspace(param.low, param.high, n)
    #     n_segment /= n
    #     for j, p in enumerate(points):
    #         idx = (j / n_segment) % n
    #         p[i] = values[idx]


    # shuffle to even out computation times
    np.random.seed(int(time()))
    idxs = np.random.permutation(n_total)
    return points[idxs, :]
    # return points


def make_kils_command(jobname, max_time, outputdir, queue='kipac-ibq'):  # 'bulletmpi'):
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
               '-n', str(16),
               '-J', jobname,
               '-oo', path.join(outputdir, log_file),
               '-W', '%d:00' % max_time,
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
                     '--qos=normal',
                     '--nodes=%d' % 1,
                     # '--exclusive',
                     '--mem-per-cpu=32000',
                     '--ntasks-per-node=%d' % 1,
                     '--cpus-per-task=%d' % 16]

    sbatch_header = '\n#SBATCH '.join(sbatch_header)

    call_str = ['python', path.join(path.dirname(__file__), 'trainingHelper.py'),
                path.join(outputdir, param_file)]

    call_str = ' '.join(call_str)
    # have to write to file in order to work.
    with open(path.join(outputdir, 'tmp.sbatch'), 'w') as f:
        f.write(sbatch_header + '\n' + call_str)

    return 'sbatch %s' % (path.join(outputdir, 'tmp.sbatch'))


def training_config_reader(filename):
    '''
    Reads specific details of the config file for this usage.
    :param filename:
        Config filename
    :return:
        method,obs n_points, system, n_jobs, max_time, outputdir, bins, cosmo_params
        Config parameters defined explicitly elsewhere.
    '''
    config = config_reader(filename)
    # I could make some of these have defaults with get()
    # I'm not sure I want to do that.
    try:
        method = config['method'].strip()
        obs = config['obs'].strip()
        n_points = int(config['n_points'])
        system = config['system']
        n_jobs = int(config['n_jobs'])
        max_time = int(config['max_time'])
        outputdir = config['outputdir'].strip()
        bins_str = config['bins']
        # need to do a little work to get this right
        bins = [float(r.strip()) for r in bins_str.strip('[ ]').split(',')]

        # cosmology information assumed to be in the remaining ones!
        # Delete the ones we've removed.
        for key in ['method', 'obs', 'n_points', 'system', 'n_jobs', 'max_time',
                    'outputdir', 'bins']:
            del config[key]

        cosmo_params = config

        # check simname and scale_factor (the 100% required ones) are in there!
        # if fails, will throw a KeyError
        cosmo_params['simname']
        # TODO change to scale factors?
        if '[' in cosmo_params['scale_factor']:  # user passed in a list
            sf_str = cosmo_params['scale_factor']
            cosmo_params['scale_factor'] = [float(a.strip()) for a in sf_str.strip('[ ]').split(',')]
        elif cosmo_params['scale_factor'] == 'all':
            cosmo_params['scale_factor'] = [cosmo_params['scale_factor'].strip()]
        else:  # float
            cosmo_params['scale_factor'] = float(cosmo_params['scale_factor'].strip())

        for cp, t in zip(['Lbox', 'npart', 'n_repops'], [float, int, int]):
            try:
                cosmo_params[cp] = t(cosmo_params[cp])
            except KeyError:
                continue

    except KeyError:
        raise KeyError("The config file %s is missing a parameter." % filename)

    return method, obs, n_points, system, n_jobs, max_time, outputdir, bins, cosmo_params


def make_training_data(config_filename, ordered_params=None):
    '''
    "Main" function. Take a config file as input and send off jobs to compute an observable
    at various points in HOD parameter space.
    :param config_filename:
        Config file.
    :return:
        None.
    '''

    method, obs, n_points, system, n_jobs, max_time, base_outputdir, bins, cosmo_params = \
        training_config_reader(config_filename)

    scale_factors = cosmo_params['scale_factor']

    # load one up to test that these scale factors maek sense.
    cat = cat_dict[cosmo_params['simname']](**cosmo_params)

    if scale_factors[0] == 'all':
        # have to load up a cat to get them
        scale_factors = cat.scale_factors
    else:  # a list
        assert all(min(a - cat.scale_factors) < 0.05 for a in scale_factors)

    if ordered_params is None:
        # raise warning?
        from .ioHelpers import DEFAULT_PARAMS as ordered_params

    # determine the specific functions needed for this setup
    # same points for each redshift
    if method == 'LHC':
        points = makeLHC(ordered_params, n_points)
    elif method == 'FHC':
        points = makeFHC(ordered_params, n_points)
    else:
        raise ValueError('Invalid method for making training data: %s' % method)

    if system == 'ki-ls':
        make_command = make_kils_command
    elif system == 'sherlock':
        make_command = make_sherlock_command
    else:
        raise ValueError('Invalid system for making training data: %s' % system)

    n_jobs_per_a = int(np.ceil(float(n_jobs) / len(scale_factors)))

    for a in scale_factors:

        cosmo_params['scale_factor'] = a  # store to pass in
        outputdir = path.join(base_outputdir, 'a_%.5f' % a)
        if not path.exists(outputdir):
            mkdir(outputdir)

        if ordered_params is None:
            from .ioHelpers import DEFAULT_PARAMS as ordered_params
            warnings.warn("Using default ordered parameters.")

        # write the global file used by all params
        # TODO Write system (maybe) and method (definetly) to file!
        header_start = ['Sampling Method: %s' % method, 'Observable: %s' % obs, 'Cosmology Params:']
        header_start.extend('%s:%s' % (key, str(val)) for key, val in cosmo_params.iteritems())
        header = '\n'.join(header_start)
        np.savetxt(path.join(outputdir, GLOBAL_FILENAME), bins, header=header)
        # Writing ordering to file.
        with open(path.join(outputdir, PARAMS_FILENAME), 'w') as f:
            pickle.dump(ordered_params, f)

        # call each job individually
        points_per_job = int(points.shape[0] / n_jobs_per_a)
        for job in xrange(n_jobs_per_a):
            # slice out a portion of the poitns
            if job == n_jobs_per_a - 1:  # last one, make sure we get the remainder
                job_points = points[job * points_per_job:, :]
            else:
                job_points = points[job * points_per_job:(job + 1) * points_per_job, :]
            jobname = 'training_data_a%.3f_%03d' % (a, job)
            param_filename = path.join(outputdir, jobname + '.npy')
            np.savetxt(param_filename, job_points)

            # TODO allow queue changing
            command = make_command(jobname, max_time, outputdir)
            # the odd shell call is to deal with minute differences in the systems.
            call(command, shell=system == 'sherlock')
