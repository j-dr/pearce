#!/bin/bash
'''This file samples points in the parameter space, and sends off jobs to perform the calculation of xi at those
points in parameter space. '''

from time import time
from os import path
from subprocess import call
from itertools import izip
from collections import namedtuple
import numpy as np

parameter = namedtuple('Parameter', ['name', 'low','high'])

#global object that defines both the names and ordering of the parameters, as well as their boundaries.
#TODO consider moving?
#TODO reading in bounds/params from config
PARAMS = [ parameter('logMmin', 11.7,12.5),
           parameter('sigma_logM', 0.2, 0.7),
           parameter('logM0', 10,13),
           parameter('logM1', 13.1, 14.3),
           parameter('alpha', 0.75, 1.25),
           parameter('f_c', 0.1, 0.5)]
#I think that it's better to have this param global, as it prevents there from being any conflicts.

#send commands to cluster

def makeLHC(N=500):
    '''Return a vector of points in parameter space that defines a latin hypercube.
    :param N:
        Number of points per dimension in the hypercube. Default is 500.
    :return
        A latin hyper cube sample in HOD space in a numpy array.
    '''
    np.random.seed(int(time()))

    #this is a bad name...
    points = []
    #by linspacing each parameter and shuffling, I ensure there is only one point in each row, in each dimension.
    for p in PARAMS:
        point = np.linspace(p.low, p.high, num=N)
        np.random.shuffle(point)#makes the cube random.
        points.append(point)
    return np.stack(points).T

def makeFHC(N=4):
    '''
    Return a vector of points in parameter space that defines a afull hyper cube.
    :param N:
        Number of points per dimension. Can be an integer or list. If it's a number, it will be the same
        across each dimension. If a list, defines points per dimension in the same ordering as PARAMS.
    :return:
        A full hyper cube sample in HOD space in a numpy array.
    '''

    if type(N) is int:
        N = [N for i in xrange(len(PARAMS))]

    assert type(N) is list

    n_total = np.prod(N)
    # TODO check if n_total is 1.
    points = np.zeros((n_total, len(PARAMS)))
    n_segment = n_total #could use the same variable, but this is clearer
    for i, (n, param) in enumerate(izip(N, PARAMS)):
        values = np.linspace(param.low, param.high, n)
        n_segment/=n
        for j, p in enumerate(points):
            idx = (j / n_segment) %n
            p[i] = values[idx]

    #shuffle to even out computation times
    np.random.seed(int(time()))
    idxs = np.random.permutation(n_total)
    return points[idxs, :]
    #return points

#cosmo params required:
#simname
#Lbox, npart
#redshift/scale_factor
#system

def make_kils_command(jobname,max_time,outputdir,global_filename=None, queue='bulletmpi'):
    '''
    Return a list of strings that comprise a bash command to call trainingHelper.py on the cluster.
    Designed to work on ki-ls's batch system
    :param jobname:
        Name of the job. Will also be used to make the parameter file and log file.
    :param max_time:
        Time for the job to run, in hours.
    :param outputdir:
        Directory to store output and param files.
    :param global_filename:
        Optional. The filename where the global parameters are stored. If left out, the default will be use.
    :param queue:
        Optional. Which queue to submit the job to.
    :return:
        Command, a list of strings that can be ' '.join'd to form a bash command.
    '''
    log_file = jobname + '.out'
    param_file = jobname+ '.npy'
    command = ['bsub',
               '-q', queue,
               '-n', str(16),
               '-J', jobname,
               '-oo', path.join(outputdir, log_file),
               '-W', '%d:00' % max_time,
               'python', path.join(path.dirname(__file__), 'trainingHelper.py'),
               param_file]
    if global_filename is not None:
        command.append(path.join(outputdir, global_filename))

    return command

def make_sherlock_command(jobname, max_time, outputdir, global_filename=None,queue=None):
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
    :param global_filename:
        Optional. The filename where the global parameters are stored. If left out, the default will be use.
    :param queue:
        Optional. Which queue to submit the job to.
    :return:
        Command, a string to call to submit the job.
    '''
    log_file = jobname + '.out'
    err_file = jobname + '.err'
    param_file = jobname+ '.npy'

    sbatch_header = ['#!/bin/bash',
                     '--job-name=%s' % jobname,
                     '-p iric',  # KIPAC queue
                     '--output=%s' % path.join(outputdir, log_file),
                     '--error=%s' % path.join(outputdir, err_file),
                     '--time=%d:00' % (max_time * 60), #max_time is in minutes
                     '--qos=normal',
                     '--nodes=%d' % 1,
                     # '--exclusive',
                     '--mem-per-cpu=32000',
                     '--ntasks-per-node=%d' % 1,
                     '--cpus-per-task=%d' % 16]

    sbatch_header = '\n#SBATCH '.join(sbatch_header)

    call_str = ['python', path.join(path.dirname(__file__), 'trainingData.py'),
                param_file]

    if global_filename is not None:
        call_str.append(path.join(outputdir, global_filename))

    call_str = ' '.join(call_str)
    #have to write to file in order to work.
    with open(path.join(outputdir, 'tmp.sbatch'), 'w') as f:
        f.write(sbatch_header + '\n' + call_str)

    return 'sbatch %s'%(path.join(outputdir, 'tmp.sbatch'))

#make_traing_data parameters
#method, n_points
#system, n_jobs
#rbins, cosmology info
#outputdir, max_time

#Could use ConfigParser maybe
#TODO move this to a different, helper folder
def config_reader(filename):
    '''
    General helper module. Turns a file of key:value pairs into a dictionary.
    :param filename:
        Config file.
    :return:
        A dictionary of key-value pairs.
    '''
    config = {}
    with open(filename) as f:
        for line in f:
            line = line.strip()
            splitline = line.split(':')
            config[splitline[0]] = config[splitline[1]]

    return config
#TODO not sure I like this, look again tomorrow
def training_config_reader(filename):
    '''
    Reads specific details of the config file for this usage.
    :param filename:
        Config file
    :return:
        method, n_points, system, n_jobs, max_time, outputdir, rbins, cosmo_params
        Config parameters defined explicitly elsewhere.
    '''
    config = config_reader(filename)
    #I could make some of these have defaults
    #I'm not sure I want to do that.
    try:
        method = config['method']
        n_points = int(config['n_points'])
        system = config['system']
        n_jobs = config['n_jobs']
        max_time = int(config['max_time'])
        outputdir = config['outputdir']

        rbins_str = config['rbins']
        #need to do a little work to get this right
        rbins = [float(r) for r in rbins_str.strip('[ ]').split(',')]

        #cosmology information assumed to be in the remaining ones!
        for key in ['method', 'n_points', 'system', 'n_jobs', 'max_time',
                    'outputdir','rbins']:
            del config[key]

        cosmo_params = config

        #check simname is in there!
        #if fails, will throw a KeyError
        cosmo_params['simname']
        cosmo_params['scale_factor']

    except KeyError:
        raise KeyError("The config file %s is missing a parameter."%filename)

    return method, n_points, system, n_jobs, max_time, outputdir, rbins, cosmo_params

def make_training_data(config_filename):
    '''
    "Main" function. Take a config file as input and send off jobs to compute xi
    at various points in HOD parameter space.
    :param config_filename:
        Config file.
    :return:
        None.
    '''

    method, n_points, system, n_jobs, max_time, outputdir, rbins, cosmo_params = \
    training_config_reader(config_filename)

    #determine the specific functions needed for this setup
    if method == 'LHC':
        points = makeLHC(n_points)
    elif method == 'FHC':
        points = makeFHC(n_points)
    else:
        raise ValueError('Invalid method for making training data: %s'%method)

    if system == 'ki-ls':
        make_command = make_kils_command
    elif system == 'sherlock':
        make_command = make_sherlock_command
    else:
        raise ValueError('Invalid system for making training data: %s'%system)

    #write the global file used by all params
    header_start = ['Cosmology Params:']
    header_start.extend('%s:%.3f' % (key, val) for key, val in cosmo_params.iteritems())
    header = '\n'.join(header_start)
    #default name.
    np.savetxt(path.join(outputdir, 'global_file.npy'), rbins)

    #call each job individually
    points_per_job = int(points.shape[0]/n_jobs)
    for job in xrange(n_jobs):
        #slice out a portion of the poitns
        job_points = points[job*points_per_job:job*(points_per_job+1), :]
        jobname = 'training_data%03d'%job
        param_filename = path.join(outputdir, jobname+'.npy' )
        np.savetxt(param_filename, job_points)

        #TODO allow queue
        command = make_command(jobname, max_time, outputdir)
        #the odd shell call is to deal with minute differences in the systems.
        call(command, shell = system=='sherlock')


