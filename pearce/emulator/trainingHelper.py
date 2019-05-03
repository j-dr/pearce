"""
Helper function that takes care of the training stuff for the queue skipper functionality
"""
import sys
sys.path.append('..')
from pearce.emulator.trainer import *

from os import remove
from glob import glob

def get_trainer(directory):
    """
    Short helper function to get the trainer from the info passed as an arguement
    :param directory:
        The directory where all the configs were written to
    :return:
        A Trainer object with the correct HOD hypercube
    """
    hod_fname = path.join(directory, HOD_FNAME)
    config_fname = path.join(directory, CONFIG_FNAME)

    trainer = Trainer(config_fname)
    # now all trainers in each job will have the same hypercube
    trainer._hod_param_vals = np.loadtxt(hod_fname)

    return trainer

def compute_on_subset(param_fname):
    """
    Computes the training job on a subset of the training set, as given by the idxs in param_fname
    :param param_fname:
    :return:
    """
    job_number = int(path.basename(param_fname).split('.')[0][-4:])
    output_directory = path.dirname(param_fname)
    print(job_number)
    trainer = get_trainer(output_directory)

    param_idxs = np.loadtxt(param_fname)
    output, output_cov = trainer.compute_measurement(param_idxs)

    np.save(path.join(output_directory, 'output_%04d.npy'%job_number), output)
    np.save(path.join(output_directory, 'output_cov_%04d.npy'%job_number), output_cov)


def consolidate_outputs(directory):
    """
    Take outputs from compute_on_subet and write them to one hdf5 file.
    :param directory:
        The directory with the outputs in them
    """
    all_output_fnames = sorted(glob(path.join(directory, 'output_*') ))
    output_fnames, output_cov_fnames = [], []

    for fname in all_output_fnames:
        if 'cov' in fname:
            output_cov_fnames.append(fname)
        else:
            output_fnames.append(fname)

    all_output, all_output_cov = [], []
    # i'd like to find a way to make the numpy arrays a priori but not sure how
    for o_fname, cov_fname in zip(output_fnames, output_cov_fnames):
        all_output.append(np.load(o_fname))
        all_output_cov.append(np.load(cov_fname))

    all_output = np.array(all_output)
    all_output_cov = np.array(all_output_cov)

    trainer = get_trainer(directory)

    trainer.write_hdf5_file(all_output, all_output_cov)

    #clean up
    for fname in all_output_fnames:
        pass
        #remove(fname)

    #remove(path.join(directory, HOD_FNAME))
    #remove(path.join(directory, CONFIG_FNAME))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Helper function called as main from "trainingData.py." Reads in \
                                                collections of HOD params and calculates their correlation functions.')
    parser.add_argument('param_fname', type = str, help='File where the vector of HOD params are stored.')
    args = vars(parser.parse_args())
    param_fname = args['param_fname']
    print(param_fname)

    compute_on_subset(param_fname)

    #would like a way to call consolidation after they've all finished, not sure how though

