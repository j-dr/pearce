"""
I'm modifying the halotools tpcf jackknfie code to return the subregion calculations so I can more
directly manipulate them.
"""


import numpy as np
from astropy.utils.misc import NumpyRNGContext
from halotools.mock_observables import *
from halotools.custom_exceptions import HalotoolsError
from halotools.mock_observables.mock_observables_helpers import *
from halotools.mock_observables.two_point_clustering import *
from halotools.mock_observables.two_point_clustering.clustering_helpers import *
from halotools.mock_observables.pair_counters.mesh_helpers import _enforce_maximum_search_length
from halotools.mock_observables.two_point_clustering.tpcf_estimators import _TP_estimator, _TP_estimator_requirements
from halotools.mock_observables.pair_counters import npairs_jackknife_3d

from halotools.mock_observables.two_point_clustering.tpcf_jackknife import \
    _tpcf_jackknife_process_args,_enclose_in_box, get_subvolume_numbers, jrandom_counts

np.seterr(divide='ignore', invalid='ignore')  # ignore divide by zero in e.g. DD/RR


__all__ = ['tpcf_subregions']

# all lifted from duncan's code
def tpcf_subregions(sample1, randoms, rbins, Nsub=[5, 5, 5],
        sample2=None, period=None, do_auto1=True, do_auto2=False, do_cross=True,
                estimator='Natural', num_threads=1, seed=None, RR=None):

    do_auto = do_auto1 or do_auto2
# process input parameters
    function_args = (sample1, randoms, rbins, Nsub, sample2, period, do_auto,
        do_cross, estimator, num_threads, seed)
    sample1, rbins, Nsub, sample2, randoms, period, do_auto, do_cross, num_threads,\
        _sample1_is_sample2, PBCs = _tpcf_jackknife_process_args(*function_args)

# determine box size the data occupies.
# This is used in determining jackknife samples.
    if PBCs is False:
        sample1, sample2, randoms, Lbox = _enclose_in_box(sample1, sample2, randoms)
    else:
        Lbox = period


    do_DD, do_DR, do_RR = _TP_estimator_requirements(estimator)

    N1 = len(sample1)
    N2 = len(sample2)
    NR = len(randoms)


    j_index_1, N_sub_vol = cuboid_subvolume_labels(sample1, Nsub, Lbox)
    j_index_2, N_sub_vol = cuboid_subvolume_labels(sample2, Nsub, Lbox)
    j_index_random, N_sub_vol = cuboid_subvolume_labels(randoms, Nsub, Lbox)

# number of points in each subvolume
    NR_subs = get_subvolume_numbers(j_index_random, N_sub_vol)
    N1_subs = get_subvolume_numbers(j_index_1, N_sub_vol)
    N2_subs = get_subvolume_numbers(j_index_2, N_sub_vol)
# number of points in each jackknife sample
    N1_subs = N1 - N1_subs
    N2_subs = N2 - N2_subs
    NR_subs = NR - NR_subs


# calculate all the pair counts
# TODO need to modify this function
    D1D1, D1D2, D2D2 = jnpair_counts(
        sample1, sample2, j_index_1, j_index_2, N_sub_vol,
            rbins, period, num_threads, do_auto1, do_cross, do_auto2, _sample1_is_sample2)
# pull out the full and sub sample results

    if _sample1_is_sample2:
        D1D1_full = D1D1[0, :]
        D1D1_sub = D1D1[1:, :]
        D1D2_full = D1D2[0, :]
        D1D2_sub = D1D2[1:, :]
        D2D2_full = D2D2[0, :]
        D2D2_sub = D2D2[1:, :]

    else:
        if do_auto1:
            D1D1_full = D1D1[0, :]
            D1D1_sub = D1D1[1:, :]
        if do_cross:
            D1D2_full = D1D2[0, :]
            D1D2_sub = D1D2[1:, :]
        if do_auto2:
            D2D2_full = D2D2[0, :]
            D2D2_sub = D2D2[1:, :]

# do random counts
# TODO figure out what of this i can skip?  

    if RR is None:
        D1R, RR = jrandom_counts(sample1, randoms, j_index_1, j_index_random, N_sub_vol,
            rbins, period, 1, do_DR, do_RR)
    else: #use the precomputed RR
        D1R, RR_dummy= jrandom_counts(sample1, randoms, j_index_1, j_index_random, N_sub_vol,
            rbins, period, 1, do_DR, do_RR=False)

    print('A')
    if _sample1_is_sample2:
        D2R = D1R
    else:
        if do_DR is True:
            D2R, RR_dummy = jrandom_counts(sample2, randoms, j_index_2, j_index_random,
                    N_sub_vol, rbins, period, num_threads, do_DR, do_RR=False)
        else:
            D2R = None
    print('B')
    if do_DR is True:
        D1R_full = D1R[0, :]
        D1R_sub = D1R[1:, :]
        D2R_full = D2R[0, :]
        D2R_sub = D2R[1:, :]
    else:
        D1R_full = None
        D1R_sub = None
        D2R_full = None
        D2R_sub = None
    if do_RR is True:
        RR_full = RR[0, :]
        RR_sub = RR[1:, :]
    else:
        RR_full = None
        RR_sub = None
# calculate the correlation function for the subsamples
    outputs = []
    print('C')

    if do_auto1 or _sample1_is_sample2:
        xi_11_sub = _TP_estimator(D1D1_sub, D1R_sub, RR_sub, N1_subs, N1_subs, NR_subs, NR_subs, estimator)
        outputs.append(xi_11_sub)
    if do_cross:
        xi_12_sub = _TP_estimator(D1D2_sub, D1R_sub, RR_sub, N1_subs, N2_subs, NR_subs, NR_subs, estimator)
        outputs.append(xi_12_sub)
    if do_auto2:
        xi_22_sub = _TP_estimator(D2D2_sub, D2R_sub, RR_sub, N2_subs, N2_subs, NR_subs, NR_subs, estimator)
        outputs.append(xi_22_sub)
    return outputs[0] if len(outputs) ==1 else tuple(outputs)

# overload to skip the xi_mm calculation
def jnpair_counts(sample1, sample2, j_index_1, j_index_2, N_sub_vol, rbins,
        period, num_threads, do_auto1 = True, do_cross=False,do_auto2=False, _sample1_is_sample2=False):
    """
    Count jackknife data pairs: DD
    """
    if do_auto1 is True:
        D1D1 = npairs_jackknife_3d(sample1, sample1, rbins, period=period,
            jtags1=j_index_1, jtags2=j_index_1,  N_samples=N_sub_vol,
            num_threads=num_threads)
        D1D1 = np.diff(D1D1, axis=1)
    else:
        D1D1 = None
        D2D2 = None

    if _sample1_is_sample2:
        D1D2 = D1D1
        D2D2 = D1D1
    else:
        if do_cross is True:
            D1D2 = npairs_jackknife_3d(sample1, sample2, rbins, period=period,
                jtags1=j_index_1, jtags2=j_index_2,
                N_samples=N_sub_vol, num_threads=num_threads)
            D1D2 = np.diff(D1D2, axis=1)
        else:
            D1D2 = None
        if do_auto2 is True:
            D2D2 = npairs_jackknife_3d(sample2, sample2, rbins, period=period,
                jtags1=j_index_2, jtags2=j_index_2,
                N_samples=N_sub_vol, num_threads=num_threads)
            D2D2 = np.diff(D2D2, axis=1)
        else:
            D2D2 = None

    return D1D1, D1D2, D2D2

