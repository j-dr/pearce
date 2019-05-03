r"""
This module contains the `~halotools.empirical_models.CorrelationAssembias` class.
This module implements an assembly bias model originally designed by Yao Yuan Mao.
It works by imposing a correlated noisy percentile on the secondary parameter and occupation.
Details can be found in` McLaughlin et al 2018 (in prep)`_.
"""

from functools import wraps
import numpy as np
from scipy.stats import poisson, bernoulli, rankdata

from halotools.empirical_models.assembias_models import HeavisideAssembias
from halotools.custom_exceptions import HalotoolsError
from halotools.utils.array_utils import custom_len
from .table_utils import compute_conditional_percentiles, compute_conditional_decorator
from halotools.empirical_models.abunmatch.noisy_percentile import noisy_percentile

__all__ = ('CorrelationAssembias',)
__author__ = ('Sean McLaughlin', )

@compute_conditional_decorator
def compute_conditional_shuffled_ranks(indices_of_prim_haloprop_bin, sec_haloprop, correlation_coeff, **kwargs):
    '''
    TODO Docs
    '''
    if sec_haloprop is None:
        msg = ("\n``sec_haloprop`` must be passed into compute_conditional_shuffled_ranks, or a table"
                       "with ``sec_haloprop_key`` as a column.\n")
        raise HalotoolsError(msg)

    try:
        assert np.all(np.logical_and(-1 <= correlation_coeff, correlation_coeff<= 1))
    except AssertionError:
        msg = ("\n``correlation_coeff`` must be passed into compute_conditional_percentiles,"
                       "and must be between -1 and 1\n")
        raise HalotoolsError(msg)


    num_in_bin = len(indices_of_prim_haloprop_bin)
    original_ranks = rankdata(sec_haloprop[indices_of_prim_haloprop_bin], 'ordinal') - 0.5
    original_ranks /= num_in_bin

    return noisy_percentile(original_ranks, correlation_coeff=correlation_coeff)


class CorrelationAssembias(HeavisideAssembias):
    """
    Class used to extend the behavior of `HeavisideAssembias` for continuous distributions.
    """
    def _get_assembias_param_dict_key(self, ipar):
        """
        """
        # changing the naem to make emulation more straightforward
        return self._method_name_to_decorate + '_' + self.gal_type + '_assembias_corr' + str(ipar+1)

    def _galprop_perturbation(self, **kwargs):
        r"""
        Has no feature in this method, purposely throw error to make sure nothing misleading is happening.

        Note: Could tehcnically "define" a perturbation, but would just involve calling the decorator and
        subtracting out the mean. Kind of backwards, so seems unecessary. 
        """
        raise NotImplementedError("_galprop_perturbation has does not have meaning in the correlation assembias method.")

    def assembias_decorator(self, func):
        r""" Primary behavior of the `CorrelationAssembias` class.
        This method is used to introduce a boost/decrement of the baseline
        function in a manner that preserves the all-halo result.
        Any function with a semi-bounded range can be decorated with
        `assembias_decorator`. The baseline behavior can be anything
        whatsoever, such as mean star formation rate or
        mean halo occupation, provided it has a semi-bounded range.

        Parameters
        -----------
        func : function object
            Baseline function whose behavior is being decorated with assembly bias.

        Returns
        -------
        wrapper : function object
            Decorated function that includes assembly bias effects.
        """
        lower_bound_key = 'lower_bound_' + self._method_name_to_decorate + '_' + self.gal_type
        baseline_lower_bound = getattr(self, lower_bound_key)
        upper_bound_key = 'upper_bound_' + self._method_name_to_decorate + '_' + self.gal_type
        baseline_upper_bound = getattr(self, upper_bound_key)

        @wraps(func)
        def wrapper(*args, **kwargs):

            #################################################################################
            #  Retrieve the arrays storing prim_haloprop and sec_haloprop
            #  The control flow below is what permits accepting an input
            #  table or a directly inputting prim_haloprop and sec_haloprop arrays

            _HAS_table = False
            if 'table' in kwargs:
                try:
                    table = kwargs['table']
                    prim_haloprop = table[self.prim_haloprop_key]
                    sec_haloprop = table[self.sec_haloprop_key]
                    _HAS_table = True
                except KeyError:
                    msg = ("When passing an input ``table`` to the "
                           " ``assembias_decorator`` method,\n"
                           "the input table must have a column with name ``%s``"
                           "and a column with name ``%s``.\n")
                    raise HalotoolsError(msg % (self.prim_haloprop_key), self.sec_haloprop_key)
            else:
                try:
                    prim_haloprop = np.atleast_1d(kwargs['prim_haloprop'])
                except KeyError:
                    msg = ("\nIf not passing an input ``table`` to the "
                           "``assembias_decorator`` method,\n"
                           "you must pass ``prim_haloprop`` argument.\n")
                    raise HalotoolsError(msg)
                try:
                    sec_haloprop = np.atleast_1d(kwargs['sec_haloprop'])
                except KeyError:
                    msg = ("\nIf not passing an input ``table`` to the "
                           "``assembias_decorator`` method,\n"
                           "you must pass ``sec_haloprop`` argument")
                    raise HalotoolsError(msg)

            #################################################################################

            #  Compute the percentile to split on as a function of the input prim_haloprop
            split = self.percentile_splitting_function(prim_haloprop)

            #  Compute the baseline, undecorated result
            result = func(*args, **kwargs)

            #  We will only decorate values that are not edge cases,
            #  so first compute the mask for non-edge cases
            no_edge_mask = (
                (split > 0) & (split < 1) &
                (result > baseline_lower_bound) & (result < baseline_upper_bound)
            )
            #  Now create convenient references to the non-edge-case sub-arrays
            no_edge_result = result[no_edge_mask]
            no_edge_split = split[no_edge_mask]

            # TODO i can maybe figure out how to cache the percentiles in the table, but for hte time being i'll just to retrieve them everytime 
            if _HAS_table is True:
                if self.sec_haloprop_key + '_percentile_values' in list(table.keys()):
#                    no_edge_percentile_values = table[self.sec_haloprop_key + '_percentile_value'][no_edge_mask]
                    pass
                else:
                    #  the value of sec_haloprop_percentile will be computed from scratch
                    #no_edge_percentile_values = compute_conditional_percentile_values( p=no_edge_split,
                    #    prim_haloprop=prim_haloprop[no_edge_mask],
                    #    sec_haloprop=sec_haloprop[no_edge_mask]
                    #)
                    pass
            else:
                pass
                '''
                try:
                    percentiles = kwargs['sec_haloprop_percentile_values']
                    if custom_len(percentiles) == 1:
                        percentiles = np.zeros(custom_len(prim_haloprop)) + percentiles
                    no_edge_percentile_values = percentiles[no_edge_mask]
                except KeyError:
                    no_edge_percentile_values = compute_conditional_percentile_values(p=no_edge_split,
                        prim_haloprop=prim_haloprop[no_edge_mask],
                        sec_haloprop=sec_haloprop[no_edge_mask]
                    )
                 '''

            #  NOTE I've removed the type 1 mask as it is not well-defined in this implementation
            strength = self.assembias_strength(prim_haloprop[no_edge_mask])

            shuffled_ranks = compute_conditional_shuffled_ranks(prim_haloprop = prim_haloprop[no_edge_mask], sec_haloprop=sec_haloprop[no_edge_mask], correlation_coeff=strength)

            dist = bernoulli if baseline_upper_bound == 1 else poisson 

            return dist.isf(1-shuffled_ranks, mu=result)

        return wrapper



