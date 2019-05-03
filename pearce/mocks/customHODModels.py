#!/bin/bash
# This module contains custom HODModels, which are subclasses of Halotools implementations.

import numpy as np
from scipy.special import erf
from scipy.interpolate import interp1d
import warnings

from halotools.empirical_models import Zheng07Cens, Zheng07Sats, OccupationComponent, model_defaults
from halotools.empirical_models import \
    HeavisideAssembias  # , ContinuousAssembias, FreeSplitAssembias, FreeSplitContinuousAssembias
# from halotools.empirical_models import CorrelationAssembias
from halotools.custom_exceptions import HalotoolsError
from halotools.utils.table_utils import compute_conditional_percentiles

from pearce.mocks.assembias_models import *


# TODO change this to use get_published_parameters and add these params in.
class RedMagicCens(Zheng07Cens):
    '''Tweak of the Zheng model to add a new parameter, f_c, denoting a modified central fraction.'''

    def __init__(self, **kwargs):
        super(RedMagicCens, self).__init__(**kwargs)

        # Default values from our analysis
        defaults = {'logMmin': 12.1, 'f_c': 0.19, 'sigma_logM': 0.46}

        self.param_dict.update(defaults)  # overwrite halotools zheng07 defaults with our own

    def mean_occupation(self, **kwargs):
        '''See Zheng07 for details.'''
        return self.param_dict['f_c'] * super(RedMagicCens, self).mean_occupation(**kwargs)


class AssembiasRedMagicCens(RedMagicCens, ContinuousAssembias):
    '''RedMagic Cens with Continuous Assembly bias'''

    def __init__(self, **kwargs):
        '''See halotools docs for more info. '''
        super(AssembiasRedMagicCens, self).__init__(**kwargs)

        fkwargs = kwargs.copy()
        if 'sec_haloprop_key' not in fkwargs:
            fkwargs['sec_haloprop_key'] = 'halo_nfw_conc'

        if 'method_name_to_decorate' not in fkwargs:
            fkwargs['method_name_to_decorate'] = 'mean_occupation'

        ContinuousAssembias.__init__(self,
                                     lower_assembias_bound=self._lower_occupation_bound,
                                     upper_assembias_bound=self._upper_occupation_bound,
                                     **fkwargs)


class HSAssembiasRedMagicCens(RedMagicCens, HeavisideAssembias):
    '''RedMagic Cens with Heaviside Assembly bias'''

    def __init__(self, **kwargs):
        '''See halotools docs for more info. '''
        super(HSAssembiasRedMagicCens, self).__init__(**kwargs)

        sec_haloprop_key = 'halo_nfw_conc'
        if 'sec_haloprop_key' not in kwargs:
            kwargs['sec_haloprop_key'] = sec_haloprop_key

        HeavisideAssembias.__init__(self,
                                    lower_assembias_bound=self._lower_occupation_bound,
                                    upper_assembias_bound=self._upper_occupation_bound,
                                    method_name_to_decorate='mean_occupation',
                                    **kwargs)


class FSAssembiasRedMagicCens(RedMagicCens, FreeSplitAssembias):
    '''RedMagic Cens with Heaviside Assembly bias'''

    def __init__(self, **kwargs):
        '''See halotools docs for more info. '''
        super(FSAssembiasRedMagicCens, self).__init__(**kwargs)

        sec_haloprop_key = 'halo_nfw_conc'
        if 'sec_haloprop_key' not in kwargs:
            kwargs['sec_haloprop_key'] = sec_haloprop_key

        FreeSplitAssembias.__init__(self,
                                    lower_assembias_bound=self._lower_occupation_bound,
                                    upper_assembias_bound=self._upper_occupation_bound,
                                    method_name_to_decorate='mean_occupation',
                                    **kwargs)


class FSCAssembiasRedMagicCens(RedMagicCens, FreeSplitContinuousAssembias):
    '''RedMagic Cens with Heaviside Assembly bias'''

    def __init__(self, **kwargs):
        '''See halotools docs for more info. '''
        super(FSCAssembiasRedMagicCens, self).__init__(**kwargs)

        sec_haloprop_key = 'halo_nfw_conc'
        if 'sec_haloprop_key' not in kwargs:
            kwargs['sec_haloprop_key'] = sec_haloprop_key

        FreeSplitContinuousAssembias.__init__(self,
                                              lower_assembias_bound=self._lower_occupation_bound,
                                              upper_assembias_bound=self._upper_occupation_bound,
                                              method_name_to_decorate='mean_occupation',
                                              **kwargs)


class CorrAssembiasRedMagicCens(RedMagicCens, CorrelationAssembias):
    '''RedMagic Cens with Heaviside Assembly bias'''

    def __init__(self, **kwargs):
        '''See halotools docs for more info. '''
        super(CorrAssembiasRedMagicCens, self).__init__(**kwargs)

        sec_haloprop_key = 'halo_nfw_conc'
        if 'sec_haloprop_key' not in kwargs:
            kwargs['sec_haloprop_key'] = sec_haloprop_key

        CorrelationAssembias.__init__(self,
                                      lower_assembias_bound=self._lower_occupation_bound,
                                      upper_assembias_bound=self._upper_occupation_bound,
                                      method_name_to_decorate='mean_occupation',
                                      **kwargs)


class RedMagicSats(Zheng07Sats):
    '''Tweak of Zheng model to add a new parameter, f_c, denoting a modified central fraction.'''

    def __init__(self, cenocc_model, **kwargs):
        # We define modulation to be true. An existing central model is required.
        super(RedMagicSats, self).__init__(modulate_with_cenocc=True, cenocc_model=cenocc_model, **kwargs)
        defaults = {'logM0': 12.20, 'logM1': 13.7, 'alpha': 1.02, 'logMmin': 12.1, 'f_c': 0.19, 'sigma_logM': 0.46}
        self.param_dict.update(defaults)
        # It does not like that we have parameters defined multiple places.
        # Required for the central occupations.
        self._suppress_repeated_param_warning = True

    def mean_occupation(self, **kwargs):
        "See Zheng07 for details"
        f_c = 1
        if 'f_c' in self.param_dict:
            f_c = self.param_dict['f_c']

        return super(RedMagicSats, self).mean_occupation(**kwargs) / f_c


class AssembiasRedMagicSats(RedMagicSats, ContinuousAssembias):
    '''RedMagic Cens with Assembly bias'''

    def __init__(self, cenocc_model, **kwargs):
        '''See halotools docs for more info. '''
        super(AssembiasRedMagicSats, self).__init__(cenocc_model, **kwargs)

        fkwargs = kwargs.copy()
        if 'sec_haloprop_key' not in fkwargs:
            fkwargs['sec_haloprop_key'] = 'halo_nfw_conc'

        if 'method_name_to_decorate' not in fkwargs:
            fkwargs['method_name_to_decorate'] = 'mean_occupation'

        ContinuousAssembias.__init__(self,
                                     lower_assembias_bound=self._lower_occupation_bound,
                                     upper_assembias_bound=self._upper_occupation_bound,
                                     **fkwargs)


class HSAssembiasRedMagicSats(RedMagicSats, HeavisideAssembias):
    '''RedMagic Cens with Assembly bias'''

    def __init__(self, cenocc_model, **kwargs):
        '''See halotools docs for more info. '''
        super(HSAssembiasRedMagicSats, self).__init__(cenocc_model, **kwargs)
        sec_haloprop_key = 'halo_nfw_conc'
        if 'sec_haloprop_key' not in kwargs:
            kwargs['sec_haloprop_key'] = sec_haloprop_key

        HeavisideAssembias.__init__(self,
                                    lower_assembias_bound=self._lower_occupation_bound,
                                    upper_assembias_bound=self._upper_occupation_bound,
                                    method_name_to_decorate='mean_occupation',
                                    **kwargs)


class FSAssembiasRedMagicSats(RedMagicSats, FreeSplitAssembias):
    '''RedMagic Cens with Assembly bias'''

    def __init__(self, cenocc_model, **kwargs):
        '''See halotools docs for more info. '''
        super(FSAssembiasRedMagicSats, self).__init__(cenocc_model, **kwargs)
        sec_haloprop_key = 'halo_nfw_conc'
        if 'sec_haloprop_key' not in kwargs:
            kwargs['sec_haloprop_key'] = sec_haloprop_key

        FreeSplitAssembias.__init__(self,
                                    lower_assembias_bound=self._lower_occupation_bound,
                                    upper_assembias_bound=self._upper_occupation_bound,
                                    method_name_to_decorate='mean_occupation',
                                    **kwargs)


class FSCAssembiasRedMagicSats(RedMagicSats, FreeSplitContinuousAssembias):
    '''RedMagic Cens with Assembly bias'''

    def __init__(self, cenocc_model, **kwargs):
        '''See halotools docs for more info. '''
        super(FSCAssembiasRedMagicSats, self).__init__(cenocc_model, **kwargs)
        sec_haloprop_key = 'halo_nfw_conc'
        if 'sec_haloprop_key' not in kwargs:
            kwargs['sec_haloprop_key'] = sec_haloprop_key

        FreeSplitContinuousAssembias.__init__(self,
                                              lower_assembias_bound=self._lower_occupation_bound,
                                              upper_assembias_bound=self._upper_occupation_bound,
                                              method_name_to_decorate='mean_occupation',
                                              **kwargs)


class CorrAssembiasRedMagicSats(RedMagicSats, CorrelationAssembias):
    '''RedMagic Cens with Assembly bias'''

    def __init__(self, cenocc_model, **kwargs):
        '''See halotools docs for more info. '''
        super(CorrAssembiasRedMagicSats, self).__init__(cenocc_model, **kwargs)
        sec_haloprop_key = 'halo_nfw_conc'
        if 'sec_haloprop_key' not in kwargs:
            kwargs['sec_haloprop_key'] = sec_haloprop_key

        CorrelationAssembias.__init__(self,
                                      lower_assembias_bound=self._lower_occupation_bound,
                                      upper_assembias_bound=self._upper_occupation_bound,
                                      method_name_to_decorate='mean_occupation',
                                      **kwargs)


class AssembiasZheng07Cens(Zheng07Cens, ContinuousAssembias):
    '''RedMagic Cens with Continuous Assembly bias'''

    def __init__(self, **kwargs):
        '''See halotools docs for more info. '''
        super(AssembiasZheng07Cens, self).__init__(**kwargs)

        fkwargs = kwargs.copy()
        if 'sec_haloprop_key' not in fkwargs:
            fkwargs['sec_haloprop_key'] = 'halo_nfw_conc'

        if 'method_name_to_decorate' not in fkwargs:
            fkwargs['method_name_to_decorate'] = 'mean_occupation'

        ContinuousAssembias.__init__(self,
                                     lower_assembias_bound=self._lower_occupation_bound,
                                     upper_assembias_bound=self._upper_occupation_bound,
                                     **fkwargs)


class HSAssembiasZheng07Cens(Zheng07Cens, HeavisideAssembias):
    '''RedMagic Cens with Heaviside Assembly bias'''

    def __init__(self, **kwargs):
        '''See halotools docs for more info. '''
        super(HSAssembiasZheng07Cens, self).__init__(**kwargs)

        sec_haloprop_key = 'halo_nfw_conc'
        if 'sec_haloprop_key' not in kwargs:
            kwargs['sec_haloprop_key'] = sec_haloprop_key

        HeavisideAssembias.__init__(self,
                                    lower_assembias_bound=self._lower_occupation_bound,
                                    upper_assembias_bound=self._upper_occupation_bound,
                                    method_name_to_decorate='mean_occupation',
                                    **kwargs)


class FSAssembiasZheng07Cens(Zheng07Cens, FreeSplitAssembias):
    '''RedMagic Cens with Heaviside Assembly bias'''

    def __init__(self, **kwargs):
        '''See halotools docs for more info. '''
        super(FSAssembiasZheng07Cens, self).__init__(**kwargs)

        sec_haloprop_key = 'halo_nfw_conc'
        if 'sec_haloprop_key' not in kwargs:
            kwargs['sec_haloprop_key'] = sec_haloprop_key

        FreeSplitAssembias.__init__(self,
                                    lower_assembias_bound=self._lower_occupation_bound,
                                    upper_assembias_bound=self._upper_occupation_bound,
                                    method_name_to_decorate='mean_occupation',
                                    **kwargs)


class FSCAssembiasZheng07Cens(Zheng07Cens, FreeSplitContinuousAssembias):
    '''RedMagic Cens with Heaviside Assembly bias'''

    def __init__(self, **kwargs):
        '''See halotools docs for more info. '''
        super(FSCAssembiasZheng07Cens, self).__init__(**kwargs)

        sec_haloprop_key = 'halo_nfw_conc'
        if 'sec_haloprop_key' not in kwargs:
            kwargs['sec_haloprop_key'] = sec_haloprop_key

        FreeSplitContinuousAssembias.__init__(self,
                                              lower_assembias_bound=self._lower_occupation_bound,
                                              upper_assembias_bound=self._upper_occupation_bound,
                                              method_name_to_decorate='mean_occupation',
                                              **kwargs)


class CorrAssembiasZheng07Cens(Zheng07Cens, CorrelationAssembias):
    '''RedMagic Cens with Heaviside Assembly bias'''

    def __init__(self, **kwargs):
        '''See halotools docs for more info. '''
        super(CorrAssembiasZheng07Cens, self).__init__(**kwargs)

        sec_haloprop_key = 'halo_nfw_conc'
        if 'sec_haloprop_key' not in kwargs:
            kwargs['sec_haloprop_key'] = sec_haloprop_key

        CorrelationAssembias.__init__(self,
                                      lower_assembias_bound=self._lower_occupation_bound,
                                      upper_assembias_bound=self._upper_occupation_bound,
                                      method_name_to_decorate='mean_occupation',
                                      **kwargs)


class AssembiasZheng07Sats(Zheng07Sats, ContinuousAssembias):
    '''RedMagic Cens with Assembly bias'''

    def __init__(self, cenocc_model, **kwargs):
        '''See halotools docs for more info. '''
        super(AssembiasZheng07Sats, self).__init__(modulate_with_cenocc=True, cenocc_model=cenocc_model, **kwargs)

        fkwargs = kwargs.copy()
        if 'sec_haloprop_key' not in fkwargs:
            fkwargs['sec_haloprop_key'] = 'halo_nfw_conc'

        if 'method_name_to_decorate' not in fkwargs:
            fkwargs['method_name_to_decorate'] = 'mean_occupation'

        ContinuousAssembias.__init__(self,
                                     lower_assembias_bound=self._lower_occupation_bound,
                                     upper_assembias_bound=self._upper_occupation_bound,
                                     **fkwargs)


class HSAssembiasZheng07Sats(Zheng07Sats, HeavisideAssembias):
    '''RedMagic Cens with Assembly bias'''

    def __init__(self, cenocc_model, **kwargs):
        '''See halotools docs for more info. '''
        super(HSAssembiasZheng07Sats, self).__init__(modulate_with_cenocc=True, cenocc_model=cenocc_model, **kwargs)
        sec_haloprop_key = 'halo_nfw_conc'
        if 'sec_haloprop_key' not in kwargs:
            kwargs['sec_haloprop_key'] = sec_haloprop_key

        HeavisideAssembias.__init__(self,
                                    lower_assembias_bound=self._lower_occupation_bound,
                                    upper_assembias_bound=self._upper_occupation_bound,
                                    method_name_to_decorate='mean_occupation',
                                    **kwargs)


class FSAssembiasZheng07Sats(Zheng07Sats, FreeSplitAssembias):
    '''RedMagic Cens with Assembly bias'''

    def __init__(self, cenocc_model, **kwargs):
        '''See halotools docs for more info. '''
        super(FSAssembiasZheng07Sats, self).__init__(modulate_with_cenocc=True, cenocc_model=cenocc_model, **kwargs)
        sec_haloprop_key = 'halo_nfw_conc'
        if 'sec_haloprop_key' not in kwargs:
            kwargs['sec_haloprop_key'] = sec_haloprop_key

        FreeSplitAssembias.__init__(self,
                                    lower_assembias_bound=self._lower_occupation_bound,
                                    upper_assembias_bound=self._upper_occupation_bound,
                                    method_name_to_decorate='mean_occupation',
                                    **kwargs)


class FSCAssembiasZheng07Sats(Zheng07Sats, FreeSplitContinuousAssembias):
    '''RedMagic Cens with Assembly bias'''

    def __init__(self, cenocc_model, **kwargs):
        '''See halotools docs for more info. '''
        super(FSCAssembiasZheng07Sats, self).__init__(modulate_with_cenocc=True, cenocc_model=cenocc_model, **kwargs)
        sec_haloprop_key = 'halo_nfw_conc'
        if 'sec_haloprop_key' not in kwargs:
            kwargs['sec_haloprop_key'] = sec_haloprop_key

        FreeSplitContinuousAssembias.__init__(self,
                                              lower_assembias_bound=self._lower_occupation_bound,
                                              upper_assembias_bound=self._upper_occupation_bound,
                                              method_name_to_decorate='mean_occupation',
                                              **kwargs)


class CorrAssembiasZheng07Sats(Zheng07Sats, CorrelationAssembias):
    '''RedMagic Cens with Assembly bias'''

    def __init__(self, cenocc_model, **kwargs):
        '''See halotools docs for more info. '''
        super(CorrAssembiasZheng07Sats, self).__init__(modulate_with_cenocc=True, cenocc_model=cenocc_model, **kwargs)
        sec_haloprop_key = 'halo_nfw_conc'
        if 'sec_haloprop_key' not in kwargs:
            kwargs['sec_haloprop_key'] = sec_haloprop_key

        CorrelationAssembias.__init__(self,
                                      lower_assembias_bound=self._lower_occupation_bound,
                                      upper_assembias_bound=self._upper_occupation_bound,
                                      method_name_to_decorate='mean_occupation',
                                      **kwargs)


class Reddick14Cens(OccupationComponent):
    r""" ``Erf`` function model for the occupation statistics of central galaxies,
    introduced in Reddick et al. 2014, arXiv:1306.4686.
    """

    # TODO figure out how to use threshold, here and in the other classes
    def __init__(self,
                 threshold=model_defaults.default_luminosity_threshold,
                 prim_haloprop_key=model_defaults.prim_haloprop_key,
                 **kwargs):
        r"""
        Parameters
        ----------
        threshold : float, optional
            Luminosity threshold of the mock galaxy sample. Unsure it will do anything at present.
            # TODO will need this later, test.

        prim_haloprop_key : string, optional
            String giving the column name of the primary halo property governing
            the occupation statistics of gal_type galaxies.
            Default value is specified in the `~halotools.empirical_models.model_defaults` module.

        Examples
        --------
        >>> cen_model = Reddick14Cens()
        >>> cen_model = Reddick14Cens(threshold=-19.5)
        >>> cen_model = Reddick14Cens(prim_haloprop_key='halo_m200b')
        """
        upper_occupation_bound = 1.0

        # Call the super class constructor, which binds all the
        # arguments to the instance.
        super(Reddick14Cens, self).__init__(gal_type='centrals',
                                            threshold=threshold, upper_occupation_bound=upper_occupation_bound,
                                            prim_haloprop_key=prim_haloprop_key,
                                            **kwargs)

        self.param_dict = self.get_published_parameters()

        self.publications = ['arXiv:1306.4686']

    def mean_occupation(self, **kwargs):
        r""" Expected number of central galaxies in a halo of mass halo_mass.
        See Equation 4 of arXiv:1306.4686.

        Parameters
        ----------
        prim_haloprop : array, optional
            Array of mass-like variable upon which occupation statistics are based.
            If ``prim_haloprop`` is not passed, then ``table`` keyword argument must be passed.

        table : object, optional
            Data table storing halo catalog.
            If ``table`` is not passed, then ``prim_haloprop`` keyword argument must be passed.

        Returns
        -------
        mean_ncen : array
            Mean number of central galaxies in the input table.

        Examples
        --------
        >>> cen_model = Reddick14Cens()

        The `mean_occupation` method of all OccupationComponent instances supports
        two different options for arguments. The first option is to directly
        pass the array of the primary halo property:

        >>> testmass = np.logspace(10, 15, num=50)
        >>> mean_ncen = cen_model.mean_occupation(prim_haloprop = testmass)

        The second option is to pass `mean_occupation` a full halo catalog.
        In this case, the array storing the primary halo property will be selected
        by accessing the ``cen_model.prim_haloprop_key`` column of the input halo catalog.
        For illustration purposes, we'll use a fake halo catalog rather than a
        (much larger) full one:

        >>> from halotools.sim_manager import FakeSim
        >>> fake_sim = FakeSim()
        >>> mean_ncen = cen_model.mean_occupation(table=fake_sim.halo_table)

        Notes
        -----
        The `mean_occupation` method computes the following function:

        :math:`\langle N_{\mathrm{cen}} \rangle_{M} =
        \frac{1}{2}\left( 1 +
        \mathrm{erf}\left( \frac{\log_{10}M -
        \log_{10}M_{min}}{\sigma_{\log_{10}M}} \right) \right)`

        """
        # Retrieve the array storing the mass-like variable
        if 'table' in list(kwargs.keys()):
            mass = kwargs['table'][self.prim_haloprop_key]
        elif 'prim_haloprop' in list(kwargs.keys()):
            mass = np.atleast_1d(kwargs['prim_haloprop'])
        else:
            msg = ("\nYou must pass either a ``table`` or ``prim_haloprop`` argument \n"
                   "to the ``mean_occupation`` function of the ``Zheng07Cens`` class.\n")
            raise HalotoolsError(msg)

        logM = np.log10(mass)
        mean_ncen = 0.5 * (1.0 + erf(
            (logM - self.param_dict['logMmin']) / self.param_dict['sigma_logM'])) * \
                    (1.0 + self.param_dict['f_cen'] * (logM - self.param_dict['logMlin']))
        # enforce upper limit
        mean_ncen[mean_ncen > self._upper_occupation_bound] = self._upper_occupation_bound
        return mean_ncen

    # TODO add parameters/threshold here
    def get_published_parameters(self):
        r"""
        Best-fit HOD parameters for Reddick 14. They aren't published, so I'm guessing at good values.

        Parameters
        ----------
        None

        Returns
        -------
        param_dict : dict
            Dictionary of model parameters whose values have been set to
            agree with the values taken from Table 1 of Zheng et al. 2007.

        Examples
        --------
        >>> cen_model = Reddick14Cens()
        >>> cen_model.param_dict = cen_model.get_published_parameters(cen_model.threshold)

        """
        return {'logMmin': 12.9,
                'sigma_logM': 0.75,
                'f_cen': 0.05,
                'logMlin': 16.0}


class AssembiasReddick14Cens(Reddick14Cens, ContinuousAssembias):
    '''Reddick14 Cens with Continuous Assembly bias'''

    def __init__(self, **kwargs):
        '''See halotools docs for more info. '''
        super(AssembiasReddick14Cens, self).__init__(**kwargs)

        sec_haloprop_key = 'halo_nfw_conc'
        if 'sec_haloprop_key' not in kwargs:
            kwargs['sec_haloprop_key'] = sec_haloprop_key

        ContinuousAssembias.__init__(self,
                                     lower_assembias_bound=self._lower_occupation_bound,
                                     upper_assembias_bound=self._upper_occupation_bound,
                                     method_name_to_decorate='mean_occupation',
                                     **kwargs)


class HSAssembiasReddick14Cens(Reddick14Cens, HeavisideAssembias):
    '''Reddick14 Cens with Heaviside Assembly bias'''

    def __init__(self, **kwargs):
        '''See halotools docs for more info. '''
        super(HSAssembiasReddick14Cens, self).__init__(**kwargs)

        sec_haloprop_key = 'halo_nfw_conc'
        if 'sec_haloprop_key' not in kwargs:
            kwargs['sec_haloprop_key'] = sec_haloprop_key

        HeavisideAssembias.__init__(self,
                                    lower_assembias_bound=self._lower_occupation_bound,
                                    upper_assembias_bound=self._upper_occupation_bound,
                                    method_name_to_decorate='mean_occupation',
                                    **kwargs)


class FSAssembiasReddick14Cens(Reddick14Cens, FreeSplitAssembias):
    '''Reddick14 Cens with Heaviside Assembly bias'''

    def __init__(self, **kwargs):
        '''See halotools docs for more info. '''
        super(FSAssembiasReddick14Cens, self).__init__(**kwargs)

        sec_haloprop_key = 'halo_nfw_conc'
        if 'sec_haloprop_key' not in kwargs:
            kwargs['sec_haloprop_key'] = sec_haloprop_key

        FreeSplitAssembias.__init__(self,
                                    lower_assembias_bound=self._lower_occupation_bound,
                                    upper_assembias_bound=self._upper_occupation_bound,
                                    method_name_to_decorate='mean_occupation',
                                    **kwargs)


class CorrAssembiasReddick14Cens(Reddick14Cens, CorrelationAssembias):
    '''Reddick14 Cens with Heaviside Assembly bias'''

    def __init__(self, **kwargs):
        '''See halotools docs for more info. '''
        super(CorrAssembiasReddick14Cens, self).__init__(**kwargs)

        sec_haloprop_key = 'halo_nfw_conc'
        if 'sec_haloprop_key' not in kwargs:
            kwargs['sec_haloprop_key'] = sec_haloprop_key

        CorrelationAssembias.__init__(self,
                                      lower_assembias_bound=self._lower_occupation_bound,
                                      upper_assembias_bound=self._upper_occupation_bound,
                                      method_name_to_decorate='mean_occupation',
                                      **kwargs)


class Reddick14Sats(OccupationComponent):
    r""" Power law model for the occupation statistics of satellite galaxies,
    introduced in Kravtsov et al. 2004, arXiv:0308519. This implementation uses
    Reddick et al. 2014, arXiv:1306.4686, to assign fiducial parameter values.

    :math:`\langle N_{sat} \rangle_{M} = \left(\frac{M_{vir}}{M_1}\right)^{\alpha}\exp(\frac{-M_{cut}}{M_{vir}}).
    """

    def __init__(self,
                 threshold=model_defaults.default_luminosity_threshold,
                 prim_haloprop_key=model_defaults.prim_haloprop_key,
                 modulate_with_cenocc=True, cenocc_model=None, **kwargs):
        r"""
        Parameters
        ----------
        threshold : float, optional
            Currently doesn't do anything. Will add in later.

        prim_haloprop_key : string, optional
            String giving the column name of the primary halo property governing
            the occupation statistics of gal_type galaxies.
            Default value is specified in the `~halotools.empirical_models.model_defaults` module.

        modulate_with_cenocc : bool, optional
            If set to True, the `Zheng07Sats.mean_occupation` method will
            be multiplied by the the first moment of the centrals:

            :math:`\langle N_{\mathrm{sat}}\rangle_{M}\Rightarrow\langle N_{\mathrm{sat}}\rangle_{M}\times\langle N_{\mathrm{cen}}\rangle_{M}`

            The ``cenocc_model`` keyword argument works together with
            the ``modulate_with_cenocc`` keyword argument to determine how
            the :math:`\langle N_{\mathrm{cen}}\rangle_{M}` prefactor is calculated.

        cenocc_model : `OccupationComponent`, optional
            If the ``cenocc_model`` keyword argument is set to its default value
            of None, then the :math:`\langle N_{\mathrm{cen}}\rangle_{M}` prefactor will be
            calculated according to `Reddick14Cens.mean_occupation`.
            However, if an instance of the `OccupationComponent` class is instead
            passed in via the ``cenocc_model`` keyword,
            then the first satellite moment will be multiplied by
            the ``mean_occupation`` function of the ``cenocc_model``.
            The ``modulate_with_cenocc`` keyword must be set to True in order
            for the ``cenocc_model`` to be operative.
            See :ref:`zheng07_using_cenocc_model_tutorial` for further details.

        Examples
        --------
        >>> sat_model = Reddick14Sats()
        >>> sat_model = ReddickSats(threshold = -21)

        The ``param_dict`` attribute can be used to build an alternate
        model from an existing instance. This feature has a variety of uses. For example,
        suppose you wish to study how the choice of halo mass definition impacts HOD predictions:

        >>> sat_model1 = ReddickSats(threshold = -19.5, prim_haloprop_key='m200b')
        >>> sat_model1.param_dict['alpha'] = 1.05
        >>> sat_model2 = ReddickSats(threshold = -19.5, prim_haloprop_key='m500c')
        >>> sat_model2.param_dict = sat_model1.param_dict

        After executing the above four lines of code, ``sat_model1`` and ``sat_model2`` are
        identical in every respect, excepting only for the difference in the halo mass definition.

        A common convention in HOD modeling of satellite populations is for the first
        occupation moment of the satellites to be multiplied by the first occupation
        moment of the associated central population.
        The ``cenocc_model`` keyword arguments allows you
        to study the impact of this choice:

        >>> sat_model1 = Reddick14Sats(threshold=-18)
        >>> sat_model2 = Reddick14Sats(threshold = sat_model1.threshold, modulate_with_cenocc=True)

        Now ``sat_model1`` and ``sat_model2`` are identical in every respect,
        excepting only the following difference:

        :math:`\langle N_{\mathrm{sat}}\rangle^{\mathrm{model 2}} = \langle N_{\mathrm{cen}}\rangle\times\langle N_{\mathrm{sat}}\rangle^{\mathrm{model 1}}`
        """
        upper_occupation_bound = float("inf")

        # Call the super class constructor, which binds all the
        # arguments to the instance.
        super(Reddick14Sats, self).__init__(
            gal_type='satellites', threshold=threshold,
            upper_occupation_bound=upper_occupation_bound,
            prim_haloprop_key=prim_haloprop_key,
            **kwargs)

        self.param_dict = self.get_published_parameters()

        if cenocc_model is None:
            cenocc_model = Reddick14Cens(
                prim_haloprop_key=prim_haloprop_key, threshold=threshold)
        else:
            if modulate_with_cenocc is False:
                msg = ("You chose to input a ``cenocc_model``, but you set the \n"
                       "``modulate_with_cenocc`` keyword to False, so your "
                       "``cenocc_model`` will have no impact on the model's behavior.\n"
                       "Be sure this is what you intend before proceeding.\n"
                       "Refer to the Zheng et al. (2007) composite model tutorial for details.\n")
                warnings.warn(msg)

        self.modulate_with_cenocc = modulate_with_cenocc
        if self.modulate_with_cenocc:
            try:
                assert isinstance(cenocc_model, OccupationComponent)
            except AssertionError:
                msg = ("The input ``cenocc_model`` must be an instance of \n"
                       "``OccupationComponent`` or one of its sub-classes.\n")
                raise HalotoolsError(msg)

            self.central_occupation_model = cenocc_model

            self.param_dict.update(self.central_occupation_model.param_dict)

        self.publications = ['arXiv:0308519']

    def mean_occupation(self, **kwargs):
        r"""Expected number of satellite galaxies in a halo of mass logM.
        See Equation 5 of arXiv:0308519.

        Parameters
        ----------
        prim_haloprop : array, optional
            Array storing a mass-like variable that governs the occupation statistics.
            If ``prim_haloprop`` is not passed, then ``table``
            keyword arguments must be passed.

        table : object, optional
            Data table storing halo catalog.
            If ``table`` is not passed, then ``prim_haloprop``
            keyword arguments must be passed.

        Returns
        -------
        mean_nsat : float or array
            Mean number of satellite galaxies in a host halo of the specified mass.

        :math:`\langle N_{\mathrm{sat}} \rangle_{M} = \left(\frac{M_{vir}}{M_1}\right)^{\alpha}\exp(\frac{-M_{cut}}{M_{vir}}).
\langle N_{\mathrm{cen}} \rangle_{M}`

        or

        :math:`\langle N_{\mathrm{sat}} \rangle_{M} = \left(\frac{M_{vir}}{M_1}\right)^{\alpha}\exp(\frac{-M_{cut}}{M_{vir}}).


        depending on whether a central model was passed to the constructor.

        Examples
        --------
        The `mean_occupation` method of all OccupationComponent instances supports
        two different options for arguments. The first option is to directly
        pass the array of the primary halo property:

        >>> sat_model = Reddick14Sats()
        >>> testmass = np.logspace(10, 15, num=50)
        >>> mean_nsat = sat_model.mean_occupation(prim_haloprop = testmass)

        The second option is to pass `mean_occupation` a full halo catalog.
        In this case, the array storing the primary halo property will be selected
        by accessing the ``sat_model.prim_haloprop_key`` column of the input halo catalog.
        For illustration purposes, we'll use a fake halo catalog rather than a
        (much larger) full one:

        >>> from halotools.sim_manager import FakeSim
        >>> fake_sim = FakeSim()
        >>> mean_nsat = sat_model.mean_occupation(table=fake_sim.halo_table)

        """

        if self.modulate_with_cenocc:
            for key, value in list(self.param_dict.items()):
                if key in self.central_occupation_model.param_dict:
                    self.central_occupation_model.param_dict[key] = value

        # Retrieve the array storing the mass-like variable
        if 'table' in list(kwargs.keys()):
            mass = kwargs['table'][self.prim_haloprop_key]
        elif 'prim_haloprop' in list(kwargs.keys()):
            mass = np.atleast_1d(kwargs['prim_haloprop'])
        else:
            msg = ("\nYou must pass either a ``table`` or ``prim_haloprop`` argument \n"
                   "to the ``mean_occupation`` function of the ``Zheng07Sats`` class.\n")
            raise HalotoolsError(msg)

        Mcut = 10. ** self.param_dict['logMcut']
        M1 = 10. ** self.param_dict['logM1']

        # Call to np.where raises a harmless RuntimeWarning exception if
        # there are entries of input logM for which mean_nsat = 0
        # Evaluating mean_nsat using the catch_warnings context manager
        # suppresses this warning

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)

            mean_nsat = ((mass / M1) ** self.param_dict['alpha']) * np.exp(-1.0 * Mcut / mass)

        # If a central occupation model was passed to the constructor,
        # multiply mean_nsat by an overall factor of mean_ncen
        if self.modulate_with_cenocc:
            mean_ncen = self.central_occupation_model.mean_occupation(**kwargs)
            mean_nsat *= mean_ncen

        return mean_nsat

    def get_published_parameters(self):
        r"""
        No published best-fit paramaters for this model, so making an educated guess.

        Parameters
        ----------
        None

        -------
        param_dict : dict
            Dictionary of model parameters whose values have been set to
            the values taken from Table 1 of Zheng et al. 2007.

        Examples
        --------
        >>> sat_model = Reddick14Sats()
        >>> sat_model.param_dict = sat_model.get_published_parameters(sat_model.threshold)
        """

        return {'logM1': 14.3,
                'alpha': 1.06,
                'logMcut': 12.5}


class AssembiasReddick14Sats(Reddick14Sats, ContinuousAssembias):
    '''Reddick14 Cens with Assembly bias'''

    def __init__(self, cenocc_model=None, **kwargs):
        '''See halotools docs for more info. '''
        super(AssembiasReddick14Sats, self).__init__(cenocc_model=cenocc_model, **kwargs)
        sec_haloprop_key = 'halo_nfw_conc'
        if 'sec_haloprop_key' not in kwargs:
            kwargs['sec_haloprop_key'] = sec_haloprop_key

        ContinuousAssembias.__init__(self,
                                     lower_assembias_bound=self._lower_occupation_bound,
                                     upper_assembias_bound=self._upper_occupation_bound,
                                     method_name_to_decorate='mean_occupation',
                                     **kwargs)


class HSAssembiasReddick14Sats(Reddick14Sats, HeavisideAssembias):
    '''Reddick14 Cens with Assembly bias'''

    def __init__(self, cenocc_model, **kwargs):
        '''See halotools docs for more info. '''
        super(HSAssembiasReddick14Sats, self).__init__(cenocc_model=cenocc_model, **kwargs)
        sec_haloprop_key = 'halo_nfw_conc'
        if 'sec_haloprop_key' not in kwargs:
            kwargs['sec_haloprop_key'] = sec_haloprop_key

        HeavisideAssembias.__init__(self,
                                    lower_assembias_bound=self._lower_occupation_bound,
                                    upper_assembias_bound=self._upper_occupation_bound,
                                    method_name_to_decorate='mean_occupation',
                                    **kwargs)


class FSAssembiasReddick14Sats(Reddick14Sats, FreeSplitAssembias):
    '''Reddick14 Cens with Assembly bias'''

    def __init__(self, cenocc_model, **kwargs):
        '''See halotools docs for more info. '''
        super(FSAssembiasReddick14Sats, self).__init__(cenocc_model=cenocc_model, **kwargs)
        sec_haloprop_key = 'halo_nfw_conc'
        if 'sec_haloprop_key' not in kwargs:
            kwargs['sec_haloprop_key'] = sec_haloprop_key

        FreeSplitAssembias.__init__(self,
                                    lower_assembias_bound=self._lower_occupation_bound,
                                    upper_assembias_bound=self._upper_occupation_bound,
                                    method_name_to_decorate='mean_occupation',
                                    **kwargs)


class CorrAssembiasReddick14Sats(Reddick14Sats, CorrelationAssembias):
    '''Reddick14 Cens with Assembly bias'''

    def __init__(self, cenocc_model, **kwargs):
        '''See halotools docs for more info. '''
        super(CorrAssembiasReddick14Sats, self).__init__(cenocc_model=cenocc_model, **kwargs)
        sec_haloprop_key = 'halo_nfw_conc'
        if 'sec_haloprop_key' not in kwargs:
            kwargs['sec_haloprop_key'] = sec_haloprop_key

        CorrelationAssembias.__init__(self,
                                      lower_assembias_bound=self._lower_occupation_bound,
                                      upper_assembias_bound=self._upper_occupation_bound,
                                      method_name_to_decorate='mean_occupation',
                                      **kwargs)


class StepFuncCens(Zheng07Cens):
    '''HOD model mainly for test purposes; a step function in centrals.'''

    def __init__(self, **kwargs):
        super(StepFuncCens, self).__init__(**kwargs)
        self.param_dict['logMmin'] = 12.1  # set default

    def mean_occupation(self, **kwargs):
        """See Zheng07 for details"""
        if 'table' in list(kwargs.keys()):
            mass = kwargs['table'][self.prim_haloprop_key]
        elif 'prim_haloprop' in list(kwargs.keys()):
            mass = kwargs['prim_haloprop']
        else:
            msg = ("\nYou must pass either a ``table`` or ``prim_haloprop`` argument \n"
                   "to the ``mean_occupation`` function of the ``StepFuncCens`` class.\n")
            raise RuntimeError(msg)
        mass = np.array(mass)
        if np.shape(mass) == ():
            mass = np.array([mass])

        Mmin = 10 ** self.param_dict['logMmin']

        return np.array(mass > Mmin, dtype=int)


class StepFuncSats(Zheng07Sats):
    """HOD model mainly for testing that is 0 for satellites."""

    def mean_occupation(self, **kwargs):
        """see Zheng07 for details."""
        "See Zheng07 for details"
        if 'table' in list(kwargs.keys()):
            mass = kwargs['table'][self.prim_haloprop_key]
        elif 'prim_haloprop' in list(kwargs.keys()):
            mass = kwargs['prim_haloprop']
        else:
            msg = ("\nYou must pass either a ``table`` or ``prim_haloprop`` argument \n"
                   "to the ``mean_occupation`` function of the ``StepFuncCens`` class.\n")
            raise RuntimeError(msg)
        mass = np.array(mass)
        if np.shape(mass) == ():
            mass = np.array([mass])

        return np.zeros_like(mass)


class TabulatedCens(OccupationComponent):
    r""" Central occupation that is fixed at observed values. Rather than being parameterized, populate with an HOD
    with a fixed, observed relationship.
    """

    def __init__(self, prim_haloprop_vals, cen_hod_vals,
                 threshold=model_defaults.default_luminosity_threshold,
                 prim_haloprop_key=model_defaults.prim_haloprop_key, **kwargs):
        r"""
        Parameters
        ----------
        prim_haloprop_vals: array
            Values of the prim haloprop that cen_hod_vals were observed

        cen_hod_vals: array
            The values of the hod observed at prim_haloprop_vals. Is interoplated to give the HOD

        threshold : float, optional
            Currently doesn't do anything. Will add in later.

        prim_haloprop_key : string, optional
            String giving the column name of the primary halo property governing
            the occupation statistics of gal_type galaxies.
            Default value is specified in the `~halotools.empirical_models.model_defaults` module.


        Examples
        --------
        >>> cen_model = TabulatedCens()
        >>> cen_model = TabulatedCens(threshold = -21)
        """

        upper_occupation_bound = 1.0

        assert np.all(cen_hod_vals >= 0) and np.all(upper_occupation_bound >= cen_hod_vals)
        assert cen_hod_vals.shape == prim_haloprop_vals.shape
        assert np.all(prim_haloprop_vals >= 0)

        self._prim_haloprop_vals = prim_haloprop_vals
        self._cen_hod_vals = cen_hod_vals

        self._mean_occupation = interp1d(np.log10(prim_haloprop_vals), cen_hod_vals, kind='cubic')

        # Call the super class constructor, which binds all the
        # arguments to the instance.
        super(TabulatedCens, self).__init__(
            gal_type='centrals', threshold=threshold,
            upper_occupation_bound=upper_occupation_bound,
            prim_haloprop_key=prim_haloprop_key,
            **kwargs)

        self.param_dict = self.get_published_parameters()

        self.publications = []

    def mean_occupation(self, **kwargs):
        r"""Expected number of satellite galaxies in a halo of mass logM.
        See Equation 5 of arXiv:0308519.

        Parameters
        ----------
        prim_haloprop : array, optional
            Array storing a mass-like variable that governs the occupation statistics.
            If ``prim_haloprop`` is not passed, then ``table``
            keyword arguments must be passed.

        table : object, optional
            Data table storing halo catalog.
            If ``table`` is not passed, then ``prim_haloprop``
            keyword arguments must be passed.

        Returns
        -------
        mean_ncen : float or array
            Mean number of central galaxies in a host halo of the specified mass, from the above interpolation

        Examples
        --------
        The `mean_occupation` method of all OccupationComponent instances supports
        two different options for arguments. The first option is to directly
        pass the array of the primary halo property:

        >>> cen_model = TabulatedCens()
        >>> testmass = np.logspace(10, 15, num=50)
        >>> mean_ncen = cen_model.mean_occupation(prim_haloprop = testmass)

        The second option is to pass `mean_occupation` a full halo catalog.
        In this case, the array storing the primary halo property will be selected
        by accessing the ``sat_model.prim_haloprop_key`` column of the input halo catalog.
        For illustration purposes, we'll use a fake halo catalog rather than a
        (much larger) full one:

        >>> from halotools.sim_manager import FakeSim
        >>> fake_sim = FakeSim()
        >>> mean_ncen = cen_model.mean_occupation(table=fake_sim.halo_table)

        """

        # Retrieve the array storing the mass-like variable
        if 'table' in list(kwargs.keys()):
            mass = kwargs['table'][self.prim_haloprop_key]
        elif 'prim_haloprop' in list(kwargs.keys()):
            mass = np.atleast_1d(kwargs['prim_haloprop'])
        else:
            msg = ("\nYou must pass either a ``table`` or ``prim_haloprop`` argument \n"
                   "to the ``mean_occupation`` function of the ``Zheng07Sats`` class.\n")
            raise HalotoolsError(msg)
        over_idxs = mass > np.max(self._prim_haloprop_vals)
        under_idxs = mass < np.min(self._prim_haloprop_vals)
        contained_indices = ~np.logical_or(under_idxs, over_idxs)  # indexs contained by the interpolator
        mean_ncen = np.zeros_like(mass)
        mean_ncen[over_idxs] = self._upper_occupation_bound
        mean_ncen[under_idxs] = self._lower_occupation_bound
        mean_ncen[contained_indices] = self._mean_occupation(np.log10(mass[contained_indices]))

        return mean_ncen

    def get_published_parameters(self):
        r"""
        No parameters for this model, so returns empty dict

        Parameters
        ----------
        None

        -------
        param_dict : dict
            Empty dictionary

        Examples
        --------
        >>> cen_model = TabulatedCens()
        >>> cen_model.param_dict = cen_model.get_published_parameters(cen_model.threshold)
        """

        return dict()


class AssembiasTabulatedCens(TabulatedCens, ContinuousAssembias):
    '''Tabulated Cens with Continuous Assembly bias'''

    def __init__(self, prim_haloprop_vals, cen_hod_vals, **kwargs):
        '''See halotools docs for more info. '''
        super(AssembiasTabulatedCens, self).__init__(prim_haloprop_vals, cen_hod_vals, **kwargs)

        sec_haloprop_key = 'halo_nfw_conc'
        if 'sec_haloprop_key' not in kwargs:
            kwargs['sec_haloprop_key'] = sec_haloprop_key

        ContinuousAssembias.__init__(self,
                                     lower_assembias_bound=self._lower_occupation_bound,
                                     upper_assembias_bound=self._upper_occupation_bound,
                                     method_name_to_decorate='mean_occupation',
                                     **kwargs)


class HSAssembiasTabulatedCens(TabulatedCens, HeavisideAssembias):
    '''Reddick14 Cens with Heaviside Assembly bias'''

    def __init__(self, prim_haloprop_vals, cen_hod_vals, **kwargs):
        '''See halotools docs for more info. '''
        super(HSAssembiasTabulatedCens, self).__init__(prim_haloprop_vals, cen_hod_vals, **kwargs)

        sec_haloprop_key = 'halo_nfw_conc'
        if 'sec_haloprop_key' not in kwargs:
            kwargs['sec_haloprop_key'] = sec_haloprop_key

        HeavisideAssembias.__init__(self,
                                    lower_assembias_bound=self._lower_occupation_bound,
                                    upper_assembias_bound=self._upper_occupation_bound,
                                    method_name_to_decorate='mean_occupation',
                                    **kwargs)


class FSAssembiasTabulatedCens(TabulatedCens, FreeSplitAssembias):
    '''Reddick14 Cens with Heaviside Assembly bias'''

    def __init__(self, prim_haloprop_vals, cen_hod_vals, **kwargs):
        '''See halotools docs for more info. '''
        super(FSAssembiasTabulatedCens, self).__init__(prim_haloprop_vals, cen_hod_vals, **kwargs)

        sec_haloprop_key = 'halo_nfw_conc'
        if 'sec_haloprop_key' not in kwargs:
            kwargs['sec_haloprop_key'] = sec_haloprop_key

        FreeSplitAssembias.__init__(self,
                                    lower_assembias_bound=self._lower_occupation_bound,
                                    upper_assembias_bound=self._upper_occupation_bound,
                                    method_name_to_decorate='mean_occupation',
                                    **kwargs)


class FSCAssembiasTabulatedCens(TabulatedCens, FreeSplitContinuousAssembias):
    '''Reddick14 Cens with Heaviside Assembly bias'''

    def __init__(self, prim_haloprop_vals, cen_hod_vals, **kwargs):
        '''See halotools docs for more info. '''
        super(FSCAssembiasTabulatedCens, self).__init__(prim_haloprop_vals, cen_hod_vals, **kwargs)

        sec_haloprop_key = 'halo_nfw_conc'
        if 'sec_haloprop_key' not in kwargs:
            kwargs['sec_haloprop_key'] = sec_haloprop_key

        FreeSplitContinuousAssembias.__init__(self,
                                              lower_assembias_bound=self._lower_occupation_bound,
                                              upper_assembias_bound=self._upper_occupation_bound,
                                              method_name_to_decorate='mean_occupation',
                                              **kwargs)


class CorrAssembiasTabulatedCens(TabulatedCens, CorrelationAssembias):
    '''Reddick14 Cens with Heaviside Assembly bias'''

    def __init__(self, prim_haloprop_vals, cen_hod_vals, **kwargs):
        '''See halotools docs for more info. '''
        super(CorrAssembiasTabulatedCens, self).__init__(prim_haloprop_vals, cen_hod_vals, **kwargs)

        sec_haloprop_key = 'halo_nfw_conc'
        if 'sec_haloprop_key' not in kwargs:
            kwargs['sec_haloprop_key'] = sec_haloprop_key

        CorrelationAssembias.__init__(self,
                                      lower_assembias_bound=self._lower_occupation_bound,
                                      upper_assembias_bound=self._upper_occupation_bound,
                                      method_name_to_decorate='mean_occupation',
                                      **kwargs)


class TabulatedSats(OccupationComponent):
    r""" Satellite ccupation that is fixed at observed values. Rather than being parameterized, populate with an HOD
    with a fixed, observed relationship.
    """

    def __init__(self, prim_haloprop_vals, sat_hod_vals,
                 threshold=model_defaults.default_luminosity_threshold,
                 prim_haloprop_key=model_defaults.prim_haloprop_key, **kwargs):
        r"""
        Parameters
        ----------
        prim_haloprop_vals: array
            Values of the prim haloprop that sat_hod_vals were observed

        sat_hod_vals: array
            The values of the hod observed at prim_haloprop_vals. Is interoplated to give the HOD

        threshold : float, optional
            Currently doesn't do anything. Will add in later.

        prim_haloprop_key : string, optional
            String giving the column name of the primary halo property governing
            the occupation statistics of gal_type galaxies.
            Default value is specified in the `~halotools.empirical_models.model_defaults` module.

        Examples
        --------
        >>> sat_model = TabulatedSats()
        >>> sat_model = TabulatedSats(threshold = -21)

        """
        upper_occupation_bound = float("inf")

        assert np.all(sat_hod_vals >= 0)
        assert sat_hod_vals.shape == prim_haloprop_vals.shape
        assert np.all(prim_haloprop_vals >= 0)

        self._prim_haloprop_vals = prim_haloprop_vals
        self._sat_hod_vals = sat_hod_vals

        self._mean_occupation = interp1d(np.log10(prim_haloprop_vals), sat_hod_vals, kind='cubic')

        # Call the super class constructor, which binds all the
        # arguments to the instance.
        super(TabulatedSats, self).__init__(
            gal_type='satellites', threshold=threshold,
            upper_occupation_bound=upper_occupation_bound,
            prim_haloprop_key=prim_haloprop_key, modulated_with_cenocc=False,
            **kwargs)

        self.param_dict = self.get_published_parameters()

        self.publications = []

    def mean_occupation(self, **kwargs):
        r"""Expected number of satellite galaxies in a halo of mass logM.

        Parameters
        ----------
        prim_haloprop : array, optional
            Array storing a mass-like variable that governs the occupation statistics.
            If ``prim_haloprop`` is not passed, then ``table``
            keyword arguments must be passed.

        table : object, optional
            Data table storing halo catalog.
            If ``table`` is not passed, then ``prim_haloprop``
            keyword arguments must be passed.

        Returns
        -------
        mean_nsat : float or array
            Mean number of satellite galaxies in a host halo of the specified mass.

        Examples
        --------
        The `mean_occupation` method of all OccupationComponent instances supports
        two different options for arguments. The first option is to directly
        pass the array of the primary halo property:

        >>> sat_model = TabulatedSats()
        >>> testmass = np.logspace(10, 15, num=50)
        >>> mean_nsat = sat_model.mean_occupation(prim_haloprop = testmass)

        The second option is to pass `mean_occupation` a full halo catalog.
        In this case, the array storing the primary halo property will be selected
        by accessing the ``sat_model.prim_haloprop_key`` column of the input halo catalog.
        For illustration purposes, we'll use a fake halo catalog rather than a
        (much larger) full one:

        >>> from halotools.sim_manager import FakeSim
        >>> fake_sim = FakeSim()
        >>> mean_nsat = sat_model.mean_occupation(table=fake_sim.halo_table)

        """

        # Retrieve the array storing the mass-like variable
        if 'table' in list(kwargs.keys()):
            mass = kwargs['table'][self.prim_haloprop_key]
        elif 'prim_haloprop' in list(kwargs.keys()):
            mass = np.atleast_1d(kwargs['prim_haloprop'])
        else:
            msg = ("\nYou must pass either a ``table`` or ``prim_haloprop`` argument \n"
                   "to the ``mean_occupation`` function of the ``Zheng07Sats`` class.\n")
            raise HalotoolsError(msg)

        over_idxs = mass > np.max(self._prim_haloprop_vals)
        under_idxs = mass < np.min(self._prim_haloprop_vals)
        contained_indices = ~np.logical_or(under_idxs, over_idxs)  # indexs contained by the interpolator
        mean_nsat = np.zeros_like(mass)
        mean_nsat[over_idxs] = self._sat_hod_vals[-1]  # not happy abotu this, no better guess
        mean_nsat[under_idxs] = self._lower_occupation_bound
        mean_nsat[contained_indices] = self._mean_occupation(np.log10(mass[contained_indices]))

        return mean_nsat

    def get_published_parameters(self):
        r"""
        No published best-fit paramaters for this model, so empty dict
        ----------
        None

        -------
        param_dict : dict
            Dictionary of model parameters whose values have been set to
            the values taken from Table 1 of Zheng et al. 2007.

        Examples
        --------
        >>> sat_model = TabulatedSats()
        >>> sat_model.param_dict = sat_model.get_published_parameters(sat_model.threshold)
        """

        return dict()


class AssembiasTabulatedSats(TabulatedSats, ContinuousAssembias):
    '''Tabulated Sats with Assembly bias'''

    def __init__(self, prim_haloprop_vals, sat_hod_vals, **kwargs):
        '''See halotools docs for more info. '''
        super(AssembiasTabulatedSats, self).__init__(prim_haloprop_vals, sat_hod_vals, cenocc_model=None, **kwargs)
        sec_haloprop_key = 'halo_nfw_conc'
        if 'sec_haloprop_key' not in kwargs:
            kwargs['sec_haloprop_key'] = sec_haloprop_key

        ContinuousAssembias.__init__(self,
                                     lower_assembias_bound=self._lower_occupation_bound,
                                     upper_assembias_bound=self._upper_occupation_bound,
                                     method_name_to_decorate='mean_occupation',
                                     **kwargs)


class HSAssembiasTabulatedSats(TabulatedSats, HeavisideAssembias):
    '''Tabulated Sats with Assembly bias'''

    def __init__(self, prim_haloprop_vals, sat_hod_vals, **kwargs):
        '''See halotools docs for more info. '''
        super(HSAssembiasTabulatedSats, self).__init__(prim_haloprop_vals, sat_hod_vals, cenocc_model=None, **kwargs)
        sec_haloprop_key = 'halo_nfw_conc'
        if 'sec_haloprop_key' not in kwargs:
            kwargs['sec_haloprop_key'] = sec_haloprop_key

        HeavisideAssembias.__init__(self,
                                    lower_assembias_bound=self._lower_occupation_bound,
                                    upper_assembias_bound=self._upper_occupation_bound,
                                    method_name_to_decorate='mean_occupation',
                                    **kwargs)


class FSAssembiasTabulatedSats(TabulatedSats, FreeSplitAssembias):
    '''Tabulated Sats with Assembly bias'''

    def __init__(self, prim_haloprop_vals, sat_hod_vals, **kwargs):
        '''See halotools docs for more info. '''
        super(FSAssembiasTabulatedSats, self).__init__(prim_haloprop_vals, sat_hod_vals, cenocc_model=None, **kwargs)
        sec_haloprop_key = 'halo_nfw_conc'
        if 'sec_haloprop_key' not in kwargs:
            kwargs['sec_haloprop_key'] = sec_haloprop_key

        FreeSplitAssembias.__init__(self,
                                    lower_assembias_bound=self._lower_occupation_bound,
                                    upper_assembias_bound=self._upper_occupation_bound,
                                    method_name_to_decorate='mean_occupation',
                                    **kwargs)


class CorrAssembiasTabulatedSats(TabulatedSats, CorrelationAssembias):
    '''Tabulated Sats with Assembly bias'''

    def __init__(self, prim_haloprop_vals, sat_hod_vals, **kwargs):
        '''See halotools docs for more info. '''
        super(CorrAssembiasTabulatedSats, self).__init__(prim_haloprop_vals, sat_hod_vals, cenocc_model=None, **kwargs)
        sec_haloprop_key = 'halo_nfw_conc'
        if 'sec_haloprop_key' not in kwargs:
            kwargs['sec_haloprop_key'] = sec_haloprop_key

        CorrelationAssembias.__init__(self,
                                      lower_assembias_bound=self._lower_occupation_bound,
                                      upper_assembias_bound=self._upper_occupation_bound,
                                      method_name_to_decorate='mean_occupation',
                                      **kwargs)


class FSCAssembiasTabulatedSats(TabulatedSats, FreeSplitContinuousAssembias):
    '''Tabulated Sats with Assembly bias'''

    def __init__(self, prim_haloprop_vals, sat_hod_vals, **kwargs):
        '''See halotools docs for more info. '''
        super(FSCAssembiasTabulatedSats, self).__init__(prim_haloprop_vals, sat_hod_vals, cenocc_model=None, **kwargs)
        sec_haloprop_key = 'halo_nfw_conc'
        if 'sec_haloprop_key' not in kwargs:
            kwargs['sec_haloprop_key'] = sec_haloprop_key

        FreeSplitContinuousAssembias.__init__(self,
                                              lower_assembias_bound=self._lower_occupation_bound,
                                              upper_assembias_bound=self._upper_occupation_bound,
                                              method_name_to_decorate='mean_occupation',
                                              **kwargs)


# considered subclassing but doesn't seem like i actually want that?
# TODO TabulatedHODComponent should be a class and Cens and Sats should be subclassed. 
class Tabulated2DCens(OccupationComponent):
    r""" Central occupation that is fixed at observed values. Rather than being parameterized, populate with an HOD
    with a fixed, observed relationship.
    """

    def __init__(self, prim_haloprop_bins, sec_haloprop_perc_bins, cen_hod_vals,
                 threshold=model_defaults.default_luminosity_threshold,
                 prim_haloprop_key=model_defaults.prim_haloprop_key,
                 sec_haloprop_key=model_defaults.sec_haloprop_key, **kwargs):
        r"""
        Parameters
        ----------
        prim_haloprop_bins: array
            Bin edges of the prim haloprop that cen_hod_vals were observed

        sec_haloprop_perc_bins: arrray
            Bin edges of the sec haloprop where cen_hod_vals were observeved

        cen_hod_vals: array
            The values of the hod observed at prim_haloprop_vals. Is interoplated to give the HOD.
            Must have shape (p,s) where p is the length of prim_haloprop_vals and s is the length of \
            sec_haloprop_vals

        threshold : float, optional
            Currently doesn't do anything. Will add in later.

        prim_haloprop_key : string, optional
            String giving the column name of the primary halo property governing
            the occupation statistics of gal_type galaxies.
            Default value is specified in the `~halotools.empirical_models.model_defaults` module.

        sec_haloprop_key : string, optional
            String giving the column name of the secondary halo property governing
            the occupation statistics of gal_type galaxies.
            Default value is specified in the `~halotools.empirical_models.model_defaults` module.


        Examples
        --------
        >>> cen_model = Tabulated2DCens()
        >>> cen_model = Tabulated2DCens(threshold = -21)
        """

        upper_occupation_bound = 1.0

        assert np.all(cen_hod_vals >= 0) and np.all(upper_occupation_bound >= cen_hod_vals)
        assert cen_hod_vals.shape[0] == prim_haloprop_bins.shape[0] - 1
        assert cen_hod_vals.shape[1] == sec_haloprop_perc_bins.shape[0] - 1
        assert np.all(prim_haloprop_bins >= 0)
        try:
            assert np.all(sec_haloprop_perc_bins >= 0) and np.all(sec_haloprop_perc_bins <= 1)
        except AssertionError:
            raise ("sec_haloprop_perc_bins are not between 0 and 1. Maybe you passed in sec_haloprop_bins instead?")

        self._prim_haloprop_bins = prim_haloprop_bins
        self._sec_haloprop_perc_bins = sec_haloprop_perc_bins
        self._cen_hod_vals = cen_hod_vals

        # Call the super class constructor, which binds all the
        # arguments to the instance.
        super(Tabulated2DCens, self).__init__(
            gal_type='centrals', threshold=threshold,
            upper_occupation_bound=upper_occupation_bound,
            prim_haloprop_key=prim_haloprop_key,
            sec_haloprop_key=sec_haloprop_key,
            **kwargs)

        self.param_dict = self.get_published_parameters()

        self.publications = []

        self.sec_haloprop_key = sec_haloprop_key  # wont be added automatically

    def mean_occupation(self, **kwargs):
        r"""Expected number of satellite galaxies in a halo of mass logM.
        See Equation 5 of arXiv:0308519.

        Parameters
        ----------
        prim_haloprop : array, optional
            Array storing a mass-like variable that governs the occupation statistics.
            If ``prim_haloprop`` is not passed, then ``table``
            keyword arguments must be passed.

        table : object, optional
            Data table storing halo catalog.
            If ``table`` is not passed, then ``prim_haloprop``
            keyword arguments must be passed.

        Returns
        -------
        mean_ncen : float or array
            Mean number of central galaxies in a host halo of the specified mass, from the above interpolation

        Examples
        --------
        The `mean_occupation` method of all OccupationComponent instances supports
        two different options for arguments. The first option is to directly
        pass the array of the primary halo property:

        >>> cen_model = Tabulated2DCens()
        >>> testmass = np.logspace(10, 15, num=50)
        >>> mean_ncen = cen_model.mean_occupation(prim_haloprop = testmass)

        The second option is to pass `mean_occupation` a full halo catalog.
        In this case, the array storing the primary halo property will be selected
        by accessing the ``sat_model.prim_haloprop_key`` column of the input halo catalog.
        For illustration purposes, we'll use a fake halo catalog rather than a
        (much larger) full one:

        >>> from halotools.sim_manager import FakeSim
        >>> fake_sim = FakeSim()
        >>> mean_ncen = cen_model.mean_occupation(table=fake_sim.halo_table)

        """

        # Retrieve the array storing the mass-like variable
        if 'table' in list(kwargs.keys()):
            table = kwargs['table']
            mass = table[self.prim_haloprop_key]
            if self.sec_haloprop_key + '_percentile' in list(table.keys()):
                sec_perc = table[self.sec_haloprop_key + '_percentile']
            else:
                # the value of sec_haloprop_percentile will be computed from scratch
                sec_perc = compute_conditional_percentiles(
                    prim_haloprop=mass,
                    sec_haloprop=table[self.sec_haloprop_key])
        else:
            try:
                mass = np.atleast_1d(kwargs['prim_haloprop'])
            except KeyError:
                msg = ("\nIf not passing an input ``table`` to the "
                       "``mean_occupation`` method,\n"
                       "you must pass ``prim_haloprop`` argument.\n")
                raise HalotoolsError(msg)
            try:
                sec_perc = np.atleast_1d(kwargs['sec_haloprop_percentile'])
            except KeyError:
                if 'sec_haloprop' not in kwargs:
                    msg = ("\nIf not passing an input ``table`` to the "
                           "``mean_occupation`` method,\n"
                           "you must pass either a ``sec_haloprop`` or "
                           "``sec_haloprop_percentile`` argument.\n")
                    raise HalotoolsError(msg)
                else:
                    sec_perc = compute_conditional_percentiles(
                        prim_haloprop=mass,
                        sec_haloprop=kwargs['sec_haloprop'])

        prim_over_idxs = mass > np.max(self._prim_haloprop_bins)
        prim_under_idxs = mass < np.min(self._prim_haloprop_bins)
        # prim_contained_indices = ~np.logical_or(prim_under_idxs, prim_over_idxs) #indexs contained by the interpolator
        contained_idxs = ~np.logical_or(prim_under_idxs, prim_over_idxs)  # indexs contained by the interpolator

        # sec_over_idxs = sec > np.max(self._sec_haloprop_vals)
        # sec_under_idxs = sec < np.min(self._sec_haloprop_vals)
        # sec_contained_indices = ~np.logical_or(sec_under_idxs, sec_over_idxs)  # indexs contained by the interpolator

        # over_idxs = np.logical_or(prim_over_idxs, sec_over_idxs)
        # under_idxs = np.logical_or(prim_under_idxs, sec_under_idxs)

        # contained_indices = ~np.logical_or(under_idxs, over_idxs)

        mean_ncen = np.zeros_like(mass)
        mean_ncen[prim_over_idxs] = self._upper_occupation_bound
        mean_ncen[prim_under_idxs] = self._lower_occupation_bound

        prim_haloprop_idxs = np.digitize(mass, self._prim_haloprop_bins, right=True) - 1
        sec_haloprop_idxs = np.digitize(sec_perc, self._sec_haloprop_perc_bins, right=True) - 1

        mean_ncen[contained_idxs] = self._cen_hod_vals[
            prim_haloprop_idxs[contained_idxs], sec_haloprop_idxs[contained_idxs]]

        # TODO dunno how i feel about this..
        # mean_sec = np.mean(sec[contained_indices])
        # mean_ncen[sec_over_idxs] = self._mean_occupation(np.log10(mass[sec_over_idxs]), mean_sec*np.ones_like(sec_over_idxs) )
        # mean_ncen[sec_under_idxs] = self._mean_occupation(np.log10(mass[sec_under_idxs]), mean_sec*np.ones_like(sec_under_idxs) )

        return mean_ncen

    def get_published_parameters(self):
        r"""
        No parameters for this model, so returns empty dict

        Parameters
        ----------
        None

        -------
        param_dict : dict
            Empty dictionary

        Examples
        --------
        >>> cen_model = TabulatedCens()
        >>> cen_model.param_dict = cen_model.get_published_parameters(cen_model.threshold)
        """

        return dict()


class Tabulated2DSats(OccupationComponent):
    r""" Satellite ccupation that is fixed at observed values. Rather than being parameterized, populate with an HOD
    with a fixed, observed relationship.
    """

    def __init__(self, prim_haloprop_bins, sec_haloprop_perc_bins, sat_hod_vals,
                 threshold=model_defaults.default_luminosity_threshold,
                 prim_haloprop_key=model_defaults.prim_haloprop_key,
                 sec_haloprop_key=model_defaults.sec_haloprop_key, **kwargs):
        r"""
        Parameters
        ----------
        prim_haloprop_bins: array
            Bin edges of the prim haloprop that sat_hod_vals were observed

        sec_haloprop_perc_bins:
            Bin edges of the sec haloprop percentile that sat_hod_vals were observed

        sat_hod_vals: array
            The values of the hod observed at prim_haloprop_vals. Is interoplated to give the HOD

        threshold : float, optional
            Currently doesn't do anything. Will add in later.

        prim_haloprop_key : string, optional
            String giving the column name of the primary halo property governing
            the occupation statistics of gal_type galaxies.
            Default value is specified in the `~halotools.empirical_models.model_defaults` module.

        Examples
        --------
        >>> sat_model = TabulatedSats()
        >>> sat_model = TabulatedSats(threshold = -21)

        """
        upper_occupation_bound = float("inf")

        assert np.all(sat_hod_vals >= 0) and np.all(upper_occupation_bound >= sat_hod_vals)
        assert sat_hod_vals.shape[0] == prim_haloprop_bins.shape[0] - 1
        assert sat_hod_vals.shape[1] == sec_haloprop_perc_bins.shape[0] - 1
        assert np.all(prim_haloprop_bins >= 0)
        try:
            assert np.all(sec_haloprop_perc_bins >= 0) and np.all(sec_haloprop_perc_bins <= 1)
        except AssertionError:
            raise ("sec_haloprop_perc_bins are not between 0 and 1. Maybe you passed in sec_haloprop_bins instead?")

        self._prim_haloprop_bins = prim_haloprop_bins
        self._sec_haloprop_perc_bins = sec_haloprop_perc_bins
        self._sat_hod_vals = sat_hod_vals

        # Call the super class constructor, which binds all the
        # arguments to the instance.
        super(Tabulated2DSats, self).__init__(
            gal_type='satellites', threshold=threshold,
            upper_occupation_bound=upper_occupation_bound,
            prim_haloprop_key=prim_haloprop_key,
            sec_haloprop_key=sec_haloprop_key,
            **kwargs)

        self.param_dict = self.get_published_parameters()

        self.publications = []

        self.sec_haloprop_key = sec_haloprop_key  # wont be added automatically

    def mean_occupation(self, **kwargs):
        r"""Expected number of satellite galaxies in a halo of mass logM.

        Parameters
        ----------
        prim_haloprop : array, optional
            Array storing a mass-like variable that governs the occupation statistics.
            If ``prim_haloprop`` is not passed, then ``table``
            keyword arguments must be passed.

        table : object, optional
            Data table storing halo catalog.
            If ``table`` is not passed, then ``prim_haloprop``
            keyword arguments must be passed.

        Returns
        -------
        mean_nsat : float or array
            Mean number of satellite galaxies in a host halo of the specified mass.

        Examples
        --------
        The `mean_occupation` method of all OccupationComponent instances supports
        two different options for arguments. The first option is to directly
        pass the array of the primary halo property:

        >>> sat_model = TabulatedSats()
        >>> testmass = np.logspace(10, 15, num=50)
        >>> mean_nsat = sat_model.mean_occupation(prim_haloprop = testmass)

        The second option is to pass `mean_occupation` a full halo catalog.
        In this case, the array storing the primary halo property will be selected
        by accessing the ``sat_model.prim_haloprop_key`` column of the input halo catalog.
        For illustration purposes, we'll use a fake halo catalog rather than a
        (much larger) full one:

        >>> from halotools.sim_manager import FakeSim
        >>> fake_sim = FakeSim()
        >>> mean_nsat = sat_model.mean_occupation(table=fake_sim.halo_table)

        """
        # Retrieve the array storing the mass-like variable
        if 'table' in list(kwargs.keys()):
            table = kwargs['table']
            mass = table[self.prim_haloprop_key]
            if self.sec_haloprop_key + '_percentile' in list(table.keys()):
                sec_perc = table[self.sec_haloprop_key + '_percentile']
            else:
                # the value of sec_haloprop_percentile will be computed from scratch
                sec_perc = compute_conditional_percentiles(
                    prim_haloprop=mass,
                    sec_haloprop=table[self.sec_haloprop_key])
        else:
            try:
                mass = np.atleast_1d(kwargs['prim_haloprop'])
            except KeyError:
                msg = ("\nIf not passing an input ``table`` to the "
                       "``mean_occupation`` method,\n"
                       "you must pass ``prim_haloprop`` argument.\n")
                raise HalotoolsError(msg)
            try:
                sec_perc = np.atleast_1d(kwargs['sec_haloprop_percentile'])
            except KeyError:
                if 'sec_haloprop' not in kwargs:
                    msg = ("\nIf not passing an input ``table`` to the "
                           "``mean_occupation`` method,\n"
                           "you must pass either a ``sec_haloprop`` or "
                           "``sec_haloprop_percentile`` argument.\n")
                    raise HalotoolsError(msg)
                else:
                    sec_perc = compute_conditional_percentiles(
                        prim_haloprop=mass,
                        sec_haloprop=kwargs['sec_haloprop'])

        prim_over_idxs = mass > np.max(self._prim_haloprop_bins)
        prim_under_idxs = mass < np.min(self._prim_haloprop_bins)
        # prim_contained_indices = ~np.logical_or(prim_under_idxs, prim_over_idxs) #indexs contained by the interpolator
        contained_idxs = ~np.logical_or(prim_under_idxs, prim_over_idxs)  # indexs contained by the interpolator

        # sec_over_idxs = sec > np.max(self._sec_haloprop_vals)
        # sec_under_idxs = sec < np.min(self._sec_haloprop_vals)
        # sec_contained_indices = ~np.logical_or(sec_under_idxs, sec_over_idxs)  # indexs contained by the interpolator

        # over_idxs = np.logical_or(prim_over_idxs, sec_over_idxs)
        # under_idxs = np.logical_or(prim_under_idxs, sec_under_idxs)

        # contained_indices = ~np.logical_or(under_idxs, over_idxs)

        mean_nsat = np.zeros_like(mass)
        mean_nsat[prim_over_idxs] = self._sat_hod_vals[-1, :].mean()  # not happy abotu this, no better guess
        mean_nsat[prim_under_idxs] = self._lower_occupation_bound

        prim_haloprop_idxs = np.digitize(mass, self._prim_haloprop_bins, right=True) - 1
        sec_haloprop_idxs = np.digitize(sec_perc, self._sec_haloprop_perc_bins, right=True) - 1
        mean_nsat[contained_idxs] = self._sat_hod_vals[
            prim_haloprop_idxs[contained_idxs], sec_haloprop_idxs[contained_idxs]]

        # TODO dunno how i feel about this..
        # mean_sec = np.mean(sec[contained_indices])
        # mean_ncen[sec_over_idxs] = self._mean_occupation(np.log10(mass[sec_over_idxs]), mean_sec*np.ones_like(sec_over_idxs) )
        # mean_ncen[sec_under_idxs] = self._mean_occupation(np.log10(mass[sec_under_idxs]), mean_sec*np.ones_like(sec_under_idxs) )

        return mean_nsat

    def get_published_parameters(self):
        r"""
        No published best-fit paramaters for this model, so empty dict
        ----------
        None

        -------
        param_dict : dict
            Dictionary of model parameters whose values have been set to
            the values taken from Table 1 of Zheng et al. 2007.

        Examples
        --------
        >>> sat_model = TabulatedSats()
        >>> sat_model.param_dict = sat_model.get_published_parameters(sat_model.threshold)
        """

        return dict()

HOD_DICT = {'abZheng07': (AssembiasZheng07Cens, AssembiasZheng07Sats), 
            'hsabZheng07': (HSAssembiasZheng07Cens, HSAssembiasZheng07Sats), 
            'fsabZheng07': (FSAssembiasZheng07Cens, FSAssembiasZheng07Sats),
            'fscabZheng07': (FSCAssembiasZheng07Cens, FSCAssembiasZheng07Sats),
            'corrZheng07': (CorrAssembiasZheng07Cens, CorrAssembiasZheng07Sats),
            'redMagic': (RedMagicCens, RedMagicSats), 
            'abRedMagic': (AssembiasRedMagicCens, AssembiasRedMagicSats),
            'hsabRedMagic': (HSAssembiasRedMagicCens, HSAssembiasRedMagicSats),
             'fsabRedMagic': (FSAssembiasRedMagicCens, FSAssembiasRedMagicSats ),
             'fscabRedMagic': (FSCAssembiasRedMagicCens, FSCAssembiasRedMagicSats ),
             'corrRedMagic': (CorrAssembiasRedMagicCens, CorrAssembiasRedMagicSats ),
             'reddick14': (Reddick14Cens, Reddick14Sats), # no modulation
             'hsabReddick14': (HSAssembiasReddick14Cens, HSAssembiasReddick14Sats), # no modulation
             'fsabReddick14': (FSAssembiasReddick14Cens, FSAssembiasReddick14Sats), # no modulation
             #'fscabReddick14': (FSCAssembiasReddick14Cens, FSCAssembiasReddick14Sats), # no modulation
             'corrReddick14': (CorrAssembiasReddick14Cens, CorrAssembiasReddick14Sats) , # no modulation
             'abReddick14': (AssembiasReddick14Cens, AssembiasReddick14Sats), # no modulation
             'tabulated': (TabulatedCens, TabulatedSats), # no modulation
             'hsabTabulated': (HSAssembiasTabulatedCens, HSAssembiasTabulatedSats),  # no modulation
             'abTabulated': (AssembiasTabulatedCens, AssembiasTabulatedSats), # no modulation
             'fsabTabulated': (FSAssembiasTabulatedCens, FSAssembiasTabulatedSats ), # no modulation
             'fscabTabulated': (FSCAssembiasTabulatedCens, FSCAssembiasTabulatedSats ), # no modulation
             'corrTabulated': (CorrAssembiasTabulatedCens, CorrAssembiasTabulatedSats ), # no modulation
             'tabulated2D': (Tabulated2DCens, Tabulated2DSats),  # no modulation
             'stepFunc': (StepFuncCens, StepFuncSats),
             'reddick14+redMagic': (Reddick14Cens, RedMagicSats)}

