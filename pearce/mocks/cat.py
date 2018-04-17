#!/bin/bash
# TODO better docstring
'''Module containing the main Cat object, which implements caching, loading, populating, and calculating observables
on catalogs. '''

from os import path
from itertools import izip
from multiprocessing import cpu_count
import inspect
import warnings
from time import time
from glob import glob
from multiprocessing import Pool

import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import linregress

from astropy import cosmology
from astropy import constants as const, units as units
from halotools.sim_manager import RockstarHlistReader, CachedHaloCatalog, UserSuppliedPtclCatalog
from halotools.empirical_models import PrebuiltHodModelFactory
from halotools.empirical_models import HodModelFactory, TrivialPhaseSpace, NFWPhaseSpace
from halotools.mock_observables import *  # i'm importing so much this is just easier

from .customHODModels import *

# try to import corrfunc, determine if it was successful
try:
    from Corrfunc.theory import xi, wp, DD
    from Corrfunc.utils import convert_3d_counts_to_cf

    CORRFUNC_AVAILABLE = True
except ImportError:
    CORRFUNC_AVAILABLE = False

VALID_HODS = {'redMagic', 'hsabRedMagic','abRedMagic','fsabRedMagic', 'fscabRedMagic','corrRedMagic',\
              'reddick14','hsabReddick14','abReddick14','fsabReddick14', 'fscabReddick14','corrReddick14','stepFunc',\
              'zheng07', 'leauthaud11', 'tinker13', 'hearin15', 'reddick14+redMagic', 'tabulated',\
              'hsabTabulated', 'abTabulated','fsabTabulated','fscabTabulated','corrTabulated', 'tabulated2D'}

DEFAULT_HODS = {'zheng07', 'leauthaud11', 'tinker13', 'hearin15'}


def observable(func):
    '''
    Decorator for observable methods. Checks that the catalog is properly loaded and calcualted.
    :param func:
        Function that calculates an observable on a populated catalog.
    :return: func
    '''

    def _func(self, *args, **kwargs):

        if 'use_corrfunc' in kwargs and (kwargs['use_corrfunc'] and not CORRFUNC_AVAILABLE):
            # NOTE just switch to halotools in this case? Or fail?
            raise ImportError("Corrfunc is not available on this machine!")

        try:
            assert self.halocat is not None
            assert self.model is not None
            assert self.populated_once
        except AssertionError:
            raise AssertionError("The cat must have a loaded model and catalog and be populated before calculating an observable.")
        return func(self, *args, **kwargs)

    # store the arguments, as the decorator destroys the spec
    _func.args = inspect.getargspec(func)[0]
    _func.__doc__ = func.__doc__

    return _func


class Cat(object):
    def __init__(self, simname='cat', loc='', filenames=[], cache_loc='/u/ki/swmclau2/des/halocats/',
                 columns_to_keep={}, halo_finder='rockstar', version_name='most_recent',
                 Lbox=1.0, pmass=1.0, scale_factors=[], cosmo=cosmology.WMAP5, gadget_loc='',
                 **kwargs):

        # TODO need both loc and filenames?
        # TODO emply filenames does a glob?
        # TODO change 'loc' to dir?
        '''
        The main object controlling manipulation and actions on catalogs.

        Many of these parameters are taken care of by subclasses; this is the most general version.
        :param simname:
            Name of the cosmology. Generally same as the object.
        :param loc:
            Directory where halo catalogs are stored.
        :param filenames:
            Names of the files available/ of interest in loc. Interchangeable use-wise with filenames.
        :param cache_loc:
            The directory to cache the halocats for halotools.
        :param columns_to_keep:
            Which columns in halo catalog to keep. For more details, see halotools documentation.
        :param halo_finder:
            Which halo finder was used. For more details, see halotools documentation.
        :param version_name:
            The version name of the catalog. For more details, see halotools documentation.
        :param scale_factors:
            What scale factors are available/ of interest for this cosmology. Interchangeable use-wise with filenames.
        :param Lbox:
            Size of the box for this simulation.
        :param pmass:
            Mass of a single dark matter particle in the simulation.
        :param cosmo:
            An astropy.cosmology object with the cosmology of this simulation.
        '''

        # relevant information for Halotools
        self.simname = simname
        self.loc = loc
        self.columns_to_keep = columns_to_keep

        # TODO allow the user access to this? Probably.
        self.columns_to_convert = set(["halo_rvir", "halo_rs","halo_rs_klypin", "halo_r200b"])
        self.columns_to_convert = list(self.columns_to_convert & set(self.columns_to_keep.keys()))

        self.halo_finder = halo_finder
        self.version_name = version_name
        self.Lbox = Lbox
        self.pmass = pmass

        self.scale_factors = np.array(sorted(scale_factors))
        self.redshifts = [1.0 / a - 1 for a in
                          self.scale_factors]  # TODO let user pass in redshift and get a scale factor

        self.cosmology = cosmo  # default cosmology
        self.h = self.cosmology.H(0).value / 100.0

        self.gadget_loc = gadget_loc  # location of original gadget files.

        if not hasattr(self, 'prim_haloprop_key'):
            self.prim_haloprop_key = 'halo_mvir'

        # TODO Well this makes me think loc doesn't do anything...
        # I use it in the subclasses though. Doesn't mean I need it here though.
        # This one doesn't need to be easy to use; just general.
        self.filenames = sorted(filenames)
        for i, fname in enumerate(self.filenames):
            self.filenames[i] = self.loc + fname

        # Confirm filenames and scale_factors have the same length
        # halotools builtins have no filenames, so there is a case filenames = []
        assert (len(self.filenames) == len(self.redshifts)) or len(self.filenames) == 0

        self.cache_filenames = [path.join(cache_loc, 'hlist_%.2f.list.%s.hdf5' % (a, self.simname)) \
                                for a in self.scale_factors]

        self.halocat = None  # halotools halocat that we wrap
        self.model = None  # same as above, but for the model
        self.populated_once = False

    def __len__(self):
        '''Number of separate catalogs contained in one object. '''
        return len(self.scale_factors)

    def __str__(self):
        '''Return an informative output string.'''
        # TODO Some things could be removed, others added
        output = []
        output.append(self.simname)
        output.append('-' * 25)
        output.append('Halo finder:\t %s' % self.halo_finder)
        output.append('Version name:\t%s' % self.version_name)
        output.append('Cosmology:\n%s' % self.cosmology)
        output.append('Redshifts:\t%s' % str(self.redshifts))
        output.append('-' * 25)
        output.append('Location:\t%s' % self.loc)
        output.append('Lbox:\t%.1f' % self.Lbox)
        output.append('Particle Mass:\t%f' % self.pmass)
        output.append('Columns to Keep:\n%s' % str(self.columns_to_keep))

        return '\n'.join(output)

    def _update_lists(self, user_kwargs, tmp_fnames, tmp_scale_factors):
        '''If the user passes in a scale factor or filename, we have to do some cropping to ensure they align.
        Used by subclasses during initialization.'''

        sf_idxs = []  # store the indicies, as they have applications elsewhere

        if 'filenames' in user_kwargs and 'scale_factors' in user_kwargs:
            assert len(user_kwargs['filenames']) == len(user_kwargs['scale_factors'])
            for kw_fname in user_kwargs['filenames']:
                # Store indicies. If not in, will throw and error.
                sf_idxs.append(tmp_fnames.index(kw_fname))
                # do nothing, we're good.
        elif 'scale_factors' in user_kwargs:
            user_kwargs['filenames'] = []
            # TODO be able to smartly round inputs 
            for a in user_kwargs['scale_factors']:
                idx = tmp_scale_factors.index(a)  # will raise an error if it's not there
                sf_idxs.append(idx)
                user_kwargs['filenames'].append(tmp_fnames[idx])  # get teh matching scale factor
        elif 'filenames' in user_kwargs:
            user_kwargs['scale_factors'] = []
            for kw_fname in user_kwargs['filenames']:
                idx = tmp_fnames.index(kw_fname)  # will throw an error if not in there.
                sf_idxs.append(idx)
                user_kwargs['scale_factors'].append(tmp_scale_factors[idx])  # get teh matching scale factor
        else:
            user_kwargs['filenames'] = tmp_fnames
            user_kwargs['scale_factors'] = tmp_scale_factors
            sf_idxs = range(len(tmp_scale_factors))

        self.sf_idxs = np.array(sf_idxs)



    def _return_nearest_sf(self, a, tol=0.05):
        '''
        Find the nearest scale factor to a that is stored, within tolerance. If none exists, return None.
        :param a:
            Scale factor to find nearest neighbor to.
        :param tol:
            Tolerance within which "near" is defined. Default is 0.05
        :return: If a nearest scale factor is found, returns it. Else, returns None.
        '''
        assert 0 < a <= 1  # assert a valid scale factor
        if a in self.scale_factors:  # try for an exact match.
            return a
        idx = np.argmin(np.abs(np.array(self.scale_factors) - a))
        if np.abs(self.scale_factors[idx] - a) < tol:
            return self.scale_factors[idx]
        else:
            return None

    def _check_cores(self, n_cores):
        '''
        Helper function that checks that a user's input for the number of cores is sensible,
        returns modifications and issues warnings as necessary.
        :param n_cores:
            User input for number of cores
        :return:
            n_cores: A sensible number of cores given context and requirements.
        '''

        assert n_cores == 'all' or n_cores > 0
        if type(n_cores) is not str:
            assert int(n_cores) == n_cores

        max_cores = cpu_count()
        if n_cores == 'all':
            return max_cores
        elif n_cores > max_cores:
            warnings.warn('n_cores invalid. Changing from %d to maximum %d.' % (n_cores, max_cores))
            return max_cores
            # else, we're good!
        else:
            return n_cores

    def cache(self, scale_factors='all', overwrite=False, add_local_density=False, add_particles = False,downsample_factor = 1e-3):
        '''
        Cache a halo catalog in the halotools format, for later use.
        :param scale_factors:
            Scale factors to cache. Default is 'all', which will cache ever scale_factor stored.
        :param overwrite:
            Overwrite the file currently on disk, if it exists. Default is false.
        :return: None
        '''
        try:
            assert not add_local_density or self.gadget_loc  # gadget loc must be nonzero here!
            assert not add_particles or self.gadget_loc
        except:
            raise AssertionError('Particle location not specified; please specify gadget location for %s' % self.simname)

        if add_local_density or add_particles:
            all_snapdirs = glob(path.join(self.gadget_loc, 'snapdir*'))
            snapdirs = [all_snapdirs[idx] for idx in self.sf_idxs]#only the snapdirs for the scale factors were interested in .
        else:
            snapdirs = ['' for i in self.scale_factors]

        for a, z, fname, cache_fnames, snapdir in izip(self.scale_factors, self.redshifts, self.filenames, self.cache_filenames, snapdirs):
            # TODO get right reader for each halofinder.
            if scale_factors != 'all' and a not in scale_factors:
                continue
            reader = RockstarHlistReader(fname, self.columns_to_keep, cache_fnames, self.simname,
                                         self.halo_finder, z, self.version_name, self.Lbox, self.pmass,
                                         overwrite=overwrite)
            reader.read_halocat(self.columns_to_convert)
            if add_local_density or add_particles:
                particles = self._read_particles(snapdir, downsample_factor=downsample_factor)
                if add_local_density:
                    self.add_local_density(reader, particles)  # TODO how to add radius?

            reader.write_to_disk()  # do these after so we have a halo table to work off of
            reader.update_cache_log()

            if add_particles:
                self.cache_particles(particles, a, downsample_factor=downsample_factor)

    def _read_particles(self, snapdir, downsample_factor):
        """
        Read in particles from a snapshot, and return them.
        :param snapdir:
            location of hte particles
        :param downsample_factor:
            The amount by which to downsample the particles. Default is 1e-3
        :return: all_particles, a numpy arrany of shape (N,3) that lists all particle positions.
        """
        from .readGadgetSnapshot import readGadgetSnapshot
        assert 0<= downsample_factor <=1
        np.random.seed(0)
        all_particles = np.array([], dtype='float32')
        # TODO should fail gracefully if memory is exceeded or if p is too small.
        for file in glob(path.join(snapdir, 'snapshot*')):
            # TODO should find out which is "fast" axis and use that.
            # Numpy uses fortran ordering.
            particles = readGadgetSnapshot(file, read_pos=True)[1]  # Think this returns some type of tuple; should check
            downsample_idxs = np.random.choice(particles.shape[0], size=int(particles.shape[0] * downsample_factor))
            particles = particles[downsample_idxs, :]
            # particles = particles[np.random.rand(particles.shape[0]) < p]  # downsample
            if particles.shape[0] == 0:
                continue

            all_particles = np.resize(all_particles, (all_particles.shape[0] + particles.shape[0], 3))
            all_particles[-particles.shape[0]:, :] = particles

        return all_particles

    def cache_particles(self,particles, scale_factor, downsample_factor ):
        """
        Add the particle to the halocatalog, so loading it will load the corresponding particles.
        :param particles:
            A (N,3) shaped numpy array of all particle positions
        """
        z = 1.0/scale_factor-1.0
        ptcl_catalog = UserSuppliedPtclCatalog(redshift=z, Lbox=self.Lbox, particle_mass=self.pmass, \
                                               x=particles[:, 0], y=particles[:, 1], z=particles[:, 2])
        ptcl_cache_loc = '/u/ki/swmclau2/des/halocats/ptcl_%.2f.list.%s_%s.hdf5'
        ptcl_cache_filename = ptcl_cache_loc % (scale_factor, self.simname, self.version_name)  # make sure we don't have redunancies.
        ptcl_catalog.add_ptclcat_to_cache(ptcl_cache_filename, self.simname, self.version_name+'_particle_%.2f'%(-1*np.log10(downsample_factor)), str(downsample_factor),overwrite=True)#TODO would be nice to make a note of the downsampling without having to do some voodoo to get it.


    def add_local_density(self, reader, all_particles, radius=[1, 5]):#[1,5,10]
        """
        Calculates the local density around each halo and adds it to the halo table, to be cached.
        :param reader:
            A RockstartHlistReader object from Halotools.
        :param snapdir:
            Gadget snapshot corresponding with this rockstar file.
        :param radius:
            Radius (in Mpc/h) around which to search for particles to estimate locas density. Default is 5 Mpc.
        :return: None
        """
        #doing imports here since these are both files i've stolen from Yao
        #Possible this will be slow

        from fast3tree import fast3tree

        if type(radius) == float:
            radius = np.array([radius])
        elif type(radius) == list:
            radius = np.array(radius)
        densities = np.ones((reader.halo_table['halo_x'].shape[0], radius.shape[0]))

        with fast3tree(all_particles) as tree:
            for r_idx, r in enumerate(radius):
                densities[:, r_idx] = densities[:, r_idx]/ (p * 4 * np.pi / 3 * r ** 3)
                for idx, halo_pos in enumerate(
                        izip(reader.halo_table['halo_x'], reader.halo_table['halo_y'], reader.halo_table['halo_z'])):
                    particle_idxs = tree.query_radius(halo_pos, r, periodic=True)
                    densities[idx,r_idx] *= reader.particle_mass * len(particle_idxs)

            reader.halo_table['halo_local_density_%d'%(int(r))] = densities[:, r_idx]
    # adding **kwargs cuz some invalid things can be passed in, hopefully not a pain
    # TODO some sort of spell check in the input file
    def load(self, scale_factor, HOD='redMagic', tol=0.05,particles=False,downsample_factor=1e-3, hod_kwargs = {}, **kwargs):
        '''
        Load both a halocat and a model to prepare for population and calculation.
        :param scale_factor:
            Scale factor of the catalog to load. If no exact match found, searches for nearest in tolerance.
        :param HOD:
            HOD model to load. Currently available options are redMagic, stepFunc, and the halotools defatuls.
        :return: None
        '''
        a = self._return_nearest_sf(scale_factor, tol)
        if a is None:
            raise ValueError('Scale factor %.3f not within given tolerance.' % scale_factor)
        self.load_catalog(a, tol, check_sf=False, particles=particles, downsample_factor=downsample_factor)
        self.load_model(a, HOD, check_sf=False, hod_kwargs=hod_kwargs)

    def load_catalog(self, scale_factor, tol=0.05, check_sf=True, particles = False, downsample_factor = 1e-3):
        '''
        Load only a specific catalog. Not reccomended to use separately from broader load function.
        Its possible for the redshift ni the model to be different fro the one from the catalog,
        though the difference will be small.
        :param a:
            The scale factor of the catalog of interest
        :param check_sf:
            Boolean whether or not to use the passed in scale_factor blindly. Default is false.
        :return: None
        '''
        if check_sf:
            a = self._return_nearest_sf(scale_factor, tol)
            if a is None:
                raise ValueError('Scale factor %.3f not within given tolerance.' % scale_factor)
        else:
            a = scale_factor  # YOLO
        z = 1.0 / a - 1
        if not particles:
            self.halocat = CachedHaloCatalog(simname=self.simname, halo_finder=self.halo_finder,
                                             version_name=self.version_name,
                                            redshift=z, dz_tol=0.01)
        else:
            self._downsample_factor = downsample_factor
            self.halocat = CachedHaloCatalog(simname=self.simname, halo_finder=self.halo_finder,
                                         version_name=self.version_name,
                                         ptcl_version_name=self.version_name+'_particle_%.2f'%(-1*np.log10(downsample_factor)),
                                            redshift=z, dz_tol = 0.01)
        # refelct the current catalog
        self.z = z
        self.a = a

    # TODO not sure if assembias should be boolean, or keep it as separate HODs?
    def load_model(self, scale_factor, HOD='redMagic', check_sf=True, hod_kwargs={}):
        '''
        Load an HOD model. Not reccomended to be used separately from the load function. It
        is possible for the scale_factor of the model and catalog to be different.
        :param scale_factor:
            Scale factor for the model
        :param HOD:
            HOD model to load. Currently available options are redMagic, stepFunc, and the halotools defatuls.
            Also may pass in a tuple of cens and sats classes, which will be instantiated here.
        :param check_sf:
            Boolean whether or not to use the passed in scale_factor blindly. Default is false.
        :param hod_kwargs:
            Kwargs to pass into the HOD model being loaded. Default is none.
        :return: None
        '''

        if check_sf:
            a = self._return_nearest_sf(scale_factor)
            if a is None:
                raise ValueError('Scale factor %.3f not within given tolerance.' % scale_factor)
        else:
            a = scale_factor  # YOLO
        z = 1.0 / a - 1
        if type(HOD) is str:
            assert HOD in VALID_HODS

            if HOD in  VALID_HODS-DEFAULT_HODS: # my custom ones
                # TODO this should be a dict of tuples, maybe two to define the modulation behavior
                if HOD == 'redMagic':
                    cens_occ = RedMagicCens(redshift=z, **hod_kwargs)
                    sats_occ = RedMagicSats(redshift=z, cenocc_model=cens_occ, **hod_kwargs)
                # the ab ones need to modulated with the baseline model
                elif HOD == 'abRedMagic':
                    cens_occ = AssembiasRedMagicCens(redshift=z, **hod_kwargs)
                    sats_occ = AssembiasRedMagicSats(redshift=z, cenocc_model=cens_occ, **hod_kwargs)
                elif HOD == 'hsabRedMagic':
                    cens_occ = HSAssembiasRedMagicCens(redshift=z, **hod_kwargs)
                    sats_occ = HSAssembiasRedMagicSats(redshift=z, cenocc_model=cens_occ, **hod_kwargs)
                elif HOD == 'fsabRedMagic':
                    cens_occ = FSAssembiasRedMagicCens(redshift=z, **hod_kwargs)
                    sats_occ = FSAssembiasRedMagicSats(redshift=z, cenocc_model=cens_occ, **hod_kwargs)
                elif HOD == 'fscabRedMagic':
                    cens_occ = FSCAssembiasRedMagicCens(redshift=z, **hod_kwargs)
                    sats_occ = FSCAssembiasRedMagicSats(redshift=z, cenocc_model=cens_occ, **hod_kwargs)
                elif HOD == 'corrRedMagic':
                    cens_occ = CorrAssembiasRedMagicCens(redshift=z, **hod_kwargs)
                    sats_occ = CorrAssembiasRedMagicSats(redshift=z, cenocc_model=cens_occ, **hod_kwargs)
                elif HOD == 'reddick14':
                    cens_occ = Reddick14Cens(redshift=z, **hod_kwargs)
                    sats_occ = Reddick14Sats(redshift=z, **hod_kwargs) # no modulation
                elif HOD == 'hsabReddick14':
                    cens_occ = HSAssembiasReddick14Cens(redshift=z, **hod_kwargs)
                    sats_occ = HSAssembiasReddick14Sats(redshift=z, **hod_kwargs) # no modulation
                elif HOD == 'fsabReddick14':
                    cens_occ = FSAssembiasReddick14Cens(redshift=z, **hod_kwargs)
                    sats_occ = FSAssembiasReddick14Sats(redshift=z, **hod_kwargs) # no modulation
                elif HOD == 'fscabReddick14':
                    cens_occ = FSCAssembiasReddick14Cens(redshift=z, **hod_kwargs)
                    sats_occ = FSCAssembiasReddick14Sats(redshift=z, **hod_kwargs) # no modulation
                elif HOD == 'corrReddick14':
                    cens_occ = CorrAssembiasReddick14Cens(redshift=z, **hod_kwargs)
                    sats_occ = CorrAssembiasReddick14Sats(redshift=z, **hod_kwargs) # no modulation
                elif HOD == 'abReddick14':
                    cens_occ = AssembiasReddick14Cens(redshift=z, **hod_kwargs)
                    sats_occ = AssembiasReddick14Sats(redshift=z, **hod_kwargs) # no modulation
                elif HOD == 'tabulated':
                    cens_occ = TabulatedCens(redshift=z, **hod_kwargs)
                    sats_occ = TabulatedSats(redshift=z,**hod_kwargs) # no modulation
                elif HOD == 'hsabTabulated':
                    cens_occ = HSAssembiasTabulatedCens(redshift=z, **hod_kwargs)
                    sats_occ = HSAssembiasTabulatedSats(redshift=z, **hod_kwargs)  # no modulation
                elif HOD == 'abTabulated':
                    cens_occ = AssembiasTabulatedCens(redshift=z, **hod_kwargs)
                    sats_occ = AssembiasTabulatedSats(redshift=z,**hod_kwargs) # no modulation
                elif HOD == 'fsabTabulated':
                    cens_occ = FSAssembiasTabulatedCens(redshift=z, **hod_kwargs)
                    sats_occ = FSAssembiasTabulatedSats(redshift=z,**hod_kwargs) # no modulation
                elif HOD == 'fscabTabulated':
                    cens_occ = FSCAssembiasTabulatedCens(redshift=z, **hod_kwargs)
                    sats_occ = FSCAssembiasTabulatedSats(redshift=z,**hod_kwargs) # no modulation
                elif HOD == 'corrTabulated':
                    cens_occ = CorrAssembiasTabulatedCens(redshift=z, **hod_kwargs)
                    sats_occ = CorrAssembiasTabulatedSats(redshift=z,**hod_kwargs) # no modulation
                elif HOD == 'tabulated2D':
                    cens_occ = Tabulated2DCens(redshift=z, **hod_kwargs)
                    sats_occ = Tabulated2DSats(redshift=z, **hod_kwargs)  # no modulation
                elif HOD == 'stepFunc':
                    cens_occ = StepFuncCens(redshift=z, **hod_kwargs)
                    sats_occ = StepFuncSats(redshift=z, **hod_kwargs)
                # TODO make it so I can pass in custom HODs like this.
                # This will be obtuse when I include assembly bias
                # This is already obtuse look at that.
                elif HOD == 'reddick14+redMagic':
                    cens_occ = Reddick14Cens(redshift=z, **hod_kwargs)
                    sats_occ = RedMagicSats(redshift=z, cenocc_model = cens_occ, **hod_kwargs)

                self.model = HodModelFactory(
                    centrals_occupation=cens_occ,
                    centrals_profile=TrivialPhaseSpace(redshift=z),
                    satellites_occupation=sats_occ,
                    satellites_profile=NFWPhaseSpace(redshift=z))

            else:
                self.model = PrebuiltHodModelFactory(HOD, **hod_kwargs)
        else:
            cens_occ = HOD[0](redshift=z, **hod_kwargs)
            sats_occ = HOD[1](redshift=z, **hod_kwargs)
            self.model = HodModelFactory(
                centrals_occupation=cens_occ,
                centrals_profile=TrivialPhaseSpace(redshift=z),
                satellites_occupation=sats_occ,
                satellites_profile=NFWPhaseSpace(redshift=z))

    def get_assembias_key(self, gal_type):
        '''
        Helper function to get the key of the assembly bias strength keys, as they are obscure to access.
        :param gal_type:
            Galaxy type to get Assembias strength. Options are 'centrals' and 'satellites'.
        :return:
        '''
        assert gal_type in {'centrals', 'satellites'}
        return self.model.input_model_dictionary['%s_occupation' % gal_type]._get_assembias_param_dict_key(0)

    # TODO this isn't a traditional observable, so I can't use the same decorator. Not sure how to handle that.
    # TODO little h's here and in hod?
    def calc_mf(self, mass_bin_range = (9,16), mass_bin_size=0.01, min_ptcl=200):
        """
        Get the mass function of the halo catalog.
        :param mass_bin_range
            A tuple of the lwoer and upper bounds of the log mass bins. Default is (9,16)
        :param mass_bin_size:
            Mass binnig size to use. Default is 0.1 dex
        :param min_ptcl:
            Minimum number of particles in a halo. Default is 200
        :return:
            mass_function, the mass function of halocat.
        """
        try:
            assert self.halocat is not None
        except AssertionError:
            raise AssertionError("Please load a halocat before calling calc_mf.")
        if hasattr(self,'_last_halocat_id'):
            if self._last_halocat_id == id(self.halocat):
                return self._last_mf
        masses = self.halocat.halo_table[self.halocat.halo_table['halo_upid']==-1]['halo_mvir']
        masses = masses[masses > min_ptcl*self.pmass]

        mass_bins = np.logspace(mass_bin_range[0], mass_bin_range[1], int( (mass_bin_range[1]-mass_bin_range[0])/mass_bin_size )+1 )

        mf = np.histogram(masses,mass_bins)[0]
        self._last_mf = mf
        self._last_halocat_id = id(self.halocat)

        return mf
    # TODO same concerns as above

    def calc_hod(self, params={}, mass_bin_range = (9,16), mass_bin_size=0.01, component='all'):
        """
        Calculate the analytic HOD for a set of parameters
        :param params:
            HOD parameters. Only those that are changed from the original are required; the rest will remain the default.
        :param mass_bin_range
            A tuple of the lwoer and upper bounds of the log mass bins. Default is (9,16)
        :param mass_bin_size:
            Mass binnig size to use. Default is 0.1 dex
        :param component:
            Which HOD component to compute for. Acceptable are "all" (default), "central" or "satellite"
        :return:
        """

        assert component in {'all','central','satellite'}
        try:
            assert self.model is not None
        except AssertionError:
            raise AssertionError("Please load a model before calling calc_hod.")

        bins = np.logspace(mass_bin_range[0], mass_bin_range[1], int( (mass_bin_range[1]-mass_bin_range[0])/mass_bin_size )+1 )
        bin_centers = (bins[:-1]+bins[1:])/2
        self.model.param_dict.update(params)
        cens_occ, sats_occ = self.model.model_dictionary['centrals_occupation'], self.model.model_dictionary['satellites_occupation']
        for key,val in params.iteritems():
            if key in cens_occ.param_dict:
                cens_occ.param_dict[key] = val
            if key in sats_occ.param_dict:
                sats_occ.param_dict[key] = val


        if component == 'all' or component == 'central':
            cen_hod = getattr(cens_occ, "baseline_mean_occupation", cens_occ.mean_occupation)(prim_haloprop=bin_centers)

            if component == 'central':
                return cen_hod
        if component == 'all' or component == 'satellite':

            sat_hod = getattr(sats_occ, "baseline_mean_occupation", sats_occ.mean_occupation)(prim_haloprop=bin_centers)
            if component == 'satellite':
                return sat_hod

        return cen_hod+sat_hod

    def calc_analytic_nd(self, params={}):
        """
        Calculate the number density from the HOD and Mass function, rather than recovering from a populatedd catalog.
        :param params:
            HOD parameters. Only those that are changed from the original are required; the rest will remain the default.
        :return: nd, a float that represents the analytic number density
        """
        mf = self.calc_mf()
        hod = self.calc_hod(params)
        return np.sum(mf*hod)/((self.Lbox/self.h)**3)

    def calc_xi_mm(self, rbins, n_cores='all', use_corrfunc=False):
        """
        Calculate the matter-matter realspace autocorrelation function
        :param rbins:
            radial binning to use during computation
        :param n_cores:
            number of cores to use during compuation. Must be an integer or 'all'. Default is 'all'
        :param use_corrfunc:
            Boolean, whether or not to use corrfunc if it is available. Default is true. If false, will use
            halotools.
        :return:
        """
        if hasattr(self, '_xi_mm_bins') and np.all(self._xi_mm_bins == rbins):# we have this one cached
            return self._xi_mm

        if use_corrfunc:
            assert CORRFUNC_AVAILABLE

        n_cores = self._check_cores(n_cores)

        x, y, z = [self.halocat.ptcl_table[c] for c in ['x', 'y', 'z']]
        pos = return_xyz_formatted_array(x, y, z, period=self.Lbox)

        if use_corrfunc:
            out = xi(self.Lbox / self.h, n_cores, rbins,
                     x.astype('float32') / self.h, y.astype('float32') / self.h,
                     z.astype('float32') / self.h)

            xi_all = out[4]  # returns a lot of irrelevant info
            # TODO jackknife with corrfunc?

        else:
            xi_all = tpcf(pos / self.h, rbins, period=self.Lbox / self.h, num_threads=n_cores,
                          estimator='Landy-Szalay')

        #cache, so we don't ahve to repeat this calculation several times.
        self._xi_mm_bins = rbins
        self._xi_mm = xi_all

        return xi_all

    def populate(self, params={}, min_ptcl=200):
        '''
        Populate the stored halocatalog with a new realization. Load must be called first.
        :param params:
            HOD parameters. Only those that are changed from the original are required; the rest will remain the default.
        :param min_ptcl:
            Minimum number of particles which constitutes a halo.
        :return: None
        '''
        try:
            assert self.model is not None
        except AssertionError:
            raise AssertionError("Please call load before calling populate.")

        self.model.param_dict.update(params)
        # might be able to check is model has_attr mock.
        if self.populated_once:
            self.model.mock.populate(Num_ptcl_requirement=min_ptcl)
        else:
            self.model.populate_mock(self.halocat, Num_ptcl_requirement=min_ptcl)
            self.populated_once = True

    # TODO how to handle analytic v observed nd
    @observable
    def calc_number_density(self):
        '''
        Return the number density for a populated box.
        :param: halo
            Whether to calculate hte number density of halos instead of galaxies. Default is false.
        :return: Number density of a populated box.
        '''
        return self.model.mock.number_density*self.h**3

    # TODO do_jackknife to cov?
    @observable
    def calc_xi(self, rbins, n_cores='all', do_jackknife=True, use_corrfunc=False, jk_args={}, halo=False):
        '''
        Calculate a 3-D correlation function on a populated catalog.
        :param rbins:
            Radial bins for the correlation function.
        :param n_cores
            Number of cores to use. default is 'all'
        :param do_jackknife:
            Whether or not to do a jackknife along with the xi calculation. Generally slower. Not supported with
            with corrfunc at present.
        :param use_corrfunc:
            Whether or not to use the halotools function to calculate xi, or Manodeep's corrfunc. Corrfunc is not
            available on all systems and currently jackknife is not supported.
        :param jk_args
            Dictionary of arguements for the jackknife call.
        :param halo
            Whether to calculate the halo correlation instead of the galaxy correlation. Default is False.
        :return: xi:
                len(rbins) 3-D correlation function
                xi_cov (if do_jacknife and ! use_corrfunc):
                (len(rbins), len(rbins)) covariance matrix for xi.
        '''
        assert not (do_jackknife and use_corrfunc)  # can't both be true.

        n_cores = self._check_cores(n_cores)
        if halo:
            x,y,z = [self.model.mock.halo_table[c] for c in ['halo_x', 'halo_y', 'halo_z']]
        else:
            x, y, z = [self.model.mock.galaxy_table[c] for c in ['x', 'y', 'z']]
        pos = return_xyz_formatted_array(x, y, z, period=self.Lbox)

        if use_corrfunc:
            '''
            # write bins to file
            # unforunately how corrfunc has to work
            # TODO A custom binfile, or one that's already written?
            bindir = path.dirname(path.abspath(__file__))  # location of files with bin edges
            with open(path.join(bindir, './binfile'), 'w') as f:
                for low, high in zip(rbins[:-1], rbins[1:]):
                    f.write('\t%f\t%f\n' % (low, high))

            # countpairs requires casting in order to work right.
            xi_all = countpairs_xi(self.model.mock.Lbox / self.h, n_cores, path.join(bindir, './binfile'),
                                   x.astype('float32') / self.h, y.astype('float32') / self.h,
                                   z.astype('float32') / self.h)
            xi_all = np.array(xi_all, dtype='float64')[:, 3]
            '''
            out = xi(self.model.mock.Lbox / self.h, n_cores, rbins,
                     x.astype('float32') / self.h, y.astype('float32') / self.h,
                     z.astype('float32') / self.h)

            xi_all = out[4]  # returns a lot of irrelevant info
            # TODO jackknife with corrfunc?

        else:
            if do_jackknife:
                np.random.seed(int(time()))
                if not jk_args:
                    # TODO customize these?
                    n_rands = 5
                    n_sub = 5
                else:
                    n_rands = jk_args['n_rands']
                    n_sub = jk_args['n_sub']

                randoms = np.random.random((pos.shape[0] * n_rands,
                                            3)) * self.Lbox / self.h  # Solution to NaNs: Just fuck me up with randoms
                xi_all, xi_cov = tpcf_jackknife(pos / self.h, randoms, rbins, period=self.Lbox / self.h,
                                                num_threads=n_cores, Nsub=n_sub, estimator='Landy-Szalay')
            else:
                xi_all = tpcf(pos / self.h, rbins, period=self.Lbox / self.h, num_threads=n_cores,
                              estimator='Landy-Szalay')

        # TODO 1, 2 halo terms?

        if do_jackknife:
            return xi_all, xi_cov
        return xi_all

    @observable
    def calc_bias(self,rbins, n_cores='all', use_corrfunc=False, **xi_kwargs):
        """
        Calculate the bias in a clustering sample. Computes the clustering and divides by the matter clustering,
        which is cached so repeated computations do not occur.
        :param rbins:
            radial bins to use for both calculations.
        :param n_cores:
            number of cores to use for both calculations. Must be a nonzero integer or 'all'. Default is 'all'
        :param use_corrfunc:
            Boolean for wheter or not to use corrfunc, if it is available. Default is False.
        :param xi_kwargs:
            Optional kwargs for the xi calculation.
        :return: bias, an array of shape(rbins.shape[0]-1) of the galaxy bias.
        """
        try:
            assert 'do_jackknife' not in xi_kwargs and 'jk_kwargs' not in xi_kwargs
        except AssertionError:
            raise NotImplementedError("Jackknife functionality is currently unavailable for bias.")

        return self.calc_xi(rbins, n_cores, use_corrfunc, **xi_kwargs)/self.calc_xi_mm(rbins, n_cores, use_corrfunc)

    @observable
    def calc_xi_gm(self, rbins, n_cores='all', use_corrfunc=False): #TODO add in halo? may not be worth it.

        n_cores = self._check_cores(n_cores)

        x_g, y_g, z_g = [self.model.mock.galaxy_table[c] for c in ['x', 'y', 'z']]
        pos_g = return_xyz_formatted_array(x_g, y_g, z_g, period=self.Lbox)

        x_m, y_m, z_m = [self.halocat.ptcl_table[c] for c in ['x', 'y', 'z']]
        pos_m = return_xyz_formatted_array(x_m, y_m, z_m, period=self.Lbox)

        if use_corrfunc:
            #corrfunc doesn't have built in cross correlations
            rand_N1 = 3 * len(x_g)

            rand_X1 = np.random.uniform(0, self.Lbox*self.h, rand_N1)
            rand_Y1 = np.random.uniform(0, self.Lbox*self.h, rand_N1)
            rand_Z1 = np.random.uniform(0, self.Lbox*self.h, rand_N1)

            rand_N2 = 3 * len(x_m)

            rand_X2 = np.random.uniform(0, self.Lbox / self.h, rand_N2)
            rand_Y2 = np.random.uniform(0, self.Lbox / self.h, rand_N2)
            rand_Z2 = np.random.uniform(0, self.Lbox / self.h, rand_N2)

            autocorr = False
            D1D2 = DD(autocorr, n_cores, rbins, x_g.astype('float32'),  y_g.astype('float32'),  z_g.astype('float32'),
                      X2= x_m.astype('float32'), Y2 = y_m.astype('float32'), Z2 = z_m.astype('float32'))
            D1R2 = DD(autocorr, n_cores, rbins, x_g.astype('float32'), y_g.astype('float32'), z_g.astype('float32'),
                      X2=rand_X2.astype('float32'), Y2=rand_Y2.astype('float32'), Z2=rand_Z2.astype('float32'))
            D2R1 = DD(autocorr, n_cores, rbins, x_m.astype('float32'), y_m.astype('float32'), z_m.astype('float32'),
                      X2=rand_X1.astype('float32'), Y2=rand_Y1.astype('float32'), Z2=rand_Z1.astype('float32'))
            R1R2 = DD(autocorr, n_cores, rbins,rand_X1.astype('float32'),rand_Y1.astype('float32'), rand_Z1.astype('float32'),
                      X2=rand_X2.astype('float32'), Y2=rand_Y2.astype('float32'), Z2=rand_Z2.astype('float32'))

            xi_all = convert_3d_counts_to_cf(len(x_g), len(x_m), rand_N1, rand_N2,
                                             D1D2, D1R2, D2R1, R1R2)
        else:
            xi_all = tpcf(pos_g / self.h, rbins, sample2 = pos_m*self.h, period=self.Lbox / self.h, num_threads=n_cores,
                          estimator='Landy-Szalay', do_auto=False)

        return xi_all

    # TODO use Joe's code. Remember to add sensible asserts when I do.
    # TODO Jackknife? A way to do it without Joe's code?
    @observable
    def calc_wp(self, rp_bins, pi_max=40, do_jackknife=True, use_corrfunc=False, n_cores='all', RSD=False, halo=False):
        '''
        Calculate the projected correlation on a populated catalog
        :param rp_bins:
            Radial bins of rp, distance perpendicular to projection direction.
        :param pi_max:
            Maximum distance to integrate to in projection. Default is 40 Mpc/h
        :param n_cores:
            Number of cores to use for calculation. Default is 'all' to use all available.
        :param RSD:
            Boolean whether or not to apply redshift space distortions. Default is True.
        :param halo:
            Whether to calculate halo correlation or galaxy correlation. Default is False.
        :return:
            wp_all, the projection correlation function.
        '''

        # could move these last parts ot the decorator or a helper. Ah well.
        n_cores = self._check_cores(n_cores)

        if halo:
            x,y,z = [self.model.mock.halo_table[c] for c in ['halo_x', 'halo_y', 'halo_z']]
        else:
            x, y, z = [self.model.mock.galaxy_table[c] for c in ['x', 'y', 'z']]

        #No RSD for halos
        if RSD and not halo:
            # for now, hardcode 'z' as the distortion dimension.
            distortion_dim = 'z'
            v_distortion_dim = self.model.mock.galaxy_table['v%s' % distortion_dim]
            # apply redshift space distortions
            pos = return_xyz_formatted_array(x, y, z, velocity=v_distortion_dim, \
                                             velocity_distortion_dimension=distortion_dim, period=self.Lbox)
        else:
            pos = return_xyz_formatted_array(x, y, z, period=self.Lbox)

        # don't forget little h!!
        if use_corrfunc:
            out = xi(self.model.mock.Lbox / self.h, pi_max / self.h, n_cores, rp_bins,
                     x.astype('float32') / self.h, y.astype('float32') / self.h,
                     z.astype('float32') / self.h)

            wp_all = out[4]  # returns a lot of irrelevant info
        else:
            wp_all = wp(pos / self.h, rp_bins, pi_max / self.h, period=self.Lbox / self.h, num_threads=n_cores)
        return wp_all

    @observable
    def calc_wt_projected(self, theta_bins, do_jackknife=True, n_cores='all', halo = False):
        '''
        Calculate the angular correlation function, w(theta), from a populated catalog.
        NOTE this is depreceated, because projecting a snapshot does not provide consistent results.
        Use calc_wt instead.
        :param theta_bins:
            The bins in theta to use.
        :param n_cores:
            Number of cores to use for calculation. Default is 'all' to use all available.
        :param halo:
            Whether or not to calculate clustering or halos or galaxies. Default is False.
        :return:
            wt_all, the angular correlation function.
        '''

        n_cores = self._check_cores(n_cores)

        if halo:
            pos = np.vstack( [self.model.mock.halo_table[c] for c in ['halo_x', 'halo_y', 'halo_z']]).T
            vels = np.vstack([self.model.mock.halo_table[c] for c in ['halo_vx', 'halo_vy', 'halo_vz']]).T
        else:
            pos = np.vstack([self.model.mock.galaxy_table[c] for c in ['x', 'y', 'z']]).T
            vels = np.vstack([self.model.mock.galaxy_table[c] for c in ['vx', 'vy', 'vz']]).T

        # TODO is the model cosmo same as the one attached to the cat?
        ra, dec, z = mock_survey.ra_dec_z(pos / self.h, vels / self.h, cosmo=self.cosmology)
        ang_pos = np.vstack((np.degrees(ra), np.degrees(dec))).T

        n_rands = 5
        rand_pos = np.random.random((pos.shape[0] * n_rands, 3)) * self.Lbox#*self.h
        rand_vels = np.zeros((pos.shape[0] * n_rands, 3))

        rand_ra, rand_dec, rand_z = mock_survey.ra_dec_z(rand_pos / self.h, rand_vels / self.h, cosmo=self.cosmology)
        rand_ang_pos = np.vstack((np.degrees(rand_ra), np.degrees(rand_dec))).T

        # NOTE I can transform coordinates and not have to use randoms at all. Consider?
        wt_all = angular_tpcf(ang_pos, theta_bins, randoms=rand_ang_pos, num_threads=n_cores)
        return wt_all

    def compute_wt_prefactor(self, zbins, dNdz):
        """
        Helper function to compute the w(theta) prefactor W from the dNdz distribution. 
        param zbins: the edges of the redshift bins in which dNdz is computed
        param dNdz: the normalized redshift distribution of the sample. Should have shape (len(zbins)-1,)

        return: W, the result of 2/c*\int_0^{\infty} dz H(z) (dN/dz)^2, in units of inverse Mpc
        """
        W = 0 
        for idx, dN in enumerate(dNdz):
            dz = zbins[idx+1] - zbins[idx]
            dNdz = dN/dz
            H = self.cosmology.H((zbins[idx+1] + zbins[idx])/2.0)
            W+= dz*H*(dNdz)**2
                                                                                            
        return  (2*W/const.c).to("1/Mpc").value


    @observable
    def calc_wt(self, theta_bins, rbins, W, n_cores='all', halo = False, xi_kwargs = {}):
        """
        TODO docs
        """
        assert 'do_jackknife' not in xi_kwargs

        if type(rbins) is list:
            rbins = np.array(rbins)
        n_cores = self._check_cores(n_cores)
        xi = self.calc_xi(rbins,do_jackknife=False,n_cores=n_cores, halo=halo, **xi_kwargs)
        rpoints = (rbins[:-1]+rbins[1:])/2.0
        
        if np.any(xi<=0):
            warnings.warn("Some values of xi are less than 0. Setting to a small nonzero value. This may have unexpected behavior, check your HOD")
            xi[xi<=0] = 1e-3

        xi_interp = interp1d(np.log10(rpoints), np.log10(xi))

        xi_mm = self.calc_xi_mm(rbins,n_cores=n_cores) 
        #if precomputed, will just load the cache

        bias2 = np.mean(xi[-3:]/xi_mm[-3:]) #estimate the large scale bias from the box
        #note i don't use the bias builtin cuz i've already computed xi_gg. 
        
        #Assume xi_mm doesn't go below 0; will fail catastrophically if it does. but if it does we can't hack around it.
        m,b,_,_,_ =linregress(np.log10(rpoints), np.log10(xi_mm))

        large_scale_model = lambda r: bias2*(10**b)*(r**m) #should i use np.power?

        tpoints = (theta_bins[1:] + theta_bins[:-1])/2.0
        wt = np.zeros_like(tpoints)
        x = self.cosmology.comoving_distance(self.z)*self.a/self.h

        assert tpoints[0]*x.to("Mpc").value/self.h >= rbins[0]
        #ubins = np.linspace(10**-6, 10**4.0, 1001)
        ubins = np.logspace(-6, 4.0, 1001)
        ubc = (ubins[1:]+ubins[:-1])/2.0
        
        def integrate_xi(bin_no):#, w_theta, bin_no, ubc, ubins) 
            int_xi = 0
            t_med = np.radians(tpoints[bin_no])
            for ubin_no, _u in enumerate(ubc):
                _du = ubins[ubin_no+1]-ubins[ubin_no]
                u = _u*units.Mpc*self.a/self.h
                du = _du*units.Mpc*self.a/self.h

                r = np.sqrt((u**2+(x*t_med)**2))#*cat.h#not sure about the h

                if r > (units.Mpc/self.h)*self.Lbox/10:
                    int_xi+=du*large_scale_model(r.value)
                else:

                    int_xi+=du*(np.power(10, \
                                xi_interp(np.log10(r.value))))
            wt[bin_no] = int_xi.to("Mpc").value/self.h

        #Currently this doesn't work cuz you can't pickle the integrate_xi function.
        #I'll just ignore for now. This is why i'm making an emulator anyway
        #p = Pool(n_cores)  
        map(integrate_xi, range(tpoints.shape[0]))
        #p.map(integrate_xi, range(tpoints.shape[0]))
        #p.terminate()

        return wt*W
    # TODO need to implement a particle check for some of these new functions
    # TODO may want to enable central/satellite cuts, etc
    @observable
    def calc_ds(self,rp_bins, n_cores='all'):
        """
        Calculate delta sigma, from a given galaxy and particle sample
        Returns in units of h*M_sun/pc^2, so be wary! 
        :param rp_bins:
            The projected radial binning in Mpc
        :param n_cores:
            Number of cores to use for the calculation, default is "all"
        :return:
            delta sigma, a numpy array of size (rp_bins.shape[0]-1,)
        """
        try:
            assert hasattr(self, "_downsample_factor")
        except AssertionError:
            raise AssertionError("The catalog loaded doesn't have a downsampling factor."
                                 "Make sure you load particles to calculate delta_sigma.")
        n_cores = self._check_cores(n_cores)

        x_g, y_g, z_g = [self.model.mock.galaxy_table[c] for c in ['x', 'y', 'z']]
        pos_g = return_xyz_formatted_array(x_g, y_g, z_g, period=self.Lbox)

        x_m, y_m, z_m = [self.halocat.ptcl_table[c] for c in ['x', 'y', 'z']]
        pos_m = return_xyz_formatted_array(x_m, y_m, z_m, period=self.Lbox)

        #Halotools wnats downsampling factor defined oppositley
        #TODO verify little h!
        # TODO maybe split into a few lines for clarity
        return self.h*delta_sigma(pos_g / self.h,pos_m*self.h, self.pmass*self.h,
                           downsampling_factor = 1./self._downsample_factor, rp_bins = rp_bins,
                           period=self.Lbox / self.h, num_threads=n_cores,cosmology = self.cosmology)[1]/(1e12)
