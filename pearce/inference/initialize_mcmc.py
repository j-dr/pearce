#/bin/bash
'''
Create the initial state for the sampler from a config. Since there are a lot of moving parts to this,
keeps everything together cleanly.
'''
from os import path
from time import time
import warnings
import numpy as np
from scipy.optimize import minimize_scalar
import h5py
import yaml
from pearce.mocks import cat_dict

# I don't know if I need an object for this a la trainer
# for now, make a main function, until there is clear need to do otherwise

def main(config_fname):
    """
    Control all other processes. Primarliy just makes the hdf5 file in the designated
    location, and copies over relevant info to the attrs
    :param config_fname:
        Filename of a YAML config file.
    """

    assert path.isfile(config_fname), "%s is not a valid config filename."%config_fname

    with open(config_fname, 'r') as ymlfile:
        cfg = yaml.load(ymlfile)

    filename = cfg['fname']

    #assert path.isfile(filename), "%s is not a valid output filename"%filename
    f = h5py.File(filename, 'w')

    emu_cfg = cfg['emu']
    data_cfg = cfg['data']
    chain_cfg = cfg['chain']

    emu_config(f, emu_cfg)
    data_config(f, data_cfg)
    chain_config(f, chain_cfg)

    f.close()


def emu_config(f, cfg):
    """
    Attach the emu config info, putting in defaults for unspecified values
    :param f:
        File handle of hdf5 file
    :param cfg:
        Emu portion of the cfg
    """
    required_emu_keys = ['emu_type', 'training_file']
    for key in required_emu_keys:
        assert key in cfg, "%s not in config but is required."%key
        f.attrs[key] = cfg[key]

    optional_keys = ['fixed_params', 'emu_hps', 'seed']
    #default_vals = [{}, {}, {}, None] #gonna None all these if empty
    # want to clafiy nothing specified

    for key in optional_keys:
        attr = cfg[key] if key in cfg else None
        if key == 'seed' and  attr is None:
            attr = int(time()) 
        else:
            attr = str(attr) if type(attr) is dict else attr 

        f.attrs[key] = attr 

def data_config(f, cfg):
    """
    Attach data config info.

    Additionally, compute new values, if required.
    :param f:
        A file hook to an HDF5 file
    :param cfg:
        cfg with the info for the data
    """
    if 'true_data_fname' in cfg:
        f.attrs['true_data_fname'] = cfg['true_data_fname']
        f.attrs['true_cov_fname'] = cfg['true_cov_fname']
        data, cov = _load_data(cfg['true_data_fname'], cfg['true_cov_fname'])

    else: #compute the data ourselves
        data, cov = _compute_data(cfg)

    for key in ['sim', 'obs', 'cov']:
        f.attrs[key] = str(cfg[key]) if key in cfg else None
        if key not in cfg:
            warnings.warn("Not all data attributes were specified. This is not reccomended, since the chain may not be well described in the future!")

    f.create_dataset('data', data = data)
    f.create_dataset('cov', data = cov)

def _load_data(true_data_fname, true_cov_fname):
    """
    Data is already computed, just load  it up
    :param true_data_fname:
        Fname(s) of the data. Str or list of strs
    :param true_cov_fname:
        Fname(s) of the covaraicnes. Str or list of strs
    :return:
        data, cov. A (n_data, n_rows) array and a (n_data, n_rows, n_rows) array
    """

    if type(true_data_fname) is str:
        true_data_fname = [true_data_fname]
    if type(true_cov_fname) is str:
        true_cov_fname = true_cov_fname

    assert len(true_data_fname) == len(true_cov_fname), "Cov and Data fnames different lengths!"

    data = []
    cov = []

    for fname in true_data_fname:
        data.append(np.loadtxt(fname))

    for fname in true_cov_fname:
        cov.append(np.loadtxt(fname))

    # NOTE stacking could be wrong here, double check
    return np.vstack(data), np.vstack(cov)

def _compute_data(cfg):
    """
    Compute the truth data explicitly
    :param cfg:
        Config for the info needed to create relevant data
        #TODO be more clear about requriements because they are extensive
    :return:
         data, cov. A (n_data, n_rows) array and a (n_data, n_rows, n_rows) array
    """
    sim_cfg = cfg['sim']
    obs_cfg = cfg['obs']
    cov_cfg = cfg['cov']

    cat = cat_dict[sim_cfg['simname']](**sim_cfg['sim_hps'])  # construct the specified catalog!

    # TODO add shams
    if sim_cfg['gal_type'] == 'HOD':
        cat.load(sim_cfg['scale_factor'], HOD=sim_cfg['hod_name'], **sim_cfg['sim_hps'])

        em_params = sim_cfg['hod_params']
        add_logMmin(em_params, cat, float(sim_cfg['nd']))

        # TODO logspace
        r_bins = obs_cfg['rbins']

        obs = obs_cfg['obs']

        if type(obs) is str:
            obs = [obs]

        emu_cov_fname = cov_cfg['emu_cov_fname']
        if type(emu_cov_fname) is str:
            emu_cov_fname = [emu_cov_fname]

        assert len(obs) == len(emu_cov_fname), "Emu cov not same length as obs!"

        n_bins = len(r_bins)-1
        data = np.zeros((len(obs)*n_bins))
        cov = np.zeros((len(obs)*n_bins, len(obs)*n_bins))

        for idx, (o, ecf) in enumerate(zip(obs, emu_cov_fname)):
            # TODO need some check that a valid configuration is specified
            y_mean, yjk = None, None
            shot_cov = covjk = np.zeros((n_bins, n_bins))

            calc_observable = getattr(cat, 'calc_%s' % o)
            if obs_cfg['mean'] or 'shot' in cov_cfg: #TODO add number of pop
                xi_vals = []
                for i in range(50):
                    cat.populate(em_params)
                    xi_vals.append(calc_observable(r_bins))

                shot_xi_vals = np.array(xi_vals)
                y_mean = np.mean(shot_xi_vals, axis = 0)
                shot_cov = np.cov(xi_vals, rowvar=False)

            if 'jackknife_hps' in cov_cfg:
                cat.populate(em_params)
                # TODO need to fix jackknife
                yjk, covjk = calc_observable(r_bins, do_jackknife=True, jk_args=cov_cfg['jackknife_hps'])

            assert path.isfile(ecf), "Invalid emu covariance specified."
            try:
                emu_cov = np.loadtxt(ecf)
            except ValueError:
                emu_cov = np.load(ecf)

            if yjk is None and y_mean is None:
                raise AssertionError("Invalid data calculation specified.")
            if obs_cfg['mean']:
                data[idx*n_bins:(idx+1)*n_bins] = y_mean
            else:
                data[idx*n_bins:(idx+1)*n_bins] = yjk

            cov[idx*n_bins:(idx+1)*n_bins, idx*n_bins:(idx+1)*n_bins] = \
                                            shot_cov+covjk+emu_cov

        if 'cross_cov_fname' in cov_cfg:
            # TODO should i be adding
            cross_cov = np.loadtxt(cov_cfg['cross_cov_fname'])
            assert cross_cov.shape[0] == n_bins, "Cross-cov matrix of wrong size."
            cov[:n_bins, n_bins:] = cross_cov
            cov[n_bins:, :n_bins] = cross_cov

    else:
        raise NotImplementedError("Non-HOD caclulation not supported")


    if 'cosmo_params' not in sim_cfg:
        cpv = cat._get_cosmo_param_names_vals()
        cosmo_param_dict = {key: val for key, val in zip(cpv[0], cpv[1])}
        sim_cfg['cosmo_params'] = cosmo_param_dict

    return data, cov

def add_logMmin(hod_params, cat, nd):
    """
    In the fixed number density case, find the logMmin value that will match the nd given hod_params
    :param: hod_params:
        The other parameters besides logMmin
    :param cat:
        the catalog in question
    :return:
        None. hod_params will have logMmin added to it.
    """
    hod_params['logMmin'] = 13.0  # initial guess

    # cat.populate(hod_params) #may be overkill, but will ensure params are written everywhere
    def func(logMmin, hod_params):
        hod_params.update({'logMmin': logMmin})
        return (cat.calc_analytic_nd(hod_params) - nd) ** 2

    res = minimize_scalar(func, bounds=(12, 16), args=(hod_params,), options={'maxiter': 100}, method='Bounded')

    # assuming this doens't fail
    hod_params['logMmin'] = res.x

def chain_config(f, cfg):
    """
    Attach relvant config info for the mcmc chains
    :param f:
        Handle to an HDF5 file to attach things to
    :param cfg:
        Cfg with MCMC data
    """

    required_mcmc_keys = ['nwalkers', 'nsteps']

    for key in required_mcmc_keys:
        assert key in cfg, "%s not in config but is required."%key
        f.attrs[key] = cfg[key]

    optional_keys = ['nburn', 'seed']#, 'fixed_params']
    for key in optional_keys:
        attr = cfg[key] if key in cfg else None
        attr = str(attr) if type(attr) is dict else attr
        f.attrs[key] = attr 

    key = 'fixed_params'
    attr = cfg[key] if key in cfg else None
    attr = str(attr) if type(attr) is dict else attr
    f.attrs['chain_fixed_params'] = attr 


if __name__ == '__main__':
    from sys import argv
    print(argv[1])
    main(argv[1])
