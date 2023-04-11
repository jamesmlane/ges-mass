# ----------------------------------------------------------------------------
#
# TITLE - mass.py
# AUTHOR - James Lane
# PROJECT - mw-dfs
#
# ----------------------------------------------------------------------------

### Docstrings and metadata:
'''Functions for density profile fitting and calculation of mass
'''

__author__ = "James Lane"

### Imports
import os
import numpy as np
import multiprocessing
import dill as pickle
import itertools
import emcee
from tqdm.notebook import tqdm
import time
import datetime
import warnings
import scipy.optimize
from scipy.stats import norm
from isodist import FEH2Z, Z2FEH
from galpy.util import coords
from galpy import potential
from astropy import units as apu
from . import densprofiles as pdens
from . import util as putil
from . import iso as piso
from . import plot as pplot

_ro = 8.275 # Gravity Collab.
_zo = 0.0208 # Bennett and Bovy

# _PRIOR_ETA_MIN = 1/np.sqrt(2) -> 45 degrees
_PRIOR_ETA_MIN = 0.5

# ----------------------------------------------------------------------------


### Fitting

def fit(hf, nprocs=10, force_fit=False, mle_init=True, just_mle=False, 
        return_walkers=True, optimizer_method='Powell',
        mass_int_type='spherical_grid', batch_masses=True, 
        make_ml_aic_bic=True, calculate_masses=True, 
        post_optimization=True, mcmc_diagnostic=False):
    '''fit:
    
    Front-facing wrapper of fit_dens, mass_from_density_samples, and likelihood 
    calculation which uses HaloFit class to set parameters and save results to 
    file.
    
    Args:
        hf (HaloFit) - Class for halo fit
        nprocs (int) - Number of processors to use [default 10]
        force_fit (bool) - Force overwrite existing results? [default False]
        mle_init (bool) - Initialize the fit with maximum likelihood? [default
            True]
        just_mle (bool) - Only calculate the maximum likelihood, no MCMC?
        return_walkers (bool) - Return the emcee sampler object? [default True]
        optimizer_method (string) - Optimizer method to use [default 'Powell']
        mass_int_type (string) - Scheme for mass integration [default 
            'spherical_grid']
        batch_masses (bool) - Batch calculate the masses [default True]
        make_ml_aic_bic (bool) - Calculate maximum likelihood, AIC, BIC using
            samples [default True]
        calculate_masses (bool) - Do mass calculations [default True]
        post_optimization (bool) - Do optimization starting from MCMC maximum 
            likelihood [default True]
    
    Returns:
        None
    '''
    # Overwrite?
    if not force_fit and os.path.exists(hf.fit_data_dir+'samples.npy') and\
                         os.path.exists(hf.fit_data_dir+'masses.npy'):
        assert False, 'Not overwriting data in '+hf.fit_data_dir+\
            ', set force_fit=True'
    
    # Prepare MCMC diagnostics
    if mcmc_diagnostic:
        mcmc_diagnostic_filename = hf.fit_data_dir+'mcmc_diagnostics.txt'
    else:
        mcmc_diagnostic_filename = None
    
    # Do MLE and MCMC
    out_fit = fit_dens(densfunc=hf.densfunc, effsel=hf.get_fit_effsel(), 
        effsel_grid=hf.get_effsel_list(), data=hf.get_data_list(), 
        init=hf.init, nprocs=nprocs, nwalkers=hf.nwalkers, nit=hf.nit, 
        ncut=hf.ncut, usr_log_prior=hf.usr_log_prior, MLE_init=mle_init, 
        just_MLE=just_mle, return_walkers=return_walkers,
        mcmc_diagnostic_filename=mcmc_diagnostic_filename,
        optimizer_method=optimizer_method)
    
    # Unpack results based on supplied keywords
    if just_mle:
        return out_fit
    if return_walkers:
        if mle_init:
            samples, opt, sampler = out_fit
        else:
            samples, sampler = out_fit
    else:
        if mle_init:
            samples, opt = out_fit
        else:
            samples = out_fit
    
    # Save fits
    np.save(hf.fit_data_dir+'samples.npy',samples)
    if mle_init:
        with open(hf.fit_data_dir+'opt.pkl','wb') as f:
            pickle.dump(opt,f)
    if return_walkers:
        with open(hf.fit_data_dir+'sampler.pkl','wb') as f:
            pickle.dump(sampler,f)
    
    # Calculate likelihoods, ML, AIC, BIC
    if make_ml_aic_bic:
        print('Calculating likelihoods for MCMC samples')
        n_samples,n_param = samples.shape
        
        if return_walkers:
            loglike = sampler.get_log_prob(flat=True, discard=hf.ncut)
        else:
            loglike = np.zeros(n_samples)
            for i in range(n_samples):
                print(str(i+1)+'/'+str(n_samples),end='\r')
                loglike[i] = loglike(samples[i], hf.densfunc, hf.get_fit_effsel(), 
                                           hf.Rgrid, hf.phigrid, hf.zgrid, 
                                           hf.Rdata, hf.phidata, hf.zdata, 
                                           hf.usr_log_prior)
        mll = np.max(loglike)
        mll_ind = np.argmax(loglike)
        aic = 2*n_param - 2*mll
        bic = np.log(hf.n_star)*n_param - 2*mll
        np.save(hf.fit_data_dir+'loglike.npy',loglike)
        np.save(hf.fit_data_dir+'mll_aic_bic.npy',
                np.array([mll,mll_ind,aic,bic]))
    
    # Calculate mass
    if calculate_masses:
        out_mass = mass_from_density_samples( samples=samples, 
            densfunc=hf.densfunc, n_star=hf.n_star, effsel=hf.get_fit_effsel(), 
            effsel_grid=hf.get_effsel_list(), iso=hf.get_iso(), 
            feh_range=hf.feh_range, logg_range=hf.logg_range, 
            jkmins=hf.jkmins, n_mass=hf.n_mass, mass_int_type=mass_int_type, 
            int_r_range=hf.int_r_range, nprocs=nprocs, batch=batch_masses, 
            ro=hf.ro, zo=hf.zo)
        masses, facs, mass_inds, isofactors = out_mass

        # Save results
        np.save(hf.fit_data_dir+'masses.npy',masses)
        np.save(hf.fit_data_dir+'facs.npy',facs)
        np.save(hf.fit_data_dir+'mass_inds.npy',mass_inds)
        np.save(hf.fit_data_dir+'isofactors.npy',isofactors)
    
    # Do post optimization from MCMC max likelihood
    if post_optimization:
        assert make_ml_aic_bic, 'must have found ML samples'
        post_init = samples[mll_ind]
        opt_post = hf.run_optimization(post_init)
        with open(hf.fit_data_dir+'opt_post.pkl','wb') as f:
            pickle.dump(opt_post,f)


def fit_dens(densfunc, effsel, effsel_grid, data, init, nprocs, nwalkers,
             nit, ncut, usr_log_prior, MLE_init, just_MLE, return_walkers, 
             convergence_n_tau=50, convergence_delta_tau=0.01, 
             optimizer_method='Powell', mcmc_diagnostic_filename=None):
    '''fit_dens:
    
    Fit a density profile to a set of data given an effective selection 
    function evaluted over a grid. For larger triaxial models the nit should 
    be at least 20k-30k to satisfy autocorrelation timescale requirements. Burn 
    in should happen within 100 steps.
    
    Args:
        densfunc (callable) - Density function with parameters to be fit,
            has the signature (R,phi,z,params) and returns density 
            normalized to solar
        effsel (array) - Array of shape (Nfield,Ndmod) where the effective
            selection function is evaluated.
        effsel_grid (list) - Length-3 list of grids of shape (effsel) 
            representing R,phi,z positions where the selection function is 
            evaluated
        data (list or 3xN array) - List of R,phi,z coordinates for data
        init (array) - Initial parameters
        nprocs (int) - Number of processors to use for multiprocessing
        nwalkers (int) - Number of MCMC walkers to use
        nit (int) - Number of iterations to sample with each walker
        ncut (int) - Number of samples to cut off the beginning of the chain 
            of each walker.
        usr_log_prior (callable) - A user-supplied log prior for convenience. 
            Will be passed to mloglike
        MLE_init (bool) - Initialize the MCMC walkers near a maximum likelihood 
            estimate calculated starting at init? If not then initializes the 
            walkers at init
        just_MLE (bool) - Only calculate the maximum likelihood estimate
        return_walkers (bool) - Return the walker instead of just the samples
        convergence_n_tau (int) - Fraction of autocorrelation timescales to
            walk before convergence is declared
        convergence_delta_tau (float) - Fraction of change in autocorrelation
            timescale to declare convergence
        optimizer_method (str) - Method to use for optimization
        mcmc_diagnostic_filename (str) - Filename to save MCMC diagnostics to
        
        
    Returns:
        opt  () - 
        samples (array) - 
    '''
    # Parameters
    ndim = len(init)
    
    # Unpack effsel grid and data
    Rgrid,phigrid,zgrid = effsel_grid
    Rdata,phidata,zdata = data
    
    # Use maximum likelihood to find a starting point
    if MLE_init:
        print('Doing maximum likelihood to find starting point')
        opt_fn = lambda x: mloglike(x, densfunc, effsel, Rgrid, phigrid, 
            zgrid, Rdata, phidata, zdata, usr_log_prior=usr_log_prior)
        opt = scipy.optimize.minimize(opt_fn, init, method=optimizer_method)
        if opt.success:
            print('MLE successful')
        else:
            print('MLE unsucessful')
            print('message: '+opt.message)
        print('MLE result: '+str(opt.x))
        if just_MLE:
            return opt
    else:
        print('Initializing with params: '+str(init))
    
    # Initialize MCMC from either init or MLE
    if MLE_init:
        pos = [opt.x + 1e-3*np.random.randn(ndim) for i in range(nwalkers)]
    else:
        pos = [init + 1e-3*np.random.randn(ndim) for i in range(nwalkers)]
    
    # Declare MCMC parameters
    if nit is None:
        print_nit = 'inf'
    else:
        print_nit = str(nit)
    autocorr_n = 500 # Interval between checking autocorrelation criterion
    allow_dvrgnt = True
    autocorr = np.zeros(0)
    old_tau = np.inf
    max_nit_tau = 0
    dvrgnt_nit_lim = int(2e4) # N to wait until checking divergence
    dvrgnt_tau_nit_fac = 0.5 # Factor of max(nit/tau) when divergence declared
    has_mcmc_diagnostic_file = False
    if isinstance(mcmc_diagnostic_filename,str):
        mcmc_diagnostic_file = open(mcmc_diagnostic_filename,'w')
        has_mcmc_diagnostic_file = True
    
    # Do MCMC
    print('Running MCMC with '+str(nprocs)+' processers')
    if has_mcmc_diagnostic_file:
        mcmc_start_txt = ('Running MCMC:'
                          'date: '+str(datetime.datetime.now())+'\n'+\
                          'nprocs: '+str(nprocs)+'\n'+\
                          'nwalkers: '+str(nwalkers)+'\n'+\
                          'nit: '+print_nit+'\n'+\
                          'ncut: '+str(ncut)+'\n'+\
                          '------------------------\n'
                          )
        mcmc_diagnostic_file.write(mcmc_start_txt)
    # with multiprocessing.Pool(nprocs) as pool:
    with multiprocessing.Pool(processes=nprocs, 
        initializer=_fit_dens_multiprocessing_init, initargs=(densfunc,
            effsel, Rgrid, phigrid, zgrid, Rdata, phidata, zdata, 
            usr_log_prior)) as pool:
        # Make sampler
        blob_dtype = [("effvol_halo", float), ("effvol_disk", float)]
        sampler = emcee.EnsembleSampler(nwalkers, ndim, loglike_mp, 
            # args=(densfunc, effsel, Rgrid, phigrid, zgrid, Rdata, phidata, 
            #       zdata, usr_log_prior), 
            pool=pool, blobs_dtype=blob_dtype)
        
        # Draw samples
        print('Generating MCMC samples...')
        t0 = time.time()
        for i, result in enumerate(sampler.sample(pos, iterations=nit)):
            # Progress
            if (i+1)%10 == 0: print('sampled '+str(i+1)+'/'+print_nit+\
                                    ' per walker', end='\r')

            # Autocorrelation analysis, only every autocorr_n samples
            if sampler.iteration % autocorr_n:
                continue

            # Compute the autocorrelation time so far
            tau = sampler.get_autocorr_time(tol=0)
            autocorr = np.append(autocorr,np.mean(tau))

            # Parameters of MCMC diagnostics
            nit_tau = np.min(sampler.iteration/tau)
            delta_tau = (old_tau - tau) / tau
            mean_tau = np.mean(tau)
            max_nit_tau = np.max([nit_tau,max_nit_tau])
            
            # Estimate time remaining
            t1 = time.time()
            t_per_sample = (t1-t0)/(i+1)
            t_remaining = t_per_sample*(nit-i-1)
            t_remaining_str = str(datetime.timedelta(seconds=t_remaining))
            

            # Record MCMC diagnostics
            mcmc_diagnostic_txt = (
                'sampled '+str(i+1)+'\n'+\
                'mean tau: '+str(mean_tau)+'\n'+\
                'mean nit/tau: '+str(sampler.iteration/mean_tau)+'\n'+\
                'min nit/tau: '+str(nit_tau)+'\n'+\
                '[min,max] delta tau: ['+\
                str(np.min(delta_tau))+','+str(np.max(delta_tau))+']\n'+\
                '[min,max] acceptance fraction: ['+\
                str(round(np.min(sampler.acceptance_fraction),2))+','+\
                str(round(np.max(sampler.acceptance_fraction),2))+']\n'+\
                'total max nit/tau '+str(max_nit_tau)+'\n'+\
                '(nit/tau)/max[(nit/tau)] '+str(nit_tau/max_nit_tau)+'\n'+\
                'estimated time remaining: '+t_remaining_str+'\n'+\
                '---\n')
            print(mcmc_diagnostic_txt)
                
            # print(mcmc_diagnostic_txt)
            if has_mcmc_diagnostic_file:
                mcmc_diagnostic_file.write(mcmc_diagnostic_txt)
            
            # Check convergence
            converged = np.all(tau * convergence_n_tau < sampler.iteration)
            converged &= np.all(np.abs(delta_tau) < convergence_delta_tau)
            if converged:
                conv_stop_txt = 'Stopped because convergence criterion met'
                print(conv_stop_txt)
                if has_mcmc_diagnostic_file:
                    mcmc_diagnostic_file.write('\n\n'+conv_stop_txt)
                break
            
            # Check divergence
            diverged = (nit_tau/max_nit_tau < dvrgnt_tau_nit_fac)
            diverged &= ((i+1) >= dvrgnt_nit_lim) & allow_dvrgnt
            if diverged:
                dvrgnt_stop_txt = 'Stopped because divergence criterion met'
                print(dvrgnt_stop_txt)
                if has_mcmc_diagnostic_file:
                    mcmc_diagnostic_file.write('\n\n'+dvrgnt_stop_txt)
                break

            old_tau = tau
            continue

        mcmc_finish_txt = 'Finished, sampled '+str(i+1)+'/'+str(nit)
        print(mcmc_finish_txt)
        if has_mcmc_diagnostic_file:
            mcmc_diagnostic_file.write('\n\n'+mcmc_finish_txt)
            mcmc_diagnostic_file.close()
        
        
    # Flatten the ensemble of walkers, remove ncut (burn in) from each
    # samples = sampler.chain[:, ncut:, :].reshape((-1, ndim))
    samples = sampler.get_chain(flat=True,discard=ncut)
    
    if MLE_init:
        if return_walkers:
            return samples, opt, sampler
        else:
            return samples, opt
    else:
        if return_walkers:
            return samples, sampler
        else:
            return samples


def _fit_dens_multiprocessing_init(_densfunc, _effsel, _Rgrid, _phigrid, _zgrid, 
                                   _Rdata, _phidata, _zdata, _usr_log_prior):
    '''_fit_dens_multiprocessing_init:

    Initialize multiprocessing for fit_dens. See fit_dens for args (no 
    underscores). Provides global variable access for multiprocessing.
    '''
    global densfunc, effsel, Rgrid, phigrid, zgrid, Rdata, phidata, zdata, \
        usr_log_prior
    densfunc = _densfunc
    effsel = _effsel
    Rgrid = _Rgrid
    phigrid = _phigrid
    zgrid = _zgrid
    Rdata = _Rdata
    phidata = _phidata
    zdata = _zdata
    usr_log_prior = _usr_log_prior
    

def mass(hf, samples=None, n_star=None, n_mass=None, int_r_range=None,
         **kwargs):
    '''mass:
    
    Lightweight wrapper of mass_from_density_samples to be used when supplying
    HaloFit object, which can account for many supplementary inputs.
    
    Args:
        hf (HaloFit) - HaloFit class containing all relevant information
        samples (np.ndarray) - Shape (n_samples, n_params) array of samples
            [default: None, which uses samples from hf]
        n_star (int) - Number of stars in sample [default: None, which uses
            n_star from hf]
        n_mass (int) - Number of masses to calculate [default: None, which
            uses n_mass from hf]
        mass_int_type (str) - Type of mass integration to use. Options are:
            'spherical_grid' - Spherical grid integration
        int_r_range (list) - Range of radii to integrate over [default: None,
            which uses int_r_range from hf]
        kwargs (dict) - Keyword arguments to pass to mass_from_density_samples
    '''
    # Handle optional inputs
    if samples is None:
        samples = hf.samples
    if n_star is None:
        n_star = hf.n_star
    if n_mass is None:
        n_mass = hf.n_mass
    if int_r_range is None:
        int_r_range = hf.int_r_range
    samples = np.atleast_2d(samples)
    
    # Handle parameters from HaloFit class
    densfunc = hf.densfunc
    effsel = hf.get_fit_effsel()
    effsel_grid = hf.get_effsel_list()
    iso = hf.get_iso()
    feh_range = hf.feh_range
    logg_range = hf.logg_range
    jkmins = hf.jkmins
    
    assert densfunc is not None
    assert effsel is not None
    assert effsel_grid is not None
    assert iso is not None
    assert feh_range is not None
    assert logg_range is not None
    assert jkmins is not None
    
    out = mass_from_density_samples(samples=samples, densfunc=densfunc, 
        n_star=n_star, effsel=effsel, effsel_grid=effsel_grid, iso=iso,
        feh_range=feh_range, logg_range=logg_range, jkmins=jkmins,
        n_mass=n_mass, int_r_range=int_r_range, **kwargs)
    return out

    
def mass_from_density_samples(samples, densfunc, n_star, effsel, effsel_grid, 
                              iso, feh_range, logg_range, jkmins, n_mass=400,
                              mass_int_type='spherical_grid', 
                              mass_analytic=False, int_r_range=[2.,70.], 
                              n_edge=[50,25,25], nprocs=None, batch=False, 
                              ro=_ro, zo=_zo, seed=0, verbose=True, 
                              _isofactors=None):
    '''mass_from_density_samples:
    
    Calculate the mass corresponding to a series of samples representing the 
    MCMC fit to a sample of stars.
    
    Args:
        samples (array) - Array of samples with size (n_samples,n_params) 
            from MCMC
        densfunc (callable) - Density function with parameters to be fit,
            has the signature (R,phi,z,params) and returns density 
            normalized to solar
        effsel (array) - Array of shape (Nfield,Ndmod) where the effective
            selection function is evaluated.
        effsel_grid (list) - Length-3 list of grids of shape (effsel) 
            representing R,phi,z positions where the selection function is 
            evaluated
        iso (np structured array) - isochrone grid (could be multiple)
        feh_range (array) - 2-element array of [feh minimum, feh maximum]
        logg_range (array)  - 2-element array of [logg minimum, logg maximum]
        jkmins (array) - array of minimum (J-Ks)0 values (length N fields)
        n_mass (int) - Number of mass calculations to create. n_mass must be 
            less than samples.shape[0]
        mass_int_type (str) - Type of integration to do for mass calculation
        mass_analytic (bool) - Analytically calculate the mass, only suitable
            for power laws.
        int_r_range (list) - List of galactocentric [r_min,r_max] for spherical 
            integration [default (2.,70.)]
        n_edge (list) - List of n_edge parameters for integration grid
        nprocs (None) - Number of processors to use in parallel
        batch (False) - Batch the parallelized computation?
        ro,vo,zo (float) - Galpy normalization parameters. ro used for 
            Sun-GC distance, zo used for height above galactic plane.
        seed (int) - Random seed for numpy [default 0]
        verbose (bool) - Be verbose?
        _isofactors ()
        
    Returns:
        masses (array) - Array of length n_mass of mass samples
        facs (array) - Array of 
    '''
    # Parameters and arrays
    assert n_mass <= samples.shape[0]
    nfield = effsel.shape[0]
    facs = np.zeros(n_mass)
    isofactors = np.zeros(nfield)
    if 'plusexpdisk' in densfunc.__name__:
        hasDisk = True
        masses = np.zeros((n_mass,3))
    else:
        hasDisk = False
        masses = np.zeros(n_mass)
    
    # Unpack effsel grid
    Rgrid,phigrid,zgrid = effsel_grid
    
    # Determine the isochrone mass fraction factors for each field
    if verbose:
        print('Calculating isochrone factors')
    
    _fast_isofactors = True
    if _isofactors is not None:
        if verbose:
            print('Warning: hard-setting isofactors')
        if isinstance(_isofactors,(list,tuple,np.ndarray)):
            isofactors = np.array(_isofactors)
        else:
            isofactors[:] = _isofactors
    else:
        if _fast_isofactors:
            unique_jkmin = np.unique(jkmins)
            if verbose:
                print('Doing fast isochrone factor calculation')
                print('Unique (J-K) minimum values: '+str(unique_jkmin))
            for i in range(len(unique_jkmin)):
                # The mass ratio mask is for all stars considered 
                massratio_isomask = (Z2FEH(iso['Zini']) > feh_range[0]) &\
                                    (Z2FEH(iso['Zini']) < feh_range[1]) &\
                                    (iso['logAge'] >= 10) &\
                                    (iso['logL'] > -9) # Eliminates WDs
                # The average mass mask extracts fitted sample based on color 
                # and logg
                avmass_isomask = massratio_isomask &\
                                (iso['Jmag']-iso['Ksmag'] > unique_jkmin[i]) &\
                                (iso['logg'] > logg_range[0]) &\
                                (iso['logg'] < logg_range[1])
                massratio = piso.mass_ratio(iso[massratio_isomask], 
                                            logg_range=logg_range,
                                            jk_range=[unique_jkmin[i],999.])
                avmass = piso.average_mass(iso[avmass_isomask])
                jkmin_mask = jkmins == unique_jkmin[i]
                isofactors[jkmin_mask] = avmass/massratio
        else:
            print('Doing slow isochrone factor calculation')
            for i in range(nfield):
                # The mass ratio mask is for all stars considered 
                massratio_isomask = (Z2FEH(iso['Zini']) > feh_range[0]) &\
                                    (Z2FEH(iso['Zini']) < feh_range[1]) &\
                                    (iso['logAge'] >= 10) &\
                                    (iso['logL'] > -9) # Eliminates WDs
                # The average mass mask extracts fitted sample based on color 
                # and logg
                avmass_isomask = massratio_isomask &\
                                (iso['Jmag']-iso['Ksmag'] > jkmins[i]) &\
                                (iso['logg'] > logg_range[0]) &\
                                (iso['logg'] < logg_range[1])
                massratio = piso.mass_ratio(iso[massratio_isomask], 
                                            logg_range=logg_range,
                                            jk_range=[jkmins[i],999.])
                avmass = piso.average_mass(iso[avmass_isomask])
                isofactors[i] = avmass/massratio
            
    if verbose:
        print(np.unique(isofactors))

    # Set up R,phi,z grid for integration
    if mass_int_type == 'cartesian_grid':
        if verbose:
            print('Cartesian grid integration not supported, using spherical grid')
        mass_int_type = 'spherical_grid'
    elif mass_int_type == 'spherical_grid':
        r_min,r_max = int_r_range
        n_edge_r,n_edge_theta,n_edge_phi = n_edge
        Rphizgrid,deltafactor = spherical_integration_grid(r_min,r_max,n_edge_r,
                                                     n_edge_theta,n_edge_phi)
    else:
        assert False, 'Must choose "spherical_grid" or "cartesian_grid"'

    # Calculate the mass
    if verbose:
        print('Calculating mass')
    
    if n_mass < samples.shape[0]:
        np.random.seed(seed)
        mass_inds = np.random.choice(samples.shape[0], n_mass, replace=False)
    else:
        mass_inds = np.arange(n_mass,dtype=int)
    samples_randind = samples[mass_inds]
    
    # Parallel?
    if nprocs and nprocs > 1:
        # Batch computing?
        if batch and (samples_randind.shape[0]%nprocs==0):
            if verbose:
                print('Batch processing masses in parallel')
            _calc_mass_generator = zip(samples_randind.reshape(
                                           nprocs,
                                           int(samples_randind.shape[0]/nprocs),
                                           samples_randind.shape[1]),
                                       itertools.repeat(densfunc),
                                       itertools.repeat(n_star),
                                       itertools.repeat(effsel),
                                       itertools.repeat(effsel_grid),
                                       itertools.repeat(isofactors),
                                       itertools.repeat(Rphizgrid),
                                       itertools.repeat(deltafactor),
                                       itertools.repeat(n_mass)
                                      )
            counter = multiprocessing.Value('i', 0)
            with multiprocessing.Pool(processes=nprocs,
                                      initializer=_calc_mass_multiprocessing_init,
                                      initargs=(counter,)) as p:
                results = p.starmap(_calc_mass_batch, _calc_mass_generator)
                masses,facs = np.moveaxis(np.array(results),2,1).reshape(
                    samples_randind.shape[0],2).T
        # No batch computing
        else:
            if verbose:
                print('Processing masses in parallel, no batching')
            _calc_mass_generator = zip(samples_randind,
                                       itertools.repeat(densfunc),
                                       itertools.repeat(n_star),
                                       itertools.repeat(effsel),
                                       itertools.repeat(effsel_grid),
                                       itertools.repeat(isofactors),
                                       itertools.repeat(Rphizgrid),
                                       itertools.repeat(deltafactor),
                                       itertools.repeat(n_mass)
                                      )
            counter = multiprocessing.Value('i', 0)
            with multiprocessing.Pool(processes=nprocs,
                                      initializer=_calc_mass_multiprocessing_init,
                                      initargs=(counter,)) as p:
                results = p.starmap(_calc_mass, _calc_mass_generator)
                masses,facs = np.array(results).T
    # Serial        
    else:
        if verbose:
            print('Processing masses in serial')
        for i,params in enumerate(samples_randind):
            if verbose:
                if (i+1)%10 == 0: print('calculated '+str(i+1)+'/'+str(n_mass), 
                                        end='\r')
            rate = densfunc(Rgrid,phigrid,zgrid,params=params)*effsel
            sumrate = np.sum(rate.T/isofactors)
            fac = n_star/sumrate
            if mass_analytic:
                # Only for spherical power law!
                if not densfunc is pdens.spherical:
                    warnings.warn('mass_analytic=True only recommended for '+\
                                  'spherical power law!')
                rsun = np.sqrt(ro**2+zo**2)
                r_min,r_max = int_r_range
                alpha = params[0] # Assume alpha is the first parameter?
                integral = 4*np.pi*rsun**alpha*((r_max**(3-alpha))/(3-alpha)-\
                                                (r_min**(3-alpha))/(3-alpha))
                masses[i] = integral*fac
            if 'plusexpdisk' in densfunc.__name__: 
                denstxyz = densfunc(Rphizgrid[:,0],Rphizgrid[:,1],
                                    Rphizgrid[:,2], params=params, 
                                    split=True)
                halodens = denstxyz[0]*fac
                diskdens = denstxyz[1]*fac
                fulldens = densfunc(Rphizgrid[:,0],Rphizgrid[:,1],
                                    Rphizgrid[:,2], params=params)*fac
                masses[i,0] = np.sum(halodens*deltafactor)
                masses[i,1] = np.sum(diskdens*deltafactor)
                masses[i,2] = np.sum(fulldens*deltafactor)
            else:
                # This is the most time consuming part of the mass calculation,
                # takes about 1 second total, about 94% of the time
                denstxyz = densfunc(Rphizgrid[:,0],Rphizgrid[:,1],
                                    Rphizgrid[:,2], params=params)*fac
                masses[i] =  np.sum(denstxyz*deltafactor)
            facs[i] = fac
            # Also maybe some sort of actual integrator?
    
    print('Calculated '+str(n_mass)+'/'+str(n_mass)+' masses')
    return masses, facs, mass_inds, isofactors


def _calc_mass_multiprocessing_init(args):
    '''_calc_mass_multiprocessing_init:
    
    Initialize a counter for the multiprocessing call
    '''
    global counter
    counter = args
    
    
def _calc_mass(params, densfunc, n_star, effsel, effsel_grid, isofactors, 
               Rphizgrid, deltafactor, n_mass):
    '''_calc_mass:
    
    Calculate the mass using spherical grid integration for one set of 
    parameters
    
    Args:
        params (array) - array of parameter arguments to densfunc
        densfunc (callable) - Density function with parameters to be fit,
            has the signature (R,phi,z,params) and returns density 
            normalized to solar
        n_star (int) - Number of stars in the fitting sample
        effsel (array) - Array of shape (Nfield,Ndmod) where the effective
            selection function is evaluated.
        effsel_grid (list) - Length-3 list of grids of shape (effsel) 
            representing R,phi,z positions where the selection function is 
            evaluated
        isofactors (array) - isochrone factors average_mass / mass_ratio
        Rphizgrid (array) - Array of shape (N,3) where N is nr*ntheta*nphi 
            of cylindrical R,phi,z coordinates for the grid
        deltafactor (array) - Array of shape (N) where N is nr*ntheta*nphi of 
            integral delta factors: r^2 sin(theta) dr dtheta dphi
    
    Returns:
        mass (float) - Mass in Msun
        fac (float) - factor to convert from normalized density to Msun/
            pc**3
    '''
    # Unpack
    Rgrid,phigrid,zgrid = effsel_grid
    # Note effsel must have area factors and Jacobians applied!
    rate = densfunc(Rgrid,phigrid,zgrid,params=params)*effsel
    sumrate = np.sum(rate.T/isofactors)
    fac = n_star/sumrate
    if 'plusexpdisk' in densfunc.__name__: 
        denstxyz = densfunc(Rphizgrid[:,0],Rphizgrid[:,1],Rphizgrid[:,2], 
                            params=params, split=True)
        halodens = denstxyz[0]*fac
        diskdens = denstxyz[1]*fac
        fulldens = densfunc(Rphizgrid[:,0],Rphizgrid[:,1],Rphizgrid[:,2], 
                         params=params)*fac
        mass = [np.sum(halodens*deltafactor),
                np.sum(diskdens*deltafactor),
                np.sum(fulldens*deltafactor)]
    else:
        denstxyz = densfunc(Rphizgrid[:,0],Rphizgrid[:,1],
                            Rphizgrid[:,2], params=params)*fac
        mass =  np.sum(denstxyz*deltafactor)
    # Increment counter
    global counter
    with counter.get_lock():
        counter.value += 1
    if (counter.value)%10 == 0: print('Calculated '+str(counter.value)+'/'+\
                                      str(n_mass)+' masses', end='\r')
    return mass,fac


def _calc_mass_batch(params, densfunc, n_star, effsel, effsel_grid, isofactors, 
                     Rphizgrid, deltafactor, n_mass):
    '''_calc_mass_batch:
    
    Calculate the mass using spherical grid integration for multiple sets of 
    parameters. This allows the parallelization to pickle fewer things.
    
    Args:
        params (array) - shape (Ns,Np) array of Ns sets of Np parameter 
            arguments to densfunc
        densfunc (callable) - Density function with parameters to be fit,
            has the signature (R,phi,z,params) and returns density 
            normalized to solar
        n_star (int) - Number of stars in the fitting sample
        effsel (array) - Array of shape (Nfield,Ndmod) where the effective
            selection function is evaluated.
        effsel_grid (list) - Length-3 list of grids of shape (effsel) 
            representing R,phi,z positions where the selection function is 
            evaluated
        isofactors (array) - isochrone factors average_mass / mass_ratio
        Rphizgrid (array) - Array of shape (N,3) where N is nr*ntheta*nphi 
            of cylindrical R,phi,z coordinates for the grid
        deltafactor (array) - Array of shape (N) where N is nr*ntheta*nphi of 
            integral delta factors: r^2 sin(theta) dr dtheta dphi
    
    Returns:
        masses (float) - Mass in Msun
        fac (float) - factor to convert from normalized density to Msun/
            pc**3
    '''
    # Unpack
    Rgrid,phigrid,zgrid = effsel_grid
    if 'plusexpdisk' in densfunc.__name__:
        hasDisk = True
        #masses = np.zeros((params.shape[0],3))
        masses = np.zeros(params.shape[0])
    else:
        hasDisk = False
        masses = np.zeros(params.shape[0])
    facs = np.zeros(params.shape[0])
    for i in range(params.shape[0]):
        # Note effsel must have area factors and Jacobians applied!
        rate = densfunc(Rgrid,phigrid,zgrid,params=params[i])*effsel
        sumrate = np.sum(rate.T/isofactors)
        facs[i] = n_star/sumrate
        if 'plusexpdisk' in densfunc.__name__: 
            denstxyz = densfunc(Rphizgrid[:,0],Rphizgrid[:,1],Rphizgrid[:,2], 
                                params=params[i], split=True)
            halodens = denstxyz[0]*facs[i]
            #diskdens = denstxyz[1]*facs[i]
            #fulldens = densfunc(Rphizgrid[:,0],Rphizgrid[:,1],Rphizgrid[:,2], 
            #                 params=params[i])*facs[i]
            masses[i] = np.sum(halodens*deltafactor)
            #masses[i,0] = np.sum(halodens*deltafactor)
            #masses[i,1] = np.sum(diskdens*deltafactor)
            #masses[i,2] = np.sum(fulldens*deltafactor)
        else:
            denstxyz = densfunc(Rphizgrid[:,0],Rphizgrid[:,1],
                                Rphizgrid[:,2], params=params[i])*facs[i]
            masses[i] =  np.sum(denstxyz*deltafactor)
        # Increment counter
        global counter
        with counter.get_lock():
            counter.value += 1
        if counter.value%10 == 0: print('Calculated '+str(counter.value)+'/'+\
                                        str(n_mass)+' masses', end='\r')
    return masses,facs


def spherical_integration_grid(r_min,r_max,n_edge_r,n_edge_theta,n_edge_phi):
    '''spherical_integration_grid:
    
    Make a spherical integration grid. For shallow power laws (alpha<3) 
    n_edge_r can be ~ 200. For steep power laws (alpha>3) and low r_min 
    (< 1kpc) consider n_edge_r ~ 500. Note the number of bins for each grid is 
    the number of edges - 1
    
    
    Args:
        r_min (float) - Minimum radius for the grid
        r_max (float) - Maximum radius for the grid
        n_edge_r (int) - Number of edges for the radial grid (n_bin = n_edge-1)
        n_edge_theta (int) - Number of edges for the theta grid (polar)
        n_edge_phi (int) - NUmber of edges for the phi grid (azimuthal)
    
    Returns:
        Rphizgrid (array) - Array of shape (N,3) where N is nr*ntheta*nphi 
            of cylindrical R,phi,z coordinates for the grid
        delta (array) - Array of shape (N) where N is nr*ntheta*nphi of 
            integral delta factors: r^2 sin(theta) dr dtheta dphi
    '''
    rgrid = np.linspace(r_min,r_max,n_edge_r)
    dr = np.diff(rgrid)[0]
    rgrid += dr/2
    thetagrid = np.linspace(0,np.pi,n_edge_theta)
    dtheta = np.diff(thetagrid)[0]
    thetagrid += dtheta/2
    phigrid = np.linspace(0,2*np.pi,n_edge_phi)
    dphi = np.diff(phigrid)[0]
    phigrid += dphi/2
    rthetaphigrid = np.asarray(np.meshgrid(rgrid[:-1],thetagrid[:-1],phigrid[:-1]))
    nbin = (n_edge_r-1)*(n_edge_theta-1)*(n_edge_phi-1)
    rthetaphigrid = rthetaphigrid.reshape(3,nbin).T
    delta = rthetaphigrid[:,0]**2*np.sin(rthetaphigrid[:,1])*dr*dtheta*dphi
    x = rthetaphigrid[:,0]*np.sin(rthetaphigrid[:,1])*np.cos(rthetaphigrid[:,2])
    y = rthetaphigrid[:,0]*np.sin(rthetaphigrid[:,1])*np.sin(rthetaphigrid[:,2])
    z = rthetaphigrid[:,0]*np.cos(rthetaphigrid[:,1])
    xyzgrid = np.dstack([x,y,z])[0]
    Rphizgrid = coords.rect_to_cyl(xyzgrid[:,0], xyzgrid[:,1], xyzgrid[:,2])
    Rphizgrid = np.dstack([Rphizgrid[0],Rphizgrid[1],Rphizgrid[2]])[0]
    return Rphizgrid,delta


def double_exponential_disk_cylindrical_mass(hR,hz,Rmin,Rmax,zmax,A=1.):
    '''double_exponential_disk_cylindrical_mass:

    Calculate the mass of a double exponential disk in cylindrical coordinates.
    Assume zmin=0 and that the vertical integration is symmetric about z=0.
    Arguments can be astropy quantities.

    Args:
        hR (float) - Scale length of the disk in cylindrical R
        hz (float) - Scale length of the disk in cylindrical z

    '''
    if isinstance(hR,apu.quantity.Quantity):
        hR = hR.to(apu.kpc).value
    if isinstance(hz,apu.quantity.Quantity):
        hz = hz.to(apu.kpc).value
    if isinstance(Rmin,apu.quantity.Quantity):
        Rmin = Rmin.to(apu.kpc).value
    if isinstance(Rmax,apu.quantity.Quantity):
        Rmax = Rmax.to(apu.kpc).value
    if isinstance(zmax,apu.quantity.Quantity):
        zmax = zmax.to(apu.kpc).value
    m = 4*np.pi*hR*hz*(np.exp(-Rmin/hR)*(Rmin+hR)-np.exp(-Rmax/hR)*(Rmax+hR))*\
        (1-np.exp(-zmax/hz))
    return A*m


def fdisk_to_number_of_stars(hf,samples=None,load_blobbed_effvol=True,
                             effvol_halo=None,effvol_disk=None,indx=None):
    '''fdisk_to_number_of_stars:
    
    Convert fdisk to number of halo and disk stars in the sample
    
    Args:
        hf (HaloFit) - HaloFit class containing all information about fit
        samples (array) - MCMC samples, shape is (nsample,ndim)
        load_blobbed_effvol (bool) - Use blobbed effective volume from the 
            sampler object if possible.
        effvol_halo (array) - Effective volume for halo stars, shape is 
            (nsample)
        effvol_disk (array) - Effective volume for disk stars, shape is
            (nsample)
        indx (array) - Index of samples to use. If None, use all samples.
    
    Returns:
        n_halo (int) - Number of halo stars
        n_disk (int) - Number of disk stars
    '''
    _loaded_samples = False
    _loaded_blobbed_effvol = False
    _has_effvol = False
    assert not ((effvol_halo is not None)^(effvol_disk is not None)),\
        'Must supply both effvol_halo and effvol_disk'
    if samples is None:
        print('Did not provide samples, using all samples from HaloFit object')
        _loaded_samples = True
        if hf.samples is None:
            hf.get_results()
            assert hf.samples is not None,\
                'No samples in supplied HaloFit instance, tried get_results()'
        samples = hf.samples
    if load_blobbed_effvol:
        if not _loaded_samples:
            print('Not loading blobbed effective volume because samples were'+
                  ' supplied, no way to find correspondance between loaded'+
                  ' effvol and samples. Supply corresponding effvol_halo and'+
                  ' effvol_disk.')
        else:
            _loaded_blobbed_effvol = True
            _has_effvol = True
            if hf.effvol_halo is None or hf.effvol_disk is None:
                hf.get_results(load_blobbed_effvol=True)
                assert hf.effvol_halo is not None\
                    and hf.effvol_disk is not None,\
                    'No blobbed effective volume in supplied HaloFit '+\
                    ' instance, tried get_results(load_blobbed_effvol=True)'
            effvol_halo = hf.effvol_halo
            effvol_disk = hf.effvol_disk
    if effvol_disk is not None:
        _has_effvol = True
        if _loaded_blobbed_effvol:
            print('Not using supplied effvol because loaded blobbed effvol')
    if indx is not None:
        print('indx was supplied, applying to samples and effvol')
        samples = samples[indx]
        if _has_effvol:
            effvol_halo = effvol_halo[indx]
            effvol_disk = effvol_disk[indx]

    samples = np.atleast_2d(samples)
    n_samples = samples.shape[0]
    n_star_halo = np.zeros(n_samples)
    n_star_disk = np.zeros(n_samples)
    
    assert ('plusexpdisk' in hf.densfunc.__name__),\
        'densfunc must have disk contamination (plusexpdisk) to have fdisk'
    
    # Unpack effective selection function and effective selection function grid
    if not _has_effvol:
        Rgrid,phigrid,zgrid = hf.get_effsel_list()
        effsel = hf.get_fit_effsel()
    n_star = hf.n_star

    if _has_effvol:
        vol_tot = effvol_halo + effvol_disk
        n_star_halo = n_star*effvol_halo/vol_tot
        n_star_disk = n_star*effvol_disk/vol_tot
    else:
        # Calculate the effective volume for both profiles
        for i in tqdm(range(n_samples)):
            dens_halo,dens_disk = hf.densfunc(Rgrid, phigrid, zgrid, 
                params=samples[i], split=True)
            vol_halo = np.sum(dens_halo*effsel)
            vol_disk = np.sum(dens_disk*effsel)
            vol_tot = vol_halo+vol_disk
            n_star_halo[i] = n_star*vol_halo/vol_tot
            n_star_disk[i] = n_star*vol_disk/vol_tot
    
    return n_star_halo,n_star_disk


### Gridding


def Rphizgrid(apo,distmods,ro=_ro,zo=_zo):
    '''Rphizgrid:
    
    Create a grid of R, phi, and z for each field in the APOGEE selection
    function corresponding to the range of distance moduli supplied.
    
    Args:
        apo (apogee.select.apogeeSelect) - APOGEE selection function
        distmods (array) - Array of distance moduli
        ro (float) - Galactocentric radius of the Sun
        zo (float) - Height of the Sun above the Galactic plane
    
    Returns:
        Rgrid (array) - Array of R positions
        phigrid (array) - Array of phi positions
        zgrid (array) - Array of z positions
    '''
    ds = 10**(distmods/5.-2.)
    Rgrid = np.zeros((len(apo._locations),len(ds)))
    phigrid = np.zeros((len(apo._locations),len(ds)))
    zgrid = np.zeros((len(apo._locations),len(ds)))
    for i in range(len(apo._locations)):
        glon,glat = apo.glonGlat(apo._locations[i])
        glon = np.ones(len(ds))*glon[0]
        glat = np.ones(len(ds))*glat[0]
        xyz = coords.lbd_to_XYZ(glon,glat,ds, degree=True)
        rphiz = coords.XYZ_to_galcencyl(xyz[:,0], xyz[:,1], xyz[:,2], 
                                             Xsun=ro, Zsun=zo)
        Rgrid[i] = rphiz[:,0]
        phigrid[i] = rphiz[:,1]
        zgrid[i] = rphiz[:,2]
    return Rgrid, phigrid, zgrid


def xyzgrid(apo,distmods,):
    '''xyzgrid:
    
    Create a grid of x, y, and z for each field in the APOGEE selection
    function corresponding to the range of distance moduli supplied.
    
    Args:
        apo (apogee.select.apogeeSelect) - APOGEE selection function
        distmods (array) - Array of distance moduli
    
    Returns:
        xgrid (array) - Array of x positions
        ygrid (array) - Array of y positions
        zgrid (array) - Array of z positions
    '''
    ds = 10**(distmods/5.-2.)
    xgrid = np.zeros((len(apo._locations),len(ds)))
    ygrid = np.zeros((len(apo._locations),len(ds)))
    zgrid = np.zeros((len(apo._locations),len(ds)))
    for i in range(len(apo._locations)):
        glon,glat = apo.glonGlat(apo._locations[i])
        glon = np.ones(len(ds))*glon[0]
        glat = np.ones(len(ds))*glat[0]
        xyz = coords.lbd_to_XYZ(glon,glat,ds,degree=True)
        xgrid[i] = xyz[:,0]
        ygrid[i] = xyz[:,1]
        zgrid[i] = xyz[:,2]
    return xgrid, ygrid, zgrid


### Model Likelihoods

def mloglike_mp(params):
    '''mloglike_mp:

    Negative log-likelihood for the inhomogeneous Poisson point process. The 
    multiprocessing version of the function for emcee.
    
    Args:
        args (args) - Arguments to pass to loglike
        kwargs (kwargs) - Keyword arguments to pass to loglike
    
    Returns:
        mloglike (array) - Negative of the loglikelihood function
    '''
    res = loglike_mp(params)
    return (-res[0],*res[1:])

def loglike_mp(params): 
    '''loglike_mp:
    
    log-likelihood for the inhomogeneous Poisson point process. Accounts for
    the prior. The multiprocessing version of the function for emcee.
    
    Args:
        params (list) - Density model parameters
    
    Returns:
        log_posterior (array) - log of the likelihood + log of the prior
    '''
    global densfunc, effsel, Rgrid, phigrid, zgrid, Rdata, phidata, zdata, usr_log_prior

    # Check for disk component
    hasDisk = False
    if 'plusexpdisk' in densfunc.__name__:
        hasDisk = True
    # Evaluate the domain of the prior
    if not domain_prior(densfunc, params):
        return -np.inf, 0., 0.
    # Evaluate the user-supplied prior
    if callable(usr_log_prior):
        usrlogprior = usr_log_prior(densfunc,params)
        if np.isneginf(usrlogprior):
            return usrlogprior, 0., 0.
    else:
        usrlogprior = 0
    # Evaluate the informative prior
    logprior = log_prior(densfunc, params)
    logdatadens = np.log(tdens(densfunc, Rdata, phidata, zdata, params=params))
    # log effective volume
    if hasDisk:
        effvol_halo,effvol_disk = effvol(densfunc,effsel,Rgrid,phigrid,zgrid,
            params=params,split=True)
    else:
        effvol_halo = effvol(densfunc,effsel,Rgrid,phigrid,zgrid,params=params)
        effvol_disk = 0.
    logeffvol = np.log(effvol_halo+effvol_disk)
    # log likelihood
    loglike = np.sum(logdatadens)-len(Rdata)*logeffvol
    if not np.isfinite(loglike):
        return -np.inf, effvol_halo, effvol_disk
    logjointprob = logprior + usrlogprior + loglike
    return logjointprob, effvol_halo, effvol_disk

def mloglike(*args, **kwargs):
    '''mloglike:
    
    Args:
        args (args) - Arguments to pass to loglike
        kwargs (kwargs) - Keyword arguments to pass to loglike
    
    Returns:
        mloglike (array) - Negative of the loglikelihood function
    '''
    return -loglike(*args,**kwargs)


def loglike(params, densfunc, effsel, Rgrid, phigrid, zgrid, Rdata, phidata, 
            zdata, usr_log_prior=None, return_effvol=False):
    '''loglike:
    
    log-likelihood for the inhomogeneous Poisson point process. Accounts for
    the prior.
    
    Args:
        params (list) - Density model parameters
        densfunc (function) - Density function
        effsel (array) - Effective selection function (Nfield x Ndistmod)
        Rgrid (array) - Array of R positions for the effective selection function
        phigrid (array) - Array of phi positions for the effective selection function
        zgrid (array) - Array of z positions for the effective selection function
        Rdata (array) - data R positions
        phidata (array) - data phi positions
        zdata (array) - data z positions
        usr_log_prior (callable) - Extra prior supplied by the user at runtime. 
            Included for flexibility so not all priors need to be hardcoded. 
            Call signature should be usr_log_prior(densfunc,params) and function 
            should return the log of the prior value. Will check for -np.inf 
            and break out of the likelihood call if it is returned. 
            [default: None]
        return_effvol (bool) - Return the effective volume of the halo and disk 
            [default: False]
    
    Returns:
        log_posterior (array) - log of the likelihood + log of the prior
    '''
    # Check for disk component
    hasDisk = False
    if 'plusexpdisk' in densfunc.__name__:
        hasDisk = True
    # Evaluate the domain of the prior
    if not domain_prior(densfunc, params):
        if return_effvol:
            return -np.inf, 0., 0.
        else:
            return -np.inf
    # Evaluate the user-supplied prior
    if callable(usr_log_prior):
        usrlogprior = usr_log_prior(densfunc,params)
        if np.isneginf(usrlogprior):
            if return_effvol:
                return usrlogprior, 0., 0.
            else:
                return usrlogprior
    else:
        usrlogprior = 0
    # Evaluate the informative prior
    logprior = log_prior(densfunc, params)
    logdatadens = np.log(tdens(densfunc, Rdata, phidata, zdata, params=params))
    # log effective volume
    if hasDisk:
        effvol_halo,effvol_disk = effvol(densfunc,effsel,Rgrid,phigrid,zgrid,
            params=params,split=True)
    else:
        effvol_halo = effvol(densfunc,effsel,Rgrid,phigrid,zgrid,params=params)
        effvol_disk = 0.
    logeffvol = np.log(effvol_halo+effvol_disk)
    # log likelihood
    loglike = np.sum(logdatadens)-len(Rdata)*logeffvol
    if not np.isfinite(loglike):
        if return_effvol:
            return -np.inf, effvol_halo, effvol_disk
        else:
            return -np.inf
    logjointprob = logprior + usrlogprior + loglike
    if return_effvol:
        return logjointprob, effvol_halo, effvol_disk
    else:
        return logjointprob


def effvol(densfunc, effsel, Rgrid, phigrid, zgrid, params=None, split=False):
    '''effvol:
    
    Returns the effective volume given a density function, an effective
    selection function, and the grid on which it was evaluated.
    
    Args:
        densfunc (function) - Density function
        effsel (array) - Effective selection function (Nfield x Ndistmod)
        Rgrid (array) - Array of R positions
        phigrid (array) - Array of phi positions
        zgrid (array) - Array of z positions
        params (list) - Density model parameters
        split (bool) - Return the effective volume for the halo and disk 
            components separately
    
    Returns:
        effvol (array) - The effective volume 
    '''
    if split:
        if params is None:
            effdens = tdens(densfunc,Rgrid,phigrid,zgrid,split=split)
        else:
            effdens = tdens(densfunc,Rgrid,phigrid,zgrid,params=params,
                split=split)
        return np.sum(effdens[0]*effsel), np.sum(effdens[1]*effsel)
    else:
        if params is None:
            effdens = tdens(densfunc,Rgrid,phigrid,zgrid)
        else:
            effdens = tdens(densfunc,Rgrid,phigrid,zgrid,params=params)
        return np.sum(effdens*effsel)


def tdens(densfunc, Rgrid, phigrid, zgrid, params=None, split=False):
    '''tdens:
    
    Deterine the densities at the locations corresponding to a supplied grid
    and density function.
    
    Args:
        densfunc (function) - Density function
        Rgrid (array) - Array of R positions
        phigrid (array) - Array of phi positions
        zgrid (array) - Array of z positions
        params (list) - Density model parameters
        split (bool) - split the density function into two parts
    
    Returns:
        dens (array) - Densities corresponding to supplied positions and density
            function
    '''
    if params is None:
        if split:
            dens = densfunc(Rgrid,phigrid,zgrid,split=split)
        else:
            dens = densfunc(Rgrid,phigrid,zgrid)
    else:
        if split:
            dens = densfunc(Rgrid,phigrid,zgrid,params=params,split=split)
        else:
            dens = densfunc(Rgrid,phigrid,zgrid,params=params)
    return dens


### Prior


def log_prior(densfunc, params):
    '''log_prior:
    
    Calculate the informative prior for the given density model and parameters.
    Only for triaxial single angle and einasto
    
    Note: Informative prior is currently suppressed, but could change in the 
    future.
    
    Args:
        densfunc (function) - Density function
        params (list) - List of parameters for the density function
    
    Returns:
        prior (bool) - Boolean representing the uninformative prior
    '''
    """
    check the (informative) prior for the given density model and parameters.
    """
    return 0
    # if densfunc is pdens.triaxial_single_angle_zvecpa:
    #     prior = norm.pdf(params[0], loc=2.5, scale=1)
    #     return np.log(prior)
    # if densfunc is pdens.triaxial_einasto_zvecpa:
    #     prior = norm.pdf(params[0], loc=20, scale=10)
    #     return np.log(prior)


def domain_prior(densfunc, params):
    '''domain_prior:
    
    Evaluate an uninformative domain prior on the parameters of each density 
    profile
    
    Args:
        densfunc (function) - Density function
        params (list) - List of parameters for the density function
    
    Returns:
        prior (bool) - Boolean corresponding to the uninformative prior    
    '''
    alpha_positive = True # Force alpha positive
    cutoff_alpha_excptn = True # Exception to above for exponential cutoff
    broken_laws_steepen = True # Force broken power laws to steepen with radius
    
    if densfunc.__name__ == 'spherical':
        a, = params[:1]
        if a < 0. and alpha_positive: return False
    
    elif densfunc.__name__ == 'spherical_cutoff':
        a,b = params[:2]
        if a < 0. and alpha_positive: return False
        if b < 0.: return False
    
    elif densfunc.__name__ == 'axisymmetric':
        a,q = params[:2]
        if a < 0. and alpha_positive: return False
        if q < 0.1: return False
        if q > 1.: return False
    
    elif densfunc.__name__ == 'triaxial_norot':
        a,p,q = params[:3]
        if a < 0. and alpha_positive: return False
        if p < 0.1: return False
        if q > 1.: return False
        if p < 0.1: return False
        if q > 1.: return False
    
    elif 'triaxial_single_angle_zvecpa' in densfunc.__name__:
        a,p,q,th,et,pa = params[:6]
        if a < 0. and alpha_positive: return False
        if p < 0.1: return False
        if p > 1.: return False
        if q < 0.1: return False
        if q > 1.: return False
        if th <= 0.: return False
        if th >= 1.: return False
        if et <= _PRIOR_ETA_MIN: return False
        if et >= 1.: return False
        if pa <= 0.: return False
        if pa >= 1.: return False
        
    elif 'triaxial_single_cutoff_zvecpa' in densfunc.__name__:
        a,r,p,q,th,et,pa = params[:7]
        if a < 0. and alpha_positive and not cutoff_alpha_excptn: return False
        if r <= 0.: return False
        if p < 0.1: return False
        if p > 1.: return False
        if q < 0.1: return False
        if q > 1.: return False
        if th <= 0.: return False
        if th >= 1.: return False
        if et <= _PRIOR_ETA_MIN: return False
        if et >= 1.: return False
        if pa <= 0.: return False
        if pa >= 1.: return False
    
    elif 'triaxial_single_cutoff_zvecpa_inv' in densfunc.__name__:
        a,b,p,q,th,et,pa = params[:7]
        if a < 0. and alpha_positive and not cutoff_alpha_excptn: return False
        if b <= 0.: return False
        if p < 0.1: return False
        if p > 1.: return False
        if q < 0.1: return False
        if q > 1.: return False
        if th <= 0.: return False
        if th >= 1.: return False
        if et <= _PRIOR_ETA_MIN: return False
        if et >= 1.: return False
        if pa <= 0.: return False
        if pa >= 1.: return False
        
    elif 'triaxial_broken_angle_zvecpa' in densfunc.__name__:
        a1,a2,r,p,q,th,et,pa = params[:8]
        if a1 < 0. and alpha_positive: return False
        if a2 < 0. and alpha_positive: return False
        if a2 <= a1 and broken_laws_steepen: return False
        if r <= 0.: return False
        if p < 0.1: return False
        if p > 1.: return False
        if q < 0.1: return False
        if q > 1.: return False
        if th <= 0.: return False
        if th >= 1.: return False
        if et <= _PRIOR_ETA_MIN: return False
        if et >= 1.: return False
        if pa <= 0.: return False
        if pa >= 1.: return False
    
    elif 'triaxial_broken_angle_zvecpa_inv' in densfunc.__name__:
        a1,a2,b,p,q,th,et,pa = params[:8]
        if a1 < 0. and alpha_positive: return False
        if a2 < 0. and alpha_positive: return False
        if a2 <= a1 and broken_laws_steepen: return False
        if b <= 0.: return False
        if p < 0.1: return False
        if p > 1.: return False
        if q < 0.1: return False
        if q > 1.: return False
        if th <= 0.: return False
        if th >= 1.: return False
        if et <= _PRIOR_ETA_MIN: return False
        if et >= 1.: return False
        if pa <= 0.: return False
        if pa >= 1.: return False
        
    elif 'triaxial_double_broken_angle_zvecpa' in densfunc.__name__:
        a1,a2,a3,r1,r2,p,q,th,et,pa = params[:10]
        if a1 < 0. and alpha_positive: return False
        if a2 < 0. and alpha_positive: return False
        if a3 < 0. and alpha_positive: return False
        if a2 <= a1 and broken_laws_steepen: return False
        if a3 <= a2 and broken_laws_steepen: return False
        if r1 <= 0.: return False
        if r2 <= r1: return False
        if p < 0.1:return False
        if p > 1.:return False
        if q < 0.1:return False
        if q > 1.:return False
        if th <= 0.:return False
        if th >= 1.:return False
        if et <= _PRIOR_ETA_MIN:return False
        if et >= 1.:return False
        if pa <= 0.:return False
        if pa >= 1.:return False
    
    elif 'triaxial_double_broken_angle_zvecpa_inv' in densfunc.__name__:
        a1,a2,a3,r1,r2,p,q,th,et,pa = params[:10]
        if a1 < 0. and alpha_positive: return False
        if a2 < 0. and alpha_positive: return False
        if a3 < 0. and alpha_positive: return False
        if a2 <= a1 and broken_laws_steepen: return False
        if a3 <= a2 and broken_laws_steepen: return False
        if r1 <= 0.: return False
        if r2 <= r1: return False
        if p < 0.1:return False
        if p > 1.:return False
        if q < 0.1:return False
        if q > 1.:return False
        if th <= 0.:return False
        if th >= 1.:return False
        if et <= _PRIOR_ETA_MIN:return False
        if et >= 1.:return False
        if pa <= 0.:return False
        if pa >= 1.:return False
        
    elif 'triaxial_single_trunc_zvecpa' in densfunc.__name__:
        a,r,p,q,th,et,pa = params[:7]
        if a < 0. and alpha_positive:return False
        if r <= 0.:return False
        if p < 0.1:return False
        if p > 1.:return False
        if q < 0.1:return False
        if q > 1.:return False
        if th <= 0.:return False
        if th >= 1.:return False
        if et <= _PRIOR_ETA_MIN:return False
        if et >= 1.:return False
        if pa <= 0.:return False
        if pa >= 1.:return False
    
    else:
        warnings.warn('Domain prior not defined for this density profile')
    
    if 'plusexpdisk' in densfunc.__name__:
        fdisk = params[-1]
        if fdisk < 0.: return False
        if fdisk > 1.: return False
    
    return True


### Distance Modulus Posterior for Model

def pdistmod_one_model(densfunc, params, effsel, Rgrid, phigrid, zgrid, distmods,
                   return_rate=False):
    '''pdistmod_one_model:
    
    Return the expected distance modulus distribution for a given model 
    including the effective selection function. Assume the effective selection
    function already has the distance modulus Jacobian applied (only the factor 
    of d**3 which matters).
    
    Args:
        densfunc (function) - Density profile
        params (list) - Parameters that describe the density model
        effsel (array) - Effective selection function (Nfield x Ndistmod)
        Rgrid (array) - Grid of R corresponding to effsel
        phigrid (array) - Grid of phi corresponding to effsel
        zgrid (array) - Grid of z corresponding to effsel
        distmods (array) - Grid of distance moduli
        return_rate (bool) - Return the rate function as well
    
    Returns:
        pd (array) - Normalized number of counts summed over all fields
        pdt (array) - Number of counts summed over all fields
        rate (array) - Raw number of counts per field, only if return_rate=True
    '''
    rate = densfunc(Rgrid,phigrid,zgrid,params=params)*effsel
    pdt = np.sum(rate,axis=0)
    pd = pdt/np.sum(pdt)/(distmods[1]-distmods[0])
    if return_rate:
        return pd, pdt, rate
    else:
        return pd, pdt

def pdistmod_sample(densfunc, samples, effsel, Rgrid, phigrid, zgrid,
                    distmods, return_rate=False, verbose=False):
    '''pdistmod_sample:
    
    Return the expected distance modulus distribution for a set of models 
    described by distribution of parameters, including the effective 
    selection function. Assume the effective selection function already 
    has the distance modulus Jacobian applied (only the factor of d**3 which 
    matters).
    
    Args:
        densfunc (function) - Density profile
        samples (array) - (Nsamples,params) shaped array to draw parameters from
        effsel (array) - Effective selection function (Nfield x Ndistmod)
        Rgrid (array) - Grid of R corresponding to effsel
        phigrid (array) - Grid of phi corresponding to effsel
        zgrid (array) - Grid of z corresponding to effsel
        distmods (array) - Grid of distance moduli
        return_rate (bool) - Return the rate function as well.
        verbose (bool) - Be verbose? [default False]
    
    Returns:
        pd (array) - Normalized number of counts summed over all fields
        pdt (array) - Number of counts summed over all fields
        rate (array) - Raw number of counts per field, only if return_rate=True
    '''
    samples = np.atleast_2d(samples)
    n_samples = samples.shape[0]
    pd = np.zeros((n_samples,effsel.shape[1]))
    pdt = np.zeros((n_samples,effsel.shape[1]))
    rate = np.zeros((n_samples,effsel.shape[0],effsel.shape[1]))
    for i,params in enumerate(samples):
        _pd,_pdt,_r = pdistmod_one_model(densfunc, params, effsel, Rgrid, 
                                         phigrid, zgrid, distmods,
                                         return_rate=True)
        pd[i,:] = _pd
        pdt[i,:] = _pdt
        rate[i,:,:] = _r
        if verbose:
            print('Calculated pdistmod for '+str(i+1)+'/'+str(n_samples),
                  end='\r')
    if return_rate:
        return pd, pdt, rate
    else:
        return pd, pdt


### Fitting class

class _HaloFit:
    '''_HaloFit:
    
    Parent class for HaloFit-type classes
    '''
    def __init__(self,
                 densfunc=None,
                 selec=None,
                 effsel=None,
                 effsel_mask=None,
                 effsel_grid=None,
                 dmods=None,
                 nwalkers=None,
                 nit=None,
                 ncut=None,
                 usr_log_prior=None,
                 n_mass=None,
                 int_r_range=None,
                 iso=None,
                 iso_filename=None,
                 jkmins=None,
                 feh_range=None,
                 logg_range=None,
                 fit_dir=None,
                 gap_dir=None,
                 ksf_dir=None,
                 version='',
                 verbose=False,
                 ro=None,
                 vo=None,
                 zo=None
                 ):
        '''__init__:
        
        Initialize the _HaloFit parent class
        
        Args:
            densfunc (callable) - Density profile
            selec (str or arr) - Kinematic selection space
            effsel (array) - Effective selection function calculated on a grid 
                of size (nfield,ndmod) without kinematic selection effects
            effsel_mask (array) - Effective selection function grid mask of 
                shape (effsel)
            effsel_grid (list) - Length-3 list of grids of shape (effsel) 
                representing R,phi,z positions where the selection function is 
                evaluated
            dmods (array) - Array of distance modulus
            nwalkers (int) - Number of MCMC walkers
            nit (int) - Number of steps to run each walker
            ncut (int) - Number of steps to trim from beginning of each chain
            usr_log_prior (callable) - User supplied log prior for densfunc
            n_mass (int) - Number of masses to calculate
            int_r_range (array) - 2-element list of spherical integration range
            iso (array) - Isochrone grid
            iso_filename (str) - Filename to access the isochrone grid
            jkmins (array) - Array of minimum (J-K) values
            feh_range (array) - 2-element list of Fe/H min, Fe/H max
            logg_range (array) - 2-element list of logg min, logg max
            fit_dir (str) - Directory for holding fitting data and figures
            gap_dir (str) - Gaia-APOGEE processed data directory
            ksf_dir (str) - kSF directory
            version (str) - Version string to add to filenames
            verbose (bool) - Print info to screen
            ro,vo,zo (float) - Galpy scales, also solar cylindrical radius. zo 
                is Solar height above the plane
            
        Returns:
            None
        '''
        # Density profile
        self.densfunc = densfunc
        
        # Kinematic selection space
        if selec is not None:
            if isinstance(selec,str): selec=[selec,]
            selec_suffix = '-'.join(selec)
        else:
            selec_suffix = None
        # Confusing but better for consistency with other code
        self.selec = selec_suffix
        self.selec_arr = selec
        
        # MCMC info
        self.nwalkers = int(nwalkers)
        self.nit = int(nit)
        self.ncut = int(ncut)
        
        # Log prior
        if usr_log_prior is not None:
            self.usr_log_prior = usr_log_prior
        else:
            self.usr_log_prior = _null_prior
        
        # Mass calculation info
        if n_mass is None:
            n_mass = int(nwalkers*(nit-ncut))
        self.n_mass = n_mass
        if int_r_range is None:
            self.int_r_range = [2.,70.]
            print('int_r_range not specified, using default of [2.,70.]')
        else:
            self.int_r_range = int_r_range
        
        # Isochrone (since iso is large it can be dynamically loaded so that 
        # pickled HaloFit classes don't carry a redundant object).
        # If this is the case then load iso with self.get_iso()
        self.iso = iso # Probably None
        self.iso_filename = iso_filename
        
        # J-K minimums
        self.jkmins = jkmins
        
        # [Fe/H], logg range
        feh_min, feh_max = feh_range
        logg_min, logg_max = logg_range
        self.feh_range = feh_range
        self.logg_range = logg_range
        self.feh_min = feh_min
        self.feh_max = feh_max
        self.logg_min = logg_min
        self.logg_max = logg_max
        
        # I/O directories
        if fit_dir[-1] != '/': fit_dir+='/'
        self.fit_dir = fit_dir
        self.gap_dir = gap_dir
        self.ksf_dir = ksf_dir
        
        # Version
        if version is None:
            version = ''
        elif version != '':
            if version[-1] != '/': version+='/'
        self.version = version
        
        # Prepare the effective selection function & grid
        self.effsel = effsel
        self.effsel_mask = effsel_mask
        Rgrid,phigrid,zgrid = effsel_grid
        self.Rgrid = Rgrid
        self.phigrid = phigrid
        self.zgrid = zgrid
        self.dmods = dmods
        
        # Initialize variables that will be set once analysis is complete
        self.opt_init = None
        self.opt_post = None
        self.samples = None
        self.sampler = None
        self.masses = None
        self.mass_inds = None
        self.facs = None
        self.isofactors = None
        self.ml = None
        self.ml_ind = None
        self.aic = None
        self.bic = None
        self.loglike = None
        self.effvol_halo = None
        self.effvol_disk = None
        self._hasResults = False
        
        # Galpy scales and zo
        self.ro = ro
        self.vo = vo
        self.zo = zo
        
        # Verbosity
        self.verbose = verbose
        
        # Warn if variables not set
        self._check_not_set()
    
    # Getters
    
    def get_effsel_list(self):
        '''get_effsel_list:
        
        Return a list of effective selection function position grids:
        [Rgrid,phigrid,zgrid]
        '''
        return [self.Rgrid,self.phigrid,self.zgrid]
    
    def get_data_list(self):
        '''get_data_grid:
        
        Return a list of data positions:
        [Rdata,phidata,zdata]
        '''
        return [self.Rdata,self.phidata,self.zdata]
    
    def get_iso(self):
        '''get_iso:
        
        Get the isochrone grid
        '''
        if self.iso is not None:
            return self.iso
        elif self.iso_filename is not None:
            return np.load(self.iso_filename)
        else:
            print('warning: iso_filename not set, returning None')
            return None
    
    def get_ksel(self,spline_type='linear',mask=True):
        '''get_ksel:
        
        Return the kinematic selection function
        
        Args:
            spline_type (str) - Type of spline, 'linear' or 'cubic' [default 
                'linear']
            mask (bool) - Mask the kinematic selection function with the 
                effective selection function mask [default False]
        '''
        ksel_filename = self.ksf_dir+'kSF_grid_'+spline_type+'_'+\
            self.selec+'.dat'
        with open(ksel_filename,'rb') as f:
            if self.verbose:
                print('Loading APOGEE kin. eff. sel. grid from '+ksel_filename)
            ksel = pickle.load(f)
        if mask and not self.effsel_mask is None:
            ksel = ksel[self.effsel_mask]
        return ksel
    
    def get_results(self,load_sampler=False,load_blobbed_effvol=True):
        '''get_results:
        
        Get results from MCMC and mass calculation. Note that the pickled
        sampler is quite big so it's optionally loaded.
        
        Args:
            load_sampler (bool) - Load and set the emcee sampler [default False]
            load_blobbed_effvol (bool) - Load and set blobbed effective volume
                from the emcee sampler [default True]
        
        Sets:
            samples - MCMC samples
            sampler - MCMC sampler object (optional)
            masses - Calculated masses
            mass_inds - indices of samples where masses were calculated
            facs - Mass calculation factors
            isofactors - Isochrone factors
            opt_init - First optimization
            opt_post - Optimization from MCMC maximum likelihood
        '''
        samples_filename = self.fit_data_dir+'samples.npy'
        masses_filename = self.fit_data_dir+'masses.npy'
        mass_inds_filename = self.fit_data_dir+'mass_inds.npy'
        facs_filename = self.fit_data_dir+'facs.npy'
        isofactors_filename = self.fit_data_dir+'isofactors.npy'
        opt_filename = self.fit_data_dir+'opt.pkl'
        opt_init_filename = self.fit_data_dir+'opt_init.pkl'
        opt_post_filename = self.fit_data_dir+'opt_post.pkl'
        
        if os.path.exists(samples_filename):
            self.samples = np.load(samples_filename)
        else:
            print('warning: samples file not found')
                
        if os.path.exists(masses_filename):
            self.masses = np.load(masses_filename)
        else:
            print('warning: masses file not found')
            
        if os.path.exists(mass_inds_filename):
            self.mass_inds = np.load(mass_inds_filename)
        else:
            print('warning: mass_inds file not found')
            
        if os.path.exists(facs_filename):
            self.facs = np.load(facs_filename)
        else:
            print('warning: facs file not found')
        
        if os.path.exists(isofactors_filename):
            self.isofactors = np.load(isofactors_filename)
        else:
            print('warning: isofactors file not found')
            
        if os.path.exists(opt_init_filename):
            with open(opt_init_filename,'rb') as f:
                self.opt_init = pickle.load(f)
        elif os.path.exists(opt_filename):
            with open(opt_filename,'rb') as f:
                self.opt_init = pickle.load(f)
        else:
            print('warning: opt_init file not found')
            
        if os.path.exists(opt_post_filename):
            with open(opt_post_filename,'rb') as f:
                self.opt_post = pickle.load(f)
        else:
            print('warning: opt_post file not found')
                
        # Also load the sampler if requested.
        if load_sampler:
            self.get_sampler(set_sampler=True,return_sampler=False)
        
        # Also load the blobbed effective volume if requested.
        if load_blobbed_effvol:
            if self.sampler is None:
                _sampler = self.get_sampler(set_sampler=False,
                    return_sampler=True)
                blobs = _sampler.get_blobs(flat=True,discard=self.ncut)
            else:
                blobs = self.sampler.get_blobs(flat=True,discard=self.ncut)
            if blobs is None:
                print('warning: blobs requested but not found in emcee sampler')
            else:
                self.effvol_halo = blobs['effvol_halo']
                self.effvol_disk = blobs['effvol_disk']
            
        self._hasResults = True
    
    def get_sampler(self,set_sampler=False,return_sampler=True):
        '''get_sampler:
        
        Load the pickled sampler. Warning, this can be a very large object
        and can cause memory issues.

        Args:
            set_sampler (bool) - Set the sampler attribute [default False]
            return_sampler (bool) - Return the sampler [default True]
        
        Sets
            sampler - MCMC sampler object
        '''
        sampler_filename = self.fit_data_dir+'sampler.pkl'
        
        if os.path.exists(sampler_filename):
            with open(sampler_filename,'rb') as f:
                _sampler = pickle.load(f)
                if np.prod(_sampler.chain.shape) > 1e7:
                    print('warning: sampler is large, beware of memory usage')
                if set_sampler:
                    self.sampler = _sampler
                if return_sampler:
                    return _sampler
                
    
    def get_ml_params(self,ml_type='mcmc_ml'):
        '''get_ml_params:
        
        Get the maximum likelihood set of parameters for the density profile
        
        ml_type can be:
         'mcmc_ml' - maximum likelihood parameters from the MCMC chain
         'mcmc_median' - median parameters from the MCMC chain
         'post' - Optimization done from highest likelihood MCMC sample
         'init' - Optimization done to start MCMC chain
        
        Args:
            ml_type (string) - String denoting type of ML solution
        
        '''
        # Make sure ml_type correct
        assert ml_type in ['mcmc_ml','mcmc_median','post','init'],\
        'ml_type must be one of "mcmc_ml", "mcmc_median", "post", "init"'
        
        # Make sure relevant variables are set
        if ml_type == 'mcmc_ml':
            assert self.ml_ind is not None and self.samples is not None,\
                'ml_ind and/or samples is not set, run get_results() & '+\
                'get_loglike_ml_aic_bic()'
            return self.samples[self.ml_ind]
        
        if ml_type == 'mcmc_median':
            assert self.samples is not None, 'samples is not set, run '+\
                'get_results()'
            return np.median(self.samples,axis=0)
        
        if ml_type == 'post':
            assert self.opt_post is not None, 'opt_post is not set, run '+\
                'get_results()'
            if not self.opt_post.success:
                print('warning: post optimization was not successful')
            return self.opt_post.x
        
        if ml_type == 'init':
            assert self.opt_init is not None, 'opt_init is not set, run '+\
                'get_results()'
            if not self.opt_init.success:
                print('warning: initial optimization was not successful')
            return self.opt_init.x
        
    def get_loglike_ml_aic_bic(self):
        '''get_loglike_ml_aic_bic:
        
        Get log likelihood, maximum likelihood, maximum likelihood inds, AIC, 
        BIC values.
                
        Sets:
            ml
            ml_ind
            aic
            bic
            loglike
        '''
        ml_aic_bic_filename = self.fit_data_dir+'mll_aic_bic.npy'
        loglike_filename = self.fit_data_dir+'loglike.npy'
        
        if os.path.exists(ml_aic_bic_filename):
            ml,ml_ind,aic,bic = np.load(ml_aic_bic_filename)
            ml_ind = int(ml_ind)
            self.ml = ml
            self.ml_ind = ml_ind
            self.aic = aic
            self.bic = bic
            if self.verbose:
                print('Set self.ml, self.ml_ind, self.aic, self.bic')
        else:
            print('warning: File containing ML, AIC, BIC does not exist')
        
        if os.path.exists(loglike_filename):
            self.loglike = np.load(loglike_filename)
            if self.verbose:
                print('Set self.loglike')
        else:
            print('warning: File containing log likelihoods does not exist')
    
    # Setters
    
    def set_selec(self,selec):
        '''set_selec
        
        Set a new select for the class
        '''
        # Probably just easier to re-initialize the whole class
        pass
    
    # Calculators
    
    def calculate_loglike(self,params,usr_log_prior=None):
        '''calculate_loglike:
        
        Compute the log likelihood x prior for a set of parameters
        '''
        if usr_log_prior is None:
            usr_log_prior = self.usr_log_prior
        return loglike(params, self.densfunc, self.get_fit_effsel(), 
                             self.Rgrid, self.phigrid, self.zgrid, 
                             self.Rdata, self.phidata, self.zdata, 
                             usr_log_prior)
    
    def calculate_isofactors(self,fast=True):
        '''calculate_isofactors:
        
        Calculate the factor omega from Mackereth+ 2019 equation 10
        '''
        if self.iso is None:
            assert self.iso_filename is not None, 'if iso not loaded, must'+\
                'have isochrone filename available'
            iso = self.get_iso()
        else:
            iso = self.iso         
            
        nfield = self.get_fit_effsel().shape[0]
        isofactors = np.zeros(nfield)
        
        # Assumes that the only variable is the changing minimum (J-K)
        if fast:
            unique_jkmin = np.unique(self.jkmins)
            if self.verbose:
                print('Doing fast isochrone factor calculation')
                print('Unique (J-K) minimum values: '+str(unique_jkmin))
            for i in range(len(unique_jkmin)):
                # The mass ratio mask is for all stars considered 
                massratio_isomask = (Z2FEH(iso['Zini']) > self.feh_min) &\
                                    (Z2FEH(iso['Zini']) < self.feh_max) &\
                                    (iso['logAge'] >= 10) &\
                                    (iso['logL'] > -9) # Eliminates WDs
                # The average mass mask extracts fitted sample based on color and logg
                avmass_isomask = massratio_isomask &\
                                 (iso['Jmag']-iso['Ksmag'] > unique_jkmin[i]) &\
                                 (iso['logg'] > self.logg_min) &\
                                 (iso['logg'] < self.logg_max)
                massratio = piso.mass_ratio(iso[massratio_isomask], 
                                            logg_range=self.logg_range,
                                            jk_range=[unique_jkmin[i],999.])
                avmass = piso.average_mass(iso[avmass_isomask])
                jkmin_mask = self.jkmins == unique_jkmin[i]
                isofactors[jkmin_mask] = avmass/massratio
            
        else:
            for i in range(nfield):
                # The mass ratio mask is for all stars considered 
                massratio_isomask = (Z2FEH(iso['Zini']) > self.feh_min) &\
                                    (Z2FEH(iso['Zini']) < self.feh_max) &\
                                    (iso['logAge'] >= 10) &\
                                    (iso['logL'] > -9) # Eliminates WDs
                # The average mass mask extracts fitted sample based on color and logg
                avmass_isomask = massratio_isomask &\
                                 (iso['Jmag']-iso['Ksmag'] > self.jkmins[i]) &\
                                 (iso['logg'] > self.logg_min) &\
                                 (iso['logg'] < self.logg_max)
                massratio = piso.mass_ratio(iso[massratio_isomask], 
                                            logg_range=self.logg_range,
                                            jk_range=[self.jkmins[i],999.])
                avmass = piso.average_mass(iso[avmass_isomask])
                isofactors[i] = avmass/massratio
                if self.verbose:
                    print('Calculating isofactor '+str(i+1)+'/'+str(nfield),
                          end='\r')
            if self.verbose:
                print('Calculating isofactor '+str(nfield)+'/'+str(nfield))
        
        if self.verbose:
            print('Unique isochrone factors: '+str(np.unique(isofactors)))
            
        return isofactors
    

    def calculate_masses(self,samples=None,n_mass=None,nprocs=1,
                         mass_int_type='spherical_grid',batch_masses=True,
                         save_results=False,overwrite=False,set_results=False,
                         return_results=True,verbose=None):
        '''calculate_masses:

        Calculate the masses of the density profiles described by the samples
        of the HaloFit class as well as the supplied effective selection
        function and stellar information.

        Args:
            samples (array) - Samples of density profile parameters of shape
                (n_samples, n_params). If none then will use self.samples
            n_mass (int) - Number of masses to calculate. Must be 
                less than or equal to the number of samples. If none then will
                use self.n_mass. Note that this may means that not all samples
                are used.
            nprocs (int) - Number of processes to use for multiprocessing
                [default: 1]
            mass_int_type (str) - Type of mass integration to use. [default:
                'spherical_grid']
            batch_masses (bool) - Whether to batch the mass calculations
                [default: True]
            save_results (bool) - Whether to save the results to file
                [default: True]
            overwrite (bool) - Whether to overwrite existing files [default:
                False]
            set_results (bool) - Whether to set the results as attributes of
                the class [default: True]
            return_results (bool) - Whether to return the results [default:
                True]

        Returns (optional):
            masses (array) - Masses of the density profiles
            facs (array) - Factors used to convert number -> mass
            mass_inds (array) - Indices of the samples used to calculate the
                masses
            isofactors (array) - Isochrone factors used to convert number of 
                red giant stars to mass of the whole stellar population
        
        Sets: (optional):
            self.masses -> masses
            self.facs -> facs
            self.mass_inds -> mass_inds
            self.isofactors -> isofactors
        '''
        # Check overwrite and whether files exist before even starting
        _files_to_check = [self.fit_data_dir+f for f in \
            ['masses.npy','facs.npy','mass_inds.npy','isofactors.npy']]
        _files_exist = np.any([os.path.exists(f) for f in _files_to_check])
        if save_results and not overwrite and _files_exist:
            raise ValueError('Files already exist and overwrite not set to True')
        
        # Handle samples
        if samples is None:
            if self.verbose:
                print('samples not supplied, using self.samples.')
            samples = self.samples

        # Handle number of masses
        if n_mass is None:
            if self.verbose:
                print('n_masses not supplied, using self.n_masses.')
            n_mass = self.n_mass

        # Calculate mass
        if self.verbose:
            print('Calculating masses')
        out_mass = mass_from_density_samples(samples=np.atleast_2d(samples), 
            densfunc=self.densfunc, n_star=self.n_star, 
            effsel=self.get_fit_effsel(), effsel_grid=self.get_effsel_list(), 
            iso=self.get_iso(), feh_range=self.feh_range, 
            logg_range=self.logg_range, jkmins=self.jkmins, n_mass=n_mass, 
            mass_int_type=mass_int_type, int_r_range=self.int_r_range, 
            nprocs=nprocs, batch=batch_masses, ro=self.ro, zo=self.zo)
        masses, facs, mass_inds, isofactors = out_mass

        if save_results:
            if self.verbose:
                print('Saving results to: '+self.fit_data_dir)
            np.save(self.fit_data_dir+'masses.npy',masses)
            np.save(self.fit_data_dir+'facs.npy',facs)
            np.save(self.fit_data_dir+'mass_inds.npy',mass_inds)
            np.save(self.fit_data_dir+'isofactors.npy',isofactors)
        
        if set_results:
            self.masses = masses
            self.facs = facs
            self.mass_inds = mass_inds
            self.isofactors = isofactors

        if return_results:
            return masses, facs, mass_inds, isofactors

    def run_optimization(self,init=None,method='Powell',optimizer_kwargs={}):
        '''run_optimization:
        
        Optimize the likelihood function using scipy.optimize.minimize
        
        Args:
            init (array) - List of parameters to begin optimization
            method (str) - Type of optimizer to use, see scipy.optimize 
                [default 'Powell']
            optimizer_kwargs (dict) - kwargs to pass to scipy.optimize.minimize
        
        Returns:
            opt (scipy OptimizeResult) - Output of scipy.optimize.minimize
        '''
        if init is None:
            init = self.init
        effsel = self.get_fit_effsel()
        if self.verbose:
            print('Doing maximum likelihood')
        opt_fn = lambda x: mloglike(x, self.densfunc, effsel, self.Rgrid, 
            self.phigrid, self.zgrid, self.Rdata, self.phidata, self.zdata,
            usr_log_prior=self.usr_log_prior)
        opt = scipy.optimize.minimize(opt_fn, init, method=method, 
                                      **optimizer_kwargs)
        if self.verbose:
            if opt.success:
                    print('MLE successful')
            else:
                print('MLE unsucessful')
                print('message: '+opt.message)
        return opt
    
    def calculate_ml_aic_bic(self,ml_type='post', n_param=None, n_star=None):
        '''calculate_ml_aic_bic:
        
        Calculate maximum likelihood, AIC, BIC.
        
        Get the maximum likelihood either from the maximum of the likelihoods 
        in the MCMC sampler chain ('mcmc_ml'), from the initial optimization
        ('init') or from the final optimization (after the MCMC is run) ('post')
        
        Args:
            ml_type (str) - Maximum likelihood type [default 'post']
        
        Returns:
            mll (float) - Maximum log likelihood
            aic (float) - Akaike information criterion
            bic (float) - Bayesian information criterion
        '''
        # Make sure ml_type correct
        assert ml_type in ['mcmc_ml','post','init'],\
        'ml_type must be one of "mcmc_ml", "post", "init"'
        
        # Make sure get_results() has been run
        assert self.loglike is not None and\
               self.opt_post is not None and\
               self.opt_init is not None, 'run self.get_results()'
        
        # Determine n_param and n_star
        if n_param is None:
            n_param = self.samples.shape[1]
        if n_star is None:
            n_star = self.n_star
        
        # Make sure relevant variables are set
        if ml_type == 'mcmc_ml':
            mll = np.max(self.loglike)
        elif ml_type == 'post':
            if not self.opt_post.success:
                print('warning: post optimization was not successful')
            mll = -self.opt_post.fun # Opt minimizes -loglike
        elif ml_type == 'init':
            if not self.opt_init.success:
                print('warning: initial optimization was not successful')
            mll = -self.opt_init.fun # Opt minimizes -loglike
        
        aic = 2*n_param - 2*mll
        bic = np.log(n_star)*n_param - 2*mll
        
        return mll,aic,bic
    
    def get_rotated_coords_in_gc_frame(self,params=None,ml_type='mcmc_median',
        vec=np.array([0,0,1])):
        '''get_rotated_coords_in_gc_frame:

        Take a vector in the rotated frame and transform it back to the 
        GC frame. This is useful for determining e.g. the principal axis 
        of a triaxial density ellipsoid

        Args:
            params (array) - List of parameters to calculate vector for,
                if None then fall back on ml_type
            ml_type (str) - Type of 'best-fit' parameter to use if the 
                params argument is None. [default 'mcmc_median']
            vec (array) - Vector to transform back to GC frame [default [0,0,1]]

        Returns:
            gc_vec (array) - Vector in GC frame
        '''
        if params is None:
            print('No params supplied, using ml_type: '+ml_type)
            params = self.get_ml_params(ml_type=ml_type)
        params = pdens.denormalize_parameters(params,self.densfunc)

        # Get the parameters to make zvec from the densfunc and params
        indx = pdens.get_densfunc_params_indx(self.densfunc,
            ['theta','eta','phi'])
        if len(params.shape) > 1:
            theta,eta,phi = params.flatten()[indx]
        else:
            theta,eta,phi = params[indx]
        zvec = np.array([np.sqrt(1-eta**2)*np.cos(theta), 
                         np.sqrt(1-eta**2)*np.sin(theta), 
                         eta])

        # Transform the coordinates out of the rotated frame
        x,y,z = pdens.transform_zvecpa(np.dstack(vec)[0],zvec,phi,inv=True)
        gc_vec = np.array([x,y,z]).T

        return gc_vec
    
    def print_mcmc_diagnostic_txt(self,return_txt=False):
        '''print_mcmc_diagnostic_txt:
        
        Access the text file with the MCMC diagnostics and print it to the
        screen

        Args:
            return_txt (bool) - If True, return the text as a string
                [default False]
        
        Returns:
            txt (str) - Text of the MCMC diagnostics
        '''
        _files = os.listdir(self.fit_data_dir)
        if 'mcmc_diagnostics.txt' in _files:
            with open(self.fit_data_dir+'mcmc_diagnostics.txt','r') as f:
                txt = f.read()
                print(txt)
            if return_txt:
                return txt
        else:
            print('mcmc_diagnostics.txt not found in dir: ',self.fit_data_dir)

    # Utils
    
    def _check_not_set(self):
        '''_check_not_set:
        
        Check to see if class attributes were left as None at instantiation
        '''
        if not self.verbose: return None
        attrs_exclude = ['opt_init','opt_post','samples','sampler','masses',
                         'mass_inds','facs','isofactors','ml','ml_ind','aic',
                         'bic','loglike','effvol_halo','effvol_disk']
        attrs = [a for a in dir(self) if not a.startswith('_') \
                 and not a in attrs_exclude and not callable(getattr(self, a))]
        for a in attrs:
            res = getattr(self,a)
            if isinstance(res,np.ndarray):
                continue
            if (a == 'selec' or a == 'selec_arr')\
                and (not hasattr(self,'fit_type') or\
                     getattr(self,'fit_type') == 'all'):
                continue
            if a == 'iso' and getattr(self,'iso_filename') not in [None,'']:
                continue
            if a == 'iso_filename' and type(getattr(self,'iso')) in \
                    [list,np.ndarray]:
                continue
            if res in [None,'']:    
                print('warning: attribute '+str(a)+' not set')
        return None

    def _validate_version_params_base(self):
        '''_validate_version_params_base:

        Hardcode some version checks into the code to make sure that when the 
        .version keyword is used that other parameters are set correctly. This 
        is the base version of the function that is called by the 
        _validate_version_params() function in the child classes.

        Returns:
            None
        
        Raises:
            Warning if version does not correspond to parameters
        '''
        if self.version == 'tau50':
            if self.nit*self.nwalkers < 1e6:
                print('''warning: version is tau50 and total number of samples 
                         < 1e6''')
        
        

class HaloFit(_HaloFit):
    '''HaloFit:
    
    Convenience class to wrap all of the information needed by various 
    fitting and plotting routines.
    '''
    
    # Initialize
    
    def __init__(self,
                 # HaloFit parameters
                 allstar=None,
                 orbs=None,
                 init=None,
                 init_type=None,
                 fit_type=None,
                 mask_disk=True,
                 mask_halo=True,
                 # _HaloFit parameters
                 densfunc=None,
                 selec=None,
                 effsel=None,
                 effsel_mask=None,
                 effsel_grid=None,
                 dmods=None,
                 nwalkers=None,
                 nit=None,
                 ncut=None,
                 usr_log_prior=None,
                 n_mass=None,
                 int_r_range=None,
                 iso=None,
                 iso_filename=None,
                 jkmins=None,
                 feh_range=None,
                 logg_range=None,
                 fit_dir=None,
                 gap_dir=None,
                 ksf_dir=None,
                 version='',
                 verbose=False,
                 ro=None,
                 vo=None,
                 zo=None
                ):
        '''__init__:
        
        Initialize a HaloFit class
        
        Args:
            allstar (array) - APOGEE allstar without any observational masking
                applied. Should just come from cleaning notebook
            orbs (orbit.Orbit) - orbit.Orbit corresponding to allstar data
            init (array) - Supply initialization parameters, if None then 
            init_type (bool) - Type of initialization to 
            fit_type (str) - string specifying the type of fit. Should be one 
                of ['gse','all']
            mask_disk (bool) - Mask the disk if fit_type='gse'? [default True]
            mask_halo (bool) - Use a halo mask from IDs if fit_type='all'?
                [default True]
            *** See _HaloFit.__init__ for other parameters
        
        Returns:
            None
        '''
        # Call parent constructor
        _HaloFit.__init__(self,
                          densfunc=densfunc,
                          selec=selec,
                          effsel=effsel,
                          effsel_mask=effsel_mask,
                          effsel_grid=effsel_grid,
                          dmods=dmods,
                          nwalkers=nwalkers,
                          nit=nit,
                          ncut=ncut,
                          usr_log_prior=usr_log_prior,
                          n_mass=n_mass,
                          int_r_range=int_r_range,
                          iso=iso,
                          iso_filename=iso_filename,
                          jkmins=jkmins,
                          feh_range=feh_range,
                          logg_range=logg_range,
                          fit_dir=fit_dir,
                          gap_dir=gap_dir,
                          ksf_dir=ksf_dir,
                          version=version,
                          verbose=verbose,
                          ro=ro,
                          vo=vo,
                          zo=zo)
        
        # Unmasked data
        if allstar is not None and orbs is not None:
            assert len(allstar)==len(orbs), 'allstar,orbs length must be equal'
        # Don't save because this is large when pickling
        # self.allstar_nomask = allstar
        # self.orbs_nomask = orbs
        
        # Fit type info
        if fit_type is not None:
            assert fit_type in ['gse','gse_map','all','all_map']
        else:
            print('warning: fit_type is required to access data')
            fit_type = ''
        self.fit_type = fit_type
        
        # Output directories
        if 'all' in fit_type:
            selec_str = ''
        else:
            selec_str = self.selec+'/'
        fit_data_dir = fit_dir+'data/'+fit_type+'/'+selec_str+str(self.feh_min)+\
                       '_feh_'+str(self.feh_max)+'/'+densfunc.__name__+'/'+self.version
        fit_fig_dir = fit_dir+'fig/'+fit_type+'/'+selec_str+str(self.feh_min)+\
                      '_feh_'+str(self.feh_max)+'/'+densfunc.__name__+'/'+self.version
        if not os.path.exists(fit_data_dir):
            os.makedirs(fit_data_dir,exist_ok=True)
        if not os.path.exists(fit_fig_dir):
            os.makedirs(fit_fig_dir,exist_ok=True)
        self.fit_data_dir = fit_data_dir
        self.fit_fig_dir = fit_fig_dir
        
        # Get the kinematic effective selection function
        if 'gse' in fit_type:
            ksel = self.get_ksel(spline_type='linear',mask=True)
            keffsel = effsel*ksel
            assert np.all( ~np.all(keffsel < 1e-9, axis=1) ),\
                'Null fields still in keffSF'
            self.keffsel = keffsel
        else:
            self.keffsel = None
        
        # Mask out GS/E stars
        if 'gse' in fit_type:
            self.mask_disk = mask_disk
            self.mask_halo = None # Not used
            self.halo_mask = None # Not used
            gse_mask_filename = gap_dir+'hb_apogee_ids_'+self.selec
            if mask_disk:
                gse_mask_filename += '_dmask.npy'
            else:
                gse_mask_filename += '.npy'
            gse_apogee_IDs = np.load(gse_mask_filename)
            gse_mask = putil.make_mask_from_apogee_ids(allstar,
                gse_apogee_IDs)
            orbs_gse = orbs[gse_mask]
            allstar_gse = allstar[gse_mask]
            self.gse_mask = gse_mask

            # Obervational masking
            obs_mask = (allstar_gse['LOGG'] > self.logg_min) &\
                       (allstar_gse['LOGG'] < self.logg_max)
                       #(allstar_gse['FE_H'] > self.feh_min)
                       #(allstar_gse['FE_H'] < self.feh_max)
            self.obs_mask = obs_mask
            orbs_obs = orbs_gse[obs_mask]
            allstar_obs = allstar_gse[obs_mask]   
       
        elif 'all' in fit_type:
            self.gse_mask = None # Not used
            self.mask_disk = None # Not used
            self.mask_halo = mask_halo
            if mask_halo:
                halo_mask_filename = gap_dir+'halo_apogee_ids.npy'
                halo_apogee_IDs = np.load(halo_mask_filename)
                halo_mask = putil.make_mask_from_apogee_ids(allstar,
                    halo_apogee_IDs)
                self.halo_mask = halo_mask
            else:
                self.halo_mask = None
            
            # A mask for observational properties, should be redundant
            obs_mask = (allstar['LOGG'] > self.logg_min) &\
                       (allstar['LOGG'] < self.logg_max)
                       #(allstar_gse['FE_H'] > self.feh_min)
                       #(allstar_gse['FE_H'] < self.feh_max)
            
            if mask_halo:
                obs_mask = obs_mask & halo_mask
            else: # Use a default metallicity selection
                pass
            
            self.obs_mask = obs_mask
            orbs_obs = orbs[obs_mask]
            allstar_obs = allstar[obs_mask] 
        
        # Set data properties
        mrpz = np.array([orbs_obs.R(use_physical=True).value,
                         orbs_obs.phi(use_physical=True).value,
                         orbs_obs.z(use_physical=True).value]).T
        Rdata,phidata,zdata = mrpz.T
        self.orbs = orbs_obs
        self.allstar = allstar_obs
        self.Rdata = Rdata
        self.phidata = phidata
        self.zdata = zdata
        self.n_star = len(orbs_obs)
        
        # Initialization
        self.init_type = init_type
        if init is None:
            if init_type is None:
                if verbose:
                    print('Using default init')
                init = pdens.get_densfunc_mcmc_init_uninformed(densfunc)
            else:
                if verbose:
                    print('Using informed init')
                init = self.get_densfunc_mcmc_init_informed(init_type=init_type,
                    verbose=verbose)
        self.init = init
        
    # Getters
    
    def get_fit_effsel(self):
        '''get_fit_effsel:
        
        Get the proper effective selection function for the fit
        '''
        if self.fit_type in ['gse','gse_map']:
            return self.keffsel
        elif self.fit_type in ['all','all_map']:
            return self.effsel
    
    def get_densfunc_mcmc_init_informed(self, init_type='ML', verbose=False):
        '''get_densfunc_mcmc_init_informed:
        
        Get an informed set of parameters to use as init. Normally load the 
        maximum likelihood set of parameters of the source densprofile. 
        init_type can be:
        'ML' - Use the maximum likelihood samples from the source densfunc
        'uninformed' - Just use default init 
        
        Args:
            init_type (string) - Type of init to load. 'ML' for maximum 
                likelihood sample, 'uninformed' for default init
            verbose (bool) - Be verbose? [default False]
            
        Returns:
            init (array) - Init parameters to use
        '''
        if self.verbose is not None:
            verbose = self.verbose
        
        assert init_type in ['ML','uninformed']
    
        densfunc = self.densfunc
        if densfunc.__name__ == 'triaxial_single_angle_zvecpa':
            print('warning: setting init_type to uninformed because densfunc is'
                  ' triaxial_single_angle_zvecpa')
            init_type = 'uninformed'

        # Unpack
        feh_min,feh_max = self.feh_range
        
        # Get the densfunc that will provide the init
        densfunc_source = pdens.get_densfunc_mcmc_init_source(densfunc)
        source_fit_data_dir = None

        # Check ML files
        if init_type=='ML':
            # Sample & ML filename
            if 'all' in self.fit_type:
                selec_str = ''
            else:
                selec_str = self.selec+'/'
            source_fit_data_dir = self.fit_dir+'data/'+self.fit_type+'/'+\
                           selec_str+str(self.feh_min)+'_feh_'+\
                           str(self.feh_max)+'/'+densfunc_source.__name__+'/'+\
                           self.version

            samples_filename = source_fit_data_dir+'samples.npy'
            ml_filename = source_fit_data_dir+'mll_aic_bic.npy'
            if (not os.path.exists(samples_filename)) or\
               (not os.path.exists(ml_filename)):
                print('warning: files required for init_type "ML" not present'
                      ', changing init_type to "uninformed"')
                init_type = 'uninformed'

        if init_type == 'uninformed':
            init = pdens.get_densfunc_mcmc_init_uninformed(densfunc)
        if init_type == 'ML':
            samples = np.load(samples_filename)
            _,ml_ind,_,_ = np.load(ml_filename)
            sample_ml = samples[int(ml_ind)]
            init = pdens.make_densfunc_mcmc_init_from_source_params( densfunc, 
                params_source=sample_ml, densfunc_source=densfunc_source)

        if verbose:
            print('init_type: '+str(init_type))
            if densfunc_source is None:
                print('densfunc_source: None')
            else:
                print('densfunc_source: '+densfunc_source.__name__)
            if source_fit_data_dir is not None:
                print('source_fit_data_dir: '+source_fit_data_dir)

        return init
    
    # Setters
    
    def set_densfunc(self,densfunc,init=None,init_type=None,usr_log_prior=None):
        '''set_densfunc:
        
        Set a new densfunc for the class rather than re-initializing the class.
        
        Args:
            densfunc (callable) - Density profile
        
        Returns:
            None
        '''
        if self.verbose:
            print('Setting densfunc to: '+densfunc.__name__)
            print('Version is: '+self.version)
        # Set the densfunc
        self.densfunc=densfunc
        
        # Re-set the directories
        if 'all' in self.fit_type:
            selec_str = ''
        else:
            selec_str = self.selec+'/'
        fit_data_dir = self.fit_dir+'data/'+self.fit_type+'/'+selec_str+\
            str(self.feh_min)+'_feh_'+str(self.feh_max)+'/'+densfunc.__name__+\
            '/'+self.version
        fit_fig_dir = self.fit_dir+'fig/'+self.fit_type+'/'+selec_str+\
            str(self.feh_min)+'_feh_'+str(self.feh_max)+'/'+densfunc.__name__+\
            '/'+self.version
        if not os.path.exists(fit_data_dir):
            os.makedirs(fit_data_dir,exist_ok=True)
        if not os.path.exists(fit_fig_dir):
            os.makedirs(fit_fig_dir,exist_ok=True)
        self.fit_data_dir = fit_data_dir
        self.fit_fig_dir = fit_fig_dir
        
        # Re-set the init
        if init_type is None:
            init_type=self.init_type
        elif init_type in ['ML','uninformed']:
            self.init_type=init_type
            
        if init is None:
            if init_type is None:
                if self.verbose:
                    print('Using default init')
                init = pdens.get_densfunc_mcmc_init_uninformed(densfunc)
            else:
                if self.verbose:
                    print('Using init_type: '+str(init_type))
                init = self.get_densfunc_mcmc_init_informed(
                    init_type=init_type, verbose=self.verbose)
        self.init = init
        
        # Re-set the user-supplied log-prior
        if usr_log_prior is not None:
            self.usr_log_prior = usr_log_prior
        else:
            self.usr_log_prior = _null_prior    

    # Utils

    def _validate_version_params(self):
        '''Hardcode some version checks into the code to make sure that when the 
        .version keyword is used that other parameters are set correctly. This 
        is the base version of the function that is called by the 
        _validate_version_params() function in the child classes.

        Returns:
            None
        
        Raises:
            Warning if version does not correspond to parameters
        '''
        self._validate_version_params_base()
        if 'ksf' in self.version and 'v2.01' in self.version:
            if 'v2.01' not in self.ksf_dir:
                print('''warning: suggested ksf version is v2.01, but
                      v2.01 not in ksf_dir''')
        if 'ksf' in self.version and 'v2.02' not in self.version:
            if 'v2.02' not in self.ksf_dir:
                print('''warning: suggested ksf version is v2.02, but
                      v2.02 not in ksf_dir''')
        if 'ksf' in self.version and 'v2.03' not in self.version:
            if 'v2.03' not in self.ksf_dir:
                print('''warning: suggested ksf version is v2.03, but
                      v2.03 not in ksf_dir''')

class MockHaloFit(_HaloFit):
    '''MockHaloFit:
    '''
    def __init__(self,
                 # MockHaloFit parameters
                 allstar=None,
                 orbs=None,
                 init=None,
                 init_type=None,
                 fit_type=None,
                 iso_feh=None,
                 iso_age=None,
                 truths=None,
                 truth_mass=None,
                 truths_normed=False,
                 # _HaloFit parameters
                 densfunc=None,
                 selec=None,
                 nwalkers=None,
                 nit=None,
                 ncut=None,
                 usr_log_prior=None,
                 n_mass=None,
                 int_r_range=None,
                 iso=None,
                 iso_filename=None,
                 jkmins=None,
                 feh_range=None,
                 logg_range=None,
                 fit_dir=None,
                 gap_dir=None,
                 ksf_dir=None,
                 version='',
                 effsel=None,
                 effsel_mask=None,
                 effsel_grid=None,
                 dmods=None,
                 verbose=False,
                 ro=None,
                 vo=None,
                 zo=None
                 ):
        '''__init__:
        
        Initialize a MockHaloFit class
        
        Args:
            * See _HaloFit.__init__ for others
            
        Returns:
            None
        '''
        # First handle feh_range
        if feh_range is None and iso_feh:
            feh_range = [iso_feh-0.1,iso_feh+0.1]

        # Call parent constructor
        _HaloFit.__init__(self,
                          densfunc=densfunc,
                          selec=selec,
                          effsel=effsel,
                          effsel_mask=effsel_mask,
                          effsel_grid=effsel_grid,
                          dmods=dmods,
                          nwalkers=nwalkers,
                          nit=nit,
                          ncut=ncut,
                          usr_log_prior=usr_log_prior,
                          n_mass=n_mass,
                          int_r_range=int_r_range,
                          iso=iso,
                          iso_filename=iso_filename,
                          jkmins=jkmins,
                          feh_range=feh_range,
                          logg_range=logg_range,
                          fit_dir=fit_dir,
                          gap_dir=gap_dir,
                          ksf_dir=ksf_dir,
                          version=version,
                          verbose=verbose,
                          ro=ro,
                          vo=vo,
                          zo=zo)
        
        # Unmasked data
        if allstar is not None and orbs is not None:
            assert len(allstar)==len(orbs), 'allstar,orbs length must be equal'
        # Don't save because this is large when pickling
        # self.allstar_nomask = allstar
        # self.orbs_nomask = orbs
        
        # Fit type info
        if fit_type is not None:
            assert fit_type in ['mock','mock+disk','mock+ksf']
        else:
            warnings.warn('fit_type is required to access data')
            fit_type = ''
        self.fit_type = fit_type
        if 'ksf' in self.fit_type:
            assert self.selec is not None, 'selec is required for mock+ksf fits'
        
        # Mock truths
        self.truths = truths
        self.truth_mass = truth_mass
        self.truths_normed = truths_normed

        # Single isochrone properties for mock
        self.iso_feh = iso_feh
        if not (iso_feh < self.feh_max and iso_feh > self.feh_min):
            raise ValueError('iso_feh should be within feh_range')
        self.iso_age = iso_age

        # Output directories
        if 'ksf' not in fit_type:
            selec_str = ''
        else:
            selec_str = self.selec+'/'
        iso_feh_str = str(round(iso_feh,3))
        fit_data_dir  = fit_dir+'data/'+fit_type+'/'+selec_str+'feh_'+\
            iso_feh_str+'/'+densfunc.__name__+'/'+self.version
        fit_fig_dir  = fit_dir+'fig/'+fit_type+'/'+selec_str+'feh_'+\
            iso_feh_str+'/'+densfunc.__name__+'/'+self.version
        if not os.path.exists(fit_data_dir):
            os.makedirs(fit_data_dir,exist_ok=True)
        if not os.path.exists(fit_fig_dir):
            os.makedirs(fit_fig_dir,exist_ok=True)
        self.fit_data_dir = fit_data_dir
        self.fit_fig_dir = fit_fig_dir

        # Get the kinematic effective selection function
        if 'ksf' in fit_type:
            ksel = self.get_ksel(spline_type='linear',mask=True)
            keffsel = effsel*ksel
            assert np.all( ~np.all(keffsel < 1e-9, axis=1) ),\
                'Null fields still in keffSF'
            self.keffsel = keffsel
        else:
            self.keffsel = None
        
        # Do kinematic masking? Or expect it to be done already?

        # Observational masking (should be redundant)
        obs_mask = (allstar['LOGG'] > self.logg_min) &\
                   (allstar['LOGG'] < self.logg_max)
        self.obs_mask = obs_mask
        orbs_obs = orbs[obs_mask]
        allstar_obs = allstar[obs_mask]
        
        # Set data properties
        mrpz = np.array([orbs_obs.R(use_physical=True).value,
                         orbs_obs.phi(use_physical=True).value,
                         orbs_obs.z(use_physical=True).value]).T
        Rdata,phidata,zdata = mrpz.T
        self.orbs = orbs_obs
        self.allstar = allstar_obs
        self.Rdata = Rdata
        self.phidata = phidata
        self.zdata = zdata
        self.n_star = len(orbs_obs)

        # Initialization
        if init_type not in [None,'default','truths']:
            print('init_type must be one of "uninformed", "truths"')
        if init_type == None:
            print('No init_type specified, using default "uninformed" init')
            init_type = 'default'
        self.init_type = init_type
        if init_type == 'default':
            init = pdens.get_densfunc_mcmc_init_uninformed(densfunc)
        if init_type == 'truths':
            init = self.get_truths(normed=True)
        self.init = init

        # if init_type is not None:
        #     print('Non-default init_type not supported for MockHaloFit, setting'
        #         ' init_type=None')
        #     init_type = None
        # self.init_type = init_type
        # if init is None:
        #     if init_type is None:
        #         if verbose:
        #             print('Using default init')
        #         init = pdens.get_densfunc_mcmc_init_uninformed(densfunc)
        #     # else:
        #     #     if verbose:
        #     #         print('Using informed init')
        #     #     init = self.get_densfunc_mcmc_init_informed(init_type=init_type,
        #     #         verbose=verbose)
        # self.init = init
            

    def get_fit_effsel(self):
        '''get_fit_effsel:
        
        Get the proper effective selection function for the fit
        '''
        if self.fit_type in ['mock+ksf']:
            return self.keffsel
        elif self.fit_type in ['mock','mock+disk']:
            return self.effsel

    def get_truths(self,normed=False,theta_in_deg=False,phi_in_deg=False):
        '''get_truths:

        Return the truths for the fit
        '''
        if self.truths is None:
            raise ValueError('No truths set for this fit')
        if normed: # Ignore radians and degrees
            if self.truths_normed:
                return self.truths
            else:
                return pdens.normalize_parameters(self.truths,self.densfunc)
        else:
            if self.truths_normed:
                _ts =  pdens.denormalize_parameters(self.truths,self.densfunc)
            else:
                _ts = self.truths[:]
            theta_indx,phi_indx = pdens.get_densfunc_params_indx(self.densfunc,
                ['theta','phi'])
            if theta_in_deg:
                _ts[theta_indx] = np.rad2deg(_ts[theta_indx])
            if phi_in_deg:
                _ts[phi_indx] = np.rad2deg(_ts[phi_indx])
            return _ts

# Null prior for class
def _null_prior(densfunc,params):
    return 0

def check_hf_versions(hf=None,densfunc=None,selec=None,feh_range=None,
                      fit_type=None,fit_dir=None,return_versions=False,
                      print_versions=True):
    '''check_hf_versions:
    
    Check the versions available for a given combination of densfunc, 
    selec, feh_range, fit_type, fit_dir. If hf is supplied then take 
    parameters from it.
    
    Args:
        hf (HaloFit) - HaloFit instance. Will look for all versions with same
            densfunc, selec, feh_range, fit_type, fit_dir
        densfunc (callable) - Density function, takes (params,R,phi,z)
        selec (str) - Selection version, 'eLz', 'AD', or 'JRLz'
        feh_range (list) - [feh_min,feh_max] for selection
        fit_type (str) - fit type e.g. 'gse' or 'all'
        fit_dir (str) - Top level directory for data products
        return_versions (bool) - Return versions
        print_versions (bool) - Print versions
        
    Returns:
        versions (list) - List of version strings for given densfunc and 
            parameters
    '''
    # Handle inputs
    if hf is None:
        ps = [densfunc,selec,feh_range,fit_type,fit_dir]
        assert np.all([(p is not None) for p in ps]),\
            'Must supply either hf or all of densfunc, selec, feh_range, '+\
            'fit_type, fit_dir'
    else:
        print('HaloFit instance supplied, using densfunc, selec, feh_range, '+\
              'fit_type, fit_dir from it')
        densfunc = hf.densfunc
        selec = hf.selec
        feh_range = hf.feh_range
        fit_type = hf.fit_type
        fit_dir = hf.fit_dir
    
    # Handle selec, assume it's a list, tuple, or array
    if isinstance(selec,(list,tuple,np.ndarray)):
        selec = selec[0]

    # Unpack [Fe/H]
    feh_min,feh_max = feh_range
    
    # Make the selection string
    if 'all' in fit_type:
        selec_str = ''
    else:
        if selec[-1] == '/':
            selec_str = selec
        else:
            selec_str = selec+'/'
    
    # Make the fit directory
    fit_data_dir = fit_dir+'data/'+fit_type+'/'+selec_str+str(feh_min)+\
        '_feh_'+str(feh_max)+'/'+densfunc.__name__+'/'
    print('Checking for versions in path: '+fit_data_dir)

    try:
        res = os.listdir(fit_data_dir)
        if print_versions:
            print(res)
        if return_versions:
            return res
    except FileNotFoundError:
        print('warning: path does not exist, path: '+str(fit_data_dir))


def fraction_stars_for_fdisk_from_halo_disk_mocks(fdisk_targ, halo_densfunc, 
    halo_params, halo_mass, disk_densfunc, disk_params, disk_mass, r_range_halo, 
    R_range_disk, z_max_disk, ro, zo, mass_from_density_samples_kwargs={}):
    '''fraction_stars_for_fdisk_from_halo_disk_mocks:

    Use the halo and disk mocks to determine the fraction of stars from the 
    disk mock that corresponds to a target fdisk:

    Args:
        fdisk_targ (float): Target value of fdisk, should be in range [0,1]
        halo_densfunc (function): Density function for the halo mock. Should be 
            from src/ges_mass/densprofiles.py
        halo_params (array): Parameters for the halo mock
        halo_mass (float): Mass of the halo mock, can be astropy unit
        disk_densfunc (function): Density function for the disk mock, should be
            exponential disk from src/ges_mass/densprofiles.py
        disk_params (array): Parameters for the disk mock, should be inverse
            scale length and inverse scale height. Can be astropy units
        disk_mass (float): Mass of the disk mock, can be astropy unit
        r_range_halo (array): Minimum and maximum radii for the halo mock, 
            can be astropy units
        R_range_disk (array): Minimum and maximum radii for the disk mock,
            can be astropy units
        z_max_disk (float): Maximum height for the disk mock, can be astropy
            unit
        ro (float): Solar cylindrical radius, equiv. to galpy ro
        zo (float): Solar cylindrical height above plane, equiv. to galpy zo
        mass_from_density_samples_kwargs (dict): Keyword arguments for
            mass_from_density_samples function. If none supplied then will be 
            filled with appropriate trivial defaults.
        
    Returns:
        Adisk (float): Factor to scale the density of the disk potential by
            to get the desired fdisk. In practice take int(Adisk*n_disk_star)
            stars from the disk mock and add them to the halo mock in order to 
            get the desired fdisk.
    '''
    # Defualts for mass_from_density_samples_kwargs. Need to be trivial so 
    # mass comes out properly
    if 'n_star' not in mass_from_density_samples_kwargs.keys():
        mass_from_density_samples_kwargs['n_star'] = 1
    if 'effsel' not in mass_from_density_samples_kwargs.keys():
        mass_from_density_samples_kwargs['effsel'] = np.array([1.])
    if 'effsel_grid' not in mass_from_density_samples_kwargs.keys():
        mass_from_density_samples_kwargs['effsel_grid'] = [ro,0.,zo]
    if 'iso' not in mass_from_density_samples_kwargs.keys():
        mass_from_density_samples_kwargs['iso'] = 1.
    if 'feh_range' not in mass_from_density_samples_kwargs.keys():
        mass_from_density_samples_kwargs['feh_range'] = [0.,0.]
    if 'logg_range' not in mass_from_density_samples_kwargs.keys():
        mass_from_density_samples_kwargs['logg_range'] = [0.,0.]
    if 'jkmins' not in mass_from_density_samples_kwargs.keys():
        mass_from_density_samples_kwargs['jkmins'] = [0.,0.]
    if 'n_mass' not in mass_from_density_samples_kwargs.keys():
        mass_from_density_samples_kwargs['n_mass'] = 1
    if 'mass_analytic' not in mass_from_density_samples_kwargs.keys():
        mass_from_density_samples_kwargs['mass_analytic'] = False
    if 'nprocs' not in mass_from_density_samples_kwargs.keys():
        mass_from_density_samples_kwargs['nprocs'] = None
    if 'batch' not in mass_from_density_samples_kwargs.keys():
        mass_from_density_samples_kwargs['batch'] = False
    if '_isofactors' not in mass_from_density_samples_kwargs.keys():
        mass_from_density_samples_kwargs['_isofactors'] = 1.
    
    # Unpack arrays
    r_min_halo, r_max_halo = r_range_halo
    R_min_disk, R_max_disk = R_range_disk
    ihR, ihz = disk_params

    # Handle astropy units
    if isinstance(halo_mass,apu.quantity.Quantity):
        halo_mass = halo_mass.to(apu.M_sun).value
    if isinstance(disk_mass,apu.quantity.Quantity):
        disk_mass = disk_mass.to(apu.M_sun).value
    if isinstance(r_min_halo,apu.quantity.Quantity):
        r_min_halo = r_min_halo.to(apu.kpc).value
    if isinstance(r_max_halo,apu.quantity.Quantity):
        r_max_halo = r_max_halo.to(apu.kpc).value
    if isinstance(R_min_disk,apu.quantity.Quantity):
        R_min_disk = R_min_disk.to(apu.kpc).value
    if isinstance(R_max_disk,apu.quantity.Quantity):
        R_max_disk = R_max_disk.to(apu.kpc).value
    if isinstance(z_max_disk,apu.quantity.Quantity):
        z_max_disk = z_max_disk.to(apu.kpc).value
    if isinstance(ihR,apu.quantity.Quantity):
        ihR = ihR.to(1/apu.kpc).value
    if isinstance(ihz,apu.quantity.Quantity):
        ihz = ihz.to(1/apu.kpc).value
    
    # Repack disk parameters
    disk_params = [ihR,ihz]

    # Ensure int_r_range set correctly
    if 'int_r_range' not in mass_from_density_samples_kwargs.keys():
        mass_from_density_samples_kwargs['int_r_range'] = [r_min_halo,r_max_halo]
    else:
        assert mass_from_density_samples_kwargs['int_r_range'][0] == r_min_halo
        assert mass_from_density_samples_kwargs['int_r_range'][1] == r_max_halo

    # mass from density samples expects 2d list of parameters
    halo_samples = np.atleast_2d(halo_params)

    # Normalize the density profiles to their masses so that the fdisk factors 
    # are calculated correctly
    print('Calculating masses to normalize mock density profiles')
    _halo_mass = mass_from_density_samples(halo_samples, halo_densfunc, 
        **mass_from_density_samples_kwargs)[0]
    _amp_halo = halo_mass/_halo_mass
    _disk_mass = double_exponential_disk_cylindrical_mass(1/ihR,1/ihz,
        R_min_disk, R_max_disk, z_max_disk)
    _amp_disk_mass = disk_mass/_disk_mass
    _exp_disk = lambda R,z,hR,hz: np.exp(-R/hR)*np.exp(-np.abs(z)/hz)
    _amp_disk_sol = _exp_disk(ro,zo,1/ihR,1/ihz)
    _amp_disk = _amp_disk_mass*_amp_disk_sol

    # Density at the solar position
    disk_dens_sol = float(disk_densfunc(ro,0,zo,disk_params)*_amp_disk)
    halo_dens_sol = float(halo_densfunc(ro,0,zo,halo_params)*_amp_halo)
    # fdisk = disk_dens/(disk_dens+halo_dens)
    # print('Current fdisk at solar position '+str(fdisk))

    # Determine the factor needed to scale the disk distribution by to get
    # The appropriate fdisk
    Adisk = fdisk_targ*halo_dens_sol/(disk_dens_sol*(1-fdisk_targ))
    return Adisk


def mock_to_hf(mock_path,dfs,mixture_arr,aAS,pot,
    kinematic_selection_mask_kwargs={},hf_kwargs={}, ro=None, vo=None, zo=None, 
    disk_mock_path=None, fdisk_seed=None, fdisk_calc_kwargs={}):
    '''mock_to_hf:

    Use a mock to generate a HaloFit instance. Kinematics are calculated 
    according to the galpy distribution functions in dfs and the mixture
    weights. Actions and eccentricities are calculated using Staeckel, and 
    the data is masked.

    Args:
        mock_path (str) - Path to mock
        dfs (list) - List of galpy distribution function objects
        mixture_arr (np.ndarray) - Array of distribution function mixture 
            weights, shape (len(dfs)), sum(mixture_arr) = 1
        aAS (galpy actionAngleStaeckel object) - ActionAngleStaeckel object for 
            eccentricity and action calculation
        pot (galpy potential object) - Potential object for eccentricity and
            action calculation
        kinematic_selection_mask_kwargs (dict) - Dictionary of keyword arguments
             for putil.kinematic_selection_mask function. 'space' or 'selection' 
             must be supplied. 'selection_version' can also be supplied.
        hf_kwargs (dict) - Dictionary of keyword arguments for MockHaloFit
            class instantiation
        ro (float) - Distance to Sun, same as galpy ro
        vo (float) - galpy vo
        zo (float) - Height of Sun above plane, same as galpy zo
        disk_mock_path (str) - Path to disk mock, must be provided if fit_type
            is 'mock+disk'
        fdisk (float) - Requested fdisk, must be provided if fit_type is
            'mock+disk'
        fdisk_calc_kwargs (dict) - Dictionary of keywords arguments for 
            fraction_stars_for_fdisk_from_halo_disk_mocks function
        
    Returns:
        hf (HaloFit) - HaloFit instance
    '''
    # Handle ro,vo,zo
    if ro is None:
        if 'ro' in hf_kwargs.keys():
            ro = hf_kwargs['ro']
        else:
            raise AttributeError
    if vo is None:
        if 'vo' in hf_kwargs.keys():
            vo = hf_kwargs['vo']
        else:
            raise AttributeError
    if zo is None:
        raise AttributeError

    # Get information from hf_kwargs and modify if need be
    fit_type = hf_kwargs['fit_type']
    assert fit_type in ['mock','mock+ksf','mock+disk']
    if 'selec' not in hf_kwargs.keys() or hf_kwargs['selec'] is None:
        if 'space' in kinematic_selection_mask_kwargs.keys():
            hf_kwargs['selec'] = kinematic_selection_mask_kwargs['space']
    # These will be loaded in
    if 'allstar' in hf_kwargs.keys():
        del hf_kwargs['allstar']
    if 'orbs' in hf_kwargs.keys():
        del hf_kwargs['orbs']
    # These are only for real data
    if 'mask_disk' in hf_kwargs.keys():
        del hf_kwargs['mask_disk']
    if 'mask_halo' in hf_kwargs.keys():
        del hf_kwargs['mask_halo']

    # Load the mock
    print('Loading halo mock from: '+str(mock_path))
    if mock_path[-1] != '/':
        mock_path += '/'
    orbs_filename = mock_path+'orbs.pkl'
    allstar_filename = mock_path+'allstar.npy'
    data_omask_filename = mock_path+'omask.npy'
    with open(orbs_filename,'rb') as f:
        orbs_nomask = pickle.load(f)
    orbs_nomask.turn_physical_on(ro=ro,vo=vo)
    allstar_nomask = np.load(allstar_filename)
    data_mask = np.load(data_omask_filename)

    # Mask the mock data
    orbs = orbs_nomask[data_mask]
    allstar = allstar_nomask[data_mask]

    # Only do kinematics if required
    if 'ksf' in fit_type:
        print('Calculating kinematics and masking...')
        # Calculate kinematics
        orbs = putil.orbit_kinematics_from_df_samples(orbs,dfs,mixture_arr)
        _,eELzs,accs,_ = putil.calculate_accs_eELzs_orbextr_Staeckel(orbs,
            pot=pot,aAS=aAS)
        # Create mask based on kinematics
        phi0 = potential.evaluatePotentials(pot,1e10,0).value
        kmask = putil.kinematic_selection_mask(orbs,eELzs,accs,phi0=phi0,
            **kinematic_selection_mask_kwargs)
    
    # Get disk contamination if required
    if 'disk' in fit_type:
        print('Calculating disk contamination...')
        # Load the disk mock
        if disk_mock_path[-1] != '/':
            disk_mock_path += '/'
        disk_orbs_filename = disk_mock_path+'orbs.pkl'
        disk_allstar_filename = disk_mock_path+'allstar.npy'
        disk_data_omask_filename = disk_mock_path+'omask.npy'
        with open(disk_orbs_filename,'rb') as f:
            disk_orbs_nomask = pickle.load(f)
        disk_orbs_nomask.turn_physical_on(ro=ro,vo=vo)
        disk_allstar_nomask = np.load(disk_allstar_filename)
        disk_data_mask = np.load(disk_data_omask_filename)
        
        # Mask the disk mock data
        disk_orbs = disk_orbs_nomask[disk_data_mask]
        disk_allstar = disk_allstar_nomask[disk_data_mask]

        # Calculate how many disk stars to take for correct fdisk
        Adisk = fraction_stars_for_fdisk_from_halo_disk_mocks(
                **fdisk_calc_kwargs)
        n_disk_contaminant = round(len(disk_orbs)*Adisk)
        rnp = np.random.default_rng(seed=fdisk_seed)
        disk_contam_indx = rnp.choice(np.arange(0,len(disk_orbs),dtype=int), 
                                size=n_disk_contaminant, replace=False)
        disk_orbs_contam = disk_orbs[disk_contam_indx]
        disk_allstar_contam = disk_allstar[disk_contam_indx]

    # Create the HaloFit object depending on fit_type
    print('Instantiating HaloFit object...')
    if fit_type == 'mock':
        hf = MockHaloFit(allstar=allstar, orbs=orbs, **hf_kwargs)
    elif fit_type == 'mock+ksf':
        orbs_kmask = orbs[kmask]
        allstar_kmask = allstar[kmask]
        hf = MockHaloFit(allstar=allstar_kmask, orbs=orbs_kmask,
            **hf_kwargs)
    else:
        # First must calculate the disk contamination
        orbs_disk_all = putil.join_orbs([orbs,disk_orbs_contam])
        # It's likely that the mock allstar will not have all the fields that
        # the real data does, so index the subset of necessary fields
        _common_fields = ['FE_H','LOGG']
        allstar_disk_all = np.append(allstar[_common_fields],
                                     disk_allstar_contam[_common_fields])
        hf = MockHaloFit(allstar=allstar_disk_all, orbs=orbs_disk_all, 
            **hf_kwargs)
    
    return hf