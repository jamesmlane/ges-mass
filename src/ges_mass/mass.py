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
import warnings
import scipy.optimize
from scipy.stats import norm
from isodist import FEH2Z, Z2FEH
from galpy.util import coords
from astropy import units as apu
from . import densprofiles as pdens
from . import util as putil
from . import iso as piso

_ro = 8.275 # Gravity Collab.
_zo = 0.0208 # Bennett and Bovy

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
        optimizer_method='Powell')
    
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
    autocorr_n = 200 # Interval between checking autocorrelation criterion
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
            
            # Record MCMC diagnostics
            mcmc_diagnostic_txt = (
                'sampled '+str(i+1)+'\n'\
                'mean tau: '+str(mean_tau)+'\n'+\
                'min nit/tau: '+str(nit_tau)+'\n'+\
                '[min,max] delta tau: ['+\
                str(np.min(delta_tau))+','+str(np.max(delta_tau))+']\n'+\
                '[min,max] acceptance fraction: ['+\
                str(round(np.min(sampler.acceptance_fraction),2))+','+\
                str(round(np.max(sampler.acceptance_fraction),2))+']\n'+\
                'total max nit/tau '+str(max_nit_tau)+'\n'+\
                '(nit/tau)/max[(nit/tau)] '+str(nit_tau/max_nit_tau)+'\n'+\
                '---')
            print(mcmc_diagnostic_txt)
                
            # print(mcmc_diagnostic_txt)
            if has_mcmc_diagnostic_file:
                mcmc_diagnostic_file.write(mcmc_diagnostic_txt)
            
            # Check convergence
            converged = np.all(tau * convergence_n_tau < sampler.iteration)
            converged &= np.all(np.abs(delta_tau) < convergence_delta_tau)
            if converged:
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
    

def mass_from_density_samples(samples, densfunc, n_star, effsel, effsel_grid, 
                              iso, feh_range, logg_range, jkmins, n_mass=400,
                              mass_int_type='spherical_grid', 
                              mass_analytic=False, int_r_range=[2.,70.], 
                              n_edge=[500,100,100], nprocs=None, batch=False, 
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
                if (i+1)%10 == 0: print('sampled '+str(i+1)+'/'+str(n_mass), 
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
            diskdens = denstxyz[1]*facs[i]
            fulldens = densfunc(Rphizgrid[:,0],Rphizgrid[:,1],Rphizgrid[:,2], 
                             params=params[i])*facs[i]
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


def fdisk_to_number_of_stars(hf,samples=None,nprocs=1):
    '''fdisk_to_number_of_stars:
    
    Convert fdisk to number of halo and disk stars in the sample
    
    Args:
        hf (HaloFit) - HaloFit class containing all information about fit
        samples (array) - MCMC samples, shape is (nsample,ndim)
        nprocs (int) - Number of processors to use [default 1]
    
    Returns:
        n_halo (int) - Number of halo stars
        n_disk (int) - Number of disk stars
    '''
    if samples is None:
        if hf.samples is None:
            hf.get_results()
            assert hf.samples is not None,\
                'No samples in supplied HaloFit instance, tried get_results()'
        samples = hf.samples

    samples = np.atleast_2d(samples)
    n_samples = samples.shape[0]
    n_star_halo = np.zeros(n_samples,dtype=int)
    n_star_disk = np.zeros(n_samples,dtype=int)
    
    assert 'plusexpdisk' in hf.densfunc.__name__,\
        'densfunc must have disk contamination (plusexpdisk) to have fdisk'
    
    # Unpack min and max for [Fe/H], logg, effective selection function grid,
    # (kinematic) effective selection function
    feh_min, feh_max = hf.feh_range
    logg_min, logg_max = hf.logg_range
    Rgrid,phigrid,zgrid = hf.get_effsel_list()
    effsel = hf.get_fit_effsel()
    
    hasEffVol = hasattr(hf,'effvol_halo') and hasattr(hf,'effvol_disk')
    n_star = hf.n_star

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
    
    def get_results(self,load_sampler=False):
        '''get_results:
        
        Get results from MCMC and mass calculation. Note that the pickled
        sampler is quite big so it's optionally loaded.
        
        Args:
            load_sampler (bool) - Load the emcee sampler [default False]
        
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
            self.get_sampler()
    
    def get_sampler(self):
        '''get_sampler:
        
        Load the pickled sampler
        
        Sets
            sampler - MCMC sampler object
        '''
        sampler_filename = self.fit_data_dir+'sampler.pkl'
        
        if os.path.exists(sampler_filename):
            with open(sampler_filename,'rb') as f:
                self.sampler = pickle.load(f)
    
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
    
    def run_optimization(self,init,method='Powell',optimizer_kwargs={}):
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
        if self.verbose:
            print('Doing maximum likelihood')
        effsel = self.get_fit_effsel()
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
            mll = -np.abs(np.max(self.loglike))
        elif ml_type == 'post':
            if not self.opt_post.success:
                print('warning: post optimization was not successful')
            mll = -np.abs(self.opt_post.fun)
        elif ml_type == 'init':
            if not self.opt_init.success:
                print('warning: initial optimization was not successful')
            mll = -np.abs(self.opt_init.fun)
        
        aic = 2*n_param - 2*mll
        bic = np.log(n_star)*n_param - 2*mll
        
        return mll,aic,bic
    
    # Utils
    
    def _check_not_set(self):
        '''_check_not_set:
        
        Check to see if class attributes were left as None at instantiation
        '''
        if not self.verbose: return None
        attrs_exclude = ['opt_init','opt_post','samples','sampler','masses',
                         'mass_inds','facs','isofactors','ml','ml_ind','aic',
                         'bic','loglike']
        attrs = [a for a in dir(self) if not a.startswith('_') \
                 and not a in attrs_exclude and not callable(getattr(self, a))]
        for a in attrs:
            res = getattr(self,a)
            if isinstance(res,np.ndarray):
                continue
            if a == 'iso' and getattr(self,'iso_filename') not in [None,'']:
                continue
            if res in [None,'']:
                print('warning: attribute '+str(a)+' not set')
        return None
        
        

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
                 iso_feh=None,
                 iso_age=None,
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
        fit_data_dir  = fit_dir+'data/'+fit_type+'/'+selec_str+'feh_'+\
            str(iso_feh)+'/'+densfunc.__name__+'/'+self.version
        fit_fig_dir  = fit_dir+'fig/'+fit_type+'/'+selec_str+'feh_'+\
            str(iso_feh)+'/'+densfunc.__name__+'/'+self.version
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
        if init_type is not None:
            print('Non-default init_type not supported for MockHaloFit, setting'
                ' init_type=None')
            init_type = None
        self.init_type = init_type
        if init is None:
            if init_type is None:
                if verbose:
                    print('Using default init')
                init = pdens.get_densfunc_mcmc_init_uninformed(densfunc)
            # else:
            #     if verbose:
            #         print('Using informed init')
            #     init = self.get_densfunc_mcmc_init_informed(init_type=init_type,
            #         verbose=verbose)
        self.init = init
            

    def get_fit_effsel(self):
        '''get_fit_effsel:
        
        Get the proper effective selection function for the fit
        '''
        if self.fit_type in ['mock+ksf']:
            return self.keffsel
        elif self.fit_type in ['mock','mock+disk']:
            return self.effsel


# Null prior for class
def _null_prior(densfunc,params):
    return 0

def check_hf_versions(densfunc,selec,feh_range,fit_type,fit_dir):
    '''check_hf_versions:
    
    For a given densfunc
    
    Args:
    
    Returns:
        versions (list) - List of version strings for given densfunc and 
            parameters
    '''
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
    
    try:
        res = os.listdir(fit_data_dir)
        print(res)
    except FileNotFoundError:
        print('warning: path does not exist, path: '+str(fit_data_dir))