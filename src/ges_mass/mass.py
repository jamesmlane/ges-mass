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
import tqdm
import warnings
import scipy.optimize
from scipy.stats import norm
from isodist import FEH2Z, Z2FEH
from galpy.util import coords
from . import densprofiles as pdens
from . import util as putil
from . import iso as piso
from . import mass as pmass

_ro = 8.275 # Gravity Collab.
_zo = 0.0208 # Bennett and Bovy

_PRIOR_ETA_MIN = 0.5

# ----------------------------------------------------------------------------


### Fitting


def fit_dens(densfunc, effsel, effsel_grid, data, init, nprocs, nwalkers=100,
             nit=250, ncut=100, usr_log_prior=None, MLE_init=True, 
             just_MLE=False, return_walkers=False, convergence_n_tau=50,
             convergence_delta_tau=0.01, optimizer_method='Nelder-Mead',
             mcmc_diagnostic_filename=None):
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
        opt_fn = lambda x: pmass.mloglike(x, densfunc, effsel, Rgrid, phigrid, 
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
    first_return = True
    has_mcmc_diagnostic_file = False
    if isinstance(mcmc_diagnostic_filename,str):
        mcmc_diagnostic_file = open(mcmc_diagnostic_filename,'w')
        has_mcmc_diagnostic_file = True
    
    # Do MCMC
    with multiprocessing.Pool(nprocs) as pool:
        # Make sampler
        sampler = emcee.EnsembleSampler(nwalkers, ndim, pmass.loglike, 
            args=(densfunc, effsel, Rgrid, phigrid, zgrid, Rdata, phidata, 
                  zdata, usr_log_prior), pool=pool)
        
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
            mcmc_diagnostic_txt = ('sampled '+str(i+1)+'\n'\
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
            converged &= np.all(np.abs(delta_tau) < 0.01)
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


def mass_from_density_samples(samples, densfunc, n_star, effsel, effsel_grid, 
                              iso, feh_range, logg_range, jkmins, n_mass=400,
                              mass_int_type='spherical_grid', 
                              mass_analytic=False, int_r_range=[2.,70.], 
                              n_edge=[500,100,100], nprocs=None, batch=False, 
                              ro=_ro, zo=_zo, seed=0, verbose=True):
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
            # The average mass mask extracts fitted sample based on color and logg
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
            # The average mass mask extracts fitted sample based on color and logg
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


def fdisk_to_number_of_stars(hf,samples,nprocs=1):
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
    samples = np.atleast_2d(samples)
    n_samples = samples.shape[0]
    n_star_halo = np.zeros(n_samples,dtype=int)
    n_star_disk = np.zeros(n_samples,dtype=int)
    
    assert 'plusexpdisk' in hf.densfunc.__name__
    
    # Unpack min and max for [Fe/H], logg, effective selection function grid,
    # (kinematic) effective selection function
    feh_min, feh_max = hf.feh_range
    logg_min, logg_max = hf.logg_range
    Rgrid,phigrid,zgrid = hf.get_effsel_grid()
    effsel = hf.get_fit_effsel()
    
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


def mloglike(*args, **kwargs):
    '''mloglike:
    
    Args:
        args (args) - Arguments to pass to loglike
        kwargs (kwargs) - Keyword arguments to pass to loglike
    
    Returns:
        mloglike (array) - Negative of the loglikelihood function
    '''
    return -loglike(*args,**kwargs)


def loglike(params, densfunc, effsel, Rgrid, phigrid, zgrid, dataR, dataphi, 
            dataz, usr_log_prior=None):
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
        dataR (array) - data R positions
        dataphi (array) - data phi positions
        dataz (array) - data z positions
        usr_log_prior (callable) - Extra prior supplied by the user at runtime. 
            Included for flexibility so not all priors need to be hardcoded. 
            Call signature should be usr_log_prior(densfunc,params) and function 
            should return the log of the prior value. Will check for -np.inf 
            and break out of the likelihood call if it is returned.
    
    Returns:
        log_posterior (array) - log of the likelihood + log of the prior
    '''
    # Evaluate the domain of the prior
    if not domain_prior(densfunc, params):
        return -np.inf
    # Evaluate the user-supplied prior
    if callable(usr_log_prior):
        usrlogprior = usr_log_prior(densfunc,params)
        if np.isneginf(usrlogprior):
            return usrlogprior
    else:
        usrlogprior = 0
    # Evaluate the informative prior
    logprior = log_prior(densfunc, params)
    logdatadens = np.log(tdens(densfunc, dataR, dataphi, dataz, params=params))
    logeffvol = np.log(effvol(densfunc,effsel,Rgrid,phigrid,zgrid,params=params))
    #log likelihood
    loglike = np.sum(logdatadens)-len(dataR)*logeffvol
    if not np.isfinite(loglike):
        return -np.inf
    return logprior + usrlogprior + loglike


def effvol(densfunc, effsel, Rgrid, phigrid, zgrid, params=None):
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
    
    Returns:
        effvol (array) - The effective volume 
    '''
    if params is None:
        effdens = tdens(densfunc,Rgrid,phigrid,zgrid)
    else:
        effdens = tdens(densfunc,Rgrid,phigrid,zgrid,params=params)
    return np.sum(effdens*effsel)


def tdens(densfunc, Rgrid, phigrid, zgrid, params=None):
    '''tdens:
    
    Deterine the densities at the locations corresponding to a supplied grid
    and density function.
    
    Args:
        densfunc (function) - Density function
        Rgrid (array) - Array of R positions
        phigrid (array) - Array of phi positions
        zgrid (array) - Array of z positions
        params (list) - Density model parameters
    
    Returns:
        dens (array) - Densities corresponding to supplied positions and density
            function
    '''
    if params is None:
        dens = densfunc(Rgrid,phigrid,zgrid)
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

### Plotting

def plot_model(model, params, minmax=[-50,50], nside=150):
    '''plot_model:
    
    Plot the density of a model in 3 planes: XY, YZ, XZ
    
    Args:
        model (function) - Density profile
        params (list) - Parameters that describe the density model
        minmax (list) - Range of model to show
        nside (int) - Number of samples for the grid to show
    
    Returns:
        None (although fig and ax are created and shown)
    '''
    xyzgrid = np.mgrid[minmax[0]:minmax[1]:nside*1j,
                       minmax[0]:minmax[1]:nside*1j,
                       minmax[0]:minmax[1]:nside*1j]
    shape = np.shape(xyzgrid.T)
    xyzgrid = xyzgrid.T.reshape(np.product(shape[:3]),shape[3])
    rphizgrid = coords.rect_to_cyl(xyzgrid[:,0], xyzgrid[:,1], xyzgrid[:,2])
    rphizgrid = np.dstack([rphizgrid[0],rphizgrid[1],rphizgrid[2]])[0]
    rphizgrid = rphizgrid.reshape(nside,nside,nside,3).T
    denstxyz = model(rphizgrid[0],rphizgrid[1],rphizgrid[2], params=params)
    fig, ax = plt.subplots(1,3)
    fig.set_size_inches(10,3.4)
    ax[0].contour(np.rot90(np.log10(np.sum(denstxyz, axis=0))), 
                  extent=[minmax[0],minmax[1],minmax[0],minmax[1]], 
                  cmap=plt.cm.cividis)
    ax[1].contour(np.rot90(np.log10(np.sum(denstxyz, axis=1))), 
                  extent=[minmax[0],minmax[1],minmax[0],minmax[1]], 
                  cmap=plt.cm.cividis)
    ax[2].contour(np.rot90(np.log10(np.sum(denstxyz, axis=2))), 
                  extent=[minmax[0],minmax[1],minmax[0],minmax[1]], 
                  cmap=plt.cm.cividis)
    xdat, ydat, zdat = coords.cyl_to_rect(Rphiz[:,0], Rphiz[:,1], Rphiz[:,2])
    ax[0].set_xlabel(r'y')
    ax[0].set_ylabel(r'z')
    ax[1].set_xlabel(r'x')
    ax[1].set_ylabel(r'z')
    ax[2].set_xlabel(r'x')
    ax[2].set_ylabel(r'y')
    for axis in ax:
        axis.set_ylim(minmax[0],minmax[1])
        axis.set_xlim(minmax[0],minmax[1])
    fig.tight_layout()


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

class HaloFit:
    '''HaloFit:
    
    Convenience class to wrap all of the information needed by various 
    fitting and plotting routines.
    '''
    
    # Initialize
    
    def __init__(self,
                 allstar=None,
                 orbs=None,
                 densfunc=None,
                 nwalkers=None,
                 nit=None,
                 ncut=None,
                 selec=None,
                 usr_log_prior=None,
                 n_mass=None,
                 effsel=None,
                 dmods=None,
                 effsel_grid=None,
                 effsel_mask=None,
                 iso=None,
                 iso_filename=None,
                 int_r_range=None,
                 feh_range=None,
                 logg_range=None,
                 jkmins=None,
                 init=None,
                 init_type=None,
                 fit_dir=None,
                 gap_dir=None,
                 ksf_dir=None,
                 fit_type=None,
                 mask_disk=True,
                 mask_halo=True,
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
            densfunc (callable) - Density profile
            nwalkers (int) - Number of MCMC walkers
            nit (int) - Number of steps to run each walker
            ncut (int) - Number of steps to trim from beginning of each chain
            selec (str or arr) - Kinematic selection space
            usr_log_prior (callable) - User supplied log prior for densfunc
            n_mass (int) - Number of masses to calculate
            effsel (array) - Effective selection function calculated on a grid 
                of size (nfield,ndmod) without kinematic selection effects
            dmods (array) - Array of distance modulus
            effsel_grid (list) - Length-3 list of grids of shape (effsel) 
                representing R,phi,z positions where the selection function is 
                evaluated
            effsel_mask (array) - Effective selection function grid mask of 
                shape (effsel)
            iso (array) - Isochrone grid
            iso_filename (str) - Filename to access the isochrone grid
            int_r_range (array) - 2-element list of spherical integration range
            feh_range (array) - 2-element list of Fe/H min, Fe/H max
            logg_range (array) - 2-element list of logg min, logg max
            jkmins (array) - Array of minimum (J-K) values
            init (array) - Supply initialization parameters, if None then 
            init_type (bool) - Type of initialization to 
            fit_dir (str) - Directory for holding fitting data and figures
            gap_dir (str) - Gaia-APOGEE processed data directory
            ksf_dir (str) - kSF directory
            fit_type (str) - string specifying the type of fit. Should be one 
                of ['gse','all']
            mask_disk (bool) - Mask the disk if fit_type='gse'? [default True]
            mask_halo (bool) - Use a halo mask from IDs if fit_type='all'?
                [default True]
            version (str) - Version string to add to filenames
            verbose (bool) - Print info to screen
            ro,vo,zo (float) - Galpy scales, also solar cylindrical radius. zo 
                is Solar height above the plane
        
        Returns:
            None
        '''
        # Unmasked data
        if allstar is not None and orbs is not None:
            assert len(allstar)==len(orbs), 'allstar,orbs length must be equal'
        # self.allstar_nomask = allstar
        # self.orbs_nomask = orbs
        
        # Density profile
        self.densfunc = densfunc
        
        # Kinematic selection space
        if selec is not None:
            if isinstance(selec,str): selec=[selec,]
            selec_suffix = '-'.join(selec)
        else:
            selec_suffix = None
        self.selec = selec_suffix
        self.selec_arr = selec
        
        # MCMC info
        self.nwalkers = nwalkers
        self.nit = nit
        self.ncut = ncut
        
        # Log prior
        if usr_log_prior is not None:
            self.usr_log_prior = usr_log_prior
        else:
            self.usr_log_prior = _null_prior
        
        # Mass calculation info
        if n_mass is None:
            n_mass = int(nwalkers*(nit-ncut))
        self.n_mass = n_mass
        self.int_r_range = int_r_range
        
        # Isochrone
        self.iso = iso
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
        
        # Fit type info
        if fit_type is not None:
            assert fit_type in ['gse','gse_map','all','all_map']
        else:
            warnings.warn('fit_type is required to access data')
            fit_type = ''
        self.fit_type = fit_type
        
        # I/O directories
        if fit_dir[-1] != '/': fit_dir+='/'
        self.fit_dir = fit_dir
        self.gap_dir = gap_dir
        self.ksf_dir = ksf_dir
        
        # Version
        if version != '':
            if version[-1] != '/': version+='/'
        else:
            version = ''
        self.version = version
        
        # Output directories
        if 'all' in fit_type:
            selec_str = ''
        else:
            selec_str = selec_suffix+'/'
        fit_data_dir = fit_dir+'data/'+fit_type+'/'+selec_str+str(feh_min)+\
                       '_feh_'+str(feh_max)+'/'+densfunc.__name__+'/'+version
        fit_fig_dir = fit_dir+'fig/'+fit_type+'/'+selec_str+str(feh_min)+\
                      '_feh_'+str(feh_max)+'/'+densfunc.__name__+'/'+version
        if not os.path.exists(fit_data_dir):
            os.makedirs(fit_data_dir,exist_ok=True)
        if not os.path.exists(fit_fig_dir):
            os.makedirs(fit_fig_dir,exist_ok=True)
        self.fit_data_dir = fit_data_dir
        self.fit_fig_dir = fit_fig_dir
        
        # Prepare the effective selection function
        self.effsel = effsel
        self.effsel_mask = effsel_mask
        Rgrid,phigrid,zgrid = effsel_grid
        self.Rgrid = Rgrid
        self.phigrid = phigrid
        self.zgrid = zgrid
        self.dmods = dmods
        
        # Get the kinematic effective selection function
        if 'gse' in fit_type:
            ksel = self.get_ksel(spline_type='linear',mask=True)
            keffsel = effsel*ksel
            assert np.all( ~np.all(keffsel < 1e-9, axis=1) ),\
                'Null fields still in keffSF'
            self.keffsel = keffsel
        else:
            self.keffsel = None
        
        # Galpy scales and zo
        self.ro = ro
        self.vo = vo
        self.zo = zo
        
        # Verbosity
        self.verbose = verbose
        
        # Mask out GS/E stars
        if 'gse' in fit_type:
            self.mask_disk = mask_disk
            self.mask_halo = None
            self.halo_mask = None
            gse_mask_filename = gap_dir+'hb_apogee_ids_'+selec_suffix
            if mask_disk:
                gse_mask_filename += '_dmask.npy'
            else:
                gse_mask_filename += '.npy'
            gse_apogee_IDs = np.load(gse_mask_filename)
            gse_mask = np.in1d(allstar['APOGEE_ID'].astype(str),gse_apogee_IDs)
            orbs_gse = orbs[gse_mask]
            allstar_gse = allstar[gse_mask]
            self.gse_mask = gse_mask

            # Obervational masking
            obs_mask = (allstar_gse['FE_H'] > feh_min) &\
                       (allstar_gse['FE_H'] < feh_max) &\
                       (allstar_gse['LOGG'] > logg_min) &\
                       (allstar_gse['LOGG'] < logg_max)
            self.obs_mask = obs_mask
            orbs_obs = orbs_gse[obs_mask]
            allstar_obs = allstar_gse[obs_mask]
            
        elif 'all' in fit_type:
            self.gse_mask = None
            self.mask_disk = None
            self.mask_halo = mask_halo
            if mask_halo:
                halo_mask_filename = gap_dir+'halo_apogee_ids.npy'
                halo_apogee_IDs = np.load(halo_mask_filename)
                halo_mask = np.in1d(allstar['APOGEE_ID'].astype(str),
                                    halo_apogee_IDs)
                self.halo_mask = halo_mask
            else:
                self.halo_mask = None
            
            obs_mask = (allstar['LOGG'] > logg_min) &\
                       (allstar['LOGG'] < logg_max)
            
            if mask_halo:
                obs_mask = obs_mask & halo_mask
            else: # Use a default metallicity selection
                obs_mask = obs_mask & (allstar['FE_H'] > feh_min) &\
                                      (allstar['FE_H'] < feh_max)
            
            self.obs_mask = obs_mask
            orbs_obs = orbs[obs_mask]
            allstar_obs = allstar[obs_mask] 
            
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
        
    # Getters
    
    def get_fit_effsel(self):
        '''get_fit_effsel:
        
        Get the proper effective selection function for the fit
        '''
        if self.fit_type in ['gse','gse_map']:
            return self.keffsel
        elif self.fit_type in ['all','all_map']:
            return self.effsel
    
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
            print('iso_filename not set, returning None')
            return None
    
    def get_ksel(self,spline_type='linear',mask=True):
        '''get_ksel:
        
        Return the kinematic effective selection function
        '''
        ksel_filename = self.ksf_dir+'kSF_grid_'+spline_type+'_'+\
            self.selec+'.dat'
        with open(ksel_filename,'rb') as f:
            print('\nLoading APOGEE kin. eff. sel. grid from '+ksel_filename)
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
                
        if os.path.exists(masses_filename):
            self.masses = np.load(masses_filename)
            
        if os.path.exists(mass_inds_filename):
            self.mass_inds = np.load(mass_inds_filename)
            
        if os.path.exists(facs_filename):
            self.facs = np.load(facs_filename)
        
        if os.path.exists(isofactors_filename):
            self.isofactors = np.load(isofactors_filename)
            
        if os.path.exists(opt_init_filename):
            with open(opt_init_filename,'rb') as f:
                self.opt_init = pickle.load(f)
        elif os.path.exists(opt_filename):
            with open(opt_filename,'rb') as f:
                self.opt_init = pickle.load(f)
            
        if os.path.exists(opt_post_filename):
            with open(opt_post_filename,'rb') as f:
                self.opt_post = pickle.load(f)
                
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
                print('warning post optimization was not successful')
            return self.opt_post.x
        
        if ml_type == 'init':
            assert self.opt_init is not None, 'opt_init is not set, run '+\
                'get_results()'
            if not self.opt_init.success:
                print('Warning initial optimization was not successful')
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
            print('Set self.ml, self.ml_ind, self.aic, self.bic')
        else:
            print('File containing ML, AIC, BIC does not exist')
        
        if os.path.exists(loglike_filename):
            self.loglike = np.load(loglike_filename)
            print('Set self.loglike')
        else:
            print('File containing log likelihoods does not exist')
    
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
            print('Setting init_type to uninformed because densfunc is '
                  'triaxial_single_angle_zvecpa')
            init_type = 'uninformed'

        # Unpack
        feh_min,feh_max = self.feh_range
        selec_suffix = self.selec
        
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
                print('Files required for init_type "ML" not present'
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
        
        Set a new densfunc for the class
        
        Args:
            densfunc (callable) - Density profile
        
        Returns:
            None
        '''
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
        return pmass.loglike(params, self.densfunc, self.get_fit_effsel(), 
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
        
        print('Unique isochrone factors: '+str(np.unique(isofactors)))
            
        return isofactors
    
    def run_optimization(self,init,method='Powell',optimizer_kwargs={}):
        '''run_optimization:
        '''
        print('Doing maximum likelihood')
        effsel = self.get_fit_effsel()
        opt_fn = lambda x: pmass.mloglike(x, self.densfunc, effsel, self.Rgrid, 
            self.phigrid, self.zgrid, self.Rdata, self.phidata, self.zdata,
            usr_log_prior=self.usr_log_prior)
        opt = scipy.optimize.minimize(opt_fn, init, method=method, 
                                      **optimizer_kwargs)
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
                print('warning post optimization was not successful')
            mll = -np.abs(self.opt_post.fun)
        elif ml_type == 'init':
            if not self.opt_init.success:
                print('Warning initial optimization was not successful')
            mll = -np.abs(self.opt_init.fun)
        
        aic = 2*n_param - 2*mll
        bic = np.log(n_star)*n_param - 2*mll
        
        return mll,aic,bic
    

class _HaloFit:
    '''_HaloFit:
    
    
    Parent class for HaloFit-type classes
    '''
    def __init__(self,
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
                 version=None,
                 effsel=None,
                 effsel_mask=None,
                 effsel_grid=None,
                 ):
        '''__init__:
        
        Initialize the _HaloFit parent class
        '''
        # Density profile
        self.densfunc = densfunc
        
        # Kinematic selection space
        if selec is not None:
            if isinstance(selec,str): selec=[selec,]
            selec_suffix = '-'.join(selec)
        else:
            selec_suffix = None
        self.selec = selec_suffix
        self.selec_arr = selec
        
        # MCMC info
        self.nwalkers = nwalkers
        self.nit = nit
        self.ncut = ncut
        
        # Log prior
        if usr_log_prior is not None:
            self.usr_log_prior = usr_log_prior
        else:
            self.usr_log_prior = _null_prior
        
        # Mass calculation info
        if n_mass is None:
            n_mass = int(nwalkers*(nit-ncut))
        self.n_mass = n_mass
        self.int_r_range = int_r_range
        
        # Isochrone
        self.iso = iso
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
        if version != '':
            if version[-1] != '/': version+='/'
        else:
            version = ''
        self.version = version
        
        # Prepare the effective selection function
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


class MockHaloFit(_HaloFit):
    '''MockHaloFit:
    '''
    def __init__(self,):
        '''__init__:
        
        Initialize a MockHaloFit class
        
        Args:
            
        Returns:
            None
        '''
        # Call parents constructor
        _HaloFit.__init__()
        
        # Do data
        
        # Do directories


# Null prior for class
def _null_prior(densfunc,params):
    return 0
        
        
        