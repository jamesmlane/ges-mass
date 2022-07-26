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
import numpy as np
import multiprocessing
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

# ----------------------------------------------------------------------------

### Fitting

def fit_dens(densfunc, effsel, effsel_grid, data, init, nprocs, nwalkers=100,
             nit=250, ncut=100, usr_log_prior=None, MLE_init=True, 
             just_MLE=False, return_walkers=False):
    '''fit_dens:
    
    Fit a density profile to a set of data given an effective selection 
    function evaluted over a grid. For larger triaxial models the nit should 
    be at least 2k-3k to satisfy autocorrelation timescale requirements. Burn 
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
        opt = scipy.optimize.fmin(lambda x: pmass.mloglike(x, densfunc, effsel, 
            Rgrid, phigrid, zgrid, Rdata, phidata, zdata, usr_log_prior), init, 
            full_output=True)
        print('MLE result: '+str(opt[0]))
        if just_MLE:
            return opt
    
    # Initialize MCMC from either init or MLE
    if MLE_init:
        pos = [opt[0] + 1e-3*np.random.randn(ndim) for i in range(nwalkers)]
    else:
        pos = [init + 1e-3*np.random.randn(ndim) for i in range(nwalkers)]
    
    # do MCMC
    with multiprocessing.Pool(nprocs) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, pmass.loglike, 
            args=(densfunc, effsel, Rgrid, phigrid, zgrid, Rdata, phidata, 
                  zdata, usr_log_prior), pool=pool)
        print('Generating MCMC samples...')
        for i, result in enumerate(sampler.sample(pos, iterations=nit)):
            if (i+1)%10 == 0: print('sampled '+str(i+1)+'/'+str(nit), end='\r')
            continue
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
                              ro=_ro, zo=_zo, seed=0):
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
    if densfunc is pdens.triaxial_single_angle_zvecpa_plusexpdisk or\
       densfunc is pdens.triaxial_single_cutoff_zvecpa_plusexpdisk:
        hasDisk = True
        masses = np.zeros((n_mass,3))
    else:
        hasDisk = False
        masses = np.zeros(n_mass)
    
    # Unpack effsel grid
    Rgrid,phigrid,zgrid = effsel_grid
    
    # Determine the isochrone mass fraction factors for each field
    print('Calculating isochrone factors')
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
    print(np.unique(isofactors))

    # Set up R,phi,z grid for integration
    if mass_int_type == 'cartesian_grid':
        print('Cartesian grid integration not supported, using spherical grid')
        mass_int_type = 'spherical_grid'
    if mass_int_type == 'spherical_grid':
        r_min,r_max = int_r_range
        n_edge_r,n_edge_theta,n_edge_phi = n_edge
        Rphizgrid,deltafactor = spherical_integration_grid(r_min,r_max,n_edge_r,
                                                     n_edge_theta,n_edge_phi)

    # Calculate the mass
    print('Calculating mass')
    
    np.random.seed(seed)
    samples_randind = samples[np.random.choice(len(samples),n_mass,
                                               replace=False)]
    # Parallel?
    if nprocs and nprocs > 1:
        # Batch computing?
        if batch and (samples_randind.shape[0]%nprocs==0):
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
            return masses, facs  
        # No batch computing
        else:
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
            return masses, facs
    # Serial        
    else:
        print('Processing masses in serial')
        for i,params in enumerate(samples_randind):
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
            if densfunc is pdens.triaxial_single_angle_zvecpa_plusexpdisk or\
               densfunc is pdens.triaxial_single_cutoff_zvecpa_plusexpdisk: 
                denstxyz = densfunc(Rphizgrid[:,0],Rphizgrid[:,1],
                                    Rphizgrid[:,2], params=params, 
                                    split=True)
                halodens = denstxyz[0]*fac
                diskdens = denstxyz[1]*fac
                fulldens = densfunc(Rphizgrid[:,0],Rphizgrid[:,1],
                                    Rphizgrid[:,2], params=params)*fac
                masses[i] = np.sum(halodens*deltafactor),\
                            np.sum(diskdens*deltafactor),\
                            np.sum(fulldens*deltafactor)
            else:
                denstxyz = densfunc(Rphizgrid[:,0],Rphizgrid[:,1],
                                    Rphizgrid[:,2], params=params)*fac
                masses[i] =  np.sum(denstxyz*deltafactor)
            facs[i] = fac
            # Also maybe some sort of actual integrator?
        return masses, facs

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
    if densfunc is pdens.triaxial_single_angle_zvecpa_plusexpdisk or\
       densfunc is pdens.triaxial_single_cutoff_zvecpa_plusexpdisk: 
        denstxyz = densfunc(Rphizgrid[:,0],Rphizgrid[:,1],Rphizgrid[:,2], 
                            params=params, split=True)
        halodens = denstxyz[0]*fac
        diskdens = denstxyz[1]*fac
        fulldens = densfunc(Rphizgrid[:,0],Rphizgrid[:,1],Rphizgrid[:,2], 
                         params=params)*fac
        mass = np.sum(halodens*deltafactor),\
                     np.sum(diskdens*deltafactor),\
                     np.sum(fulldens*deltafactor)
    else:
        denstxyz = densfunc(Rphizgrid[:,0],Rphizgrid[:,1],
                            Rphizgrid[:,2], params=params)*fac
        mass =  np.sum(denstxyz*deltafactor)
    # Increment counter
    global counter
    with counter.get_lock():
        counter.value += 1
    if counter.value%10 == 0: print('sampled '+str(counter.value+1)+'/'+str(n_mass), 
                                    end='\r')
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
        mass (float) - Mass in Msun
        fac (float) - factor to convert from normalized density to Msun/
            pc**3
    '''
    # Unpack
    Rgrid,phigrid,zgrid = effsel_grid
    mass = np.zeros(params.shape[0])
    facs = np.zeros(params.shape[0])
    for i in range(params.shape[0]):
        # Note effsel must have area factors and Jacobians applied!
        rate = densfunc(Rgrid,phigrid,zgrid,params=params[i])*effsel
        sumrate = np.sum(rate.T/isofactors)
        facs[i] = n_star/sumrate
        if densfunc is pdens.triaxial_single_angle_zvecpa_plusexpdisk or\
           densfunc is pdens.triaxial_single_cutoff_zvecpa_plusexpdisk: 
            denstxyz = densfunc(Rphizgrid[:,0],Rphizgrid[:,1],Rphizgrid[:,2], 
                                params=params[i], split=True)
            halodens = denstxyz[0]*facs[i]
            diskdens = denstxyz[1]*facs[i]
            fulldens = densfunc(Rphizgrid[:,0],Rphizgrid[:,1],Rphizgrid[:,2], 
                             params=params[i])*facs[i]
            mass[i] = np.sum(halodens*deltafactor),\
                         np.sum(diskdens*deltafactor),\
                         np.sum(fulldens*deltafactor)
        else:
            denstxyz = densfunc(Rphizgrid[:,0],Rphizgrid[:,1],
                                Rphizgrid[:,2], params=params[i])*facs[i]
            mass[i] =  np.sum(denstxyz*deltafactor)
        # Increment counter
        global counter
        with counter.get_lock():
            counter.value += 1
        if counter.value%10 == 0: print('sampled '+str(counter.value+1)+'/'+str(n_mass), 
                                        end='\r')
    return mass,facs

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
    alpha_positive = False
    if densfunc is pdens.spherical:
        # alpha
        if params[0] < 0. and alpha_positive:return False
        return True
    if densfunc is pdens.axisymmetric:
        # alpha, q
        if params[0] < 0. and alpha_positive:return False
        if params[1] < 0.1:return False
        if params[1] > 1.:return False
        return True
    if densfunc is pdens.triaxial_norot:
        # alpha, p, q
        if params[0] < 0. and alpha_positive:return False
        if params[1] < 0.1:return False
        if params[1] > 1.:return False
        if params[2] < 0.1:return False
        if params[2] > 1.:return False
        return True
    if densfunc is pdens.triaxial_single_angle_aby:
        # alpha, p, q, A, B, Y
        if params[0] < 0. and alpha_positive:return False
        if params[1] < 0.1:return False
        if params[1] > 10.:return False
        if params[2] < 0.1:return False
        if params[2] > 10.:return False
        if params[3] < 0.:return False
        if params[3] > 1.:return False
        if params[4] < 0.:return False
        if params[4] > 1.:return False
        if params[5] < 0.:return False
        if params[5] > 1.:return False
        return True
    if densfunc is pdens.triaxial_single_angle_zvecpa:
        # alpha, p, q, theta, eta, phi
        if params[0] < 0. and alpha_positive:return False
        if params[1] < 0.1:return False
        if params[1] > 1.:return False
        if params[2] < 0.1:return False
        if params[2] > 1.:return False
        if params[3] < 0.:return False
        if params[3] > 1.:return False
        if params[4] < 0.:return False
        if params[4] > 1.:return False
        if params[5] < 0.:return False
        if params[5] > 1.:return False
        return True
    if densfunc is pdens.triaxial_single_cutoff_zvecpa:
        # alpha, beta, p, q, theta, eta, phi
        if params[0] < 0. and alpha_positive:return False
        if params[1] < 0.:return False
        if params[2] < 0.1:return False
        if params[2] > 1.:return False
        if params[3] < 0.1:return False
        if params[3] > 1.:return False
        if params[4] <= 0.:return False
        if params[4] >= 1.:return False
        if params[5] <= 0.:return False
        if params[5] >= 1.:return False
        if params[6] <= 0.:return False
        if params[6] >= 1.:return False
        return True
    if densfunc is pdens.triaxial_broken_angle_zvecpa:
        if params[0] < 0. and alpha_positive:return False
        if params[1] < 0.:return False
        if params[2] < 0.:return False
        if params[3] < 0.1:return False
        if params[3] > 1.:return False
        if params[4] < 0.1:return False
        if params[4] > 1.:return False
        if params[5] <= 0.:return False
        if params[5] >= 1.:return False
        if params[6] <= 0.:return False
        if params[6] >= 1.:return False
        if params[7] <= 0.:return False
        if params[7] >= 1.:return False
        return True
    if densfunc is pdens.triaxial_single_angle_zvecpa_plusexpdisk:
        # alpha, p, q, theta, eta, phi, fdisc
        if params[0] < 0. and alpha_positive:return False
        if params[1] < 0.1:return False
        if params[1] > 1.:return False
        if params[2] < 0.1:return False
        if params[2] > 1.:return False
        if params[3] <= 0.:return False
        if params[3] >= 1.:return False
        if params[4] <= 0.:return False
        if params[4] >= 1.:return False
        if params[5] <= 0.:return False
        if params[5] >= 1.:return False
        if params[6] < 0.:return False
        if params[6] > 1.:return False
        return True
    if densfunc is pdens.triaxial_single_cutoff_zvecpa_plusexpdisk:
        # alpha, beta, p, q, theta, eta, phi, fdisc
        if params[0] < 0. and alpha_positive:return False
        if params[1] < 0.:return False
        if params[2] < 0.1:return False
        if params[2] > 1.:return False
        if params[3] < 0.1:return False
        if params[3] > 1.:return False
        if params[4] <= 0.:return False
        if params[4] >= 1.:return False
        if params[5] <= 0.:return False
        if params[5] >= 1.:return False
        if params[6] <= 0.:return False
        if params[6] >= 1.:return False
        if params[7] < 0.:return False
        if params[7] > 1.:return False
        return True
    if densfunc is pdens.triaxial_broken_angle_zvecpa_plusexpdisk:
        if params[0] < 0. and alpha_positive:return False
        if params[1] < 0.:return False
        if params[2] < 0.:return False
        if params[3] < 0.1:return False
        if params[3] > 1.:return False
        if params[4] < 0.1:return False
        if params[4] > 1.:return False
        if params[5] <= 0.:return False
        if params[5] >= 1.:return False
        if params[6] <= 0.:return False
        if params[6] >= 1.:return False
        if params[7] <= 0.:return False
        if params[7] >= 1.:return False
        if params[8] < 0.:return False
        if params[8] > 1.:return False
        return True
#     if densfunc is pdens.triaxial_einasto_zvecpa:
#         if params[0] < 0. and alpha_positive:return False
#         if params[1] < 0.5:return False
#         if params[2] < 0.1:return False
#         if params[2] > 1.:return False
#         if params[3] < 0.1:return False
#         if params[3] > 1.:return False
#         if params[4] <= 0.:return False
#         if params[4] >= 1.:return False
#         if params[5] <= 0.:return False
#         if params[5] >= 1.:return False
#         if params[6] <= 0.:return False
#         if params[6] >= 1.:return False
#         if params[7] <= 0.:return False
#         if params[7] >= 1.:return False
#         return True
#     if densfunc is pdens.triaxial_einasto_zvecpa_plusexpdisk:
#         if params[0] < 0. and alpha_positive:return False
#         if params[1] < 0.5:return False
#         if params[2] < 0.1:return False
#         if params[2] > 1.:return False
#         if params[3] < 0.1:return False
#         if params[3] > 1.:return False
#         if params[4] <= 0.:return False
#         if params[4] >= 1.:return False
#         if params[5] <= 0.:return False
#         if params[5] >= 1.:return False
#         if params[6] <= 0.:return False
#         if params[6] >= 1.:return False
#         if params[7] <= 0.:return False
#         if params[7] >= 1.:return False
#         if params[8] <= 0.:return False
#         if params[8] >= 1.:return False
        return True
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

def pdistmod_sample(densfunc, samples, n_samples, effsel, Rgrid, phigrid, zgrid,
                    distmods, return_rate=False):
    '''pdistmod_sample:
    
    Return the expected distance modulus distribution for a set of models 
    described by distribution of parameters, including the effective 
    selection function. Assume the effective selection function already 
    has the distance modulus Jacobian applied (only the factor of d**3 which 
    matters).
    
    Args:
        densfunc (function) - Density profile
        samples (array) - (Nsamples,params) shaped array to draw parameters from
        n_samples (int) - Number of times to draw parameters from samples
        effsel (array) - Effective selection function (Nfield x Ndistmod)
        Rgrid (array) - Grid of R corresponding to effsel
        phigrid (array) - Grid of phi corresponding to effsel
        zgrid (array) - Grid of z corresponding to effsel
        distmods (array) - Grid of distance moduli
        return_rate (bool) - Return the rate function as well.
    
    Returns:
        pd (array) - Normalized number of counts summed over all fields
        pdt (array) - Number of counts summed over all fields
        rate (array) - Raw number of counts per field, only if return_rate=True
    '''
    pd = np.zeros((n_samples,effsel.shape[1]))
    pdt = np.zeros((n_samples,effsel.shape[1]))
    rate = np.zeros((n_samples,effsel.shape[0],effsel.shape[1]))
    sample_randind = np.random.choice(len(samples),n_samples,replace=False)
    for i,params in enumerate(samples[sample_randind]):
        _pd,_pdt,_r = pdistmod_one_model(densfunc, params, effsel, Rgrid, 
                                         phigrid, zgrid, distmods,
                                         return_rate=True)
        pd[i,:] = _pd
        pdt[i,:] = _pdt
        rate[i,:,:] = _r
    if return_rate:
        return pd, pdt, rate
    else:
        return pd, pdt    

# def pdistmod_check_fit(densfunc, samp, effsel, Rgrid, phigrid, zgrid, ds, 
#                        goodindx, sample=False):
#     '''pdistmod_check_fit:
    
#     Determine the distance modulus distribution for a given model
    
#     Args:
#         densfunc (function) - Density profile
#         samp (array) - Fitted set of parameters for the density model
#         effsel (array) - Effective selection function (Nfield x Ndistmod)
#         Rgrid (array) - Grid of R corresponding to effsel
#         phigrid (array) - Grid of phi corresponding to effsel
#         zgrid (array) - Grid of z corresponding to effsel
#         ds (array) - Grid of distances corresponding to the distance modulus 
#             grid
#         goodindx (array) - Usable fields in the effective selection function
#         sample (bool) - Sample based on a range of model parameters
    
#     Returns:
        
#     '''
#     pds = np.empty((200,len(ds)))
#     if sample:
#         for ii,params in tqdm.tqdm_notebook(enumerate(samp[np.random.randint(len(samp), size=200)]), total=200):
#             pd, pdt,rate = pdistmod_model(densfunc, params, effsel, Rgrid, 
#                 phigrid, zgrid, ds, goodindx, returnrate=True)
#             pds[ii] = pd
#         return pds
#     else:
#         pd, pdt, rate = pdistmod_model(densfunc, np.median(samp,axis=0), effsel, 
#             Rgrid, phigrid, zgrid, ds, goodindx, returnrate=True)
#         return pd
#     ##ie
# #def

    
# def fit_bin_mask(mask, effsel, goodindx, fehrange, 
#                  model=pdens.triaxial_single_angle_aby, 
#                  just_MLE=True, just_MCMC=False, mass=False, 
#                  init=[2.,0.5,0.5,0.8,1/10.,0.5,0.5,0.5], ncut=40, 
#                  mass_analytic=False, inttype='spherical'):
#     '''fit_bin_mask:
    
#     Fit a density profile to data in a bin defined by a mask. Can use either 
#     maximum likelihood, MCMC, or both.
    
#     Args:
#         mask (bool array)
#         effsel (array) - Effective selection function (Nfield x Ndistmod)
#         goodindx (array) - Usable fields in the effective selection function
#         fehrange (list) - The range in [Fe/H] defined by the mask
        
#         model (function) - Density model function
#         just_MLE
#         just_MCMC
#         mass
#         init
#         ncut
#         mass_analytic (bool) - Compute the density profile normalizing mass 
#             analytically. Only works for spherical density profiles.
#         mass_int_type (str) - Coordinate basis for integration to calculate the 
#             normalizing mass of the density profile. Can be either
#             'spherical' or 'cartesian'.
        
    
#     Returns:
        
#     '''
    
    
#     '''Fits the stars defined by a mask to the APOGEE low metallicity sample
#     IN:
#     mask - must be same length as gaia2_matches[omask], and must have Fe/H limits that go into fehrange.
#     fehrange - the range in Fe/H spanned by the sample defined by mask
#     effsel - the effective selection function corresponding to the sample in mask
#     model - the density model to be fit
#     just_MLE - do the Maximum Likelihood and return opt
#     just_MCMC - do the MCMC
#     mass - also compute the total mass
#     init - initial input parameters for the density model
#     ncut - number of samples to cut from each MCMC chain
#     analytic - compute the mass integral analytically (only works for spherical density models)
#     inttype - the coordinate scheme for the integration grid for mass computation
#     OUTPUT:
#     opt - opt from op.minimize
#     samples - MCMC samples
#     masses - MC samples of mass
#     facs - the normalisation factor corresponding to each mass sample
#     '''
#     #needs goodindx defined above. 'mask' must be same length as gaia2_matches[omask].
#     #print(sum(mask))
    
#     # If we're not just doing MCMC then find the MLE starting point
#     if not just_MCMC:
#         # do MLE. 
#         opt = op.fmin(lambda x: putil.mloglike(x,model, 
#             effsel[goodindx]*ds**3.*np.log(10)/5.*(distmods[1]-distmods[0]), 
#             Rgrid[goodindx], phigrid[goodindx], zgrid[goodindx], 
#             mrpz[:,0][mask]*8., mrpz[:,1][mask], mrpz[:,2][mask]*8.), 
#             init, full_output=True)
#         print(opt[0])
#         if just_MLE:
#             return opt
#         ##fi
#     ##fi
    
#     # do MCMC initialised from best result from MLE or from init params given
#     ndim, nwalkers = len(init), 200
#     if just_MCMC:
#         pos = [init + 1e-3*np.random.randn(ndim) for i in range(nwalkers)]
#     else:
#         pos = [opt[0] + 1e-3*np.random.randn(ndim) for i in range(nwalkers)]
#     nit = 200
#     threads = 4
#     #only effsel for good fields
#     effsel_in = effsel[goodindx]*ds**3*np.log(10)/5.*(distmods[1]-distmods[0])
#     #re-build Rphiz grid
#     Rgrid_in, phigrid_in, zgrid_in = Rgrid[goodindx], phigrid[goodindx], zgrid[goodindx]
    
#     #set up sampler and do the sampling.
#     with multiprocessing.Pool(nprocs) as pool:
#         sampler = emcee.EnsembleSampler(nwalkers, ndim, util.loglike, 
#             args=(model, effsel_in, Rgrid_in, phigrid_in, zgrid_in, 
#             mrpz[:,0][mask]*8., mrpz[:,1][mask], mrpz[:,2][mask]*8.), 
#             pool=pool)
#         print('Generating MCMC samples...')
#         for i, result in enumerate(sampler.sample(pos, iterations=nit)):
#             if (i+1)%10 == 0: print('sampled '+str(i+1)+'/'+str(nit))
#             continue
#         ###i
#         #cut ncut samples from each chain
#         samples = sampler.chain[:, ncut:, :].reshape((-1, ndim))
#     ##wi
    
#     # Return if not determining mass
#     if not mass:
#         if just_MCMC:
#             return samples
#         else:
#             return opt, samples
#         ##ie
#     ##fi
    
#     # Determine the isochrone mass fraction factors
#     isofactors = np.zeros(len(effsel[goodindx]))
#     for i in range(len(isofactors)):   
#         isomask = (Z2FEH(isorec['Zini']) > fehrange[0]) & (Z2FEH(isorec['Zini']) < fehrange[1]) & (isorec['Jmag']-isorec['Ksmag'] > jkmins[goodindx][i]) & (isorec['logg'] < 3) & (isorec['logg'] > 1) & (isorec['logAge'] > 10)
#         avmass = util.average_mass(isorec[isomask], lowfehgrid=True)
#         isomask = (Z2FEH(isorec['Zini']) > fehrange[0]) & (Z2FEH(isorec['Zini']) < fehrange[1]) & (isorec['logAge'] > 10)  #& (isorec['J']-isorec['K'] > 0.3) & (isorec['logg'] < 3) & (isorec['logg'] > 1)
#         massratio = util.mass_ratio(isorec[isomask], lowfehgrid=True, minjk=jkmins[goodindx][i])
#         isofactors[i] = avmass/massratio
#     print(np.unique(isofactors))
    
#     #set up grid for integration
#     if mass_int_type == 'spherical':
#         rthetaphigrid = np.mgrid[2.:70:150j,0:np.pi:150j,0:2*np.pi:150j]
#         dr = (70-2.)/149
#         dtheta = (np.pi-0.)/149
#         dphi = (2*np.pi-0.)/149
#         shape = np.shape(rthetaphigrid.T)
#         rthetaphigrid = rthetaphigrid.T.reshape(np.product(shape[:3]),shape[3])
#         deltafactor = rthetaphigrid[:,0]**2*np.sin(rthetaphigrid[:,1])*dr*dtheta*dphi
#         x = rthetaphigrid[:,0]*np.sin(rthetaphigrid[:,1])*np.cos(rthetaphigrid[:,2])
#         y = rthetaphigrid[:,0]*np.sin(rthetaphigrid[:,1])*np.sin(rthetaphigrid[:,2])
#         z = rthetaphigrid[:,0]*np.cos(rthetaphigrid[:,1])
#         xyzgrid = np.dstack([x,y,z])[0]
#         rphizgrid = coords.rect_to_cyl(xyzgrid[:,0], xyzgrid[:,1], xyzgrid[:,2])
#         rphizgrid = np.dstack([rphizgrid[0],rphizgrid[1],rphizgrid[2]])[0]
#     if mass_int_type == 'cartesian':
#         xyzgrid = np.mgrid[-50.:50.:150j,-50.:50.:150j,-50.:50.:150j]
#         delta = xyzgrid[0,:,0,0][1]-xyzgrid[0,:,0,0][0]
#         deltafactor = delta**3
#         shape = np.shape(xyzgrid.T)
#         xyzgrid = xyzgrid.T.reshape(np.product(shape[:3]),shape[3])
#         rphizgrid = coords.rect_to_cyl(xyzgrid[:,0], xyzgrid[:,1], xyzgrid[:,2])
#         rphizgrid = np.dstack([rphizgrid[0],rphizgrid[1],rphizgrid[2]])[0]
#     if model is pdens.triaxial_single_angle_zvecpa_plusexpdisk or model is pdens.triaxial_einasto_zvecpa_plusexpdisk or model is pdens.triaxial_broken_angle_zvecpa_plusexpdisk or model is pdens.triaxial_single_cutoff_zvecpa_plusexpdisk:
#         masses = np.zeros((400,3))
#     else:
#         masses = np.zeros(400)
#     facs = np.zeros(400)
    
#     # Calculate the mass
#     print('Calculating mass')
#     for ii,params in enumerate(samples[np.random.choice(len(samples), 400, replace=False)]):
#         rate=model(Rgrid[goodindx],phigrid[goodindx],zgrid[goodindx],params=params)*effsel[goodindx]*ds**3*np.log(10)/5.*(distmods[1]-distmods[0])
#         sumrate = np.sum(rate.T/isofactors)
#         norm = sum(mask)/sumrate
#         fac = norm*(180./np.pi)**2
#         if mass_analytic:
#             #only for spherical power law!
#             rsun = np.sqrt(8.**2+0.02**2)
#             min_r = 2.
#             max_r = 70.
#             alpha = params[0]
#             integral = 4*np.pi*((rsun**alpha*max_r**(3-alpha))/(3-alpha)-(rsun**alpha*min_r**(3-alpha))/(3-alpha))
#             masses[ii] = integral*fac
#         else:
#             if model is pdens.triaxial_single_angle_zvecpa_plusexpdisk or model is pdens.triaxial_einasto_zvecpa_plusexpdisk or model is pdens.triaxial_broken_angle_zvecpa_plusexpdisk or model is pdens.triaxial_single_cutoff_zvecpa_plusexpdisk:
#                 denstxyz = model(rphizgrid[:,0],rphizgrid[:,1],rphizgrid[:,2], params=params, split=True)
#                 halodens = denstxyz[0]*fac
#                 diskdens = denstxyz[1]*fac
#                 fulldens = model(rphizgrid[:,0],rphizgrid[:,1],rphizgrid[:,2], params=params)*fac
#                 masses[ii] = np.sum(halodens*deltafactor), np.sum(diskdens*deltafactor), np.sum(fulldens*deltafactor)
#             else:
#                 denstxyz = model(rphizgrid[:,0],rphizgrid[:,1],rphizgrid[:,2], params=params)*fac
#                 masses[ii] =  np.sum(denstxyz*deltafactor)
#         #densfunc = lambda r,phi,z: r*model(r,phi,z,params=params)
#         #integral = nquad(densfunc, [[3.,50.],[0.,2*np.pi],[-30.,30.]])
#         #masses[ii] = integral[0]*fac
#         facs[ii] = fac
#     ###i
    
#     if just_MCMC:
#         return samples, masses, facs
#     return opt, samples, masses, facs
# #def

### Defunct

# def join_on_id(dat1,dat2,joinfield='APOGEE_ID'):
#     '''join_on_id:
# 
#     Takes two recarrays and joins them based on a ID string. Only overlapping 
#     entries will be present in the output.
# 
#     Args:
#         dat1 (numpy.recarray) - First set of data
#         dat2 (numpy.recarray) - Second set of data
#         joinfield (string) - Field that joins the data
# 
#     Returns:
#         dat1 (numpy.recarray) - Joined data
#     '''
#     #find common fields
#     names1 = [dat1.dtype.descr[i][0] for i in range(len(dat1.dtype.descr))]
#     names2 = [dat2.dtype.descr[i][0] for i in range(len(dat2.dtype.descr))]
#     namesint = np.intersect1d(names1,names2)
#     if joinfield not in namesint:
#         return NameError('Field '+joinfield+' is not present in both arrays.')
#     #work out which fields get appended from dat2
#     descr2 = dat2.dtype.descr
#     fields_to_append = []
#     names_to_append = []
#     for i in range(len(names2)):
#         if names2[i] not in namesint:
#             fields_to_append.append(descr2[i])
#             names_to_append.append(names2[i])
#         else:
#             continue   
#     newdtype= dat1.dtype.descr+fields_to_append
#     newdata= np.empty(len(dat1),dtype=newdtype)
#     for name in dat1.dtype.names:
#         newdata[name]= dat1[name]
#     for f in names_to_append:
#         newdata[f]= np.zeros(len(dat1))-9999.
#     dat1= newdata
# 
#     hash1= dict(zip(dat1[joinfield],
#                     np.arange(len(dat1))))
#     hash2= dict(zip(dat2[joinfield],
#                     np.arange(len(dat2))))
#     common, indx1, indx2 = np.intersect1d(dat1[joinfield],dat2[joinfield],return_indices=True)
#     for f in names_to_append:
#         dat1[f][indx1]= dat2[f][indx2]
#     return dat1
# #def