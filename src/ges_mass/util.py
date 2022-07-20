# ----------------------------------------------------------------------------
#
# TITLE - util.py
# AUTHOR - James Lane
# PROJECT - ges-mass
# CONTENTS:
#
# ----------------------------------------------------------------------------

### Docstrings and metadata:
'''Utilities and other misc functions. Includes config file loading, 
'''
__author__ = "James Lane"

### Imports
import numpy as np
import os
import copy
from astropy import units as apu
from astropy import coordinates
from galpy import orbit
import pdb

# ----------------------------------------------------------------------------

# Config file loading and parsing

def load_config_to_dict(fname='config.txt'):
    '''load_config_to_dict:
    
    Load a config file and convert to dictionary. Will search for configuration 
    file matching fname recursively upwards through the directory structure. 
    Config file takes the form:
    
    KEYWORD1 = VALUE1 # comment
    KEYWORD2 = VALUE2
    etc..
    
    = sign must separate keywords from values. Trailing # indicates comment
    
    Args:
        fname (str) - Name of the configuration file to search for 
            ['config.txt']
        
    Returns:
        cdict (dict) - Dictionary of config keyword-value pairs
    '''
    cdict = {}
    fname_path = _find_config_file(fname)
    print('Loading config file from '+os.path.realpath(fname_path))
    with open(fname_path,'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.split('#')[0].strip() == '': continue # Empty line
            assert '=' in line, 'Keyword-Value pairs must be separated by "="'
            # Remove comments and split at =
            line_vals = line.split('#')[0].strip().split('=') 
            cdict[line_vals[0].strip().upper()] = line_vals[1].strip()
    return cdict

def _find_config_file(fname):
    '''_find_config_file:
    
    Recursively find a config.txt file search upwards through the directory
    structure
    
    Args:
        fname (str) - Name of the configuration file to search for
    
    Returns:
        config_path (str) - Path to the configuration file
    '''
    config_dir = ''
    while True:
        if os.path.realpath(config_dir).split('/')[-1] == 'ges-mass':
            raise FileNotFoundError('Could not find configuration file within'+
                                    ' project directory structure')
        if os.path.realpath(config_dir) == '/':
            raise RuntimeError('Reached base directory')
        if os.path.exists(config_dir+fname):
            return config_dir+fname
        config_dir = config_dir+'../'

def parse_config_dict(cdict,keyword):
    '''parse_config_dict:
    
    Parse config dictionary for keyword-value pairs. Valid keywords are:
        RO (float) - galpy distance scale
        VO (float) - galpy velocity scale
        ZO (float) - galpy vertical solar position
        HOME_DIR (string) - Project directory (not used)
        BASE_DIR (string) - Data directory
        APOGEE_DR (string) - APOGEE data release of the form dr#
        APOGEE_RESULTS_VERS (string) - APOGEE results version for data release 
        GAIA_DR (string) - Gaia data release of the form dr# or edr#
        NDMOD (int) - Number of distance moduli in effective survey selection
            function grid
        DMOD_MIN (float) - Minimum distance modulus in effective survey
            selection function grid
        DMOD_MAX (float) - Maximum distance modulus in effective survey
            selection function grid
        LOGG_MIN (float) - Minimum log(g) for fitting sample
        LOGG_MAX (float) - Maximum log(g) for fitting sample
        FEH_MIN (float) - Minimum [Fe/H] for fitting sample
        FEH_MAX (float) - Maximum [Fe/H] for fitting sample
        NPROCS (int) - Number of processors to use when multiprocessing
        
    Args:
        cdict (dict) - Dictionary of keyword-value pairs
        keyword (str or arr) - Keyword to extract, or array of keywords

    Returns:
        value (variable) - Value or result of the keyword
    '''
    if isinstance(keyword,(list,tuple,np.ndarray)): # many keywords
        _islist = True
        _keyword = []
        _value = []
        for key in keyword:
            assert key.upper() in cdict, 'Keyword '+key.upper()+' not in cdict'
            _keyword.append(key.upper())
    else: # Assume string, just one keyword
        _islist = False
        _keyword = [keyword.upper(),]
        assert _keyword[0] in cdict, 'Keyword '+_keyword[0]+' not in cdict'
    
    float_kws =  ['RO','VO','ZO','DMOD_MIN','DMOD_MAX','LOGG_MIN','LOGG_MAX',
                  'FEH_MIN','FEH_MAX','FEH_MIN_GSE','FEH_MAX_GSE']
    int_kws =    ['NPROCS','NDMOD']
    string_kws = ['HOME_DIR','BASE_DIR','APOGEE_DR','APOGEE_RESULTS_VERS',
                  'GAIA_DR']
    
    for key in _keyword:
        if key in float_kws:
            if _islist:
                _value.append( float(cdict[key]) )
            else:
                return float(cdict[key]) 
            ##ie    
        elif key in int_kws:
            if _islist:
                _value.append( int(cdict[key]) )
            else:
                return int(cdict[key])
            ##ie
        elif key in string_kws:
            if _islist:
                _value.append( cdict[key] )
            else:
                return cdict[key]
            ##ie
        ##ei
        # No code, just pass value
        else:
            print('Warning: keyword '+key+' has no parsing code,'+
            ' just passing value')
            if _islist:
                _value.append( cdict[key] )
            else:
                return cdict[key]
    # Assume single key has returned already
    return _value
#def

# ----------------------------------------------------------------------------

def make_dmod_grid(n,dmod_min,dmod_max):
    '''make_dmod_grid:
    
    Make the grid of distance moduli on which the effective survey selection 
    function is calculated
    
    Args:
        ndm (int) - Number of points in the grid
        dm_min (float) - maximum distance modulus
        dm_max (float) - minimum distance modulus
    
    Returns:
        dmod (array) - Distance modulus
        dist (array) - Distance in kpc
    '''
    dmod = np.linspace(dmod_min, dmod_max, n)
    dist = 10.**(dmod/5-2)
    return dmod,dist

def parse_mixture(arr, mixture_arr, seed=None, absolute=True, return_inds=False):
    '''parse_mixture:
    
    Parses an array that denotes fractional numbers of arr elements to be 
    returned as well as a list of objects and returns a list of objects 
    corresponding to the mixture. Note that the first argument can be any 
    array of objects, such as orbit.Orbit or np.ndarray as long as they have
    the same length (np.ndarray can have extra dimensions).
    
    In practice, the idea is to take many objects that have the 
    same number of things in them, then generate a new list of objects which 
    has the same length as each of the original objects, but is a mixing of the 
    incoming array.
    
    if you had a list of objects, each of which 
    have length 1000:
    
    [obj1,obj2]
    
    and a mixture_arr that looked like:
    
    [0.4,0.6]
    
    Then 400 elements are taken from obj1 and 600 from obj2 and a list of 
    those objects is returned:
    
    [obj1_mixture,obj2_mixture]
    
    Note that if absolute=False, actual numbers are normalized, so a mixture 
    arr which is [4,6] Does the same thing as one which is [0.4,0.6]
    
    Args:
        arr (list) - List of galpy.orbit.Orbit or np.ndarray objects. For 
            np.ndarray the shape should be (M,N) where M is number of quantities
            (e.g. e,E,Lz gives M=3) and N is the number of stars.
        mixture_arr (np.ndarray) - Array of numbers corresponding to the 
            fractional amount of each element of arr that should be returned
        seed (int) - Seed to use for sampling
        absolute (bool) - The fractions given in mixture_arr are absolute, 
            not relative. If True no element of absolute may be greater than 
            1.0 [True]
    
    Returns:
        arr_mixture (list) - List of objects with fractional 
            amounts corresponding to mixture_arr
    '''
    if not absolute:
        warnings.warn('Warning, absolute=False. Not using absolute mixture array fractions!')
    assert len(mixture_arr) == len(arr)
    assert isinstance(mixture_arr,np.ndarray)
    
    # Normalization
    norm = np.sum(mixture_arr)
    
    arr_mixture = []
    sample_inds_mixture = []
    
    # Loop over the length of mixture_arr
    for i in range(len(mixture_arr)):
        
        if mixture_arr[i] == 0: # Skip arr[i]
            continue
        elif mixture_arr[i] == norm and not absolute: # All the elements come from arr[i]
            return [arr[i],]
        else:
            if absolute:
                orb_frac = mixture_arr[i]
                assert orb_frac <= 1., 'absolute fractions cannot exceed 1'
                if orb_frac == 1.: # Take all elements of arr[i]
                    arr_mixture.append(arr[i])
                    continue
            else:
                orb_frac = mixture_arr[i]/norm
            if isinstance(arr[i],np.ndarray):
                n_arr = np.atleast_2d(arr[i]).shape[1]
                _isNumpy = True
            else:
                _isNumpy = False
            
            n_samples = int(n_arr*orb_frac)
            if seed is not None:
                np.random.seed(seed)
            sample_inds = np.random.choice(n_arr,n_samples,replace=False)
            if _isNumpy:
                arr_mixture.append(arr[i][:,sample_inds]) 
            else:
                arr_mixture.append(arr[i][sample_inds])
            ##ie
            sample_inds_mixture.append(sample_inds)
        ##ie
    ###i
    if return_inds:
        return arr_mixture, sample_inds_mixture
    else:
        return arr_mixture
    ##ie
#def

def find_orbit_nearest_neighbor(orbs1,orbs2,ro=8,vo=220):
    '''find_orbit_nearest_neighbor:
    
    Find the nearest neighbor matching between two sets 
    of orbit.Orbit objects. Finds the closest match for each 
    orbit in orbs1 from orbs2.
    
    So orbs2[match_index] are the nearest neighbors 
    
    Args
        orbs1 (orbit.Orbit) = orbits to find nearest neighbors
        orbs2 (orbit.Orbit) = orbits from which nearest neighbors will be find
    
    Returns:
        match_indx (array) - Match index for 
        dist
    '''
    orbs1_copy = copy.deepcopy(orbs1)
    orbs2_copy = copy.deepcopy(orbs2)
    orbs1_copy.turn_physical_on(ro=ro,vo=vo)
    orbs2_copy.turn_physical_on(ro=ro,vo=vo)
    
    # If orbs1 has only 1 element make this into an array
    if len(orbs1_copy) == 1:
        sc1 = coordinates.SkyCoord(l=np.array([orbs1_copy.ll().value])*apu.deg, 
                                   b=np.array([orbs1_copy.bb().value])*apu.deg, 
                                   distance=np.array([orbs1_copy.dist().value])*apu.kpc, 
                                   frame='galactic')
    else:
        sc1 = coordinates.SkyCoord(l=orbs1_copy.ll(), b=orbs1_copy.bb(), distance=orbs1_copy.dist(), 
                                   frame='galactic')
    ##ie
    
    sc2 = coordinates.SkyCoord(l=orbs2_copy.ll(), b=orbs2_copy.bb(), distance=orbs2_copy.dist(), 
                               frame='galactic')
    
    match_indx, sep2d, dist3d = sc1.match_to_catalog_3d(sc2,nthneighbor=1)
    
    return match_indx, sep2d, dist3d
#def

def perturb_orbit_with_Gaia_APOGEE_uncertainties(orbs,gaia,allstar,
                                                 only_velocities=True,
                                                 ro=8.,vo=220,zo=0):
    '''perturb_orbit_with_Gaia_APOGEE_uncertainties:
    
    Perturb an orbit object by the uncertainties from Gaia and APOGEE data
    
    Note that correlations between distance and anything is zero.
    
    Gaia data must be queried by:
        'ra_error'
        'dec_error'
        'pmra_error'
        'pmdec_error'
        'ra_dec_corr'
        'ra_pmra_corr'
        'ra_pmdec_corr'
        'dec_pmra_corr'
        'dec_pmdec_corr'
        'pmra_pmdec_corr'
    
    APOGEE data must be queried by:
        'VERR'
        'weighted_dist_error'
    
    Args
        orbs (orbit.Orbit) - original orbit(s)
        gaia (array) - Gaia data
        allstar (array) - APOGEE data
        only_velocities (bool) - Only perturb velocities, not positions
        ro,vo,zo (floats) - galpy unit scales
    
    Returns
        orbs_sample (orbit.Orbit) - New orbit(s) perturbed by uncertainties
    '''
    # If either gaia or allstar are numpy.void the array manipulation won't work 
    # correctly
    assert not isinstance(gaia,np.void) and not isinstance(allstar,np.void),\
        '''either gaia or allstar are np.void, array manipulation will not work properly. 
           Try indexing the arrays as gaia[[0,]] instead of gaia[0] for example'''
    assert ( len(gaia) == len(allstar) and len(orbs) == len(gaia) ) or\
           ( len(gaia) == 1 and len(allstar) == 1 and len(orbs) > 1),\
        'either gaia, allstar, and orbs are same length, or gaia and allstar are 1D, and orbs is >1D'
    
    n_samples = len(orbs)
    n_errors = len(gaia)
    
    ra_sample,dec_sample,pmra_sample,pmdec_sample,dist_sample,rv_sample = np.zeros((6,n_samples))
    
    # Positions / velocities from the DF-sampled orbit
    ra,dec = orbs.ra().value, orbs.dec().value, 
    pmra,pmdec = orbs.pmra().value, orbs.pmdec().value
    dist,rv = orbs.dist().value, orbs.vlos().value 
    
    # Uncertainties. Note 'weighted_dist_error' is in pc (want kpc)
    ra_e,dec_e = gaia['ra_error'], gaia['dec_error']
    pmra_e,pmdec_e = gaia['pmra_error'], gaia['pmdec_error']
    dist_e,rv_e = allstar['weighted_dist_error']/1e3, allstar['VERR']
    
    if only_velocities:
        ra_e = dec_e = dist_e = np.zeros_like(ra_e)
    ##fi
    
    # Get covariances. Note that nothing covaries with radial velocity
    radec, radist, rapmra, rapmdec = gaia['ra_dec_corr'], 0., gaia['ra_pmra_corr'], gaia['ra_pmdec_corr']
    decdist, decpmra, decpmdec = 0., gaia['dec_pmra_corr'], gaia['dec_pmdec_corr']
    distpmra, distpmdec = 0., 0.
    pmrapmdec = gaia['pmra_pmdec_corr']
    
    if only_velocities:
        radec = radist = rapmra = rapmdec = np.zeros_like(radec)
        decdist = decpmra = decpmdec = np.zeros_like(decdist)
        distpmra = distpmdec = np.zeros_like(distpmra)
    ##fi
    
    # Covariance matrix
    cov = np.zeros((n_samples,6,6))
    rv_zeroed = np.zeros(n_errors)
        
    cov[:,0,:]  = np.dstack([ra_e**2, ra_e*dec_e*radec, ra_e*dist_e*radist, 
        ra_e*pmra_e*rapmra, ra_e*dec_e*rapmdec, rv_zeroed])[0] 
    cov[:,1,1:] = np.dstack([dec_e**2, dec_e*dist_e*decdist, 
        dec_e*pmra_e*decpmra, dec_e*pmdec_e*decpmdec, rv_zeroed])[0]
    cov[:,2,2:] = np.dstack([dist_e**2, dist_e*pmra_e*distpmra, 
        dist_e*pmdec_e*distpmdec, rv_zeroed])[0]
    cov[:,3,3:] = np.dstack([pmra_e**2, pmra_e*pmdec_e*pmrapmdec, 
        rv_zeroed])[0]
    cov[:,4,4:] = np.dstack([pmdec_e**2, rv_zeroed])[0]
    cov[:,5,5] = rv_e**2
    
    # Matrix is symmetric
    cov[:,:,0] = cov[:,0,:]
    cov[:,1:,1] = cov[:,1,1:]
    cov[:,2:,2] = cov[:,2,2:]
    cov[:,3:,3] = cov[:,3,3:]
    cov[:,4:,4] = cov[:,4,4:]
    
    # Mean vectors
    mean = np.dstack([ra,dec,dist,pmra,pmdec,rv])[0]
    
    vxvv_resample = np.empty((n_samples,6))
    for i in range(n_samples):
        try:
            vxvv_resample[i] = np.random.multivariate_normal(mean[i], cov[i], 1)
        except ValueError:
            print(mean[o_mask][i])
            print(cov[o_mask][i]) 
        ##te
    ###i  
    
    orbs_sample = orbit.Orbit(vxvv=vxvv_resample, radec=True, ro=ro, vo=vo, zo=zo)
    return orbs_sample
#def

def trim_gaia_allstar_input(gaia_input,allstar_input):
    '''trim_gaia_allstar_input:
    
    Remove unused fields from Gaia and APOGEE data arrays
    
    Args:
        gaia_input (np.ndarray) - Input Gaia data
        allstar_input (np.ndarray) - Input APOGEE data
    
    Returns:
        gaia_input (np.ndarray) - Output Gaia data
        allstar_input (np.ndarray) - Output APOGEE data
    '''
    
    gaia_fields = ['source_id',
                   'ra',
                   'ra_error',
                   'dec',
                   'dec_error',
                   'parallax',
                   'parallax_error',
                   'pmra',
                   'pmra_error',
                   'pmdec',
                   'pmdec_error',
                   'ra_dec_corr',
                   'ra_parallax_corr',
                   'ra_pmra_corr',
                   'ra_pmdec_corr',
                   'dec_parallax_corr',
                   'dec_pmra_corr',
                   'dec_pmdec_corr',
                   'parallax_pmra_corr',
                   'parallax_pmdec_corr',
                   'pmra_pmdec_corr',
                   'radial_velocity',
                   'radial_velocity_error'
                   ]
    allstar_fields = ['APOGEE_ID',
                      'FIELD',
                      'VHELIO_AVG',
                      'VERR',
                      'weighted_dist',
                      'weighted_dist_error'
                      ]
    return gaia_input[gaia_fields], allstar_input[allstar_fields]
#def

def get_metallicity(allstar, metallicity):
    '''get_metallicity:
    
    Function to parse metallicity string arguments of the form: X_Y where 
    X and Y are species. Assume that either X_Y is in allstar, callable as 
    'X_Y' or that it can be constructed from X_FE-Y_FE
    
    Args:
        allstar (np.ndarray) - ndarray representing the allstar file
        metallicity (str) - string representing the metallicity, of the form 
            'X_Y' where X and Y are species
    
    Returns:
        abundance (array) - abundance corresponding to metallicity
        abundance_err (array) - uncertainty corresponding to metallicity
    '''
    try:
        abundance = allstar[metallicity]
        abundance_err = allstar[metallicity+'_ERR']
        return abundance, abundance_err
    except ValueError:
        spec1,spec2 = metallicity.split('_')
        assert spec1+'_FE' in allstar.dtype.names,\
            spec1+'_FE not in allstar'
        assert spec2+'_FE' in allstar.dtype.names,\
            spec2+'_FE not in allstar'
        abundance = allstar[spec1+'_FE'] - allstar[spec2+'_FE']
        abundance_err = np.sqrt( np.square(allstar[spec1+'_FE_ERR']) +\
                                 np.square(allstar[spec2+'_FE_ERR']) )
        return abundance,abundance_err
    ##te
#def

def make_SF_grid_orbits(apogee_fields,ds,ro,vo,zo,fudge_ll_instability=True):
    '''make_SF_grid_orbits:
    
    Create a list of orbits of length len(apogee_fields) x len(ds)
    representing the locations of a selection function grid over 
    distances and apogee pointings. The orbits correspond to 
    SF.flatten() as long as the SF has dimensions [nfields,nds]
    
    Args:
        apogee_fields (np recarray) - apogee field information array
        ds (array) - list of distances
        ro,vo,zo (float) - galpy scale lengths
        fudge_ll_instability (bool) - Apply fudge for weird ll=90+5e-7 
            galpy error
    
    Returns:
        orbs_grid (galpy.orbit.Orbits) - orbits object corresponding to the 
            flattened SF grid of shape [nfields,nds]
    '''
    # Make linear grid
    ds_lingrid = np.tile(ds,reps=len(apogee_fields))
    ls_lingrid = np.repeat(apogee_fields['GLON'],repeats=len(ds))
    bs_lingrid = np.repeat(apogee_fields['GLAT'],repeats=len(ds))
    
    # Make orbits
    vzero = np.zeros_like(ds_lingrid)
    vxvvs = np.array([ls_lingrid,bs_lingrid,ds_lingrid,vzero,vzero,vzero]).T
    orbs_grid = orbit.Orbit(vxvvs,lb=True,ro=ro,vo=vo,zo=zo)
    
    if fudge_ll_instability:
        print('Addressing galactic ll error')
        where_ll_unstable = np.isnan(orbs_grid.ll())
        ls_lingrid[where_ll_unstable] += 1e-5
        vxvvs = np.array([ls_lingrid,bs_lingrid,ds_lingrid,vzero,vzero,vzero]).T
        orbs_grid = orbit.Orbit(vxvvs,lb=True,ro=ro,vo=vo,zo=zo)
        assert np.all(np.isfinite(orbs_grid.ll())) and\
               np.all(np.isfinite(orbs_grid.bb())) and\
               np.all(np.isfinite(orbs_grid.dist())), 'll fudge did not work!'
    ##fi
    
    return orbs_grid
#def