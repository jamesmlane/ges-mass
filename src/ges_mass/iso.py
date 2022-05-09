# ----------------------------------------------------------------------------
#
# TITLE - iso.py
# AUTHOR - James Lane
# PROJECT - ges-mass
# CONTENTS:
#
# ----------------------------------------------------------------------------

### Docstrings and metadata:
'''Utilities for dealing with isochrones, re-written from Mackereth
'''
__author__ = "James Lane"

### Imports
import os
import numpy as np
import isodist
from isodist import Z2FEH,FEH2Z
import tqdm
from scipy.interpolate import interp1d

# ----------------------------------------------------------------------------

### Isochrone properties

def average_mass(iso, weight_inverse_z=True, weight_log_age=False, 
                 iso_type='parsec1.2', weights_key='weights_imf'):
    '''average_mass:
    
    Find the average mass for a given slice of the isochrone recarray. Assume 
    the isochrone recarray is sliced coming into the function. iso must 
    have IMF weights accessible by 'weights_imf'
    
    Args:
        iso (recarray) - Isochrone recarray
        weight_inverse_z (bool) - Weight by inverse of Z, accounts for 
            linear spacing in Z, so that sampling is even in [Fe/H]
        weight_log_age (bool) - Weight by logAge to account for logarithmic 
            spacing in isochrone ages, not necessary for linear spacing
        iso_type (string) - String denoting isochrone library for key access
        weights_key (string) - Special key denoting a set of weights either 
            supplied in the isochrone or added to the isochrone by user. 
            Denotes IMF weights.
    
    Returns:
        m_weight (float) - Weighted mass
    '''
    _iso_keys = iso_keys(iso_type)
    age_key = _iso_keys['logage']
    z_key = _iso_keys['z_initial']
    mass_key = _iso_keys['mass_initial']
    
    # Get the IMF weights.
    try:
        weights = iso[weights_key]
    except ValueError:
        sys.exit('Must have weights_imf key')
    
    if weight_log_age:
        weights *= 10**(iso[logage_key]-9)
    if weight_inverse_z:
        weights *= 1/iso[z_key]
    
    return np.sum(iso[mass_key]*weights)/np.sum(weights)

def mass_ratio(iso, logg_range, jk_range, weight_inverse_z=True, 
               weight_log_age=False, iso_type='parsec1.2', 
               weights_key='weights_imf'):
    '''mass_ratio:
    
    Find the mass ratio between stars in the cuts adopted for giants in APOGEE, 
    and the rest of the isochrones. iso must have IMF weights accessible by 
    'weights_imf'
    
    Args:
        iso (recarray) - Isochrone recarray
        logg_range (list) - Limits for logg [lower,upper]
        jk_range (array) - Limits for J-K if applicable [lower,upper] [None]
        weight_inverse_z (bool) - Weight by inverse of Z, accounts for 
            linear spacing in Z, so that sampling is even in [Fe/H]
        weight_log_age (bool) - Weight by logAge to account for logarithmic 
            spacing in isochrone ages, not necessary for linear spacing
        iso_type (string) - String denoting isochrone library for key access
        weights_key (string) - Special key denoting a set of weights either 
            supplied in the isochrone or added to the isochrone by user. 
            Denotes IMF weights.
    
    Returns:
        mass_ratio (float) - Mass ratio
    '''
    _iso_keys = iso_keys(iso_type)
    age_key = _iso_keys['logage']
    z_key = _iso_keys['z_initial']
    mass_key = _iso_keys['mass_initial']
    logg_key = _iso_keys['logg']
    j_key = _iso_keys['jmag']
    ks_key = _iso_keys['ksmag']
    
    try:
        weights = iso[weights_key]
    except ValueError:
        sys.exit('Must have weights_imf key')
    
    if weight_log_age:
        weights *= 10**(iso[logage_key]-9)
    if weight_inverse_z:
        weights *= 1/iso[z_key]
    
    # Mask the fitting sample
    mask = (iso[logg_key] > logg_range[0]) &\
           (iso[logg_key] < logg_range[1]) &\
           (iso[j_key]-iso[ks_key] > jk_range[0]) &\
           (iso[j_key]-iso[ks_key] < jk_range[1])
    
    return np.sum(iso[mass_key][mask]*weights[mask])/np.sum(iso[mass_key]*weights)

def iso_keys(iso_type):
    '''iso_keys:
    
    Get keys for various isochrones. Allows the code to use a single set 
    of keys to access numerous isochrone libraries. The dictionary _iso_keys 
    links the relevent keys for the isochrone array with this common set of 
    keys used by the code:
        
    The following are required for core functionality
    'mass_initial' - Initial mass of each point in the isochrone
    'logage' - Log base 10 age of each point in the isochrone
    'z_initial' - Initial metal fraction (z) of each point in the isochrone
    'jmag' - J-band magnitude
    'ksmag' - Ks-band magnitude
    'logg' - Log base 10 surface gravity

    So for example if the initial mass in the isochrone is accessed 
    by calling iso['Mini'], then one element of iso_keys should be 
    {...,'mass_initial':'Mini',...} and so on. See below for example
    
    Currently supported libraries are:
        - PARSEC v1.2 'parsec1.2'
    
    Args:
        iso_type (str) - String denoting the type of isochrone library.
        
    Returns:
        _iso_keys (dict) - Dictionary of isochrone key pairs
    '''
    if iso_type == 'parsec1.2':
        _iso_keys = {'mass_initial':'Mini',
                     'z_initial':'Zini',
                     'logage':'logAge',
                     'jmag':'Jmag',
                     'hmag':'Hmag',
                     'ksmag':'Ksmag',
                     'logg':'logg',
                     'logteff':'logTe'}
    # elif: Add more dictionaries
    return _iso_keys

### Isochrone sampling

def sampleiso(N, iso, return_inds=False, return_iso=False, lowfeh=True):
    '''sampleiso:
    
    Sample isochrone recarray iso weighted by lognormal Chabrier IMF. Function 
    is used by APOGEE_iso_samples.
    
    Args:
        N (int) - Number of samples
        iso (recarray) - Isochrone recarray.
        return_inds (bool) - Return the random sample indices
        return_iso (bool) - Return the full isochrone
        lowfeh (bool) - Use the low [Fe/H] isochrone grid
    
    Returns:
        randinds (array) - Random indices, only if return_inds=True
        iso_j (array) - J samples, only if return_iso=False or return_inds=True
        iso_h (array) - H samples, only if return_iso=False or return_inds=True
        iso_k (array) - K samples, only if return_iso=False or return_inds=True
        iso (recarray) - Full isochrone recarray, only if return_iso=True
    '''
    if lowfeh:
        logagekey = 'logAge'
        zkey = 'Zini'
        jkey, hkey, kkey = 'Jmag', 'Hmag', 'Ksmag'
    else:
        logagekey = 'logageyr'
        zkey = 'Z'
        jkey, hkey, kkey = 'J', 'H', 'K'
    weights = iso['deltaM']*(10**(iso[logagekey]-9)/iso[zkey])
    sort = np.argsort(weights)
    tinter = interp1d(np.cumsum(weights[sort])/np.sum(weights), 
                      range(len(weights[sort])), kind='linear')
    randinds = np.round(tinter(np.random.rand(N))).astype(np.int64)
    if return_inds:
        return randinds, iso[jkey][sort][randinds], iso[hkey][sort][randinds], iso[kkey][sort][randinds]
    elif return_iso:
        return iso[sort][randinds]
    else:
        return iso[jkey][sort][randinds], iso[hkey][sort][randinds], iso[kkey][sort][randinds]
    ##ie
#def

def APOGEE_iso_samples(nsamples, rec, fehrange=[-1,-1.5], lowfehgrid=True):
    '''APOGEE_iso_samples:
    
    Sample from an isochrone grid in accordance with the APOGEE selection, 
    including logg cuts for giants and minimum mass. Function is used 
    in the calculation of the effective selection function.
    
    Note: The limits in logg and J-K are hardcoded. Make sure this does not 
    present a problem anywhere in the code.
    
    Args:
        nsamples (int) - Number of samples to draw
        rec (recarray) - isochrone recarray
        ferange (list) - Range of [Fe/H]
        lowfehgrid (bool) - Use the low [Fe/H] isochrone grid 
    
    Returns:
        niso () - The normal isochrone grid samples (J-K > 0.5)
        p3niso () - The blue isochrone grid samples (J-K > 0.3)
    '''
    trec = np.copy(rec)
    if lowfehgrid:
        logagekey = 'logAge'
        zkey = 'Zini'
        mkey = 'Mini'
        jkey, hkey, kkey = 'Jmag', 'Hmag', 'Ksmag'
    else:
        logagekey = 'logageyr'
        zkey = 'Z'
        mkey ='M_ini'
        jkey, hkey, kkey = 'J', 'H', 'K'
    mask = (trec[jkey]-trec[kkey] > 0.5)  &\
           (Z2FEH(trec[zkey]) > fehrange[0]) &\
           (Z2FEH(trec[zkey]) < fehrange[1]) &\
           (trec[mkey] > 0.75) & (trec['logg'] < 3.) &\
           (trec['logg'] > 1.) & (trec[logagekey] >= 10.)
    niso = sampleiso(nsamples,trec[mask], return_iso=True, lowfeh=lowfehgrid)
    mask = (trec[jkey]-trec[kkey] > 0.3001)  &\
           (Z2FEH(trec[zkey]) > fehrange[0]) &\
           (Z2FEH(trec[zkey]) < fehrange[1]) &\
           (trec[mkey] > 0.75) & (trec['logg'] < 3.) &\
           (trec['logg'] > 1.) & (trec[logagekey] >= 10.)
    p3niso = sampleiso(nsamples,trec[mask], return_iso=True, lowfeh=lowfehgrid)
    return niso, p3niso
#def