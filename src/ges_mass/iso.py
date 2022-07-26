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
import warnings
import isodist
from isodist import Z2FEH,FEH2Z
import scipy.interpolate
import scipy.integrate

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

# ----------------------------------------------------------------------------

### Isochrone weights

def calculate_weights_imf(iso, weights_key='weights_imf', imf=None, m_min=0.,
                          norm=1., diff=False, diff_key='weights_imf', 
                          overwrite=False, iso_type='parsec1.2'):
    '''calculate_weights_imf:
    
    Calculate 'weights_imf' field for isochrones. Can either be done by 
    using an IMF to calculate the weight between two mass points in the 
    isochrone, or by taking the difference between an existing cumulative IMF 
    field in the isochrone (e.g. 'int_IMF' for Parsec1.2 isochrones).
    
    Weights are calculated for each unique pair of Zini, logAge
    
    Args:
        iso (np structured array) - Isochrone array. Will be appended to
        weights_key (string) - Key name for the new weights field
            [default 'weights_imf']
        imf (callable) - IMF function to calculate the weights for each mass 
            interval [default None]
        m_min (float) - Initial mass to integrate the first mass point
        norm (float) - Normalizing factor to apply divide weights [default 1.]
        diff (bool) - Calculate the weights by differencing an existing 
            field in the isochrone? [default False]
        diff_key (string) - If diff=True which field in the isochrone to 
            use for differencing [default None]
        overwrite (bool) - If the array field already exists overwrite?
            [default False]
        iso_type (string) - Isochrone type for keys [default 'parsec1.2']
    
    Returns:
        iso (np structured array) - Isochrone array with weights field appended
    '''
    # Checks
    assert callable(imf) or diff, 'Either imf must be a callable or diff=True'
    
    # Isochrone keys
    _iso_keys = iso_keys(iso_type)
    age_key = _iso_keys['logage']
    z_key = _iso_keys['z_initial']
    mass_key = _iso_keys['mass_initial']
    
    # Unique metallicity and ages
    unique_zini = np.unique(iso[z_key])
    unique_age = np.unique(iso[age_key])
    
    # Make the weights
    w_imf = np.zeros(len(iso), dtype=[(weights_key,'f8'),])
    for i in range(len(unique_zini)):
        for j in range(len(unique_age)):
            print('doing metallicity '+str(i+1)+'/'+str(len(unique_zini)),
                  end='\r')
            iso_mask = np.where((iso['Zini']==unique_zini[i]) &\
                                (iso['logAge']==unique_age[j]))[0]
            if diff:
                w_imf[iso_mask[1:]] = np.diff(iso[iso_mask][diff_key])/norm
                w_imf[iso_mask[0]] = iso[iso_mask[0]][diff_key]/norm
            else:
                for k in range(len(iso_mask)):
                    if iso[iso_mask[0]][mass_key]<m_min:
                        warnings.warn('First mass entry in isochrone for '+\
                                      'z='+str(unique_zini[i])+' and '+\
                                      'logAge='+str(unique_age[j])+' is '+\
                                      'less than m_min.')
                    
                    if k == 0:
                        w_imf[iso_mask[k]] = scipy.integrate.quad(imf, m_min, 
                            iso[iso_mask[k]][mass_key])[0]/norm
                    else:
                        w_imf[iso_mask[k]] = scipy.integrate.quad(
                            imf, iso[iso_mask[k-1]][mass_key], 
                            iso[iso_mask[k]][mass_key])[0]/norm
    print('')
    # Check whether or not the field that's being appended already exists
    if weights_key in iso.dtype.names:
        if overwrite:
            iso[weights_key] = w_imf[weights_key]
        else:
            warnings.warn('Field already present in iso, not overwriting')
    else:
        iso = np.lib.recfunctions.merge_arrays((iso,w_imf), flatten=True)
    return iso

# ----------------------------------------------------------------------------

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
    tinter = scipy.interpolateinterp1d(np.cumsum(weights[sort])/np.sum(weights), 
                                       range(len(weights[sort])), kind='linear')
    randinds = np.round(tinter(np.random.rand(N))).astype(np.int64)
    if return_inds:
        return randinds, iso[jkey][sort][randinds], iso[hkey][sort][randinds], iso[kkey][sort][randinds]
    elif return_iso:
        return iso[sort][randinds]
    else:
        return iso[jkey][sort][randinds], iso[hkey][sort][randinds], iso[kkey][sort][randinds]

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


# ----------------------------------------------------------------------------

### IMFs

def chabrier01_lognormal(m,A=0.141):
    '''chabrier01_lognormal:
    
    Chabrier (2001) lognormal IMF
    
    Args:
        m (float or np.array) - mass, use Msol if normalization matters
        A (float) - Normalization [default 0.141 as per Chabrier (2001)]
        
    Returns
        dn/dm (np.ndarray) - Value of the IMF for given linear mass interval
    '''
    return _chabrier_lognormal(m,m0=0.1,sigma=0.627,A=A)

def chabrier01_exponential(m,A=3.0):
    '''chabrier01_exponential:
    
    Chabrier (2001) exponential IMF
    
    Args:
        m (float or np.array) - mass, use Msol if normalization matters
        A (float) - Normalization [default 3.0 as per Chabrier (2001)]
        
    Returns
        dn/dm (np.ndarray) - Value of the IMF for given linear mass interval
    '''
    m0 = 716.4
    alpha = -3.3
    beta = 0.25
    return A*(m**alpha)*np.exp(-(m/m0)**beta)
    
def chabrier03_lognormal(m,A=0.158):
    '''chabrier01_lognormal:
    
    Args:
        m (float or np.array) - mass, use Msol if normalization matters
        A (float) - Normalization [default 0.158 as per Chabrier (2003)]
        
    Returns
        dn/dm (np.ndarray) - Value of the IMF for given linear mass interval
    '''
    return _chabrier_lognormal(m,m0=0.079,sigma=0.69,A=A)
    

def chabrier05_lognormal(m,A=0.093):
    '''chabrier01_lognormal:
    
    Args:
        m (float or np.array) - mass, use Msol if normalization matters
        A (float) - Normalization [default 0.093 as per Chabrier (2005)]
        
    Returns
        dn/dm (np.ndarray) - Value of the IMF for given linear mass interval
    '''
    return _chabrier_lognormal(m,m0=0.2,sigma=0.55,A=A)
    

def _chabrier_lognormal(m,m0=0.,sigma=1.,A=1.):
    '''_chabrier_lognormal:
    
    Lognormal-type Chabrier initial mass function
    
    Args:
    
    Returns:
        dn/dm (np.ndarray) - Value of the IMF for given linear mass interval
    '''
    dNdlogm = np.exp(-(np.log10(m)-np.log10(m0))**2/(2.*sigma**2.))
    dlogmdm = 1./m/np.log(10.)
    return A*dNdlogm*dlogmdm
                     
def kroupa(m,A=0.48):
    '''kroupa:
    
    Kroupa initial mass function
    
    Args:
        m (np.ndarray) - Masses [solar]
        A (float) - Normalization for the first power law (all other follow
            to make sure boundaries are continuous) [default 0.48 matches
            normalization of Chabrier 2003 lognormal at 1 solar mass]
    
    Returns:
        Nm (np.ndarray) - Value of the IMF for given masses  
    '''
    a1,a2,a3 = 0.3,1.3,2.3
    k2 = 0.08*A
    k3 = 0.5*k2
    
    if not isinstance(m,np.ndarray):
        m = np.atleast_1d(m)
    ##fi
    
    where_m_1 = np.logical_and(m>=0.01,m<0.08)
    where_m_2 = np.logical_and(m>=0.08,m<0.5)
    where_m_3 = m>=0.5
    Nm = np.empty(len(m))
    Nm[where_m_1] = A*m[where_m_1]**(-a1)
    Nm[where_m_2] = k2*m[where_m_2]**(-a2)
    Nm[where_m_3] = k3*m[where_m_3]**(-a3)
    Nm[m<0.01] = 0
    return Nm

def cimf(f,a,b,intargs=()):
    '''cimf:
    
    Calculate the cumulative of the initial mass function
    
    Args:
        f (callable) - IMF function
        a,b (float) - lower and upper integration bounds
        intargs (dict) - Dictionary of parameters to pass to f [optional]
    '''
    return scipy.integrate.quad(f,a,b,args=intargs)[0]