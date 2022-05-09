# ----------------------------------------------------------------------------
#
# TITLE - iso.py
# AUTHOR - James Lane
# PROJECT - ges-mass
# CONTENTS:
#
# ----------------------------------------------------------------------------

### Docstrings and metadata:
'''Utilities for dealing with isochrones, stolen from Mackereth
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

def generate_isogrid():
    '''generate_isogrid:
    
    Make a Padova grid of isochrones.
    
    Args:
        None
    
    Returns:
        rec (recarray) - concatenated list of isochrones
    '''
    zs = np.arange(0.0005,0.0605, 0.0005)
    zlist = []
    for i in range(len(zs)):
        zlist.append(format(zs[i],'.4f'))
    iso = isodist.PadovaIsochrone(type='2mass-spitzer-wise', Z=zs, parsec=True)
    logages, mets, js, hs, ks, loggs, teffs = [], [], [], [], [], [], []
    imf, deltam, M_ini, M_act, logL = [], [], [], [], []
    iso_logages = iso._logages
    iso_Zs = iso._ZS
    for i in tqdm.tqdm(range(len(iso_logages))):
        for j in range(len(iso_Zs)):
            thisage = iso_logages[i]
            thisZ = iso_Zs[j]
            thisiso = iso(thisage, Z=thisZ)
            so = np.argsort(thisiso['M_ini'])
            loggs.extend(thisiso['logg'][so][1:])
            logages.extend(thisiso['logage'][so][1:])
            mets.extend(np.ones(len(thisiso['H'][so])-1)*thisZ)
            js.extend(thisiso['J'][so][1:])
            hs.extend(thisiso['H'][so][1:])
            ks.extend(thisiso['Ks'][so][1:])
            teffs.extend(thisiso['logTe'][so][1:])
            imf.extend(thisiso['int_IMF'][so][1:])
            deltam.extend(thisiso['int_IMF'][so][1:]-thisiso['int_IMF'][so][:-1])
            M_ini.extend(thisiso['M_ini'][so][1:])
            M_act.extend(thisiso['M_act'][so][1:])
            logL.extend(thisiso['logL'][so][1:])
    logages = np.array(logages)
    mets = np.array(mets)
    js = np.array(js)
    hs = np.array(hs)
    ks = np.array(ks)
    loggs = np.array(loggs)
    teffs = 10**np.array(teffs)
    imf = np.array(imf)
    deltam = np.array(deltam)
    M_ini = np.array(M_ini)
    M_act = np.array(M_act)
    logL = np.array(logL)
    rec = np.recarray(len(deltam), dtype=[('logageyr', float),
                                          ('Z', float),
                                          ('J', float),
                                          ('H', float),
                                          ('K', float),
                                          ('logg', float),
                                          ('teff', float),
                                          ('int_IMF', float),
                                          ('deltaM', float),
                                          ('M_ini', float),
                                          ('M_act', float),
                                          ('logL', float)])

    rec['logageyr'] = logages
    rec['Z'] = mets
    rec['J'] = js
    rec['H'] = hs
    rec['K'] = ks
    rec['logg'] = loggs
    rec['teff'] = teffs
    rec['int_IMF'] = imf
    rec['deltaM'] = deltam
    rec['M_ini'] = M_ini
    rec['M_act'] = M_act
    rec['logL'] = logL
    return rec
#def

def generate_lowfeh_isogrid(mag='2mass-spitzer-wise-old'):
    '''generate_lowfeh_isogrid
    
    Make a linear grid of low [Fe/H] isochrones. Fields are:
    
    'Zini', 'MH', 'logAge', 'Mini', 'int_IMF', 'Mass', 'logL', 'logTe', 'logg', 
    'label', 'McoreTP', 'C_O', 'period0', 'period1', 'pmode', 'Mloss', 'tau1m', 
    'X', 'Y', 'Xc', 'Xn', 'Xo', 'Cexcess', 'Z', 'mbolmag', 'Jmag', 'Hmag', 
    'Ksmag', 'IRAC_36mag', 'IRAC_45mag', 'IRAC_58mag', 'IRAC_80mag', 
    'MIPS_24mag', 'MIPS_70mag', 'MIPS_160mag', 'W1mag', 'W2mag', 'W3mag', 
    'W4mag', 'deltaM'
    
    'Zini' runs from 0.0001 to 0.0031 with spacing 0.0001. This is roughly the 
    range -2.28 < [Fe/H] < -0.79 (Calculated using isodist.Z2FEH)
    
    Note: I'm not sure how the function np.lib.recfunctions.append_fields
    works because it shouldn't be in the version of numpy that I use. For 
    some reason it works though. For example if you copy paste this code into 
    ipython then it won't work.
    
    Args:
        mag (string) - Magnitude system to use [2mass-spitzer-wise-old]
        
    Returns:
        fulliso (recarray) - concatenated list of isochrones
    '''
    zs = np.arange(0.0001,0.0031,0.0001)
    base = os.environ['ISODIST_DATA']
    isoname = 'parsec1.2-'
    isolist = []
    for i in tqdm.tqdm(range(len(zs))): 
        file = os.path.join(base,isoname+mag,isoname+mag+'-Z-%5.4f.dat.gz' % zs[i])
        alliso = np.genfromtxt(file, dtype=None, names=True, skip_header=11)
        ages = np.unique(alliso['logAge'])
        for age in ages:
            mask = alliso['logAge'] == round(age,2)
            iso = alliso[mask]
            iso = iso[np.argsort(iso['Mini'])]
            deltam = iso['int_IMF'][1:]-iso['int_IMF'][:-1]
            iso = np.lib.recfunctions.append_fields(iso[1:], 'deltaM', deltam)
            isolist.append(iso)
    fulliso = np.concatenate([entry for entry in isolist])
    return fulliso
#def

### Isochrone Sampling and Properties

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