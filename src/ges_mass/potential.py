# ----------------------------------------------------------------------------
#
# TITLE - potential.py
# AUTHOR - James Lane
# PROJECT - ges-mass
# CONTENTS:
#
# ----------------------------------------------------------------------------

### Docstrings and metadata:
'''Potential functions
'''
__author__ = "James Lane"

### Imports

## Basic
import numpy as np
import copy
from astropy import units as apu
from galpy import potential
import scipy.special

# ----------------------------------------------------------------------------

def make_interpolated_mwpot(mwpot='MWPotential2014',rmin=1/8.,rmax=800.,
    ngrid=201,ro=8.,vo=220.,match_type='mass'):
    '''make_interpolated_mwpot:
    
    Make an interpolated version of MW Potential using 
    potential.interpSphericalPotential. Can either be made to match the 
    radial mass profile of the MW or the radial force profile. 
    
    Args:
        mwpot (string or galpy.potential.Potential) - Options are 'MWPotential2014'
        rmin (float) - Minimum radial position used to make the interpolated 
            grid
        rmax (float) - Maximum radial position used to make the interpolated
            grid
        ngrid (int) - Number of points in the radial interpolation grid
        ro (float) - galpy radial scale
        vo (float) - galpy velocity scale
        match_type (string) - Match the radial mass profile ('mass') or the 
            in-plane radial force profile ('force') ['mass'] 
    
    Returns:
        interpot (potential.interpSphericalPotential) - interpolated 
            MW Potential
    '''
    if isinstance(rmin,apu.quantity.Quantity):
        rmin = rmin.value/ro
    if isinstance(rmax,apu.quantity.Quantity):
        rmax = rmax.value/ro

    if isinstance(mwpot,potential.Potential):
        mwp = copy.deepcopy(mwpot)
    else:
        assert isinstance(mwpot,str), 'If not potential.Potential, mwpot must be str'
        if mwpot == 'MWPotential2014':
            mwp = potential.MWPotential2014
        else:
            'MWPotential2014 only implemented mwpot'
    
    rgrid = np.geomspace(rmin,rmax,ngrid)
    assert match_type in ['mass','force'], 'match_type must be "mass" or "force"'
    if match_type == 'mass':
        rforce_fn = lambda r: -potential.mass(mwp,r,use_physical=False)/r**2
    elif match_type == 'force':
        rforce_fn = lambda r: potential.evaluaterforce(mwp,r,0,use_physical=False)
    ##ie
    interpot = potential.interpSphericalPotential(rforce=rforce_fn, 
        rgrid=rgrid, Phi0=potential.evaluatePotentials(mwp,rmin,0,use_physical=False), 
        ro=ro, vo=vo)
    return interpot
#def

# Define density, enclosed mass, force for PowerSphericalPotentialwCutoff
# but with beta=1 (implying the exponent in the exponential cutoff is 1)
def PowerSphericalPotentialwCutoff_beta1_dens(r,alpha,rc,amp=1.):
    '''PowerSphericalPotentialwCutoff_beta1_dens:
    
    Density of the PowerSphericalPotentialwCutoff with beta=1. (implying
    the exponent in the exponential cutoff is 1).

    Args:
        r (float) - radius
        alpha (float) - Power law index
        rc (float) - exponential cutoff radius
    
    Returns:
        dens (float) - Density at r
    '''
    return amp/r**alpha*np.exp(-(r/rc)**1.)

def PowerSphericalPotentialwCutoff_beta1_mass(r, alpha, rc, amp=1.):
    '''PowerSphericalPotentialwCutoff_beta1_mass:

    Enclosed mass of the PowerSphericalPotentialwCutoff with beta=1. (implying
    the exponent in the exponential cutoff is 1).

    Args:
        r (float) - radius
        alpha (float) - power law index
        rc (float) - exponential cutoff radius
    
    Returns:
        mass (float) - mass enclosed at r
    '''
    return amp*4.*np.pi*r**(3-alpha)/(3-alpha)*scipy.special.hyp1f1(3-alpha, 4-alpha, -r/rc)

def PowerSphericalPotentialwCutoff_beta1_force(r, alpha, rc, amp=1.):
    '''PowerSphericalPotentialwCutoff_beta1_force:

    Radial force of the PowerSphericalPotentialwCutoff with beta=1. (implying
    the exponent in the exponential cutoff is 1).

    Args:
        r (float) - radius
        alpha (float) - power law index
        rc (float) - exponential cutoff radius
    
    Returns:
        force (float) - radial force at r
    '''
    return -amp*PowerSphericalPotentialwCutoff_beta1_mass(r,alpha,rc,amp=amp)/r**2

def PowerSphericalPotentialwCutoff_beta1_interp(alpha, rc, amp=1.):
    '''PowerSphericalPotentialwCutoff_beta1_interp:

    Wrapper to make a galpy potential.interpSphericalPotential object for a 
    PowerSphericalPotentialwCutoff with beta=1. (implying the exponent in the
    exponential cutoff is 1).

    Args:
        alpha (float) - power law index
        rc (float) - exponential cutoff radius
        amp (float) - amplitude
    
    Returns:
        interpot (potential.interpSphericalPotential) - interpolated 
            PowerSphericalPotentialwCutoff with beta=1
    '''
    rgrid = np.geomspace(1e-2,100.,10001)
    rforce = lambda r: PowerSphericalPotentialwCutoff_beta1_force(r,alpha,rc,amp=amp)
    interpot = potential.interpSphericalPotential(rforce=rforce, rgrid=rgrid)
    return interpot

def normalize_potential_from_mass(pot, mass, rmin=0., rmax=1e10):
    '''normalize_potential_from_mass:

    Normalize a potential to have a given mass between two radial limits

    Args:
        pot (galpy.potential.Potential) - Potential to normalize
        mass (float) - Mass to normalize to
        rmin (float) - Minimum radial position to integrate over [default 0.]
        rmax (float) - Maximum radial position to integrate over [default 1e10]

    Returns:
        pot (galpy.potential.Potential) - Normalized potential
    '''
    # Check astropy
    if isinstance(mass,apu.quantity.Quantity):
        _is_physical = True
        assert pot._roSet, 'ro must be set if mass is a Quantity'
        assert pot._voSet, 'vo must be set if mass is a Quantity'
    # Candidate mass
    m = potential.mass(pot,rmax)-potential.mass(pot,rmin)
    fac = mass/m
    if _is_physical:
        fac = fac.value
    # Normalize
    pot_norm = pot.__mul__(fac)
    # Run a few checks
    if _is_physical:
        assert np.isclose(potential.mass(pot_norm,rmax).value\
                          -potential.mass(pot_norm,rmin).value,
                          mass.value,rtol=1e-3,atol=1e-5),\
            'Mass normalization failed'
        rs = np.geomspace(rmin.value,rmax.value,11)*apu.kpc
        assert np.all(np.isclose(potential.evaluateDensities(pot_norm,rs,0).value/fac,
                                 potential.evaluateDensities(pot,rs,0).value,
                                 rtol=1e-3,atol=1e-5)),\
            'Density normalization failed'
    else:
        assert np.isclose(potential.mass(pot_norm,rmax)\
                          -potential.mass(pot_norm,rmin),
                          mass,rtol=1e-3,atol=1e-5),\
            'Mass normalization failed'
        rs = np.geomspace(rmin,rmax,11)
        assert np.all(np.isclose(potential.evaluateDensities(pot_norm,rs,0)/fac,
                                 potential.evaluateDensities(pot,rs,0),
                                 rtol=1e-3,atol=1e-5)),\
            'Density normalization failed'
    return pot_norm
