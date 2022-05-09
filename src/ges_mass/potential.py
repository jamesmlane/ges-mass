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