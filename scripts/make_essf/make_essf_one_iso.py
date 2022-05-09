# ----------------------------------------------------------------------------
#
# TITLE - make_essf_one_iso.py
# AUTHOR - James Lane 
# PROJECT - ges-mass
# CONTENTS:
#
# ----------------------------------------------------------------------------

### Docstrings and metadata:
'''Makes the effective survey selection function for APOGEE using a single 
isochrone drawn from a larger grid of isochrones
'''
__author__ = "James Lane"

import numpy as np, pdb, sys, os, dill as pickle

# Set APOGEE version for the package
apogee_results_vers = 'l33'
# Year 7 is appropriate for DR16 (l33)
apo_combined_select_year = 7
os.environ['RESULTS_VERS'] = apogee_results_vers

import apogee.select as apsel
import apogee.tools as apotools
from isodist import Z2FEH
import mwdust
import multiprocessing
import tqdm

# Project specific
sys.path.insert(0,'../../src/')
from ges_mass import iso as piso
from ges_mass import util as putil

# ----------------------------------------------------------------------------

### Preliminaries

# Keywords
cdict = putil.load_config_to_dict()
keywords = ['DATA_DIR','APOGEE_DR','APOGEE_RESULTS_VERS','GAIA_DR']
data_dir_base,apogee_dr,apogee_results_vers,gaia_dr = putil.parse_config_dict(cdict,keywords)
data_dir = data_dir_base+'gaia_apogee/'
gaia_apogee_dir = 'apogee_'+apogee_dr+'_'+apogee_results_vers+'_gaia_'+gaia_dr+'/'

# Sanity
assert apotools.path._APOGEE_REDUX == apogee_results_vers, 'Not using correct results version!'

# Filenames
apogee_SF_filename = data_dir+gaia_apogee_dir+'apogee_SF.dat'

# Forcing
force_effSF = False # Force calculation of the essf even if one exists?

# Parameters for effective survey selection function
n_iso_samples=2000
ndistmods=301
minmax_distmods=[7.,19.]
distmods = np.linspace(minmax_distmods[0], minmax_distmods[1], ndistmods)
ds = 10.**(distmods/5-2)
nthreads = int(multiprocessing.cpu_count()//4)

# ----------------------------------------------------------------------------

### Loading

# Selection function
print('APOGEE data release is: '+apogee_dr+', and results version is: '+apogee_results_vers)
if os.path.exists(apogee_SF_filename):
    print('Loading APOGEE selection function from '+apogee_SF_filename)
    with open(apogee_SF_filename, 'rb') as f:
        apogee_SF = pickle.load(f)
    ##wi
else:
    sys.exit('Could not find APOGEE selection function, make it. Exiting...')
##ie

# Dustmap and isochrones
print('Loading isochrone grid')
iso_filename = data_dir+gaia_apogee_dir+'iso_grid.npy'
iso = np.load(iso_filename)
print('Loading Combined19 dust map from mwdust')
dmap = mwdust.Combined19(filter='2MASS H') # dustmap from mwdust, use most recent

# ----------------------------------------------------------------------------

### Effective selection function

# Generate isochrone samples
# Choose a specific_isochrone
Zini = 0.0010
logAge = 10.
iso_mask = (iso['logg'] > 1.) &\
           (iso['logg'] < 3.) &\
           (iso['Zini'] == Zini) &\
           (iso['logAge'] == logAge)
H = iso['Hmag'][iso_mask]
J = iso['Jmag'][iso_mask]
K = iso['Ksmag'][iso_mask]
weights = iso['weights_imf'][iso_mask]/iso['Zini'][iso_mask]

# Create an APOGEE effective selection function object
apogee_effSF = apsel.apogeeEffectiveSelect(apogee_SF, dmap3d=dmap, MH=H, 
                                           JK0=(J-K), weights=weights)
apogee_effSF_filename = data_dir+gaia_apogee_dir+\
    'apogee_effSF_grid_inclArea_z'+str(Zini)+'_logAge'+str(logAge)+'.dat'

# print('Generating isochrone samples')
# niso, p3niso = piso.APOGEE_iso_samples(n_iso_samples, iso, 
#                                        fehrange=[-1.6, -0.6], lowfehgrid=True)
# H = np.array(niso['Hmag'].data)
# J = np.array(niso['Jmag'].data)
# K = np.array(niso['Ksmag'].data)
# p3H = np.array(p3niso['Hmag'].data)
# p3J = np.array(p3niso['Jmag'].data)
# p3K = np.array(p3niso['Ksmag'].data)

# Function to calculate the effective selection function at one location, 
# Based on the loaded isochrone, the dust map, and including the area factor
def _calc_effSF_inclArea_one_loc(i): 
    loc = apogee_SF._locations[i]
    if np.sum([np.nansum(apogee_SF._nspec_short[i]),
               np.nansum(apogee_SF._nspec_medium[i]),
               np.nansum(apogee_SF._nspec_long[i])]) < 1.:
        effSF_inclArea = np.zeros(len(ds))
    else:
        effSF_inclArea = apogee_effSF(loc, ds, MH=H, JK0=(J-K), weights=weights)\
                         *apogee_SF.area(loc)
#     elif apogee_SF.JKmin(loc) >= 0.5:
#         effSF_inclArea = apogee_effSF(loc, ds, MH=H, JK0=(J-K), weights=weights)\
#                          *apogee_SF.area(loc)
#     elif apogee_SF.JKmin(loc) < 0.5:
#         effSF_inclArea = apogee_effSF(loc, ds, MH=p3H, JK0=(p3J-p3K))\
#                          *apogee_SF.area(loc)
    return effSF_inclArea
#def


print('Calculating effective selection function for APOGEE fields...')
with multiprocessing.Pool(nthreads) as p:
    out = list(tqdm.tqdm(p.imap(_calc_effSF_inclArea_one_loc, 
                                range(0,len(apogee_SF._locations))), 
                         total=len(apogee_SF._locations)))
##wi
apogee_effSF_grid_inclArea = np.array(out)
with open(apogee_effSF_filename, 'wb') as f:
    pickle.dump(apogee_effSF_grid_inclArea, f)
##wi

# ----------------------------------------------------------------------------
