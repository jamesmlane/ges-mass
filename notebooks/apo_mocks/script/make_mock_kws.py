# ------------------------------------------------------------------------
#
# TITLE - make_apo_mock
# AUTHOR - James Lane
# PROJECT - ges-mass
#
# ------------------------------------------------------------------------
#
# Docstrings and metadata:
'''Make APOGEE mocks
'''

__author__ = "James Lane"

### Imports

## Basic
import numpy as np
import sys, os, pdb, copy, glob, subprocess, warnings, dill as pickle, time
import gc, psutil, argparse, shutil
from astropy import units as apu

## galpy
from galpy import orbit
from galpy import potential

## APOGEE, isochrones, dustmaps
import mwdust

## APOGEE mocks
import apomock
from apomock.util.util import join_orbs

### Scale parameters
ro = 8.275 # Gravity
vo = 220
zo = 0.0208 # Bennett+ 2019

# ------------------------------------------------------------------------

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--rmin',type=float,required=True)
parser.add_argument('--mock',type=str,required=True)
args = parser.parse_args()
mock_number = str(args.mock)
r_min = float(args.rmin)/ro

# Mock path
mock_path = '../data/mock_'+str(mock_number)+'/'
if not os.path.exists(mock_path):
    os.makedirs(mock_path)
else:
    sys.exit(mock_path+' already exists, not overwritting')

# Selection function
aposf_data_dir = '/geir_data/scr/lane/projects/ges-mass/data/gaia_apogee/'+\
                 'apogee_dr16_l33_gaia_dr2/'
with open(aposf_data_dir+'apogee_SF.dat','rb') as f:
    aposf = pickle.load(f)

# Dustmap
dmap = mwdust.Combined19(filter='2MASS H') # dustmap from mwdust
#dmap = mwdust.Zero(filter='2MASS H')

# Isochrone
_parsec_1_2_iso_keys = {'mass_initial':'Mini',
                        'z_initial':'Zini',
                        'log_age':'logAge',
                        'jmag':'Jmag',
                        'hmag':'Hmag',
                        'ksmag':'Ksmag',
                        'logg':'logg',
                        'logteff':'logTe'
                        }
z = 0.0010
log_age = 10.0
iso_grid = np.load('/geir_data/scr/lane/projects/ges-mass/data/gaia_apogee/apogee_dr16_l33_gaia_dr2/iso_grid.npy')
iso = iso_grid[(iso_grid['Zini']==z) & (iso_grid['logAge']==log_age)]
iso = iso[iso['logL']>-9.]

# Density potential
alpha = 3.5
denspot_args = {'alpha':alpha}
denspot = potential.PowerSphericalPotential(**denspot_args, ro=ro, vo=vo)

fallstar = []
orbs = []
m_tot = 5e8
n_chunk = 10

# RAM profiling script
f1 = open(mock_path+'memory_usage.txt','w')

# Write arguments
with open(mock_path+'args.txt','w') as f:
    f.write(str(sys.argv))

# Copy this script
this_filename = os.path.abspath(__file__)
shutil.copyfile(this_filename,mock_path+'script.py')

for i in range(n_chunk):
    f1.write('Chunk '+str(i+1)+'\n')
    mock = apomock.APOGEEMock(denspot, ro=ro, vo=vo)
    mock.load_isochrone(iso=iso, iso_keys=_parsec_1_2_iso_keys)

    print('Sampling masses')
    t1 = time.time()
    
    m_min = 0.08
    mock.sample_masses(m_tot/n_chunk, m_min=m_min)
    t2 = time.time()
    mem = psutil.Process(os.getpid()).memory_info().rss/1024**3
    f1.write('mass '+str(mem)+'\n')
    print('Took '+str(t2-t1)+' s')
    
    print('Sampling positions')
    t1 = time.time()
    # r_min = 2.0/ro
    r_max = 70./ro
    mock.sample_positions(r_min=r_min, r_max=r_max)
    t2 = time.time()
    mem = psutil.Process(os.getpid()).memory_info().rss/1024**3
    f1.write('orbs '+str(mem)+'\n')
    print('Took '+str(t2-t1)+' s')

    print('Applying selection function')
    t1 = time.time()
    mock.apply_selection_function(aposf, dmap)
    mem = psutil.Process(os.getpid()).memory_info().rss/1024**3
    f1.write('ssf '+str(mem)+'\n')
    t2 = time.time()
    print('Took '+str(t2-t1)+' s')

    fallstar.append( mock.make_allstar() )
    orbs.append( mock.orbs )
    
    # Write a summary of the mock
    mock_summary = mock._write_mock_summary()
    with open(mock_path+'/mock_summary_chunk_'+str(i+1)+'.txt.','w') as f2:
        for line in mock_summary:
            f2.write(line)
            f2.write('\n')

    del mock
    gc.collect()
    
    mem = psutil.Process(os.getpid()).memory_info().rss/1024**3
    f1.write('end '+str(mem)+'\n\n')
    
print('Found '+str(len(fallstar))+' samples')
f1.close()

# Orbits
orbs = join_orbs(orbs)
with open(mock_path+'/orbs.pkl','wb') as f:
    pickle.dump(orbs,f)
##wi

# Fake allstar
fallstar = np.concatenate(fallstar)
np.save(mock_path+'/allstar.npy',fallstar)

# Masking
# Cut bulge fields. Within 20 degrees of the galactic center
omask_bulge = ~(((orbs.ll().value > 340.) |\
                (orbs.ll().value < 20.)) &\
               (np.fabs(orbs.bb().value) < 20.)
              )
# Cut fields containing enhancements of globular cluster stars
gc_locid = [2011,4353,5093,5229,5294,5295,5296,5297,5298,5299,5300,5325,5328,
            5329,5438,5528,5529,5744,5801]
omask_gc = ~np.isin(fallstar['LOCATION_ID'],gc_locid)
omask_logg = (fallstar['LOGG'] > 1.) & (fallstar['LOGG'] < 3.)

omask = omask_bulge & omask_gc & omask_logg
print(str(np.sum(omask))+' samples survived observational masking')

np.save(mock_path+'/omask.npy',omask)