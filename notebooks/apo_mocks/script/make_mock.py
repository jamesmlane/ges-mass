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
import sys, os, pdb, glob, subprocess, dill as pickle, time, shutil
import gc, psutil
from astropy import units as apu

## galpy
from galpy import orbit
from galpy import potential

## APOGEE, isochrones, dustmaps
import mwdust

## APOGEE mocks
import apomock
from apomock.util.util import join_orbs

# Project-specific packages
sys.path.insert(0,'../../../src/')
from ges_mass import ssf as pssf

### Scale parameters
ro = 8.275 # Gravity
vo = 220
zo = 0.0208 # Bennett+ 2019

# ------------------------------------------------------------------------

# Timing
t1 = time.time()

# Mock
mock_index = '81'
mock_path = '../data/mock_'+str(mock_index)+'/'
if not os.path.exists(mock_path):
    os.makedirs(mock_path)
else:
    sys.exit(mock_path+' already exists, not overwritting')
# Copy this script over to the output
shutil.copyfile('./make_mock_ind1.py', mock_path+'make_mock_script.py')
    
# Selection function
aposf_data_dir = '/geir_data/scr/lane/projects/ges-mass/data/gaia_apogee/'+\
                 'apogee_dr16_l33_gaia_edr3/'
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
iso_grid = np.load('/geir_data/scr/lane/projects/ges-mass/data/gaia_apogee/apogee_dr16_l33_gaia_edr3/iso_grid.npy')
iso = iso_grid[(iso_grid['Zini']==z) & (iso_grid['logAge']==log_age)]
iso = iso[iso['logL']>-9.]

# Density potential
alpha = 3.5
# rc = 25*apu.kpc
denspot_args = {'alpha':alpha,
                #'rc':rc,
                }
denspot = potential.PowerSphericalPotential(**denspot_args, ro=ro, vo=vo)

# Triaxiality & rotation
_b = 0.8
_c = 0.5
_eta = 1./np.sqrt(2.)
_theta = np.pi/2.
_zvec = [np.sqrt(1.-_eta**2.)*np.cos(_theta),np.sqrt(1.-_eta**2.)*np.sin(_theta),_eta]
#_zvec = [0.,1./np.sqrt(2.),1./np.sqrt(2.)] # theta=pi/2, eta=1/sqrt(2) 
_pa = np.pi/5. # pi/5

fallstar = []
orbs = []
m_tot = 2e8
n_chunk = 4

# profiling script
f1 = open(mock_path+'profiling.txt','w')
    
for i in range(n_chunk):
    f1.write('Chunk '+str(i+1)+'\n')
    mock = apomock.APOGEEMockSpherical(denspot, ro=ro, vo=vo)
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
    r_min = 2.0/ro
    r_max = 55.0/ro
    mock.sample_positions(r_min=r_min, r_max=r_max, b=_b, c=_c, 
                          zvec=_zvec, pa=_pa)
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
        f2.write('-----\n\n')
        f2.write('Total mass of mock (e8 Msun): '+str(m_tot/1e8)+'\n')
        f2.write('Number of chunks: '+str(n_chunk)+'\n')
        f2.write('Number of stars in chunk: '+str(len(mock.orbs)))

    del mock
    gc.collect()
    
    mem = psutil.Process(os.getpid()).memory_info().rss/1024**3
    f1.write('end '+str(mem)+'\n\n')

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
gc_locid = pssf.get_globular_cluster_fields()
omask_gc = ~np.isin(fallstar['LOCATION_ID'],gc_locid)
omask_logg = (fallstar['LOGG'] > 1.) & (fallstar['LOGG'] < 3.)

# Number of samples
print('Found '+str(len(fallstar))+' samples')

omask = omask_bulge & omask_gc & omask_logg
print(str(np.sum(omask))+' samples survived observational masking')

np.save(mock_path+'/omask.npy',omask)

# Timing
t2 = time.time()
f1.write('\nTook '+str(t2-t1)+'s')

# Close profiling script
f1.close()