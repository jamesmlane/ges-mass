# ----------------------------------------------------------------------------
#
# TITLE - ssf.py
# AUTHOR - James Lane
# PROJECT - ges-mass
# CONTENTS:
#
# ----------------------------------------------------------------------------

### Docstrings and metadata:
'''Utilities for dealing with the selection function, effective selection 
function, and kinematic selection function
'''
__author__ = "James Lane"

### Imports
import numpy as np
import os
import copy
from scipy import interpolate
from matplotlib import pyplot as plt
# plt.style.use('../mpl/project.mplstyle')
import dill as pickle
from galpy import orbit
from galpy import potential
from galpy.util import multi as galpy_multi
from astropy import table
from astropy import units as apu
import urllib

from . import plot as pplot
from . import util as putil

# ----------------------------------------------------------------------------

# Define the function that will take all input and calculate results in parallelized manner
def calc_kinematics_parallel(ncores,this_df,n_samples,orbs_locs,do_perturb_orbs,
                             gaia_input,allstar_input,delta,aAS,pot,ro,vo,zo):
    '''calc_kinematics_parallel:
    
    Calculate the kinemaitcs according to a DF. Returns 
    orbs, eELz, accs
    
    Args:
        ncores (int) - Number of processors to use
        this_df (df instance) - DF to use to calculate kinematics
        n_samples (int) - Number of samples to draw
        orbs_locs (orbit.Orbit instance) - Orbit object representing location 
            to draw samples
        do_perturb_orbs (bool) - Perturb the orbits by an amount equal to 
            the uncertainty from nearby data
        gaia_input (array) - Gaia data
        allstar_input (array) - APOGEE allstar data
        delta (array) - Staeckel deltas for location
        aAS (actionAngleStaeckel instance) - Staeckel action angle object 
            to calculate kinematic quantities
        pot (potential.Potential instance) - Milky Way potential
        ro,vo,zo (float) - Galpy scales and Sun position
    
    Returns:
        results (array) - Array of [orbs,eELz,accs]
    '''
     
    lambda_func = (lambda x: calc_kinematics_one_loc(this_df,n_samples,
        orbs_locs[x], do_perturb_orbs, gaia_input[[x,]], allstar_input[[x,]],
        delta[x], aAS, pot, ro, vo, zo))
    
    n_calls = len(orbs_locs)
    print('Using '+str(ncores)+' cores')
    results = (galpy_multi.parallel_map(lambda_func, 
               np.arange(0,n_calls,1,dtype='int'),  
               numcores=ncores,progressbar=True))
    
    # By wrapping in numpy array results can be querried as 
    # results[:,0]: array of n_calls orbits, each orbit n_samples long
    # results[:,1]: array of n_calls eELzs, each is (3,1000) of e,E,Lz
    # results[:,2]: array of n_calls actions, each is (3,1000) of JR,Lz,Jz
    return np.array(results,dtype='object')


def calc_kinematics_one_loc(df,n_samples,orbs_locs,do_perturb_orbs,gaia_input,
                    allstar_input,delta,aAS,pot,ro,vo,zo):
    '''calc_kinematics_one_loc:
    
    Calculate the kinemaics for a single location
    
    Args:
        df (df object) - DF to use to calculate kinematics
        n_samples (int) - Number of samples to draw
        orbs_locs (orbit.Orbit instance) - Orbit object representing location 
            to draw samples
        do_perturb_orbs (bool) - Perturb the orbits by an amount equal to 
            the uncertainty from nearby data
        gaia_input (array) - Gaia data
        allstar_input (array) - APOGEE allstar data
        delta (array) - Staeckel deltas for location
        aAS (actionAngleStaeckel instance) - Staeckel action angle object 
            to calculate kinematic quantities
        pot (potential.Potential instance) - Milky Way potential
        ro,vo,zo (float) - Galpy scales and Sun position
    
    Returns:
        results (arr) - Array of orbs, eELzs, actions
    '''
    orbs_samp = df.sample(n=n_samples,R=np.ones_like(n_samples)*orbs_locs.R(),
                          phi=np.ones_like(n_samples)*orbs_locs.phi(),
                          z=np.ones_like(n_samples)*orbs_locs.z())

    # Resample orbits based on positional matches
    if do_perturb_orbs:
        orbs_samp = putil.perturb_orbit_with_Gaia_APOGEE_uncertainties(orbs_samp,
            gaia_input,allstar_input,only_velocities=True,ro=ro,vo=vo,zo=zo)
    
    ecc,_,_,_ = aAS.EccZmaxRperiRap(orbs_samp, delta=delta, 
                                    use_physical=True, c=True)
    accs = aAS(orbs_samp, delta=delta, c=True)
    E = orbs_samp.E(pot=pot)
    Lz = orbs_samp.Lz()

    try:
        ecc = ecc.value # No units
    except AttributeError:
        pass
    try:
        jr = accs[0].to(apu.kpc*apu.km/apu.s).value
        jp = accs[1].to(apu.kpc*apu.km/apu.s).value
        jz = accs[2].to(apu.kpc*apu.km/apu.s).value
    except AttributeError:
        jr = accs[0]*ro*vo
        jp = accs[1]*ro*vo
        jz = accs[2]*ro*vo
    try:
        E = E.to(apu.km*apu.km/apu.s/apu.s).value
        Lz = Lz.to(apu.kpc*apu.km/apu.s).value
    except AttributeError:
        E *= (vo*vo)
        Lz *= (ro*vo)
    
    eELzs = np.array([ecc,E,Lz])
    actions = np.array([jr,jp,jz])
    
    return [orbs_samp,eELzs,actions]

def calc_beta(orbs_locs):
    '''calc_beta:

    Calculate the beta values for a set of DFs, and orbits for locations

    Args:
        orbs_locs (list) - List of lists of orbit.Orbit instances, one for 
            each location with n_samples orbits in each.
    
    Returns:
        betas (np.array) - Array of beta values. shape (n_dfs,n_locs)
    '''
    # Assume these dimensions
    n_dfs = len(orbs_locs)
    n_locs = len(orbs_locs[0])
    betas = np.zeros((n_dfs,n_locs))
    for i in range(n_dfs):
        for j in range(n_locs):
            betas[i,j] = calc_beta_one_loc(orbs_locs[i][j])
    return betas

def calc_beta_one_loc(orbs):
    '''calc_beta_one_loc:

    Calculate the beta value for a set of orbits

    Args:
        orbs (orbit.Orbit instance) - Orbit object representing location 
            to draw samples
    
    Returns:
        beta (float) - Beta value
    '''
    # Calculate the beta value
    vr = orbs.vr()
    vtheta = orbs.vtheta()
    vphi = orbs.vT()

    if isinstance(vr,apu.Quantity):
        vr = vr.value
        vtheta = vtheta.value
        vphi = vphi.value
    
    beta = 1 - ((np.std(vtheta)**2 + np.std(vphi)**2) / (2*np.std(vr)**2))
    return beta

def fit_smooth_spline(x,y,s=0):
    '''fit_smooth_spline:
    
    Fit a smooth spline to KSF purity and completeness data. 
    
    Args:
        x (np.ndarray) - X data
        y (np.ndarray) - Y data
        s (float) - Smoothing factor
        
    Returns:
        smooth_spl () - Smooth spline
    '''
    # Trim both edges of the data if they're 0
    where_y_good = np.ones_like(x,dtype=bool)
    leftInd, rightInd = 0,-1
    while True:
        if x[leftInd] == 0:
            where_y_good[leftInd] = False
            leftInd+=1
        else:
            break
    while True:
        if x[rightInd] == 0:
            where_y_good[rightInd] = False
            rightInd-=1
        else:
            break
    
    smooth_spl = interpolate.UnivariateSpline(x[where_y_good], y[where_y_good], 
                                              k=3, s=s, ext=1)
    return smooth_spl

def fit_linear_spline(x,y):
    '''fit_linear_spline:
    
    Fit a linear spline to KSF purity and completeness data. 
    
    Args:
        x (np.ndarray) - X data
        y (np.ndarray) - Y data
        
    Returns:
        linear_spl () - Linear spline
    '''    
    interpolation_mask = np.isfinite(x) & np.isfinite(y)
    linear_spl = interpolate.interp1d(x[interpolation_mask], 
                                      y[interpolation_mask], kind='linear')
    return linear_spl


def make_completeness_purity_splines(selec_spaces, orbs, eELzs, actions,
    mixture_arr, denspots, halo_selection_dict, phi0, lblocids_pointing, 
    ds_individual, fs, kSF_dir, fig_dir, force_splines=False, force_cp=False, 
    spline_type='linear', make_spline_plots=None, n_spline_plots=50):
    '''make_completeness_purity_splines:
    
    Calculate completeness and purity at all locations for the kinematic 
    data given a selection. Then create completeness-distance and 
    purity-distance splines. orbs, eELzs, and actions should be 2-element lists 
    containing the low beta [0] and high beta [1] samples at each location 
    where the spline is calculated.
    
    Args:
        selec_spaces (string or arr) - Kinematic selection string or array 
            of strings representing a combined selection
        orbs (list of orbit.Orbit instances) - list of size = number of 
            DFs containing list of size = number of locations in the spline 
            fitting grid  containing orbits representing kinematic samples
        eELzs (list of arrays) - list of size = number of 
            DFs containing list of size = number of locations in the spline 
            fitting grid  containing e,E,Lz
        actions (list of arrays) - list of size = number of 
            DFs containing list of size = number of locations in the spline 
            fitting grid  containing JR,Lz,Jz
        mixture_arr () - Array of size = number of DFs containing the fractions 
            of samples used to generate the mixtures for completeness and purity 
            calculations
        denspots (list of galpy Potentials instances) - List of size = number of
            DFs containing the potentials which are the denspots= arguments of
            the DFs. These are evaluated (evaluateDensities) at each location
            to produce weights for the purity calculation (completeness is
            independent of the density)
        halo_selection_dict (dict) - Dictionary containing halo selections
        phi0 (float) - Value of the potential at infinity
        lblocids_pointing (list) - List containing the ls, bs, and locids 
            of each pointing
        ds_individual (array) - List of individual distances used to fit 
            a spline along each pointing
        fs (array) - Array of length number of distance points x number of 
            pointings containing the APOGEE location ID for each point
        kSF_dir (string) - Directory to store the kinematic effective 
            selection function data products
        fig_dir (string) - Directory to store figures
        force_splines (bool) - Force creation of new splines even if old 
            ones exist
        force_cp (bool) - Force calculation of completeness & purity, 
            even if saved values exist
        spline_type (str) - Type of spline to create, either 'linear', 
            'cubic', or 'both' ['linear']
        make_spline_plots (str) - Make plots of completeness splines 
            'completeness', purity splines 'purity' or both 'both'
        n_spline_plots (int) - Number of splines to plot, use 'all' to 
            plot everything
    
    Returns:
        None
    '''
    assert spline_type in ['linear','cubic','both'], 'spline_type must be'+\
        ' either "linear", "cubic", or "both"'
    
    print('\nSelection is: ')
    print(selec_spaces)
    
    # Unpack info for each pointing
    ls_pointing,bs_pointing,locids_pointing = lblocids_pointing
    n_pointing = len(locids_pointing)
    
    # Program is only setup to handle 2 samples as input
    assert len(orbs) == 2, 'Function is only setup to handle 2 samples'
    assert len(eELzs) == 2, 'Function is only setup to handle 2 samples'
    assert len(actions) == 2, 'Function is only setup to handle 2 samples'
    assert len(mixture_arr) == 2, 'Function is only setup to handle 2 samples'
    assert len(denspots) == 2, 'Function is only setup to handle 2 samples'
    
    # Assert that orbs faithfully holds location and sample number info
    n_locs = len(orbs[0])
    n_samples = len(orbs[0][0])
    
    # Determine how weights for purity will be calculated
    for d in denspots:
        assert d is None or isinstance(d,potential.Potential), \
            'denspots[i] must be None or a galpy.potential.Potential instance'
    assert isinstance(mixture_arr,np.ndarray), 'mixture_arr must be a numpy array'
    for m in mixture_arr:
        assert m is None or isinstance(m,(float,int)), \
            'mixture_arr[i] must be None or a number'
    _has_mixture_arr = not (np.all(mixture_arr == 0) or np.all(mixture_arr == None))
    _has_denspot = not (denspots[0] == None or denspots[1] == None)
    if _has_mixture_arr and _has_denspot:
        print('Provided mixture_arr and denspot for purity calculation,'
              ' prioritizing denspot')
        _has_mixture_arr = False

    completeness = np.zeros(n_locs)
    purity = np.zeros(n_locs)
    if isinstance(selec_spaces,str): selec_spaces = [selec_spaces,]
    selec_spaces_suffix = '-'.join(selec_spaces)
    if spline_type in ['linear','both']:
        spline_linear_filename = kSF_dir+'kSF_splines_linear_'+\
            selec_spaces_suffix+'.pkl'
    if spline_type in ['cubic','both']:
        spline_cubic_filename = kSF_dir+'kSF_splines_cubic_'+\
            selec_spaces_suffix+'.pkl'
    
    if make_spline_plots in ['completeness','both']:
        completeness_fig_dir = fig_dir+selec_spaces_suffix+'/completeness/'
        os.makedirs(completeness_fig_dir, exist_ok=True)
    if make_spline_plots in ['purity','both']:
        purity_fig_dir = fig_dir+selec_spaces_suffix+'/purity/'
        os.makedirs(purity_fig_dir, exist_ok=True)
    
    completeness_filename = kSF_dir+'completeness_'+selec_spaces_suffix+'.npy'
    purity_filename = kSF_dir+'purity_'+selec_spaces_suffix+'.npy'
    if not (   os.path.exists(completeness_filename)\
            or os.path.exists(purity_filename)) or force_cp:
        # Calculate purity and completeness at each KSF location
        print('Calculating completeness and purity')
        for i in range(n_locs):
            for j in range(len(selec_spaces)):
                lowbeta_x,lowbeta_y = pplot.get_plottable_data( [orbs[0][i],], 
                    [eELzs[0][i],], [actions[0][i],], 
                    np.atleast_2d(mixture_arr[0]), 
                    selec_spaces[j], phi0=phi0, absolute=True)
                highbeta_x,highbeta_y = pplot.get_plottable_data( [orbs[1][i],], 
                    [eELzs[1][i],], [actions[1][i],], 
                    np.atleast_2d(mixture_arr[1]), 
                    selec_spaces[j], phi0=phi0, absolute=True)
                this_selection = halo_selection_dict[selec_spaces[j]]

                if j == 0:
                    lowbeta_selec = pplot.is_in_scaled_selection(lowbeta_x, 
                            lowbeta_y, this_selection, factor=[1.,1.])
                    highbeta_selec = pplot.is_in_scaled_selection(highbeta_x, 
                            highbeta_y, this_selection, factor=[1.,1.])
                else:
                    lowbeta_selec = lowbeta_selec & pplot.is_in_scaled_selection(
                        lowbeta_x, lowbeta_y, this_selection, factor=[1.,1.])
                    highbeta_selec = highbeta_selec & pplot.is_in_scaled_selection(
                        highbeta_x, highbeta_y, this_selection, factor=[1.,1.])

            # Calculate the purity factor. This will be multiplied to each 
            # number of counts in the selection regions to account for the 
            # variation in density profile / arbitrary mixture
            if _has_mixture_arr:
                n_low_beta_purity = np.sum(lowbeta_selec)*mixture_arr[0]
                n_high_beta_purity = np.sum(highbeta_selec)*mixture_arr[1]
            elif _has_denspot:
                # Make some assertions about the orbits for sanity
                assert np.all(orbs[0][i].R().value == orbs[1][i].R().value[0])
                assert np.all(orbs[0][i].R().value == orbs[1][i].R().value)
                assert np.all(orbs[0][i].z().value == orbs[1][i].z().value) 
                loc_R = orbs[0][i].R()[0]
                loc_z = orbs[0][i].z()[0]
                dens_low_beta_purity = potential.evaluateDensities(
                    denspots[0],loc_R,loc_z,use_physical=False)
                dens_high_beta_purity = potential.evaluateDensities(
                    denspots[1],loc_R,loc_z,use_physical=False)
                n_low_beta_purity = np.sum(lowbeta_selec)*dens_low_beta_purity
                n_high_beta_purity = np.sum(highbeta_selec)*dens_high_beta_purity


            if np.sum(highbeta_selec) == 0:
                completeness[i] = 0
                purity[i] = 0
            else:
                completeness[i] = np.sum(highbeta_selec)/n_samples
                purity[i] = n_high_beta_purity/\
                    (n_high_beta_purity+n_low_beta_purity)
                # purity[i] = np.sum(highbeta_selec)/(np.sum(highbeta_selec)\
                #                                     +np.sum(lowbeta_selec))
        print('Saving purity and completeness to '+purity_filename+' and '+\
              completeness_filename)
        np.save(completeness_filename, completeness, allow_pickle=True)
        np.save(purity_filename, purity, allow_pickle=True)
    else:
        print('Loading purity and completeness from '+purity_filename+' and '+\
              completeness_filename)
        completeness = np.load(completeness_filename, allow_pickle=True)
        purity = np.load(purity_filename, allow_pickle=True)
    
    # Create the purity and completeness splines for each location
    if spline_type in ['linear','both']:
        if not os.path.exists(spline_linear_filename) or force_splines:
            print('Creating linear completeness and purity splines')
            spl_linear_completeness_arr = []
            spl_linear_purity_arr = []

            # Loop over all pointings
            for i in range(n_pointing):
                # Find where elements of the larger pointing-distance grid are 
                # from this location
                where_pointing = np.where(fs == locids_pointing[i])[0]

                # Get spline-fitting data
                spl_xs = np.log10(ds_individual)
                spl_cs = completeness[where_pointing]
                spl_ps = purity[where_pointing]
                spl_completeness = fit_linear_spline(spl_xs, spl_cs)
                spl_purity = fit_linear_spline(spl_xs, spl_ps)
                spl_linear_completeness_arr.append(spl_completeness)
                spl_linear_purity_arr.append(spl_purity)

            # Save splines
            print('Saving splines to '+spline_linear_filename)
            with open(spline_linear_filename,'wb') as f:
                pickle.dump([spl_linear_completeness_arr,spl_linear_purity_arr,
                             locids_pointing],f)
        else:
            # Load splines
            print('Loading splines from '+spline_linear_filename)
            with open(spline_linear_filename,'rb') as f:
                spl_linear_completeness_arr,spl_linear_purity_arr,_ = pickle.load(f)
        
    if spline_type in ['cubic','both']:
        if not os.path.exists(spline_cubic_filename) or force_splines:
            print('Creating cubic completeness and purity splines')
            spl_cubic_completeness_arr = []
            spl_cubic_purity_arr = []

            # Loop over all pointings
            for i in range(n_pointing):
                # Find where elements of the larger pointing-distance grid are 
                # from this location
                where_pointing = np.where(fs == locids_pointing[i])[0]

                # Get spline-fitting data
                spl_xs = np.log10(ds_individual)
                spl_cs = completeness[where_pointing]
                spl_ps = purity[where_pointing]
                spl_s = 0.2
                spl_completeness = fit_smooth_spline(spl_xs, spl_cs, s=spl_s)
                spl_purity = fit_smooth_spline(spl_xs, spl_ps, s=spl_s)
                spl_cubic_completeness_arr.append(spl_completeness)
                spl_cubic_purity_arr.append(spl_purity)

            # Save splines
            print('Saving splines to '+spline_cubic_filename)
            with open(spline_cubic_filename,'wb') as f:
                pickle.dump([spl_cubic_completeness_arr,spl_cubic_purity_arr,
                             locids_pointing],f)
        else:
            # Load splines
            print('Loading splines from '+spline_cubic_filename)
            with open(spline_cubic_filename,'rb') as f:
                spl_cubic_completeness_arr,spl_cubic_purity_arr,_ = pickle.load(f)
        
    if make_spline_plots:
        if n_spline_plots == 'all':
            print('Making all spline plots')
        else:
            print('Making '+str(n_spline_plots)+' spline plots')
        # Some keywoards
        label_fontsize = 8
        mock_xs = np.log10(np.linspace(ds_individual[0],ds_individual[-1],301))

        # Make plots of purity and completeness at each location
        for i in range(n_pointing):
            
            if n_spline_plots == 'all': pass
            elif i+1 > n_spline_plots: continue

            if i+1 == n_pointing:
                print('plotting location '+str(i+1)+'/'+str(n_pointing))
            else:
                print('plotting location '+str(i+1)+'/'+str(n_pointing), end='\r')

            # Get spline data
            where_pointing = np.where(fs == locids_pointing[i])[0]
            spl_xs = np.log10(ds_individual)
            spl_cs = completeness[where_pointing]
            spl_ps = purity[where_pointing]

            # Completeness
            if make_spline_plots in ['completeness','both']:
                fig = None
                ax = None
                if spline_type in ['linear','both']:
                    fig,ax = pplot.plot_completeness_purity_distance_spline(
                        spl_xs,spl_cs,spl_linear_completeness_arr[i],mock_xs,
                        'completeness',spl_color='DodgerBlue')
                if spline_type in ['cubic','both']:
                    fig,ax = pplot.plot_completeness_purity_distance_spline(
                        spl_xs,spl_cs,spl_cubic_completeness_arr[i],mock_xs,
                        'completeness',spl_color='Red',fig=fig,ax=ax)
                ax.axhline(0,color='Black',linestyle='dashed')
                fig.suptitle(r'ID: '+str(locids_pointing[i])+', '\
                             +'$\ell = '+str(round(ls_pointing[i],2))+'^{\circ}, '\
                             +'b = '+str(round(bs_pointing[i],2))+'^{\circ}$, '\
                             +'N = '+str(n_samples),
                             fontsize=label_fontsize)
                fig_title = completeness_fig_dir+'ID-'+str(locids_pointing[i])\
                            +'_l'+str(round(ls_pointing[i],2))\
                            +'_b'+str(round(bs_pointing[i],2))\
                            +'_distance.png'
                fig.savefig(fig_title, dpi=100)
                plt.close(fig)
            
            # Purity
            if make_spline_plots in ['purity','both']:
                fig = None
                ax = None
                if spline_type in ['linear','both']:
                    fig,ax = pplot.plot_completeness_purity_distance_spline(
                        spl_xs,spl_cs,spl_linear_purity_arr[i],mock_xs,
                        'completeness',spl_color='DodgerBlue')
                if spline_type in ['cubic','both']:
                    fig,ax = pplot.plot_completeness_purity_distance_spline(
                        spl_xs,spl_cs,spl_cubic_purity_arr[i],mock_xs,
                        'completeness',spl_color='Red',fig=fig,ax=ax)
                ax.axhline(0,color='Black',linestyle='dashed')
                fig.suptitle(r'ID: '+str(locids_pointing[i])+', '\
                             +'$\ell = '+str(round(ls_pointing[i],2))+'^{\circ}, '\
                             +'b = '+str(round(bs_pointing[i],2))+'^{\circ}$, '\
                             +'N = '+str(n_samples),
                             fontsize=label_fontsize)
                fig_title = purity_fig_dir+'ID-'+str(locids_pointing[i])\
                            +'_l'+str(round(ls_pointing[i],2))\
                            +'_b'+str(round(bs_pointing[i],2))\
                            +'_distance.png'
                fig.savefig(fig_title, dpi=200)
                plt.close(fig)
                

def create_kSF_grid(selec_spaces, apogee_fields, ds, kSF_dir, 
    spline_type='linear', make_purity_grid=True):
    '''apply_kSF_splines_to_effSF:

    Use the kSF spline grid to generate the kSF correction over the distance 
    modulus grid

    Args:
        selec_spaces (list) - List of strings corresponding to the kinematic 
            selection
        apogee_fields (np recarray) APOGEE field information array
        ds (np array) - Distance grid in kpc
        kSF_dir (string) - Kinematic selection function directory where the 
            spline files are stored
        spline_type (str) - Type of spline to create, either 'linear',
            'cubic', or 'both' ['linear']
        make_purity_grid (bool) - Whether to make the purity grid as well as
            the completeness grid [True]

    Returns:
        keffSF_grid (array) - kinematic effective selection function grid with 
            shape [nfields,ndistmods]
    '''
    # Definitions and sanity
    logds = np.log10(ds)
    ndistmods = len(ds)
    nfields = len(apogee_fields)
    
    print('\nUsing '+str(nfields)+' fields')
    print('Using '+str(ndistmods)+' distance moduli')
    
    # Filenames based on kinematic spaces used to select GES stars
    selec_spaces_suffix = '-'.join(selec_spaces)
    if spline_type in ['linear','both']:
        spline_linear_filename = kSF_dir+'kSF_splines_linear_'+\
            selec_spaces_suffix+'.pkl'
        print('\nLoading linear kinematic selection function spline grid from '\
          +spline_linear_filename)
        with open(spline_linear_filename,'rb') as f:
            completeness_linear_splines,purity_linear_splines,spline_locids = \
                pickle.load(f)
    if spline_type in ['cubic','both']:
        spline_cubic_filename = kSF_dir+'kSF_splines_cubic_'+\
            selec_spaces_suffix+'.pkl'
        print('\nLoading cubic kinematic selection function spline grid from '\
          +spline_cubic_filename)
        with open(spline_cubic_filename,'rb') as f:
            completeness_cubic_splines,purity_cubic_splines,spline_locids = \
                pickle.load(f)

    # Create a grid to map the splines onto
    grid_shape = (nfields,ndistmods)
    kSF_grid_cubic = np.zeros(grid_shape)
    kSF_grid_linear = np.zeros(grid_shape)
    purity_grid_cubic = np.zeros(grid_shape)
    purity_grid_linear = np.zeros(grid_shape)

    # Loop over each location and apply the kSF splines to the grid
    for i in range(nfields):
        assert spline_locids[i] == apogee_fields['LOCATION_ID'][i]
        
        if spline_type in ['linear','both']:
            spline_kSF_raw_linear = completeness_linear_splines[i](logds)
            kSF_grid_linear[i,:] = spline_kSF_raw_linear
            if make_purity_grid:
                spline_purity_raw_linear = purity_linear_splines[i](logds)
                purity_grid_linear[i,:] = spline_purity_raw_linear
            
        if spline_type in ['cubic','both']:
            # Make sure the cubic kSF is positive everywhere
            spline_kSF_raw_cubic = completeness_cubic_splines[i](logds)
            spline_kSF_raw_cubic[spline_kSF_raw_cubic < 0] = 0
            kSF_grid_cubic[i,:] = spline_kSF_raw_cubic
            if make_purity_grid:
                spline_purity_raw_cubic = purity_cubic_splines[i](logds)
                purity_grid_cubic[i,:] = spline_purity_raw_cubic
    
    # Save the new grids
    if spline_type in ['linear','both']:
        # kinematic selection function grid
        kSF_filename_linear = kSF_dir+'kSF_grid_linear_'+\
            selec_spaces_suffix+'.dat'
        print('\nSaving kinematic selection function grid to '+\
              kSF_filename_linear)
        with open(kSF_filename_linear,'wb') as f:
            pickle.dump(kSF_grid_linear,f)
        if make_purity_grid:
            purity_filename_linear = kSF_dir+'purity_grid_linear_'+\
                selec_spaces_suffix+'.dat'
            print('\nSaving purity grid to '+\
                  purity_filename_linear)
            with open(purity_filename_linear,'wb') as f:
                pickle.dump(purity_grid_linear,f)
        
    if spline_type in ['cubic','both']:
        # kinematic selection function grid
        kSF_filename_cubic = kSF_dir+'kSF_grid_cubic_'+\
            selec_spaces_suffix+'.dat'
        print('\nSaving kinematic selection function grid to '+\
              kSF_filename_cubic)
        with open(kSF_filename_cubic,'wb') as f:
            pickle.dump(kSF_grid_cubic,f)
        if make_purity_grid:
            purity_filename_cubic = kSF_dir+'purity_grid_cubic_'+\
                selec_spaces_suffix+'.dat'
            print('\nSaving purity grid to '+\
                  purity_filename_cubic)
            with open(purity_filename_cubic,'wb') as f:
                pickle.dump(purity_grid_cubic,f)

# -----------------------------------------------------------------------------

# Harris catalog and globular cluster stuff

def get_harris_catalog( download_catalog=True, filename=None,
    download_url='http://physwww.mcmaster.ca/~harris/mwgc.dat',
    output_filename = './mwgc.FIT', return_catalog=False):
    '''get_harris_catalog:
    
    Read and parse the Harris catalog of globular clusters and return it as an 
    Astropy table object
    
    Args:
        download_catalog (bool) - Download the catalog from the web? [True]
        filename (str) - If not downloading from web then read the filename [None]
        download_url (str) - URL to get the catalog from
        output_filename (str) = Filename to output Harris catalog. If None 
            then don't output catalog.
        return_catalog (bool) = Return the catalog as a variable [False]
        
    Returns:
        table (Astropy Table) - Harris catalog in table format
    '''

    # Get the catalog via download if requested
    if download_catalog:
        print('Downloading '+download_url+' ...')
        filename = './mwgc_temp.dat'
        urllib.request.urlretrieve(download_url, filename)
        
    # Open file
    with open(filename, "r") as fp:
        contents = fp.readlines()
    os.system('rm -f '+filename)

    # Separate into 3 parts
    parts = _separate_harris_catalog(contents)
    
    # Define the columns
    columns2 = ("FE_H", "WT", "E(B-V)", "V_HB", "DISTANCE_MOD_V", "V_T", "M_V",
        "U-B", "B-V", "V-R", "V-I", "SPEC_TYPE", "ELLIPTICITY")
    columns3 = ("V_HELIO", "V_HELIO_ERR", "V_LSR", "SIGMA_V", "SIGMA_V_ERR",
        "CONC", "R_CONC", "R_HALF", "MU_V", "RHO_0", "LOG_T_RELAX_CORE", 
        "LOG_T_RELAX_MEDIAN")

    char_indices2 = [
        (13, 18, float),
        (19, 21, int),
        (24, 28, float),
        (29, 34, float),
        (35, 40, float),
        (41, 46, float),
        (47, 53, float),
        (55, 60, float),
        (61, 66, float),
        (67, 72, float),
        (73, 78, float),
        (79, 85, str),
        (86, 90, float)
    ]
    char_indices3 = [
        (12, 18, float),
        (20, 24, float),
        (25, 32, float),
        (36, 42, float),
        (43, 48, float),
        (49, 53, float),
        (59, 63, float),
        (65, 70, float),
        (72, 77, float),
        (79, 84, float),
        (86, 91, float),
        (92, None, float)
    ]
    
    # Read the first part
    
    # Read each part
    part1 = list(map(_parse_harris_catalog_line_part1, parts[0]))
    part2 = _parse_harris_catalog_part(parts[1], columns2, char_indices2)
    part3 = _parse_harris_catalog_part(parts[2], columns3, char_indices3)

    joined_parts = []
    for i, (cluster_id, part1_data) in enumerate(part1):

        assert cluster_id == part2[i][0]
        assert cluster_id == part3[i][0]

        joined_data = { "ID": cluster_id }
        joined_data.update(part1_data)
        joined_data.update(part2[i][1])
        joined_data.update(part3[i][1])

        joined_parts.append(joined_data)

    tab = table.Table(rows=joined_parts)
    
    if output_filename is not None:
        tab.write(output_filename, overwrite=True)
    if return_catalog:
        return tab


def _separate_harris_catalog(contents):
    '''separate_harris_catalog:
    
    Separate the line-by-line contents of the Harris catalog into the three
    prescribed parts. Each section begins with '   ID' and ends with a blank 
    line.
    
    Args:
        contents (list) List of lines corresponding to the read catalog
        
    Returns:
        parts (arr) Array of 3 sections of the catalog
    '''
    
    # Loop over each line, find the indices where the parts begin and end
    in_part, line_indices = False, []
    for i, line in enumerate(contents):
        
        # If we just saw a '   ID' then ignore the next blank line, which 
        # separates the column headers from the first line of data
        if len(line_indices) > 0 and line_indices[-1] == i + 1:
            continue
        
        # Find the lines that begin and end the data columns
        if line.lstrip().startswith('ID'):
            line_indices.append(i + 2)
            in_part = True # Signal that we are within a data part
        elif in_part and len(line.strip()) == 0:
            line_indices.append(i)
            in_part = False # Signal that we have left a data part

    # line_indices contains start, end indices for each part in a flat list.
    # Now send back separate parts.
    N = int(len(line_indices)/2) # Should always be 3 parts, but whatever.
    parts = []
    for i in range(N):
        start_index, end_index = line_indices[2*i:2*i+2]
        parts.append(contents[ start_index : end_index ])

    return parts


def _parse_harris_catalog_part(part, columns, char_indices):
    '''_parse_harris_catalog_part:
    
    Read part of the Harris catalog
    
    Args:
        part (list) - List of part lines
        columns (list) - List of column names
        char_indices (list) - List of length-3 tuples corresponding to the 
            start, end, and data type of each column of data
    
    Returns:
        
    '''
    # Output list
    parts_out = []
    
    # Loop over each line
    for line in part:
        
        # Get the ID first
        cluster_id = line[:13].strip()
        
        # Get the rest of the data
        data_columns = _parse_harris_catalog_line(line, char_indices)
        
        # zip into a dictionary
        assert len(columns) == len(data_columns)
        data = dict(zip(columns, data_columns))
        
        parts_out.append( (cluster_id,data) )
    
    return parts_out


def _parse_harris_catalog_line(line, char_indices):
    '''_parse_harris_catalog_line:
    
    Args:
        line (string) - Line of the catalog
        char_indices (list) - List of length-3 tuples corresponding to the 
            start, end, and data type of each column of data
        
    Returns:
        data_columns (list) - Output parsed into list
        
    '''

    data_columns = []
    for si, ei, kind in char_indices:
        try:
            _ = kind(line[si:ei].strip())
        except:
            _ = np.nan
        data_columns.append(_)
    return data_columns


def _parse_harris_catalog_line_part1(line):
    '''_parse_harris_catalog_line_part1:
    
    Part 1 of the Harris catalog is special
    
    Args:
        line (str) - Line of part 1
    
    Returns:
        data (list) - Line parsed into cluster id and data
    '''

    columns = ("NAME", "RA", "DEC", "GLON", "GLAT", "R_SUN", "R_GC", "X", "Y", "Z")

    # Get the ID and name first.
    cluster_id, cluster_name = (line[:12].strip(), line[12:25])

    _ = line[25:].split()
    ra, dec = ":".join(_[:3]), ":".join(_[3:6])
    data_columns = [cluster_name.strip(), ra, dec] + list(map(float, _[6:]))

    assert len(columns) == len(data_columns)
    data = dict(zip(columns, data_columns))
    return (cluster_id, data)


def get_globular_cluster_fields():
    '''get_globular_cluster_fields:
    
    Return a list of APOGEE LOCATIONS_IDs for fields with globular cluster 
    stars in them. These should not be considered for modelling.
    
    Args:
        None
    
    Returns:
        gc_locid (list) - List of APOGEE LOCATION_IDs for fields with globular 
            cluster stars in them
    '''
    gc_locid = [2011, 2247, 4260, 5295, 5299, 5300, 5328, 5329, 5801]
    # gc_locid = [2011,4353,5093,5229,5294,5295,5296,5297,5298,5299,5300,5325,
    #             5328,5329,5438,5528,5529,5744,5801]
    return gc_locid