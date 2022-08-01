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
import urllib

from . import plot as pplot
from . import util as putil

# ----------------------------------------------------------------------------

# Define the function that will take all input and calculate results in parallelized manner
def calc_kinematics_parallel(ncores,this_df,n_samples,orbs_locs,do_perturb_orbs,
                             gaia_input,allstar_input,deltas,aAS,ro,vo,zo):
    '''calc_kinematics_parallel:
    
    Calculate the kinemaitcs according to a DF. Returns 
    orbs, eELz, accs
    
    Args:
        ncores
        this_df
        n_samples
        orbs_locs
        do_perturb_orbs
        gaia_input
        allstar_input
        deltas
        aAS
        ro
        vo
        zo
    
    Returns:
        results (array) - Array of [orbs,eELz,accs]
    '''
     
    lambda_func = (lambda x: calc_kinematics_on_loc(this_df,n_samples,
        orbs_locs[x], do_perturb_orbs,gaia_input[[x,]], allstar_input[[x,]],
        deltas[x], aAS, ro, vo, zo))
    
    n_calls = len(orbs_locs)
    print('Using '+str(ncores)+' cores')
    results = (galpy_multi.parallel_map(lambda_func, 
               np.arange(0,n_calls,1,dtype='int'),  
               numcores=ncores))
    
    # By wrapping in numpy array results can be querried as 
    # results[:,0]: array of n_calls orbits, each orbit n_samples long
    # results[:,1]: array of n_calls eELzs, each is (3,1000) of e,E,Lz
    # results[:,2]: array of n_calls actions, each is (3,1000) of JR,Lz,Jz
    return np.array(results,dtype='object')
#def

def calc_kinematics_one_loc(df,n_samples,orbs_locs,do_perturb_orbs,gaia_input,
                    allstar_input,delta,aAS,ro,vo,zo):
    '''calc_kinematics_one_loc:
    
    Calculate the kinemaics for a single location
    
    Args:
        df
        n_samples
        orbs_locs
        do_perturb_orbs
        gaia_input
        allstar_input
        delta
        aAS
        ro
        vo
        zo
    
    Returns:
        results (arr) - Array of orbs, eELzs, actions
    '''
    
    # Timing?

    orbs_samp = df.sample(n=n_samples,R=np.ones_like(n_samples)*orbs_locs.R(),
                                      phi=np.ones_like(n_samples)*orbs_locs.phi(),
                                      z=np.ones_like(n_samples)*orbs_locs.z())

    # Resample orbits based on positional matches
    if do_perturb_orbs:
        orbs_samp = putil.perturb_orbit_with_Gaia_APOGEE_uncertainties(orbs_samp,
            gaia_input,allstar_input,only_velocities=True,ro=ro,vo=vo,zo=zo)
    ##fi

    ecc,_,_,_ = aAS.EccZmaxRperiRap(orbs_samp, delta=delta, 
                                    use_physical=True, c=True)
    accs = aAS(orbs_samp, delta=delta, c=True)
    E = orbs_samp.E(pot=mwpot)
    Lz = orbs_samp.Lz(pot=mwpot)

    try:
        ecc = ecc.value
    except AttributeError:
        pass
    try:
        jr = accs[0].value
        Lz = accs[1].value
        jz = accs[2].value
    except AttributeError:
        jr = accs[0]
        Lz = accs[1]
        jz = accs[2]
    try:
        E = E.value
        Lz = Lz.value
    except AttributeError:
        pass
    ##te
    
    eELzs = np.array([ecc,E,Lz])
    actions = np.array([jr,Lz,jz])
    
    return [orbs_samp,eELzs,actions]
#def  

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
        ##ie
    ##wh
    while True:
        if x[rightInd] == 0:
            where_y_good[rightInd] = False
            rightInd-=1
        else:
            break
        ##ie
    ##wh
    
    smooth_spl = interpolate.UnivariateSpline(x[where_y_good], y[where_y_good], 
                                              k=3, s=s, ext=1)
    return smooth_spl
#def

def make_completeness_purity_splines(selec_spaces, orbs, eELzs, actions, 
    halo_selection_dict, phi0,
    lblocids_pointing, ds_individual, fs, out_dir, gaia_apogee_dir, fig_dir, 
    force_splines=False, make_spline_plots=False,n_spline_plots=50):
    '''make_completeness_purity_splines:
    
    Calculate completeness and purity at all locations for the kinematic 
    data given a selection. Then create completeness-distance and 
    purity-distance splines. 
    
    Args:
    
    Returns:
        None
    '''
    print('\nSelection is: ')
    print(selec_spaces)
    
    # Unpack
    ls_pointing,bs_pointing,locids_pointing = lblocids_pointing
    n_pointing = len(locids_pointing)
    
    # Assert that orbs faithfully holds location and sample number info
    n_locs = len(orbs[0])
    n_samples = len(orbs[0][0])
    
    completeness = np.zeros(n_locs)
    purity = np.zeros(n_locs)
    selec_spaces_suffix = '-'.join(selec_spaces)
    spline_filename = out_dir+gaia_apogee_dir+'ksf_splines_'\
                      +selec_spaces_suffix+'.pkl'
    
    if make_spline_plots in ['completeness','both']:
        completeness_fig_dir = fig_dir+selec_spaces_suffix+'/completeness/'
        os.makedirs(completeness_fig_dir, exist_ok=True)
    if make_spline_plots in ['purity','both']:
        purity_fig_dir = fig_dir+selec_spaces_suffix+'/purity/'
        os.makedirs(purity_fig_dir, exist_ok=True)
    ##fi
    
    # Calculate purity and completeness at each KSF location
    print('Calculating completeness and purity')
    for i in range(n_locs):
        for j in range(len(selec_spaces)):
            lowbeta_x,lowbeta_y = pplot.get_plottable_data( [orbs[0][i],], 
                [eELzs[0][i],], [actions[0][i],], np.array([1,]), selec_spaces[j], 
                phi0=phi0, absolute=True)
            highbeta_x,highbeta_y = pplot.get_plottable_data( [orbs[1][i],], 
                [eELzs[1][i],], [actions[1][i],], np.array([1,]), selec_spaces[j], 
                phi0=phi0, absolute=True)
            this_selection = halo_selection_dict[selec_spaces[j]]

            if j == 0:
                lowbeta_selec = pplot.is_in_scaled_selection(lowbeta_x, lowbeta_y, 
                        this_selection, factor=[1.,1.])
                highbeta_selec = pplot.is_in_scaled_selection(highbeta_x, highbeta_y, 
                        this_selection, factor=[1.,1.])
            else:
                lowbeta_selec = lowbeta_selec & pplot.is_in_scaled_selection(
                    lowbeta_x, lowbeta_y, this_selection, factor=[1.,1.])
                highbeta_selec = highbeta_selec & pplot.is_in_scaled_selection(
                    highbeta_x, highbeta_y, this_selection, factor=[1.,1.])
            ##ie
        ###j
        if np.sum(highbeta_selec) == 0:
            purity[i] = 0
        else:
            completeness[i] = np.sum(highbeta_selec)/n_samples
            purity[i] = np.sum(highbeta_selec)/(np.sum(highbeta_selec)\
                                                +np.sum(lowbeta_selec))
        ##ie
    ###i
    
    # Create the purity and completeness splines for each location
    if not os.path.exists(spline_filename) or force_splines:
        print('Creating completeness-distance and purity-distance splines')
        spl_completeness_arr = []
        spl_purity_arr = []

        # Loop over all pointings
        for i in range(n_pointing):
            # Find where elements of the larger pointing-distance grid are from 
            # this location
            where_pointing = np.where(fs == locids_pointing[i])[0]

            # Get spline-fitting data
            spl_xs = np.log10(ds_individual)
            spl_cs = completeness[where_pointing]
            spl_ps = purity[where_pointing]
            spl_s = 0.2
            spl_completeness = fit_smooth_spline(spl_xs, spl_cs,s=spl_s)
            spl_purity = fit_smooth_spline(spl_xs, spl_ps,s=spl_s)
            spl_completeness_arr.append(spl_completeness)
            spl_purity_arr.append(spl_purity)
        ###i
        # Save splines
        print('Saving splines to '+spline_filename)
        with open(spline_filename,'wb') as f:
            pickle.dump([spl_completeness_arr,spl_purity_arr,locids_pointing],f)
        ##wi
    else:
        # Load splines
        print('Loading splines from '+spline_filename)
        with open(spline_filename,'rb') as f:
            spl_completeness_arr,spl_purity_arr,_ = pickle.load(f)
        ##wi
    ##ie
    
    if make_spline_plots:
        if n_spline_plots == None:
            print('Making all spline plots')
        else:
            print('Making '+str(n_spline_plots)+' spline plots')
        ##ie
        # Some keywoards
        label_fontsize = 8
        mock_xs = np.log10(np.linspace(ds_individual[0],ds_individual[-1],301))

        # Make plots of purity and completeness at each location
        for i in range(n_pointing):
            
            if n_spline_plots == None: pass
            elif i+1 > n_spline_plots: continue

            if i+1 == n_pointing:
                print('plotting location '+str(i+1)+'/'+str(n_pointing))
            else:
                print('plotting location '+str(i+1)+'/'+str(n_pointing), end='\r')
            ##ie

            # Get spline data
            where_pointing = np.where(fs == locids_pointing[i])[0]
            spl_xs = np.log10(ds_individual)
            spl_cs = completeness[where_pointing]
            spl_ps = purity[where_pointing]

            # Completeness
            if make_spline_plots in ['completeness','both']:
                fig,ax = pplot.plot_completeness_purity_distance_spline(spl_xs,
                    spl_cs,spl_completeness_arr[i],mock_xs,'completeness')
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
            ##fi
            
            # Purity
            if make_spline_plots in ['purity','both']:
                fig,ax = pplot.plot_completeness_purity_distance_spline(spl_xs,
                    spl_ps,spl_purity_arr[i],mock_xs,'purity')
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
        ###i
    ##fi
#def

def apply_kSF_splines_to_effSF(selec_spaces,effSF_grid,apogee_fields,ds,kSF_dir,
    fig_dir,ro,vo,zo,make_SF_plots=False,denspot=None):
    '''apply_kSF_splines_to_effSF:

    Use the kSF spline grid to generate the kSF correction over the distance 
    modulus grid and apply it to the effective selection function grid.

    Args:
        selec_spaces (list) - List of selection spaces corresponding to the 
        effSF_grid (np array) - Effective selection function grid with shape 
            [nfields,ndistmods]
        apogee_fields (np recarray) apogee field information array
        ds (np array) - Distance grid
        kSF_dir (string) - Kinematic selection function directory where the 
            spline files are stored
        fig_dir (string) - Place to store figures
        make_sf_plots (bool) - Make plots of the SFs
        denspot (galpy.potential.Potential) - Density profile to weight 
            SFs along LOS for proper visualization

    Returns:
        None
    '''
    # Definitions and sanity
    logds = np.log10(ds)
    ndistmods = len(ds)
    nfields = len(apogee_fields)
    assert effSF_grid.shape == (nfields,ndistmods), 'insane :('
    
    # Filenames based on kinematic spaces used to select GES stars
    selec_spaces_suffix = '-'.join(selec_spaces)
    if make_SF_plots:
        os.makedirs(fig_dir+selec_spaces_suffix,exist_ok=True)
    ##fi
    kSF_spline_filename = kSF_dir+'ksf_splines_'+selec_spaces_suffix+'.pkl'
    print('Loading kinematic selection function spline grid from '\
          +kSF_spline_filename)
    with open(kSF_spline_filename,'rb') as f:
        completeness_splines,_,spline_locids = pickle.load(f)
    ##wi

    # Create a grid to map the splines onto
    kSF_grid = np.zeros_like(effSF_grid)

    # Loop over each location and apply the kSF splines to the grid
    for i in range(nfields):
        assert spline_locids[i] == apogee_fields['LOCATION_ID'][i]

        # Make sure the kSF is positive everywhere
        spline_kSF_raw = completeness_splines[i](logds)
        spline_kSF_raw[spline_kSF_raw < 0] = 0
        kSF_grid[i,:] = spline_kSF_raw
    ###i
    
    # Apply the kinematic selection function to the effective selection function
    keffSF_grid = effSF_grid*kSF_grid
    
    kSF_filename = kSF_dir+'ksf_grid_'+selec_spaces_suffix+'.dat'
    print('Saving kinematic selection function grid to '+kSF_filename)
    with open(kSF_filename,'wb') as f:
        pickle.dump(kSF_grid,f)
        
    keffSF_filename = kSF_dir+'apogee_keffSF_grid_inclArea_'\
                             +selec_spaces_suffix+'.dat'
    print('Saving kinematic effective selection function to '+keffSF_filename)
    with open(keffSF_filename,'wb') as f:
        pickle.dump(keffSF_grid,f)
    
    # Make plots 
    if make_SF_plots:
        
        # Create a large linear grid of orbits representing positions where 
        # the SF is evaluated, corresponding to SF.flatten()
        orbs_grid = putil.make_SF_grid_orbits(apogee_fields,ds,ro,vo,zo,
            fudge_ll_instability=True)
        
        # Weight selection functions by stellar density for proper viewing
        effSF_grid_weighted = np.zeros(nfields)
        kSF_grid_weighted = np.zeros(nfields)
        keffSF_grid_weighted = np.zeros(nfields)
        
        for i in range(nfields):
            ll = apogee_fields['GLON'][i].repeat(ndistmods)
            bb = apogee_fields['GLAT'][i].repeat(ndistmods)
            field_vxvv = np.zeros((ndistmods,6))
            field_vxvv[:,0] = ll
            field_vxvv[:,1] = bb
            field_vxvv[:,2] = ds
            field_orbs = orbit.Orbit(vxvv=field_vxvv,lb=True)
            dens_weights = potential.evaluateDensities(denspot,field_orbs.R(),
                field_orbs.z())

            # Weight the SF along the LOS by the density of the stellar halo
            effSF_grid_weighted[i] = np.average(effSF_grid[i],
                weights=dens_weights)
            kSF_grid_weighted[i] = np.average(kSF_grid[i],
                weights=dens_weights)
            keffSF_grid_weighted[i] = np.average(keffSF_grid[i],
                weights=dens_weights)
        
        fig = plt.figure(figsize=(15,5))
        axs = fig.subplots(nrows=1,ncols=3)
        
        SF_plot_arr = [np.log10(effSF_grid_weighted),
                       kSF_grid_weighted,
                       np.log10(keffSF_grid_weighted)]
        labels = ['eff. SF','kin. SF',r'kin. SF $\times$ eff. SF']
        # SF_grid_arr = [effSF_grid,kSF_grid,keffSF_grid]
        
        for i in range(3):
            
            vmin,vmax = -3,1
            if i == 1:
                vmin,vmax = 0.,1.
            ##fi
            glon_plot = copy.deepcopy(apogee_fields['GLON'])
            glon_plot[glon_plot>180] = glon_plot[glon_plot>180]-360
            pts = axs[i].scatter(glon_plot, apogee_fields['GLAT'], 
                                 c=SF_plot_arr[i], vmin=vmin, vmax=vmax)
            axs[i].set_xlabel('longitude [deg]')
            axs[i].set_ylabel('latitude [deg]')
            axs[i].set_xlim(180,-180)
            axs[i].set_ylim(-90,90)
            axs[i].set_xticks([180,90,0,-90,-180])
            axs[i].set_yticks([-90,-45,0,45,90])
            # axs[i].suptitle(labels[i],fontsize=12)
            cbar = fig.colorbar(pts, ax=axs[i], orientation='horizontal')
            cbar.set_label(labels[i])
        
        fig.savefig(fig_dir+selec_spaces_suffix+'/SF_map.pdf')
        plt.close(fig)
    ##fi
    return keffSF_grid
#def

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
    gc_locid = [2011,4353,5093,5229,5294,5295,5296,5297,5298,5299,5300,5325,
                5328,5329,5438,5528,5529,5744,5801]
    return gc_locid