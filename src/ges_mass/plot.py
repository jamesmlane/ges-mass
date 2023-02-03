# ----------------------------------------------------------------------------
#
# TITLE - plot.py
# AUTHOR - James Lane
# PROJECT - ges-mass
# CONTENTS:
#
# ----------------------------------------------------------------------------

### Docstrings and metadata:
'''Functions to deal with plotting and kinematic spaces
'''
__author__ = "James Lane"

### Imports
import numpy as np
import numbers
import copy
import dill as pickle
import os
from astropy import units as apu
from galpy import orbit
import matplotlib
from matplotlib import patches
from matplotlib import pyplot as plt
import corner
from scipy.special import erf

from . import util as putil
from . import mass as pmass
from . import densprofiles as pdens

_MEDIAN_MASS_FOR_1E8 = 1e5

# ----------------------------------------------------------------------------

# Plotting density profiles

def get_plotting_data(hf, return_dirs=False):
    '''get_plotting_data:
    
    A wrapper function to get data so that all code cells are consistent
    
    Args:
        hf (HaloFit) - HaloFit class
        return_dirs (bool) - Return directories
    
    Returns:
        out (list) - List of relevant data and variables for plotting:
            samples (array) - MCMC output samples, shape is (nit,ndim)
            opt (list) - scipy.optimize output
            sampler () - emcee sampler object
            masses (array) - Masses
            facs (array) - Mass factors
            samples_mass (array) - Subset of samples used to generated masses, 
                only not None when masses < samples
    '''
    samples = np.load(hf.fit_data_dir+'samples.npy')
    masses = np.load(hf.fit_data_dir+'masses.npy')
    facs = np.load(hf.fit_data_dir+'facs.npy')
    samples_mass = np.load(hf.fit_data_dir+'mass_sample_inds.npy')
    with open(hf.fit_data_dir+'opt.pkl','rb') as f:
        opt = pickle.load(f)
    if os.path.exists(hf.fit_data_dir+'sampler.pkl'):
        with open(hf.fit_data_dir+'sampler.pkl','rb') as f:
            sampler = pickle.load(f)
    else:
        sampler = None
    
    out = [samples,opt,sampler,masses,facs,samples_mass]
    if return_dirs:
        out = out+[hf.fit_data_dir,hf.fit_fig_dir]
    
    return out


def xyz_to_Rphiz(x,y,z):
    '''xyz_to_Rphiz:
    
    Calculate galactocentric R,phi,z from X,Y,Z
    
    Args:
        x,y,z (np.array) - galactocentric rectangular coordinates
    
    Returns:
        R,phi,z (np.array) - galactocentric cylindrical coordinates
    '''
    R = (x**2.+y**2.)**0.5
    phi = np.arctan2(y,x)
    return R,phi,z


def Rphiz_to_xyz(R,phi,z):
    '''Rphi_to_xyz:
    
    Calculate galactocentric X,Y,Z from R,phi,z
    
    Args:
        R,phi,z (np.array) - galactocentric cylindrical coordinates
    
    Returns:
        x,y,z (np.array) - galactocentric rectangular coordinates
    '''
    x = R*np.cos(phi)
    y = R*np.sin(phi)
    return x,y,z


def plot_corner(hf,samples=None,plot_mass=False,thin=None,thin_to=None,
                quantiles=[0.16,0.5,0.84],show_titles=True,corner_kwargs={}, 
                truths='None'):
    '''plot_corner:
    
    Plot posterior samples
    
    Args:
        hf (HaloFit) - HaloFit class containing all 
        samples (array) - MCMC samples, shape is (nsample,ndim). If None then 
            samples will be hf.samples [default None]
        plot_mass (bool) - Also plot masses? If so len(masses) must be 
            n_samples [default False]
        thin (int) - Factor to thin by, so will plot samples[::thin]
        thin_to (int) - Number of samples that should be plotted, all others 
            will be thinned to achieve this [default None]
        truths (str) - Add maximum likelihood truths. Must be string supplied 
            to hf.get_ml_params() [default None]
        corner_kwargs (dict) - Dictionary of keywords to pass to corner.corner()
            [default {}]
    
    Returns:
        fig (pyplot.Figure) - Figure with all samples
    '''
    # Copy kwargs so not overwriting
    _corner_kwargs = copy.deepcopy(corner_kwargs)
    
    # Get mcmc labels, denormalize parameters
    mcmc_labels = pdens.get_densfunc_mcmc_labels(hf.densfunc, 
                                                 physical_units=True)
    if samples is None:
        samples = hf.samples
    samples = pdens.denormalize_parameters(samples, hf.densfunc)
    
    # Including mass or not?
    if plot_mass:
        masses = np.masses
        if np.median(masses)>_MEDIAN_MASS_FOR_1E8:
            masses /= 1e8
            mcmc_labels.append(r'M $[10^{8} \textrm{M}_{\odot}]$')
        else:
            mcmc_labels.append(r'M $[\textrm{M}_{\odot}]$')
            
        if len(masses) == samples.shape[0]:
            samples = np.concatenate((samples,np.atleast_2d(masses).T),axis=1)
        if len(masses) < samples.shape[0]:
            assert hf.mass_inds is not None, 'if number of masses less'+\
                ' than number of samples, hf.mass_inds must be not None'
            samples = samples[hf.mass_inds]
            samples = np.concatenate((samples,np.atleast_2d(masses).T),axis=1)
    
    # Thinning if required
    n_samples = samples.shape[0]
    if thin is not None:
        thin = int(thin)
        print('thinning by factor '+str(thin))
        samples = samples[::thin,:]
    elif thin_to is not None:
        thin = np.floor(n_samples/thin_to).astype(int)
        print('thinning to N='+str(thin_to)+', thinning by factor '+str(thin))
        samples = samples[::thin,:]
    
    # Include truths
    valid_truths = ['mcmc_ml','mcmc_median','post','init','mock_truths']
    if (truths is not None) and (truths in valid_truths):
        if truths == 'mock_truths':
            if hf.truths is None:
                print('mock truths requested, but hf.truths is None')                
            truth_values = np.ravel(
                pdens.denormalize_parameters(hf.truths,hf.densfunc))
        else:
            truth_values = hf.get_ml_params(truths)
        if plot_mass: # Need to account for ml_ind for mass and other truths
            print('Truths for mass not yet implemented')
            truth_values.append(None)
        if 'truths' not in _corner_kwargs.keys():
            _corner_kwargs.update({'truths':truth_values})
        else:
            print('Not including truths in _corner_kwargs, key already present')
    
    # Plot
    fig = corner.corner(samples, quantiles=quantiles, labels=mcmc_labels, 
                        show_titles=show_titles, **_corner_kwargs)
    return fig


def plot_masses(hf,mass_in_log=True,quantiles=[0.16,0.5,0.84],show_titles=True,
                truths='None',corner_kwargs={}):
    '''plot_masses:
    
    Args:
        hf (HaloFit) - HaloFit class containing all information
        m_ten_eight
    
    Returns:
        fig (matplotlib Figure object)
        axs (matplotlib Axis object)
    '''
    # Copy kwargs so not overwriting
    _corner_kwargs = copy.deepcopy(corner_kwargs)
    
    masses = hf.masses
    if mass_in_log:
        if np.any(~np.isfinite(masses)):
            masses[~np.isfinite(masses)] = np.median(masses)
        masses = np.log10(masses)
        labels = [r'$\log_{10}(\mathrm{M}/\mathrm{M}_{\odot})$']
    else:
        if np.median(masses)>_MEDIAN_MASS_FOR_1E8:
            masses /= 1e8
            labels = [r'M $[10^{8} \textrm{M}_{\odot}]$']
        else:
            labels = [r'M $\textrm{M}_{\odot}$']
          
    # Include truths
    valid_truths = ['mock_truths']
    if (truths is not None) and (truths in valid_truths):
        if truths == 'mock_truths':
            if hf.truths is None:
                print('mock truths requested, but hf.truth_mass is None')
            truth_mass = np.array([hf.truth_mass])
            if mass_in_log:
                truth_mass = np.log10(truth_mass)
        if 'truths' not in _corner_kwargs.keys():
            _corner_kwargs.update({'truths':truth_mass})
        else:
            print('Not including truths in _corner_kwargs, key already present')
    
    # Plot
    fig = corner.corner(np.atleast_2d(masses).T, quantiles=quantiles, 
                        labels=labels, show_titles=show_titles, **_corner_kwargs)
    return fig


def plot_density_xyz(hf,params=None,n=100,scale=20.,contour=False,imshow_kwargs={},
                     contour_kwargs={},fig=None,axs=None):
    '''plot_density_xyz:
    
    Plot XYZ plane views of a density profile, can be contour
    
    Args:
        hf (HaloFit) - HaloFit class containing all information
        params (list) - List of parameters for the density function
        n (int) - Number of bins in each dimension [default 100]
        scale (float) - Size of the box, units depend on model and params 
            [default 20]
        contour (bool) - plot as contour instead of image [default False]
        imshow_kwargs (dict) - keywords to supply to imshow if contour=False. 
            Will be supplied with some default keywords if is {} [default {}]
        contour_kwargs (dict) - keywords to supply to contour if contour=True.
            Will be supplied with some default keywords if is {} [default {}]
        fig (matplotlib Figure instance) - Figure, if None will make one
        axs (matplotlib Axes instance) - Axes, must be 3, if None will make one
    
    Returns:
        fig (matplotlib Figure object)
    '''
    # Default kwargs
    if imshow_kwargs == {}:
        contour_kwargs = {'cmap':'rainbow', 'vmin':-1, 'vmax':3, 
                          'extent':(-scale,scale,-scale,scale), 
                          'origin':'lower'}
    if contour_kwargs == {}:
        imshow_kwargs = {'levels':[0.,0.5,1.,1.5,2.,2.5,3.], 
                         'extent':(-scale,scale,-scale,scale),
                         'origin':'lower'}
    
    # Default params to a best-fit value
    if params is None:
        print('params not supplied, trying post optimization result')
        if not hf._hasResults:
            hf.get_results()
        try:
            params = hf.get_ml_params('post')
        except AssertionError:
            print('Could not access post optimization result, trying MCMC ML')
            try:
                params = hf.get_ml_params('mcmc_ml')
            except AssertionError:
                print('Could not access MCMC ML, trying MCMC median')
                try:
                    params = hf.get_ml_params('mcmc_median')
                except AssertionError:
                    raise RuntimeError('Could not access params')
    
    # Index strings
    ind_arr = [[0,1],[0,2],[1,2]]
    ind_str = [['X','Y'],['X','Z'],['Y','Z']]
    
    # tick_locs = np.arange(-10,12.5,2.5)
    if fig is None or axs is None:
        fig = plt.figure(figsize=(15,5))
        axs = fig.subplots(nrows=1,ncols=3)
    
    # Make the X,Y,Z grid and R,phi,z grid
    xs = np.linspace(-scale,scale,endpoint=False,num=n)+(scale/n)
    ys = np.linspace(-scale,scale,endpoint=False,num=n)+(scale/n)
    zs = np.linspace(-scale,scale,endpoint=False,num=n)+(scale/n)
    
    for i in range(3):
        # Grid
        if i == 0: # XY plane
            c1grid,c2grid = np.meshgrid(xs,ys)
            zgrid = np.zeros_like(c1grid)
            Rgrid,phigrid,zgrid = xyz_to_Rphiz(c1grid,c2grid,zgrid)
        if i == 1: # XZ plane
            c1grid,c2grid = np.meshgrid(xs,zs)
            ygrid = np.zeros_like(c1grid)
            Rgrid,phigrid,zgrid = xyz_to_Rphiz(c1grid,ygrid,c2grid)
        if i == 2: # YZ plane
            c1grid,c2grid = np.meshgrid(ys,zs)
            xgrid = np.zeros_like(c1grid)
            Rgrid,phigrid,zgrid = xyz_to_Rphiz(xgrid,c1grid,c2grid)
    
        # Calculate densities
        dgrid = hf.densfunc(Rgrid,phigrid,zgrid,params)
        
        if contour:
            axs[i].contour(c1grid,c2grid,np.log10(dgrid), **contour_kwargs)
        else:
            axs[i].imshow(np.log10(dgrid), **imshow_kwargs)
        axs[i].set_xlabel(ind_str[i][0]+' [kpc]')
        axs[i].set_ylabel(ind_str[i][1]+' [kpc]')

    fig.tight_layout()
    fig.show()
    
    return fig


def plot_density_re(hf, re_range=[0.1,100], nre=100, nrand=None, 
                    plot_kwargs={}):
    '''plot_density_re:
    
    Make a figure of density vs effective radius
    
    Args:
        hf (HaloFit) - HaloFit class containing all information
        re_range (list) - Range of effective radius to plot [default 0.1 to 100]
        nre (int) - Number of effective radii to consider [default 100]
        nrand (int) - Number of samples to randomly select [default None]
        plot_kwargs (dict) - kwargs to send to plt.plot [default {}]
        
    Returns:
        fig (matplotlib Figure object)
        axs (matplotlib Axis object)
    '''
    # Figure & axis
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    # Effective radius grid
    re = np.logspace(np.log10(re_range[0]), np.log10(re_range[1]), nre)
    _phi = np.zeros_like(re)
    _z = np.zeros_like(re)
    log_re = np.log10(re)
    
    # Get the indices of the p and q parameters
    params = ['p','q']
    indx = pdens.get_densfunc_params_indx(hf.densfunc,params)
    
    # Determine samples for which to calculate density
    if nrand is not None:
        nrand = int(nrand)
        rnp = np.random.default_rng()
        randind = rnp.integers(0,hf.samples.shape[0],size=nrand)
        samples_in = hf.samples[randind]
    else:
        samples_in = hf.samples[:]
    
    # Calculate density and plot
    for i in range(len(samples_in)):
        _sample = samples_in[i]
        _sample[indx] = [1.,1.]
        dens = hf.densfunc(re,_phi,_z,params=_sample)
        ax.plot(log_re, np.log10(dens), **plot_kwargs)
    
    ax.set_xlabel(r'$\log_{10} (r_{e})$ [kpc]')
    ax.set_ylabel(r'$\log_{10} (\nu)$')
    
    return fig

def plot_distmod_posterior(hf, pd=None, nrand=None, posterior_type='lines', 
    fill_quantiles=[0.16,0.5,0.84], hist_kwargs=None, lines_kwargs=None, 
    fill_kwargs=None):
    '''plot_distmod_posterior:
    
    Plot the distance modulus posterior of the data vs the model
    
    Args:
        hf (HaloFit) - HaloFit class containing all information
        pd (array) - Supplied posterior distribution for models to plot, if 
            None then will be calculated. Should be shape (nsamples,ndmod)
            [default None]
        nrand (int) - Number of samples to get posteriors for, randomly. If None 
            then don't do random subsample, do all samples. [default None]
        posterior_type (string) - How to plot posteriors, either 'lines' so each 
            sample gets it's own line, or 'fill' so the median is between 
            two quantiles (see fill_quantiles) [default 'lines']
        fill_quantiles (array) - 3-element array describing quantiles if 
            posterior_type='fill', [default (0.16,0.5,0.84)]
        hist_kwargs (dict) - Dict of kwargs passed to ax.hist for data plotting
            [default {}]
        lines_kwargs (dict) - Dict of kwargs passed to ax.plot for 'lines'
            posterior plotting [default {}]
        fill_kwargs (dict) - Dict of kwargs passed to ax.fill_between for 
            'fill' posterior plotting [default {}]
        verbose (bool) - Be verbose?
        ro,vo (float) - galpy ro,vo scales [default 8,220]

    Returns:
        fig,ax
    '''
    if hist_kwargs is None:
        hist_kwargs = {'color':'Black'}
    if fill_kwargs is None:
        fill_kwargs = {'alpha':0.5,'color':'DarkOrange'}
    if lines_kwargs is None:
        lines_kwargs = {'color':'DarkOrange','linewidth':2.}
    
    # Checks
    assert posterior_type in ['lines','fill']
    
    # Safety valve for nrand
    if nrand is None and hf.samples.shape[0] > 100:
        nrand = 100
    
    # Calculate posterior
    if pd is None:
        if hf.verbose:
            print('Calculating p(distmod|model)..')
        if nrand is not None:
            nrand = int(nrand)
            rnp = np.random.default_rng()
            randind = rnp.integers(0,hf.samples.shape[0],size=nrand)
            samples_in = hf.samples[randind]
        else:
            samples_in = hf.samples[:]
        Rgrid,phigrid,zgrid = hf.get_effsel_list()
        pd,pdt,rate = pmass.pdistmod_sample(hf.densfunc, samples_in, 
            hf.get_fit_effsel(), hf.Rgrid, hf.phigrid, hf.zgrid, hf.dmods, 
            return_rate=True, verbose=hf.verbose)
    else:
        assert pd.shape[1] == len(hf.dmods)
    
    if hf.verbose:
        print('plotting..')
        
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Data histogram
    #zero_arr = np.zeros_like(hf.Rdata)
    #vxvv = np.array([hf.Rdata/hf.ro,zero_arr,zero_arr,hf.zdata/hf.ro,
    #                 zero_arr,hf.phidata]).T
    #orbs = orbit.Orbit(vxvv=vxvv, ro=hf.ro, vo=hf.vo)
    orbs = hf.orbs
    dm = 5*np.log10(orbs.dist(use_physical=True).value)+10
    ax.hist(dm, histtype='step', zorder=2, range=(np.min(hf.dmods),
            np.max(hf.dmods)), density=True, **hist_kwargs)

    if posterior_type == 'lines':
        for i in range(len(pd)):
            ax.plot(hf.dmods, pd[i], zorder=3, **lines_kwargs)

    if posterior_type == 'fill':
        lqt_pd, mqt_pd, uqt_pd = np.quantile(pd, fill_quantiles, axis=0)
        ax.fill_between(hf.dmods, lqt_pd, uqt_pd, zorder=3, **fill_kwargs)
        ax.plot(hf.dmods, mqt_pd, zorder=4, **lines_kwargs)

    ax.set_xlabel('Distance Modulus')
    ax.set_ylabel('Density')

    return fig 

# ----------------------------------------------------------------------------

# Plotting utilities for kinematic spaces

def add_diamond_boundary(ax,dedge=1.2,zorder=7,draw_tick_fac=50):
    '''add_diamond_boundary:
    
    Add the diamond-shaped boundary and other things to the action diamond
    panels.
    
    Args:
        ax (matplotlib axis object) - Axis
        dedge (float) - how far to draw the blank patch polygon 
    
    Returns:
        None
    '''
    
    # Plot the white background to clean up the edges of the diamond?
    white_background = True
    if white_background:
        blank_contour_pts_R = [[0.,-1.],[1.,0.],[0,1.],[-0.5,dedge],
                                [dedge,dedge],[dedge,-dedge],[-0.5,-dedge],
                                [0.,-1.]]
        blank_contour_pts_L = [[0.,-1.],[-1.,0.],[0,1.],[0.5,dedge],
                                [-dedge,dedge],[-dedge,-dedge],[0.5,-dedge],
                                [0.,-1.]]
        blank_contour_poly_R = patches.Polygon(blank_contour_pts_R, 
            closed=True, fill=True, zorder=zorder-1, edgecolor='None', 
            facecolor='White')
        blank_contour_poly_L = patches.Polygon(blank_contour_pts_L, 
            closed=True, fill=True, zorder=zorder-1, edgecolor='None', 
            facecolor='White')
        ax.add_artist(blank_contour_poly_R)
        ax.add_artist(blank_contour_poly_L)
        
        # Do artificial ticks because the ticks get overwritten by the 
        # drawing of the white patch
        ticklen = 2*dedge/draw_tick_fac
        xticks = [-1,-0.5,0,0.5,1]
        yticks = [-1,-0.5,0,0.5,1]
        tickwidth = 1.
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        for i in range(len(xticks)):
            ax.plot( [xticks[i],xticks[i]], [-dedge,-dedge+ticklen], 
                linewidth=tickwidth, color='Black', zorder=zorder )
            ax.plot( [xticks[i],xticks[i]], [dedge,dedge-ticklen], 
                linewidth=tickwidth, color='Black', zorder=zorder )
            ax.plot( [-dedge,-dedge+ticklen], [yticks[i],yticks[i]], 
                linewidth=tickwidth, color='Black', zorder=zorder )
            ax.plot( [dedge,dedge-ticklen], [yticks[i],yticks[i]], 
                linewidth=tickwidth, color='Black', zorder=zorder )
        ###i
    #def
    
    # Plot the diamond itself
    ax.plot([-1,0], [0,-1], linestyle='solid', color='Black', linewidth=0.5, 
            zorder=zorder)
    ax.plot([0,1],  [-1,0], linestyle='solid', color='Black', linewidth=0.5, 
            zorder=zorder)
    ax.plot([1,0],  [0,1],  linestyle='solid', color='Black', linewidth=0.5, 
            zorder=zorder)
    ax.plot([0,-1], [1,0],  linestyle='solid', color='Black', linewidth=0.5, 
            zorder=zorder)
#def

def add_selection_boundaries(ax,selection,plot_cent=False,plot_kwargs={}):
    '''add_selection_boundaries:
    
    Draw selection boundaries. Argument is a list of selection boundaries. 
    Each selection is a list of 3 elements, a string denoting the type of 
    boundary ('ellipse' or 'line'), then two arrays which each have two 
    elements. The two arrays are either [xcent,ycent] and [xsemi,ysemi]
    (for ellipse) or [x1,x2] and [y1,y2] (for line).
    
    Args:
        ax (matplotlib axis object) - axis
        selection (list) - list of selections
        plot_cent (bool) - Plot the center of the bounding region (ellipse only)
        plot_kwargs (dict) - keyword arguments for the plot
    
    Returns:
        None
    '''
    n_selec = len(selection)
    for i in range(n_selec):
        cp_plot_kwargs = copy.deepcopy(plot_kwargs)
        if selection[i][0] == 'ellipse':
            cp_plot_kwargs.pop('color', None)
            _,cent,semi = selection[i]
            add_ellipse(ax,cent,semi,plot_kwargs=cp_plot_kwargs)
            if plot_cent:
                ax.scatter(cent[0], cent[1], marker='x', s=10, color='Black')
        if selection[i][0] == 'line':
            cp_plot_kwargs = copy.deepcopy(plot_kwargs)
            cp_plot_kwargs.pop('facecolor', None)
            cp_plot_kwargs.pop('edgecolor', None)
            _,xs,ys = selection[i]
            ax.plot(xs,ys,**cp_plot_kwargs)
        ##fi
    ###i
#def

def add_legend(ax, disk_selection_kws=None, halo_selection_kws=None):
    '''add_legend:
    
    Args:
        ax (matplotlib Axis object) - axis
        disk_selection_kws (dict) - dictionary of disk selection keywords
        halo_selection_kws (dict) - dictionary of halo selection keywords
        
    Returns:
        None
    '''
    cp_disk_selection_kws = copy.deepcopy(disk_selection_kws)
    cp_halo_selection_kws = copy.deepcopy(halo_selection_kws)
    cp_disk_selection_kws.pop('facecolor', None)
    cp_disk_selection_kws.pop('edgecolor', None)
    cp_halo_selection_kws.pop('facecolor', None)
    cp_halo_selection_kws.pop('edgecolor', None)
    ax.plot([],[], **cp_disk_selection_kws)
    ax.plot([],[], **cp_halo_selection_kws)
    legend = ax.legend(loc='lower left', fontsize=7, handlelength=2.2, 
                       frameon=False, ncol=1, handletextpad=0.2)
#def

def axis_limits_and_labels(ax,xlim,ylim,xlabel,ylabel,mixture_text,
                            is_left_edge=False,is_top_edge=False,
                            label_fontsize=8):
    '''axis_spacing_and_labels:
    
    Do the labels and limits for the axis
    
    Args:
        ax (matplotlib axis object) - axis
        xlim (list) - x limits
        ylim (list) - y limits
        xlabel (string) - x label
        ylabel (string) - y label
        is_edge (bool) - Is the left-most edge panel?
    
    Returns:
        None
    '''
    ax.set_xlabel(xlabel, fontsize=label_fontsize, labelpad=-0.1)
    ax.tick_params(axis='both',labelsize=label_fontsize)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    if is_left_edge:
        ax.set_ylabel(ylabel, fontsize=label_fontsize)
    else:
        ax.tick_params(labelleft=False)
    ##ie
    if is_top_edge:
        ax.set_title(mixture_text, fontsize=label_fontsize, loc='center')
    ##fi
#def

def add_ellipse(ax,cent,semi,plot_kwargs={}):
    '''add_ellipse:
    
    Add a bounding ellipse to an axis
    
    Args:
        ax (matplotlib axis object) - Axis
        cent (list) - ellipse center [x,y]
        semi (list) - ellipse semi-major axis [x,y]
        plot_kws (dict) - Plot keywords
    
    Returns:
        None
    '''
    ellipse = patches.Ellipse(cent, width=semi[0]*2, height=semi[1]*2, 
        **plot_kwargs)
    ax.add_artist(ellipse)
#def


# ----------------------------------------------------------------------------

# Lines equations

def line_equation(x,m,b):
    '''line_equation:
    
    Args:
        x (array) - x values
        m (float) - slope
        b (float) - y-intercept
    
    Returns:
        y (array) - y values
    '''
    return m*x + b
#def


def get_params_from_line(xs,ys):
    '''get_params_from_line:
    
    Get line parameters from 2 points
    
    Args:
        xs (list) - 2-element list of x points
        ys (list) - 2-element list of y points
    
    Returns:
        m (float) - slope
        b (float) - y-intercept
    '''
    m = (ys[1]-ys[0])/(xs[1]-xs[0])
    b = ys[0] - m*xs[0]
    return m,b
#def

# ----------------------------------------------------------------------------

# Kinematic spaces functions

def get_plottable_data(orbs,eELz,accs,mixture_arr,plot_type,phi0=0,seed=0,
                        absolute=False):
    '''get_plottable_data:
    
    Take in all the arrays of data for the different beta models, and output 
    a mixture of the correct type of data to be plotted. plot type can be 
    any of ['vRvT','Toomre','ELz','JRLz','eLz','AD']. Can also be 'Rz' to 
    get radius and height above the disk.
    
    Args:
        orbs (list) - list of orbit.Orbit objects for each beta
        eELz (list) - list of np.array([e,E,Lz]) for each beta
        accs (list) - list of np.array([JR,Lz,Jz]) for each beta
        mixture_arr (list) - 
        plot_type (str) - Type of data to extract for plot
        phi0 (float) - Potential at infinity to subtract from energies [0]
        seed (int) - seed to use when randomly extracting data
        absolute (int) - Use absolute fractions of total amount of data in 
            mixture_arr [False]

    Returns:
        plot_x (np.array) - y coordinate
        plot_y (np.array) - x coordinates
    '''
    
    orbs_mix = putil.parse_mixture(orbs, mixture_arr, seed=seed, absolute=absolute)
    eELz_mix = putil.parse_mixture(eELz, mixture_arr, seed=seed, absolute=absolute)
    accs_mix = putil.parse_mixture(accs, mixture_arr, seed=seed, absolute=absolute)
    n_in_mix = len(orbs_mix)
    
    if plot_type == 'vRvT':
        vR = np.array([])
        vT = np.array([])
        for i in range(n_in_mix):
            vR = np.append( vR, orbs_mix[i].vR().value )
            vT = np.append( vT, orbs_mix[i].vT().value )
        return vR,vT
    
    elif plot_type == 'Toomre':
        vT = np.array([])
        vperp = np.array([])
        for i in range(n_in_mix):
            vT = np.append( vT, orbs_mix[i].vT().value )
            this_vperp = np.sqrt( np.square( orbs_mix[i].vR().value ) +\
                                  np.square( orbs_mix[i].vz().value ) )
            vperp = np.append( vperp, this_vperp )
        return vT,vperp
    
    elif plot_type == 'ELz':
        E = np.array([])
        Lz = np.array([])
        for i in range(n_in_mix):
            E = np.append( E, (eELz_mix[i][1]-phi0)/1e5 )
            Lz = np.append( Lz, eELz_mix[i][2] )
        return Lz,E
    
    elif plot_type == 'JRLz':
        JR = np.array([])
        Lz = np.array([])
        for i in range(n_in_mix):
            JR = np.append( JR, np.sqrt(accs_mix[i][0]) )
            Lz = np.append( Lz, eELz_mix[i][2] )
        return Lz,JR
    
    elif plot_type == 'eLz':
        e = np.array([])
        Lz = np.array([])
        for i in range(n_in_mix):
            e = np.append( e, eELz_mix[i][0] )
            Lz = np.append( Lz, eELz_mix[i][2] )
        return Lz,e
    
    elif plot_type == 'AD':
        Jz_JR = np.array([])
        Jphi = np.array([])
        for i in range(n_in_mix):
            JR,Lz,Jz = accs_mix[i]
            Jtot = np.abs(JR) + np.abs(Lz) + np.abs(Jz)
            Jz_JR_norm = (Jz-JR) / Jtot
            Jphi_norm = Lz / Jtot
            Jz_JR = np.append(Jz_JR, Jz_JR_norm)
            Jphi = np.append(Jphi, Jphi_norm)
        return Jphi,Jz_JR
    
    elif plot_type == 'Rz':
        R = np.array([])
        z = np.array([])
        for i in range(n_in_mix):
            R = np.append( R, orbs_mix[i].R().value )
            z = np.append( z, orbs_mix[i].z().value )
        return R,z
    
    else:
        raise ValueError('plot_type not recognized')
    ##ie
#def

def is_in_scaled_selection(x,y,selection,factor=1):
    '''is_in_scaled_selection:
    
    Lightweight wrapper of is_in_ellipse to allow scaling the ellipse by a 
    constant factor in either dimension or both 
    
    Args:
        x (array) - X coordinates
        y (array) - Y coordinates
        selection (list) - List of selections. Must be ellipses
        factor (float or array) - 2-element array of scaling factors for x 
            direction and y direction. If is float then will be cast as 
            np.array([factor,factor])
    
    Returns:
        is_inside (array) - Boolean array same size as x and y
    '''
    if isinstance(factor,numbers.Number):
        factor = np.array([factor,factor],dtype='float')
    assert len(x) == len(y), 'x and y must be same shape'
    is_in_ellipse_bools = np.zeros_like(x,dtype='bool')
    n_selec = len(selection)
    for i in range(n_selec):
        assert selection[i][0] == 'ellipse', 'selections must be ellipses'
        _,cent,semi = selection[i]
        semi_in = [semi[0]*factor[0],semi[1]*factor[1]]
        is_in_ellipse_bools = is_in_ellipse_bools |\
                              is_in_ellipse(x,y,cent,semi_in)
    ###i
    return is_in_ellipse_bools
#def

def is_in_ellipse(x,y,ellipse_cent,ellipse_ab):
    '''is_in_ellipse:
    
    Determine if points x,y are in an ellipse
    
    Args:
        x (array) - X coordinates
        y (array) - Y coordinates
        ellipse_cent (list) - 2-element list of ellipse center
        ellipse_ab (list) - 2-element list of semi-major axes
    
    Returns:
        is_inside (array) - Boolean array same size as x and y
    '''
    a,b = ellipse_ab
    xo,yo = ellipse_cent
    if isinstance(x,apu.quantity.Quantity):
        x = x.value
    if isinstance(y,apu.quantity.Quantity):
        y = y.value
    ##fi
    elliptical_dist = np.power(x-xo,2)/a**2 + np.power(y-yo,2)/b**2
    return elliptical_dist < 1
#def

# ----------------------------------------------------------------------------

# Plotting for KSF creation

def plot_completeness_purity_distance_spline(xs,ys,spl,mock_xs,spl_type,
                                             spl_color='DodgerBlue',fig=None,
                                             ax=None):
    '''plot_completeness_purity_distance_spline:
    
    Make a plot of the completeness-distance spline or purity-distance spline.
    
    Args:
        xs (np array) - Distances, probably log
        ys (np array) - Completeness
        spl (scipy.UnivariateSpline) - Completeness-Distance or purity-distance 
            spline
        mock_xs (np array) - Distances at which spline should be evaluated
        spl_type (string) - either 'completeness' or 'purity'
        spl_color (string) - Spline color [default 'DodgerBlue']
        fig (matplotlib Figure) - Figure object [default None, makes fig]
        ax (matplotlib Axis) - Axis object [default None, makes ax]
        
        
    Returns:
        fig (matplotlib Figure) - Figure object
        ax (matplotlib Axes) - Axis object
    '''
    assert spl_type in ['completeness','purity'],\
        "spl_type must be 'completeness' or 'purity'"
    if fig == None or ax == None:
        fig = plt.figure(figsize=(3,2))
        ax = fig.add_subplot(111)
    ax.scatter( xs, ys, s=4, color='Black' )
    ax.plot( mock_xs, spl(mock_xs), color=spl_color, zorder=2, linestyle='solid')
    ax.set_xlabel(r'$\log_{10}$(Distance/kpc)', fontsize=8)
    ax.set_ylabel(spl_type.capitalize(), fontsize=8)
    fig.tight_layout()
    return fig,ax
#def


# ----------------------------------------------------------------------------

# Better colourmaps

class colors(object):
    def __init__(self):

        # colour table in HTML hex format
        self.hexcols = ['#332288', '#88CCEE', '#44AA99', '#117733', '#999933', 
                        '#DDCC77', '#CC6677', '#882255', '#AA4499', '#661100', 
                        '#6699CC', '#AA4466', '#4477AA']

        self.greysafecols = ['#809BC8', '#FF6666', '#FFCC66', '#64C204']

        self.xarr = [[12], 
                [12, 6], 
                [12, 6, 5], 
                [12, 6, 5, 3], 
                [0, 1, 3, 5, 6], 
                [0, 1, 3, 5, 6, 8], 
                [0, 1, 2, 3, 5, 6, 8], 
                [0, 1, 2, 3, 4, 5, 6, 8], 
                [0, 1, 2, 3, 4, 5, 6, 7, 8], 
                [0, 1, 2, 3, 4, 5, 9, 6, 7, 8], 
                [0, 10, 1, 2, 3, 4, 5, 9, 6, 7, 8], 
                [0, 10, 1, 2, 3, 4, 5, 9, 6, 11, 7, 8]]

    # get specified nr of distinct colours in HTML hex format.
    # usage: colour_list = safe_colours.distinct_list(num_colours_required)
    # returns: list of distinct colours in HTML hex
    def distinct_list(self,nr):

        # check if nr is in correct range
        if nr < 1 or nr > 12:
            print("wrong nr of distinct colours!")
            return

        # get list of indices
        lst = self.xarr[nr-1]
        
        # generate colour list by stepping through indices and looking them up
        # in the colour table
        i_col = 0
        col = [0] * nr
        for idx in lst:
            col[i_col] = self.hexcols[idx]
            i_col+=1
        return col

    # Generate a dictionary of all the safe colours which can be addressed by colour name.
    def distinct_named(self):
        cl = self.hexcols

        outdict = {'navy':cl[0],\
                   'cyan':cl[1],\
                   'turquoise':cl[2],\
                   'green':cl[3],\
                   'olive':cl[4],\
                   'sandstone':cl[5],\
                   'coral':cl[6],\
                   'maroon':cl[7],\
                   'magenta':cl[8],\
                   'brown':cl[9],\
                   'skyblue':cl[10],\
                   'pink':cl[11],\
                   'blue':cl[12]}

        return outdict


    # For making colourmaps.
    # Usage: cmap = safe_colours.colourmap('rainbow')
    def colourmap(self,maptype,invert=False):

        if maptype == 'diverging':
            # Deviation around zero colormap (blue--red)
            cols = []
            for x in np.linspace(0,1, 256):
                rcol = 0.237 - 2.13*x + 26.92*x**2 - 65.5*x**3 + 63.5*x**4 - 22.36*x**5
                gcol = ((0.572 + 1.524*x - 1.811*x**2)/(1 - 0.291*x + 0.1574*x**2))**2
                bcol = 1/(1.579 - 4.03*x + 12.92*x**2 - 31.4*x**3 + 48.6*x**4 - 23.36*x**5)
                cols.append((rcol, gcol, bcol))

            if invert==True:
                cols = cols[::-1]

            return plt.get_cmap(matplotlib.colors.LinearSegmentedColormap.from_list("PaulT_plusmin", cols))

        elif maptype == 'heat':
            # Linear colormap (white--red)
            cols = []
            for x in np.linspace(0,1, 256):
                rcol = (1 - 0.392*(1 + erf((x - 0.869)/ 0.255)))
                gcol = (1.021 - 0.456*(1 + erf((x - 0.527)/ 0.376)))
                bcol = (1 - 0.493*(1 + erf((x - 0.272)/ 0.309)))
                cols.append((rcol, gcol, bcol))

            if invert==True:
                cols = cols[::-1]

            return plt.get_cmap(matplotlib.colors.LinearSegmentedColormap.from_list("PaulT_linear", cols))

        elif maptype == 'rainbow':
            # Linear colormap (rainbow)
            cols = []
            for x in np.linspace(0,1, 254):
                rcol = (0.472-0.567*x+4.05*x**2)/(1.+8.72*x-19.17*x**2+14.1*x**3)
                gcol = 0.108932-1.22635*x+27.284*x**2-98.577*x**3+163.3*x**4-131.395*x**5+40.634*x**6
                bcol = 1./(1.97+3.54*x-68.5*x**2+243*x**3-297*x**4+125*x**5)
                cols.append((rcol, gcol, bcol))

            if invert==True:
                cols = cols[::-1]

            return plt.get_cmap(matplotlib.colors.LinearSegmentedColormap.from_list("PaulT_rainbow", cols))

        else:
            raise KeyError('Please pick a valid colourmap, options are'\
                           +' "diverging", "heat" or "rainbow"')
    ##ie
#cls