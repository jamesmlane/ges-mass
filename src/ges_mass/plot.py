# ----------------------------------------------------------------------------
#
# TITLE - util.py
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
from astropy import units as apu
import matplotlib
from matplotlib import patches
from matplotlib import pyplot as plt
from scipy.special import erf

from . import util as putil

# ----------------------------------------------------------------------------

# Plotting density profiles

# Functions
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

def plot_density_xyz(model,params,n=100,scale=20.,fig=None,axs=None):
    '''plot_density_xyz:
    
    Plot XYZ plane views of a density profile
    
    Args:
        model (callable) - Density function that takes R,phi,z and params
        params (list) - List of parameters for the density function
        n (int) - Number of bins in each dimension [default 100]
        scale (float) - Size of the box, units depend on model and params 
            [default 20]
        fig (matplotlib Figure instance) - Figure, if None will make one
        axs (matplotlib Axes instance) - Axes, must be 3, if None will make one
    
    Returns:
        
    '''
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
        if i == 0: # XY plane
            xgrid,ygrid = np.meshgrid(xs,ys)
            zgrid = np.zeros_like(xgrid)
            Rgrid,phigrid,zgrid = xyz_to_Rphiz(xgrid,ygrid,zgrid)
        if i == 1: # XZ plane
            xgrid,zgrid = np.meshgrid(xs,zs)
            ygrid = np.zeros_like(xgrid)
            Rgrid,phigrid,zgrid = xyz_to_Rphiz(xgrid,ygrid,zgrid)
        if i == 2: # YZ plane
            ygrid,zgrid = np.meshgrid(ys,zs)
            xgrid = np.zeros_like(ygrid)
            Rgrid,phigrid,zgrid = xyz_to_Rphiz(xgrid,ygrid,zgrid)
    
        # Calculate densities
        dgrid = model(Rgrid,phigrid,zgrid,params)
        
        axs[i].imshow(np.log10(dgrid), cmap='rainbow', 
                      extent=(-scale,scale,-scale,scale), vmin=-1, vmax=3, 
                      origin='lower')
        axs[i].set_xlabel(ind_str[i][0]+' [kpc]')
        axs[i].set_ylabel(ind_str[i][1]+' [kpc]')

    fig.tight_layout()
    fig.show()
    
    return fig,axs

# ----------------------------------------------------------------------------

# Plotting utilities for kinematic spaces

def add_diamond_boundary(ax,dedge=1.2,zorder=5):
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
            closed=True, fill=True, zorder=2, edgecolor='None', 
            facecolor='White')
        blank_contour_poly_L = patches.Polygon(blank_contour_pts_L, 
            closed=True, fill=True, zorder=2, edgecolor='None', 
            facecolor='White')
        ax.add_artist(blank_contour_poly_R)
        ax.add_artist(blank_contour_poly_L)
        
        # Do artificial ticks because the ticks get overwritten by the 
        # drawing of the white patch
        ticklen = 2*dedge/20
        xticks = [-1,0,1]
        yticks = [-1,0,1]
        tickwidth = 1.
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

def plot_completeness_purity_distance_spline(xs,ys,spl,mock_xs,spl_type):
    '''plot_completeness_purity_distance_spline:
    
    Make a plot of the completeness-distance spline or purity-distance spline.
    
    Args:
        xs (np array) - Distances, probably log
        ys (np array) - Completeness
        spl (scipy.UnivariateSpline) - Completeness-Distance or purity-distance 
            spline
        mock_xs (np array) - Distances at which spline should be evaluated
        spl_type (string) - either 'completeness' or 'purity'
        
    Returns:
        fig (matplotlib Figure) - Figure object
        ax (matplotlib Axes) - Axis object
    '''
    assert spl_type in ['completeness','purity'],\
        "spl_type must be 'completeness' or 'purity'"
    fig = plt.figure(figsize=(3,2))
    ax = fig.add_subplot(111)
    ax.scatter( xs, ys, s=4, color='Black' )
    ax.plot( mock_xs, spl(mock_xs), color='DodgerBlue', zorder=2, linestyle='solid' )
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