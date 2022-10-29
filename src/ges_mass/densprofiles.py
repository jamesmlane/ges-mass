# ----------------------------------------------------------------------------
#
# TITLE - fit_GE_mass_mock.py
# AUTHOR - James Lane
# PROJECT - ges-mass
#
# -----------------------------------------------------------------------------

### Imports

import numpy as np
from galpy.util import bovy_coords, _rotate_to_arbitrary_vector
from scipy.optimize import newton
from scipy.special import erfinv
import warnings

# Scales
_ro = 8.275 # Gravity Collab.
_zo = 0.0208 # Bennett and Bovy

# Normalization convention
norm_eta = lambda x : x # (x+1)/2
denorm_eta = lambda x : x # 2*x-1
theta_scale = 2*np.pi
phi_scale = np.pi

# Utilities for normalization and unit conversion

def normalize_parameters(params,model):
    '''normalize_parameters:
    
    Transform parameters from a finite domain to a normalized [0,1] domain.
    Parameters that have an infinite domain, such as power law indices are 
    not transformed
    
    Args:
        params (list) - Density function parameters 
        model (callable) - Density function
    
    Returns:
        params (list) - Normalized density function parameters
    '''
    # params should be 2d for indexing
    params = np.atleast_2d(params)
    
    # Non-rotated density profiles have trivial transformations
    if model.__name__ == 'spherical':
        params_out = [params[:,0],]
    if model.__name__ == 'spherical_cutoff':
        params_out = [params[:,0],params[:,1]]
    if model.__name__ == 'axisymmetric':
        params_out = [params[:,0],params[:,1],]
    if model.__name__ == 'triaxial_norot':
        params_out = [params[:,0],params[:,1],params[:,2]]
    
    # Handle profiles that might have a disk component
    if 'triaxial_single_angle_zvecpa' in model.__name__:
        # params are [alpha,p,q,theta,eta,pa]
        # theta is [0,2pi], eta is [-1,1], pa is [0,pi]
        params_out = np.array([params[:,0],
                               params[:,1],
                               params[:,2],
                               params[:,3]/theta_scale,
                               norm_eta(params[:,4]), 
                               params[:,5]/phi_scale
                              ]).T
    if 'triaxial_single_cutoff_zvecpa' in model.__name__:
        # params are [alpha,beta,p,q,theta,eta,pa]
        # theta is [0,2pi], eta is [-1,1], pa is [0,pi]
        params_out = np.array([params[:,0],
                               params[:,1],
                               params[:,2],
                               params[:,3],
                               params[:,4]/theta_scale,
                               norm_eta(params[:,5]), 
                               params[:,6]/phi_scale
                              ]).T
    if 'triaxial_broken_angle_zvecpa' in model.__name__:
        # params are [alpha_in,alpha_out,r1,p,q,theta,eta,pa]
        # theta is [0,2pi], eta is [-1,1], pa is [0,pi]
        params_out = np.array([params[:,0],
                               params[:,1],
                               params[:,2],
                               params[:,3], 
                               params[:,4],
                               params[:,5]/theta_scale,
                               norm_eta(params[:,6]),
                               params[:,7]/phi_scale
                              ]).T
    if 'triaxial_double_broken_angle_zvecpa' in model.__name__:
        # params are [alpha_in,alpha_mid,alpha_out,r1,r2,p,q,theta,eta,pa]
        # theta is [0,2pi], eta is [-1,1], pa is [0,pi]
        params_out = np.array([params[:,0],
                               params[:,1],
                               params[:,2],
                               params[:,3], 
                               params[:,4],
                               params[:,5],
                               params[:,6], 
                               params[:,7]/theta_scale,
                               norm_eta(params[:,8]), 
                               params[:,9]/phi_scale
                              ]).T
    if 'triaxial_single_trunc_zvecpa' in model.__name__:
        # params are [alpha,r1,p,q,theta,eta,pa]
        # theta is [0,2pi], eta is [-1,1], pa is [0,pi]
        params_out = np.array([params[:,0],
                               params[:,1],
                               params[:,2],
                               params[:,3],
                               params[:,4]/theta_scale,
                               norm_eta(params[:,5]), 
                               params[:,6]/phi_scale
                              ]).T
        
    if 'plusexpdisk' in model.__name__:
        # Add the disk contamination fraction, assume it's the last parameter
        params_out = np.concatenate((params_out,
                                     np.atleast_2d(params[:,-1]).T),
                                    axis=1)
        
    return params_out

def denormalize_parameters(params,model):
    '''denormalize_parameters:
    
    Transform parameters from a normalized [0,1] domain to a finite domain
    
    Args:
        params (list) - Normalized density function parameters 
        model (callable) - Density function
    
    Returns:
        params (list) - Density function parameters
    '''
    # params should be 2d for indexing
    params = np.atleast_2d(params)
    
    # Non-rotated density profiles have trivial transformations
    if model.__name__ == 'spherical':
        params_out = np.array([params[:,0],]).T
    if model.__name__ == 'spherical_cutoff':
        params_out = np.array([params[:,0],params[:,1]]).T
    if model.__name__ == 'axisymmetric':
        params_out = np.array([params[:,0],params[:,1],]).T
    if model.__name__ == 'triaxial_norot':
        params_out = np.array([params[:,0],params[:,1],params[:,2]]).T
        
    if 'triaxial_single_angle_zvecpa' in model.__name__:
        # params are [alpha,p,q,theta,eta,pa]
        # theta is [0,2pi], eta is [-1,1], pa is [0,pi]
        params_out = np.array([params[:,0],
                               params[:,1],
                               params[:,2],
                               params[:,3]*theta_scale,
                               denorm_eta(params[:,4]),
                               params[:,5]*phi_scale
                              ]).T
    if 'triaxial_single_cutoff_zvecpa' in model.__name__:
        # params are [alpha,beta,p,q,theta,eta,pa]
        # theta is [0,2pi], eta is [-1,1], pa is [0,pi]
        params_out = np.array([params[:,0],
                               params[:,1],
                               params[:,2],
                               params[:,3],
                               params[:,4]*theta_scale,
                               denorm_eta(params[:,5]),
                               params[:,6]*phi_scale
                              ]).T
    if 'triaxial_broken_angle_zvecpa' in model.__name__:
        # params are [alpha_in,alpha_out,beta,p,q,theta,eta,pa]
        # theta is [0,2pi], eta is [-1,1], pa is [0,pi]
        params_out = np.array([params[:,0],
                               params[:,1],
                               params[:,2],
                               params[:,3],
                               params[:,4],
                               params[:,5]*theta_scale,
                               denorm_eta(params[:,6]),
                               params[:,7]*phi_scale
                              ]).T
    if 'triaxial_double_broken_angle_zvecpa' in model.__name__:
        # params are [alpha_in,alpha_mid,alpha_out,r1,r2,p,q,theta,eta,pa]
        # theta is [0,2pi], eta is [-1,1], pa is [0,pi]
        params_out = np.array([params[:,0],
                               params[:,1],
                               params[:,2],
                               params[:,3],
                               params[:,4],
                               params[:,5],
                               params[:,6],
                               params[:,7]*theta_scale,
                               denorm_eta(params[:,8]),
                               params[:,9]*phi_scale
                              ]).T
    if 'triaxial_single_trunc_zvecpa' in model.__name__:
        # params are [alpha,beta,p,q,theta,eta,pa]
        # theta is [0,2pi], eta is [-1,1], pa is [0,pi]
        params_out = np.array([params[:,0],
                               params[:,1],
                               params[:,2],
                               params[:,3],
                               params[:,4]*theta_scale,
                               denorm_eta(params[:,5]),
                               params[:,6]*phi_scale
                              ]).T
    
    if 'plusexpdisk' in model.__name__:
        # Add the disk contamination fraction, assume it's the last parameter
        params_out = np.concatenate((params_out,
                                     np.atleast_2d(params[:,-1]).T),
                                    axis=1)
    
    return params_out

# Utilities for transformations

# def transform_aby(xyz,alpha,beta,gamma):
#     '''transform_aby:

#     Transform xyz coordinates by rotation around x-axis (alpha), transformed 
#     y-axis (beta) and twice transformed z-axis (gamma)
    
#     Args:
#         xyz (np.array) - Galactocentric coordinate array
#         alpha (float) - X-axis rotation angle
#         beta (float) - Transformed Y-axis rotation angle
#         gamma (float) - Twice-transformed Z-axis rotation angle
    
#     Returns:
#         x,y,z (np.arrays) - Galactocentric rectangular coordinates
#     '''
#     Rx = np.zeros([3,3])
#     Ry = np.zeros([3,3])
#     Rz = np.zeros([3,3])
#     Rx[0,0] = 1
#     Rx[1] = [0, np.cos(alpha), -np.sin(alpha)]
#     Rx[2] = [0, np.sin(alpha), np.cos(alpha)]
#     Ry[0] = [np.cos(beta), 0, np.sin(beta)]
#     Ry[1,1] = 1
#     Ry[2] = [-np.sin(beta), 0, np.cos(beta)]
#     Rz[0] = [np.cos(gamma), -np.sin(gamma), 0]
#     Rz[1] = [np.sin(gamma), np.cos(gamma), 0]
#     Rz[2,2] = 1
#     R = np.matmul(Rx,np.matmul(Ry,Rz))
#     if np.ndim(xyz) == 1:
#         tgalcenrect = np.dot(R, xyz)
#         x, y, z = tgalcenrect[0], tgalcenrect[1], tgalcenrect[2]
#     else:
#         tgalcenrect = np.einsum('ij,aj->ai', R, xyz)
#         x, y, z = tgalcenrect[:,0], tgalcenrect[:,1], tgalcenrect[:,2]
#     return x, y, z

def transform_zvecpa(xyz,zvec,pa):
    '''transform_zvecpa:

    Transform xyz coordinates by rotation of galactocentric Z-axis to zvec, 
    then a rotation of pa.
    
    Args:
        xyz (np.array) - Galactocentric coordinate array
        zvec (float) - New z-vector in the original galactocentric coordinate
            frame
        pa (float) - Position angle
    
    Returns:
        x,y,z (np.arrays) - Galactocentric rectangular coordinates
    '''
    pa_rot= np.array([[np.cos(pa),np.sin(pa),0.],
                         [-np.sin(pa),np.cos(pa),0.],
                         [0.,0.,1.]])

    zvec/= np.sqrt(np.sum(zvec**2.))
    zvec_rot= _rotate_to_arbitrary_vector(np.array([[0.,0.,1.]]),zvec,inv=True)[0]
    trot= np.dot(pa_rot,zvec_rot)
    if np.ndim(xyz) == 1:
        tgalcenrect = np.dot(trot, xyz)
        x, y, z = tgalcenrect[0], tgalcenrect[1], tgalcenrect[2]
    else:
        tgalcenrect = np.einsum('ij,aj->ai', trot, xyz)
        x, y, z = tgalcenrect[:,0], tgalcenrect[:,1], tgalcenrect[:,2]
    return x, y, z

def eta_theta_to_zvec(eta,theta):
    '''eta_theta_to_zvec:
    
    Transform eta and theta parameters to a z-vector according to:
    
    zvec = [ sqrt(1-eta^2)*cos(theta) ]
           [ sqrt(1-eta^2)*sin(theta) ]
           [ eta                      ]
    
    Either eta,theta can be floats, one can be float and the other array, 
    or both can be arrays of same length. If arrays then output is (3,N)
    
    Args:
        eta (float or np.array) - Z scale of z-vector (see above)
        theta (float or np.array) - XY scale of z-vector (see above)
    
    Returns:
        zvec (np.array) - Z-vector corresponding to eta and theta
    '''
    if (isinstance(eta,float) and isinstance(theta,float)):
        zvec = np.empty(3)
    else:
        assert (isinstance(eta,float) or isinstance(theta,float)) or\
            (len(eta)==len(theta))
        if isinstance(eta,float):
            nz = len(theta)
        else:
            nz = len(eta)
        zvec = np.empty((3,nz))
    zvec[0] = (1-eta**2.)**0.5*np.cos(theta)
    zvec[1] = (1-eta**2.)**0.5*np.sin(theta)
    zvec[2] = eta
    zvec /= np.sum(zvec**2.,axis=0)**0.5
    return zvec

def zvec_to_eta_theta(zvec):
    '''zvec_to_eta_theta:
    
    Transform z-vector to eta and theta parameters according to:
    
    theta = acos(z1/(1-z3**2)**0.5)
    eta = z3
    
    either zvec is a length-3 array in which case eta and theta are floats, 
    or zvec is a shape (3,N) array in which case eta and theta are
    
    Args:
        zvec (np.array) - z-vector corresponding to eta and theta
    
    Returns:
        eta (float or np.array) - Z scale of z-vector (see above)
        theta (float or np.array) - XY scale of z-vector (see above)
    '''
    zvec = np.asarray(zvec)
    assert zvec.shape==(3,) or (zvec.shape[0]==3), 'wrong shape'
    if len(zvec.shape)==1:
        zvec /= np.sum(zvec**2.)**0.5
    else:
        zvec /= np.sum(zvec**2.,axis=0)**0.5
    eta = zvec[2]
    theta = np.arccos(zvec[0]/(1-eta**2.)**0.5)
    # theta = np.arcsin(zvec[1]/(1-eta**2.)**0.5)
    try:
        eta = float(eta)
        theta = float(theta)
    except TypeError:
        pass
    if np.any(np.isnan(theta)): # Handle when eta=1.
        if isinstance(theta,np.ndarray):
            theta[np.isnan(theta)]=0.
        else:
            theta = 0.
    return eta,theta

# Utilities to get some information from density functions

def get_densfunc_params_indx(densfunc,param):
    '''get_densfunc_param_indx:
    
    Get the indices of a set of parameters
    
    Args:
        densfunc (callable) - densfunc to find the parameter indices
        param (list of str) - List of strings representing parameters 
    
    Returns:
        indx (array) - List of indices representing locations of parameters
    '''
    # Wrap up param if int
    if isinstance(param,int):
        param = [param,]
    
    # Lists of parameters
    if 'triaxial_single_angle_zvecpa' in densfunc.__name__:
        param_names = np.array(['alpha','p','q','theta','eta','phi'])
    
    elif 'triaxial_single_cutoff_zvecpa' in densfunc.__name__:
        param_names = np.array(['alpha','beta','p','q','theta','eta','phi'])
    
    elif 'triaxial_broken_angle_zvecpa' in densfunc.__name__:
        param_names = np.array(['alpha_in','alpha_out','r1','p','q','theta',
                                'eta','phi'])
        
    elif 'triaxial_double_broken_angle_zvecpa' in densfunc.__name__:
        param_names = np.array(['alpha_in','alpha_mid','alpha_out','r1',
                                'r2','p','q','theta','eta','phi'])
        
    if 'plusexpdisk' in densfunc.__name__:
        param_names = np.concatenate((param_names,['fdisk']))
    
    indx = []
    for i in range(len(param)):
        assert param[i] in param_names, '"'+params[i]+'" not in densfunc list'
        indx = indx+[ np.where(param_names==param[i])[0][0] ]
    
    return indx


def get_densfunc_nodisk(densfunc):
    '''get_densfunc_nodisk:
    
    Get a density profile corresponding to the input which does not have a 
    disk. If the input density profile has no disk to begin with then return 
    the input density profile.
    
    Args:
        densfunc (callable) - densfunc to find the counterpart without a 
            disk
    
    Returns:
        densfunc_nodisk (callable) - counterpart to densfunc which does not 
            have a disk component
    '''
    densfuncs_nodisk_already = [spherical, spherical_cutoff, axisymmetric, 
                                triaxial_norot, triaxial_single_angle_aby,
                                triaxial_single_angle_zvecpa, 
                                triaxial_single_cutoff_zvecpa, 
                                triaxial_broken_angle_zvecpa, 
                                triaxial_single_trunc_zvecpa]
    if densfunc in densfuncs_nodisk_already:
        return densfunc
    elif densfunc == triaxial_broken_angle_zvecpa_plusexpdisk:
        return triaxial_broken_angle_zvecpa
    elif densfunc == triaxial_single_angle_zvecpa_plusexpdisk:
        return triaxial_single_angle_zvecpa
    elif densfunc == triaxial_single_cutoff_zvecpa_plusexpdisk:
        return triaxial_single_cutoff_zvecpa
    elif densfunc == triaxial_single_trunc_zvecpa_plusexpdisk:
        return triaxial_single_trunc_zvecpa
    elif densfunc == exp_disk:
        warnings.warn('exp_disk has no component without a disk')
        return None

def get_densfunc_mcmc_labels(densfunc, physical_units=False):
    '''get_densfunc_mcmc_labels:
    
    Args:
        densfunc (callable) - density function
        physical_units (bool) - 
    
    Returns:
        labels (arr) - String array
    '''
    if  'triaxial_single_angle_zvecpa' in densfunc.__name__:
        labels = [r'$\alpha$', r'$p$', r'$q$', r'$\theta$', r'$\eta$', 
                  r'$\phi$']
    elif 'triaxial_single_cutoff_zvecpa' in densfunc.__name__:
        labels = [r'$\alpha$', r'$\beta$', r'$p$', r'$q$', r'$\theta$', 
                  r'$\eta$', r'$\phi$']
    elif 'triaxial_broken_angle_zvecpa' in densfunc.__name__:
        labels = [r'$\alpha_{in}$', r'$\alpha_{out}$', r'$\beta$', r'$p$', 
                  r'$q$', r'$\theta$', r'$\eta$', r'$\phi$']
    elif 'triaxial_single_trunc_zvecpa' in densfunc.__name__:
        labels = [r'$\alpha$', r'$\beta$', r'$p$', r'$q$', r'$\theta$', 
                  r'$\eta$', r'$\phi$']
                    
    if densfunc.__name__[-11:] == 'plusexpdisk':
        labels.append(r'$f_{disk}$')
    
    if physical_units:
        for i in range(len(labels)):
            if 'beta' in labels[i]:
                labels[i] = labels[i]+' [kpc]'
            if 'theta' in labels[i]:
                labels[i] = labels[i]+' [rad]'
            if 'phi' in labels[i]:
                labels[i] = labels[i]+' [rad]'
    
    return labels
    
def get_densfunc_mcmc_init(densfunc):
    '''get_densfunc_mcmc_init:
    
    Get the initialization for MCMC
    
    Args:
        densfunc (callable) - density function
        
    Returns:
        init (array) - Initia
    '''
    if  'triaxial_single_angle_zvecpa' in densfunc.__name__:
        init = np.array([2.0, 0.5, 0.5, 0.5, 0.5, 0.5])
    elif 'triaxial_single_cutoff_zvecpa' in densfunc.__name__:
        init = np.array([2.0, 20., 0.5, 0.5, 0.5, 0.5, 0.5])
    elif 'triaxial_broken_angle_zvecpa' in densfunc.__name__:
        init = np.array([2., 2., 20., 0.5, 0.5, 0.5, 0.5, 0.5])
    elif 'triaxial_single_trunc_zvecpa' in densfunc.__name__:
        init = np.array([2.0, 20., 0.5, 0.5, 0.5, 0.5, 0.5])
    
    if 'plusexpdisk' in densfunc.__name__:
        init = np.concatenate((init,[0.01,]))
    
    return init
    
# Density models
    
def spherical(R,phi,z,params=[2.5,]):
    '''spherical:
    
    general spherical power-law density model
    
    Args: 
        R, phi, z (np.arrays) - Galactocentric cylindrical coordinates
        params (float array) - [alpha,]
            alpha (float) - Power law index
        
    Returns
        dens (np.array) - density at coordinates (normalized to 1 at _ro,_zo)
    '''
    grid = False
    if np.ndim(R) > 1:
        grid = True
        dim = np.shape(R)
        R = R.reshape(np.product(dim))
        phi = phi.reshape(np.product(dim))
        z = z.reshape(np.product(dim))
    x, y, z = R*np.cos(phi), R*np.sin(phi), z
    dens = np.sqrt(x**2+y**2+z**2)**(-params[0])
    dens = dens/(np.sqrt(_ro**2+_zo**2)**(-params[0]))
    if grid:
        dens = dens.reshape(dim)
    return dens

def spherical_cutoff(R,phi,z,params=[2.5,0.1]):
    '''spherical_cutoff:
    
    general spherical power-law density model with exponential cutoff
    
    Args: 
        R, phi, z (np.arrays) - Galactocentric cylindrical coordinates
        params (float array) - [alpha,beta]
            alpha (float) - Power law index
            beta (float) - Inverse exponential truncation scale
        
    Returns
        dens (np.array) - density at coordinates (normalized to 1 at _ro,_zo)
    '''
    grid = False
    if np.ndim(R) > 1:
        grid = True
        dim = np.shape(R)
        R = R.reshape(np.product(dim))
        phi = phi.reshape(np.product(dim))
        z = z.reshape(np.product(dim))
    x, y, z = R*np.cos(phi), R*np.sin(phi), z
    r = np.sqrt(x**2+y**2+z**2)
    dens = r**(-params[0])*np.exp(-params[1]*r)
    dens = dens/(np.sqrt(_ro**2+_zo**2)**(-params[0])\
                 *np.exp(-params[1]*np.sqrt(_ro**2+_zo**2)))
    if grid:
        dens = dens.reshape(dim)
    return dens

def axisymmetric(R,phi,z,params=[2.5,1.]):
    '''axisymmetric:
    
    general axisymmetric power-law density model
    
    Args: 
        R, phi, z (np.arrays) - Galactocentric cylindrical coordinates
        params (float array) - [alpha,q]
            alpha (float) - Power law index
            q (float) - Ratio of Z to X scale lengths
        
    Returns
        dens (np.array) - density at coordinates (normalized to 1 at _ro,_zo)
    '''
    grid = False
    if np.ndim(R) > 1:
        grid = True
        dim = np.shape(R)
        R = R.reshape(np.product(dim))
        phi = phi.reshape(np.product(dim))
        z = z.reshape(np.product(dim))
    x, y, z = R*np.cos(phi), R*np.sin(phi), z
    dens = np.sqrt(x**2+y**2+z**2/params[1]**2)**-params[0]
    dens = dens/np.sqrt(_ro**2+_zo**2/params[1]**2)**-params[0]
    if grid:
        dens = dens.reshape(dim)
    return dens

def triaxial_norot(R,phi,z,params=[2.5,1.,1.]):
    '''triaxial_norot:
    
    General triaxial power-law density profile without rotation (aligned with
    galactocentric system)
    
    Args: 
        R, phi, z (np.arrays) - Galactocentric cylindrical coordinates
        params (float array) - [alpha,p,q]
            alpha (float) - Power law index
            p (float) - Ratio of Y to X scale lengths
            q (float) - Ratio of Z to X scale lengths
        
    Returns
        dens (np.array) - density at coordinates (normalized to 1 at _ro,_zo)
    '''
    grid = False
    if np.ndim(R) > 1:
        grid = True
        dim = np.shape(R)
        R = R.reshape(np.product(dim))
        phi = phi.reshape(np.product(dim))
        z = z.reshape(np.product(dim))
    x, y, z = R*np.cos(phi), R*np.sin(phi), z
    dens = np.sqrt(x**2+y**2/params[1]**2+z**2/params[2]**2)**-params[0]
    dens = dens/np.sqrt(_ro**2+_zo**2/params[2]**2)**-params[0]
    if grid:
        dens = dens.reshape(dim)
    return dens

# def triaxial_single_angle_aby(R,phi,z,params=[2.,0.5,0.5,0.5,0.5,0.5]):
#     '''triaxial_single_angle_aby:
# 
#     Triaxial power-law density profile rotated using the alpha-beta-gamma 
#     scheme (see transform_aby)
# 
#     Args: 
#         R, phi, z (np.arrays) - Galactocentric cylindrical coordinates
#         params (float array) - [alpha,p,q,A,B,Y]
#             alpha (float) - Power law index
#             p (float) - Ratio of Y to X scale lengths
#             q (float) - Ratio of Z to X scale lengths
#             A (float) - Alpha rotation angle
#             B (float) - Beta rotation angle
#             Y (float) - Gamma rotation angle
# 
#     Returns
#         dens (np.array) - density at coordinates (normalized to 1 at _ro,_zo)
#     '''
#     grid = False
#     # alpha = 0.9*np.pi*params[3]+0.05*np.pi-np.pi/2.
#     # beta = 0.9*np.pi*params[4]+0.05*np.pi-np.pi/2.
#     # gamma = 0.9*np.pi*params[5]+0.05*np.pi-np.pi/2.
#     if np.ndim(R) > 1:
#         grid = True
#         dim = np.shape(R)
#         R = R.reshape(np.product(dim))
#         phi = phi.reshape(np.product(dim))
#         z = z.reshape(np.product(dim))
#     x, y, z = R*np.cos(phi), R*np.sin(phi), z
#     x, y, z = transform_aby(np.dstack([x,y,z])[0], alpha,beta,gamma)
#     xsun, ysun, zsun = transform_aby([_ro, 0., _zo],alpha,beta,gamma)
#     r_e = np.sqrt(x**2+y**2/params[1]**2+z**2/params[2]**2)
#     r_e_sun = np.sqrt(xsun**2+ysun**2/params[1]**2+zsun**2/params[2]**2)
#     dens = (r_e)**(-params[0])
#     sundens = (r_e_sun)**(-params[0])
#     dens = dens/sundens
#     if grid:
#         dens = dens.reshape(dim)
#     return dens

def triaxial_single_angle_zvecpa(R,phi,z,params=[2.,0.5,0.5,0.,0.,0.]):
    '''triaxial_single_angle_zvecpa:
    
    Triaxial power-law density profile rotated using the zvec-pa scheme 
    (see transform_zvecpa). Note that zvec is parameterized using two 
    parameters eta and theta as follows:
    
    zvec = [ sqrt(1-eta^2)*cos(theta) ]
           [ sqrt(1-eta^2)*sin(theta) ]
           [ eta                      ]
    
    Args: 
        R, phi, z (np.arrays) - Galactocentric cylindrical coordinates
        params (float array) - [alpha,p,q,theta,eta,pa]
            alpha (float) - Power law index
            p (float) - Ratio of Y to X scale lengths
            q (float) - Ratio of Z to X scale lengths
            theta (float) - Sets scale / orientation of zvec in XY plane
            eta (float) - Sets scale of zvec along Z-axis
            pa (float) - Final rotation angle
        
    Returns
        dens (np.array) - density at coordinates (normalized to 1 at _ro,_zo)
    '''
    grid = False
    theta = params[3]*2*np.pi
    tz = (params[4]*2)-1
    zvec = np.array([np.sqrt(1-tz**2)*np.cos(theta), 
                     np.sqrt(1-tz**2)*np.sin(theta), 
                     tz])
    pa = (params[5]*np.pi)#-np.pi/2.
    if np.ndim(R) > 1:
        grid = True
        dim = np.shape(R)
        R = R.reshape(np.product(dim))
        phi = phi.reshape(np.product(dim))
        z = z.reshape(np.product(dim))
    x, y, z = R*np.cos(phi), R*np.sin(phi), z
    x, y, z = transform_zvecpa(np.dstack([x,y,z])[0], zvec,pa)
    xsun,ysun,zsun = transform_zvecpa([_ro,0.,_zo],zvec,pa)
    r_e = np.sqrt(x**2+y**2/params[1]**2+z**2/params[2]**2)
    r_e_sun = np.sqrt(xsun**2+ysun**2/params[1]**2+zsun**2/params[2]**2)
    dens = (r_e)**(-params[0])
    sundens = (r_e_sun)**(-params[0])
    dens = dens/sundens
    if grid:
        dens = dens.reshape(dim)
    return dens

def triaxial_single_cutoff_zvecpa(R,phi,z,params=[2.,1.,0.5,0.5,0.,0.,0.]):
    '''triaxial_single_cutoff_zvecpa:
    
    Triaxial power-law density profile with exponential cutoff rotated 
    using the zvec-pa scheme (see transform_zvecpa). Note that zvec is 
    parameterized using two parameters eta and theta as follows:
    
    zvec = [ sqrt(1-eta^2)*cos(theta) ]
           [ sqrt(1-eta^2)*sin(theta) ]
           [ eta                      ]
    
    Args: 
        R, phi, z (np.arrays) - Galactocentric cylindrical coordinates
        params (float array) - [alpha,p,q,eta,theta,pa]
            alpha (float) - Power law index
            beta (float) - Inverse exponential truncation scale
            p (float) - Ratio of Y to X scale lengths
            q (float) - Ratio of Z to X scale lengths
            theta (float) - Sets scale / orientation of zvec in XY plane
            eta (float) - Sets scale of zvec along Z-axis
            pa (float) - Final rotation angle
        
    Returns
        dens (np.array) - density at coordinates (normalized to 1 at _ro,_zo)
    '''
    # params[1] = 0. # Test
    grid = False
    theta = params[4]*2*np.pi
    tz = (params[5]*2)-1
    zvec = np.array([np.sqrt(1-tz**2)*np.cos(theta), np.sqrt(1-tz**2)*np.sin(theta), tz])
    pa = (params[6]*np.pi)
    if np.ndim(R) > 1:
        grid = True
        dim = np.shape(R)
        R = R.reshape(np.product(dim))
        phi = phi.reshape(np.product(dim))
        z = z.reshape(np.product(dim))
    x, y, z = R*np.cos(phi), R*np.sin(phi), z
    x, y, z = transform_zvecpa(np.dstack([x,y,z])[0], zvec,pa)
    xsun,ysun,zsun = transform_zvecpa([_ro,0.,_zo],zvec,pa)
    r_e = np.sqrt(x**2+y**2/params[2]**2+z**2/params[3]**2)
    r_e_sun = np.sqrt(xsun**2+ysun**2/params[2]**2+zsun**2/params[3]**2)
    dens = (r_e)**(-params[0])*np.exp(-params[1]*r_e)
    sundens = (r_e_sun)**(-params[0])*np.exp(-params[1]*r_e_sun)
    dens = dens/sundens
    if grid:
        dens = dens.reshape(dim)
    return dens

def triaxial_broken_angle_zvecpa(R,phi,z,
    params=[2.,3.,5.,0.5,0.5,0.,0.,0.],split=False):
    '''triaxial_broken_angle_zvecpa:
    
    Triaxial broken angle power-law density profile rotated using the zvec-pa 
    scheme (see transform_zvecpa). Note that zvec is parameterized using two 
    parameters eta and theta as follows:
    
    zvec = [ sqrt(1-eta^2)*cos(theta) ]
           [ sqrt(1-eta^2)*sin(theta) ]
           [ eta                      ]
    
    Args: 
        R, phi, z (np.arrays) - Galactocentric cylindrical coordinates
        params (float array) - [alpha,beta,p,q,eta,theta,pa,fdisk]
            alpha_in (float) - Inner power law index
            alpha_out (float) - Outer power law index
            beta (float) - Radius where the power law index changes
            p (float) - Ratio of Y to X scale lengths
            q (float) - Ratio of Z to X scale lengths
            theta (float) - Sets scale / orientation of zvec in XY plane
            eta (float) - Sets scale of zvec along Z-axis
            pa (float) - Final rotation angle
                contained in the disk, halo density fraction is then (1-fdisk)
        
    Returns
        dens (np.array) - density at coordinates (normalized to 1 at _ro,_zo)
    '''
    grid = False
    theta = params[5]*2*np.pi
    tz = (params[6]*2)-1
    zvec = np.array([np.sqrt(1-tz**2)*np.cos(theta), np.sqrt(1-tz**2)*np.sin(theta), tz])
    pa = (params[7]*np.pi)
    if np.ndim(R) > 1:
        grid = True
        dim = np.shape(R)
        R = R.reshape(np.product(dim))
        phi = phi.reshape(np.product(dim))
        z = z.reshape(np.product(dim))
    x, y, z = R*np.cos(phi), R*np.sin(phi), z
    x, y, z = transform_zvecpa(np.dstack([x,y,z])[0], zvec,pa)
    xsun,ysun,zsun = transform_zvecpa([_ro,0.,_zo],zvec,pa)
    r_e = np.sqrt(x**2+y**2/params[3]**2+z**2/params[4]**2)
    r_e_sun = np.sqrt(xsun**2+ysun**2/params[3]**2+zsun**2/params[4]**2)
    dens = np.zeros(len(r_e))
    dens[r_e < params[2]] = (r_e[r_e < params[2]])**(-params[0])
    dens[r_e > params[2]] = (params[2])**(params[1]-params[0])*(r_e[r_e > params[2]])**(-params[1])
    if params[2] < r_e_sun:
        sundens = (params[2])**(params[1]-params[0])*(r_e_sun)**(-params[1])
    else:
        sundens = (r_e_sun)**(-params[0])
    dens = dens/sundens
    if grid:
        dens = dens.reshape(dim)
    return dens

def triaxial_single_trunc_zvecpa(R,phi,z,
    params=[2.,10.,0.5,0.5,0.,0.,0.],split=False):
    '''triaxial_single_trunc_zvecpa:
    
    Triaxial truncated (density goes to 0 at some radius) density profile 
    rotated using the zvec-pa scheme (see transform_zvecpa). Note that zvec is 
    parameterized using two parameters eta and theta as follows:
    
    zvec = [ sqrt(1-eta^2)*cos(theta) ]
           [ sqrt(1-eta^2)*sin(theta) ]
           [ eta                      ]
    
    Args: 
        R, phi, z (np.arrays) - Galactocentric cylindrical coordinates
        params (float array) - [alpha,beta,p,q,eta,theta,pa,fdisk]
            alpha (float) - power law index
            beta (float) - Radius where the density profile goes to 0
            p (float) - Ratio of Y to X scale lengths
            q (float) - Ratio of Z to X scale lengths
            theta (float) - Sets scale / orientation of zvec in XY plane
            eta (float) - Sets scale of zvec along Z-axis
            pa (float) - Final rotation angle
                contained in the disk, halo density fraction is then (1-fdisk)
        
    Returns
        dens (np.array) - density at coordinates (normalized to 1 at _ro,_zo)
    '''
    grid = False
    theta = params[4]*2*np.pi
    tz = (params[5]*2)-1
    zvec = np.array([np.sqrt(1-tz**2)*np.cos(theta), np.sqrt(1-tz**2)*np.sin(theta), tz])
    pa = (params[6]*np.pi)
    if np.ndim(R) > 1:
        grid = True
        dim = np.shape(R)
        R = R.reshape(np.product(dim))
        phi = phi.reshape(np.product(dim))
        z = z.reshape(np.product(dim))
    x, y, z = R*np.cos(phi), R*np.sin(phi), z
    x, y, z = transform_zvecpa(np.dstack([x,y,z])[0], zvec,pa)
    xsun,ysun,zsun = transform_zvecpa([_ro,0.,_zo],zvec,pa)
    r_e = np.sqrt(x**2+y**2/params[2]**2+z**2/params[3]**2)
    r_e_sun = np.sqrt(xsun**2+ysun**2/params[2]**2+zsun**2/params[3]**2)
    dens = np.zeros(len(r_e))
    dens[r_e < params[1]] = (r_e[r_e < params[1]])**(-params[0])
    dens[r_e > params[1]] = 0
    sundens = (r_e_sun)**(-params[0])
    dens = dens/sundens
    if grid:
        dens = dens.reshape(dim)
    return dens

def exp_disk(R,phi,z,params=[1/1.8,1/0.8]):
    '''exp_disk:
    
    Exponential disk
    
    Args:
        R, phi, z (np.arrays) - Galactocentric cylindrical coordinates
        params (float array) - [1/hr,1/hz]
            1/hr (float) - Inverse radial scale
            1/hz (float) - Inverse vertical scale
    
    Returns:
        dens (np.array) - density at coordinates (normalized to 1 at _ro,_zo)
    '''
    hr = params[0]
    hz = params[1]
    diskdens = np.exp(-hr*(R-_ro)-hz*np.fabs(z))
    diskdens_sun = np.exp(-hr*(_ro-_ro)-hz*np.fabs(_zo))
    return diskdens/diskdens_sun


def triaxial_single_angle_zvecpa_plusexpdisk(R,phi,z,
    params=[2.,0.5,0.5,0.,0.,0.,0.01],split=False):
    '''triaxial_single_angle_zvecpa_plusexpdisk:
    
    Triaxial power-law density profile rotated using the zvec-pa scheme 
    (see transform_zvecpa) with exponential disk contamination. Note that zvec 
    is parameterized using two parameters eta and theta as follows:
    
    zvec = [ sqrt(1-eta^2)*cos(theta) ]
           [ sqrt(1-eta^2)*sin(theta) ]
           [ eta                      ]
    
    Note that the parameters for the disk are fixed
    
    Args: 
        R, phi, z (np.arrays) - Galactocentric cylindrical coordinates
        params (float array) - [alpha,p,q,eta,theta,pa,fdisk]
            alpha (float) - Power law index
            p (float) - Ratio of Y to X scale lengths
            q (float) - Ratio of Z to X scale lengths
            theta (float) - Sets scale / orientation of zvec in XY plane
            eta (float) - Sets scale of zvec along Z-axis
            pa (float) - Final rotation angle
            fdisk (float) - Fraction of density at the location of the Sun
                contained in the disk, halo density fraction is then (1-fdisk)
        
    Returns
        dens (np.array) - density at coordinates (normalized to 1 at _ro,_zo)
    '''
    hr = 1/2. # Fixed disk parameters
    hz = 1/0.8
    original_z = np.copy(z)
    grid = False
    theta = params[3]*2*np.pi
    tz = (params[4]*2)-1
    zvec = np.array([np.sqrt(1-tz**2)*np.cos(theta),
                     np.sqrt(1-tz**2)*np.sin(theta), 
                     tz])
    pa = (params[5]*np.pi)#-np.pi/2.
    if np.ndim(R) > 1:
        grid = True
        dim = np.shape(R)
        R = R.reshape(np.product(dim))
        phi = phi.reshape(np.product(dim))
        z = z.reshape(np.product(dim))
        original_z = original_z.reshape(np.product(dim))
    x, y, z = R*np.cos(phi), R*np.sin(phi), z
    x, y, z = transform_zvecpa(np.dstack([x,y,z])[0], zvec,pa)
    xsun,ysun,zsun = transform_zvecpa([_ro,0.,_zo],zvec,pa)
    r_e = np.sqrt(x**2+y**2/params[1]**2+z**2/params[2]**2)
    r_e_sun = np.sqrt(xsun**2+ysun**2/params[1]**2+zsun**2/params[2]**2)
    diskdens = np.exp(-hr*(R-_ro)-hz*np.fabs(original_z))
    diskdens_sun = np.exp(-hr*(_ro-_ro)-hz*np.fabs(_zo))
    dens = (r_e)**(-params[0])
    sundens = (r_e_sun)**(-params[0])
    if split:
        dens, diskdens = (1-params[6])*dens/sundens, (params[6])*diskdens/diskdens_sun
        if grid:
            dens = dens.reshape(dim)
            diskdens = diskdens.reshape(dim)
        return dens, diskdens
    else:
        dens = (1-params[6])*dens/sundens+(params[6]*diskdens/diskdens_sun)
        #dens = ((1-params[6])*dens+params[6]*diskdens)/((1-params[6])*sundens+params[6]*diskdens_sun)
        if grid:
            dens = dens.reshape(dim)
        return dens

def triaxial_single_cutoff_zvecpa_plusexpdisk(R,phi,z,
    params=[2.,1.,0.5,0.5,0.,0.,0.,0.01],split=False):
    '''triaxial_single_cutoff_zvecpa_plusexpdisk:
    
    Triaxial power-law density profile with exponential cutoff rotated using 
    the zvec-pa scheme (see transform_zvecpa) with exponential disk 
    contamination. Note that zvec is parameterized using two parameters eta and 
    theta as follows:
    
    zvec = [ sqrt(1-eta^2)*cos(theta) ]
           [ sqrt(1-eta^2)*sin(theta) ]
           [ eta                      ]
    
    Note that the parameters for the disk are fixed
    
    Args: 
        R, phi, z (np.arrays) - Galactocentric cylindrical coordinates
        params (float array) - [alpha,beta,p,q,eta,theta,pa,fdisk]
            alpha (float) - Power law index
            beta (float) - Inverse exponential truncation scale
            p (float) - Ratio of Y to X scale lengths
            q (float) - Ratio of Z to X scale lengths
            theta (float) - Sets scale / orientation of zvec in XY plane
            eta (float) - Sets scale of zvec along Z-axis
            pa (float) - Final rotation angle
            fdisk (float) - Fraction of density at the location of the Sun
                contained in the disk, halo density fraction is then (1-fdisk)
        
    Returns
        dens (np.array) - density at coordinates (normalized to 1 at _ro,_zo)
    '''
    # params[1] = 0. # Test
    hr = 1/2. # Fixed disk parameters
    hz = 1/0.8
    original_z = np.copy(z)
    grid = False
    theta = params[4]*2*np.pi
    tz = (params[5]*2)-1
    zvec = np.array([np.sqrt(1-tz**2)*np.cos(theta), np.sqrt(1-tz**2)*np.sin(theta), tz])
    pa = (params[6]*np.pi)
    if np.ndim(R) > 1:
        grid = True
        dim = np.shape(R)
        R = R.reshape(np.product(dim))
        phi = phi.reshape(np.product(dim))
        z = z.reshape(np.product(dim))
        original_z = original_z.reshape(np.product(dim))
    x, y, z = R*np.cos(phi), R*np.sin(phi), z
    x, y, z = transform_zvecpa(np.dstack([x,y,z])[0], zvec,pa)
    xsun,ysun,zsun = transform_zvecpa([_ro,0.,_zo],zvec,pa)
    r_e = np.sqrt(x**2+y**2/params[2]**2+z**2/params[3]**2)
    r_e_sun = np.sqrt(xsun**2+ysun**2/params[2]**2+zsun**2/params[3]**2)
    diskdens = np.exp(-hr*(R-_ro)-hz*np.fabs(original_z))
    diskdens_sun = np.exp(-hr*(_ro-_ro)-hz*np.fabs(_zo))
    dens = (r_e)**(-params[0])*np.exp(-params[1]*r_e)
    sundens = (r_e_sun)**(-params[0])*np.exp(-params[1]*r_e_sun)
    if split:
        dens, diskdens = (1-params[7])*dens/sundens, (params[7])*diskdens/diskdens_sun
        if grid:
            dens = dens.reshape(dim)
            diskdens = diskdens.reshape(dim)
        return dens, diskdens
    else:
        dens = (1-params[7])*dens/sundens+(params[7]*diskdens/diskdens_sun)
        if grid:
            dens = dens.reshape(dim)
        return dens

def triaxial_broken_angle_zvecpa_plusexpdisk(R,phi,z,
    params=[2.,3.,5.,0.5,0.5,0.,0.,0.,0.01],split=False):
    '''triaxial_broken_angle_zvecpa_zvecpa_plusexpdisk:
    
    Triaxial broken angle power-law density profile rotated using the zvec-pa 
    scheme (see transform_zvecpa) with exponential disk contamination. Note 
    that zvec is parameterized using two parameters eta and theta as follows:
    
    zvec = [ sqrt(1-eta^2)*cos(theta) ]
           [ sqrt(1-eta^2)*sin(theta) ]
           [ eta                      ]
    
    Note that the parameters for the disk are fixed
    
    Args: 
        R, phi, z (np.arrays) - Galactocentric cylindrical coordinates
        params (float array) - [alpha,beta,p,q,eta,theta,pa,fdisk]
            alpha_in (float) - Inner power law index
            alpha_out (float) - Outer power law index
            beta (float) - Radius where the power law index changes
            p (float) - Ratio of Y to X scale lengths
            q (float) - Ratio of Z to X scale lengths
            theta (float) - Sets scale / orientation of zvec in XY plane
            eta (float) - Sets scale of zvec along Z-axis
            pa (float) - Final rotation angle
            fdisk (float) - Fraction of density at the location of the Sun
                contained in the disk, halo density fraction is then (1-fdisk)
        
    Returns
        dens (np.array) - density at coordinates (normalized to 1 at _ro,_zo)
    '''
    hr = 1/2. # Fixed disk parameters
    hz = 1/0.8
    original_z = np.copy(z)
    grid = False
    theta = params[5]*2*np.pi
    tz = (params[6]*2)-1
    zvec = np.array([np.sqrt(1-tz**2)*np.cos(theta), np.sqrt(1-tz**2)*np.sin(theta), tz])
    pa = (params[7]*np.pi)
    if np.ndim(R) > 1:
        grid = True
        dim = np.shape(R)
        R = R.reshape(np.product(dim))
        phi = phi.reshape(np.product(dim))
        z = z.reshape(np.product(dim))
        original_z = original_z.reshape(np.product(dim))
    x, y, z = R*np.cos(phi), R*np.sin(phi), z
    x, y, z = transform_zvecpa(np.dstack([x,y,z])[0], zvec,pa)
    xsun,ysun,zsun = transform_zvecpa([_ro,0.,_zo],zvec,pa)
    r_e = np.sqrt(x**2+y**2/params[3]**2+z**2/params[4]**2)
    r_e_sun = np.sqrt(xsun**2+ysun**2/params[3]**2+zsun**2/params[4]**2)
    diskdens = np.exp(-hr*(R-_ro)-hz*np.fabs(original_z))
    diskdens_sun = np.exp(-hr*(_ro-_ro)-hz*np.fabs(_zo))
    dens = np.zeros(len(r_e))
    dens[r_e < params[2]] = (r_e[r_e < params[2]])**(-params[0])
    dens[r_e > params[2]] = (params[2])**(params[1]-params[0])*(r_e[r_e > params[2]])**(-params[1])
    if params[2] < r_e_sun:
        sundens = (params[2])**(params[1]-params[0])*(r_e_sun)**(-params[1])
    else:
        sundens = (r_e_sun)**(-params[0])
    if split:
        dens, diskdens = (1-params[8])*dens/sundens, (params[8])*diskdens/diskdens_sun
        if grid:
            dens = dens.reshape(dim)
            diskdens = diskdens.reshape(dim)
        return dens, diskdens
    else:
        dens = (1-params[8])*dens/sundens+(params[8]*diskdens/diskdens_sun)
        if grid:
            dens = dens.reshape(dim)
        return dens

def triaxial_single_trunc_zvecpa_plusexpdisk(R,phi,z,
    params=[2.,10.,0.5,0.5,0.,0.,0.,0.01],split=False):
    '''triaxial_single_trunc_zvecpa_zvecpa_plusexpdisk:
    
    Triaxial truncated (density goes to 0 at some radius) density profile 
    rotated using the zvec-pa scheme (see transform_zvecpa) with exponential 
    disk contamination. Note that zvec is parameterized using two parameters 
    eta and theta as follows:
    
    zvec = [ sqrt(1-eta^2)*cos(theta) ]
           [ sqrt(1-eta^2)*sin(theta) ]
           [ eta                      ]
    
    Note that the parameters for the disk are fixed
    
    Args: 
        R, phi, z (np.arrays) - Galactocentric cylindrical coordinates
        params (float array) - [alpha,beta,p,q,eta,theta,pa,fdisk]
            alpha (float) - Inner power law index
            beta (float) - Radius where the density goes to 0
            p (float) - Ratio of Y to X scale lengths
            q (float) - Ratio of Z to X scale lengths
            theta (float) - Sets scale / orientation of zvec in XY plane
            eta (float) - Sets scale of zvec along Z-axis
            pa (float) - Final rotation angle
            fdisk (float) - Fraction of density at the location of the Sun
                contained in the disk, halo density fraction is then (1-fdisk)
        
    Returns
        dens (np.array) - density at coordinates (normalized to 1 at _ro,_zo)
    '''
    hr = 1/2. # Fixed disk parameters
    hz = 1/0.8
    original_z = np.copy(z)
    grid = False
    theta = params[4]*2*np.pi
    tz = (params[5]*2)-1
    zvec = np.array([np.sqrt(1-tz**2)*np.cos(theta), np.sqrt(1-tz**2)*np.sin(theta), tz])
    pa = (params[6]*np.pi)
    if np.ndim(R) > 1:
        grid = True
        dim = np.shape(R)
        R = R.reshape(np.product(dim))
        phi = phi.reshape(np.product(dim))
        z = z.reshape(np.product(dim))
        original_z = original_z.reshape(np.product(dim))
    x, y, z = R*np.cos(phi), R*np.sin(phi), z
    x, y, z = transform_zvecpa(np.dstack([x,y,z])[0], zvec,pa)
    xsun,ysun,zsun = transform_zvecpa([_ro,0.,_zo],zvec,pa)
    r_e = np.sqrt(x**2+y**2/params[2]**2+z**2/params[3]**2)
    r_e_sun = np.sqrt(xsun**2+ysun**2/params[2]**2+zsun**2/params[3]**2)
    diskdens = np.exp(-hr*(R-_ro)-hz*np.fabs(original_z))
    diskdens_sun = np.exp(-hr*(_ro-_ro)-hz*np.fabs(_zo))
    dens = np.zeros(len(r_e))
    dens[r_e < params[1]] = (r_e[r_e < params[1]])**(-params[0])
    dens[r_e > params[1]] = 0
    sundens = (r_e_sun)**(-params[0])
    if split:
        dens, diskdens = (1-params[7])*dens/sundens, (params[7])*diskdens/diskdens_sun
        if grid:
            dens = dens.reshape(dim)
            diskdens = diskdens.reshape(dim)
        return dens, diskdens
    else:
        dens = (1-params[7])*dens/sundens+(params[7]*diskdens/diskdens_sun)
        if grid:
            dens = dens.reshape(dim)
        return dens

# def triaxial_einasto_zvecpa(R,phi,z,params=[10.,3.,0.8,0.8,0.,0.99,0.]):
#     """
#     triaxial einasto profile, with zvec,pa rotation
#     INPUT
#         R, phi, z - Galactocentric cylindrical coordinates
#         params - [n,r_eb,p,q,theta,phi,pa]
#     OUTPUT
#         density at R, phi, z
#     """
#     grid = False
#     r_eb = params[0]
#     n = params[1]
#     p = params[2]
#     q = params[3]
#     theta = params[4]*2*np.pi
#     tz = (params[5]*2)-1
#     zvec = np.array([np.sqrt(1-tz**2)*np.cos(theta), np.sqrt(1-tz**2)*np.sin(theta), tz])
#     pa = (params[6]*np.pi)
#     if np.ndim(R) > 1:
#         grid = True
#         dim = np.shape(R)
#         R = R.reshape(np.product(dim))
#         phi = phi.reshape(np.product(dim))
#         z = z.reshape(np.product(dim))
#     x, y, z = R*np.cos(phi), R*np.sin(phi), z
#     x, y, z = transform_zvecpa(np.dstack([x,y,z])[0], zvec,pa)
#     xsun,ysun,zsun = transform_zvecpa([_R0,0.,_z0],zvec,pa)
#     r_e = np.sqrt(x**2+y**2/p**2+z**2/q**2)
#     r_e_sun = np.sqrt(xsun**2+ysun**2/p**2+zsun**2/q**2)
#     dn = 3*n - 1./3. + 0.0079/n
#     dens = np.exp(-dn*((r_e/r_eb)**(1/n)-1))
#     sundens = np.exp(-dn*((r_e_sun/r_eb)**(1/n)-1))
#     dens = dens/sundens
#     if grid:
#         dens = dens.reshape(dim)
#     return dens

# def triaxial_einasto_zvecpa_plusexpdisk(R,phi,z,params=[10.,3.,0.8,0.8,0.,0.99,0.,0.], split=False):
#     """
#     triaxial einasto profile, with zvec,pa rotation plus expdisk contaminant
#     INPUT
#         R, phi, z - Galactocentric cylindrical coordinates
#         params - [n,r_eb,p,q,theta,phi,pa,fdisc]
#     OUTPUT
#         density at R, phi, z
#     """
#     grid = False
#     original_z = np.copy(z)
#     r_eb = params[0]
#     n = params[1]
#     p = params[2]
#     q = params[3]
#     theta = params[4]*2*np.pi
#     tz = (params[5]*2)-1
#     zvec = np.array([np.sqrt(1-tz**2)*np.cos(theta), np.sqrt(1-tz**2)*np.sin(theta), tz])
#     pa = (params[6]*np.pi)
#     if np.ndim(R) > 1:
#         grid = True
#         dim = np.shape(R)
#         R = R.reshape(np.product(dim))
#         phi = phi.reshape(np.product(dim))
#         z = z.reshape(np.product(dim))
#         original_z = original_z.reshape(np.product(dim))
#     x, y, z = R*np.cos(phi), R*np.sin(phi), z
#     x, y, z = transform_zvecpa(np.dstack([x,y,z])[0], zvec,pa)
#     xsun,ysun,zsun = transform_zvecpa([_R0,0.,_z0],zvec,pa)
#     r_e = np.sqrt(x**2+y**2/p**2+z**2/q**2)
#     r_e_sun = np.sqrt(xsun**2+ysun**2/p**2+zsun**2/q**2)
#     dn = 3.*n - 1./3. + 0.0079/n
#     dens = np.exp(-dn*((r_e/r_eb)**(1/n)-1))
#     sundens = np.exp(-dn*((r_e_sun/r_eb)**(1/n)-1))
#     hr = 1/2.
#     hz = 1/0.8
#     diskdens = np.exp(-hr*(R-_R0)-hz*np.fabs(original_z))
#     diskdens_sun = np.exp(-hr*(_R0-_R0)-hz*np.fabs(_z0))
#     if split:
#         dens, diskdens = (1-params[7])*dens/sundens, (params[7])*diskdens/diskdens_sun
#         if grid:
#             dens = dens.reshape(dim)
#             diskdens = diskdens.reshape(dim)
#         return dens, diskdens
#     else:
#         dens = (1-params[7])*dens/sundens+(params[7]*diskdens/diskdens_sun)
#         if grid:
#             dens = dens.reshape(dim)
#         return dens
#     return dens
