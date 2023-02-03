# ----------------------------------------------------------------------------
#
# TITLE - fit_GE_mass_mock.py
# AUTHOR - James Lane
# PROJECT - ges-mass
#
# -----------------------------------------------------------------------------

### Imports

import numpy as np
from galpy.util import _rotate_to_arbitrary_vector
import warnings

# Scales
_ro = 8.275 # Gravity Collab.
_zo = 0.0208 # Bennett and Bovy

# Normalization convention
norm_eta = lambda x : x # (x+1)/2
denorm_eta = lambda x : x # 2*x-1
theta_scale = 2*np.pi
phi_scale = np.pi
rad_to_degr = 180./np.pi

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
    if 'triaxial_single_cutoff_zvecpa' in model.__name__: # Also inverse
        # params are [alpha,r1 or beta1,p,q,theta,eta,pa]
        # theta is [0,2pi], eta is [-1,1], pa is [0,pi]
        params_out = np.array([params[:,0],
                               params[:,1],
                               params[:,2],
                               params[:,3],
                               params[:,4]/theta_scale,
                               norm_eta(params[:,5]), 
                               params[:,6]/phi_scale
                              ]).T
    if 'triaxial_broken_angle_zvecpa' in model.__name__: # Also inverse
        # params are [alpha_in,alpha_out,r1 or beta1,p,q,theta,eta,pa]
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

def denormalize_parameters(params,model,theta_in_degr=False,phi_in_degr=False):
    '''denormalize_parameters:
    
    Transform parameters from a normalized [0,1] domain to a finite domain
    
    Args:
        params (list) - Normalized density function parameters with shape (n,)
        model (callable) - Density function
        theta_in_degr (bool) - If physical_units return theta in degrees 
            instead of radians
        phi_in_degr (bool) - If physical_units return phi in degrees instead
            of radians
        
    
    Returns:
        params (list) - Density function parameters
    '''
    # Handle degrees and radians
    _theta_scale = float(theta_scale)
    if theta_in_degr:
        _theta_scale *= rad_to_degr
    _phi_scale = float(phi_scale)
    if phi_in_degr:
        _phi_scale *= rad_to_degr
    
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
                               params[:,3]*_theta_scale,
                               denorm_eta(params[:,4]),
                               params[:,5]*_phi_scale
                              ]).T
    if 'triaxial_single_cutoff_zvecpa' in model.__name__: # Also inverse
        # params are [alpha,r1 or beta1,p,q,theta,eta,pa]
        # theta is [0,2pi], eta is [-1,1], pa is [0,pi]
        params_out = np.array([params[:,0],
                               params[:,1],
                               params[:,2],
                               params[:,3],
                               params[:,4]*_theta_scale,
                               denorm_eta(params[:,5]),
                               params[:,6]*_phi_scale
                              ]).T
    if 'triaxial_broken_angle_zvecpa' in model.__name__: # Also inverse
        # params are [alpha_in,alpha_out,r1 or beta1,p,q,theta,eta,pa]
        # theta is [0,2pi], eta is [-1,1], pa is [0,pi]
        params_out = np.array([params[:,0],
                               params[:,1],
                               params[:,2],
                               params[:,3],
                               params[:,4],
                               params[:,5]*_theta_scale,
                               denorm_eta(params[:,6]),
                               params[:,7]*_phi_scale
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
                               params[:,7]*_theta_scale,
                               denorm_eta(params[:,8]),
                               params[:,9]*_phi_scale
                              ]).T
    if 'triaxial_single_trunc_zvecpa' in model.__name__:
        # params are [alpha,beta,p,q,theta,eta,pa]
        # theta is [0,2pi], eta is [-1,1], pa is [0,pi]
        params_out = np.array([params[:,0],
                               params[:,1],
                               params[:,2],
                               params[:,3],
                               params[:,4]*_theta_scale,
                               denorm_eta(params[:,5]),
                               params[:,6]*_phi_scale
                              ]).T
    
    if 'plusexpdisk' in model.__name__:
        # Add the disk contamination fraction, assume it's the last parameter
        params_out = np.concatenate((params_out,
                                     np.atleast_2d(params[:,-1]).T),
                                    axis=1)
    
    return params_out


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
        return tgalcenrect[0], tgalcenrect[1], tgalcenrect[2]
    else:
        tgalcenrect = np.einsum('ij,aj->ai', trot, xyz)
        return tgalcenrect[:,0], tgalcenrect[:,1], tgalcenrect[:,2]


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
    dname = densfunc.__name__
    
    # Wrap up param if int
    if isinstance(param,int):
        param = [param,]
    
    # Lists of parameters
    if 'triaxial_single_angle_zvecpa' in dname:
        param_names = np.array(['alpha','p','q','theta','eta','phi'])
    
    elif 'triaxial_single_cutoff_zvecpa' in dname:
        if 'inv' in dname:
            param_names = np.array(['alpha','beta1','p','q','theta','eta',
                                    'phi'])
        else:
            param_names = np.array(['alpha','r1','p','q','theta','eta','phi'])
    
    elif 'triaxial_broken_angle_zvecpa' in dname:
        if 'inv' in dname:
            param_names = np.array(['alpha_in','alpha_out','beta1','p','q',
                                    'theta','eta','phi'])
        else:
            param_names = np.array(['alpha_in','alpha_out','r1','p','q','theta',
                                    'eta','phi'])
        
    elif 'triaxial_double_broken_angle_zvecpa' in dname:
        param_names = np.array(['alpha_in','alpha_mid','alpha_out','r1',
                                'r2','p','q','theta','eta','phi'])
        
    if 'plusexpdisk' in dname:
        param_names = np.concatenate((param_names,['fdisk']))
    
    indx = []
    for i in range(len(param)):
        assert param[i] in param_names, '"'+param[i]+'" not in densfunc list'
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
                                triaxial_norot,
                                triaxial_single_angle_zvecpa, 
                                triaxial_single_cutoff_zvecpa, 
                                triaxial_broken_angle_zvecpa,
                                triaxial_double_broken_angle_zvecpa,
                                triaxial_single_trunc_zvecpa]
    if densfunc in densfuncs_nodisk_already:
        return densfunc
    elif densfunc == triaxial_single_angle_zvecpa_plusexpdisk:
        return triaxial_single_angle_zvecpa
    elif densfunc == triaxial_single_cutoff_zvecpa_plusexpdisk:
        return triaxial_single_cutoff_zvecpa
    elif densfunc == triaxial_broken_angle_zvecpa_plusexpdisk:
        return triaxial_broken_angle_zvecpa
    elif densfunc == triaxial_double_broken_angle_zvecpa_plusexpdisk:
        return triaxial_double_broken_angle_zvecpa
    elif densfunc == triaxial_single_trunc_zvecpa_plusexpdisk:
        return triaxial_single_trunc_zvecpa
    elif densfunc == exp_disk:
        warnings.warn('exp_disk has no component without a disk')
        return None

    
def get_densfunc_mcmc_labels(densfunc, physical_units=False, 
    theta_in_degr=False, phi_in_degr=False):
    '''get_densfunc_mcmc_labels:
    
    Args:
        densfunc (callable) - density function
        physical_units (bool) - Return the labels with physical units attached
        theta_in_degr (bool) - If physical_units return theta in degrees 
            instead of radians
        phi_in_degr (bool) - If physical_units return phi in degrees instead
            of radians
    
    Returns:
        labels (arr) - String array
    '''
    dname = densfunc.__name__
    if  'triaxial_single_angle_zvecpa' in dname:
        labels = [r'$\alpha_{1}$', r'$p$', r'$q$', r'$\theta$', r'$\eta$', 
                  r'$\phi$']
    elif 'triaxial_single_cutoff_zvecpa' in dname:
        if 'inv' in dname:
            labels = [r'$\alpha_{1}$', r'$\beta$', r'$p$', r'$q$', r'$\theta$', 
                      r'$\eta$', r'$\phi$']
        else:
            labels = [r'$\alpha_{1}$', r'$r_{1}$', r'$p$', r'$q$', r'$\theta$', 
                      r'$\eta$', r'$\phi$']
    elif 'triaxial_broken_angle_zvecpa' in dname:
        if 'inv' in dname:
            labels = [r'$\alpha_{1}$', r'$\alpha_{2}$', r'$r_{1}$', r'$p$', 
                      r'$q$', r'$\theta$', r'$\eta$', r'$\phi$']
        else:
            labels = [r'$\alpha_{1}$', r'$\alpha_{2}$', r'$r_{1}$', r'$p$', 
                      r'$q$', r'$\theta$', r'$\eta$', r'$\phi$']
    elif 'triaxial_double_broken_angle_zvecpa' in dname:
        labels = [r'$\alpha_{1}$', r'$\alpha_{2}$', r'$\alpha_{3}$', 
                  r'$r_{1}$', r'$r_{2}$', r'$p$', r'$q$', r'$\theta$', 
                  r'$\eta$', r'$\phi$']
    elif 'triaxial_single_trunc_zvecpa' in dname:
        labels = [r'$\alpha$', r'$r_{1}$', r'$p$', r'$q$', r'$\theta$', 
                  r'$\eta$', r'$\phi$']
                    
    if dname[-11:] == 'plusexpdisk':
        labels.append(r'$f_{disk}$')
    
    if physical_units:
        for i in range(len(labels)):
            if 'beta' in labels[i]:
                labels[i] = labels[i]+r' [kpc$^{-1}$]'
            if 'r_' in labels[i]:
                labels[i] = labels[i]+r' [kpc]'
            if 'theta' in labels[i]:
                if theta_in_degr:
                    labels[i] = labels[i]+' [deg]'
                else:
                    labels[i] = labels[i]+' [rad]'
                
            if 'phi' in labels[i]:
                if phi_in_degr:
                    labels[i] = labels[i]+' [deg]'
                else:
                    labels[i] = labels[i]+' [rad]'
    
    return labels


def get_densfunc_mcmc_init_uninformed(densfunc):
    '''get_densfunc_mcmc_init_uninformed:
    
    Get the initialization for MCMC naively.
    
    Args:
        densfunc (callable) - density function
        
    Returns:
        init (array) - Initial parameters
    '''
    dname = densfunc.__name__
    if  'triaxial_single_angle_zvecpa' in dname:
        init = np.array([2.0, 0.5, 0.5, 0.01, 0.99, 0.01])
    elif 'triaxial_single_cutoff_zvecpa' in dname:
        if 'inv' in dname:
            init = np.array([2.0, 1./20., 0.5, 0.5, 0.01, 0.99, 0.01])
        else:
            init = np.array([2.0, 20., 0.5, 0.5, 0.01, 0.99, 0.01])
    elif 'triaxial_broken_angle_zvecpa' in dname:
        if 'inv' in dname:
            init = np.array([2., 4., 1./20., 0.5, 0.5, 0.01, 0.99, 0.01])
        else:
            init = np.array([2., 4., 20., 0.5, 0.5, 0.01, 0.99, 0.01])
    elif 'triaxial_double_broken_angle_zvecpa' in dname:
        init = np.array([2., 3., 4., 20., 40., 0.5, 0.5, 0.01, 0.99, 0.01])
    elif 'triaxial_single_trunc_zvecpa' in dname:
        init = np.array([2.0, 50., 0.5, 0.5, 0.01, 0.99, 0.01])
    
    if 'plusexpdisk' in dname:
        init = np.concatenate((init,[0.01,]))
    
    return init


def get_densfunc_mcmc_init_source(densfunc):
    '''get_densfunc_mcmc_init_source:
    
    Where to get the init for MCMC? In general profiles with disk contamination 
    inherit from the same profile without disk contamination. More complicated 
    profiles inherit from simpler profiles
    
    Args:
        densfunc (callable) - Density profile that needing init
    
    Returns:
        densfunc_source (callable) - Density profile that init is inherited from
    '''
    corr = {'triaxial_single_angle_zvecpa':None,
            'triaxial_single_angle_zvecpa_plusexpdisk':\
                triaxial_single_angle_zvecpa,
            
            'triaxial_single_cutoff_zvecpa':\
                triaxial_single_angle_zvecpa,
            'triaxial_single_cutoff_zvecpa_inv':\
                triaxial_single_angle_zvecpa,
            'triaxial_single_cutoff_zvecpa_plusexpdisk':\
                triaxial_single_cutoff_zvecpa,
            
            'triaxial_broken_angle_zvecpa':\
                triaxial_single_angle_zvecpa,
            'triaxial_broken_angle_zvecpa_inv':\
                triaxial_single_angle_zvecpa,
            'triaxial_broken_angle_zvecpa_plusexpdisk':\
                triaxial_broken_angle_zvecpa,
            
            'triaxial_broken_angle_zvecpa_inv':\
                triaxial_single_angle_zvecpa,
           
            'triaxial_double_broken_angle_zvecpa':\
                triaxial_broken_angle_zvecpa,
            'triaxial_double_broken_angle_zvecpa_plusexpdisk':\
                triaxial_double_broken_angle_zvecpa,
            }
    return corr[densfunc.__name__]


# def get_densfunc_mcmc_init_informed(densfunc, feh_range, init_type='ML', 
#                                     verbose=False):
#         '''get_densfunc_mcmc_init_informed:
        
#         Get an informed set of parameters to use as init. Normally load the 
#         maximum likelihood set of parameters of the source densprofile. 
#         init_type can be:
#         'ML' - Use the maximum likelihood samples from the source densfunc
#         'uninformed' - Just use default init
        
#         Args:
#             densfunc (callable) - Density profile to get init for
#             feh_range (array) - 2-element array of [feh minimum, feh maximum]
#             init_type (string) - Type of init to load. 'ML' for maximum 
#                 likelihood sample, 'uninformed' for default init
#             verbose (bool) - Be verbose? [default False]
            
#         Returns:
#             init (array) - Init parameters to use
#         '''
#         if densfunc is None:
#             assert densfunc is not None, 'Must have densfunc'
#         if feh_range is None:
#             feh_range = self.feh_range
#             assert feh_range is not None, 'Must have feh_range'
        
#         assert init_type in ['ML','uninformed']

#         if densfunc.__name__ == 'triaxial_single_angle_zvecpa':
#             init_type = 'uninformed'

#         # Unpack
#         feh_min,feh_max = feh_range

#         # Kinematic selection space
#         if isinstance(selec,str): selec=[selec,]
#         selec_suffix = '-'.join(selec)

#         # Get the densfunc that will provide the init
#         densfunc_source = pdens.get_densfunc_mcmc_init_source(densfunc)

#         # Check ML files
#         if init_type=='ML':
#             # Sample & ML filename
#             samples_filename = self.fit_data_dir+'samples.npy'
#             ml_filename = self.fit_data_dir+'mll_aic_bic.npy'
#             if (not os.path.exists(samples_filename)) or\
#                (not os.path.exists(ml_filename)):
#                 warnings.warn('Files required for init_type "ML" not present'
#                               ', changing init_type to "uninformed"')
#                 init_type = 'uninformed'

#         if init_type == 'uninformed':
#             init = pdens.get_densfunc_mcmc_init_uninformed(densfunc)
#         if init_type == 'ML':
#             samples = np.load(samples_filename)
#             _,ml_ind,_,_ = np.load(ml_filename)
#             sample_ml = samples[int(ml_ind)]
#             init = pdens.make_densfunc_mcmc_init_from_source_params( densfunc, 
#                 params_source=sample_ml, densfunc_source=densfunc_source)

#         if verbose:
#             print('init_type: '+str(init_type))
#             print('densfunc_source: '+densfunc_source.__name__)
#             print('fit_data_dir: '+fit_data_dir)

#         return init


def make_densfunc_mcmc_init_from_source_params(densfunc,params_source,
    densfunc_source=None):
    '''make_densfunc_mcmc_init_from_source_params:
    
    densfunc_source needs to match get_densfunc_mcmc_init_source(densfunc)
    
    Args:
        densfunc (callable) - Density profile for which init is being 
            created
        params_source (array) - Parameters from source density profile
        densfunc_source - Density profile that donates params_source
    
    Returns:
        params
    '''
    # A few hardcoded limits that roughly correspond to the user-supplied
    # domain priors.
    _prior_alpha_max = 10.
    _prior_r_max = 55.
    
    dname = densfunc.__name__
    if densfunc_source is not None:
        if not densfunc_source.__name__ ==\
            get_densfunc_mcmc_init_source(densfunc).__name__:
            warnings.warn('densfunc_source does not match densfunc from'
                             'get_densfunc_mcmc_init_source()')
    else:
        densfunc_source = get_densfunc_mcmc_init_source(densfunc)
    
    params = get_densfunc_mcmc_init_uninformed(densfunc)
    if dname == 'triaxial_single_angle_zvecpa':
        return params
    elif dname == 'triaxial_single_cutoff_zvecpa':
        params[0] = params_source[0]
        params[2:] = params_source[1:]
    elif dname == 'triaxial_single_cutoff_zvecpa_inv':
        params[0] = params_source[0]
        params[2:] = params_source[1:]
    elif dname == 'triaxial_broken_angle_zvecpa':
        params[0] = params_source[0]
        params[1] = np.min([params[0]+1.,_prior_alpha_max])
        params[3:] = params_source[1:]
    elif dname == 'triaxial_broken_angle_zvecpa_inv':
        params[0] = params_source[0]
        params[1] = np.min([params[0]+1.,_prior_alpha_max])
        params[3:] = params_source[1:]
    elif dname == 'triaxial_double_broken_angle_zvecpa':
        params[:2] = params_source[:2]
        params[2] = np.min([params[1]+1.,_prior_alpha_max])
        params[3] = params_source[2]
        params[4] = np.min([params[3]+10.,_prior_r_max])
        params[5:] = params_source[3:]
    elif 'plusexpdisk' in dname:
        params[:-1] = params_source
    else:
        warnings.warn('Could not find densfunc')
        return None
    return params

   
def get_densfunc_minimization_constraint(densfunc):
    '''get_densfunc_minimization_bounds:
    
    Get a set of bounds for the minimization routine to start the MCMC
    
    Args:
        densfunc (callable) - density function
        
    Return:
        None
    '''
    return None


def get_default_thick_disk_params():
    '''get_default_thick_disk_params:
    
    Get a default set of thick disk parameters. These are from 
    Mackereth+2017, and used by Mackereth+2020. User can edit these.
    
    Args:
        None
    
    Returns:
        hr (float) - Radial scale length in inverse kpc
        hz (float) - vertical scale length in inverse kpc
    '''
    return 1./2.2, 1./0.8


def check_grid(R,phi,z):
    '''check_grid:
    
    Check if the input R,phi,z are a grid or not, if they are then flatten
    
    Args:
        R, phi, z (np.arrays) - Galactocentric cylindrical coordinates
    
    Returns:
        R, phi, z (np.arrays) - Galactocentric cylindrical coordinates
        grid (bool) - Input data is multidimensional
        dim (array) - Input data dimension array
    '''
    grid = False
    dim = None
    if np.ndim(R) > 1:
        grid = True
        dim = np.shape(R)
        R = R.reshape(np.product(dim))
        phi = phi.reshape(np.product(dim))
        z = z.reshape(np.product(dim))
    return R,phi,z,grid,dim


def make_zvec_r_e(R,phi,z,p,q,theta,eta,pa):
    '''make_zvec_r_e:
    
    Make effective radii for data and Sun
    
    Args:
        R, phi, z (np.arrays) - Galactocentric cylindrical coordinates
        p,q (floats) - Y and Z axis scale lengths
        theta,eta,pa (floats) - zvec transformation parameters
    
    Returns:
        r_e (np.array) - Effective radii after transformation and axis scaling
        r_e_sun (float) - Effective radius of solar position after 
            transformation and axis scaling
    '''
    zvec = np.array([np.sqrt(1-eta**2)*np.cos(theta), 
                     np.sqrt(1-eta**2)*np.sin(theta), 
                     eta])
    x, y, z = R*np.cos(phi), R*np.sin(phi), z
    x, y, z = transform_zvecpa(np.dstack([x,y,z])[0], zvec,pa)
    xsun,ysun,zsun = transform_zvecpa([_ro,0.,_zo],zvec,pa)
    r_e = np.sqrt(x**2+y**2/p**2+z**2/q**2)
    r_e_sun = np.sqrt(xsun**2+ysun**2/p**2+zsun**2/q**2)
    return r_e,r_e_sun
    
# Density models
    
def spherical(R,phi,z,params=[2.,]):
    '''spherical:
    
    general spherical power-law density model
    
    Args: 
        R, phi, z (np.arrays) - Galactocentric cylindrical coordinates
        params (float array) - [alpha,]
            alpha (float) - Power law index
        
    Returns
        dens (np.array) - density at coordinates (normalized to 1 at _ro,_zo)
    '''
    alpha = params[0]
    R,phi,z,grid,dim = check_grid(R,phi,z)
    x, y, z = R*np.cos(phi), R*np.sin(phi), z
    
    dens = np.sqrt(x**2+y**2+z**2)**(-alpha)
    dens = dens/(np.sqrt(_ro**2+_zo**2)**(-alpha))
    if grid:
        dens = dens.reshape(dim)
    return dens


def spherical_cutoff(R,phi,z,params=[2.,0.1]):
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
    alpha,beta = params
    R,phi,z,grid,dim = check_grid(R,phi,z)
    x, y, z = R*np.cos(phi), R*np.sin(phi), z
    r = np.sqrt(x**2+y**2+z**2)
    
    dens = r**(-alpha)*np.exp(-beta*r)
    dens = dens/(np.sqrt(_ro**2+_zo**2)**(-alpha)\
                 *np.exp(-beta*np.sqrt(_ro**2+_zo**2)))
    if grid:
        dens = dens.reshape(dim)
    return dens


def axisymmetric(R,phi,z,params=[2.,0.5]):
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
    alpha,q = params
    grid = False
    R,phi,z,grid,dim = check_grid(R,phi,z)
    x, y, z = R*np.cos(phi), R*np.sin(phi), z
    
    dens = np.sqrt(x**2+y**2+z**2/q**2)**-alpha
    dens = dens/np.sqrt(_ro**2+_zo**2/q**2)**-alpha
    if grid:
        dens = dens.reshape(dim)
    return dens


def triaxial_norot(R,phi,z,params=[2.,0.5,0.5]):
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
    alpha,p,q = params
    R,phi,z,grid,dim = check_grid(R,phi,z)
    x, y, z = R*np.cos(phi), R*np.sin(phi), z
    
    dens = np.sqrt(x**2+y**2/p**2+z**2/q**2)**-alpha
    dens = dens/np.sqrt(_ro**2+_zo**2/q**2)**-alpha
    if grid:
        dens = dens.reshape(dim)
    return dens


def triaxial_single_angle_zvecpa(R,phi,z,params=[2.,0.5,0.5,0.01,0.99,0.01]):
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
    alpha,p,q,theta,eta,pa = denormalize_parameters(
        params, triaxial_single_angle_zvecpa)[0]
    R,phi,z,grid,dim = check_grid(R,phi,z)
    r_e,r_e_sun = make_zvec_r_e(R,phi,z,p,q,theta,eta,pa)
    
    dens = (r_e)**(-alpha)
    sundens = (r_e_sun)**(-alpha)
    dens = dens/sundens 
    if grid:
        dens = dens.reshape(dim)
    return dens


def triaxial_single_cutoff_zvecpa(
    R,phi,z,params=[2.,10.,0.5,0.5,0.01,0.99,0.01]):
    '''triaxial_single_cutoff_zvecpa:
    
    Triaxial power-law density profile with exponential cutoff length (defined 
    as a linear distance) rotated using the zvec-pa scheme (see 
    transform_zvecpa). Note that zvec is parameterized using two parameters 
    eta and theta as follows:
    
    zvec = [ sqrt(1-eta^2)*cos(theta) ]
           [ sqrt(1-eta^2)*sin(theta) ]
           [ eta                      ]
    
    Args: 
        R, phi, z (np.arrays) - Galactocentric cylindrical coordinates
        params (float array) - [alpha,p,q,eta,theta,pa]
            alpha (float) - Power law index
            r1 (float) - Inverse exponential truncation scale
            p (float) - Ratio of Y to X scale lengths
            q (float) - Ratio of Z to X scale lengths
            theta (float) - Sets scale / orientation of zvec in XY plane
            eta (float) - Sets scale of zvec along Z-axis
            pa (float) - Final rotation angle
        
    Returns
        dens (np.array) - density at coordinates (normalized to 1 at _ro,_zo)
    '''
    alpha,r1,p,q,theta,eta,pa = denormalize_parameters(
        params, triaxial_single_cutoff_zvecpa)[0]
    R,phi,z,grid,dim = check_grid(R,phi,z)
    r_e,r_e_sun = make_zvec_r_e(R,phi,z,p,q,theta,eta,pa)
    
    dens = (r_e)**(-alpha)*np.exp(-r_e/r1)
    sundens = (r_e_sun)**(-alpha)*np.exp(-r_e_sun/r1)
    dens = dens/sundens
    if grid:
        dens = dens.reshape(dim)
    return dens


def triaxial_single_cutoff_zvecpa_inv(
    R,phi,z,params=[2.,0.1,0.5,0.5,0.01,0.99,0.01]):
    '''triaxial_single_cutoff_zvecpa_inv:
    
    Triaxial power-law density profile with exponential cutoff scale (defined 
    as the inverse of the distance) and rotated using the zvec-pa scheme 
    (see transform_zvecpa). Note that zvec is parameterized using two 
    parameters eta and theta as follows:
    
    zvec = [ sqrt(1-eta^2)*cos(theta) ]
           [ sqrt(1-eta^2)*sin(theta) ]
           [ eta                      ]
    
    Args: 
        R, phi, z (np.arrays) - Galactocentric cylindrical coordinates
        params (float array) - [alpha,p,q,eta,theta,pa]
            alpha (float) - Power law index
            beta1 (float) - Inverse exponential truncation scale
            p (float) - Ratio of Y to X scale lengths
            q (float) - Ratio of Z to X scale lengths
            theta (float) - Sets scale / orientation of zvec in XY plane
            eta (float) - Sets scale of zvec along Z-axis
            pa (float) - Final rotation angle
        
    Returns
        dens (np.array) - density at coordinates (normalized to 1 at _ro,_zo)
    '''
    alpha,beta1,p,q,theta,eta,pa = denormalize_parameters(
        params, triaxial_single_cutoff_zvecpa_inv)[0]
    R,phi,z,grid,dim = check_grid(R,phi,z)
    r_e,r_e_sun = make_zvec_r_e(R,phi,z,p,q,theta,eta,pa)
    
    dens = (r_e)**(-alpha)*np.exp(-beta1*r_e)
    sundens = (r_e_sun)**(-alpha)*np.exp(-beta1*r_e_sun)
    dens = dens/sundens
    if grid:
        dens = dens.reshape(dim)
    return dens


def triaxial_broken_angle_zvecpa(R,phi,z,
    params=[2.,3.,10.,0.5,0.5,0.01,0.99,0.01],split=False):
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
            r1 (float) - Radius where the power law index changes
            p (float) - Ratio of Y to X scale lengths
            q (float) - Ratio of Z to X scale lengths
            theta (float) - Sets scale / orientation of zvec in XY plane
            eta (float) - Sets scale of zvec along Z-axis
            pa (float) - Final rotation angle
                contained in the disk, halo density fraction is then (1-fdisk)
        
    Returns
        dens (np.array) - density at coordinates (normalized to 1 at _ro,_zo)
    '''
    alpha_in,alpha_out,r1,p,q,theta,eta,pa = denormalize_parameters(
        params, triaxial_broken_angle_zvecpa)[0]
    R,phi,z,grid,dim = check_grid(R,phi,z)
    r_e,r_e_sun = make_zvec_r_e(R,phi,z,p,q,theta,eta,pa)
    
    dens = np.zeros(len(r_e))
    inner_mask = r_e <= r1
    outer_mask = r_e > r1
    r1norm = r1**(alpha_out-alpha_in)
    dens[inner_mask] = (r_e[inner_mask])**(-alpha_in)
    dens[outer_mask] = r1norm*(r_e[outer_mask])**(-alpha_out)
    if r_e_sun <= r1:
        sundens = r_e_sun**(-alpha_in)
    else:
        sundens = r1norm*r_e_sun**(-alpha_out)
    dens = dens/sundens
    if grid:
        dens = dens.reshape(dim)
    return dens


def triaxial_broken_angle_zvecpa_inv(R,phi,z,
    params=[2.,3.,0.1,0.5,0.5,0.01,0.99,0.01],split=False):
    '''triaxial_broken_angle_zvecpa:
    
    Triaxial broken angle power-law density profile with inverse scale lengths 
    rotated using the zvec-pa scheme (see transform_zvecpa). Note that zvec is 
    parameterized using two parameters eta and theta as follows:
    
    zvec = [ sqrt(1-eta^2)*cos(theta) ]
           [ sqrt(1-eta^2)*sin(theta) ]
           [ eta                      ]
    
    Args: 
        R, phi, z (np.arrays) - Galactocentric cylindrical coordinates
        params (float array) - [alpha,beta,p,q,eta,theta,pa,fdisk]
            alpha_in (float) - Inner power law index
            alpha_out (float) - Outer power law index
            beta1 (float) - Inverse radius where the power law index changes
            p (float) - Ratio of Y to X scale lengths
            q (float) - Ratio of Z to X scale lengths
            theta (float) - Sets scale / orientation of zvec in XY plane
            eta (float) - Sets scale of zvec along Z-axis
            pa (float) - Final rotation angle
                contained in the disk, halo density fraction is then (1-fdisk)
        
    Returns
        dens (np.array) - density at coordinates (normalized to 1 at _ro,_zo)
    '''
    alpha_in,alpha_out,beta1,p,q,theta,eta,pa = denormalize_parameters(
        params, triaxial_broken_angle_zvecpa_inv)[0]
    r1 = 1./beta1
    R,phi,z,grid,dim = check_grid(R,phi,z)
    r_e,r_e_sun = make_zvec_r_e(R,phi,z,p,q,theta,eta,pa)
    
    dens = np.zeros(len(r_e))
    inner_mask = r_e <= r1
    outer_mask = r_e > r1
    r1norm = r1**(alpha_out-alpha_in)
    dens[inner_mask] = (r_e[inner_mask])**(-alpha_in)
    dens[outer_mask] = r1norm*(r_e[outer_mask])**(-alpha_out)
    if r_e_sun <= r1:
        sundens = r_e_sun**(-alpha_in)
    else:
        sundens = r1norm*r_e_sun**(-alpha_out)
    dens = dens/sundens
    if grid:
        dens = dens.reshape(dim)
    return dens


def triaxial_double_broken_angle_zvecpa(R,phi,z,
    params=[2.,3.,4.,10.,20.,0.5,0.5,0.,1.,0.],split=False):
    '''triaxial_double_broken_angle_zvecpa:
    
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
            alpha_mid (float) - Middle power law index
            alpha_out (float) - Outer power law index
            r1 (float) - Radius where the power law index changes from inner
                to middle
            r2 (float) - Radius where the power law index changes from middle 
                to outer
            p (float) - Ratio of Y to X scale lengths
            q (float) - Ratio of Z to X scale lengths
            theta (float) - Sets scale / orientation of zvec in XY plane
            eta (float) - Sets scale of zvec along Z-axis
            pa (float) - Final rotation angle
                contained in the disk, halo density fraction is then (1-fdisk)
        
    Returns
        dens (np.array) - density at coordinates (normalized to 1 at _ro,_zo)
    '''
    alpha_in,alpha_mid,alpha_out,r1,r2,p,q,theta,eta,pa = denormalize_parameters(
        params, triaxial_double_broken_angle_zvecpa)[0]
    R,phi,z,grid,dim = check_grid(R,phi,z)
    r_e,r_e_sun = make_zvec_r_e(R,phi,z,p,q,theta,eta,pa)

    dens = np.zeros(len(r_e))
    inner_mask = r_e <= r1
    middle_mask = (r_e > r1) & (r_e <= r2)
    outer_mask = r_e > r2
    r1norm = r1**(alpha_mid-alpha_in)
    r2norm = r2**(alpha_out-alpha_mid)
    dens[inner_mask] = (r_e[inner_mask])**(-alpha_in)
    dens[middle_mask] = r1norm*(r_e[middle_mask])**(-alpha_mid)
    dens[outer_mask] = r1norm*r2norm*(r_e[outer_mask])**(-alpha_out)
    if r_e_sun <= r1:
        sundens = r_e_sun**(-alpha_in)
    elif (r_e_sun > r1) & (r_e_sun <= r2):
        sundens = r1norm*r_e_sun**(-alpha_mid)
    else:
        sundens = r1norm*r2norm*r_e_sun**(-alpha_out)
    dens = dens/sundens
    if grid:
        dens = dens.reshape(dim)
    return dens


def triaxial_single_trunc_zvecpa(R,phi,z,
    params=[2.,10.,0.5,0.5,0.01,0.99,0.01],split=False):
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
            r1 (float) - Radius where the density profile goes to 0
            p (float) - Ratio of Y to X scale lengths
            q (float) - Ratio of Z to X scale lengths
            theta (float) - Sets scale / orientation of zvec in XY plane
            eta (float) - Sets scale of zvec along Z-axis
            pa (float) - Final rotation angle
                contained in the disk, halo density fraction is then (1-fdisk)
        
    Returns
        dens (np.array) - density at coordinates (normalized to 1 at _ro,_zo)
    '''
    alpha,r1,p,q,theta,eta,pa = denormalize_parameters(
        params, triaxial_single_trunc_zvecpa)[0]
    R,phi,z,grid,dim = check_grid(R,phi,z)
    r_e,r_e_sun = make_zvec_r_e(R,phi,z,p,q,theta,eta,pa)
    
    dens = np.zeros(len(r_e))
    inner_mask = r_e <= r1
    outer_mask = r_e > r1
    dens[inner_mask] = (r_e[inner_mask])**(-alpha)
    dens[outer_mask] = 0
    sundens = (r_e_sun)**(-alpha)
    dens = dens/sundens
    if grid:
        dens = dens.reshape(dim)
    return dens


def exp_disk(R,phi,z,params=[1/2.,1/0.8]):
    '''exp_disk:
    
    Exponential disk
    
    Args:
        R, phi, z (np.arrays) - Galactocentric cylindrical coordinates
        params (float array) - [1/hr,1/hz]
            1/hr (float) - Radial scale
            1/hz (float) - Vertical scale
    
    Returns:
        dens (np.array) - density at coordinates (normalized to 1 at _ro,_zo)
    '''
    hr,hz = params
    R,phi,z,grid,dim = check_grid(R,phi,z)
    diskdens = np.exp(-hr*(R-_ro)-hz*np.fabs(z))
    diskdens_sun = np.exp(-hr*(_ro-_ro)-hz*np.fabs(_zo))
    diskdens = diskdens/diskdens_sun
    if grid:
        diskdens = diskdens.reshape(dim)
    return diskdens


def triaxial_single_angle_zvecpa_plusexpdisk(R,phi,z,
    params=[2.,0.5,0.5,0.01,0.99,0.01,0.01],split=False):
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
    fdisk = denormalize_parameters(
        params, triaxial_single_angle_zvecpa_plusexpdisk)[0][-1]
    dens = triaxial_single_angle_zvecpa(R,phi,z,params=params[:-1])
    hr,hz = get_default_thick_disk_params()
    diskdens = exp_disk(R,phi,z,[hr,hz])
    if split:
        return (1-fdisk)*dens, fdisk*diskdens
    else:
        return (1-fdisk)*dens+fdisk*diskdens


def triaxial_single_cutoff_zvecpa_plusexpdisk(R,phi,z,
    params=[2.,0.1,0.5,0.5,0.01,0.99,0.01,0.01],split=False):
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
    fdisk = denormalize_parameters(
        params, triaxial_single_cutoff_zvecpa_plusexpdisk)[0][-1]
    dens = triaxial_single_cutoff_zvecpa(R,phi,z,params=params[:-1])
    hr,hz = get_default_thick_disk_params()
    diskdens = exp_disk(R,phi,z,[hr,hz])
    if split:
        return (1-fdisk)*dens, fdisk*diskdens
    else:
        return (1-fdisk)*dens+fdisk*diskdens


def triaxial_broken_angle_zvecpa_plusexpdisk(R,phi,z,
    params=[2.,3.,10.,0.5,0.5,0.01,0.99,0.01,0.01],split=False):
    '''triaxial_broken_angle_zvecpa_plusexpdisk:
    
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
            r1 (float) - Radius where the power law index changes
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
    fdisk = denormalize_parameters(
        params, triaxial_broken_angle_zvecpa_plusexpdisk)[0][-1]
    dens = triaxial_broken_angle_zvecpa(R,phi,z,params=params[:-1])
    hr,hz = get_default_thick_disk_params()
    diskdens = exp_disk(R,phi,z,[hr,hz])
    if split:
        return (1-fdisk)*dens, fdisk*diskdens
    else:
        return (1-fdisk)*dens+fdisk*diskdens


def triaxial_double_broken_angle_zvecpa_plusexpdisk(R,phi,z,
    params=[2.,3.,4.,10.,20.,0.5,0.5,0.01,0.99,0.01,0.01],split=False):
    '''triaxial_double_broken_angle_zvecpa_plusexpdisk:
    
    Triaxial double broken angle power-law density profile rotated using the 
    zvec-pa scheme (see transform_zvecpa) with exponential disk contamination. 
    Note that zvec is parameterized using two parameters eta and theta as 
    follows:
    
    zvec = [ sqrt(1-eta^2)*cos(theta) ]
           [ sqrt(1-eta^2)*sin(theta) ]
           [ eta                      ]
    
    Note that the parameters for the disk are fixed
    
    Args: 
        R, phi, z (np.arrays) - Galactocentric cylindrical coordinates
        params (float array) - [alpha,beta,p,q,eta,theta,pa,fdisk]
            alpha_in (float) - Inner power law index
            alpha_mid (float) - Middle power law index
            alpha_out (float) - Outer power law index
            r1 (float) - Radius where the power law index changes from inner
                to middle
            r2 (float) - Radius where the power law index changes from middle 
                to outer
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
    fdisk = denormalize_parameters(
        params, triaxial_double_broken_angle_zvecpa_plusexpdisk)[0][-1]
    dens = triaxial_double_broken_angle_zvecpa(R,phi,z,params=params[:-1])
    hr,hz = get_default_thick_disk_params()
    diskdens = exp_disk(R,phi,z,[hr,hz])
    if split:
        return (1-fdisk)*dens, fdisk*diskdens
    else:
        return (1-fdisk)*dens+fdisk*diskdens


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
            r1 (float) - Radius where the density goes to 0
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
    fdisk = denormalize_parameters(
        params, triaxial_single_trunc_zvecpa_plusexpdisk)[0][-1]
    dens = triaxial_single_trunc_zvecpa(R,phi,z,params=params[:-1])
    hr,hz = get_default_thick_disk_params()
    diskdens = exp_disk(R,phi,z,[hr,hz])
    if split:
        return (1-fdisk)*dens, fdisk*diskdens
    else:
        return (1-fdisk)*dens+fdisk*diskdens


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
