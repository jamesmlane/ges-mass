import numpy as np
import sys
import pdb
sys.path.insert(0,'../src/')
from ges_mass import densprofiles as pdens

## Test density profile basic properties

# Test density profiles go to 0 at infinity
def test_densprofile_zero_at_infinity():
    tol = 1e-8
    inf = 1e10
    
    dps = [pdens.triaxial_single_angle_zvecpa,
           pdens.triaxial_single_cutoff_zvecpa,
           pdens.triaxial_broken_angle_zvecpa,
           pdens.triaxial_single_trunc_zvecpa]
    ddps = [pdens.triaxial_single_angle_zvecpa_plusexpdisk,
            pdens.triaxial_single_cutoff_zvecpa_plusexpdisk,
            pdens.triaxial_broken_angle_zvecpa_plusexpdisk,
            pdens.triaxial_single_trunc_zvecpa_plusexpdisk]
    dargs = [[2.,0.5,0.7,0.5,0.5,0.5], # alpha, p, q, theta, eta, phi
             [3.5,0.1,1.,1.,0.5,0.5,0.5], # alpha, beta, p, q, theta, eta, phi
             [2.,4.,20.,0.8,0.5,0.5,0.5,0.5], # alpha_in, alpha_out, beta, p, q, theta, eta, phi
             [5.,20.,0.4,0.8,0.5,0.5,0.5] # alpha, beta, p, q, theta, eta, phi
            ]
    
    for i in range(len(dps)):
        # Non disk profiles
        assert dps[i](inf,0,0,dargs[i]) < tol,\
            'Densprofile '+str(dps[i].__name__)+' does not go to '+\
            '0 at R=infinity'
        assert dps[i](0,0,inf,dargs[i]) < tol,\
            'Densprofile '+str(dps[i].__name__)+' does not go to '+\
            '0 at z=infinity'
        
        # Disk profiles
        ddarg = dargs[i]+[0.2,]
        assert ddps[i](inf,0,0,ddarg) < tol,\
            'Densprofile '+str(ddps[i].__name__)+' does not go to '+\
            '0 at R=infinity'
        assert ddps[i](0,0,inf,ddarg) < tol,\
            'Densprofile '+str(ddps[i].__name__)+' does not go to '+\
            '0 at z=infinity'

# Test power law equivalent for p=q=1 triaxial density profiles
def test_triaxial_densprofile_power_law_equivalent():
    tol = 1e-8
    
    dps = [pdens.triaxial_single_angle_zvecpa,
           pdens.triaxial_single_cutoff_zvecpa,
           pdens.triaxial_broken_angle_zvecpa,
           pdens.triaxial_single_trunc_zvecpa]
    ddps = [pdens.triaxial_single_angle_zvecpa_plusexpdisk,
            pdens.triaxial_single_cutoff_zvecpa_plusexpdisk,
            pdens.triaxial_broken_angle_zvecpa_plusexpdisk,
            pdens.triaxial_single_trunc_zvecpa_plusexpdisk]
    dargs = [[2.,1.,1.,0.5,0.5,0.5], # alpha, p, q, theta, eta, phi
             [3.5,1e-10,1.,1.,0.5,0.5,0.5], # alpha, beta, p, q, theta, eta, phi
             [4.,4.,20.,1.,1.,0.5,0.5,0.5], # alpha_in, alpha_out, beta, p, q, theta, eta, phi
             [5.,1e10,1.,1.,0.5,0.5,0.5] # alpha, beta, p, q, theta, eta, phi
            ]
    
    rs = np.linspace(0.1,100.,num=5)
    zero = np.zeros_like(rs)
    sdp = pdens.spherical
    
    for i in range(len(dps)):
        rhos = pdens.spherical(rs,zero,zero,[dargs[i][0]])
        
        # Non disk profiles
        assert np.all(np.fabs((rhos-dps[i](rs,zero,zero,dargs[i]))/rhos<tol)),\
            'Densprofile '+str(dps[i].__name__)+' is does not match '+\
            'corresponding spherical power law when p=q=1 along R range'
        assert np.all(np.fabs((rhos-dps[i](zero,zero,rs,dargs[i]))/rhos<tol)),\
            'Densprofile '+str(dps[i].__name__)+' is does not match '+\
            'corresponding spherical power law when p=q=1 along z range'
        
        # Disk profiles
        ddarg = dargs[i]+[0.,]
        assert np.all(np.fabs((rhos-ddps[i](rs,zero,zero,ddarg))/rhos<tol)),\
            'Densprofile '+str(ddps[i].__name__)+' is does not match '+\
            'corresponding spherical power law when p=q=1, fdisc=0. along '+\
            'R range'
        assert np.all(np.fabs((rhos-ddps[i](zero,zero,rs,ddarg))/rhos<tol)),\
            'Densprofile '+str(ddps[i].__name__)+' is does not match '+\
            'corresponding spherical power law when p=q=1, fdisc=0. along '+\
            'z range'

# Test spherical symmetry for p=q=1 triaxial density profiles
def test_triaxial_densprofile_spherical_symmetry():
    tol = 1e-10
    dps = [pdens.triaxial_single_angle_zvecpa,
           pdens.triaxial_single_cutoff_zvecpa,
           pdens.triaxial_broken_angle_zvecpa,
           pdens.triaxial_single_trunc_zvecpa]
    ddps = [pdens.triaxial_single_angle_zvecpa_plusexpdisk,
            pdens.triaxial_single_cutoff_zvecpa_plusexpdisk,
            pdens.triaxial_broken_angle_zvecpa_plusexpdisk,
            pdens.triaxial_single_trunc_zvecpa_plusexpdisk]
    dargs = [[2.,1.,1.,0.5,0.5,0.5], # alpha, p, q, theta, eta, phi
             [3.5,1e-10,1.,1.,0.5,0.5,0.5], # alpha, beta, p, q, theta, eta, phi
             [2.,4.,20.,1.,1.,0.5,0.5,0.5], # alpha_in, alpha_out, beta, p, q, theta, eta, phi
             [2.,1e10,1.,1.,0.5,0.5,0.5] # alpha, beta, p, q, theta, eta, phi
            ]
    rs = np.linspace(0.1,100.,num=5)
    phis = np.linspace(0.,2*np.pi,num=11)
    
    for i in range(len(dps)):
        for j in range(len(rs)):            
            # Non disk profiles
            rho = dps[i](rs[j],phis,np.zeros_like(phis),dargs[i])
            assert np.all(np.fabs((rho-np.mean(rho))/np.mean(rho))<tol),\
                'Densprofile '+str(dps[i].__name__)+' is not '+\
                'spherically symmetric when p=q=1'
            
            # Disk profiles
            ddarg = dargs[i]+[0.2,]
            rho = ddps[i](rs[j],phis,np.zeros_like(phis),ddarg)
            assert np.all(np.fabs((rho-np.mean(rho))/np.mean(rho))<tol),\
                'Densprofile '+str(ddps[i].__name__)+' is not '+\
                'spherically symmetric when p=q=1, fdisc=0.'

# Test that the density profiles behave predictably when theta changed.
# theta=0. should equal theta=1. theta=0.
def test_triaxial_densprofile_theta_rotation():
    tol = 1e-10
    
    dps = [pdens.triaxial_single_angle_zvecpa,
           pdens.triaxial_single_cutoff_zvecpa,
           pdens.triaxial_broken_angle_zvecpa,
           pdens.triaxial_single_trunc_zvecpa]
    ddps = [pdens.triaxial_single_angle_zvecpa_plusexpdisk,
            pdens.triaxial_single_cutoff_zvecpa_plusexpdisk,
            pdens.triaxial_broken_angle_zvecpa_plusexpdisk,
            pdens.triaxial_single_trunc_zvecpa_plusexpdisk]
    dargs = [[2.,0.8,0.4,0.,0.5,0.], # alpha, p, q, theta, eta, phi
             [3.5,1e-10,0.5,0.6,0.,0.5,0.], # alpha, beta, p, q, theta, eta, phi
             [2.,4.,20.,0.7,0.3,0.,0.5,0.], # alpha_in, alpha_out, beta, p, q, theta, eta, phi
             [2.,1e10,0.2,0.9,0.,0.5,0.] # alpha, beta, p, q, theta, eta, phi
            ]
            
    rs = np.linspace(0.1,100.,num=11)
    phis = np.linspace(0.,2*np.pi,num=11)
    zs = np.linspace(0.1,10.,num=11)
    
    for i in range(len(dps)):
        rho = dps[i](rs,phis,zs,dargs[i])
        darg_rot = dargs[i]
        darg_rot[-3] = 1.
        rho_rot = dps[i](rs,phis,zs,darg_rot)
        assert np.all(np.fabs((rho-rho_rot)/rho)<tol),\
            'Densprofile '+str(dps[i].__name__)+' is not '+\
            'invariant under a rotation of phi=pi'

# Test that the density profiles behave predictably when phi changed.
# phi=0. should equal phi=1.
def test_triaxial_densprofile_phi_rotation():
    tol = 1e-10
    
    dps = [pdens.triaxial_single_angle_zvecpa,
           pdens.triaxial_single_cutoff_zvecpa,
           pdens.triaxial_broken_angle_zvecpa,
           pdens.triaxial_single_trunc_zvecpa]
    ddps = [pdens.triaxial_single_angle_zvecpa_plusexpdisk,
            pdens.triaxial_single_cutoff_zvecpa_plusexpdisk,
            pdens.triaxial_broken_angle_zvecpa_plusexpdisk,
            pdens.triaxial_single_trunc_zvecpa_plusexpdisk]
    dargs = [[2.,0.8,0.4,0.5,0.5,0.], # alpha, p, q, theta, eta, phi
             [3.5,1e-10,0.5,0.6,0.5,0.5,0.], # alpha, beta, p, q, theta, eta, phi
             [2.,4.,20.,0.7,0.3,0.5,0.5,0.], # alpha_in, alpha_out, beta, p, q, theta, eta, phi
             [2.,1e10,0.2,0.9,0.5,0.5,0.] # alpha, beta, p, q, theta, eta, phi
            ]
            
    rs = np.linspace(0.1,100.,num=11)
    phis = np.linspace(0.,2*np.pi,num=11)
    zs = np.linspace(0.1,10.,num=11)
    
    for i in range(len(dps)):
        rho = dps[i](rs,phis,zs,dargs[i])
        darg_rot = dargs[i]
        darg_rot[-1] = 1.
        rho_rot = dps[i](rs,phis,zs,darg_rot)
        rho_rot = dps[i](rs,phis,zs,darg_rot)
        assert np.all(np.fabs((rho-rho_rot)/rho)<tol),\
            'Densprofile '+str(dps[i].__name__)+' is not '+\
            'invariant under a rotation of phi=pi'