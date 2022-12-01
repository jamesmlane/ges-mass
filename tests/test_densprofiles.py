import numpy as np
import sys
import pdb
sys.path.insert(0,'../src/')
from ges_mass import densprofiles as pdens
from ges_mass import mass as pmass

## Test density profile basic properties

# Test density profiles go to 0 at infinity
def test_densprofile_zero_at_infinity():
    tol = 1e-8
    inf = 1e10
    
    dps = [pdens.triaxial_single_angle_zvecpa,
           pdens.triaxial_single_cutoff_zvecpa,
           pdens.triaxial_broken_angle_zvecpa,
           pdens.triaxial_double_broken_angle_zvecpa,
           pdens.triaxial_single_trunc_zvecpa]
    ddps = [pdens.triaxial_single_angle_zvecpa_plusexpdisk,
            pdens.triaxial_single_cutoff_zvecpa_plusexpdisk,
            pdens.triaxial_broken_angle_zvecpa_plusexpdisk,
            pdens.triaxial_double_broken_angle_zvecpa_plusexpdisk,
            pdens.triaxial_single_trunc_zvecpa_plusexpdisk]
    dargs = [[2.,0.5,0.7,0.5,0.5,0.5], # a,p,q,th,et,pa
             [3.5,0.1,1.,1.,0.5,0.5,0.5], # a,b,p,q,th,et,pa
             [2.,4.,20.,0.8,0.5,0.5,0.5,0.5], # a1,a2,r1,p,q,th,et,pa
             [2.,3.,4.,20.,40.,0.8,0.5,0.5,0.5,0.5], # a1,a2,a3,r1,r2,p,q,th,et,pa
             [5.,20.,0.4,0.8,0.5,0.5,0.5]] # a,r,p,q,th,et,pa
    
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


# Test power law equivalence for p=q=1 triaxial density profiles
def test_triaxial_densprofile_power_law_equivalent():
    tol = 1e-8
    
    dps = [pdens.triaxial_single_angle_zvecpa,
           pdens.triaxial_single_cutoff_zvecpa,
           pdens.triaxial_broken_angle_zvecpa,
           pdens.triaxial_double_broken_angle_zvecpa,
           pdens.triaxial_single_trunc_zvecpa]
    ddps = [pdens.triaxial_single_angle_zvecpa_plusexpdisk,
            pdens.triaxial_single_cutoff_zvecpa_plusexpdisk,
            pdens.triaxial_broken_angle_zvecpa_plusexpdisk,
            pdens.triaxial_double_broken_angle_zvecpa_plusexpdisk,
            pdens.triaxial_single_trunc_zvecpa_plusexpdisk]
    dargs = [[2.,1.,1.,0.5,0.5,0.5], # a,p,q,th,et,pa
             [3.5,1e-10,1.,1.,0.5,0.5,0.5], # a,b,p,q,th,et,pa
             [4.,4.,20.,1.,1.,0.5,0.5,0.5], # a1,a2,r,p,q,th,et,pa
             [4.,4.,4.,20.,40.,1.,1.,0.5,0.5,0.5], # a1,a2,a3,r1,r2,p,q,th,et,pa
             [5.,1e10,1.,1.,0.5,0.5,0.5]] # a,r,p,q,th,et,pa
    
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
           pdens.triaxial_double_broken_angle_zvecpa,
           pdens.triaxial_single_trunc_zvecpa]
    ddps = [pdens.triaxial_single_angle_zvecpa_plusexpdisk,
            pdens.triaxial_single_cutoff_zvecpa_plusexpdisk,
            pdens.triaxial_broken_angle_zvecpa_plusexpdisk,
            pdens.triaxial_double_broken_angle_zvecpa_plusexpdisk,
            pdens.triaxial_single_trunc_zvecpa_plusexpdisk]
    dargs = [[2.,1.,1.,0.5,0.5,0.5], # a,p,q,th,et,pa
             [3.5,1e-10,1.,1.,0.5,0.5,0.5], # a,b,p,q,th,et,pa
             [2.,4.,20.,1.,1.,0.5,0.5,0.5], # a1,a2,r,p,q,th,et,pa
             [2.,3.,4.,20.,40.,1.,1.,0.5,0.5,0.5], # a1,a2,a3,r1,r2,p,q,th,et,pa
             [2.,1e10,1.,1.,0.5,0.5,0.5] # a,r,p,q,th,et,pa
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
# theta=0. should equal theta=1
def test_triaxial_densprofile_theta_rotation():
    tol = 1e-10
    
    dps = [pdens.triaxial_single_angle_zvecpa,
           pdens.triaxial_single_cutoff_zvecpa,
           pdens.triaxial_broken_angle_zvecpa,
           pdens.triaxial_double_broken_angle_zvecpa,
           pdens.triaxial_single_trunc_zvecpa]
    ddps = [pdens.triaxial_single_angle_zvecpa_plusexpdisk,
            pdens.triaxial_single_cutoff_zvecpa_plusexpdisk,
            pdens.triaxial_broken_angle_zvecpa_plusexpdisk,
            pdens.triaxial_double_broken_angle_zvecpa_plusexpdisk,
            pdens.triaxial_single_trunc_zvecpa_plusexpdisk]
    dargs = [[2.,0.8,0.4,0.,0.5,0.], # a,p,q,th,et,pa
             [3.5,1e-10,0.5,0.6,0.,0.5,0.], # a,b,p,q,th,et,pa
             [2.,4.,20.,0.7,0.3,0.,0.5,0.], # a1,a2,r,p,q,th,et,pa
             [2.,3.,4.,20.,40.,0.6,0.9,0.,0.5,0.5], # a1,a2,a3,r1,r2,p,q,th,et,pa
             [2.,1e10,0.2,0.9,0.,0.5,0.] # a,r,p,q,th,et,pa
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
            'invariant under a rotation of theta=2pi'


# Test that the density profiles behave predictably when phi changed.
# phi=0. should equal phi=1.
def test_triaxial_densprofile_phi_rotation():
    tol = 1e-10
    
    dps = [pdens.triaxial_single_angle_zvecpa,
           pdens.triaxial_single_cutoff_zvecpa,
           pdens.triaxial_broken_angle_zvecpa,
           pdens.triaxial_double_broken_angle_zvecpa,
           pdens.triaxial_single_trunc_zvecpa]
    ddps = [pdens.triaxial_single_angle_zvecpa_plusexpdisk,
            pdens.triaxial_single_cutoff_zvecpa_plusexpdisk,
            pdens.triaxial_broken_angle_zvecpa_plusexpdisk,
            pdens.triaxial_double_broken_angle_zvecpa_plusexpdisk,
            pdens.triaxial_single_trunc_zvecpa_plusexpdisk]
    dargs = [[2.,0.8,0.4,0.5,0.5,0.], # a,p,q,th,et,pa
             [3.5,1e-10,0.5,0.6,0.5,0.5,0.], # a,b,p,q,th,et,pa
             [2.,4.,20.,0.7,0.3,0.5,0.5,0.], # a1,a2,r,p,q,th,et,pa
             [2.,3.,4.,20.,40.,0.6,0.9,0.5,0.5,0.], # a1,a2,a3,r1,r2,p,q,th,et,pa
             [2.,1e10,0.2,0.9,0.5,0.5,0.] # a,r,p,q,th,et,pa
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


# Test that profiles with disk fraction equal counterparts without disk 
# fraction when fdisk=0
def test_profile_fdisk0_correspondance():
    tol = 1e-8
    Rs = np.array([1.,5.,10.,50.,100.,500.])
    phis = np.array([0.1,1.,1.5,2.,2.5,3.])
    zs = np.array([0.,0.,0.2,0.5,-0.2,-0.5])
    npos = len(Rs)
    
    dps = [pdens.triaxial_single_angle_zvecpa,
           pdens.triaxial_single_cutoff_zvecpa,
           pdens.triaxial_broken_angle_zvecpa,
           pdens.triaxial_double_broken_angle_zvecpa,
           pdens.triaxial_single_trunc_zvecpa]
    ddps = [pdens.triaxial_single_angle_zvecpa_plusexpdisk,
            pdens.triaxial_single_cutoff_zvecpa_plusexpdisk,
            pdens.triaxial_broken_angle_zvecpa_plusexpdisk,
            pdens.triaxial_double_broken_angle_zvecpa_plusexpdisk,
            pdens.triaxial_single_trunc_zvecpa_plusexpdisk]
    dargs = [[2.,0.5,0.7,0.5,0.5,0.5], # a,p,q,th,et,pa
             [3.5,0.1,1.,1.,0.5,0.5,0.5], # a,b,p,q,th,et,pa
             [2.,4.,20.,0.8,0.5,0.5,0.5,0.5], # a1,a2,r1,p,q,th,et,pa
             [2.,3.,4.,20.,40.,0.8,0.5,0.5,0.5,0.5], # a1,a2,a3,r1,r2,p,q,th,et,pa
             [5.,1000.,0.4,0.8,0.5,0.5,0.5]] # a,r,p,q,th,et,pa
    
    for i in range(len(dps)):
        for j in range(npos):
            rho_dp = dps[i](Rs[j],phis[j],zs[j],dargs[i])
            rho_ddp_f = ddps[i](Rs[j],phis[j],zs[j],dargs[i]+[0.5,])
            rho_ddp_no_f = ddps[i](Rs[j],phis[j],zs[j],dargs[i]+[0.,])
            
            assert np.fabs(rho_dp-rho_ddp_f)/rho_dp > tol,\
                'Densprofile '+str(dps[i].__name__)+' equals version with '+\
                'disk when fdisk > 0'
            assert np.fabs(rho_dp-rho_ddp_no_f)/rho_dp < tol,\
                'Densprofile '+str(dps[i].__name__)+' does not equal version '+\
                'with disk when fdisk = 0'


# Test the domain prior
def test_densprofile_domain_prior():
    # Default params that should lie within domain
    densfuncs = [pdens.triaxial_single_angle_zvecpa,
                 pdens.triaxial_single_cutoff_zvecpa,
                 pdens.triaxial_broken_angle_zvecpa,
                 pdens.triaxial_double_broken_angle_zvecpa,
                 pdens.triaxial_single_trunc_zvecpa,
                 pdens.triaxial_single_angle_zvecpa_plusexpdisk,
                 pdens.triaxial_single_cutoff_zvecpa_plusexpdisk,
                 pdens.triaxial_broken_angle_zvecpa_plusexpdisk,
                 pdens.triaxial_double_broken_angle_zvecpa_plusexpdisk,
                 pdens.triaxial_single_trunc_zvecpa_plusexpdisk,
                ]
    params = [pdens.get_densfunc_mcmc_init_uninformed(densfuncs[0]),
              pdens.get_densfunc_mcmc_init_uninformed(densfuncs[1]),
              pdens.get_densfunc_mcmc_init_uninformed(densfuncs[2]),
              pdens.get_densfunc_mcmc_init_uninformed(densfuncs[3]),
              pdens.get_densfunc_mcmc_init_uninformed(densfuncs[4]),
              pdens.get_densfunc_mcmc_init_uninformed(densfuncs[5]),
              pdens.get_densfunc_mcmc_init_uninformed(densfuncs[6]),
              pdens.get_densfunc_mcmc_init_uninformed(densfuncs[7]),
              pdens.get_densfunc_mcmc_init_uninformed(densfuncs[8]),
              pdens.get_densfunc_mcmc_init_uninformed(densfuncs[9])
             ]
    ndens = len(densfuncs)
    p_index = [1,2,3,5,2]*2
    q_index = [2,3,4,6,3]*2
    th_index = [3,4,5,7,4]*2
    et_index = [4,5,6,8,5]*2
    pa_index = [5,6,7,9,6]*2
    fdisk_indx = -1
    
    # Test defaults
    for i in range(ndens):
        dns = densfuncs[i]
        pms = params[i]
        assert pmass.domain_prior(dns,pms),\
            'Default MCMC parameters fail domain prior for '+str(dns.__name__)
    
    # Test p
    for i in range(ndens):
        dns = densfuncs[i]
        pms = list(params[i])
        idx = p_index[i]
        pms[idx] = 1. # Allowed
        assert pmass.domain_prior(dns,pms)==True,\
            'p domain prior fails for '+str(dns.__name__)
        pms[idx] = 0.1 # Allowed
        assert pmass.domain_prior(dns,pms)==True,\
            'p domain prior fails for '+str(dns.__name__)
        pms[idx] = 1.1 # not Allowed
        assert pmass.domain_prior(dns,pms)==False,\
            'p domain prior fails for '+str(dns.__name__)
        pms[idx] = 0.05 # not Allowed
        assert pmass.domain_prior(dns,pms)==False,\
            'p domain prior fails for '+str(dns.__name__)
        pms[idx] = -0.1 # not Allowed
        assert pmass.domain_prior(dns,pms)==False,\
            'p domain prior fails for '+str(dns.__name__)
    
    # Test q
    for i in range(ndens):
        dns = densfuncs[i]
        pms = list(params[i])
        idx = q_index[i]
        pms[idx] = 1. # Allowed
        assert pmass.domain_prior(dns,pms)==True,\
            'q domain prior fails for '+str(dns.__name__)
        pms[idx] = 0.1 # Allowed
        assert pmass.domain_prior(dns,pms)==True,\
            'q domain prior fails for '+str(dns.__name__)
        pms[idx] = 0.05 # not Allowed
        assert pmass.domain_prior(dns,pms)==False,\
            'q domain prior fails for '+str(dns.__name__)
        pms[idx] = -0.1 # not Allowed
        assert pmass.domain_prior(dns,pms)==False,\
            'q domain prior fails for '+str(dns.__name__)
    
    # Test theta
    for i in range(ndens):
        dns = densfuncs[i]
        pms = list(params[i])
        idx = th_index[i]
        pms[idx] = 0.999 # Allowed
        assert pmass.domain_prior(dns,pms)==True,\
            'theta domain prior fails for '+str(dns.__name__)
        pms[idx] = 0.001 # Allowed
        assert pmass.domain_prior(dns,pms)==True,\
            'theta domain prior fails for '+str(dns.__name__)
        pms[idx] = 1. # not Allowed
        assert pmass.domain_prior(dns,pms)==False,\
            'theta domain prior fails for '+str(dns.__name__)
        pms[idx] = 0. # not Allowed
        assert pmass.domain_prior(dns,pms)==False,\
            'theta domain prior fails for '+str(dns.__name__)
        pms[idx] = 1.1 # not Allowed
        assert pmass.domain_prior(dns,pms)==False,\
            'theta domain prior fails for '+str(dns.__name__)
        pms[idx] = -0.1 # not Allowed
        assert pmass.domain_prior(dns,pms)==False,\
            'theta domain prior fails for '+str(dns.__name__)
        
    # Test eta
    for i in range(ndens):
        dns = densfuncs[i]
        pms = list(params[i])
        idx = et_index[i]
        pms[idx] = 0.999 # Allowed
        assert pmass.domain_prior(dns,pms)==True,\
            'eta domain prior fails for '+str(dns.__name__)
        pms[idx] = 0.001 # Allowed
        assert pmass.domain_prior(dns,pms)==True,\
            'eta domain prior fails for '+str(dns.__name__)
        pms[idx] = 1. # not Allowed
        assert pmass.domain_prior(dns,pms)==False,\
            'eta domain prior fails for '+str(dns.__name__)
        pms[idx] = 0. # not Allowed
        assert pmass.domain_prior(dns,pms)==False,\
            'eta domain prior fails for '+str(dns.__name__)
        pms[idx] = 1.1 # not Allowed
        assert pmass.domain_prior(dns,pms)==False,\
            'eta domain prior fails for '+str(dns.__name__)
        pms[idx] = -0.1 # not Allowed
        assert pmass.domain_prior(dns,pms)==False,\
            'eta domain prior fails for '+str(dns.__name__)
    
    # Test pa
    for i in range(ndens):
        dns = densfuncs[i]
        pms = list(params[i])
        idx = pa_index[i]
        pms[idx] = 0.999 # Allowed
        assert pmass.domain_prior(dns,pms)==True,\
            'pa domain prior fails for '+str(dns.__name__)
        pms[idx] = 0.001 # Allowed
        assert pmass.domain_prior(dns,pms)==True,\
            'pa domain prior fails for '+str(dns.__name__)
        pms[idx] = 1. # not Allowed
        assert pmass.domain_prior(dns,pms)==False,\
            'pa domain prior fails for '+str(dns.__name__)
        pms[idx] = 0. # not Allowed
        assert pmass.domain_prior(dns,pms)==False,\
            'pa domain prior fails for '+str(dns.__name__)
        pms[idx] = 1.1 # not Allowed
        assert pmass.domain_prior(dns,pms)==False,\
            'pa domain prior fails for '+str(dns.__name__)
        pms[idx] = -0.1 # not Allowed
        assert pmass.domain_prior(dns,pms)==False,\
            'pa domain prior fails for '+str(dns.__name__)
    
    # Test fdisk
    for i in range(ndens):
        dns = densfuncs[i]
        if 'plusexpdisk' not in dns.__name__: continue
        pms = list(params[i])
        idx = -1
        pms[idx] = 0.999 # Allowed
        assert pmass.domain_prior(dns,pms)==True,\
            'pa domain prior fails for '+str(dns.__name__)
        pms[idx] = 0.001 # Allowed
        assert pmass.domain_prior(dns,pms)==True,\
            'pa domain prior fails for '+str(dns.__name__)
        pms[idx] = 1. # Allowed
        assert pmass.domain_prior(dns,pms)==True,\
            'pa domain prior fails for '+str(dns.__name__)
        pms[idx] = 0. # Allowed
        assert pmass.domain_prior(dns,pms)==True,\
            'pa domain prior fails for '+str(dns.__name__)
        pms[idx] = 1.1 # not Allowed
        assert pmass.domain_prior(dns,pms)==False,\
            'pa domain prior fails for '+str(dns.__name__)
        pms[idx] = -0.1 # not Allowed
        assert pmass.domain_prior(dns,pms)==False,\
            'pa domain prior fails for '+str(dns.__name__)
    
    # Test alpha
    for i in range(ndens):
        dns = densfuncs[i]
        if 'cutoff' in dns.__name__: continue
        pms = list(params[i])
        idx = 0
        pms[idx] = -1 # not Allowed
        assert pmass.domain_prior(dns,pms)==False,\
            'alpha domain prior fails for '+str(dns.__name__)
        
        if 'broken_angle' in dns.__name__:
            pms = list(params[i])
            idx = 1
            pms[idx] = -1 # not Allowed
            assert pmass.domain_prior(dns,pms)==False,\
                'alpha domain prior fails for '+str(dns.__name__)
        
        if 'double_broken_angle' in dns.__name__:
            pms = list(params[i])
            idx = 2
            pms[idx] = -1 # not Allowed
            assert pmass.domain_prior(dns,pms)==False,\
                'alpha domain prior fails for '+str(dns.__name__)
    
    ## Test characteristic radii
    
    # Exponential cutoff
    i = 1
    dns = densfuncs[i]
    pms = list(params[i])
    idx = 1
    pms[idx] = -1
    assert pmass.domain_prior(dns,pms)==False,\
        'beta domain prior fails for '+str(dns.__name__)
    
    # Broken power law
    i = 2
    dns = densfuncs[i]
    pms = list(params[i])
    idx = 2
    pms[idx] = -1
    assert pmass.domain_prior(dns,pms)==False,\
        'r domain prior fails for '+str(dns.__name__)
    
    # Double broken power law
    i = 3
    dns = densfuncs[i]
    pms = list(params[i])
    idx = 3
    pms[idx] = -1
    assert pmass.domain_prior(dns,pms)==False,\
        'r1 domain prior fails for '+str(dns.__name__)
    pms = list(params[i])
    idx = 4
    pms[idx] = 1.
    assert pmass.domain_prior(dns,pms)==False,\
        'r2 domain prior fails for '+str(dns.__name__)
    
    # Truncated
    i = 1
    dns = densfuncs[i]
    pms = list(params[i])
    idx = 1
    pms[idx] = -1
    assert pmass.domain_prior(dns,pms)==False,\
        'r domain prior fails for '+str(dns.__name__)
    