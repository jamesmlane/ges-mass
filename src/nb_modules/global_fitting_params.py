# general kwargs
verbose = True

# HaloFit kwargs
init = None
init_type = 'ML'
nwalkers = 100
nit = int(1e6)
ncut = int(1e3)
n_mass = 2000
#n_mass = int(nwalkers*(nit-ncut))
usr_log_prior = None
int_r_range = [2.,70.]
iso = None
mask_disk = True
mask_halo = True

# fit function kwargs
force_fit = True
mle_init = True
just_mle = False
return_walkers = True
mass_int_type = 'spherical_grid'
batch_masses = True
calculate_masses = True
make_ml_aic_bic = True
mcmc_diagnostic = True

hf_kwargs = {'allstar':allstar_nomask,
             'orbs':orbs_nomask,
             'init':init,
             'init_type':init_type,
             'nwalkers':nwalkers,
             'nit':nit,
             'ncut':ncut,
             'effsel':apof,
             'effsel_grid':effsel_grid,
             'effsel_mask':apogee_effSF_mask,
             'dmods':dmods,
             'usr_log_prior':usr_log_prior,
             'n_mass':n_mass,
             'int_r_range':int_r_range,
             'iso':iso,
             'iso_filename':iso_grid_filename,
             'jkmins':jkmins,
             'logg_range':logg_range,
             'mask_disk':mask_disk,
             'mask_halo':mask_halo,
             'fit_dir':fit_dir,
             'gap_dir':gap_dir,
             'ksf_dir':ksf_dir,
             #'verbose':verbose,
             'ro':ro,
             'vo':vo,
             'zo':zo}

fit_data_kwargs = {'force_fit':force_fit,
                   'mle_init':mle_init,
                   'just_mle':just_mle,
                   'return_walkers':return_walkers,
                   'mass_int_type':mass_int_type,
                   'batch_masses':batch_masses,
                   'calculate_masses':calculate_masses,
                   'make_ml_aic_bic':make_ml_aic_bic,
                   'mcmc_diagnostic':mcmc_diagnostic,
                   }