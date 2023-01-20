## general kwargs
verbose = True

## HaloFit kwargs (ordering follows HaloFit.__init__)
# allstar and orbs loaded in prep cell
init = None
init_type = 'ML'
# fit_type provided at runtime
mask_disk = True
mask_halo = True
# densfunc, selec provided at runtime
# effsel, effsel_grid, effsel_mask, dmods loaded in prep cell
nwalkers = 100
nit = int(1e6)
ncut = int(1e3)
# usr_log_prior provided at runtime
n_mass = 2000 # int(nwalkers*(nit-ncut))
int_r_range = [2.,70.]
iso = None # Will read from iso_grid_filename
# iso_filename, jkmins loaded in prep cell
# feh_range provided at runtime
# logg_range loaded in config cell
# fit_dir, gap_dir, ksf_dir loaded in prep cell
# version provided at runtime
# ro, vo, zo loaded in config cell

hf_kwargs = {## HaloFit parameters
             'allstar':allstar_nomask,
             'orbs':orbs_nomask,
             'init':init,
             'init_type':init_type,
             # 'fit_type':fit_type, # provided at runtime
             'mask_disk':mask_disk,
             'mask_halo':mask_halo,
             ## _HaloFit parameters
             # 'densfunc':densfunc, # provided at runtime
             # 'selec':selec, # provided at runtime
             'effsel':apof,
             'effsel_mask':apogee_effSF_mask,
             'effsel_grid':effsel_grid,
             'dmods':dmods,
             'nwalkers':nwalkers,
             'nit':nit,
             'ncut':ncut,
             # 'usr_log_prior':usr_log_prior, # provided at runtime
             'n_mass':n_mass,
             'int_r_range':int_r_range,
             'iso':iso,
             'iso_filename':iso_grid_filename,
             'jkmins':jkmins,
             # 'feh_range':feh_range, # provided at runtime
             'logg_range':logg_range,
             'fit_dir':fit_dir,
             'gap_dir':gap_dir,
             'ksf_dir':ksf_dir,
             # 'version':version, # provided at runtime
             'verbose':verbose,
             'ro':ro,
             'vo':vo,
             'zo':zo}

## pmass.fit() function kwargs
# nprocs set in config file
force_fit = True
mle_init = True
just_mle = False
return_walkers = True
optimizer_method = 'Powell'
mass_int_type = 'spherical_grid'
batch_masses = True
make_ml_aic_bic = True
calculate_masses = True
post_optimization = True
mcmc_diagnostic = True

fit_kwargs = {# 'nprocs':nprocs, # Normally given at runtime 
              'force_fit':force_fit,
              'mle_init':mle_init,
              'just_mle':just_mle,
              'return_walkers':return_walkers,
              'optimizer_method':optimizer_method,
              'mass_int_type':mass_int_type,
              'batch_masses':batch_masses,
              'make_ml_aic_bic':make_ml_aic_bic,
              'calculate_masses':calculate_masses,
              'post_optimization':post_optimization,
              'mcmc_diagnostic':mcmc_diagnostic,
              }