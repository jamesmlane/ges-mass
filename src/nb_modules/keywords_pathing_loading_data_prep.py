## Keywords
cdict = putil.load_config_to_dict()
keywords = ['BASE_DIR','APOGEE_DR','APOGEE_RESULTS_VERS','GAIA_DR','NDMOD',
            'DMOD_MIN','DMOD_MAX','LOGG_MIN','LOGG_MAX','FEH_MIN','FEH_MAX',
            'FEH_MIN_GSE','FEH_MAX_GSE','DF_VERSION','KSF_VERSION','NPROCS',
            'RO','VO','ZO']
base_dir,apogee_dr,apogee_results_vers,gaia_dr,ndmod,dmod_min,dmod_max,\
    logg_min,logg_max,feh_min,feh_max,feh_min_gse,feh_max_gse,df_version,\
    ksf_version,nprocs,ro,vo,zo = putil.parse_config_dict(cdict,keywords)
logg_range = [logg_min,logg_max]
feh_range = [feh_min,feh_max]
feh_range_gse = [feh_min_gse,feh_max_gse]
feh_range_all = [feh_min,feh_max_gse]

## Pathing
fit_paths = putil.prepare_paths(base_dir,apogee_dr,apogee_results_vers,gaia_dr,
                                df_version,ksf_version)
data_dir,version_dir,ga_dir,gap_dir,df_dir,ksf_dir,fit_dir = fit_paths

## Filenames
fit_filenames = putil.prepare_filenames(ga_dir,gap_dir,feh_range_gse)
apogee_SF_filename,apogee_effSF_filename,apogee_effSF_mask_filename,\
    iso_grid_filename,clean_kinematics_filename = fit_filenames

## File loading and data preparation
fit_stuff,other_stuff = putil.prepare_fitting(fit_filenames,
    [ndmod,dmod_min,dmod_max], ro,zo,return_other=True)
apogee_effSF_mask,dmap,iso_grid,jkmins,dmods,ds,effsel_grid,apof,\
    allstar_nomask,orbs_nomask = fit_stuff
Rgrid,phigrid,zgrid = effsel_grid