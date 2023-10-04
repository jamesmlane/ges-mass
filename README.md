# Lane, Bovy & Mackereth, (2023, MNRAS)

This repository contains code to replicate the results of our paper.
- [MNRAS](https://academic.oup.com/mnras/article-abstract/526/1/1209/7276634?)
- [arXiv](https://arxiv.org/abs/2306.03084) (updated Sept. 2023)

For questions about the repository use contact information listed in the paper.

## Requirements

The normal `scipy`/`numpy` stack as well as `matplotlib` and `astropy`. We recommend the development version of [galpy](https://github.com/jobovy/galpy), as that may contain recent updates to the spherical distribution function code (and if not at least v1.9). If you need to download APOGEE data we use [apogee](https://github.com/jobovy/apogee) and if you need to download Gaia data we use [gaia_tools](https://github.com/jobovy/gaia_tools). Also uses [emcee](https://emcee.readthedocs.io/en/stable/) for MCMC sampling and [corner](https://corner.readthedocs.io/en/latest/) for corner plots. We use [mwdust](https://github.com/jobovy/mwdust) to interact with dust maps, [apomock](https://github.com/jamesmlane/apomock) for mock generation (optional), and [isodist](https://github.com/jobovy/isodist) to interact with isochrones.

Other packages used include, but these could probably be removed with minimal code changes:
- `dill`
- `tqdm`

The project also has an internally accessed `./src/` directory that contains a lot of code that is accessed by the notebooks. This module is not installed, rather notebooks just path to the directory to access the code.

## Data

- Project requires downloading APOGEE data and Gaia data. If using `apogee` and `gaia_tools` should work well with notebooks as-is.
- Also isochrones from PARSEC v1.2s are required. See paper for description of isochrones.

## Notebooks

Notebooks are separated into 7 stages, plus some extras. All notebooks are located in `./notebooks/`. Most notebooks should hopefully run as-is, but some will require pathing modifications. Sometimes `./fig/`, `./data/`, or `./tables/` directories need to be created, but hopefully each notebook makes them as-needed.

Project-level keywords are contained in `notebooks/config.txt` and are read-in at the beginning of each notebook. The path keywords obviously need to be changed.

- 1-get-data
  - Notebook to get APOGEE DR16 and Gaia DR3 data. Also an old notebook to get Gaia DR2 data.
- 2-make-iso
  - Notebook will wrangle input isochrones. See notebook for how to path to isochrones.
- 3-make-essf
  - Notebooks to make effective survey selection functions. First make the ESSF, then mask. Making the one-isochrone ESSF is done for the mocks. The notebook showing the removal of GCs is simply demonstrative, and a list of fields ignored based on the results of the notebooks is hardcoded into `./src/`
- 4-make-ksf
  - A single notebooks that makes the kinematic selection function. Multiple sets of commented-out code shows how different KSFs are constructed. Refer to paper on details for each.
- 5-prepare-data
  - The data cleaning notebook prepares the raw data for fitting and further analysis. The examination notebook constructs the specific fitting samples. Unfortunately this notebook is a bit confusing in its current state, but should work.
- 6-fit-dens
  - fit_dens_* are notebooks where fitting of various samples is done. plot_fit_* are the corresponding notebooks where plots are generated (non paper quality)
- 7-paper
  - Many notebooks that generate the figures and tables for the paper.
- 8-extra
  - Extra notebooks.

Finally mock generation is done in `notebooks/apo_mocks/`. Beware when generating mocks as it can take a long time, and use a lot of system resources. The scripts show how the mocks can be constructed in pieces to circumvent these issues. See the paper for the specific mock parameters used, and `apomock` documentation for more information about how to generate mocks.