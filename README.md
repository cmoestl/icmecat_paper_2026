# ICMECAT paper 2026

Code for producing the results and figures for the Möstl et al. 2026 ICMECAT paper. This paper is on arxiv at: https://arxiv.org/abs/2512.04730

Everything is produced with the notebook **moestl_icmecat_results.ipynb**, see instructions on top of this file. A .py file of the same name is also available, converted from the .ipynb file.

The conda environment is "dro", see folder /envs. For installation instructions, see the end of this readme.

Figures in the article are in the folder /results, designated as fig1_... , fig_2 ... and available as pdf and png. 

A file containing spacecraft positions as a numpy array in pickle format (positions_2020_all_HEEQ_1h_rad_cm.p) can be found in folder /positions.

Data files for Parker Solar Probe (psp_2018_now_rtn.p) and Solar Orbiter (solo_2020_now_rtn.p) in pickle format as numpy arrays are in folder /data. 

The folder /icmecat contains the data files for the ICMECAT catalog, which can be found in this figshare repository: https://doi.org/10.6084/m9.figshare.6356420 (version 24 on figshare was used for this paper).

How to read and use these files is shown in the notebook moestl_icmecat_results.ipynb.


## Installation

Install python with miniconda:

on Linux:

    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh

on MacOS:

    curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
    bash Miniconda3-latest-MacOSX-x86_64.sh


go to a directory of your choice

    git clone https://github.com/cmoestl/icmecat_paper_2026

Create a conda environment using the "envs/env_dro.yml", and activate the environment:

    conda env create -f env_dro.yml

    conda activate dro
