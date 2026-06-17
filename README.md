# PyCMT

PyCMT is a comprehensive Python package designed for downloading rainfall data, computing spatial precipitation distributions, and generating time series visualizations for climate monitoring.

This README is directed at PyCMT users.

# Installation Guide
Follow these steps to set up a clean virtual environment and install all the required dependencies using Conda.

# Prerequisites
Make sure you have Miniconda or Anaconda installed on your system.

# Step 1: Create the Conda Environment
Open your Anaconda Prompt or Miniconda Terminal and execute the following command to create the environment with all necessary geospatial and meteorological packages:

conda create -c conda-forge -c hallkjc01 -n pycmt_env xcast xarray netcdf4 matplotlib cartopy cfgrib jupyter ipykernel -y

# Step 2: Activate the Environment
Once the installation is complete, activate your newly created environment:

conda activate pycmt_env

# Step 3: Register the Kernel for Jupyter Notebooks
To use this environment inside Jupyter Notebooks, register it as a custom Python kernel:

python -m ipykernel install --user --name=pycmt_env

# Step 4: Install PyCMT Package
Install the latest development version of PyCMT directly from the GitHub repository using pip:

pip install git+https://github.com/elmamounedieye/pycmt.git



### This version puts the focus only on AFRICA.

Download this  "Test Notebook" to use the PyCMT https://github.com/elmamounedieye/pycmt/blob/adf33d16765cc05bbd792cb166d87c48e20dfefe/Test_notebook.ipynb.

### You don't need to use the "run_upload" functionality for Senegal or Africa as area since the package integrates STATIONS' files for both.
