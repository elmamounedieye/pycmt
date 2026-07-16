# PyCMT

PyCMT is a comprehensive Python package designed for downloading rainfall data, computing spatial precipitation distributions, and generating time series visualizations for climate monitoring.

This README is directed at PyCMT users.

# Installation Guide
Follow these steps to set up a clean virtual environment and install all the required dependencies using Conda.

# Prerequisites
Make sure you have Miniconda or Anaconda installed on your system.

# Step 1: Create the Conda Environment
Open your Anaconda Prompt or Miniconda Terminal and execute the following command to create the environment with all necessary geospatial and meteorological packages:

```bash
conda create -n cpc_pycmt_env -c conda-forge python=3.10 geopandas shapely jupyter ipykernel cartopy xarray dask netcdf4 python-wget rasterio bottleneck pillow -y
```

# Step 2: Activate the Environment
Once the installation is complete, activate your newly created environment:

```bash
conda activate cpc_pycmt_env
```

# Step 3: Register the Kernel for Jupyter Notebooks
To use this environment inside Jupyter Notebooks, register it as a custom Python kernel:
```bash
python -m ipykernel install --user --name cpc_pycmt_env --display-name "Python (cpc_pycmt_env)
```
# Step 4: Install PyCMT Package
Install the latest development version of PyCMT directly from the GitHub repository using pip:
```bash
pip install git+https://github.com/elmamounedieye/pycmt.git
```

# Step 5: Run jupyter notebook on your terminal

```bash
jupyter notebook
```
### This version puts the focus only on AFRICA.

Download this  "Test Notebook" to use the PyCMT https://github.com/elmamounedieye/pycmt/blob/adf33d16765cc05bbd792cb166d87c48e20dfefe/Test_notebook.ipynb.

#Kernel choice

Choose "Python (cpc_pycmt_env)" kernel to run the notebook.


### You don't need to use the "run_upload" functionality for Senegal or Africa as area since the package integrates STATIONS' files for both.
