import pytest
import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd
import shapely.geometry as sgeom
import matplotlib.pyplot as plt
from unittest.mock import patch, MagicMock
from pycmt.modules.hydrology import apply_clean_smoothing1, generate_runoff, generate_soilmoisture

# =========================================================================
# 1. TEST UNITAIRE : FONCTION DE LISSAGE (apply_clean_smoothing1)
# =========================================================================
def test_apply_clean_smoothing1():
    data = np.ones((5, 5))
    data[2, 2] = np.nan
    da = xr.DataArray(
        data=data,
        coords=[("lat", np.linspace(0, 10, 5)), ("lon", np.linspace(0, 10, 5))],
        name="test_var"
    )
    da_smoothed = apply_clean_smoothing1(da, sigma=1.0)
    assert isinstance(da_smoothed, xr.DataArray)
    assert np.isnan(da_smoothed.values[2, 2])

# =========================================================================
# 2. TESTS DES PIPELINES RUNOFF ET SOIL MOISTURE
# =========================================================================
@pytest.mark.filterwarnings("ignore:No artists with labels found to put in legend")
@pytest.mark.filterwarnings("ignore:Attempting to set identical low and high")
@patch("pycmt.modules.hydrology.xr.open_dataset")
@patch("pycmt.modules.hydrology.open_CtlDataset")
@patch("pycmt.modules.hydrology.gpd.read_file")
@patch("pycmt.modules.hydrology.plt.savefig")
@patch("pycmt.modules.hydrology.plt.colorbar")  # Neutralise le moteur de colorbar géométrique
@patch("pycmt.modules.hydrology.plt.subplots")  # Neutralise les erreurs de repères Cartopy/Gridlines
def test_hydrology_workflows(
    mock_subplots,
    mock_colorbar,
    mock_savefig,
    mock_read_file,
    mock_open_ctl,
    mock_open_dataset
):
    # --- MOCK DU MASQUE PAYS VIA UN MAGICMOCK COMPLET ---
    mock_mask_ds = MagicMock()
    mock_mask_var = MagicMock()
    mock_mask_var.sortby.return_value = mock_mask_var
    mock_mask_var.interp.return_value = mock_mask_var
    mock_mask_var.fillna.return_value = np.ones((2, 2))
    mock_mask_ds.__getitem__.return_value = mock_mask_var
    mock_open_dataset.return_value = mock_mask_ds

    # --- CONFIGURATION DE VRAIS JEUX DE DONNÉES XARRAY DATASET ---
    dates = pd.date_range(start="1980-01-01", periods=1)
    lon_vals = np.array([14.0, 15.0])
    lat_vals = np.array([12.0, 13.0])

    def create_valid_test_dataset(var_name):
        da = xr.DataArray(
            np.ones((1, 1, 2, 2)),
            coords=[("time", dates), ("lev", [0]), ("lat", lat_vals), ("lon", lon_vals)],
            name=var_name
        )
        return da.to_dataset()

    ds_mask = create_valid_test_dataset("mask")
    ds_r = create_valid_test_dataset("r")
    ds_dm = create_valid_test_dataset("dm")
    # Anticipation de la KeyError : La variable cherchée par Xarray est nommée 'w'
    ds_w = create_valid_test_dataset("w")

    # Intercepteur dynamique pour open_CtlDataset
    def side_effect_open_ctl(file_path):
        path_str = str(file_path)
        if "landmask" in path_str:
            return ds_mask.copy(deep=True)
        elif "drymask" in path_str:
            return ds_dm.copy(deep=True)
        elif "soilmoisture" in path_str:
            # Pour l'humidité du sol, on renvoie le dataset contenant la variable 'w'
            return ds_w.copy(deep=True)
        else:  # Fichiers 'runoff.*.mo.ctl'
            return ds_r.copy(deep=True)

    mock_open_ctl.side_effect = side_effect_open_ctl

    # --- MOCK DE LA COUCHE VECTORIELLE GEOPANDAS ---
    mock_gdf = MagicMock(spec=gpd.GeoDataFrame)
    mock_gdf.geometry.unary_union = sgeom.Polygon([(14, 12), (14, 13), (15, 13), (15, 12)])
    mock_gdf.total_bounds = np.array([14.0, 12.0, 15.0, 13.0])
    mock_gdf.is_valid = [True]
    mock_gdf.to_crs.return_value = mock_gdf
    mock_read_file.return_value = mock_gdf

    # --- MOCK DE L'AFFICHAGE ET DES GRIDS PLOT ---
    mock_fig, mock_ax = MagicMock(), MagicMock()
    mock_subplots.return_value = (mock_fig, mock_ax)
    mock_ax.gridlines.return_value = MagicMock()

    # --- EXÉCUTION DES DEUX WORKFLOWS INDÉPENDANTS ---
    generate_runoff("SEN", "Senegal")
    generate_soilmoisture("SEN", "Senegal")

    # --- VALIDATIONS ---
    assert mock_open_dataset.call_count == 2
    assert mock_read_file.call_count == 2
    # 5 périodes pour Runoff + 5 périodes pour Soil Moisture = 10 cartes attendues
    assert mock_savefig.call_count == 10