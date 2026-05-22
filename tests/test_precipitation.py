import pytest
import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd
import shapely.geometry as sgeom
from unittest.mock import patch, MagicMock
from pathlib import Path
from pycmt.modules.precipitation import apply_clean_smoothing, plot_precip

# =========================================================================
# 1. TEST UNITAIRE : FONCTION DE LISSAGE (apply_clean_smoothing)
# =========================================================================

def test_apply_clean_smoothing():
    data = np.ones((5, 5))
    data[2, 2] = np.nan
    
    lon = np.linspace(0, 10, 5)
    lat = np.linspace(0, 10, 5)
    
    da = xr.DataArray(
        data=data,
        coords=[("lat", lat), ("lon", lon)],
        name="precip",
        attrs={"unit": "mm"}
    )
    
    da_smoothed = apply_clean_smoothing(da, sigma=1.0)
    
    assert isinstance(da_smoothed, xr.DataArray)
    assert da_smoothed.name == "precip"
    assert da_smoothed.attrs["unit"] == "mm"
    assert np.isnan(da_smoothed.sel(lat=5.0, lon=5.0).values)

# =========================================================================
# 2. TEST D'ORCHESTRATION DU WORKFLOW DE PLOT (plot_precip)
# =========================================================================

@pytest.mark.filterwarnings("ignore:No artists with labels found to put in legend")
@patch("pycmt.modules.precipitation.xr.open_dataset")
@patch("pycmt.modules.precipitation.open_CtlDataset")
@patch("pycmt.modules.precipitation.gpd.read_file")
@patch("pycmt.modules.precipitation.plt.subplots")
@patch("pycmt.modules.precipitation.Path.mkdir")
@patch("pycmt.modules.precipitation.plt.colorbar")  # Ajout du Mock pour intercepter l'appel colorbar
def test_plot_precip_workflow(
    mock_colorbar,
    mock_mkdir,
    mock_subplots,
    mock_read_file,
    mock_open_ctl,
    mock_open_dataset
):
    # --- MOCK ET JEU DE DONNÉES DU MASQUE NC ---
    mock_mask_ds = MagicMock()
    mock_mask_var = MagicMock()
    mock_mask_ds.__getitem__.return_value = mock_mask_var
    mock_open_dataset.return_value = mock_mask_ds

    # --- MOCK DES FICHIERS BINAIRES CTL (DATASET ET SLICES TEMPORELS) ---
    dates = pd.date_range(start="2026-01-01", periods=185)
    
    lon_grid = [14.0, 15.0]
    lat_grid = [12.0, 13.0]
    fake_matrix = np.ones((185, 2, 2))
    
    mock_data_array = xr.DataArray(
        data=fake_matrix,
        coords=[("time", dates), ("lat", lat_grid), ("lon", lon_grid)],
        name="pmer2"
    )
    
    mock_ctl_dataset = MagicMock()
    mock_ctl_dataset.__getitem__.return_value = mock_data_array
    mock_open_ctl.return_value = mock_ctl_dataset

    # --- MOCK DE LA COUCHE VECTORIELLE SHAPEFILE (GADM) ---
    mock_gdf = MagicMock(spec=gpd.GeoDataFrame)
    mock_gdf.geometry = MagicMock()
    
    fake_poly = sgeom.Polygon([(14, 12), (14, 13), (15, 13), (15, 12)])
    mock_gdf.geometry.unary_union = fake_poly
    mock_gdf.geometry.apply.return_value = mock_gdf
    
    mock_gdf.total_bounds = np.array([14.0, 12.0, 15.0, 13.0])
    mock_gdf.is_valid = [True]
    mock_gdf.to_crs.return_value = mock_gdf
    mock_read_file.return_value = mock_gdf

    # --- MOCK DES COMPOSANTS GRAPHIQUES MATPLOTLIB ---
    mock_fig, mock_ax = MagicMock(), MagicMock()
    mock_subplots.return_value = (mock_fig, mock_ax)

    # --- EXÉCUTION DU PIPELINE SPATIAL ---
    with patch("pycmt.modules.precipitation.plt.savefig") as mock_savefig:
        plot_precip("0p25", "0p25", "SEN", "Senegal", "arc2")
        
        # --- VERIFICATIONS DES APPELS ---
        assert mock_open_dataset.call_count == 1
        assert mock_open_ctl.call_count == 2
        assert mock_read_file.call_count == 1
        
        # L'appel à la colorbar doit être intercepté à chaque création de carte
        assert mock_colorbar.call_count == 24
        
        # 6 périodes * 4 types de cartes = 24 images simulées avec succès
        assert mock_savefig.call_count == 24