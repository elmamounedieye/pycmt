import pytest
import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd
import shapely.geometry as sgeom
from unittest.mock import patch, MagicMock, mock_open

from pycmt.modules.indices import generate_spi, run_orchestrator_spp, do_vhi

# Constantes textuelles virtuelles pour shunter les accès disques physiques
FAKE_LATLON_CONTENT = "Senegal 12.0 16.0 -17.5 -11.5 0.25 0.25"
FAKE_CAPTIONS_CONTENT = (
    "T1\nST1\nT2\nST2\nT3\nST3\n"
    "T4\nST4\nT5\nST5\nT6\nST6\n"
)


# =========================================================================
# 1. TESTS : GENERATE_SPI
# =========================================================================
@pytest.mark.filterwarnings("ignore:No artists with labels found to put in legend")
@pytest.mark.filterwarnings("ignore:Attempting to set identical low and high")
@patch("pycmt.modules.indices.xr.open_dataset")
@patch("pycmt.modules.indices.open_CtlDataset")
@patch("pycmt.modules.indices.gpd.read_file")
@patch("pycmt.modules.indices.plt.savefig")
@patch("pycmt.modules.indices.plt.colorbar")
@patch("pycmt.modules.indices.plt.subplots")
def test_generate_spi_workflow(
    mock_subplots,
    mock_colorbar,
    mock_savefig,
    mock_read_file,
    mock_open_ctl,
    mock_open_dataset
):
    mock_mask_ds = MagicMock()
    mock_mask_var = xr.DataArray(
        np.ones((2, 2)),
        coords=[("lat", [12.0, 13.0]), ("lon", [14.0, 15.0])],
        name="mask_data"
    )
    mock_mask_ds.__getitem__.return_value = mock_mask_var
    mock_open_dataset.return_value = mock_mask_ds

    dates = pd.date_range(start="1980-01-01", periods=1)
    coords = [("time", dates), ("lev", [0]), ("lat", [12.0, 13.0]), ("lon", [14.0, 15.0])]
    
    da_p = xr.DataArray(np.ones((1, 1, 2, 2)), coords=coords, name="p")
    da_dm = xr.DataArray(np.ones((1, 1, 2, 2)), coords=coords, name="dm")

    def side_effect_spi_ctl(file_path):
        path_str = str(file_path)
        if "drymask" in path_str:
            return da_dm.to_dataset()
        return da_p.to_dataset()

    mock_open_ctl.side_effect = side_effect_spi_ctl

    mock_gdf = MagicMock(spec=gpd.GeoDataFrame)
    mock_gdf.geometry.unary_union = sgeom.Polygon([(14, 12), (14, 13), (15, 13), (15, 12)])
    mock_gdf.total_bounds = np.array([14.0, 12.0, 15.0, 13.0])
    mock_gdf.is_valid = [True]
    mock_gdf.to_crs.return_value = mock_gdf
    mock_read_file.return_value = mock_gdf

    mock_fig, mock_ax = MagicMock(), MagicMock()
    mock_subplots.return_value = (mock_fig, mock_ax)
    mock_ax.gridlines.return_value = MagicMock()

    generate_spi("SEN", "Senegal", "cmorph")

    assert mock_open_ctl.call_count == 10
    assert mock_savefig.call_count == 5


# =========================================================================
# 2. TESTS : RUN_ORCHESTRATOR_SPP
# =========================================================================
@pytest.mark.filterwarnings("ignore:Attempting to set identical low and high")
@patch("pycmt.modules.indices.xr.open_dataset")
@patch("pycmt.modules.indices.open_CtlDataset")
@patch("pycmt.modules.indices.gpd.read_file")
@patch("pycmt.modules.indices.plt.savefig")
@patch("pycmt.modules.indices.plt.colorbar")
@patch("pycmt.modules.indices.plt.subplots")
def test_run_orchestrator_spp_with_mask(
    mock_subplots,
    mock_colorbar,
    mock_savefig,
    mock_read_file,
    mock_open_ctl,
    mock_open_dataset
):
    mock_mask_ds = MagicMock()
    mock_mask_var = xr.DataArray(np.ones((2, 2)), coords=[("lat", [12.0, 13.0]), ("lon", [14.0, 15.0])], name="mask_data")
    mock_mask_ds.__getitem__.return_value = mock_mask_var
    mock_open_dataset.return_value = mock_mask_ds

    dates = pd.date_range(start="1980-01-01", periods=1)
    coords = [("time", dates), ("lev", [0]), ("lat", [12.0, 13.0]), ("lon", [14.0, 15.0])]
    
    da_spp1 = xr.DataArray(np.ones((1, 1, 2, 2)), coords=coords, name="spp1")
    da_spp2 = xr.DataArray(np.ones((1, 1, 2, 2)), coords=coords, name="spp2")
    da_spp3 = xr.DataArray(np.ones((1, 1, 2, 2)), coords=coords, name="spp3")
    ds_spp = xr.merge([da_spp1, da_spp2, da_spp3])
    mock_open_ctl.return_value = ds_spp

    mock_gdf = MagicMock(spec=gpd.GeoDataFrame)
    mock_gdf.geometry.unary_union = sgeom.Polygon([(14, 12), (14, 13), (15, 13), (15, 12)])
    mock_gdf.total_bounds = np.array([14.0, 12.0, 15.0, 13.0])
    mock_gdf.is_valid = [True]
    mock_gdf.to_crs.return_value = mock_gdf
    mock_read_file.return_value = mock_gdf

    mock_fig, mock_ax = MagicMock(), MagicMock()
    mock_subplots.return_value = (mock_fig, mock_ax)
    mock_ax.gridlines.return_value = MagicMock()

    def side_effect_open_files(filepath, *args, **kwargs):
        path_str = str(filepath)
        if "spp_timescales" in path_str:
            return mock_open(read_data=FAKE_CAPTIONS_CONTENT).return_value
        return mock_open(read_data=FAKE_LATLON_CONTENT).return_value

    with patch("builtins.open", side_effect=side_effect_open_files):
        run_orchestrator_spp("Senegal", "SEN", "rfe2", mask_enabled=True)

    assert mock_open_ctl.call_count == 6
    assert mock_savefig.call_count == 6


@pytest.mark.filterwarnings("ignore:Attempting to set identical low and high")
@patch("pycmt.modules.indices.open_CtlDataset")
@patch("pycmt.modules.indices.gpd.read_file")
@patch("pycmt.modules.indices.plt.savefig")
@patch("pycmt.modules.indices.plt.subplots")
def test_run_orchestrator_spp_no_mask_and_shp_error(
    mock_subplots,
    mock_savefig,
    mock_read_file,
    mock_open_ctl
):
    mock_read_file.side_effect = RuntimeError("Simulated Shapefile Error")

    dates = pd.date_range(start="1980-01-01", periods=1)
    coords = [("time", dates), ("lev", [0]), ("lat", [12.0, 13.0]), ("lon", [14.0, 15.0])]
    
    da_spp1 = xr.DataArray(np.ones((1, 1, 2, 2)), coords=coords, name="spp1")
    da_spp2 = xr.DataArray(np.ones((1, 1, 2, 2)), coords=coords, name="spp2")
    da_spp3 = xr.DataArray(np.ones((1, 1, 2, 2)), coords=coords, name="spp3")
    mock_open_ctl.return_value = xr.merge([da_spp1, da_spp2, da_spp3])

    mock_fig, mock_ax = MagicMock(), MagicMock()
    mock_subplots.return_value = (mock_fig, mock_ax)
    mock_ax.gridlines.return_value = MagicMock()

    def side_effect_open_files(filepath, *args, **kwargs):
        path_str = str(filepath)
        if "spp_timescales" in path_str:
            return mock_open(read_data=FAKE_CAPTIONS_CONTENT).return_value
        return mock_open(read_data=FAKE_LATLON_CONTENT).return_value

    with patch("builtins.open", side_effect=side_effect_open_files):
        run_orchestrator_spp("Senegal", "SEN", "rfe2", mask_enabled=False)

    assert mock_savefig.call_count == 6


# =========================================================================
# 3. TESTS : DO_VHI
# =========================================================================
@pytest.mark.filterwarnings("ignore:Attempting to set identical low and high")
@patch("pycmt.modules.indices.os.path.exists", return_value=True)  
@patch("pycmt.modules.indices.xr.open_dataset")  
@patch("pycmt.modules.indices.gpd.read_file")
@patch("pycmt.modules.indices.shutil.copy")  
@patch("pycmt.modules.indices.plot_vhi_map")  # Isole le moteur graphique externe
def test_do_vhi_workflow(
    mock_plot_vhi,
    mock_copy,
    mock_read_file,
    mock_xr_open,
    mock_exists
):
    mock_ds_noaa = MagicMock()
    mock_xr_open.return_value = mock_ds_noaa

    mock_gdf = MagicMock(spec=gpd.GeoDataFrame)
    mock_gdf.geometry.unary_union = sgeom.Polygon([(14, 12), (14, 13), (15, 13), (15, 12)])
    mock_gdf.total_bounds = np.array([14.0, 12.0, 15.0, 13.0])
    mock_gdf.is_valid = [True]
    mock_gdf.to_crs.return_value = mock_gdf
    mock_read_file.return_value = mock_gdf

    with patch("builtins.open", mock_open(read_data=FAKE_LATLON_CONTENT)):
        do_vhi("Senegal", "SEN")

    # CORRECTION : La boucle rétroactive tourne sur la fenêtre complète de 6 semaines.
    # On valide qu'il y a bien 6 déplacements de fichiers virtuels et 6 appels de rendu de cartes.
    assert mock_copy.call_count == 6
    assert mock_plot_vhi.call_count == 6