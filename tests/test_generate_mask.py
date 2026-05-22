import pytest
import numpy as np
import xarray as xr
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path
from pycmt.core.generate_mask import Maskgenerator, run_workflow

# =========================================================================
# 1. TESTS DE LA CLASSE MASKGENERATOR
# =========================================================================

@patch("pycmt.core.generate_mask.gpd.read_file")
def test_mask_generator_init(mock_read_file):
    mock_gdf = MagicMock()
    mock_gdf.geometry = [MagicMock()]
    mock_gdf.total_bounds = np.array([-17.5, 12.0, -11.5, 16.5])
    mock_read_file.return_value = mock_gdf

    generator = Maskgenerator("fake_shapefile.shp")
    
    assert len(generator.shapes) == 1
    assert np.array_equal(generator.raw_bounds, np.array([-17.5, 12.0, -11.5, 16.5]))

@patch("pycmt.core.generate_mask.gpd.read_file")
def test_align_grid_with_origin_none(mock_read_file):
    mock_gdf = MagicMock()
    mock_gdf.total_bounds = np.array([-17.5, 12.0, -11.5, 16.5])
    mock_read_file.return_value = mock_gdf

    generator = Maskgenerator("fake_shapefile.shp")
    bounds = generator.align_grid(0.1, None, None)
    
    assert np.array_equal(bounds, np.array([-17.5, 12.0, -11.5, 16.5]))

@patch("pycmt.core.generate_mask.gpd.read_file")
def test_align_grid_with_origin_values(mock_read_file):
    mock_gdf = MagicMock()
    mock_gdf.total_bounds = np.array([-17.5, 12.0, -11.5, 16.5])
    mock_read_file.return_value = mock_gdf

    generator = Maskgenerator("fake_shapefile.shp")
    bounds = generator.align_grid(0.25, 0.125, -89.875)
    
    # Vérification que les nouvelles limites calculées sont bien renvoyées
    assert len(bounds) == 4
    assert bounds[0] < -17.5
    assert bounds[2] > -11.5

@patch("pycmt.core.generate_mask.gpd.read_file")
@patch("pycmt.core.generate_mask.rasterize")
@patch("pycmt.core.generate_mask.xr.DataArray")
def test_create_and_save_mask(mock_data_array, mock_rasterize, mock_read_file, tmp_path):
    mock_gdf = MagicMock()
    mock_gdf.total_bounds = np.array([-17.0, 12.0, -12.0, 16.0])
    mock_read_file.return_value = mock_gdf
    
    # Configuration des retours simulés de rasterize pour mask1 et mask2
    mock_rasterize.side_effect = [
        np.ones((16, 20), dtype=np.int32),  # mask1
        np.zeros((16, 20), dtype=np.int32)  # mask2
    ]
    
    mock_ds = MagicMock()
    mock_data_array.return_value = mock_ds
    mock_ds.where.return_value = mock_ds

    generator = Maskgenerator("fake_shapefile.shp")
    output_file = tmp_path / "SEN_mask.nc"
    
    res = generator.create_and_save_mask(0.25, (-17.0, 12.0, -12.0, 16.0), output_file)
    
    assert res == output_file
    mock_ds.to_netcdf.assert_called_once_with(output_file)

# =========================================================================
# 2. TEST DE LA FONCTION RUN_WORKFLOW
# =========================================================================

@patch("pycmt.core.generate_mask.Maskgenerator")
@patch("pycmt.core.generate_mask.os.makedirs")
@patch("builtins.open", new_callable=mock_open)
def test_run_workflow(mock_file, mock_makedirs, mock_mask_gen_cls):
    mock_generator_instance = MagicMock()
    mock_generator_instance.align_grid.return_value = (-17.0, 12.0, -12.0, 16.0)
    mock_generator_instance.create_and_save_mask.return_value = "fake_output_path.nc"
    mock_mask_gen_cls.return_value = mock_generator_instance
    
    run_workflow("SEN", "Senegal")
    
    # Vérification que les 6 résolutions de la configuration ont été traitées
    assert mock_generator_instance.align_grid.call_count == 6
    assert mock_generator_instance.create_and_save_mask.call_count == 6
    assert mock_makedirs.call_count == 6
    
    # Vérification qu'à la fin de la fonction, le fichier d'info latlon a été écrit
    mock_file.assert_called_once()
    mock_file().write.assert_called_once()