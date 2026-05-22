import pytest
import numpy as np
import xarray as xr
from unittest.mock import patch, MagicMock
from pycmt.visualization.generate_maps import plot_vhi_map

# =========================================================================
# 1. CAS DE TEST : LE SHAPEFILE DES FRONTIÈRES EXISTE
# =========================================================================
@patch("pycmt.visualization.generate_maps.get_vhi_colormap", create=True)
@patch("pycmt.visualization.generate_maps.plt.figure")
@patch("pycmt.visualization.generate_maps.plt.axes")
@patch("pycmt.visualization.generate_maps.os.path.exists")
@patch("pycmt.visualization.generate_maps.gpd.read_file")
@patch("pycmt.visualization.generate_maps.plt.savefig")
def test_plot_vhi_map_shp_exists(
    mock_savefig,
    mock_read_file,
    mock_exists,
    mock_axes,
    mock_figure,
    mock_get_cmap
):
    # Configuration des valeurs simulées (Mocks)
    mock_get_cmap.return_value = (MagicMock(), MagicMock(), [0, 6, 12, 24, 36, 48, 60, 72, 84, 100])
    mock_fig = MagicMock()
    mock_figure.return_value = mock_fig
    mock_ax = MagicMock()
    mock_axes.return_value = mock_ax
    mock_exists.return_value = True
    
    mock_gdf = MagicMock()
    mock_read_file.return_value = mock_gdf

    # Création d'un jeu de données DataArray minimal réel (2x2)
    da = xr.DataArray(
        np.ones((2, 2)),
        coords=[("lat", [12.0, 13.0]), ("lon", [14.0, 15.0])],
        name="VHI"
    )
    
    extent_info = {
        "name": "Senegal",
        "lon_range": (-17.5, -11.5),
        "lat_range": (12.0, 16.0)
    }
    
    wk_info = {
        "end_str": "20MAY2026",
        "week_str": "20"
    }

    # Appel de la fonction cible
    plot_vhi_map(da, extent_info, "fake_path_regions.shp", wk_info)

    # Vérifications des appels stratégiques du pipeline
    mock_get_cmap.assert_called_once()
    mock_axes.assert_called_once()
    mock_exists.assert_called_once_with("fake_path_regions.shp")
    mock_read_file.assert_called_once_with("fake_path_regions.shp")
    
    # CORRECTION : assert_any_call intercepte la présence de l'initialisation 
    # n'importe où dans l'historique des appels, filtrant les effets de bord de colorbar()
    mock_figure.assert_any_call(figsize=(11, 8.5))
    
    # Validation du dessin vectoriel Geopandas sur l'axe Cartopy
    mock_gdf.plot.assert_called_once_with(ax=mock_ax, edgecolor='black', facecolor='none', linewidth=0.5, zorder=2)
    mock_savefig.assert_called_once()


# =========================================================================
# 2. CAS DE TEST : LE SHAPEFILE DES FRONTIÈRES EST INTROUVABLE (FALLBACK)
# =========================================================================
@patch("pycmt.visualization.generate_maps.get_vhi_colormap", create=True)
@patch("pycmt.visualization.generate_maps.plt.figure")
@patch("pycmt.visualization.generate_maps.plt.axes")
@patch("pycmt.visualization.generate_maps.os.path.exists")
@patch("pycmt.visualization.generate_maps.plt.savefig")
def test_plot_vhi_map_shp_not_exists(
    mock_savefig,
    mock_exists,
    mock_axes,
    mock_figure,
    mock_get_cmap
):
    # Configuration pour simuler l'absence du fichier shapefile
    mock_get_cmap.return_value = (MagicMock(), MagicMock(), [0, 6, 12, 24, 36, 48, 60, 72, 84, 100])
    mock_fig = MagicMock()
    mock_figure.return_value = mock_fig
    mock_ax = MagicMock()
    mock_axes.return_value = mock_ax
    mock_exists.return_value = False

    da = xr.DataArray(
        np.ones((2, 2)),
        coords=[("lat", [12.0, 13.0]), ("lon", [14.0, 15.0])],
        name="VHI"
    )
    
    extent_info = {
        "name": "Senegal",
        "lon_range": (-17.5, -11.5),
        "lat_range": (12.0, 16.0)
    }
    
    wk_info = {
        "end_str": "20MAY2026",
        "week_str": "20"
    }

    plot_vhi_map(da, extent_info, "missing_path_regions.shp", wk_info)

    # Vérifications
    mock_get_cmap.assert_called_once()
    mock_axes.assert_called_once()
    mock_exists.assert_called_once_with("missing_path_regions.shp")
    
    # CORRECTION : Application du même filtrage d'historique d'appels
    mock_figure.assert_any_call(figsize=(11, 8.5))
    mock_savefig.assert_called_once()