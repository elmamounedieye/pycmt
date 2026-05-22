import pytest
import numpy as np
import pandas as pd
import xarray as xr
import shapely.geometry as sgeom
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path
from pycmt.visualization.generate_grid import (
    calculate_pixels_corrected,
    get_px,
    generate_pixel_arguments,
    get_exact_pixels,
    check_interior_point,
    plot_pix_coordinates
)

# Constantes textuelles de simulation pour court-circuiter l'accès au disque
FAKE_LATLON_CONTENT = "Senegal 12.0 16.0 -17.5 -11.5 0.25 0.25\n"
FAKE_STNS_CONTENT = "1 14.5 -16.5 Dakar\n"

# Structure de dictionnaire de configuration géographique attendue
columns_structure = ['idx', 'lat', 'lon', 'lat_min', 'lat_max', 'lon_min', 'lon_max', 'px_x', 'px_y', 'name']


# =========================================================================
# 1. TESTS UNITAIRES : LOGIQUE DE CONVERSION DE COORDONNÉES EN PIXELS
# =========================================================================

def test_calculate_pixels_corrected():
    # Validation du positionnement mathématique en pouces vers l'image finale 1200x927
    x_px, y_px = calculate_pixels_corrected(14.5, -16.5, 12.0, 16.0, -17.5, -11.5)
    assert isinstance(x_px, float)
    assert isinstance(y_px, float)
    assert x_px > 0
    assert y_px > 0

def test_get_px():
    # Validation du ratio d'affichage vectoriel brut de l'image de sortie
    x_px, y_px = get_px(14.5, -16.5, 12.0, 16.0, -17.5, -11.5)
    assert x_px == (( -16.5 - (-17.5) ) / ( -11.5 - (-17.5) )) * 1200
    assert y_px == (( 16.0 - 14.5 ) / ( 16.0 - 12.0 )) * 927

def test_check_interior_point():
    # Point situé au cœur du rectangle
    poly = sgeom.box(10, 10, 20, 20)
    assert check_interior_point(15, 15, poly) is True
    # Point situé à l'extérieur
    assert check_interior_point(5, 5, poly) is False


# =========================================================================
# 2. TEST D'ORCHESTRATION : GENERATION DES ARGUMENTS DU MAILLAGE GRID
# =========================================================================

@patch("pycmt.visualization.generate_grid.pd.DataFrame.to_csv")
def test_generate_pixel_arguments_pipeline(mock_to_csv):
    # Simulation des ouvertures en chaînes successives (Fichier latlon puis Fichier stations)
    def side_effect_open(file_path, *args, **kwargs):
        path_str = str(file_path)
        if "stns.txt" in path_str:
            return mock_open(read_data=FAKE_STNS_CONTENT).return_value
        return mock_open(read_data=FAKE_LATLON_CONTENT).return_value

    with patch("builtins.open", side_effect=side_effect_open):
        generate_pixel_arguments(0.25, "SEN", "Senegal")

    # Vérification qu'un DataFrame structuré a bien été converti et écrit sur le disque virtuel
    assert mock_to_csv.call_count == 1


# =========================================================================
# 3. TEST DE RENDU : PRODUCTION DE LA CARTE HTML ET DES COORDONNÉES PIXELS
# =========================================================================

@patch("pycmt.visualization.generate_grid.xr.open_dataset")
@patch("pycmt.visualization.generate_grid.gpd.read_file")
@patch("pycmt.visualization.generate_grid.pd.read_csv")
@patch("pycmt.visualization.generate_grid.plt.subplots")
@patch("pycmt.visualization.generate_grid.plt.savefig")
@patch("pycmt.visualization.generate_grid.json.dump")
@patch("pycmt.visualization.generate_grid.Path.mkdir")
def test_plot_pix_coordinates_pipeline(
    mock_mkdir,
    mock_json_dump,
    mock_savefig,
    mock_subplots,
    mock_read_csv,
    mock_read_file,
    mock_xr_open
):
    # --- MOCK DU MASQUE GEOSPATIAL NETCDF ---
    # Construction d'un xr.DataArray minimal 2x2 simulé en DataFrame
    mock_mask_ds = MagicMock()
    da_mask = xr.DataArray(np.ones((2, 2)), coords=[("lat", [12.0, 13.0]), ("lon", [14.0, 15.0])], name="mask_data")
    mock_mask_ds.__getitem__.return_value = da_mask
    mock_xr_open.return_value = mock_mask_ds

    # --- MOCK DES DATAFRAMES D'ARGUMENTS PIXELS (10 COLONNES EXIGÉES) ---
    # Simulation des lignes renvoyées par le parser pour la grille régulière et les stations météo
    fake_grid_row = [1, 14.5, -16.5, 14.25, 14.75, -16.75, -16.25, 150.0, 200.0, "grid"]
    fake_stn_row = [2, 14.5, -16.5, "Dakar"]
    
    df_pix = pd.DataFrame([fake_grid_row], columns=columns_structure)
    df_stns = pd.DataFrame([fake_stn_row], columns=['idx', 'lat', 'lon', 'name'])
    
    # Distribution successive lors des requêtes d'indexations pandas
    mock_read_csv.side_effect = [df_pix, df_stns]

    # --- MOCK DE LA COUCHE GÉOMÉTRIQUE SHAPEFILE GADM (ADM0 & ADM1) ---
    mock_gdf_adm0 = MagicMock()
    mock_gdf_adm1 = MagicMock()
    
    # Géométrie interne servant au maillage topologique
    fake_poly = sgeom.Polygon([(13, 11), (13, 17), (17, 17), (17, 11)])
    mock_gdf_adm0.geometry.iloc.__getitem__.return_value = fake_poly
    mock_gdf_adm0.geometry.unary_union = fake_poly
    
    # Distribution successive pour adm0.shp puis adm1.shp
    mock_read_file.side_effect = [mock_gdf_adm0, mock_gdf_adm1]

    # --- MOCK DU MOTEUR GRAPHIQUE MATPLOTLIB ---
    mock_fig, mock_ax = MagicMock(), MagicMock()
    mock_subplots.return_value = (mock_fig, mock_ax)
    
    # Configuration des dimensions du canevas pour simulateur get_exact_pixels
    mock_fig.canvas.get_width_height.return_value = (1200, 927)
    
    # Simulation du repère géospatial transData (Géo -> Pixels d'affichage)
    mock_ax.transData.transform.return_value = (400.0, 500.0)

    # --- EXÉCUTION DU SCRIPT DE SYNCHRONISATION ET INTERCEPTION DISQUE ---
    # On isole l'écriture du fichier final avec un mock d'ouverture générique
    with patch("builtins.open", mock_open(read_data=FAKE_LATLON_CONTENT)) as mock_file:
        plot_pix_coordinates("Senegal", "SEN", "arc2")

        # --- VALIDATIONS DES ETAPES ---
        assert mock_xr_open.call_count == 1
        assert mock_read_file.call_count == 2  # Doit charger adm0 et adm1
        assert mock_read_csv.call_count == 2   # Doit charger pixel_args et country_stns
        assert mock_savefig.call_count == 1    # Doit sauvegarder l'image finale de vérification de grille
        assert mock_json_dump.call_count == 1   # Doit générer le dictionnaire formatted_areas au format JSON