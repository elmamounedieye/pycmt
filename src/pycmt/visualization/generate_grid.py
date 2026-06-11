import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
from pathlib import Path
from shapely.geometry import Point
import matplotlib.ticker as ticker
from matplotlib.patches import PathPatch
#from matplotlib.path import Path as PathPlt
import matplotlib.path as mpath
import matplotlib
matplotlib.use('Agg')
import json
import os





# --- Configuration ---
# --- 1. CONFIGURATION ---

####### Pixel coordinates generation #######
def calculate_pixels_corrected(lat, lon, lat1, lat2, lon1, lon2):
    # Marges standards de GrADS sur une page de 11x8.5 pouces
    parea_left = 1.1    # Marge pour les labels de latitude
    parea_right = 10.5  # Marge à droite
    parea_bottom = 0.75 # Marge pour les labels de longitude
    parea_top = 7.75    # Marge en haut (titre)

    # 1. Calcul de la position relative en pouces (Simulation de 'q w2xy')
    xx = parea_left + (lon - lon1) / (lon2 - lon1) * (parea_right - parea_left)
    yy = parea_bottom + (lat - lat1) / (lat2 - lat1) * (parea_top - parea_bottom)

    # 2. Conversion vers l'image finale (Base 800x618 agrandie 1.5x pour atteindre 1200x927)
    # On utilise 8.5 - yy pour inverser l'axe Y (0 en haut pour le HTML)
    xxxx1 = (800 / 11.0) * xx * 1.5
    yyyy1 = (618 / 8.5) * (8.5 - yy) * 1.5

    return xxxx1, yyyy1


def get_px(lat, lon, lat1, lat2, lon1, lon2):
    IMG_WIDTH = 1200
    IMG_HEIGHT = 927
    x_ratio = (lon - lon1) / (lon2 - lon1)
    y_ratio = (lat2 - lat) / (lat2 - lat1)
    return x_ratio * IMG_WIDTH, y_ratio * IMG_HEIGHT


def generate_pixel_arguments(
    dtarsl,
    country_iso,
    country_name
    ):
    """
    Génère un fichier de paramètres de pixels
    à partir d'une grille régulière définie
    par :
        - lat1, lat2
        - lon1, lon2
        - dtarsl (pas de grille)
    """

    # =========================================================
    # 1. PATHS
    # =========================================================
    latlon_path = Path(__file__).resolve().parents[1] / "data" / f"{country_name}_latlon"
    stn_file_path = Path(__file__).resolve().parents[1] / "data" / f"{country_name}_stns.txt"

    output_file = Path(__file__).resolve().parents[1] / "data" / f"pixelargs_{country_name}.txt"

    # =========================================================
    # 2. LECTURE DES LIMITES
    # =========================================================
    with open(latlon_path, 'r') as f:

        lines = f.readline().split()

        country = lines[0]

        lat1, lat2 = float(lines[1]), float(lines[2])
        lon1, lon2 = float(lines[3]), float(lines[4])

    # =========================================================
    # 3. CONSTRUCTION DE LA GRILLE
    # =========================================================
    # Nord -> Sud
    lats_unique = np.arange(
        max(lat1, lat2),
        min(lat1, lat2) - dtarsl,
        -dtarsl
    )

    # Ouest -> Est
    lons_unique = np.arange(
        min(lon1, lon2),
        max(lon1, lon2) + dtarsl,
        dtarsl
    )

    pixel_data = []

    idx = 1

    # =========================================================
    # 4. TRAITEMENT DE LA GRILLE
    # =========================================================
    for lat in lats_unique:

        for lon in lons_unique:

            lat_r = round(lat, 4)
            lon_r = round(lon, 4)

            px_x, px_y = calculate_pixels_corrected(
                lat_r,
                lon_r,
                lat1,
                lat2,
                lon1,
                lon2
            )

            pixel_data.append({

                'id': idx,

                'lat_cent': f"{lat_r:.2f}",
                'lon_cent': f"{lon_r:.2f}",

                'lat_min': f"{(lat_r - dtarsl):.2f}",
                'lat_max': f"{(lat_r + dtarsl):.2f}",

                'lon_min': f"{(lon_r - dtarsl):.2f}",
                'lon_max': f"{(lon_r + dtarsl):.2f}",

                'px_x': f"{px_x:.2f}",
                'px_y': f"{px_y:.2f}",

                'type': 'grid'
            })

            idx += 1

    # =========================================================
    # 5. TRAITEMENT DES STATIONS
    # =========================================================
    with open(stn_file_path, 'r') as f:

        for line in f:

            parts = line.strip().split()

            s_lat = float(parts[1])
            s_lon = float(parts[2])
            name = parts[3]
            px_x, px_y = get_px(
                s_lat,
                s_lon,
                lat1,
                lat2,
                lon1,
                lon2
            )
            pixel_data.append({

                'id': idx,

                'lat_cent': f"{s_lat:.2f}",
                'lon_cent': f"{s_lon:.2f}",

                'lat_min': f"{(s_lat - dtarsl):.2f}",
                'lat_max': f"{(s_lat + dtarsl):.2f}",

                'lon_min': f"{(s_lon - dtarsl):.2f}",
                'lon_max': f"{(s_lon + dtarsl):.2f}",

                'px_x': f"{px_x:.2f}",
                'px_y': f"{px_y:.2f}",

                'type': name
            })

            idx += 1


    # =========================================================
    # 6. EXPORT FINAL
    # =========================================================
    df_final = pd.DataFrame(pixel_data)
    #file = Path(output_file)
    #file.unlink()
    output_file = Path(__file__).resolve().parents[1] / "data" / f"pixelargs_{country_name}.txt"
    df_final.to_csv(
        output_file,
        sep=' ',
        index=True,
        header=False
    )

    #print(f"✅ Fichier généré : {output_file} ({len(df_final)} lignes)")

    #return output_file


####### Generating grid pixel points########
  # N'oubliez pas l'import en haut du script
def get_exact_pixels(lat, lon, ax, fig):
    # Transformation Data (Géo) -> Display (Pixels)
    x_display, y_display = ax.transData.transform((lon, lat))
    width, height = fig.canvas.get_width_height()
    
    # Inversion Y pour le HTML (0 en haut)
    px_x = x_display
    px_y = height - y_display
    return px_x, px_y

def check_interior_point(lon, lat, area_geom):
    return Point(lon, lat).within(area_geom)



def plot_pix_coordinates(country, country_iso, rndta):
    # --- Chemins ---
    base_data = Path(__file__).resolve().parents[1] / "data"
    country_latlon = base_data / f"{country}_latlon"
    pixel_args_path = base_data / f"pixelargs_{country}.txt" #Path(pixel_args)
    country_stns_path = base_data / f"{country}_stns.txt"
    shpfile_path = base_data / "gis_resources" / "countries" / f"{country_iso}_adm"
    mask_path = base_data / "gis_resources" / f"country_masks0p0375" / "365dcal" / f"{country_iso}_mask.nc"

    # --- Lecture des fichiers ---
    mask_pix = xr.open_dataset(mask_path)
    mask_nc = mask_pix['mask_data'].to_dataframe(name='val').reset_index()

    with open(country_latlon, 'r') as f:
        line = f.readline().split()
        lat1, lat2 = float(line[1]), float(line[2])
        lon1, lon2 = float(line[3]), float(line[4])

    # Lecture sécurisée des DataFrames avec les 10 colonnes bien ordonnées
    columns_structure = ['idx', 'lat', 'lon', 'lat_min', 'lat_max', 'lon_min', 'lon_max', 'px_x', 'px_y', 'name']
    pix_df = pd.read_csv(pixel_args_path, sep=r'\s+', header=None, names=columns_structure)
    stns_df = pd.read_csv(country_stns_path, sep=r'\s+', header=None, names=['idx', 'lat', 'lon', 'name'])

    nstns = len(stns_df)
    stngrd = len(pix_df) - nstns

    fig, ax = plt.subplots(figsize=(12, 9), dpi=100)

    try:
        adm0 = gpd.read_file(shpfile_path / f"{country_iso}_adm0.shp")
        adm1 = gpd.read_file(shpfile_path / f"{country_iso}_adm1.shp")

        #area_geom = adm0.geometry.iloc[0]
        area_geom = adm0.geometry.unary_union
        if not area_geom.is_valid:
            area_geom = area_geom.buffer(0)

        # --- Gestion du Clipping Path ---
        def get_coords(geom):
            if geom.geom_type == 'Polygon':
                return [list(geom.exterior.coords)]
            elif geom.geom_type == 'MultiPolygon':
                return [list(p.exterior.coords) for p in geom.geoms]
            return []

        all_coords = get_coords(area_geom)
        poly_path = mpath.Path.make_compound_path(*[mpath.Path(c) for c in all_coords])
        
        patch = PathPatch(poly_path, transform=ax.transData, facecolor='none', edgecolor='none')
        ax.add_patch(patch)

        # Affichage du masque avec clipping
        mesh = ax.scatter(mask_nc.lon, mask_nc.lat, c=mask_nc.val, cmap='Reds', alpha=0.3, s=1, zorder=1)

        # --- Points de Grille et Stations ---
        grid_points_raw = pix_df.iloc[:stngrd].copy()
        stn_points_raw = pix_df.iloc[stngrd:].copy()

        # --- Filtrage Géospatial (Points Intérieurs Uniquement) ---
        grid_interior_mask = grid_points_raw.apply(
            lambda row: check_interior_point(row['lon'], row['lat'], area_geom), axis=1
        )
        grid_points = grid_points_raw[grid_interior_mask].copy()

        #####Addition
        #grid_points = grid_points_raw.copy()
        stn_points = stn_points_raw.copy()

        """stn_interior_mask = stn_points_raw.apply(
            lambda row: check_interior_point(row['lon'], row['lat'], area_geom), axis=1
        )
        stn_points = stn_points_raw[stn_interior_mask].copy()"""


        # =====================================================================
        # ÉTAPE DE SAUVEGARDE STRICTE DU FICHIER SOURCE AVEC LES 10 COLONNES
        # =====================================================================
        # 1. On concatène les lignes intérieures
        updated_pix_df = pd.concat([grid_points, stn_points_raw], ignore_index=True)
        
        # 2. Sécurité : On force l'alignement exact selon l'en-tête d'origine
        # (idx en colonne 1 ... et name en colonne 10 à sa place)
        updated_pix_df = updated_pix_df[columns_structure]
        
        # 3. Écriture physique propre sur le disque
        updated_pix_df.to_csv(pixel_args_path, sep=' ', header=False, index=False)
        #print(f"💾 {pixel_args_path.name} synchronisé avec succès (Structure 10-colonnes respectée).")
        # =====================================================================

        # Tracé des points filtrés
        ax.scatter(grid_points['lon'], grid_points['lat'], s=20, c='cyan', label='Grid', zorder=3)
        ax.scatter(stn_points['lon'], stn_points['lat'], s=25, c='red', label='Stations', marker='s', zorder=4)

        # --- Plot des frontières ---
        adm0.plot(ax=ax, facecolor="#B7A9CD", edgecolor='black', linewidth=2, alpha=0.4, zorder=2)
        adm1.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=0.5, zorder=2)

        ax.set_xlim(lon1, lon2)
        ax.set_ylim(lat1, lat2)
        ax.legend()

    except Exception as e:
        print(f"Error processing geographical coordinates : {e}")
        import traceback
        traceback.print_exc()

    # --- Configuration finale du graphique ---
    ax.set_xlim(lon1, lon2)
    ax.set_ylim(lat1, lat2)

    #ax.xaxis.set_major_locator(ticker.MultipleLocator(10)) 
    #ax.yaxis.set_major_locator(ticker.MultipleLocator(10)) 

    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=6, steps=[1, 2, 5, 10]))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=6, steps=[1, 2, 5, 10]))

    ax.set_xlabel('E', fontweight='bold')
    ax.set_ylabel('N', fontweight='bold')

    ax.set_title(f"Validation Grille et Stations - {line[0]}")
    ax.grid(True, linestyle='--', alpha=0.8)
    plt.legend()

    # --- Capture des coordonnées Pixels (Pour la carte HTML) ---
    html_areas_list = []
    period = [7, 10, 30, 60, 90, 180]
    formatted_areas = {}
    
    for prd in period:
        for _, row in grid_points.iterrows():
            px_x, px_y = get_exact_pixels(row['lat'], row['lon'], ax, fig)
            area = f'<area href="{row["idx"]}_{prd}.png" shape="circle" coords="{px_x:.2f},{px_y:.2f},5" title="Lat:{row["lat"]:.2f}°N, Lon:{row["lon"]:.2f}°E">'
            html_areas_list.append(area)

        for _, row in stn_points.iterrows():
            px_x, px_y = get_exact_pixels(row['lat'], row['lon'], ax, fig)
            name_stn = row.get('name', row['name'])
            area = f'<area href="{row["idx"]}_{prd}.png" shape="circle" coords="{px_x:.2f},{px_y:.2f},5" title="{name_stn}: {row["lat"]:.2f}°N, {row["lon"]:.2f}°E ">'
            html_areas_list.append(area)

        format_txt = "\n".join(html_areas_list)
        html_areas_list = []
        formatted_areas[str(prd)] = format_txt
        
    ts_path = (Path(__file__).resolve().parents[1] / "data" / "ts_maps" / f"{country}" / rndta).resolve()
    ts_path.mkdir(parents=True, exist_ok=True)
    
    filename = f"{country}_grid.png"
    full_save_path = ts_path / filename
    
    #print(f"🚀 Tentative de sauvegarde dans : {full_save_path}")
    plt.savefig(full_save_path, dpi=100) 
    plt.close(fig) 
    
    #print(f"✅ Image sauvegardée avec succès !")
    with open(base_data / f"formatted_areas_{country}.json", "w", encoding="utf-8") as f:
        json.dump(formatted_areas, f, indent=4, ensure_ascii=False)
        f.close()
