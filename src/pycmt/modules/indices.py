import os
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.patches import PathPatch
from shapely.ops import unary_union
import geopandas as gpd
import pandas as pd
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm
from xgrads import open_CtlDataset
from pathlib import Path
from scipy.ndimage import gaussian_filter
from datetime import datetime, timedelta
import shutil
import re




####### Beginning VHI ######

# --- 1. CONFIGURATION ---
CNTRY_CODE = "AFR"        # Correspond à $1
CNTRY_NAME = "Africa"    # Correspond à $2
INIT_DAY = 0              # Correspond à $3 (initday)
MASK_OPT = "mask_yes"     # Correspond à $4

# Chemins
#FIX_DIR = "../fix"

def get_week_logic(days_ago):
    """
    Reproduit la logique de calendrier du script original.
    Calcul des semaines de 7 jours à partir du 1er Janvier.
    """
    print(f" starting getting week info")
    target_date = datetime.now() - timedelta(days=days_ago)
    year = target_date.year
    
    # Jour julien (1-366)
    julian_day = int(target_date.strftime("%j"))
    
    # Numéro de semaine (Logique GrADS : tranches de 7 jours)
    week_num = ((julian_day - 1) // 7) + 1
    week_num = min(week_num, 52)
    
    # Calcul des bornes de la semaine pour le titre (du script original)
    # On trouve le jour de la semaine du 1er Janvier de l'année concernée
    jan1 = datetime(year, 1, 1)
    start_day_val = jan1.weekday() + 1 # 1=Lundi, 7=Dimanche
    
    # Calcul du décalage
    cur_wkday = target_date.weekday() + 1
    diff = cur_wkday - start_day_val
    if diff < 0: diff += 7
    
    bgn_date = target_date - timedelta(days=diff)
    end_date = bgn_date + timedelta(days=6)
    
    # Sécurité fin d'année
    if julian_day > 357:
        end_date = datetime(year, 12, 31)
    print(f" Ending Getting week info")
    return {
        "year": year,
        "week_str": f"{week_num:02d}",
        "file_tag": f"{year}0{week_num:02d}", # ex: 2026017
        "bgn_str": bgn_date.strftime("%d%b%Y").upper(),
        "end_str": end_date.strftime("%d%b%Y").upper()
    }

def prepare_data_for_week(wk_info, week_idx, vhi_path):
    """Gère les fichiers et prépare les variables pour chaque semaine."""
    #os.chdir("/arc2")
    # 1. Identification du fichier satellite source
    # Le script cherche NPP ou J01
    FIX_DIR = Path(__file__).resolve().parents[1] / "data"
    print(f" starting preparing data for week")
    prefix = "VHP.G04.C07"
    suffix = f"P{wk_info['file_tag']}.VH.nc"
    
    source_file = None
    for sat in ["npp", "j01"]:
        f_name = f"{prefix}.{sat}.{suffix}"
        print(f"{f_name}")
        print(f"{vhi_path}")
        f_name = Path(vhi_path) / f_name
        print(f"{f_name}")
        if os.path.exists(f_name):
            source_file = f_name
            break
    cur = os.getcwd()
    print(f"current path === {cur}")
    if not source_file:
        print(f"!!! Données manquantes : Semaine {wk_info['week_str']} ({wk_info['year']})")
        return

    # 2. Préparation du fichier de travail (vhi.nc)
    shutil.copy(source_file, Path(vhi_path)/"vhi.nc")
    
    # 3. Lecture des paramètres de géométrie (équivalent awk)
    ydel = "0.1"
    latlon_file = os.path.join(FIX_DIR, f"{CNTRY_NAME}_latlon")
    if os.path.exists(latlon_file):
        with open(latlon_file, 'r') as f:
            lines = f.readlines()
            if len(lines) >= 2:
                # Colonne 9 (index 8) de la ligne 2
                ydel = lines[1].split()[8]

    # --- LOG ---
    print(f"--- SEMAINE {week_idx} (Sem. {wk_info['week_str']}) ---")
    print(f"Période : {wk_info['bgn_str']} au {wk_info['end_str']}")
    print(f"Source  : {source_file}")
    print(f"YDEL    : {ydel}")    
    #if os.path.exists("vhi.nc"):
    #    os.remove("vhi.nc")
    print(f" Ending preparing data for week")


def get_vhi_colormap():
    hex_colors = [
        "#FF00FF", # 0-6
        "#E10032", # 6-12
        "#FF7D7D", # 12-24
        "#FFAA00", # 24-36
        "#FFFF64", # 36-48
        "#64FF64", # 48-60
        "#009600", # 60-72
        "#5050FF", # 72-84
        "#0000C8"  # 84-100
    ]
    levels = [0, 6, 12, 24, 36, 48, 60, 72, 84, 100]
    
    cmap = ListedColormap(hex_colors)
    norm = BoundaryNorm(levels, len(hex_colors))
    
    return cmap, norm, levels # On retourne aussi levels pour le plot

def load_config_extent(path_fix):
    print(f" starting loading extents")
    print(f"Loading extents")
    dir = os.getcwd()
    print(f"Load extents  CWD: {dir}")
    with open(path_fix, 'r') as f:
        line = f.readline().split()
        country = line[0]
        lat1, lat2 = float(line[1]), float(line[2])
        lon1, lon2 = float(line[3]), float(line[4])
    print(f" Ending loading extents")
    return {
        "name": country,
        "lat_range": (lat1, lat2),
        "lon_range": (lon1, lon2)
    }


def prepare_vhi_data(path_vhi, path_mask):

    
    print(f" Starting Preparing VHI data")
    """print(f"VHI CWD: {dir}")
    print(f"{path_vhi}")"""
    ds = xr.open_dataset(path_vhi) #r'vhi.nc')
    ds = ds.assign_coords({
                "lat": ds.latitude,
                "lon": ds.longitude
            }).rename({
                "HEIGHT": "lat",
                "WIDTH": "lon"
            })
    vhi = ds["VHI"].load()
    
    mask_ds = xr.open_dataset(path_mask)
    #land_mask = xr.open_dataset("GLDASp5_landmask_025d.nc4")
    mask_interp = mask_ds['mask_data'].interp(lat=vhi.lat, lon=vhi.lon, method="nearest")
    #land_interp = land_mask["GLDAS_mask"].isel(time=-1).interp(lat=vhi.lat, lon=vhi.lon, method="nearest").load()
    print(f" Ending Preparing VHI data")
    return vhi.where(mask_interp == 1) #&(land_interp ==1))


def plot_vhi_map(da, extent_info, path_shp, wk_info, output_dir, country):
    """Génère et sauvegarde la carte Cartopy."""
    print(f"Starting Generating VHI maps")
    cmap, norm, levels = get_vhi_colormap()
    
    fig = plt.figure(figsize=(11, 8.5))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    da = da.where(da > 0)
    im = ax.pcolormesh(
        da.lon,
        da.lat, 
        da, 
        #levels=levels,
        cmap=cmap, 
        norm=norm,
        transform=ccrs.PlateCarree(),
        zorder=1)
    
    if os.path.exists(path_shp):
        gdf = gpd.read_file(path_shp)
        gdf.plot(ax=ax, edgecolor='black', facecolor='none', linewidth=0.5, zorder=2)
    else:
        print(f"Attention: {path_shp} introuvable. Utilisation des frontières par défaut.")
        #ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='black')

    # Configuration géographique
    ax.set_extent([extent_info['lon_range'][0], extent_info['lon_range'][1], 
                   extent_info['lat_range'][0], extent_info['lat_range'][1]],
                  crs=ccrs.PlateCarree()) # Best practice to specify CRS for extent
                  
    date_val = wk_info['end_str']
    plt.title(f"Vegetation Health Index - {extent_info['name']}",loc='left', fontsize=9, fontweight='bold', pad=7)
    plt.title(f"Week Ending {date_val}", loc='right', fontsize=9, style='italic')

    cbar = fig.colorbar(im, ax=ax, orientation='horizontal', fraction=0.04, pad=0.08, extend='both')
    cbar.set_label('VHI (%)', fontsize=10)
    cbar.ax.set_xticklabels([str(l) for l in levels])
    output_path= output_dir / f"{country}_vhi{wk_info['week_str']}.png"

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Fichier créé : {output_path}")
    print(f"Ending Generating VHI maps")


def do_vhi(country, country_iso):
 
    base_dir = Path(__file__).resolve().parents[1] / "data" 
    vhi_dir_path = base_dir / "vhi" / "data" # On garde l'objet Path ici
    
    if not vhi_dir_path.exists():
        vhi_dir_path.mkdir(parents=True, exist_ok=True)

    # Maintenant on peut transformer en chemins relatifs pour tes fonctions
    vhi_dir_str = os.path.relpath(vhi_dir_path)
    #captions_info = os.path.relpath(captions_file_path)
    country_info = os.path.relpath(base_dir / f"{country}_latlon")
    output_dir = base_dir / "vhi" /"vhi_maps" /f"{country}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    #########
    country_mask = base_dir / "gis_resources"/f"country_masks0p036/365dcal/{country_iso}_mask.nc"
    FILE_VHI = base_dir / "vhi" / "data" / "vhi.nc"
    FILE_SHP = base_dir / "gis_resources"/"countries" / f"{country_iso}_adm" /f"{country_iso}_adm1.shp"
    
    print(f"Traitement VHI pour {country}...")

    for i in range(1, 7):
        # Calcul du décalage de jours (7, 14, 21...)
        nd = INIT_DAY + (i * 7)
        
        # Extraction des infos de calendrier
        wk_info = get_week_logic(nd)
        print(f"week info: {wk_info}")
        # Traitement
        prepare_data_for_week(wk_info, i, vhi_dir_str)
        try:
            conf = load_config_extent(country_info)
            #os.chdir("..")          
            curr =os.getcwd()
            print(f"currently dur : {curr}")
            vhi_final = prepare_vhi_data(FILE_VHI, country_mask)
            plot_vhi_map(vhi_final, conf, FILE_SHP, wk_info, output_dir, country)
        except Exception as e:
            print(f"Erreur lors de l'exécution : {e}")
        #if os.path.exists(FILE_VHI):
        #    os.remove(FILE_VHI)
        #break
            

    print("\nFin du script.")
######End VHI #######

###### Beginning SPP ######


import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from xgrads import open_CtlDataset
from matplotlib.colors import ListedColormap, BoundaryNorm

# --- 1. CONFIGURATION & LECTURE DES FICHIERS FIXES ---

def read_country_config(filepath):
    with open(filepath, 'r') as f:
        data = f.readline().split()
        # On saute l'en-tête GrADS si présent, on prend la ligne de données
        #data = lines[1].split() 
        return {
            'name': data[0],
            'lat1': float(data[1]), 'lat2': float(data[2]),
            'lon1': float(data[3]), 'lon2': float(data[4])
            #'xlint': float(data[5]), 'ylint': float(data[6])
        }

def read_captions(filepath):
    captions = []
    with open(filepath, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
        for i in range(0, len(lines), 2):
            captions.append((lines[i], lines[i+1]))
    return captions

# --- 2. DÉFINITION DES COULEURS (Styles GrADS) ---

def get_grads_colors():
    
    #Recrée les échelles de couleurs extraites de l'image :
    #- Below: Jaune à Rouge foncé
    #- Normal: Vert clair à Vert foncé
    #- Above: Bleu clair à Bleu foncé
    
    # Les seuils correspondent aux étiquettes 40, 60, 75, 90 de votre image
    levels = [20, 40, 60, 75, 90, 100]
    
    cmaps = {
        # BELOW: Jaune (#ffff80), Orange clair (#ffc800), Orange foncé (#ff8000), Rouge (#cc000
        # 0)
        'below':  ListedColormap(["#FFFFFF", '#ffff80', '#ffc800', '#ff8000', '#cc0000']),
        
        # NORMAL: Vert pâle (#c8ffc8), Vert clair (#78ff78), Vert moyen (#00cc00), Vert foncé (#007d00)
        'normal': ListedColormap(["#FFFFFF", '#c8ffc8', '#78ff78', '#00cc00', '#007d00']),
        
        # ABOVE: Bleu très clair (#b4ffff), Bleu ciel (#78d2ff), Bleu royal (#0078ff), Bleu marine (#003cff)
        'above':  ListedColormap(["#FFFFFF", '#b4ffff', '#78d2ff', '#0078ff', '#003cff'])
    }
    
    # BoundaryNorm garantit que chaque intervalle [40-60], [60-75], etc., utilise la bonne couleur
    norms = {k: BoundaryNorm(levels, v.N) for k, v in cmaps.items()}
    
    return cmaps, norms

# --- 3. LOGIQUE DE CALCUL ET TRACÉ ---

import os
import re
import xarray as xr
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import shapely.geometry as sgeom
from pathlib import Path
from shapely.validation import make_valid

def run_orchestrator_spp(country, country_iso, rndta, mask_enabled=True):
    ctl_files = [
        f'spp_{rndta}_comb_1ic-0proj.ctl', f'spp_{rndta}_comb_1ic-1proj.ctl',
        f'spp_{rndta}_comb_1ic-2proj.ctl', f'spp_{rndta}_comb_2ic-0proj.ctl',
        f'spp_{rndta}_comb_2ic-1proj.ctl', f'spp_{rndta}_comb_3ic-0proj.ctl'
    ]
    base_dir = Path(__file__).resolve().parents[1] / "data" 
    spp_dir_path = base_dir / "spp" / f"spp_data_{rndta}" 
    
    if not spp_dir_path.exists():
        spp_dir_path.mkdir(parents=True, exist_ok=True)

    captions_file_path = spp_dir_path / 'spp_timescales.txt'

    spp_dir_str = os.path.relpath(spp_dir_path)
    captions_info = os.path.relpath(captions_file_path)
    country_info = os.path.relpath(base_dir / f"{country}_latlon")
    output_dir = base_dir / "spp" / "spp_maps" / f"{country}" / f"{rndta}"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    config = read_country_config(country_info)
    captions = read_captions(captions_info)
    cmaps, norms = get_grads_colors()

    print("🗺️ Chargement et traitement des limites géographiques...")
    shp_error = False
    try:
        shp_path = base_dir / "gis_resources" / "countries" / f"{country_iso}_adm" / f"{country_iso}_adm1.shp"
        shp_adm1 = gpd.read_file(shp_path)
        shp_adm1 = shp_adm1.to_crs("EPSG:4326")
        
        shp_adm1["geometry"] = shp_adm1.geometry.apply(make_valid)
        shp_adm1 = shp_adm1[shp_adm1.is_valid]
        
        bounds = shp_adm1.total_bounds
        country_geom = shp_adm1.geometry.unary_union
        if not country_geom.is_valid:
            country_geom = country_geom.buffer(0)
            
        world_box = sgeom.box(bounds[0]-10, bounds[1]-10, bounds[2]+10, bounds[3]+10)
        inverse_mask = world_box.difference(country_geom)
        
        pad = 0.5
        extent_box = [bounds[0]-pad, bounds[2]+pad, bounds[1]-pad, bounds[3]+pad]
    except Exception as e:
        print(f"⚠️ Erreur d'initialisation SHP : {e}")
        shp_error = True
        extent_box = [config['lon1'], config['lon2'], config['lat1'], config['lat2']]

    mask_ds = None
    if mask_enabled:
        #mask_path = base_dir / "gis_resources" / f"country_masks{rsl_name}" / "365dcal" / f"{country_iso}_mask.nc"
        mask_path = base_dir / "gis_resources" / f"country_masks0p1" / "365dcal" / f"{country_iso}_mask.nc"

        mask_ds = xr.open_dataset(mask_path)
        mask_sorted = mask_ds['mask_data']

    for i, ctl in enumerate(ctl_files):
        print(f"Processing {ctl}...")
        ds = open_CtlDataset(os.path.relpath(spp_dir_path / ctl))
        
        def lon_360_to_180(ds):
            ds.coords['lon'] = (ds.coords['lon'] + 180) % 360 - 180.125
            return ds.sortby('lon')

        ds = lon_360_to_180(ds)
        
        v = list(ds.data_vars)
        p1 = ds[v[0]].isel(time=-1).load()
        p2 = ds[v[1]].isel(time=-1).load()
        p3 = ds[v[2]].isel(time=-1).load()

        max_p = np.maximum(np.maximum(p1, p2), p3)
        threshold = 100 / 3.0

        if mask_enabled:
            m = mask_sorted
            if not (m.lat.equals(ds.lat) and m.lon.equals(ds.lon)):
                m = m.interp(lat=ds.lat, lon=ds.lon, method='nearest')
            
            p1, p2, p3 = p1.where(m > 0), p2.where(m > 0), p3.where(m > 0)
            max_p = max_p.where(m > 0)

        fig = plt.figure(figsize=(10, 8.5))
        ax = plt.axes(projection=ccrs.PlateCarree())
        
        ax.set_extent(extent_box, crs=ccrs.PlateCarree())
        ax.set_facecolor("#ffffff")

        if not shp_error:
            ax.add_geometries([country_geom], ccrs.PlateCarree(),
                              facecolor="#e0e0e0", edgecolor='none', zorder=1)

        categories = [
            ('below', p1, (p1 == max_p) & (p1 > threshold)),
            ('normal', p2, (p2 == max_p) & (p2 > threshold)),
            ('above', p3, (p3 == max_p) & (p3 > threshold))
        ]

        for cat_name, data, mask in categories:
            plot_data = data.where(mask)
            if not plot_data.isnull().all():
                ax.pcolormesh(
                    ds.lon, ds.lat, plot_data,
                    cmap=cmaps[cat_name], 
                    norm=norms[cat_name],
                    shading='nearest',
                    transform=ccrs.PlateCarree(),
                    zorder=2
                )

        if not shp_error:
            ax.add_geometries([inverse_mask], ccrs.PlateCarree(), 
                              facecolor="#ffffff", edgecolor='none', zorder=5)

            ax.add_geometries(shp_adm1.geometry, ccrs.PlateCarree(),
                              facecolor='none', edgecolor='#4a4a4a', linewidth=0.5, zorder=6)
            
            ax.add_geometries([country_geom], ccrs.PlateCarree(),
                              facecolor='none', edgecolor='black', linewidth=1.5, zorder=7)

        plt.title(f"{captions[i][0]}\n{captions[i][1]}", fontsize=12, fontweight='bold', pad=15)
        caption_text = f"{captions[i][0]}\n{captions[i][1]}"

        match_month = re.search(r"Period\s*=\s*(\d+)", captions[i][0])
        match_proj = re.search(r"Period\s*=\s*(\d+)", captions[i][1])

        month_val = match_month.group(1) if match_month else "0"
        proj_val = match_proj.group(1) if match_proj else "0"

        for j, cat in enumerate(['below', 'normal', 'above']):
            cax = fig.add_axes([0.18 + j * 0.24, 0.05, 0.18, 0.01])
            cb = plt.colorbar(
                plt.cm.ScalarMappable(norm=norms[cat], cmap=cmaps[cat]), 
                cax=cax, orientation='horizontal'
            )
            cb.set_label(cat.upper(), fontsize=7, fontweight='bold', labelpad=2)
            cb.ax.tick_params(labelsize=6)

        # =====================================================================
        # 2. FIX ANCHOR : AJOUT DE LA LÉGENDE DRYMASK INCORPORÉE SUR LA CARTE
        # =====================================================================
        dry_patch = mpatches.Patch(facecolor='#e0e0e0', edgecolor='#4a4a4a', linewidth=0.5)
        ax.legend(
            [dry_patch], 
            ['Drymask'], 
            loc='lower left',        # Positionné dans le coin inférieur gauche de l'Afrique (dans l'océan Atlantique)
            fontsize=7, 
            frameon=True, 
            facecolor='#ffffff', 
            edgecolor='#4a4a4a',
            framealpha=0.9
        )
        gl = ax.gridlines(draw_labels=True, linewidth=0.6, color='gray', alpha=0.6, linestyle='--', zorder=8)
        gl.top_labels = gl.right_labels = False
        gl.xlocator = gl.ylocator = plt.MultipleLocator(10)

        plt.savefig(output_dir / f"spp_{country}_{rndta}_Month{month_val}Proj{proj_val}.png", dpi=150)#, bbox_inches='tight')
        plt.clf()
        plt.close(fig)

    print(f"✅ Toutes les cartes SPP {rndta} ont été générées et masquées avec succès.")       #print(f"Saved: {output_name}")
"""
if __name__ == "__main__":
    run_orchestrator_spp()"""
######END SPP #######

###### SPI ##########

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
from xgrads import open_CtlDataset
import os
from datetime import datetime





# --- 2. FONCTION DE LISSAGE (Basée sur ton script) ---

####last edit clean_smoothing
def apply_clean_smoothing1(da, sigma=1.0):
    # 1. Sauvegarde des coordonnées et attributs d'origine
    coords = da.coords
    dims = da.dims
    attrs = da.attrs
    name = da.name

    # 2. Gestion des NaNs
    # Le filtre gaussien propage les NaNs (tout devient blanc) si on ne les gère pas.
    # On remplit les zones hors-pays par 0 le temps du calcul.
    filled_data = da.fillna(0).values

    # 3. Application du filtre (renvoie un numpy array)
    smoothed_values = gaussian_filter(filled_data, sigma=sigma)

    # 4. RECONSTRUCTION du DataArray
    # On réinjecte les valeurs lissées dans la structure d'origine
    da_smoothed = xr.DataArray(
        data=smoothed_values,
        coords=coords,
        dims=dims,
        name=name,
        attrs=attrs
    )

    # 5. Rétablissement du masque
    # On remet les NaNs là où ils étaient (en dehors du pays) 
    # pour que le lissage ne dépasse pas sur l'océan.
    return da_smoothed.where(da.notnull())

# --- 3. CHARGEMENT DU MASQUE PAYS NC ---


import shapely.geometry as sgeom
from shapely.validation import make_valid  # Blindage topologique

def generate_spi(country_iso, country, rndta):
    base_dir = Path(__file__).resolve().parents[1] / "data" 
    mask_path = base_dir / "gis_resources" / f"country_masks0p036" / "365dcal" / f"{country_iso}_mask.nc"
    spi_dir = base_dir / "spi" / "data" / f"{rndta}"
    
    if not os.path.exists(spi_dir):
        os.makedirs(spi_dir)
    spi_dir = os.path.relpath(spi_dir)
    
    output_dir = base_dir / "SPI" / "spi_maps" / f"{country}" / f"{rndta}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    PATH_MASK_NC = xr.open_dataset(mask_path)
    mask_sorted = PATH_MASK_NC['mask_data'].sortby('lat')

    # =========================================================================
    # 1. CHARGEMENT ET RÉPARATION STRICTE DU SHAPEFILE (Hors boucle)
    # =========================================================================
    print("🗺️ Chargement et nettoyage des frontières géographiques...")
    shp_path = base_dir / "gis_resources" / "countries" / f"{country_iso}_adm" / f"{country_iso}_adm1.shp"
    shp_adm1 = gpd.read_file(shp_path)
    
    # Précision de la projection géographique (WGS84)
    shp_adm1 = shp_adm1.to_crs("EPSG:4326")
    
    # Réparation chirurgicale des géométries défectueuses (side location conflicts)
    shp_adm1["geometry"] = shp_adm1.geometry.apply(make_valid)
    shp_adm1 = shp_adm1[shp_adm1.is_valid]
    
    bounds = shp_adm1.total_bounds
    
    # Fusion des contours pour la frontière nationale
    country_geom = shp_adm1.geometry.unary_union
    if not country_geom.is_valid:
        country_geom = country_geom.buffer(0)
        
    # Création de la boîte globale pour le masque inversé extérieur
    world_box = sgeom.box(bounds[0]-10, bounds[1]-10, bounds[2]+10, bounds[3]+10)
    inverse_mask = world_box.difference(country_geom)
    # =========================================================================

    # --- 2. CONFIGURATION DES COULEURS (Styles GrADS) ---
    spi_levels = [-2, -1.6, -1.3, -0.8, -0.5, 0.5, 0.8, 1.3, 1.6, 2]
    spi_colors = [
        '#B20000', '#FF0000', '#FF6600', '#FFBD33', '#FFFF99', 
        '#FFFFFF', '#B2EBF2', '#80B3FF', '#3385FF', '#1A53FF', '#311B92'
    ]

    # --- 3. BOUCLE SUR LES PÉRIODES ---
    periods = [1, 3, 6, 12, 24]

    for p in periods:
        print(f"🔄 Traitement : SPI {p}-Month...")
        
        ds_spi = open_CtlDataset(os.path.join(spi_dir, f'{rndta}.spi.{p}.mo.ctl'))
        ds_m = open_CtlDataset(os.path.join(spi_dir, f'drymask{p}.ctl'))

        def lon_360_to_180(ds):
            ds.coords['lon'] = (ds.coords['lon'] + 180) % 360 - 180.25
            return ds.sortby('lon')

        ds_spi = lon_360_to_180(ds_spi)
        ds_m = lon_360_to_180(ds_m)
        
        spi = ds_spi['p'].isel(time=-1, lev=0).load()
        drymask_raw = ds_m['dm'].isel(time=-1, lev=0).load()

        drymask = drymask_raw.interp(
            lat=mask_sorted.lat, 
            lon=mask_sorted.lon, 
            method="nearest"
        ).fillna(0)

        # --- 4. LISSAGE ET MASQUAGE FINAL ---
        spi_smooth = apply_clean_smoothing1(spi, sigma=1.5)
        spi_smooth = spi_smooth.interp(lat=mask_sorted.lat, lon=mask_sorted.lon, method="nearest").fillna(0)

        spi_masked = spi_smooth.where((drymask == 1.) & (mask_sorted == 1))

        # --- 5. DESSIN ---
        fig = plt.figure(figsize=(12, 9.27))
        ax = plt.axes(projection=ccrs.PlateCarree())
        
        # 🟡 GESTION DE LA COULEUR EXTERNE (Fond global initial)
        #ax.set_facecolor("#e0e0e0") 
        ax.set_facecolor("#ffffff")

        # =====================================================================
        # 🟢 GESTION DE LA COULEUR INTERNE : LE TAPIS DE ZONE SÈCHE (zorder 1)
        # =====================================================================
        # Tout l'intérieur du pays reçoit cette couleur. Les zones sèches (NaN) 
        # laisseront apparaître ce fond blanc au lieu du gris externe.
        ax.add_geometries([country_geom], ccrs.PlateCarree(),
                          facecolor="#e0e0e0", edgecolor='none', zorder=1)
        # =====================================================================

        # Tracé des contours de données SPI (zorder 4)
        im = ax.contourf(
            spi_masked.lon, 
            spi_masked.lat, 
            spi_masked,
            levels=spi_levels,
            colors=spi_colors,
            extend='both',
            transform=ccrs.PlateCarree(),
            zorder=4
        )
        
        # Définition stricte du cadrage géographique
        pad = 0.5
        ax.set_extent([bounds[0]-pad, bounds[2]+pad, bounds[1]-pad, bounds[3]+pad], crs=ccrs.PlateCarree())

        # =====================================================================
        # 🟡 APPLICATION DU MASQUE INVERSÉ EXTERNE (zorder 5)
        # =====================================================================
        # Applique la couleur de votre choix sur les zones débordantes et pays voisins
        ax.add_geometries([inverse_mask], ccrs.PlateCarree(), 
                          facecolor="#ffffff", edgecolor='none', zorder=5)
        # =====================================================================

        # --- 6. HAUTEURS DE TRACÉ DES FRONTIÈRES (zorder 6 & 7) ---
        # 1. Frontières régionales internes (fines)
        ax.add_geometries(shp_adm1.geometry, ccrs.PlateCarree(),
                          facecolor='none', edgecolor='#4a4a4a', linewidth=0.5, zorder=6)
        
        # 2. Frontière nationale principale (noire, épaisse et au premier plan)
        ax.add_geometries([country_geom], ccrs.PlateCarree(),
                          facecolor='none', edgecolor='black', linewidth=1.5, zorder=7)

        # Grille cartographique
        gl = ax.gridlines(draw_labels=True, linewidth=0.6, color='gray', alpha=0.6, linestyle='--', zorder=8)
        gl.top_labels = gl.right_labels = False
        gl.xlocator = gl.ylocator = plt.MultipleLocator(10) # <-- Supprime les superpositions floues

        # Titre dynamique
        date_str = pd.to_datetime(ds_spi.time.values[-1]).strftime("%b %Y")
        plt.title(f"{rndta.upper()} ADJ 00Z SPI \n{p}-Month Period Ending {date_str}", fontsize=14)
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
        cbar.set_label('Percentile (%)')

        # --- 7. SAUVEGARDE NETTE ---
        plt.savefig(output_dir / f"{country}_spi_{rndta}_{p}mo.png", dpi=150, bbox_inches='tight')
        plt.clf()
        plt.close(fig)
        print(f"💾 Carte validée et sauvegardée pour {p} mois.")

    print(f"✅ Tous les calculs SPI terminés avec succès pour {country}.")
######### End SPI ########