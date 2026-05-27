import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
from xgrads import open_CtlDataset
import os
from datetime import datetime
from pathlib import Path
import pandas as pd
from scipy.ndimage import gaussian_filter
import matplotlib.path as mpath
import shapely.geometry as sgeom




#################### RUNOFF COMPUTING##############
# --- 2. FONCTION DE LISSAGE (Basée sur ton script) ---
####last edit clean_smoothing
def apply_clean_smoothing1(da, sigma=1.0):
    # 1. Sauvegarde des coordonnées et attributs d'origine
    coords = da.coords
    dims = da.dims
    attrs = da.attrs
    name = da.name

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


    return da_smoothed.where(da.notnull())



import os
import xarray as xr
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import shapely.geometry as sgeom
from pathlib import Path
from shapely.validation import make_valid  # Blindage topologique

def generate_runoff(country_iso, country):
    base_dir = Path(__file__).resolve().parents[1] / "data" 
    mask_path = base_dir / "gis_resources" / f"country_masks0p036" / "365dcal" / f"{country_iso}_mask.nc"
    runoff_dir = base_dir / "SPI" / "data" / "Runoff"
    
    if not os.path.exists(runoff_dir):
        os.makedirs(runoff_dir)
    runoff_dir = os.path.relpath(runoff_dir)
    
    output_dir = base_dir / "SPI" / "Runoff_maps" / f"{country}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # 1. CHARGEMENT ET RÉPARATION DU SHAPEFILE (Hors boucle pour la vitesse)
    # =========================================================================
    print("🗺️ loading geographic boundaries...")
    shap_path = base_dir / "gis_resources" / "countries" / f"{country_iso}_adm" / f"{country_iso}_adm1.shp"
    shp_adm1 = gpd.read_file(shap_path)
    
    # Précision du système de coordonnées géographiques standard (WGS84)
    shp_adm1 = shp_adm1.to_crs("EPSG:4326")
    
    # Réparation chirurgicale des géométries invalides (side location conflicts)
    shp_adm1["geometry"] = shp_adm1.geometry.apply(make_valid)
    shp_adm1 = shp_adm1[shp_adm1.is_valid]
    
    bounds = shp_adm1.total_bounds
    
    # Fusion des contours pour obtenir la frontière nationale externe
    country_geom = shp_adm1.geometry.unary_union
    if not country_geom.is_valid:
        country_geom = country_geom.buffer(0)
        
    # Création de la boîte pour le masque de cache inversé extérieur
    world_box = sgeom.box(bounds[0]-10, bounds[1]-10, bounds[2]+10, bounds[3]+10)
    inverse_mask = world_box.difference(country_geom)
    # =========================================================================

    # --- 2. MASQUES ET DONNÉES ---
    PATH_MASK_NC = xr.open_dataset(mask_path)
    landmask = open_CtlDataset(os.path.join(runoff_dir, 'landmask.ctl'))
    
    def lon_360_to_180(ds):
        ds.coords['lon'] = (ds.coords['lon'] + 180) % 360 - 180.25
        return ds.sortby('lon')

    landmask_recordinated = lon_360_to_180(landmask)
    land = landmask_recordinated["mask"].isel(time=-1, lev=-1).load()
    mask_sorted = PATH_MASK_NC['mask_data'].sortby('lat')

    # Configuration des niveaux de couleurs
    pct_colors = ['#8B0000', '#FF0000', '#FF4500', '#FFA500', '#FFFF00', 
                  "#FFFFFF", '#ADFF2F', '#00FF00', '#008000', '#006400', '#000080']
    pct_levels = [2, 5, 10, 20, 30, 70, 80, 90, 95, 98]

    periods = [1, 3, 6, 12, 24]

    # =========================================================================
    # 3. BOUCLE SUR LES PÉRIODES RUNOFF
    # =========================================================================
    print(f"🎨  Processing runoff for {country}...")
    for p in periods:
        
        
        ds_r = open_CtlDataset(os.path.join(runoff_dir, f'runoff.{p}.mo.ctl'))
        ds_m = open_CtlDataset(os.path.join(runoff_dir, f'drymask{p}.ctl'))

        ds_r = lon_360_to_180(ds_r)
        ds_m = lon_360_to_180(ds_m)
        
        runoff = ds_r['r'].isel(time=-1, lev=0).load()
        drymask_raw = ds_m['dm'].isel(time=-1, lev=0).load()
        
        mask_resized = mask_sorted.interp(lat=runoff.lat, lon=runoff.lon, method="nearest").fillna(0)

        # --- LISSAGE ET MASQUAGE DU RUNOFF ---
        runoff_smooth = apply_clean_smoothing1(runoff.where(runoff >= 0), sigma=1.8)
        runoff_masked = runoff_smooth.where(drymask_raw == 1.) #& (mask_resized == 1))# & (land >= 10))

        # --- 4. DESSIN MULTI-COUCHES PERFORMANCE ---
        fig = plt.figure(figsize=(12, 9.27))
        ax = plt.axes(projection=ccrs.PlateCarree())
        
        # 🟡 CONFIGURATION DE LA COULEUR EXTERNE INITIALE (Ex: Gris clair ou blanc)
        #ax.set_facecolor("#e0e0e0")
        ax.set_facecolor("#ffffff")

        # =====================================================================
        # 🟢 ÉTAPE 0 : LE TAPIS DE ZONE SÈCHE INTERNE (zorder 1)
        # =====================================================================
        # On peint l'intérieur du pays en blanc pur. Tout ce qui est filtré par le
        # drymask ou exclu par l'indice land s'affichera proprement en blanc.
        ax.add_geometries([country_geom], ccrs.PlateCarree(),
                          facecolor="#e0e0e0", edgecolor='none', zorder=1)
        # =====================================================================

        # Tracé des contours de données Runoff (zorder 2)
        im = ax.contourf(
            runoff_masked.lon, 
            runoff_masked.lat, 
            runoff_masked,
            levels=pct_levels,
            colors=pct_colors,
            extend='both',
            transform=ccrs.PlateCarree(),
            zorder=2
        )
        
        # Cadrage serré autour du pays avec marge de respiration (pad)
        pad = 0.5
        ax.set_extent([bounds[0]-pad, bounds[2]+pad, bounds[1]-pad, bounds[3]+pad], crs=ccrs.PlateCarree())

        # =====================================================================
        # 🟡 APPLICATION DU MASQUE INVERSÉ EXTERNE (zorder 5)
        # =====================================================================
        # On applique le cache de couleur sur l'extérieur (synchronisé avec set_facecolor)
        ax.add_geometries([inverse_mask], ccrs.PlateCarree(), 
                          facecolor="#ffffff", edgecolor='none', zorder=5)

        # =====================================================================
        # 🔴 TRACÉ STRATIFIÉ DES FRONTIÈRES VIA CARTOPY (zorder 6 & 7)
        # =====================================================================
        # 1. Frontières administratives régionales (fines)
        ax.add_geometries(shp_adm1.geometry, ccrs.PlateCarree(),
                          facecolor='none', edgecolor='#4a4a4a', linewidth=0.5, zorder=6)
        
        # 2. Grande frontière nationale externe (noire et épaisse)
        ax.add_geometries([country_geom], ccrs.PlateCarree(),
                          facecolor='none', edgecolor='black', linewidth=1.5, zorder=7)

        # Grille cartographique
        gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, alpha=0.3)
        gl.top_labels = gl.right_labels = False

        # Titre dynamique
        date_str = pd.to_datetime(ds_r.time.values[-1]).strftime("%b %Y")
        plt.title(f"CPC Leaky Bucket Runoff Percentile\n{p}-Month Period Ending {date_str}", fontsize=14)
        
        # Barre d'échelle de couleurs
        cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
        cbar.set_label('Percentile (%)')
        
        gl = ax.gridlines(draw_labels=True, linewidth=0.6, color='gray', alpha=0.6, linestyle='--', zorder=8)
        gl.top_labels = gl.right_labels = False
        gl.xlocator = gl.ylocator = plt.MultipleLocator(10) # <-- Supprime les superpositions floues

        # --- 5. SAUVEGARDE ET VIDAGE DE LA MÉMOIRE ---
        plt.savefig(output_dir / f"{country}_r_{p}mo.png", dpi=150, bbox_inches='tight')
        plt.clf()
        plt.close(fig)

    print(f"✅ Runoff maps processed for {country}.")


############## SOIL MOISTURE #############


# --- 1. CONFIGURATION DES CHEMINS ---

# Chemins vers les ressources

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

    return da_smoothed.where(da.notnull())


import matplotlib.path as mpath
from shapely.validation import make_valid  # Blindage topologique anti-crash

def generate_soilmoisture(country_iso, country):
    base_dir = Path(__file__).resolve().parents[1] / "data" 
    mask_path = base_dir / "gis_resources" / f"country_masks0p036" / "365dcal" / f"{country_iso}_mask.nc"
    soilmoisture_dir = os.path.relpath(base_dir / "SPI" / "data" / "soilmoisture")
    output_dir = base_dir / "SPI" / "Soilmoisture_maps" / f"{country}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # =========================================================================
    # 1. CHARGEMENT ET RÉPARATION DU SHAPEFILE (Hors boucle pour la performance)
    # =========================================================================
    shap_path = base_dir / "gis_resources" / "countries" / f"{country_iso}_adm" / f"{country_iso}_adm1.shp"
    shp_adm1 = gpd.read_file(shap_path)
    
    # Force la projection géographique standard (WGS84)
    shp_adm1 = shp_adm1.to_crs("EPSG:4326")
    
    # Correction des anomalies de géométries (side location conflicts)
    shp_adm1["geometry"] = shp_adm1.geometry.apply(make_valid)
    shp_adm1 = shp_adm1[shp_adm1.is_valid]
    
    bounds = shp_adm1.total_bounds
    
    # Fusion des contours pour obtenir la frontière nationale externe
    country_geom = shp_adm1.geometry.unary_union
    if not country_geom.is_valid:
        country_geom = country_geom.buffer(0)
        
    # Création du grand masque inverse pour le cache gris extérieur
    world_box = sgeom.box(bounds[0]-10, bounds[1]-10, bounds[2]+10, bounds[3]+10)
    inverse_mask = world_box.difference(country_geom)
    # =========================================================================

    # --- 2. CONFIGURATION DES DONNÉES ET COULEURS ---
    mask_nc = xr.open_dataset(mask_path)
    mask_sorted = mask_nc['mask_data'].sortby('lat')
    
    pct_colors = ['#C10000', '#7F4F42', '#A67C6D', '#C9A596', '#E9D6CE', 
                  "#FFFFFFFF", '#B9F59D', '#6EF06E', '#26D426', '#00A100', '#2E67F8']
    pct_levels = [2, 5, 10, 20, 30, 70, 80, 90, 95, 98]
    periods = [1, 3, 6, 12, 24]

    pad = 0.5
    extent_box = [bounds[0]-pad, bounds[2]+pad, bounds[1]-pad, bounds[3]+pad]

    # =========================================================================
    # 3. BOUCLE SUR LES PÉRIODES SOIL MOISTURE
    # =========================================================================
    print(f"Processing : {country} Soil Moisture...")
    for p in periods:
       
        
        ds_sm = open_CtlDataset(os.path.join(soilmoisture_dir, f'soilmoisture.{p}.mo.ctl'))
        ds_m = open_CtlDataset(os.path.join(soilmoisture_dir, f'drymask{p}.ctl'))

        def lon_360_to_180(ds):
            ds.coords['lon'] = (ds.coords['lon'] + 180) % 360 - 180.25
            return ds.sortby('lon')

        soilmoisture = lon_360_to_180(ds_sm)['w'].isel(time=-1, lev=0).load()
        drymask = lon_360_to_180(ds_m)['dm'].isel(time=-1, lev=0).load()
        
        mask_resized = mask_sorted.interp(lat=soilmoisture.lat, lon=soilmoisture.lon, method="nearest").fillna(0)
        
        # Lissage de la matrice
        soilmoisture_smooth = apply_clean_smoothing1(soilmoisture.where(soilmoisture >= 0), sigma=1.8)
        soilmoisture_final = soilmoisture_smooth.where(drymask == 1.) #& (mask_resized == 1))

        # --- 4. DESSIN MULTI-COUCHES PERFORMANCE ---
        fig = plt.figure(figsize=(12, 9.27))
        ax = plt.axes(projection=ccrs.PlateCarree())
        
        # 🟡 CONFIGURATION DE LA COULEUR EXTERNE PAR DÉFAUT
        #ax.set_facecolor("#e0e0e0")
        ax.set_facecolor("#ffffff")
        ax.set_extent(extent_box, crs=ccrs.PlateCarree())

        # =====================================================================
        # 🟢 ÉTAPE 0 : LE TAPIS DE ZONE SÈCHE INTERNE (zorder 1)
        # =====================================================================
        # On tapisse l'intérieur du pays de blanc. Les pixels de l'océan limitrophe 
        # ou les zones intérieures éliminées par le drymask (NaN) seront d'un blanc impeccable.
        ax.add_geometries([country_geom], ccrs.PlateCarree(),
                          facecolor="#e0e0e0", edgecolor='none', zorder=1)
        # =====================================================================

        # Tracé des contours de données d'humidité (zorder 2)
        im = ax.contourf(
            soilmoisture_final.lon, 
            soilmoisture_final.lat, 
            soilmoisture_final,
            levels=pct_levels,
            colors=pct_colors,
            extend='both',
            transform=ccrs.PlateCarree(),
            zorder=2
        )

        # =====================================================================
        # 🟡 APPLICATION DU MASQUE INVERSÉ EXTERNE (zorder 5)
        # =====================================================================
        # Le calque de cache gris uniforme vient se superposer à l'extérieur des frontières
        ax.add_geometries([inverse_mask], ccrs.PlateCarree(), 
                          facecolor="#ffffff", edgecolor='none', zorder=5)

        # =====================================================================
        # 🔴 TRACÉ CHIRURGICAL DES FRONTIÈRES VIA CARTOPY (zorder 6 & 7)
        # =====================================================================
        # 1. Frontières administratives régionales ADM1 (Fines et grises)
        ax.add_geometries(shp_adm1.geometry, ccrs.PlateCarree(),
                          facecolor='none', edgecolor='#4a4a4a', linewidth=0.5, zorder=6)
        
        # 2. Grande frontière nationale externe ADM0 (Noire et épaisse au premier plan)
        ax.add_geometries([country_geom], ccrs.PlateCarree(),
                          facecolor='none', edgecolor='black', linewidth=1.5, zorder=7)

        # Grille de coordonnées cartographiques
        gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, alpha=0.3)
        gl.top_labels = gl.right_labels = False
        
        # Titre de la carte
        date_str = pd.to_datetime(ds_sm.time.values[-1]).strftime("%b %Y")
        plt.title(f"{country} Soil Moisture Percentile\n{p}-Month Period Ending {date_str}", fontsize=14)
        
        # Configuration de la Colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
        cbar.set_label('Percentile (%)')

        gl = ax.gridlines(draw_labels=True, linewidth=0.6, color='gray', alpha=0.6, linestyle='--', zorder=8)
        gl.top_labels = gl.right_labels = False
        gl.xlocator = gl.ylocator = plt.MultipleLocator(10) # <-- Supprime les superpositions floues

        # Sauvegarde propre et réinitialisation de la mémoire graphique
        plt.savefig(output_dir / f"{country}_sm_{p}mo.png", dpi=150, bbox_inches='tight')
        plt.clf()
        plt.close(fig)

    print(f"✅ Soil moisture maps generated for {country}.")

