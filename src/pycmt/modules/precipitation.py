import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
import geopandas as gpd
from xgrads import open_CtlDataset
import os
import cartopy.crs as ccrs
import scipy.ndimage as ndimage
from scipy.ndimage import gaussian_filter
from pathlib import Path
import matplotlib.path as mpath
import shapely.geometry as sgeom
from shapely.validation import make_valid
import platform


def apply_clean_smoothing(da, sigma=1.0):
    # 1. Sauvegarde des coordonnées et attributs d'origine
    coords = da.coords
    dims = da.dims
    attrs = da.attrs
    name = da.name

    filled_data = da.fillna(0).values


    smoothed_values = gaussian_filter(filled_data, sigma=sigma)

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

def plot_precip(rsl, rsl_name, country_iso, country, rndta):

    # =========================================================
    # 1. PATHS
    # =========================================================
    base_dir = Path(__file__).resolve().parents[1] / "data"
    mask_path = base_dir / "gis_resources" / f"country_masks{rsl_name}" / "365dcal" / f"{country_iso}_mask.nc"
    if rndta.lower() == "arc2":
        daily_precip_path = (base_dir / "ARC2" / "arc2" / "arc2.ctl")#.resolve()
        clim_path = (base_dir / "ARC2" / "arc2_clim" / "arc2_clim.ctl")#.resolve()
        precip_var = "pmer2"
    elif rndta.lower() == "rfe2": 
        daily_precip_path = (base_dir / "rfe2_data" / "rfe2_daily" / "rfe2daily.ctl")#.resolve()
        clim_path = (base_dir / "rfe2_data" / "rfe2_clim" / "rfe2clim.ctl")#.resolve()
        precip_var = "r"
    shap_path = base_dir / "gis_resources" / "countries" / f"{country_iso}_adm" / f"{country_iso}_adm1.shp"

    # =========================================================
    # 2. DATA LOAD & REPARATION UNIQUE SHP (Hors boucle)
    # =========================================================
    print("Loading data...")

    mask_nc = xr.open_dataset(mask_path)
    #daily_data = open_CtlDataset(os.path.relpath(daily_precip_path))
    #clim_data = open_CtlDataset(os.path.relpath(clim_path))

    abs_daily_path = daily_precip_path.resolve()
    abs_clim_path = clim_path.resolve()

    # 4. Normalisation pour contrer le bug de xgrads sous Windows
    if platform.system() == "Windows":
        # Force l'usage de slashes '/' pour empêcher xgrads d'ajouter './' devant 'C:/'
        final_daily_path = abs_daily_path.as_posix()
        final_clim_path = abs_clim_path.as_posix()
    else:
        # Format natif pour Linux et macOS (contient déjà des slashes '/')
        final_daily_path = str(abs_daily_path)
        final_clim_path = str(abs_clim_path)

    # 5. Ouverture sécurisée des fichiers de données GrADS
    daily_data = open_CtlDataset(final_daily_path)
    clim_data = open_CtlDataset(final_clim_path)


    # Chargement et réparation unique du Shapefile
    shp_adm1 = gpd.read_file(shap_path)
    shp_adm1 = shp_adm1.to_crs("EPSG:4326")
    shp_adm1["geometry"] = shp_adm1.geometry.apply(make_valid)
    shp_adm1 = shp_adm1[shp_adm1.is_valid]

    bounds = shp_adm1.total_bounds
    
    # Fusion des contours pour obtenir la frontière nationale externe
    country_geom = shp_adm1.geometry.unary_union
    if not country_geom.is_valid:
        country_geom = country_geom.buffer(0)

    # Création du masque inversé (le calque de cache extérieur)
    world_box = sgeom.box(bounds[0]-10, bounds[1]-10, bounds[2]+10, bounds[3]+10)
    inverse_mask = world_box.difference(country_geom)

    periodes = [7, 10, 30, 60, 90, 180]

    # =========================================================
    # 3. COLORS & LIMITS
    # =========================================================
    colors_precip = [
        "#FFFFFF", "#D3FFCB", "#93F78F", "#4BC34B",
        "#C3F3FB", "#73B7F7", "#4B8BEF", "#E3E3FF",
        "#B3A3FF", "#8D80E3", "#FFFBBB", "#FFB333",
        "#E74333", "#B73333", "#EBA3A3", "#FFEBEB"
    ]
    limits_precip = [2, 5, 10, 25, 50, 75, 100, 150, 200, 300, 500, 750, 1000, 1500, 2500]

    anom_colors = [
        "#C30000", "#FF3300", "#FFA200", "#FFEB7A",
        "#7A5147", "#B78E84", "#F4DFD5", "#FFFFFF",
        "#CBFFC1", "#7AF975", "#1EB71E", "#98D5FE",
        "#2984F4", "#DFDFFF", "#8272EF"
    ]
    limits_anom = [-500, -300, -200, -100, -50, -25, -10, 10, 25, 50, 100, 200, 300, 500]

    mask_sorted = mask_nc['mask_data']

    pad = 0.5
    extent_box = [bounds[0]-pad, bounds[2]+pad, bounds[1]-pad, bounds[3]+pad]

    out_dir = base_dir / "spatial_maps" / country / rndta 
    out_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================
    # 4. LOOP PERIODS
    # =========================================================
    for p in periodes:
        print(f"\n🕒 Période : {p} jours")

        daily_slice = daily_data[precip_var].isel(time=slice(-p, None))
        clim_slice = clim_data[precip_var].isel(time=slice(-p, None))

        pcur = p * daily_slice.mean(dim='time').load()
        pclim = p * clim_slice.mean(dim='time').load()
        
        time_values = daily_slice.time
        start_date = pd.to_datetime(time_values[1].values).strftime('%d%b%Y')
        end_date = pd.to_datetime(time_values[-1].values).strftime('%d%b%Y')

        anom = pcur - pclim
        pclimrate = pclim 
        drymask = xr.where(pclimrate >= 0.25, 1, 0)

        pcnp = ((pcur / pclim.where(pclim > 1)) * 100)
        pcnp = pcnp.where(np.isfinite(pcnp)).clip(max=400)
        pcnp_clean = pcnp.where(drymask == 1)

        mask_resized = mask_sorted.interp(
            lat=pcur.lat,
            lon=pcur.lon,
            method="nearest"
        ).fillna(0)
        
        # Application des fonctions de lissage
        anom = apply_clean_smoothing(anom, sigma=0)
        pcnp = apply_clean_smoothing(pcnp_clean, sigma=0)
        pcur = apply_clean_smoothing(pcur, sigma=0)
        pclim = apply_clean_smoothing(pclim, sigma=0)
        
        anom_masked = anom.where(drymask == 1) #& (mask_resized == 1))
        pcur_masked = pcur.where(drymask == 1) #& (mask_resized == 1))
        pclim_masked = pclim.where(drymask == 1)# & (mask_resized == 1))
        pcnp_masked = pcnp_clean.where(mask_resized == 1)

        plot_configs = {
            "precip_anomaly": {"data": anom_masked, "levels": limits_anom, "colors": anom_colors, "title": "Anomaly", "unit": "mm"},
            "total_precip": {"data": pcur_masked, "levels": limits_precip, "colors": colors_precip, "title": "Current Precip", "unit": "mm"},
            "normal_precip": {"data": pclim_masked, "levels": limits_precip, "colors": colors_precip, "title": "Climatology", "unit": "mm"},
            "percent_normal_precip": {"data": pcnp_masked, "levels": limits_precip, "colors": colors_precip, "title": "Percent of Normal", "unit": "%"}
        }

        # =========================================================
        # 5. DESSIN MULTI-COUCHES PERFORMANCE
        # =========================================================
        for key, config in plot_configs.items():
            print(f"🎨 {config['title']}")

            fig, ax = plt.subplots(figsize=(12, 9.27), subplot_kw={'projection': ccrs.PlateCarree()})

            ax.set_extent(extent_box, crs=ccrs.PlateCarree())
            
            # 🟡 COULEUR EXTERNE INITIALE
            #ax.set_facecolor("#e0e0e0")
            ax.set_facecolor("#ffffff")

            # =====================================================================
            # 🟢 ÉTAPE 0 : LE TAPIS DE FOND INTERNE (Zones Sèches Masquées)
            # =====================================================================
            # Dessine la forme du pays complète en blanc (zorder 1) sous les données.
            # Les zones exclues par le drymask (NaN) hériteront de ce blanc impeccable.
            ax.add_geometries([country_geom], ccrs.PlateCarree(),
                              facecolor="#e0e0e06c", edgecolor='none', zorder=1)
            # =====================================================================

            # Étape B : Dessin de la matrice de données cartographiques (zorder 2)
            im = ax.contourf(
                config['data'].lon,
                config['data'].lat,
                config['data'],
                levels=config['levels'],
                colors=config['colors'],
                extend='both',
                transform=ccrs.PlateCarree(),
                zorder=2
            )

            # =====================================================================
            # 🟡 APPLICATION DU MASQUE INVERSÉ EXTERNE (zorder 5)
            # =====================================================================
            # Recouvre hermétiquement toutes les coulures de lissage hors frontières
            ax.add_geometries([inverse_mask], ccrs.PlateCarree(), 
                              facecolor="#ffffff", edgecolor='none', zorder=5)

            # =====================================================================
            # 🔴 TRACÉ CHIRURGICAL ET STRATIFIÉ DES FRONTIÈRES (zorder 6 & 7)
            # =====================================================================
            # 1. Frontières administratives régionales internes (Discrètes)
            ax.add_geometries(shp_adm1.geometry, ccrs.PlateCarree(),
                              facecolor='none', edgecolor='#4a4a4a', linewidth=0.5, zorder=6)
            
            # 2. Frontière nationale principale (Contour souverain net au premier plan)
            ax.add_geometries([country_geom], ccrs.PlateCarree(),
                              facecolor='none', edgecolor='black', linewidth=1.5, zorder=7)

            # Barre d'échelle
            cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
            cbar.set_label(config['unit'])

            # Grille de coordonnées
            gl = ax.gridlines(draw_labels=True, alpha=0.6)
            gl.top_labels = False
            gl.right_labels = False

            # Titre appliqué (Rouge de style GrADS)
            if key == "normal_precip":
                plt.title(f"{rndta.upper()} - {config['title']} : {p}-day \n Period: {start_date[:-4]} - {end_date[:-4]}", color='red')
            else:
                plt.title(f"{rndta.upper()} - {config['title']} : {p}-day \n Period: {start_date} - {end_date}", color='black')

            # Sauvegarde et libération immédiate de la RAM
            plt.savefig(out_dir / f"{country}_{p}day_{rndta}_{key}.png", dpi=150, bbox_inches='tight')
            plt.clf()
            plt.close(fig)

            print("✅ saved")
            
    print(f"✅ Cartes de précipitations {rndta} terminées avec double masque couleur.")