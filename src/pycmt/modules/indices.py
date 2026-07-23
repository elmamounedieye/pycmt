import gc
import os
import re
import shutil
from datetime import datetime, timedelta
from pathlib import Path

import cartopy.crs as ccrs
import geopandas as gpd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import shapely.geometry as sgeom
import xarray as xr
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
from matplotlib.colors import BoundaryNorm, ListedColormap
from scipy.ndimage import gaussian_filter
from shapely.validation import make_valid
from xgrads import open_CtlDataset

# =============================================================================
# 1. HELPER & UTILITY FUNCTIONS
# =============================================================================

def get_week_logic(days_ago):
    """Calculates week number and target dates matching GrADS logic."""
    target_date = datetime.now() - timedelta(days=days_ago)
    year = target_date.year
    julian_day = int(target_date.strftime("%j"))
    
    week_num = min(((julian_day - 1) // 7) + 1, 52)
    
    jan1 = datetime(year, 1, 1)
    start_day_val = jan1.weekday() + 1
    cur_wkday = target_date.weekday() + 1
    diff = cur_wkday - start_day_val
    if diff < 0:
        diff += 7
        
    bgn_date = target_date - timedelta(days=diff)
    end_date = bgn_date + timedelta(days=6)
    
    if julian_day > 357:
        end_date = datetime(year, 12, 31)
        
    return {
        "year": year,
        "week_str": f"{week_num:02d}",
        "file_tag": f"{year}0{week_num:02d}",
        "bgn_str": bgn_date.strftime("%d%b%Y").upper(),
        "end_str": end_date.strftime("%d%b%Y").upper()
    }


def read_country_config(filepath):
    """Reads bounding box configuration for a given country."""
    with open(filepath, 'r') as f:
        data = f.readline().split()
        return {
            'name': data[0],
            'lat1': float(data[1]), 'lat2': float(data[2]),
            'lon1': float(data[3]), 'lon2': float(data[4]),
            'lat_range': (float(data[1]), float(data[2])),
            'lon_range': (float(data[3]), float(data[4]))
        }


def read_captions(filepath):
    """Reads caption pairs from a metadata text file."""
    captions = []
    with open(filepath, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
        for i in range(0, len(lines), 2):
            captions.append((lines[i], lines[i+1]))
    return captions


def get_vhi_colormap():
    """Returns ListedColormap, Norm, and Levels for VHI visualization."""
    hex_colors = [
        "#FF00FF", "#E10032", "#FF7D7D", "#FFAA00", "#FFFF64",
        "#64FF64", "#009600", "#5050FF", "#0000C8"
    ]
    levels = [0, 6, 12, 24, 36, 48, 60, 72, 84, 100]
    cmap = ListedColormap(hex_colors)
    norm = BoundaryNorm(levels, len(hex_colors))
    return cmap, norm, levels


def get_grads_colors():
    """Returns GrADS-styled colormaps and norms for Below, Normal, and Above probability categories."""
    levels = [20, 40, 60, 75, 90, 100]
    cmaps = {
        'below': ListedColormap(["#FFFFFF", '#ffff80', '#ffc800', '#ff8000', '#cc0000']),
        'normal': ListedColormap(["#FFFFFF", '#c8ffc8', '#78ff78', '#00cc00', '#007d00']),
        'above': ListedColormap(["#FFFFFF", '#b4ffff', '#78d2ff', '#0078ff', '#003cff'])
    }
    norms = {k: BoundaryNorm(levels, v.N) for k, v in cmaps.items()}
    return cmaps, norms


def apply_clean_smoothing1(da, sigma=1.0):
    """Applies Gaussian smoothing while preserving NaN boundaries."""
    coords, dims, attrs, name = da.coords, da.dims, da.attrs, da.name
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


# =============================================================================
# 2. VHI (VEGETATION HEALTH INDEX) MODULE
# =============================================================================

def prepare_data_for_week(wk_info, week_idx, vhi_path, country_name="Africa"):
    """Prepares weekly VHI satellite files safely."""
    FIX_DIR = Path(__file__).resolve().parents[1] / "data"
    prefix = "VHP.G04.C07"
    suffix = f"P{wk_info['file_tag']}.VH.nc"
    
    source_file = None
    for sat in ["npp", "j01"]:
        f_name = Path(vhi_path) / f"{prefix}.{sat}.{suffix}"
        if os.path.exists(f_name):
            source_file = f_name
            break

    if not source_file:
        print(f"!!! Missing data : Week {wk_info['week_str']} ({wk_info['year']})")
        return

    shutil.copy(source_file, Path(vhi_path) / "vhi.nc")
    print(f"--- WEEK {week_idx} (Wk. {wk_info['week_str']}) ---")
    print(f"Période : {wk_info['bgn_str']} to {wk_info['end_str']}")


def prepare_vhi_data(path_vhi, path_mask):
    """Loads and masks VHI dataset using context managers."""
    with xr.open_dataset(path_vhi) as ds:
        ds_assigned = ds.assign_coords({
            "lat": ds.latitude,
            "lon": ds.longitude
        }).rename({
            "HEIGHT": "lat",
            "WIDTH": "lon"
        })
        vhi = ds_assigned["VHI"].load()
    
    with xr.open_dataset(path_mask) as mask_ds:
        mask_interp = mask_ds['mask_data'].interp(lat=vhi.lat, lon=vhi.lon, method="nearest").load()

    return vhi.where(mask_interp == 1)


def plot_vhi_map(da, extent_info, gdf_shp, wk_info, output_dir, country):
    """Renders VHI spatial plot using a pre-loaded GeoDataFrame."""
    print(f"Starting VHI map generation for Week {wk_info['week_str']}")
    cmap, norm, levels = get_vhi_colormap()

    fig = plt.figure(figsize=(11, 8.5))
    ax = plt.axes(projection=ccrs.PlateCarree())

    da = da.where(da > 0)
    im = ax.pcolormesh(
        da.lon, da.lat, da,
        cmap=cmap, norm=norm,
        transform=ccrs.PlateCarree(),
        zorder=1
    )

    if gdf_shp is not None:
        gdf_shp.plot(ax=ax, edgecolor="black", facecolor="none", linewidth=0.5, zorder=2)

    lon_min, lon_max = extent_info["lon_range"]
    lat_min, lat_max = extent_info["lat_range"]
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

    lon_delta = lon_max - lon_min
    step = 1.0 if lon_delta <= 5 else (3.0 if lon_delta <= 15 else (5.0 if lon_delta <= 30 else 10.0))

    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.5, color="gray", alpha=0.5, linestyle="--")
    gl.top_labels = gl.right_labels = False
    gl.xformatter, gl.yformatter = LONGITUDE_FORMATTER, LATITUDE_FORMATTER
    gl.xlocator = gl.ylocator = mticker.MultipleLocator(step)
    gl.xlabel_style = gl.ylabel_style = {"size": 9}

    plt.title(f"Vegetation Health Index - {extent_info['name']}", loc="left", fontsize=9, fontweight="bold", pad=7)
    plt.title(f"Week Ending {wk_info['end_str']}", loc="right", fontsize=9, style="italic")

    cbar = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.04, pad=0.08, extend="both")
    cbar.set_label("VHI (%)", fontsize=10)
    cbar.ax.set_xticklabels([str(l) for l in levels])

    output_path = output_dir / f"{country}_vhi{wk_info['week_str']}.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    
    plt.close(fig)
    plt.close('all')


def do_vhi(country, country_iso, init_day=0):
    """Main orchestrator for weekly VHI mapping."""
    base_dir = Path(__file__).resolve().parents[1] / "data" 
    vhi_dir_path = base_dir / "vhi" / "data"
    vhi_dir_path.mkdir(parents=True, exist_ok=True)

    vhi_dir_str = os.path.relpath(vhi_dir_path)
    country_info = base_dir / f"{country}_latlon"
    output_dir = base_dir / "vhi" / "vhi_maps" / f"{country}"
    output_dir.mkdir(parents=True, exist_ok=True)

    country_mask = base_dir / "gis_resources" / f"country_masks0p036/365dcal/{country_iso}_mask.nc"
    file_vhi = base_dir / "vhi" / "data" / "vhi.nc"
    file_shp = base_dir / "gis_resources" / "countries" / f"{country_iso}_adm" / f"{country_iso}_adm1.shp"
    
    # Pre-load Shapefile ONCE to avoid loop read-thrashing
    gdf_shp = gpd.read_file(file_shp) if file_shp.exists() else None

    print(f"Processing VHI for {country}...")

    for i in range(1, 7):
        nd = init_day + (i * 7)
        wk_info = get_week_logic(nd)
        prepare_data_for_week(wk_info, i, vhi_dir_str, country)
        try:
            conf = read_country_config(country_info)
            vhi_final = prepare_vhi_data(file_vhi, country_mask)
            plot_vhi_map(vhi_final, conf, gdf_shp, wk_info, output_dir, country)
        except Exception as e:
            print(f"Error during VHI execution iteration {i}: {e}")
            
        gc.collect()


# =============================================================================
# 3. SPP (SEASONAL PRECIPITATION PROBABILITY) MODULE
# =============================================================================

def run_orchestrator_spp(country, country_iso, rndta, mask_enabled=True):
    """Main orchestrator for Seasonal Precipitation Probability mapping."""
    ctl_files = [
        f'spp_{rndta}_comb_1ic-0proj.ctl', f'spp_{rndta}_comb_1ic-1proj.ctl',
        f'spp_{rndta}_comb_1ic-2proj.ctl', f'spp_{rndta}_comb_2ic-0proj.ctl',
        f'spp_{rndta}_comb_2ic-1proj.ctl', f'spp_{rndta}_comb_3ic-0proj.ctl'
    ]
    base_dir = Path(__file__).resolve().parents[1] / "data" 
    spp_dir_path = base_dir / "spp" / f"spp_data_{rndta}"
    spp_dir_path.mkdir(parents=True, exist_ok=True)

    captions_file_path = spp_dir_path / 'spp_timescales.txt'
    country_info = base_dir / f"{country}_latlon"
    output_dir = base_dir / "spp" / "spp_maps" / f"{country}" / f"{rndta}"
    output_dir.mkdir(parents=True, exist_ok=True)

    config = read_country_config(country_info)
    captions = read_captions(captions_file_path)
    cmaps, norms = get_grads_colors()

    print("Loading geographic boundaries for SPP...")
    shp_error = False
    try:
        shp_path = base_dir / "gis_resources" / "countries" / f"{country_iso}_adm" / f"{country_iso}_adm1.shp"
        shp_adm1 = gpd.read_file(shp_path).to_crs("EPSG:4326")
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
        print(f"⚠️ Error initializing SHP for SPP: {e}")
        shp_error = True
        extent_box = [config['lon1'], config['lon2'], config['lat1'], config['lat2']]

    mask_sorted = None
    if mask_enabled:
        mask_path = base_dir / "gis_resources" / "country_masks0p1" / "365dcal" / f"{country_iso}_mask.nc"
        with xr.open_dataset(mask_path) as mask_ds:
            mask_sorted = mask_ds['mask_data'].load()

    for i, ctl in enumerate(ctl_files):
        ds_ctl_path = spp_dir_path / ctl
        if not ds_ctl_path.exists():
            continue
            
        ds = open_CtlDataset(os.path.relpath(ds_ctl_path))
        ds.coords['lon'] = (ds.coords['lon'] + 180) % 360 - 180.125
        ds = ds.sortby('lon')
        
        v = list(ds.data_vars)
        p1 = ds[v[0]].isel(time=-1).load()
        p2 = ds[v[1]].isel(time=-1).load()
        p3 = ds[v[2]].isel(time=-1).load()
        
        # Close xgrads handle immediately after copying data arrays
        ds.close()

        max_p = np.maximum(np.maximum(p1, p2), p3)
        threshold = 100 / 3.0

        if mask_enabled and mask_sorted is not None:
            m = mask_sorted
            if not (m.lat.equals(p1.lat) and m.lon.equals(p1.lon)):
                m = m.interp(lat=p1.lat, lon=p1.lon, method='nearest')
            
            p1, p2, p3 = p1.where(m > 0), p2.where(m > 0), p3.where(m > 0)
            max_p = max_p.where(m > 0)

        fig = plt.figure(figsize=(10, 8.5))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_extent(extent_box, crs=ccrs.PlateCarree())
        ax.set_facecolor("#ffffff")

        if not shp_error:
            ax.add_geometries([country_geom], ccrs.PlateCarree(), facecolor="#e0e0e0", edgecolor='none', zorder=1)

        categories = [
            ('below', p1, (p1 == max_p) & (p1 > threshold)),
            ('normal', p2, (p2 == max_p) & (p2 > threshold)),
            ('above', p3, (p3 == max_p) & (p3 > threshold))
        ]

        for cat_name, data, mask in categories:
            plot_data = data.where(mask)
            if not plot_data.isnull().all():
                ax.pcolormesh(
                    p1.lon, p1.lat, plot_data,
                    cmap=cmaps[cat_name], norm=norms[cat_name],
                    shading='nearest', transform=ccrs.PlateCarree(), zorder=2
                )

        if not shp_error:
            ax.add_geometries([inverse_mask], ccrs.PlateCarree(), facecolor="#ffffff", edgecolor='none', zorder=5)
            ax.add_geometries(shp_adm1.geometry, ccrs.PlateCarree(), facecolor='none', edgecolor='#4a4a4a', linewidth=0.5, zorder=6)
            ax.add_geometries([country_geom], ccrs.PlateCarree(), facecolor='none', edgecolor='black', linewidth=1.5, zorder=7)

        plt.title(f"{captions[i][0]}\n{captions[i][1]}", fontsize=12, fontweight='bold', pad=15)

        match_month = re.search(r"Period\s*=\s*(\d+)", captions[i][0])
        match_proj = re.search(r"Period\s*=\s*(\d+)", captions[i][1])
        month_val = match_month.group(1) if match_month else "0"
        proj_val = match_proj.group(1) if match_proj else "0"

        for j, cat in enumerate(['below', 'normal', 'above']):
            cax = fig.add_axes([0.18 + j * 0.24, 0.05, 0.18, 0.01])
            cb = plt.colorbar(plt.cm.ScalarMappable(norm=norms[cat], cmap=cmaps[cat]), cax=cax, orientation='horizontal')
            cb.set_label(cat.upper(), fontsize=7, fontweight='bold', labelpad=2)
            cb.ax.tick_params(labelsize=6)

        dry_patch = mpatches.Patch(facecolor='#e0e0e0', edgecolor='#4a4a4a', linewidth=0.5)
        ax.legend([dry_patch], ['Drymask'], loc='lower left', fontsize=7, frameon=True, facecolor='#ffffff', edgecolor='#4a4a4a', framealpha=0.9)
        
        gl = ax.gridlines(draw_labels=True, linewidth=0.6, color='gray', alpha=0.6, linestyle='--', zorder=8)
        gl.top_labels = gl.right_labels = False
        gl.xlocator = gl.ylocator = mticker.MultipleLocator(10)

        plt.savefig(output_dir / f"spp_{country}_{rndta}_Month{month_val}Proj{proj_val}.png", dpi=150)
        
        plt.close(fig)
        plt.close('all')
        gc.collect()

    print(f"✅ SPP maps for {rndta} are generated.")


# =============================================================================
# 4. SPI (STANDARDIZED PRECIPITATION INDEX) MODULE
# =============================================================================

def generate_spi(country_iso, country, rndta):
    """Main orchestrator for Standardized Precipitation Index mapping."""
    base_dir = Path(__file__).resolve().parents[1] / "data" 
    mask_path = base_dir / "gis_resources" / "country_masks0p036" / "365dcal" / f"{country_iso}_mask.nc"
    spi_dir = base_dir / "spi" / "data" / f"{rndta}"
    spi_dir.mkdir(parents=True, exist_ok=True)
    spi_dir_str = os.path.relpath(spi_dir)
    
    output_dir = base_dir / "spi" / "spi_maps" / f"{country}" / f"{rndta}"
    output_dir.mkdir(parents=True, exist_ok=True)

    with xr.open_dataset(mask_path) as path_mask_nc:
        mask_sorted = path_mask_nc['mask_data'].sortby('lat').load()

    shp_path = base_dir / "gis_resources" / "countries" / f"{country_iso}_adm" / f"{country_iso}_adm1.shp"
    shp_adm1 = gpd.read_file(shp_path).to_crs("EPSG:4326")
    shp_adm1["geometry"] = shp_adm1.geometry.apply(make_valid)
    shp_adm1 = shp_adm1[shp_adm1.is_valid]
    
    bounds = shp_adm1.total_bounds
    country_geom = shp_adm1.geometry.unary_union
    if not country_geom.is_valid:
        country_geom = country_geom.buffer(0)
        
    world_box = sgeom.box(bounds[0]-10, bounds[1]-10, bounds[2]+10, bounds[3]+10)
    inverse_mask = world_box.difference(country_geom)

    spi_levels = [-2, -1.6, -1.3, -0.8, -0.5, 0.5, 0.8, 1.3, 1.6, 2]
    spi_colors = [
        '#B20000', '#FF0000', '#FF6600', '#FFBD33', '#FFFF99', 
        '#FFFFFF', '#B2EBF2', '#80B3FF', '#3385FF', '#1A53FF', '#311B92'
    ]

    periods = [1, 3, 6, 12, 24]

    for p in periods:
        spi_ctl = os.path.join(spi_dir_str, f'{rndta}.spi.{p}.mo.ctl')
        mask_ctl = os.path.join(spi_dir_str, f'drymask{p}.ctl')
        
        if not (os.path.exists(spi_ctl) and os.path.exists(mask_ctl)):
            continue

        ds_spi = open_CtlDataset(spi_ctl)
        ds_m = open_CtlDataset(mask_ctl)

        ds_spi.coords['lon'] = (ds_spi.coords['lon'] + 180) % 360 - 180.25
        ds_spi = ds_spi.sortby('lon')

        ds_m.coords['lon'] = (ds_m.coords['lon'] + 180) % 360 - 180.25
        ds_m = ds_m.sortby('lon')

        spi = ds_spi['p'].isel(time=-1, lev=0).load()
        drymask_raw = ds_m['dm'].isel(time=-1, lev=0).load()

        ds_spi.close()
        ds_m.close()

        drymask = drymask_raw.interp(
            lat=mask_sorted.lat, lon=mask_sorted.lon, method="nearest"
        ).fillna(0)

        spi_smooth = apply_clean_smoothing1(spi, sigma=1.5)
        spi_smooth = spi_smooth.interp(lat=mask_sorted.lat, lon=mask_sorted.lon, method="nearest").fillna(0)
        spi_masked = spi_smooth.where((drymask == 1.) & (mask_sorted == 1))

        fig = plt.figure(figsize=(12, 9.27))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_facecolor("#ffffff")

        ax.add_geometries([country_geom], ccrs.PlateCarree(), facecolor="#e0e0e0", edgecolor='none', zorder=1)

        im = ax.contourf(
            spi_masked.lon, spi_masked.lat, spi_masked,
            levels=spi_levels, colors=spi_colors, extend='both',
            transform=ccrs.PlateCarree(), zorder=4
        )

        pad = 0.5
        ax.set_extent([bounds[0]-pad, bounds[2]+pad, bounds[1]-pad, bounds[3]+pad], crs=ccrs.PlateCarree())

        ax.add_geometries([inverse_mask], ccrs.PlateCarree(), facecolor="#ffffff", edgecolor='none', zorder=5)
        ax.add_geometries(shp_adm1.geometry, ccrs.PlateCarree(), facecolor='none', edgecolor='#4a4a4a', linewidth=0.5, zorder=6)
        ax.add_geometries([country_geom], ccrs.PlateCarree(), facecolor='none', edgecolor='black', linewidth=1.5, zorder=7)

        gl = ax.gridlines(draw_labels=True, linewidth=0.6, color='gray', alpha=0.6, linestyle='--', zorder=8)
        gl.top_labels = gl.right_labels = False
        gl.xlocator = gl.ylocator = mticker.MultipleLocator(10)

        date_str = pd.to_datetime(ds_spi.time.values[-1]).strftime("%b %Y")
        plt.title(f"{rndta.upper()} ADJ 00Z SPI \n{p}-Month Period Ending {date_str}", fontsize=14)

        cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
        cbar.set_label('Percentile (%)')

        plt.savefig(output_dir / f"{country}_spi_{rndta}_{p}mo.png", dpi=150, bbox_inches='tight')

        plt.close(fig)
        plt.close('all')
        gc.collect()

    print(f"✅ SPI maps generated for {country}.")
