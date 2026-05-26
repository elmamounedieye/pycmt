import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') # Pour ne pas afficher de fenêtre
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MultipleLocator
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
from xgrads import open_CtlDataset
import os
import time
from pathlib import Path


def process_data(data):
    """Calcule les données journalières et les cumuls."""
    s = pd.Series(data.values)
    s = s.replace(-999.0, np.nan).clip(lower=0)
    daily = s.fillna(0)
    cumul = s.fillna(0).cumsum()
    return daily, cumul

def get_nice_step(max_val, is_cumul=False):
    """Calcule l'échelle intelligente des axes Y."""
    imax = int(round(max_val + (max_val / 50.0)))
    target_del = imax / 10.0
    steps = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100]
    if is_cumul:
        steps += [125, 175, 250, 350, 500]
    for s in steps:
        if s >= target_del: return imax, s
    return imax, steps[-1]

def generate_rainfall_plot(pcur_ts, pclim_ts, stnnm, lt, ln, stncnt, period, country, rndta):
    """Génère et sauvegarde le graphique final avec des axes parfaitement alignés."""
    r1, r2 = process_data(pcur_ts)
    r3, r4 = process_data(pclim_ts)
    
    dates = pcur_ts.time.values
    r4_high, r4_low = 1.2 * r4, 0.8 * r4
    
    imax1, del1 = get_nice_step(max(r1.max(), r3.max()))
    imax2, del2 = get_nice_step(max(r2.max(), r4.max()), is_cumul=True)
    
    # FIX 1 : sharex=True force l'alignement géométrique strict des deux graphiques
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 11), sharex=True, gridspec_kw={'height_ratios': [2, 2]})
    plt.subplots_adjust(hspace=0.15) # Réduit un peu l'espace vide entre les deux

    # --- Panneau Supérieur ---
    ax1.fill_between(dates, r2, r4_high, where=(r2 > r4_high), interpolate=True, color='#32CD32', alpha=0.5, label='120% Of Normal')
    ax1.fill_between(dates, r2, r4_low, where=(r2 < r4_low), interpolate=True, color='#A52A2A', alpha=0.5, label='80% Of Normal')
    ax1.fill_between(dates, r4_low, r4_high, color="#8B8B95", alpha=0.3)

    perc = (r2.iloc[-1] / r4.iloc[-1]) * 100 if r4.iloc[-1] != 0 else 0
    perf_txt = f"Current Performance: {perc:.1f}% of Normal"

    ax1.plot(dates, r4, color='gray', linestyle='--', linewidth=2, label='Climatological Normal')
    ax1.plot(dates, r2, color='black', linewidth=1, label=perf_txt)
    
    # --- Panneau Inférieur ---
    ax2.bar(dates, r1, color='#4682B4', width=0.7)

    # Formattage des axes
    my_fmt = mdates.DateFormatter('%d %b')
    for i, ax in enumerate([ax1, ax2]):
        ax.tick_params(labelbottom=True)
        ax.xaxis.set_major_formatter(my_fmt)
        ax.grid(True, linestyle='-', alpha=0.5)
        
        # FIX 2 : On place les légendes au même endroit sans empiéter sur la zone de tracé
        if i == 0:
            ax.legend(loc='upper left', fontsize='small', framealpha=0.9)
        else:
            ax.legend(loc='upper left', fontsize='small').set_visible(False) # Optionnel: cache la légende vide du bas

        plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
        ax.set_xlim(dates[0], dates[-1])
        
        if i == 0: # AX1 (Cumul)
            y_max = max(imax2 * 1.05, 7) 
            ax.set_ylim(bottom=0, top=y_max)
            ax.yaxis.set_major_locator(MultipleLocator(del2 if del2 > 0 else 1))
            ax.set_ylabel('Rainfall vs Normal (mm)', fontweight='bold')
        else: # AX2 (Journalier)
            y_max = max(imax1, 7)
            ax.set_ylim(bottom=0, top=y_max)
            ax.yaxis.set_major_locator(MultipleLocator(del1 if del1 > 0 else 1))
            ax.set_ylabel('Rainfall (mm)', fontweight='bold')

    plt.suptitle(f"{rndta.upper()} Point Time Series\n{len(dates)}-Day Rainfall @ {stnnm} ({lt}N, {ln}E)", fontsize=14, y=0.94)
    
    save_dir = Path(__file__).resolve().parents[1] / "data" / "ts_maps" / f"{country}" / rndta
    save_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{stncnt}_{period}.png"
    
    # FIX 3 : bbox_inches='tight' est réactivé pour nettoyer proprement les débordements de texte
    plt.savefig(save_dir / filename, dpi=100)#, bbox_inches='tight')
    plt.close(fig)
    #return filename



####### Execution ########
def generate_tseries(country_iso, country, rndta):
    base_dir = Path(__file__).resolve().parents[1] / "data" 
    output_dir = base_dir / "ts_maps"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    save_path = os.path.abspath(output_dir)
    
    if rndta.lower() == "arc2":
        daily_precip_path = base_dir / "ARC2" / "arc2" / "arc2.ctl"
        clim_path = base_dir / "ARC2" / "arc2_clim" / "arc2_clim.ctl"
        precip_var = "pmer2"
    elif rndta.lower() == "rfe2": 
        daily_precip_path = base_dir / "rfe2_data" / "rfe2_daily" / "rfe2daily.ctl"
        clim_path = base_dir / "rfe2_data" / "rfe2_clim" / "rfe2clim.ctl"
        precip_var = "r"

    #daily_data = open_CtlDataset(os.path.relpath(daily_precip_path))
    #clim_data = open_CtlDataset(os.path.relpath(clim_path))
    print(f"daily precip path: {daily_precip_path}")
    print(f"==================================")
    print(f"daily precip path with Path: {Path(daily_precip_path)}")
    print(f"==================================")
    print(f"daily precip path with Path 22: {str(daily_precip_path)}")
    print(f"==================================")
    """daily_data = open_CtlDataset(Path(daily_precip_path).resolve().as_posix())
    clim_data = open_CtlDataset(Path(clim_path).resolve().as_posix())"""
    #daily_data = open_CtlDataset(str(daily_precip_path)[2:])
    daily_data = open_CtlDataset(os.path.abspath(daily_precip_path).replace('./', ''))
    #clim_data = open_CtlDataset(str(clim_path)[2:])
    clim_data = open_CtlDataset(os.path.abspath(clim_path).replace('./', ''))
    #os.path.abspath(clim_path).replace('\\', '/')

    #######added######

    start_time = time.time()
    #with open('latlon_senegal', 'r') as f:
    with open(base_dir /f"pixelargs_{country}.txt", 'r') as f:
        for line_content in f:
            line = line_content.split()
            if not line: continue  # Saute les lignes vides
            
            stncnt, lt, ln, stnnm = line[0], float(line[1]), float(line[2]), line[-1]
            periodes = [7, 10, 30, 60, 90, 180]

            for p in periodes:
                pcur_ts = daily_data[precip_var].sel(lat=lt, lon=ln, method='nearest').isel(time=slice(-p, None)).load()
                pclim_ts = clim_data[precip_var].sel(lat=lt, lon=ln, method='nearest').isel(time=slice(-p, None)).load()

                # 2. Appel de la fonction du module
                #image_generee = generate_rainfall_plot(pcur_ts, pclim_ts, stnnm, lt, ln, stncnt, 30, country)
                generate_rainfall_plot(pcur_ts, pclim_ts, stnnm, lt, ln, stncnt, p, country, rndta)
            

    duration = time.time() - start_time
    print(f"Total duration : {duration}")