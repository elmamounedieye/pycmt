import os
import shutil
import time
from pathlib import Path

# CORRECTION DES IMPORTS : Utilisation du nom du package global installé
from pycmt.core.downloader import (
    download_arc2_data,
    download_gadm_country,
    download_rfe2,
    download_runoff_data,
    download_spi,
    download_spp_noaa,
    download_xsm_data,
    get_country_iso,
    run_retrieval_vhi,
)
from pycmt.core.generate_mask import run_workflow
from pycmt.modules.generate_timeseries import generate_tseries
from pycmt.modules.hydrology import generate_runoff, generate_soilmoisture
from pycmt.modules.indices import generate_spi as calc_spi
from pycmt.modules.indices import do_vhi, run_orchestrator_spp
from pycmt.modules.precipitation import plot_precip
from pycmt.visualization.generate_grid import generate_pixel_arguments, plot_pix_coordinates
from pycmt.visualization.generate_html import build_country_dashboard, generate_html_map


def plot_precip_ts(country: str, rndta: str, rsl, ts_rsl):
    if country == "Africa":
        country_iso = "AFR"
        source = Path.cwd().resolve().parent[2] / "data" / "AFR_adm"
        destination = Path.re / "data" / "gis_resources" / "countries"
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, destination)
    else:
        country_iso = get_country_iso(country)
        download_gadm_country(country_iso)
        print("✅ Téléchargement du shapefile terminé !")
    
    config = {
        0.036: '0p036',
        0.0375: '0p0375',
        0.25: '0p25',
        0.5: '0p5',
        0.1: '0p1',
        1.0: '1p0'
    }
    
    run_workflow(country_iso, country)
    generate_pixel_arguments(ts_rsl, country_iso, country)
    plot_pix_coordinates(country, country_iso, rndta)
    
    if rndta.lower() == "arc2":
        download_arc2_data()
    if rndta.lower() == "rfe2":
        download_rfe2()

    plot_precip(rsl, config[rsl], country_iso, country, rndta)
    generate_tseries(country_iso, country, rndta)
    generate_html_map(country, rndta)
    print(f"🎉 Génération des précipitations et séries temporelles complétée.")
    return country_iso

def generate_spp(country, country_iso):
    print(f"📥 Téléchargement RFE2 SPP...")
    download_spp_noaa("rfe2")
    print(f"📥 Téléchargement CMORPH SPP...")
    download_spp_noaa("cmorph")
    
    print(f"⚙️ Génération RFE2 SPP...")
    run_orchestrator_spp(country, country_iso, "rfe2")
    print(f"⚙️ Génération CMORPH SPP...")
    run_orchestrator_spp(country, country_iso, "cmorph")
    print(f"✅ SPP complété.")

def generate_spi_(country, country_iso):
    print(f"📥 Téléchargement Données Runoff...")
    download_runoff_data()
    print(f"📥 Téléchargement Données Soil Moisture...")
    download_xsm_data()
    download_spi("cmorph")
    download_spi("rfe2")

    print(f"⚙️ Génération Runoff...")
    generate_runoff(country_iso, country)
    print(f"⚙️ Génération Soil Moisture...")
    generate_soilmoisture(country_iso, country)
    print(f"⚙️ Génération SPI CMORPH...")
    calc_spi(country_iso, country, "cmorph")
    print(f"⚙️ Génération SPI RFE2...")
    calc_spi(country_iso, country, "rfe2")
    print(f"✅ SPI complété.")

def generated_vhi(country, country_iso):
    print(f"📥 Téléchargement VHI...")
    run_retrieval_vhi()
    print(f"⚙️ Génération VHI...")
    do_vhi(country, country_iso)
    print(f"✅ VHI complété.")

def generate_dashboard(country, rndta):
    build_country_dashboard(country, rndta)


# =========================================================================
# BLOC D'EXÉCUTION PRINCIPAL (PRODUCTION)
# =========================================================================
"""
if __name__ == "__main__":
    country_target = "Senegal"
    data_source = "arc2"  # ou "rfe2"
    
    print(f"🏁 Démarrage du Climonitor pour : {country_target}")
    starting = time.time()
    
    # 1. Pipeline Précipitations et Séries Temporelles
    iso = plot_precip_ts(country_target, data_source, 0.25, 0.5)
    
    # 2. Décommentez les blocs suivants selon vos besoins de production :
    # generate_spp(country_target, iso)
    # generate_spi_(country_target, iso)
    # generated_vhi(country_target, iso)
    
    # 3. Assemblage de l'interface Dashboard HTML
    generate_dashboard(country_target, data_source)
    
    duration = time.time() - starting
    print(f"⏱️ Durée totale des calculs : {duration:.2f} secondes")"""