import os
import shutil
import time
from pathlib import Path

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

from pathlib import Path
from IPython.display import display
import ipywidgets as widgets
import pycmt

def run_uploader(allowed_extensions=".ctl, .nc, .txt, .idx", subfolder="data"):
    dest_dir = Path(pycmt.__file__).resolve().parent / subfolder
    dest_dir.mkdir(parents=True, exist_ok=True)

    uploader = widgets.FileUpload(accept=allowed_extensions, multiple=False)
    btn = widgets.Button(description="Save in source directory", button_style="success", icon="check")
    out = widgets.Output()

    def upload_data(b):
        with out:
            out.clear_output()
            if not uploader.value:
                return print("⚠️ Please, select a file first.")
            try:
                info = uploader.value[0] if isinstance(uploader.value, (list, tuple)) else list(uploader.value.values())[0]
                name = info.get("name") or info.get("metadata", {}).get("name")
                
                dest_path = dest_dir / name
                with open(dest_path, "wb") as f:
                    f.write(bytes(info["content"]))
                print(f"✅ SUCCESS: [{name}] saved.\nPath: {dest_path.resolve().as_posix()}")
            except Exception as e:
                print(f"❌ Error: {str(e)}")

    btn.on_click(upload_data)
    print("Select the file and click validate:")
    display(uploader, btn, out)

def plot_precip_ts(country: str, rndta: str, rsl, ts_rsl):
    if country == "Africa":
        country_iso = "AFR"
        
        package_root = Path(__file__).resolve().parent
        
        source = package_root / "data" / "AFR_adm"
        
        destination = package_root / "data" / "gis_resources" / "countries" / "AFR_adm"
        
        destination.parent.mkdir(parents=True, exist_ok=True)

        shutil.copytree(source, destination, dirs_exist_ok=True)
        print("✅ AFR_adm folder copied!")
    else:
        country_iso = get_country_iso(country)
        print(f" country iso : {country_iso}")
        download_gadm_country(country_iso)
        print("✅ Downloading shapefile complete!")
    
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
    print(f"🎉 Spatial maps and time series complete.")
    return country_iso

def generate_spp(country, country_iso):
    print(f"📥 Downloading RFE2 SPP...")
    download_spp_noaa("rfe2")
    print(f"📥 Downloading CMORPH SPP...")
    download_spp_noaa("cmorph")
    
    print(f"⚙️ Generating RFE2 SPP...")
    run_orchestrator_spp(country, country_iso, "rfe2")
    print(f"⚙️ Generating CMORPH SPP...")
    run_orchestrator_spp(country, country_iso, "cmorph")
    print(f"✅ SPP complete.")

def generate_spi_(country, country_iso):
    print(f"📥 Downloading Runoff...")
    download_runoff_data()
    print(f"📥 Downloading Soil Moisture...")
    download_xsm_data()
    download_spi("cmorph")
    download_spi("rfe2")

    print(f"⚙️ Generating Runoff...")
    generate_runoff(country_iso, country)
    print(f"⚙️ Generating Soil Moisture...")
    generate_soilmoisture(country_iso, country)
    print(f"⚙️ Generating SPI CMORPH...")
    calc_spi(country_iso, country, "cmorph")
    print(f"⚙️ Generating SPI RFE2...")
    calc_spi(country_iso, country, "rfe2")
    print(f"✅ SPI complete.")

def generated_vhi(country, country_iso):
    print(f"📥 Téléchargement VHI...")
    run_retrieval_vhi()
    print(f"⚙️ Generating VHI...")
    do_vhi(country, country_iso)
    print(f"✅ VHI complete.")

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
    #starting = time.time()
    
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