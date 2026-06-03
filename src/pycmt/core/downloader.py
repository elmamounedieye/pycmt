import gzip
import json
import os
import shutil
import socket
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import geopandas as gpd
import numpy as np
import requests
import rioxarray
import urllib3
import urllib3.util.ssl_ as urllib3_ssl
from requests.adapters import HTTPAdapter
import xarray as xr
from rasterio.features import rasterize
from rasterio.transform import from_origin

import pycmt

# =========================================================================
# SYSTEM-LEVEL NETWORK OPTIMIZATION FOR MACOS
# =========================================================================

def allowed_gai_family():
    """Forces IPv4 to eliminate the implicit 5-second DNS handshake timeouts on Mac."""
    return socket.AF_INET

import urllib3.util.connection as urllib3_cn
urllib3_cn.allowed_gai_family = allowed_gai_family

# Configuration de la session avec pool de connexions élargi pour le multi-threading
http_session = requests.Session()
adapter = HTTPAdapter(pool_connections=25, pool_maxsize=25, max_retries=3)
http_session.mount("https://", adapter)
http_session.mount("http://", adapter)

http_session.headers.update({
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
})

class DESAdapter(HTTPAdapter):
    def init_poolmanager(self, *args, **kwargs):
        context = urllib3_ssl.create_urllib3_context()
        context.load_default_certs()
        context.set_ciphers('DEFAULT@SECLEVEL=1')
        kwargs['ssl_context'] = context
        return super(DESAdapter, self).init_poolmanager(*args, **kwargs)

http_session.mount('https://sgbd.acmad.org', DESAdapter())

# =========================================================================
# ASYNC-BUFFERED CORE DOWNLOADING FUNCTIONS
# =========================================================================

def download_file(url, filename):
    """Downloads files with an expanded 128KB block buffer."""
    try:
        response = http_session.get(url, stream=True, timeout=30)
        if response.status_code == 200:
            with open(filename, "wb") as f:
                for chunk in response.iter_content(chunk_size=131072):
                    f.write(chunk)
            print(f"Succeeded : {Path(filename).name} downloading.")
            return True
        else:
            return False
    except Exception as e:
        print(f"URL failure : {e}")
        return False

def manage_download(url, dest_path, is_gzip=False, min_size_kb=2000):
    """High-speed download utilizing memory buffers before hitting Mac disk storage."""
    if dest_path.exists() and (dest_path.stat().st_size / 1024) > min_size_kb:
        return True 

    try:
        response = http_session.get(url, stream=True, timeout=15)
        response.raise_for_status()
        
        if is_gzip:
            # Téléchargement complet en RAM puis décompression vectorielle (évite les verrous disques macOS)
            compressed_data = response.content
            decompressed_data = gzip.decompress(compressed_data)
            with open(dest_path, 'wb') as f_out:
                f_out.write(decompressed_data)
        else:
            with open(dest_path, 'wb') as f_out:
                for chunk in response.iter_content(chunk_size=131072):
                    f_out.write(chunk)
        return True
    except Exception as e:
        if dest_path.exists():
            dest_path.unlink()
        return False

# =========================================================================
# GEOSPATIAL & METADATA MANAGEMENT
# =========================================================================

def get_country_iso(country: str):
    BASE_DIR = Path(__file__).resolve().parents[1]
    data_file = BASE_DIR / "data" / "countries_iso_dict.json"
    try:
        with open(data_file, 'r', encoding='utf-8') as f:
            countries = json.load(f)
        return countries[country]
    except FileNotFoundError:
        return f"❌ Error : Data file {data_file} not found."
    except KeyError:
        return f"❌ Error : '{country}' is not in the dictionary."

def download_gadm_country(iso_code, file_format="shp"):
    iso_code = iso_code.upper()
    filename = f"{iso_code}_adm.zip"
    save_directory = Path(__file__).resolve().parents[1] / "data" / "gis_resources" / "countries"
    full_save_path = save_directory / filename

    if full_save_path.exists():
        print(f"Directory already exists : {full_save_path}")
        return os.path.splitext(full_save_path)[0]
    
    save_directory.mkdir(parents=True, exist_ok=True)
    url = f"https://geodata.ucdavis.edu/gadm/gadm4.1/{file_format}/gadm41_{iso_code}_{file_format}.zip"
    print(f"Downloading {iso_code} shapefile...")

    if download_file(url, full_save_path):
        with zipfile.ZipFile(full_save_path, 'r') as zip_ref:
            country_folder = save_directory / f"{iso_code}_adm"
            os.makedirs(country_folder, exist_ok=True)
            zip_ref.extractall(country_folder)
        rename_country_shapefiles(iso_code)

def rename_country_shapefiles(country_iso):
    folder = Path(__file__).resolve().parents[1] / "data" / "gis_resources" / "countries" / f"{country_iso}_adm"
    expected_final_file = folder / f"{country_iso}_adm0.shp"
    if expected_final_file.exists():
        return
    
    for path in folder.glob(f"gadm41_{country_iso}_*.shp"):
        parts = path.stem.split('_')
        if len(parts) < 3: continue
        adm_level = parts[2]
        for ext in ["shx", "shp", "prj", "dbf", "cpg"]:
            src = folder / f"gadm41_{country_iso}_{adm_level}.{ext}"
            dst = folder / f"{country_iso}_adm{adm_level}.{ext}"
            if src.exists():
                try:
                    shutil.copy2(src, dst)
                except Exception:
                    pass

def generate_ctl(template_name, output_path, replacements):
    template_path = Path(template_name)
    if not template_path.exists():
        return
    content = template_path.read_text()
    for key, value in replacements.items():
        if "DIR" in key:
            content = content.replace(key, "^")
        else:
            content = content.replace(key, str(value))
    content = content.replace('^^', '^').replace('?\\', '^').replace('?/', '^')
    output_path.write_text(content)

# =========================================================================
# PARALLELIZED FAST WORKFLOW PIPELINES
# =========================================================================

def download_arc2_data(init_day_offset=1):
    base_dir = Path(__file__).resolve().parents[1]
    daily_dir = base_dir / "data" / "ARC2" / "arc2"
    clim_dir = base_dir / "data" / "ARC2" / "arc2_clim"
    
    daily_dir.mkdir(parents=True, exist_ok=True)
    clim_dir.mkdir(parents=True, exist_ok=True)

    today = datetime.now() - timedelta(days=init_day_offset)
    
    while True:
        target_date = today - timedelta(days=1)
        date_str = target_date.strftime("%Y%m%d")
        daily_filename = f"daily_clim.bin.{date_str}"
        daily_path = daily_dir / daily_filename
        daily_url = f"https://ftp.cpc.ncep.noaa.gov/fews/fewsdata/africa/arc2/bin/{daily_filename}.gz"
        
        if manage_download(daily_url, daily_path, is_gzip=True):
            break
        today -= timedelta(days=1)
            
    print(f"🚀 Mac Parallel Engine: Launching multi-threaded download for ARC2...")
    tasks = []
    for i in range(1, 181):
        t_date = today - timedelta(days=i)
        d_str = t_date.strftime("%Y%m%d")
        m_nd = t_date.strftime("%m%d")
        
        tasks.append((f"https://ftp.cpc.ncep.noaa.gov/fews/fewsdata/africa/arc2/bin/daily_clim.bin.{d_str}.gz", daily_dir / f"daily_clim.bin.{d_str}", True))
        tasks.append((f"https://ftp.cpc.ncep.noaa.gov/fews/fewsdata/africa/arc2/bin/daily_clim.bin.{d_str}.gz", daily_dir / f"daily_clim.bin.{d_str}", True))
        tasks.append((f"https://ftp.cpc.ncep.noaa.gov/fews/AFR_CLIM/ARC2/CLIMATOLOGY_DATA/DAILY_MEANS/clim.bin.{m_nd}", clim_dir / f"clim.bin.{m_nd}", False))

    with ThreadPoolExecutor(max_workers=8) as executor:
        executor.map(lambda p: manage_download(p[0], p[1], p[2]), tasks)

    start_date_str = (today - timedelta(days=180)).strftime("%d%b%Y")
    generate_ctl(base_dir / "data" / "template_arc2.ctl", daily_dir / "arc2.ctl", {"DLDATADIR": "^", "STDATE": start_date_str})
    generate_ctl(base_dir / "data" / "template_arc2_clim.ctl", clim_dir / "arc2_clim.ctl", {"CLMDLDATADIR": "^", "STDATE": start_date_str})
    print(f"✅ ARC2 Synchronization Complete.")

def download_rfe2(days_offset=1):
    base_dir = Path(pycmt.__file__).resolve().parent / "data"
    output_dir = base_dir / "rfe2_data" / "rfe2_daily"
    clim_dir = base_dir / "rfe2_data" / "rfe2_clim"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    clim_dir.mkdir(parents=True, exist_ok=True)
    
    yesterday = datetime.now().date() - timedelta(days=(days_offset + 1))
    existing_files = list(output_dir.glob("all_products.bin.*"))
    
    if not existing_files:
        n_missing = 180
    else:
        dates_in_disk = []
        for f in existing_files:
            try:
                date_part = f.name.split(".")[-1]
                dates_in_disk.append(datetime.strptime(date_part, "%Y%m%d").date())
            except ValueError:
                continue
        n_missing = (yesterday - max(dates_in_disk)).days if dates_in_disk else 180

    if n_missing <= 0 and (output_dir / "rfe2daily.ctl").exists():
        return
        
    print(f"🚀 Mac Parallel Engine: Downloading RFE2 payloads in parallel (8 threads)...")
    tasks = []
    for i in range(1, 181):
        target_date = datetime.now() - timedelta(days=(days_offset + i))
        date_str = target_date.strftime("%Y%m%d")  
        mmdd = target_date.strftime("%m%d")        
        
        tasks.append((f"https://ftp.cpc.ncep.noaa.gov/fews/fewsdata/africa/rfe2/bin/all_products.bin.{date_str}.gz", output_dir / f"all_products.bin.{date_str}", True))
        tasks.append((f"https://ftp.cpc.ncep.noaa.gov/fews/clim_dly/clim_RFE2/clim_dly.{mmdd}", clim_dir / f"clim_dly.{mmdd}", False))

    with ThreadPoolExecutor(max_workers=8) as executor:
        executor.map(lambda p: manage_download(p[0], p[1], p[2]), tasks)

    start_date = datetime.now() - timedelta(days=(days_offset + 180))
    start_date_str = start_date.strftime("%d%b%Y")
    generate_ctl(base_dir / "template_rfe2.ctl", output_dir / "rfe2daily.ctl", {"DLDATADIR": "^", "NNDAYS": 180, "STDATE": start_date_str})
    generate_ctl(base_dir / "rfe2clim.ctl", clim_dir / "rfe2clim.ctl", {"NNDAYS": 180, "STDATE": start_date_str})
    print(f"✅ RFE2 Synchronization Complete.")

def run_retrieval_vhi():
    INIT_DAY = 0
    TARGET_DIR = Path(__file__).resolve().parents[1] / "data" / "vhi" / "data"
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    
    for f in os.listdir(TARGET_DIR):
        if f.endswith(".png"):
            os.remove(TARGET_DIR / f)

    tasks = []
    for i in range(1, 7):
        nd = INIT_DAY + (i * 7)
        target_date = datetime.now() - timedelta(days=nd)
        year = target_date.year
        julian_day = int(target_date.strftime("%j"))
        week_num = min(((julian_day - 1) // 7) + 1, 52)
        wk_tag = f"{year}{week_num:03d}"
        
        save_path = TARGET_DIR / f"VHP.G04.C07.j01.P{wk_tag}.VH.nc"
        if not save_path.exists():
            url = f"https://www.star.nesdis.noaa.gov/data/pub0018/VHPdata4users/data/Blended_VH_4km/VH/VHP.G04.C07.j01.P{wk_tag}.VH.nc"
            tasks.append((url, save_path))

    if tasks:
        print(f"🚀 Mac Parallel Engine: Fetching VHI time series (Parallel)...")
        with ThreadPoolExecutor(max_workers=3) as executor:
            executor.map(lambda p: download_file(p[0], p[1]), tasks)

def download_spp_noaa(rndta):
    base_urls = [
        f"https://ftp.cpc.ncep.noaa.gov/International/spp/{rndta}/spp_{rndta}_comb_1ic-0proj.ctl",
        f"https://ftp.cpc.ncep.noaa.gov/International/spp/{rndta}/spp_{rndta}_comb_1ic-0proj.dat",
        f"https://ftp.cpc.ncep.noaa.gov/International/spp/{rndta}/spp_{rndta}_comb_1ic-1proj.ctl",
        f"https://ftp.cpc.ncep.noaa.gov/International/spp/{rndta}/spp_{rndta}_comb_1ic-1proj.dat",
        f"https://ftp.cpc.ncep.noaa.gov/International/spp/{rndta}/spp_{rndta}_comb_1ic-2proj.ctl",
        f"https://ftp.cpc.ncep.noaa.gov/International/spp/{rndta}/spp_{rndta}_comb_1ic-2proj.dat",
        f"https://ftp.cpc.ncep.noaa.gov/International/spp/{rndta}/spp_{rndta}_comb_2ic-0proj.ctl",
        f"https://ftp.cpc.ncep.noaa.gov/International/spp/{rndta}/spp_{rndta}_comb_2ic-0proj.dat",
        f"https://ftp.cpc.ncep.noaa.gov/International/spp/{rndta}/spp_{rndta}_comb_2ic-1proj.ctl",
        f"https://ftp.cpc.ncep.noaa.gov/International/spp/{rndta}/spp_{rndta}_comb_2ic-1proj.dat",
        f"https://ftp.cpc.ncep.noaa.gov/International/spp/{rndta}/spp_{rndta}_comb_3ic-0proj.ctl",
        f"https://ftp.cpc.ncep.noaa.gov/International/spp/{rndta}/spp_{rndta}_comb_3ic-0proj.dat",
        f"https://ftp.cpc.ncep.noaa.gov/International/spp/{rndta}/spp_timescales.txt"
    ]
    target_dir = Path(__file__).resolve().parents[1] / "data" / "spp" / f"spp_data_{rndta}"
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"🚀 Mac Parallel Engine: Syncing SPP {rndta.upper()} fields...")
    with ThreadPoolExecutor(max_workers=4) as executor:
        executor.map(lambda url: download_file(url, target_dir / url.split('/')[-1]), base_urls)

    for ctl_file in target_dir.glob("*.ctl"):
        content = ctl_file.read_text()
        if "DSET ^" not in content:
            ctl_file.write_text(content.replace("DSET ", "DSET ^"))

def download_spi(rndta: str):
    # Les dictionnaires d'URLs restent identiques...
    spi_data_urls = {
        "cmorph": [f"https://ftp.cpc.ncep.noaa.gov/fews/DroughtMonitor/spi/cmorph/{f}" for f in ["drymask12.bin","drymask12.ctl","drymask1.bin","drymask1.ctl","drymask24.bin","drymask24.ctl","drymask3.bin","drymask3.ctl","drymask6.bin","drymask6.ctl","globalmask0.25.dat","landmask.ctl","cmorph.spi.12.mo.bin","cmorph.spi.12.mo.ctl","cmorph.spi.1.mo.bin","cmorph.spi.1.mo.ctl","cmorph.spi.24.mo.bin","cmorph.spi.24.mo.ctl","cmorph.spi.3.mo.bin","cmorph.spi.3.mo.ctl","cmorph.spi.6.mo.bin","cmorph.spi.6.mo.ctl"]],
        "rfe2": [f"https://ftp.cpc.ncep.noaa.gov/fews/DroughtMonitor/spi/rfe2/{f}" for f in ["drymask12.bin","drymask12.ctl","drymask1.bin","drymask1.ctl","drymask24.bin","drymask24.ctl","drymask3.bin","drymask3.ctl","drymask6.bin","drymask6.ctl","mask.ctl","mask.gra","rfe2.spi.12.mo.bin","rfe2.spi.12.mo.ctl","rfe2.spi.1.mo.bin","rfe2.spi.1.mo.ctl","rfe2.spi.24.mo.bin","rfe2.spi.24.mo.ctl","rfe2.spi.3.mo.bin","rfe2.spi.3.mo.ctl","rfe2.spi.6.mo.bin","rfe2.spi.6.mo.ctl"]]
    }
    if rndta not in spi_data_urls: return
    base_dir = Path(__file__).resolve().parents[1] / "data" / "spi" / "data" / f"{rndta}"
    base_dir.mkdir(parents=True, exist_ok=True)

    print(f"🚀 Mac Parallel Engine: Downloading SPI {rndta.upper()} fields...")
    with ThreadPoolExecutor(max_workers=6) as executor:
        executor.map(lambda url: download_file(url, base_dir / url.split('/')[-1]), spi_data_urls[rndta])

def download_runoff_data():
    base_dir = Path(__file__).resolve().parents[1] / "data" / "spi" / "data" / "Runoff"
    base_dir.mkdir(parents=True, exist_ok=True)
    urls = [f"https://ftp.cpc.ncep.noaa.gov/fews/DroughtMonitor/Runoff/{f}" for f in ["drymask12.bin","drymask12.ctl","drymask1.bin","drymask1.ctl","drymask24.bin","drymask24.ctl","drymask3.bin","drymask3.ctl","drymask6.bin","drymask6.ctl","globalmask0.5.dat","landmask.ctl","runoff.12.mo.bin","runoff.12.mo.ctl","runoff.1.mo.bin","runoff.1.mo.ctl","runoff.24.mo.bin","runoff.24.mo.ctl","runoff.3.mo.bin","runoff.3.mo.ctl","runoff.6.mo.bin","runoff.6.mo.ctl"]]
    
    print(f"🚀 Mac Parallel Engine: Downloading Hydrology Runoff fields...")
    with ThreadPoolExecutor(max_workers=6) as executor:
        executor.map(lambda url: download_file(url, base_dir / url.split('/')[-1]), urls)

def download_xsm_data():
    base_dir = Path(__file__).resolve().parents[1] / "data" / "spi" / "data" / "Soilmoisture"
    base_dir.mkdir(parents=True, exist_ok=True)
    urls = [f"https://ftp.cpc.ncep.noaa.gov/fews/DroughtMonitor/SoilMoisture/{f}" for f in ["drymask12.bin","drymask12.ctl","drymask1.bin","drymask1.ctl","drymask24.bin","drymask24.ctl","drymask3.bin","drymask3.ctl","drymask6.bin","drymask6.ctl","globalmask0.5.dat","landmask.ctl","soilmoisture.12.mo.bin","soilmoisture.12.mo.ctl","soilmoisture.1.mo.bin","soilmoisture.1.mo.ctl","soilmoisture.24.mo.bin","soilmoisture.24.mo.ctl","soilmoisture.3.mo.bin","soilmoisture.3.mo.ctl","soilmoisture.6.mo.bin","soilmoisture.6.mo.ctl"]]
    
    print(f"🚀 Mac Parallel Engine: Downloading Soil Moisture fields...")
    with ThreadPoolExecutor(max_workers=6) as executor:
        executor.map(lambda url: download_file(url, base_dir / url.split('/')[-1]), urls)