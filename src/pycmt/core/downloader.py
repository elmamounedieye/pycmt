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
# MACOS NETWORK OPTIMIZATION HACKS (FORCES IPV4 & PERSISTENT KEEPALIVE)
# =========================================================================

def allowed_gai_family():
    """Forces IPv4 resolution to prevent macOS hidden timeout bugs on NOAA servers."""
    return socket.AF_INET

import urllib3.util.connection as urllib3_cn
urllib3_cn.allowed_gai_family = allowed_gai_family

# Global persistent session setup
http_session = requests.Session()
http_session.headers.update({
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
})

class DESAdapter(HTTPAdapter):
    """SSL Adapter designed to downgrade connection security protocols for older servers (e.g. ACMAD)."""
    def init_poolmanager(self, *args, **kwargs):
        context = urllib3_ssl.create_urllib3_context()
        context.load_default_certs()
        context.set_ciphers('DEFAULT@SECLEVEL=1')
        kwargs['ssl_context'] = context
        return super(DESAdapter, self).init_poolmanager(*args, **kwargs)

http_session.mount('https://sgbd.acmad.org', DESAdapter())

# =========================================================================
# CORE DOWNLOADING FUNCTIONS
# =========================================================================

def download_file(url, filename):
    """Generic file downloader optimized with a 64KB stream buffer chunk."""
    try:
        response = http_session.get(url, stream=True, timeout=30)
        if response.status_code == 200:
            with open(filename, "wb") as f:
                for chunk in response.iter_content(chunk_size=65536):
                    f.write(chunk)
            print(f"Succeeded : {Path(filename).name} downloading.")
            return True
        else:
            print(f"URL HTTP Error : {response.status_code} for {url}")
            return False
    except Exception as e:
        print(f"URL failure : {e}")
        return False

def manage_download(url, dest_path, is_gzip=False, min_size_kb=2000):
    """Validates local file completeness before firing standard stream downloading pipelines."""
    if dest_path.exists() and (dest_path.stat().st_size / 1024) > min_size_kb:
        return True 

    print(f"⏳ Downloading : {dest_path.name}...")
    try:
        response = http_session.get(url, stream=True, timeout=15)
        response.raise_for_status()
        
        if is_gzip:
            with gzip.GzipFile(fileobj=response.raw) as gfile:
                with open(dest_path, 'wb') as f_out:
                    shutil.copyfileobj(gfile, f_out)
        else:
            with open(dest_path, 'wb') as f_out:
                for chunk in response.iter_content(chunk_size=65536):
                    f_out.write(chunk)
        print(f"   ✅ OK")
        return True
    except Exception as e:
        print(f"   ❌ Error on {dest_path.name}: {e}")
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
        print(f"Country file path: {full_save_path}")
        return os.path.splitext(full_save_path)[0]
    else:
        save_directory.mkdir(parents=True, exist_ok=True)
        print(f"Directory created: {save_directory}")
        url = f"https://geodata.ucdavis.edu/gadm/gadm4.1/{file_format}/gadm41_{iso_code}_{file_format}.zip"
        print(f"Downloading {iso_code} from {url}...")

        try:
            if download_file(url, full_save_path):
                print(f"Download successful")
                with zipfile.ZipFile(full_save_path, 'r') as zip_ref:
                    country_folder = save_directory / f"{iso_code}_adm"
                    os.makedirs(country_folder, exist_ok=True)
                    zip_ref.extractall(country_folder)
                    print(f"Files extracted into {country_folder}")
                rename_country_shapefiles(iso_code)
        except Exception as e:
            print(f"An Error occurred: {e}")

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
                    print(f"   ✅ Copied : {src.name} -> {dst.name}")
                except Exception as e:
                    print(f"   ❌ Error copying {src.name} : {e}")

def generate_ctl(template_name, output_path, replacements):
    template_path = Path(template_name)
    if not template_path.exists():
        print(f"⚠️ Template {template_name} not found.")
        return
    content = template_path.read_text()
    for key, value in replacements.items():
        if "DIR" in key:
            content = content.replace(key, "^")
        else:
            content = content.replace(key, str(value))
    content = content.replace('^^', '^').replace('?\\', '^').replace('?/', '^')
    output_path.write_text(content)
    print(f"📝 CTL generated : {output_path.name}")

# =========================================================================
# CLIMATE PRODUCT PIPELINES (ARC2, RFE2, VHI, SPP, SPI, HYDROLOGY)
# =========================================================================

def download_arc2_data(init_day_offset=1):
    base_dir = Path(__file__).resolve().parents[1]
    daily_dir = base_dir / "data" / "ARC2" / "arc2"
    clim_dir = base_dir / "data" / "ARC2" / "arc2_clim"
    
    daily_dir.mkdir(parents=True, exist_ok=True)
    clim_dir.mkdir(parents=True, exist_ok=True)

    today = datetime.now() - timedelta(days=init_day_offset)
    print(f"🔍 Searching for latest available ARC2 dataset starting from : {today.strftime('%Y-%m-%d')}...")
    
    while True:
        target_date = today - timedelta(days=1)
        date_str = target_date.strftime("%Y%m%d")
        daily_filename = f"daily_clim.bin.{date_str}"
        daily_path = daily_dir / daily_filename
        daily_url = f"https://ftp.cpc.ncep.noaa.gov/fews/fewsdata/africa/arc2/bin/{daily_filename}.gz"
        
        if manage_download(daily_url, daily_path, is_gzip=True):
            print(f"✨ Verified start date found : {target_date.strftime('%Y-%m-%d')}")
            break
        else:
            print(f"⚠️ Dataset for {target_date.strftime('%Y-%m-%d')} not online. Shifting back to previous day...")
            today -= timedelta(days=1)
            
    print(f"🚀 Downloading ARC2 (180 days batch payload)...")
    for i in range(1, 181):
        target_date = today - timedelta(days=i)
        date_str = target_date.strftime("%Y%m%d")
        date_short = target_date.strftime("%m%d")
        
        daily_filename = f"daily_clim.bin.{date_str}"
        daily_path = daily_dir / daily_filename
        daily_url = f"https://ftp.cpc.ncep.noaa.gov/fews/fewsdata/africa/arc2/bin/{daily_filename}.gz"
        manage_download(daily_url, daily_path, is_gzip=True)

        clim_filename = f"clim.bin.{date_short}"
        clim_path = clim_dir / clim_filename
        clim_url = f"https://ftp.cpc.ncep.noaa.gov/fews/AFR_CLIM/ARC2/CLIMATOLOGY_DATA/DAILY_MEANS/{clim_filename}"
        manage_download(clim_url, clim_path, is_gzip=False)

    start_date_str = (today - timedelta(days=180)).strftime("%d%b%Y")
    generate_ctl(base_dir / "data" / "template_arc2.ctl", daily_dir / "arc2.ctl", {"DLDATADIR": "^", "STDATE": start_date_str})
    generate_ctl(base_dir / "data" / "template_arc2_clim.ctl", clim_dir / "arc2_clim.ctl", {"CLMDLDATADIR": "^", "STDATE": start_date_str})

def download_rfe2(days_offset=1):
    base_dir = Path(pycmt.__file__).resolve().parent / "data"
    output_dir = base_dir / "rfe2_data" / "rfe2_daily"
    clim_dir = base_dir / "rfe2_data" / "rfe2_clim"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    clim_dir.mkdir(parents=True, exist_ok=True)
    
    NORMAL_SIZE = 2406204
    yesterday = datetime.now().date() - timedelta(days=(days_offset + 1))
    existing_files = list(output_dir.glob("all_products.bin.*"))
    
    if not existing_files:
        print("ℹ️ Missing local cache assets. Initializing full 180-day batch payload workflow.")
        n_missing = 180
    else:
        dates_in_disk = []
        for f in existing_files:
            try:
                date_part = f.name.split(".")[-1]
                dates_in_disk.append(datetime.strptime(date_part, "%Y%m%d").date())
            except ValueError:
                continue
        if dates_in_disk:
            last_disk_date = max(dates_in_disk)
            n_missing = (yesterday - last_disk_date).days
            print(f"📊 Last cached file found on disk : {last_disk_date}")
        else:
            n_missing = 180

    ctl_file = output_dir / "rfe2daily.ctl"
    if n_missing <= 0 and ctl_file.exists():
        print("✅ Cached RFE2 assets are up to date. Processing skipped.")
        return
        
    print(f"🔄 Lag sync window calculation : {n_missing} days overdue. Reloading standard 180-day series pipeline...")
    for i in range(1, 181):
        target_date = datetime.now() - timedelta(days=(days_offset + i))
        date_str = target_date.strftime("%Y%m%d")  
        mmdd = target_date.strftime("%m%d")        
        
        file_name = f"all_products.bin.{date_str}"
        gz_name = f"{file_name}.gz"
        file_url = f"https://ftp.cpc.ncep.noaa.gov/fews/fewsdata/africa/rfe2/bin/{gz_name}"
        file_path = output_dir / file_name

        if not (file_path.exists() and file_path.stat().st_size == NORMAL_SIZE):
            print(f" 📥 Downloading RFE2 daily precipitation data : {date_str}...")
            try:
                response = http_session.get(file_url, stream=True, timeout=20)
                if response.status_code == 200:
                    temp_gz = base_dir / gz_name
                    with open(temp_gz, 'wb') as f:
                        f.write(response.content)
                    with gzip.open(temp_gz, 'rb') as f_in:
                        with open(file_path, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    temp_gz.unlink()
                else:
                    print(f" ⚠️ File missing on NOAA server endpoint: {gz_name} (HTTP {response.status_code})")
            except Exception as e:
                print(f" ❌ Precipitation download error on date {date_str} : {e}")

        clim_file_name = f"clim_dly.{mmdd}"
        clim_file_path = clim_dir / clim_file_name
        clim_url = f"https://ftp.cpc.ncep.noaa.gov/fews/clim_dly/clim_RFE2/{clim_file_name}"

        if not (clim_file_path.exists() and clim_file_path.stat().st_size == NORMAL_SIZE):
            print(f" 📥 Downloading day-by-day climatology metadata : {mmdd}...")
            try:
                response = http_session.get(clim_url, timeout=15)
                if response.status_code == 200:
                    with open(clim_file_path, "wb") as f:
                        f.write(response.content)
                else:
                    print(f" ⚠️ Climatology missing on remote source for index {mmdd}")
            except Exception as e:
                print(f" ❌ Climatology extraction failure on index {mmdd} : {e}")

    post_files = list(output_dir.glob("all_products.bin.*"))
    valid_files_with_dates = []
    for f in post_files:
        try:
            d_str = f.name.split(".")[-1]
            if len(d_str) == 8:
                valid_files_with_dates.append((datetime.strptime(d_str, "%Y%m%d").date(), f))
        except ValueError:
            continue

    if len(valid_files_with_dates) > 180:
        valid_files_with_dates.sort(key=lambda x: x[0])
        to_delete = valid_files_with_dates[:len(valid_files_with_dates) - 180]
        print(f"\n🧹 Rolling historical purging triggered ({len(to_delete)} obsolete files to clear)...")
        for f_date, f_path in to_delete:
            try:
                f_path.unlink()
                print(f"   Purged : {f_path.name}")
            except Exception as e:
                print(f"   ⚠️ Erasure failure on {f_path.name} : {e}")

    start_date = datetime.now() - timedelta(days=(days_offset + 180))
    start_date_str = start_date.strftime("%d%b%Y")
    generate_ctl(base_dir / "template_rfe2.ctl", output_dir / "rfe2daily.ctl", {"DLDATADIR": "^", "NNDAYS": 180, "STDATE": start_date_str})
    generate_ctl(base_dir / "rfe2clim.ctl", clim_dir / "rfe2clim.ctl", {"NNDAYS": 180, "STDATE": start_date_str})
    print(f"\n✅ RFE2 execution sync pipeline successfully achieved (GrADS anchor : {start_date_str}).")

def run_retrieval_vhi():
    INIT_DAY = 0
    TARGET_DIR = Path(__file__).resolve().parents[1] / "data" / "vhi" / "data"
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    WEEKS_TO_FETCH = 6
    for f in os.listdir(TARGET_DIR):
        if f.endswith(".png"):
            os.remove(TARGET_DIR / f)

    for i in range(1, WEEKS_TO_FETCH + 1):
        nd = INIT_DAY + (i * 7)
        target_date = datetime.now() - timedelta(days=nd)
        year = target_date.year
        julian_day = int(target_date.strftime("%j"))
        week_num = min(((julian_day - 1) // 7) + 1, 52)

        wk_tag = f"{year}{week_num:03d}"
        target_filename = f"VHP.G04.C07.j01.P{wk_tag}.VH.nc"
        print(f"\n--- Week {i} (Tag: {wk_tag}) ---")
        save_path = TARGET_DIR / target_filename

        if save_path.exists():
            print(f"File {target_filename} already downloaded.")
            continue

        url = f"https://www.star.nesdis.noaa.gov/data/pub0018/VHPdata4users/data/Blended_VH_4km/VH/VHP.G04.C07.j01.P{wk_tag}.VH.nc"
        download_file(url, save_path)

def download_spp_file(url, target_folder):
    filename = url.split('/')[-1]
    save_path = os.path.join(target_folder, filename)
    os.makedirs(target_folder, exist_ok=True)
    try:
        response = http_session.get(url, timeout=100, stream=True)
        if response.status_code != 200 and url.startswith("https://sgbd.acmad.org"):
            http_url = url.replace("https://", "http://")
            response = http_session.get(http_url, timeout=20, stream=True)

        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=65536):
                    f.write(chunk)
            return f"SUCCEEDED: {filename}"
        else:
            return f"ERROR {response.status_code}: {filename}"
    except Exception as e:
        return f"FAILURE: {filename} (Error: {e})"

def generate_ctl_spp(template_name, output_path, replacements):
    template_path = Path(template_name)
    if not template_path.exists():
        print(f"⚠️ Template {template_name} not found.")
        return
    content = template_path.read_text()
    for key, value in replacements.items():
        if "DSET " in key:
            content = content.replace(key, "DSET ^")
        else:
            content = content.replace(key, str(value))
    output_path.write_text(content)
    print(f"📝 CTL generated : {output_path.name}")

def download_spp_noaa(rndta):
    base_urls = [
        "https://ftp.cpc.ncep.noaa.gov/International/spp/{rndta}/spp_{rndta}_comb_1ic-0proj.ctl",
        "https://ftp.cpc.ncep.noaa.gov/International/spp/{rndta}/spp_{rndta}_comb_1ic-0proj.dat",
        "https://ftp.cpc.ncep.noaa.gov/International/spp/{rndta}/spp_{rndta}_comb_1ic-1proj.ctl",
        "https://ftp.cpc.ncep.noaa.gov/International/spp/{rndta}/spp_{rndta}_comb_1ic-1proj.dat",
        "https://ftp.cpc.ncep.noaa.gov/International/spp/{rndta}/spp_{rndta}_comb_1ic-2proj.ctl",
        "https://ftp.cpc.ncep.noaa.gov/International/spp/{rndta}/spp_{rndta}_comb_1ic-2proj.dat",
        "https://ftp.cpc.ncep.noaa.gov/International/spp/{rndta}/spp_{rndta}_comb_2ic-0proj.ctl",
        "https://ftp.cpc.ncep.noaa.gov/International/spp/{rndta}/spp_{rndta}_comb_2ic-0proj.dat",
        "https://ftp.cpc.ncep.noaa.gov/International/spp/{rndta}/spp_{rndta}_comb_2ic-1proj.ctl",
        "https://ftp.cpc.ncep.noaa.gov/International/spp/{rndta}/spp_{rndta}_comb_2ic-1proj.dat",
        "https://ftp.cpc.ncep.noaa.gov/International/spp/{rndta}/spp_{rndta}_comb_3ic-0proj.ctl",
        "https://ftp.cpc.ncep.noaa.gov/International/spp/{rndta}/spp_{rndta}_comb_3ic-0proj.dat",
        "https://ftp.cpc.ncep.noaa.gov/International/spp/{rndta}/spp_timescales.txt"
    ]
    target_dir = Path(__file__).resolve().parents[1] / "data" / "spp" / f"spp_data_{rndta}"
    target_dir.mkdir(parents=True, exist_ok=True)

    for url_template in base_urls:
        url = url_template.format(rndta=rndta)
        filename = url.split('/')[-1]
        save_path = target_dir / filename
        if save_path.exists():
            save_path.unlink() 
        download_file(url, save_path)

    for ctl_file in target_dir.glob("*.ctl"):
        print(f"📄 Correcting CTL ({rndta.upper()}) : {ctl_file.name}")
        generate_ctl_spp(ctl_file, ctl_file, {"DSET ": "DSET ^"})

def download_spi(rndta: str):
    spi_data_urls = {
        "cmorph": [
            "https://ftp.cpc.ncep.noaa.gov/fews/DroughtMonitor/spi/cmorph/drymask12.bin",
            "https://ftp.cpc.ncep.noaa.gov/fews/DroughtMonitor/spi/cmorph/drymask12.ctl",
            "https://ftp.cpc.ncep.noaa.gov/fews/DroughtMonitor/spi/cmorph/drymask1.bin",
            "https://ftp.cpc.ncep.noaa.gov/fews/DroughtMonitor/spi/cmorph/drymask1.ctl",
            "https://ftp.cpc.ncep.noaa.gov/fews/DroughtMonitor/spi/cmorph/drymask24.bin",
            "https://ftp.cpc.ncep.noaa.gov/fews/DroughtMonitor/spi/cmorph/drymask24.ctl",
            "https://ftp.cpc.ncep.noaa.gov/fews/DroughtMonitor/spi/cmorph/drymask3.bin",
            "https://ftp.cpc.ncep.noaa.gov/fews/DroughtMonitor/spi/cmorph/drymask3.ctl",
            "https://ftp.cpc.ncep.noaa.gov/fews/DroughtMonitor/spi/cmorph/drymask6.bin",
            "https://ftp.cpc.ncep.noaa.gov/fews/DroughtMonitor/spi/cmorph/drymask6.ctl",
            "https://ftp.cpc.ncep.noaa.gov/fews/DroughtMonitor/spi/cmorph/globalmask0.25.dat",
            "https://ftp.cpc.ncep.noaa.gov/fews/DroughtMonitor/spi/cmorph/landmask.ctl",
            "https://ftp.cpc.ncep.noaa.gov/fews/DroughtMonitor/spi/cmorph/cmorph.spi.12.mo.bin",
            "https://ftp.cpc.ncep.noaa.gov/fews/DroughtMonitor/spi/cmorph/cmorph.spi.12.mo.ctl",
            "https://ftp.cpc.ncep.noaa.gov/fews/DroughtMonitor/spi/cmorph/cmorph.spi.1.mo.bin",
            "https://ftp.cpc.ncep.noaa.gov/fews/DroughtMonitor/spi/cmorph/cmorph.spi.1.mo.ctl",
            "https://ftp.cpc.ncep.noaa.gov/fews/DroughtMonitor/spi/cmorph/cmorph.spi.24.mo.bin",
            "https://ftp.cpc.ncep.noaa.gov/fews/DroughtMonitor/spi/cmorph/cmorph.spi.24.mo.ctl",
            "https://ftp.cpc.ncep.noaa.gov/fews/DroughtMonitor/spi/cmorph/cmorph.spi.3.mo.bin",
            "https://ftp.cpc.ncep.noaa.gov/fews/DroughtMonitor/spi/cmorph/cmorph.spi.3.mo.ctl",
            "https://ftp.cpc.ncep.noaa.gov/fews/DroughtMonitor/spi/cmorph/cmorph.spi.6.mo.bin",
            "https://ftp.cpc.ncep.noaa.gov/fews/DroughtMonitor/spi/cmorph/cmorph.spi.6.mo.ctl"
        ],
        "cpcuni": [
            "https://ftp.cpc.ncep.noaa.gov/fews/DroughtMonitor/spi/cpcu/drymask12.bin",
            "https://ftp.cpc.ncep.noaa.gov/fews/DroughtMonitor/spi/cpcu/drymask12.ctl",
            "https://ftp.cpc.ncep.noaa.gov/fews/DroughtMonitor/spi/cpcu/drymask1.bin",
            "https://ftp.cpc.ncep.noaa.gov/fews/DroughtMonitor/spi/cpcu/drymask1.ctl",
            "https://ftp.cpc.ncep.noaa.gov/fews/DroughtMonitor/spi/cpcu/drymask24.bin",
            "https://ftp.cpc.ncep.noaa.gov/fews/DroughtMonitor/spi/cpcu/drymask24.ctl",
            "https://ftp.cpc.ncep.noaa.gov/fews/DroughtMonitor/spi/cpcu/drymask3.bin",
            "https://ftp.cpc.ncep.noaa.gov/fews/DroughtMonitor/spi/cpcu/drymask3.ctl",
            "https://ftp.cpc.ncep.noaa.gov/fews/DroughtMonitor/spi/cpcu/drymask6.bin",
            "https://ftp.cpc.ncep.noaa.gov/fews/DroughtMonitor/spi/cpcu/drymask6.ctl",
            "https://ftp.cpc.ncep.noaa.gov/fews/DroughtMonitor/spi/cpcu/globalmask0.5.dat",
            "https://ftp.cpc.ncep.noaa.gov/fews/DroughtMonitor/spi/cpcu/landmask.ctl",
            "https://ftp.cpc.ncep.noaa.gov/fews/DroughtMonitor/spi/cpcu/cpcu.spi.12.mo.bin",
            "https://ftp.cpc.ncep.noaa.gov/fews/DroughtMonitor/spi/cpcu/cpcu.spi.12.mo.ctl",
            "https://ftp.cpc.ncep.noaa.gov/fews/DroughtMonitor/spi/cpcu/cpcu.spi.1.mo.bin",
            "https://ftp.cpc.ncep.noaa.gov/fews/DroughtMonitor/spi/cpcu/cpcu.spi.1.mo.ctl",
            "https://ftp.cpc.ncep.noaa.gov/fews/DroughtMonitor/spi/cpcu/cpcu.spi.24.mo.bin",
            "https://ftp.cpc.ncep.noaa.gov/fews/DroughtMonitor/spi/cpcu/cpcu.spi.24.mo.ctl",
            "https://ftp.cpc.ncep.noaa.gov/fews/DroughtMonitor/spi/cpcu/cpcu.spi.3.mo.bin",
            "https://ftp.cpc.ncep.noaa.gov/fews/DroughtMonitor/spi/cpcu/cpcu.spi.3.mo.ctl",
            "https://ftp.cpc.ncep.noaa.gov/fews/DroughtMonitor/spi/cpcu/cpcu.spi.6.mo.bin",
            "https://ftp.cpc.ncep.noaa.gov/fews/DroughtMonitor/spi/cpcu/cpcu.spi.6.mo.ctl"
        ],
        "rfe2": [
            "https://ftp.cpc.ncep.noaa.gov/fews/DroughtMonitor/spi/rfe2/drymask12.bin",
            "https://ftp.cpc.ncep.noaa.gov/fews/DroughtMonitor/spi/rfe2/drymask12.ctl",
            "https://ftp.cpc.ncep.noaa.gov/fews/DroughtMonitor/spi/rfe2/drymask1.bin",
            "https://ftp.cpc.ncep.noaa.gov/fews/DroughtMonitor/spi/rfe2/drymask1.ctl",
            "https://ftp.cpc.ncep.noaa.gov/fews/DroughtMonitor/spi/rfe2/drymask24.bin",
            "https://ftp.cpc.ncep.noaa.gov/fews/DroughtMonitor/spi/rfe2/drymask24.ctl",
            "https://ftp.cpc.ncep.noaa.gov/fews/DroughtMonitor/spi/rfe2/drymask3.bin",
            "https://ftp.cpc.ncep.noaa.gov/fews/DroughtMonitor/spi/rfe2/drymask3.ctl",
            "https://ftp.cpc.ncep.noaa.gov/fews/DroughtMonitor/spi/rfe2/drymask6.bin",
            "https://ftp.cpc.ncep.noaa.gov/fews/DroughtMonitor/spi/rfe2/drymask6.ctl",
            "https://ftp.cpc.ncep.noaa.gov/fews/DroughtMonitor/spi/rfe2/mask.ctl",
            "https://ftp.cpc.ncep.noaa.gov/fews/DroughtMonitor/spi/rfe2/mask.gra",
            "https://ftp.cpc.ncep.noaa.gov/fews/DroughtMonitor/spi/rfe2/rfe2.spi.12.mo.bin",
            "https://ftp.cpc.ncep.noaa.gov/fews/DroughtMonitor/spi/rfe2/rfe2.spi.12.mo.ctl",
            "https://ftp.cpc.ncep.noaa.gov/fews/DroughtMonitor/spi/rfe2/rfe2.spi.1.mo.bin",
            "https://ftp.cpc.ncep.noaa.gov/fews/DroughtMonitor/spi/rfe2/rfe2.spi.1.mo.ctl",
            "https://ftp.cpc.ncep.noaa.gov/fews/DroughtMonitor/spi/rfe2/rfe2.spi.24.mo.bin",
            "https://ftp.cpc.ncep.noaa.gov/fews/DroughtMonitor/spi/rfe2/rfe2.spi.24.mo.ctl",
            "https://ftp.cpc.ncep.noaa.gov/fews/DroughtMonitor/spi/rfe2/rfe2.spi.3.mo.bin",
            "https://ftp.cpc.ncep.noaa.gov/fews/DroughtMonitor/spi/rfe2/rfe2.spi.3.mo.ctl",
            "https://ftp.cpc.ncep.noaa.gov/fews/DroughtMonitor/spi/rfe2/rfe2.spi.6.mo.bin",
            "https://ftp.cpc.ncep.noaa.gov/fews/DroughtMonitor/spi/rfe2/rfe2.spi.6.mo.ctl"
        ]
    }
    base_dir = Path(__file__).resolve().parents[1] / "data" / "spi" / "data" / f"{rndta}"
    base_dir.mkdir(parents=True, exist_ok=True)

    print(f"🧹 Cleaning repository in '{rndta}'...")
    for filename in os.listdir(base_dir):
        if filename.lower().endswith(('.ctl', '.bin', '.dat')):
            try:
                os.remove(base_dir / filename)
            except Exception as e:
                print(f"   Error deleting {filename}: {e}")

    for url in spi_data_urls[rndta]:
        filename = url.split('/')[-1]
        download_file(url, base_dir / filename)

def download_runoff_data():
    base_dir = Path(__file__).resolve().parents[1] / "data" / "spi" / "data" / "Runoff"
    base_dir.mkdir(parents=True, exist_ok=True)
    print(f"🚀 Initializing download sequence to destination : {base_dir.resolve()}")
    urls_runoff = [
        "https://ftp.cpc.ncep.noaa.gov/fews/DroughtMonitor/Runoff/drymask12.bin",
        "https://ftp.cpc.ncep.noaa.gov/fews/DroughtMonitor/Runoff/drymask12.ctl",
        "https://ftp.cpc.ncep.noaa.gov/fews/DroughtMonitor/Runoff/drymask1.bin",
        "https://ftp.cpc.ncep.noaa.gov/fews/DroughtMonitor/Runoff/drymask1.ctl",
        "https://ftp.cpc.ncep.noaa.gov/fews/DroughtMonitor/Runoff/drymask24.bin",
        "https://ftp.cpc.ncep.noaa.gov/fews/DroughtMonitor/Runoff/drymask24.ctl",
        "https://ftp.cpc.ncep.noaa.gov/fews/DroughtMonitor/Runoff/drymask3.bin",
        "https://ftp.cpc.ncep.noaa.gov/fews/DroughtMonitor/Runoff/drymask3.ctl",
        "https://ftp.cpc.ncep.noaa.gov/fews/DroughtMonitor/Runoff/drymask6.bin",
        "https://ftp.cpc.ncep.noaa.gov/fews/DroughtMonitor/Runoff/drymask6.ctl",
        "https://ftp.cpc.ncep.noaa.gov/fews/DroughtMonitor/Runoff/globalmask0.5.dat",
        "https://ftp.cpc.ncep.noaa.gov/fews/DroughtMonitor/Runoff/landmask.ctl",
        "https://ftp.cpc.ncep.noaa.gov/fews/DroughtMonitor/Runoff/runoff.12.mo.bin",
        "https://ftp.cpc.ncep.noaa.gov/fews/DroughtMonitor/Runoff/runoff.12.mo.ctl",
        "https://ftp.cpc.ncep.noaa.gov/fews/DroughtMonitor/Runoff/runoff.1.mo.bin",
        "https://ftp.cpc.ncep.noaa.gov/fews/DroughtMonitor/Runoff/runoff.1.mo.ctl",
        "https://ftp.cpc.ncep.noaa.gov/fews/DroughtMonitor/Runoff/runoff.24.mo.bin",
        "https://ftp.cpc.ncep.noaa.gov/fews/DroughtMonitor/Runoff/runoff.24.mo.ctl",
        "https://ftp.cpc.ncep.noaa.gov/fews/DroughtMonitor/Runoff/runoff.3.mo.bin",
        "https://ftp.cpc.ncep.noaa.gov/fews/DroughtMonitor/Runoff/runoff.3.mo.ctl",
        "https://ftp.cpc.ncep.noaa.gov/fews/DroughtMonitor/Runoff/runoff.6.mo.bin",
        "https://ftp.cpc.ncep.noaa.gov/fews/DroughtMonitor/Runoff/runoff.6.mo.ctl"
    ]
    for url in urls_runoff:
        filename = url.split("/")[-1]
        download_file(url, base_dir / filename)

def download_xsm_data():
    base_dir = Path(__file__).resolve().parents[1] / "data" / "spi" / "data" / "Soilmoisture"
    base_dir.mkdir(parents=True, exist_ok=True)
    urls_xsm = [
        "https://ftp.cpc.ncep.noaa.gov/fews/DroughtMonitor/SoilMoisture/drymask12.bin",
        "https://ftp.cpc.ncep.noaa.gov/fews/DroughtMonitor/SoilMoisture/drymask12.ctl",
        "https://ftp.cpc.ncep.noaa.gov/fews/DroughtMonitor/SoilMoisture/drymask1.bin",
        "https://ftp.cpc.ncep.noaa.gov/fews/DroughtMonitor/SoilMoisture/drymask1.ctl",
        "https://ftp.cpc.ncep.noaa.gov/fews/DroughtMonitor/SoilMoisture/drymask24.bin",
        "https://ftp.cpc.ncep.noaa.gov/fews/DroughtMonitor/SoilMoisture/drymask24.ctl",
        "https://ftp.cpc.ncep.noaa.gov/fews/DroughtMonitor/SoilMoisture/drymask3.bin",
        "https://ftp.cpc.ncep.noaa.gov/fews/DroughtMonitor/SoilMoisture/drymask3.ctl",
        "https://ftp.cpc.ncep.noaa.gov/fews/DroughtMonitor/SoilMoisture/drymask6.bin",
        "https://ftp.cpc.ncep.noaa.gov/fews/DroughtMonitor/SoilMoisture/drymask6.ctl",
        "https://ftp.cpc.ncep.noaa.gov/fews/DroughtMonitor/SoilMoisture/globalmask0.5.dat",
        "https://ftp.cpc.ncep.noaa.gov/fews/DroughtMonitor/SoilMoisture/landmask.ctl",
        "https://ftp.cpc.ncep.noaa.gov/fews/DroughtMonitor/SoilMoisture/soilmoisture.12.mo.bin",
        "https://ftp.cpc.ncep.noaa.gov/fews/DroughtMonitor/SoilMoisture/soilmoisture.12.mo.ctl",
        "https://ftp.cpc.ncep.noaa.gov/fews/DroughtMonitor/SoilMoisture/soilmoisture.1.mo.bin",
        "https://ftp.cpc.ncep.noaa.gov/fews/DroughtMonitor/SoilMoisture/soilmoisture.1.mo.ctl",
        "https://ftp.cpc.ncep.noaa.gov/fews/DroughtMonitor/SoilMoisture/soilmoisture.24.mo.bin",
        "https://ftp.cpc.ncep.noaa.gov/fews/DroughtMonitor/SoilMoisture/soilmoisture.24.mo.ctl",
        "https://ftp.cpc.ncep.noaa.gov/fews/DroughtMonitor/SoilMoisture/soilmoisture.3.mo.bin",
        "https://ftp.cpc.ncep.noaa.gov/fews/DroughtMonitor/SoilMoisture/soilmoisture.3.mo.ctl",
        "https://ftp.cpc.ncep.noaa.gov/fews/DroughtMonitor/SoilMoisture/soilmoisture.6.mo.bin",
        "https://ftp.cpc.ncep.noaa.gov/fews/DroughtMonitor/SoilMoisture/soilmoisture.6.mo.ctl"
    ]
    for url in urls_xsm:
        filename = url.split("/")[-1]
        download_file(url, base_dir / filename)