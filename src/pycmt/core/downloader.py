import requests
import os
import json
from pathlib import Path
import shutil
import zipfile
import numpy as np
import geopandas as gpd
import xarray as xr
from rasterio.features import rasterize
from rasterio.transform import from_origin
import rioxarray 
from datetime import datetime, timedelta
import gzip



#Country information
def get_country_iso(country: str,):
    #crr = os.getcwd()
    #print(f"crr : {crr}")
    BASE_DIR = Path(__file__).resolve().parents[1]
    #dataDir = BASE_DIR / "data"
    data_file = BASE_DIR / "data" / "countries_iso_dict.json"
    #currrr = os.getcwd()
    #print(f"currrr :{currrr}")

    try:
        with open(data_file, 'r', encoding='utf-8') as f:
            countries = json.load(f)
        
        country_iso = countries[country]
        return country_iso
    
    except FileNotFoundError:
        return f"❌ Error : Data file {data_file} not found."
    except KeyError:
        return f"❌ Error : '{country}' is not in the dictionary."

#Downloading country shapefile
def download_gadm_country(iso_code, file_format="shp"):
    
    iso_code = iso_code.upper()
    filename = f"{iso_code}_adm.zip"
    save_directory = Path(__file__).resolve().parents[1] / "data" / "gis_resources" / "countries"
    full_save_path = os.path.join(save_directory, filename)
    # 1. VERIFICATION : Si le dossier existe, on s'arrête ici
    if os.path.exists(full_save_path):
        print(f"Le dossier existe déjà : {full_save_path}")
        print(f"country file path: {full_save_path}")
        return os.path.splitext(full_save_path)[0]

    else:
        save_directory.mkdir(parents=True, exist_ok=True)
        print(f"Directory created: {save_directory}")

        url = f"https://geodata.ucdavis.edu/gadm/gadm4.1/{file_format}/gadm41_{iso_code}_{file_format}.zip"
        print(f"Downloading {iso_code} from {url}...")

        try:
            response = requests.get(url, stream=True)
            response.raise_for_status 

            with open(full_save_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            print(f"Download successful")

            with zipfile.ZipFile(full_save_path, 'r') as zip_ref:
                country_folder = os.path.splitext(full_save_path)[0]
                # 2. Créer le dossier s'il n'existe pas
                os.makedirs(country_folder, exist_ok=True)
                zip_ref.extractall(country_folder)
                print(f"Files extracted into {country_folder}")
            
            rename_country_shapefiles(iso_code)
            #return country_folder
        except requests.exceptions.HTTPError:
            print(f"Error : Could not find the data for '{iso_code}'. Check the ISO code")
        except Exception as e:
            print(f"An Error occured: {e}")


def rename_country_shapefiles(country_iso):

    #folder = Path(shpfolder)
    folder = Path(__file__).resolve().parents[1] / "data" / "gis_resources" / "countries" / f"{country_iso}_adm"
    expected_final_file = folder / f"{country_iso}_adm0.shp"
    if os.path.exists(expected_final_file):
        #print(f"##################################################")
        #print(f"Files are already renamed in {shpfolder}")
        #print(f"##################################################")
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



def download_arc2_data(init_day_offset=1):
    # 1. Configuration des chemins
    base_dir = Path(__file__).resolve().parents[1]
    daily_dir = Path(__file__).resolve().parents[1] / "data" / "ARC2" / "arc2"
    clim_dir = Path(__file__).resolve().parents[1] / "data" / "ARC2" / "arc2_clim"
    
    daily_dir.mkdir(parents=True, exist_ok=True)
    clim_dir.mkdir(parents=True, exist_ok=True)

    # 2. Détermination de la date de départ réelle (Validation du premier fichier disponible)
    today = datetime.now() - timedelta(days=init_day_offset)
    print(f"🔍 Recherche de la dernière donnée ARC2 disponible à partir de : {today.strftime('%Y-%m-%d')}...")
    
    while True:
        # On teste le fichier le plus récent requis (qui correspond à i=1 dans ton ancienne logique)
        target_date = today - timedelta(days=1)
        date_str = target_date.strftime("%Y%m%d")
        daily_filename = f"daily_clim.bin.{date_str}"
        daily_path = daily_dir / daily_filename
        daily_url = f"https://ftp.cpc.ncep.noaa.gov/fews/fewsdata/africa/arc2/bin/{daily_filename}.gz"
        
        # Tentative de téléchargement du premier fichier
        if manage_download(daily_url, daily_path, is_gzip=True):
            print(f"✨ Date de départ validée : {target_date.strftime('%Y-%m-%d')}")
            break # On a trouvé notre date de départ, on sort de la recherche
        else:
            print(f"⚠️ Donnée du {target_date.strftime('%Y-%m-%d')} indisponible. Tentative sur le jour précédent...")
            today -= timedelta(days=1) # On décale "today" d'un jour et on recommence
            
    # 3. Téléchargement des N (180) données à partir de la nouvelle date validée
    print(f"🚀 Downloading ARC2 (180 days) starting from verified date...")
    
    # Note : Le premier fichier (i=1) a déjà été téléchargé par la boucle while ci-dessus,
    # mais manage_download le détectera et passera directement sans le retélécharger.
    for i in range(1, 181):
        target_date = today - timedelta(days=i)
        date_str = target_date.strftime("%Y%m%d")      # AAAAMMDD
        date_short = target_date.strftime("%m%d")      # MMDD
        
        # --- PHASE A : DONNÉES QUOTIDIENNES ---
        daily_filename = f"daily_clim.bin.{date_str}"
        daily_path = daily_dir / daily_filename
        daily_url = f"https://ftp.cpc.ncep.noaa.gov/fews/fewsdata/africa/arc2/bin/{daily_filename}.gz"
        
        manage_download(daily_url, daily_path, is_gzip=True)

        # --- PHASE B : DONNÉES CLIMATOLOGIQUES ---
        clim_filename = f"clim.bin.{date_short}"
        clim_path = clim_dir / clim_filename
        clim_url = f"https://ftp.cpc.ncep.noaa.gov/fews/AFR_CLIM/ARC2/CLIMATOLOGY_DATA/DAILY_MEANS/{clim_filename}"
        
        manage_download(clim_url, clim_path, is_gzip=False)

    # 4. GÉNÉRATION DES FICHIERS .CTL (Ajusté sur la vraie date de fin de la série)
    start_date_str = (today - timedelta(days=180)).strftime("%d%b%Y")
    
    generate_ctl(
        template_name=base_dir / "data" / "template_arc2.ctl",
        output_path=daily_dir / "arc2.ctl",
        replacements={"DLDATADIR": "^", "STDATE": start_date_str}
    )
    
    generate_ctl(
        template_name=base_dir / "data" / "template_arc2_clim.ctl",
        output_path=clim_dir / "arc2_clim.ctl",
        replacements={"CLMDLDATADIR": "^", "STDATE": start_date_str}
    )

def manage_download(url, dest_path, is_gzip=False, min_size_kb=2000):
    """Gère la vérification, le téléchargement et la décompression.
       Retourne True si le fichier est dispo/téléchargé, False sinon."""
    
    # Vérification si le fichier existe et est valide localement
    if dest_path.exists() and (dest_path.stat().st_size / 1024) > min_size_kb:
        return True 

    print(f"⏳ Downloading : {dest_path.name}...")
    
    try:
        response = requests.get(url, stream=True, timeout=15) # Timeout réduit pour ne pas bloquer trop longtemps si indisponible
        response.raise_for_status()
        
        if is_gzip:
            with gzip.GzipFile(fileobj=response.raw) as gfile:
                with open(dest_path, 'wb') as f_out:
                    shutil.copyfileobj(gfile, f_out)
        else:
            with open(dest_path, 'wb') as f_out:
                for chunk in response.iter_content(chunk_size=8192):
                    f_out.write(chunk)
        print(f"   ✅ OK")
        return True
        
    except Exception as e:
        print(f"   ❌ Error on {dest_path.name}: {e}")
        # Si le fichier incomplet/corrompu a été créé à cause de l'erreur, on le supprime
        if dest_path.exists():
            dest_path.unlink()
        return False

def generate_ctl(template_name, output_path, replacements):
    # (Ton code reste identique pour cette fonction)
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
    
    content = content.replace('^^', '^')
    content = content.replace('^\\', '^')
    content = content.replace('^/', '^')
    
    output_path.write_text(content)
    print(f"📝 CTL généré : {output_path.name}")


########## RFE2 
import gzip
import shutil
from datetime import datetime, timedelta
from pathlib import Path
import requests

# Point d'ancrage absolu du package installé
import pycmt

def download_rfe2(days_offset=1):
    
    # 1. Configuration des chemins absolus au sein du package
    base_dir = Path(pycmt.__file__).resolve().parent / "data"
    output_dir = base_dir / "rfe2_data" / "rfe2_daily"
    clim_dir = base_dir / "rfe2_data" / "rfe2_clim"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    clim_dir.mkdir(parents=True, exist_ok=True)
    
    NORMAL_SIZE = 2406204  # Taille d'un fichier binaire RFE2 valide (décompressé)
    yesterday = datetime.now().date() - timedelta(days=(days_offset + 1))
    
    # 2. Analyse des fichiers déjà présents sur le disque
    existing_files = list(output_dir.glob("all_products.bin.*"))
    
    if not existing_files:
        print("ℹ️ Aucun fichier trouvé sur le disque. Téléchargement initial (180 jours).")
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
            print(f"📊 Dernier fichier détecté sur le disque : {last_disk_date}")
        else:
            n_missing = 180

    # Si les données sont à jour et que le fichier de contrôle existe, on s'arrête
    ctl_file = output_dir / "rfe2daily.ctl"
    if n_missing <= 0 and ctl_file.exists():
        print("✅ Données RFE2 déjà à jour. Aucun téléchargement requis.")
        return
        
    print(f"🔄 Retard détecté : {n_missing} jours. Lancement de la synchronisation sur 180 jours...")

    # 3. Boucle de téléchargement (Précipitations + Climatologie)
    for i in range(1, 181):
        target_date = datetime.now() - timedelta(days=(days_offset + i))
        date_str = target_date.strftime("%Y%m%d")  
        mmdd = target_date.strftime("%m%d")        
        
        # --- BLOC PRÉCIPITATIONS DAILY ---
        file_name = f"all_products.bin.{date_str}"
        gz_name = f"{file_name}.gz"
        
        # URL CORRIGÉE (Suppression de /fewsdata/ qui causait la 404)
        file_url = f"https://ftp.cpc.ncep.noaa.gov/fews/fewsdata/africa/rfe2/bin/{gz_name}"
        file_path = output_dir / file_name

        if not (file_path.exists() and file_path.stat().st_size == NORMAL_SIZE):
            print(f" 📥 Téléchargement pluie RFE2 : {date_str}...")
            try:
                response = requests.get(file_url, stream=True, timeout=20)
                if response.status_code == 200:
                    temp_gz = base_dir / gz_name
                    with open(temp_gz, 'wb') as f:
                        f.write(response.content)
                    
                    # Décompression native du fichier .gz
                    with gzip.open(temp_gz, 'rb') as f_in:
                        with open(file_path, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    
                    temp_gz.unlink()  # Nettoyage du fichier compressé temporaire
                else:
                    print(f" ⚠️ Non disponible sur le serveur NOAA : {gz_name} (Code HTTP {response.status_code})")
            except Exception as e:
                print(f" ❌ Erreur téléchargement pluie {date_str} : {e}")

        # --- BLOC CLIMATOLOGIE DLY ---
        clim_file_name = f"clim_dly.{mmdd}"
        clim_file_path = clim_dir / clim_file_name
        
        # URL STABLE POUR LA CLIMATOLOGIE RFE2 AFRIQUE
        clim_url = f"https://ftp.cpc.ncep.noaa.gov/fews/clim_dly/clim_RFE2/{clim_file_name}"

        if not (clim_file_path.exists() and clim_file_path.stat().st_size == NORMAL_SIZE):
            print(f" 📥 Téléchargement climatologie jour par jour : {mmdd}...")
            try:
                response = requests.get(clim_url, timeout=15)
                if response.status_code == 200:
                    with open(clim_file_path, "wb") as f:
                        f.write(response.content)
                else:
                    print(f" ⚠️ Climatologie non disponible pour le {mmdd}")
            except Exception as e:
                print(f" ❌ Erreur climatologie {mmdd} : {e}")

    # 4. Nettoyage de l'historique (Sécurité pour ne conserver STRICTEMENT que 180 jours)
    post_files = list(output_dir.glob("all_products.bin.*"))
    valid_files_with_dates = []

    for f in post_files:
        try:
            d_str = f.name.split(".")[-1]
            if len(d_str) == 8:  # ignore les fichiers .ctl ou autres
                valid_files_with_dates.append((datetime.strptime(d_str, "%Y%m%d").date(), f))
        except ValueError:
            continue

    if len(valid_files_with_dates) > 180:
        valid_files_with_dates.sort(key=lambda x: x[0])
        to_delete = valid_files_with_dates[:len(valid_files_with_dates) - 180]
        
        print(f"\n🧹 Nettoyage de l'ancien historique ({len(to_delete)} jours obsolètes)...")
        for f_date, f_path in to_delete:
            try:
                f_path.unlink()
                print(f"   Supprimé : {f_path.name}")
            except Exception as e:
                print(f"   ⚠️ Impossible de supprimer {f_path.name} : {e}")

    # 5. Régénération dynamique des fichiers d'indexation .ctl
    start_date = datetime.now() - timedelta(days=(days_offset + 180))
    start_date_str = start_date.strftime("%d%b%Y")

    # Import local de la fonction de génération CTL (située dans le même module)
    from pycmt.core.downloader import generate_ctl

    generate_ctl(
        template_name=base_dir / "template_rfe2.ctl",
        output_path=output_dir / "rfe2daily.ctl",
        replacements={"DLDATADIR": "^", "NNDAYS": 180, "STDATE": start_date_str}
    )

    generate_ctl(
        template_name=base_dir / "rfe2clim.ctl",
        output_path=clim_dir / "rfe2clim.ctl",
        replacements={"NNDAYS": 180, "STDATE": start_date_str}
    )

    print(f"\n✅ Synchronisation RFE2 achevée avec succès (Début GrADS : {start_date_str}).")








# --- CONFIGURATION --- VHI

# Liste des URL modèles (le tag {wk} sera remplacé par l'année + semaine)



def download_file(url, filename):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    }
    try:
        response = requests.get(url, headers=headers, stream=True, timeout=30)
        if response.status_code == 200:
            with open(filename, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Succeeded : {filename} downloading.")
            return True
        else:
            print(f"URL HTTP Error : {response.status_code} for {url}")
            return False
    except Exception as e:
        print(f"URL failure : {e}")
        return False


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

        if os.path.exists(save_path):
            print(f"File {target_filename} already downloaded.")
            continue

        URL_SOURCES = [
            "https://www.star.nesdis.noaa.gov/data/pub0018/VHPdata4users/data/Blended_VH_4km/VH/VHP.G04.C07.j01.P{wk}.VH.nc"
        ]
        for url_template in URL_SOURCES:
            source_url = url_template.format(wk=wk_tag)
            if download_file(source_url, save_path):
                break

####### Download SPP #######
import os
import requests
import ssl
from urllib3.util.ssl_ import create_urllib3_context
from requests.adapters import HTTPAdapter
from concurrent.futures import ThreadPoolExecutor

# --- CONFIGURATION OPTIONNELLE ---
SOURCES_SELECTIONNEES = ["rfe2"] 

# Classe pour forcer une sécurité SSL plus faible (DH_KEY_TOO_SMALL bypass)
class DESAdapter(HTTPAdapter):
    def init_poolmanager(self, *args, **kwargs):
        context = create_urllib3_context()
        context.load_default_certs()
        # On baisse le niveau de sécurité pour accepter les clés DH anciennes
        context.set_ciphers('DEFAULT@SECLEVEL=1')
        kwargs['ssl_context'] = context
        return super(DESAdapter, self).init_poolmanager(*args, **kwargs)

def download_spp_file(url, target_folder):
    filename = url.split('/')[-1]
    save_path = os.path.join(target_folder, filename)
    os.makedirs(target_folder, exist_ok=True)
    
    # On utilise un session pour appliquer l'Adapter SSL
    session = requests.Session()
    session.mount('https://', DESAdapter())
    
    try:
        # Tentative en HTTPS (avec sécurité réduite)
        response = session.get(url, timeout=100, stream=True)
        
        # Si HTTPS échoue encore, on tente un repli automatique en HTTP
        if response.status_code != 200 and url.startswith("https://sgbd.acmad.org"):
            http_url = url.replace("https://", "http://")
            response = requests.get(http_url, timeout=20, stream=True)

        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=16384):
                    f.write(chunk)
            return f"SUCCEEDED: {filename}"
        else:
            return f"ERROR {response.status_code}: {filename}"
            
    except Exception as e:
        return f"FAILURE: {filename} (Erreur: {e})"
    

def generate_ctl_spp(template_name, output_path, replacements):
    template_path = Path(template_name)
    if not template_path.exists():
        print(f"⚠️ Template {template_name} not found.")
        return

    content = template_path.read_text()
    
    for key, value in replacements.items():
        if "DSET " in key:
            # On remplace la balise par le chapeau seul
            # Ainsi, ^DLDATADIR devient ^
            # Et si le template n'avait pas de ^, on l'ajoute ici
            content = content.replace(key, "DSET ^")
        else:
            content = content.replace(key, str(value))
    
    # --- NETTOYAGE DES DOUBLONS ---
    #content = content.replace('^^', '^')   # Au cas où le template avait déjà un ^
    #content = content.replace('^\\', '^')  # Supprime l'antislash parasite
    #content = content.replace('^/', '^')   # Supprime le slash parasite
    
    output_path.write_text(content)
    print(f"📝 CTL generated : {output_path.name}")


def run_spp_retrieval(rndta):
    base_urls = [
        "https://sgbd.acmad.org/thredds/fileServer/ACMAD/CDD/climatedataservice/SPP_data/spp_{rndta}/spp_{rndta}_comb_1ic-0proj.ctl",
        "https://sgbd.acmad.org/thredds/fileServer/ACMAD/CDD/climatedataservice/SPP_data/spp_{rndta}/spp_{rndta}_comb_1ic-0proj.dat",
        "https://sgbd.acmad.org/thredds/fileServer/ACMAD/CDD/climatedataservice/SPP_data/spp_{rndta}/spp_{rndta}_comb_1ic-1proj.ctl",
        "https://sgbd.acmad.org/thredds/fileServer/ACMAD/CDD/climatedataservice/SPP_data/spp_{rndta}/spp_{rndta}_comb_1ic-1proj.dat",
        "https://sgbd.acmad.org/thredds/fileServer/ACMAD/CDD/climatedataservice/SPP_data/spp_{rndta}/spp_{rndta}_comb_1ic-2proj.ctl",
        "https://sgbd.acmad.org/thredds/fileServer/ACMAD/CDD/climatedataservice/SPP_data/spp_{rndta}/spp_{rndta}_comb_1ic-2proj.dat",
        "https://sgbd.acmad.org/thredds/fileServer/ACMAD/CDD/climatedataservice/SPP_data/spp_{rndta}/spp_{rndta}_comb_2ic-0proj.ctl",
        "https://sgbd.acmad.org/thredds/fileServer/ACMAD/CDD/climatedataservice/SPP_data/spp_{rndta}/spp_{rndta}_comb_2ic-0proj.dat",
        "https://sgbd.acmad.org/thredds/fileServer/ACMAD/CDD/climatedataservice/SPP_data/spp_{rndta}/spp_{rndta}_comb_2ic-1proj.ctl",
        "https://sgbd.acmad.org/thredds/fileServer/ACMAD/CDD/climatedataservice/SPP_data/spp_{rndta}/spp_{rndta}_comb_2ic-1proj.dat",
        "https://sgbd.acmad.org/thredds/fileServer/ACMAD/CDD/climatedataservice/SPP_data/spp_{rndta}/spp_{rndta}_comb_3ic-0proj.ctl",
        "https://sgbd.acmad.org/thredds/fileServer/ACMAD/CDD/climatedataservice/SPP_data/spp_{rndta}/spp_{rndta}_comb_3ic-0proj.dat",
        "https://sgbd.acmad.org/thredds/fileServer/ACMAD/CDD/climatedataservice/SPP_data/spp_{rndta}/spp_timescales.txt"
    ]

    base_urls_cpp = [
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
    base_dir = Path(__file__).resolve().parents[1] / "data" / "spp" /f"spp_data_{rndta}"
    base_dir.mkdir(parents=True, exist_ok=True)

    all_tasks = []
    #for rndta in SOURCES_SELECTIONNEES:
    #target_dir = base_dir / f"spp_data_{rndta}"
    for url_template in base_urls:
        all_tasks.append((url_template.format(rndta=rndta), base_dir))

    #print(f"Démarrage avec contournement SSL pour : {', '.join(SOURCES_SELECTIONNEES)}")

    with ThreadPoolExecutor(max_workers=3) as executor:
        results = list(executor.map(lambda p: download_spp_file(*p), all_tasks))

    """for res in results:
        print(res)"""


    # Boucler sur tous les fichiers finissant par .ctl
    for ctl_file in base_dir.glob("*.ctl"):
        #print(f"DOWNLOADING FILE : {ctl_file.name}")
        generate_ctl_spp(
        template_name=ctl_file,
        output_path=ctl_file,
        replacements={"DSET ": "DSET ^"}
    )

#if __name__ == "__main__":
#    run_spp_retrieval()

####### END download SPP #######
####### NEW SPP download #######


def download_spp_noaa(rndta):
    # 1. Définition des URLs avec le template intégré
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

    # 2. Configuration des dossiers
    target_dir = Path(__file__).resolve().parents[1] / "data" / "spp" / f"spp_data_{rndta}"
    target_dir.mkdir(parents=True, exist_ok=True)

    #print(f"🛰️  Mise à jour des données NOAA SPP pour : {rndta.upper()}")
    #print(f"📂 Dossier : {target_dir.resolve()}\n")

    # 3. Boucle de téléchargement
    for url_template in base_urls:
        url = url_template.format(rndta=rndta)
        filename = url.split('/')[-1]
        save_path = target_dir / filename

        # --- LOGIQUE DE MISE À JOUR : Suppression si le fichier existe déjà ---
        if save_path.exists():
            #print(f"  [UPDATE] Suppression de l'ancienne version : {filename}")
            save_path.unlink() 

        #print(f"  [TÉLÉCHARGEMENT] {filename}...", end="", flush=True)

        try:
            response = requests.get(url, stream=True, timeout=45)
            
            if response.status_code == 200:
                with open(save_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=65536):
                        f.write(chunk)
                print(" ✅ Done")
            else:
                print(f" ❌ Error {response.status_code} ")

        except Exception as e:
            print(f" 💥 Failure : {str(e)}")

    # 4. Traitement des fichiers CTL après téléchargement
    # (Note : correction de la faute de frappe "cmoprh" -> "cmorph")
    if rndta == "cmorph":
        for ctl_file in target_dir.glob("*.ctl"):
            print(f"📄 Correcting CTL (CMORPH) : {ctl_file.name}")
            generate_ctl_spp(
                template_name=ctl_file,
                output_path=ctl_file,
                replacements={"DSET ": "DSET ^"}
            )
    elif rndta == "rfe2":
        for ctl_file in target_dir.glob("*.ctl"):
            print(f"📄 Correcting CTL (RFE2) : {ctl_file.name}")
            generate_ctl_spp(
                template_name=ctl_file,
                output_path=ctl_file,
                replacements={"DSET ": "DSET ^"}
            )
##### END OF NEW SPP ######



####### Download SPI ###########

# Liste des URLs (extraites de vos commandes curl)





def download_spi(rndta: str):
    spi_data_urls ={
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

    #### SPI CPCU
    "cpcuni" :[
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
    
    "rfe2" :[
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
    base_dir = Path(__file__).resolve().parents[1] / "data" / "spi" /"data" /f"{rndta}"
    #daily_dir = base_dir / "data" / "spi" /"data"
    #clim_dir = base_dir / "data" / "ARC2" / "arc2_clim"
    
    # Création des dossiers s'ils n'existent pas
    base_dir.mkdir(parents=True, exist_ok=True)
    #clim_dir.mkdir(parents=True, exist_ok=True)

    # 1. Création du dossier s'il n'existe pas
    #if not os.path.exists(rndta):
    #    os.makedirs(rndta)
    #    print(f"📁 Dossier '{rndta}' créé.")
    #else:
    #    print(f"📂 Utilisation du dossier existant : '{rndta}'")

    ##### Clean repo
    ext = ('.ctl', '.bin', '.dat')
    print(f"🧹 Cleaning repository in '{rndta}'...")
    
    # On itère maintenant sur le contenu du dossier cible
    for filename in os.listdir(base_dir):
        if filename.lower().endswith(ext):
            file_path = os.path.join(base_dir, filename)
            try:
                os.remove(file_path)
                print(f"   Deleting : {filename}")
            except Exception as e:
                print(f"   Error deletion {filename}: {e}")

    #### Download phase
    for url in spi_data_urls[rndta]:
        filename = url.split('/')[-1]
        # On définit le chemin complet (dossier + nom de fichier)
        save_path = os.path.join(base_dir, filename)
        
        #print(f"⏳ Téléchargement de {filename} vers {base_dir}...")
        
        try:
            with requests.get(url, stream=True) as r:
                r.raise_for_status() 
                
                with open(save_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            
            #print(f"✅ Terminé : {filename}")
            
        except requests.exceptions.RequestException as e:
            print(f" Error downloading {filename} : {e}")

#if __name__ == "__main__":
#    download_files(urls_spi_cpcuni, "cpcuni")




def download_runoff_data():
    # 1. Configuration du dossier de destination
    # On place les données dans ../../data/Runoff
    base_dir = Path(__file__).resolve().parents[1] / "data" / "spi" / "data" / "Runoff"
    base_dir.mkdir(parents=True, exist_ok=True)

    print(f"🚀 Début du téléchargement vers : {base_dir.resolve()}")
    urls_runoff=[
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
        # Extraction du nom du fichier depuis l'URL
        filename = url.split("/")[-1]
        destination = base_dir / filename

        #print(f"⏳ Téléchargement de {filename}...", end="\r")

        try:
            # Requête de téléchargement
            response = requests.get(url, stream=True, timeout=20)
            
            # Vérification si l'URL est valide (Statut 200)
            if response.status_code == 200:
                with open(destination, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                print(f" Done! : {filename}          ")
            else:
                print(f" Error {response.status_code} pour : {filename}")

        except Exception as e:
            print(f"💥 Failure for {filename} : {e}")


###### Soil Moisture ######
def download_xsm_data():
    # 1. Configuration du dossier de destination
    # On place les données dans ../../data/Soilmoisture
    base_dir = Path(__file__).resolve().parents[1] / "data" / "spi" / "data" / "Soilmoisture"
    base_dir.mkdir(parents=True, exist_ok=True)

    #print(f"🚀 Début du téléchargement vers : {base_dir.resolve()}")
    urls_xsm= [
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
        # Extraction du nom du fichier depuis l'URL
        filename = url.split("/")[-1]
        destination = base_dir / filename

        #print(f"⏳ Téléchargement de {filename}...", end="\r")

        try:
            # Requête de téléchargement
            response = requests.get(url, stream=True, timeout=20)
            
            # Vérification si l'URL est valide (Statut 200)
            if response.status_code == 200:
                with open(destination, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                print(f"✅ Done : {filename}          ")
            else:
                print(f" Error {response.status_code} pour : {filename}")

        except Exception as e:
            print(f" Failure for {filename} : {e}")

# --- Ta liste d'URLs ---

