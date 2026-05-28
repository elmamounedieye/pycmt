from fastapi import FastAPI, BackgroundTasks, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from pathlib import Path
import shutil
import pycmt

# Réutilisation de tes fonctions de calcul pycmt
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

app = FastAPI(
    title="⚡ PyCMT Climate Monitor API",
    description="API de suivi, monitoring et traitement de données géospatiales et climatiques",
    version="1.0.0"
)

# Configuration de la correspondance de résolution
RESOLUTION_CONFIG = {
    0.036: '0p036', 0.0375: '0p0375', 0.25: '0p25',
    0.5: '0p5', 0.1: '0p1', 1.0: '1p0'
}

# =========================================================================
# PIPELINES DE TRAITEMENT (Fonctions exécutées en arrière-plan)
# =========================================================================

def pipeline_climonitor_complet(country: str, rndta: str, rsl: float, ts_rsl: float):
    """Orchestrateur complet de calcul s'exécutant en arrière-plan."""
    try:
        # 1. Gestion du cas spécifique Afrique ou pays classique
        if country == "Africa":
            country_iso = "AFR"
            package_root = Path(__file__).resolve().parent
            source = package_root / "data" / "AFR_adm"
            destination = package_root / "data" / "gis_resources" / "countries" / "AFR_adm"
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source, destination, dirs_exist_ok=True)
        else:
            country_iso = get_country_iso(country)
            download_gadm_country(country_iso)
        
        # 2. Génération des masques et coordonnées pixels
        run_workflow(country_iso, country)
        generate_pixel_arguments(ts_rsl, country_iso, country)
        plot_pix_coordinates(country, country_iso, rndta)
        
        # 3. Téléchargements sources de précipitations
        if rndta.lower() == "arc2":
            download_arc2_data()
        elif rndta.lower() == "rfe2":
            download_rfe2()

        # 4. Traitements géospatiaux et graphiques
        plot_precip(rsl, RESOLUTION_CONFIG[rsl], country_iso, country, rndta)
        generate_tseries(country_iso, country, rndta)
        generate_html_map(country, rndta)
        
        # 5. Déclenchement automatique du Dashboard final
        build_country_dashboard(country, rndta)
        print(f"🎉 Pipeline API réussi pour {country} ({rndta})")
        
    except Exception as e:
        print(f"❌ Échec de la pipeline en arrière-plan : {str(e)}")

@app.post("/upload-data/", tags=["Données"])
async def upload_custom_file(subfolder: str = "data", file: UploadFile = File(...)):
    """
    Remplace l'uploader ipywidgets. Permet d'envoyer un fichier (.ctl, .nc, etc.) 
    directement dans les répertoires internes du package.
    """
    dest_dir = Path(pycmt.__file__).resolve().parent / subfolder
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / file.filename

    try:
        with open(dest_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        return {"status": "success", "filename": file.filename, "saved_at": str(dest_path.resolve())}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la sauvegarde : {str(e)}")
# =========================================================================
# ENDPOINTS / ROUTES API
# =========================================================================

@app.post("/run-monitor/", tags=["Calculs"])
async def run_climate_monitor(
    country: str, 
    source_data: str, 
    resolution: float, 
    timeseries_resolution: float,
    background_tasks: BackgroundTasks
):
    """
    Lance la pipeline complète de monitoring pour un pays donné.
    Le calcul s'exécute en tâche de fond pour éviter de bloquer l'API.
    """
    if resolution not in RESOLUTION_CONFIG:
        raise HTTPException(status_code=400, detail=f"Résolution invalide. Choisissez parmi : {list(RESOLUTION_CONFIG.keys())}")
        
    if source_data.lower() not in ["arc2", "rfe2"]:
        raise HTTPException(status_code=400, detail="Source de données invalide. Choisissez 'arc2' ou 'rfe2'.")

    # Ajout de la pipeline lourde dans les tâches de fond
    background_tasks.add_task(
        pipeline_climonitor_complet, 
        country, source_data, resolution, timeseries_resolution
    )
    
    return {
        "status": "processing",
        "message": f"Le traitement pour le pays '{country}' a été démarré avec succès en arrière-plan."
    }





@app.get("/download-dashboard/{country}/{source_data}", tags=["Résultats"])
async def get_dashboard_html(country: str, source_data: str):
    """
    Permet de télécharger ou de visualiser le Dashboard HTML généré pour un pays.
    """
    # Remplace ce chemin par l'endroit exact où build_country_dashboard enregistre le fichier HTML final
    dashboard_path = Path.cwd() / "outputs" / "dashboards" / f"{country}_{source_data}_index.html"
    
    if not dashboard_path.exists():
        raise HTTPException(status_code=404, detail="Dashboard non trouvé. Veuillez d'abord lancer les calculs via /run-monitor/.")
        
    return FileResponse(path=dashboard_path, filename=f"{country}_dashboard.html", media_type='text/html')