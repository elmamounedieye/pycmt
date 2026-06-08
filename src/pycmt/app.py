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
    download_cmorph_data,
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
    title="⚡ PyCMT Climate Monitoring API",
    description="Monitoring API for climate and geospatial processing",
    version="1.0.0"
)

RESOLUTION_CONFIG = {
    0.036: '0p036', 0.0375: '0p0375', 0.25: '0p25',
    0.5: '0p5', 0.1: '0p1', 1.0: '1p0'
}

# =========================================================================
# =========================================================================

def pipeline_climonitor_complet(
    country: str, 
    rndta: str, 
    rsl: float, 
    ts_rsl: float,
    run_spp: bool,
    run_spi: bool,
    run_vhi: bool
):
    try:
        if country == "Africa":
            country_iso = "AFR"
            package_root = Path(__file__).resolve().parent
            source = package_root / "data" / "AFR_adm"
            destination = package_root / "data" / "gis_resources" / "countries" / "AFR_adm"
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source, destination, dirs_exist_ok=True)
        elif country == "World":
            country_iso = "WRL"
            package_root = Path(__file__).resolve().parent
            source = package_root / "data" / "WRL_adm"
            destination = package_root / "data" / "gis_resources" / "countries" / "WRL_adm"
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source, destination, dirs_exist_ok=True)
        else:
            country_iso = get_country_iso(country)
            download_gadm_country(country_iso)
        
        run_workflow(country_iso, country)
        generate_pixel_arguments(ts_rsl, country_iso, country)
        plot_pix_coordinates(country, country_iso, rndta)
        
        if rndta.lower() == "arc2":
            download_arc2_data()
        elif rndta.lower() == "rfe2":
            download_rfe2()
        if rndta.lower()== "cmorph":
            download_cmorph_data()

        plot_precip(rsl, RESOLUTION_CONFIG[rsl], country_iso, country, rndta)
        generate_tseries(country_iso, country, rndta)
        generate_html_map(country, rndta)
        
        
        # --- Module SPP ---
        if run_spp:
            print(f"📥 Downloading and generating SPP for {country}...")
            download_spp_noaa("rfe2")
            download_spp_noaa("cmorph")
            run_orchestrator_spp(country, country_iso, "rfe2")
            run_orchestrator_spp(country, country_iso, "cmorph")
            print(f"✅ SPP Done.")

        # --- Module SPI / Runoff / Soil Moisture ---
        if run_spi:
            print(f"📥 Downloading and generating SPI, Runoff and Soil Moisture for {country}...")
            download_runoff_data()
            download_xsm_data()
            download_spi("cmorph")
            download_spi("rfe2")
            generate_runoff(country_iso, country)
            generate_soilmoisture(country_iso, country)
            calc_spi(country_iso, country, "cmorph")
            calc_spi(country_iso, country, "rfe2")
            print(f"✅ SPI, Runoff and Soil Moisture Done.")

        # --- Module VHI ---
        if run_vhi:
            print(f"📥 Donwloading and generating VHI for {country}...")
            run_retrieval_vhi()
            do_vhi(country, country_iso)
            print(f"✅ VHI Done.")

        build_country_dashboard(country, rndta)
        print(f"🎉 API pipeline successful for {country} ({rndta})")
        
    except Exception as e:
        print(f"Background pipeline failure : {str(e)}")


# =========================================================================
# ENDPOINTS / ROUTES API
# =========================================================================

@app.post("/upload-data/", tags=["Data"])
async def upload_custom_file(subfolder: str = "data", file: UploadFile = File(...)):
    
    dest_dir = Path(pycmt.__file__).resolve().parent / subfolder
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / file.filename

    try:
        with open(dest_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        return {"status": "success", "filename": file.filename, "saved_at": str(dest_path.resolve())}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during saving : {str(e)}")


@app.post("/run-monitor/", tags=["Calculs"])
async def run_climate_monitor(
    country: str, 
    source_data: str, 
    resolution: float, 
    timeseries_resolution: float,
    background_tasks: BackgroundTasks,
    run_spp: bool = True,
    run_spi: bool = True,
    run_vhi: bool = True
):
   
    if resolution not in RESOLUTION_CONFIG:
        raise HTTPException(status_code=400, detail=f"Invalid resolution. Please choose among : {list(RESOLUTION_CONFIG.keys())}")
        
    if source_data.lower() not in ["arc2", "rfe2", "cmorph"]:
        raise HTTPException(status_code=400, detail="Invalid data soure. Choose 'arc2' or 'rfe2' or 'cmorph.")

    background_tasks.add_task(
        pipeline_climonitor_complet, 
        country, source_data, resolution, timeseries_resolution,
        run_spp, run_spi, run_vhi
    )
    
    return {
        "status": "processing",
        "message": f"Processing for '{country}' has started in background.",
        "modules_actives": {
            "spp": run_spp,
            "spi_hydrology": run_spi,
            "vhi": run_vhi
        }
    }


@app.get("/download-dashboard/{country}/{source_data}", tags=["Results"])
async def get_dashboard_html(country: str, source_data: str):
  
    dashboard_path = Path.cwd() / "outputs" / "dashboards" / f"{country}_{source_data}_index.html"
    
    if not dashboard_path.exists():
        raise HTTPException(status_code=404, detail="Dashboard not found. Please, run the computation with /run-monitor/.")
        
    return FileResponse(path=dashboard_path, filename=f"{country}_dashboard.html", media_type='text/html')