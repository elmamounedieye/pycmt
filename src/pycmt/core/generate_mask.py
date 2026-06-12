import xarray as xr
from rasterio.features import rasterize
from rasterio.transform import from_origin
import rioxarray 
import numpy as np
import geopandas as gpd
import os
import pathlib
from pathlib import Path


class Maskgenerator:
    def __init__(self, shapefile_path):
        #print(f"#####################Mask generator#############################")
        #print(os.getcwd())
        #print(f"shapefile path : {shapefile_path}")
        #print(f"##################################################")
        self.gdf = gpd.read_file(shapefile_path)
        self.shapes = [geom for geom in self.gdf.geometry]
        self.raw_bounds = self.gdf.total_bounds
        #print(f"=================*******===============")
        #print(f"raw bounds : {self.raw_bounds}")
        #print(f"=================*******===============")

    def align_grid(self, rsl, origin_lon, origin_lat):
        if origin_lon is None:
            return self.raw_bounds
        
        #offset calculation (Nptsraw)
        nptsL = int((self.raw_bounds[0] - origin_lon) / rsl)
        nptsR = int((self.raw_bounds[2] - origin_lon) / rsl)
        nptsS = int((self.raw_bounds[1] - origin_lat) / rsl)
        nptsN = int((self.raw_bounds[3] - origin_lat) / rsl)


        #Ajustement
        new_minlon = ((nptsL - 1) *rsl) + origin_lon
        new_maxlon = ((nptsR + 1) *rsl) + origin_lon
        new_minlat = ((nptsS - 1) *rsl) + origin_lat
        new_maxlat = ((nptsN + 1) *rsl) + origin_lat

        return new_minlon, new_minlat, new_maxlon, new_maxlat
    
    def create_and_save_mask(self, rsl, bounds, output_path):

        minlon, minlat, maxlon, maxlat = bounds
        width = int(round((maxlon - minlon) /rsl))
        height = int(round((maxlat - minlat) /rsl))

        #Transformation Rasterio
        transform = from_origin(minlon, maxlat, rsl, rsl)

        #Mask1 (1 inside, 0 outside)
        mask1 = rasterize(self.shapes, out_shape=(height, width),
                          transform=transform, fill=0, default_value=1)
        
        #Mask2 (-1 outside, 1 dedans)
        mask2 = rasterize(self.shapes, out_shape=(height, width),
                          transform=transform, fill=-1, default_value=0)
        
        combined_mask = (mask1 + mask2).astype(np.float32)

        #Calculating pixels' center for Netcdf
        lats = np.linspace(maxlat - rsl/2, minlat +rsl/2, height)
        lons = np.linspace(minlon + rsl/2, maxlon -rsl/2, width)

        ds = xr.DataArray(
            combined_mask,
            coords=[("lat", lats), ("lon", lons)],
            name="mask_data",
            attrs={"Title": "Mask data", "res": rsl, "calendar": "365_day"}
        )

        ds = ds.where(ds != 0, -9999.0)

        #Adding coordinates system (CRS)
        ds.rio.write_crs("epsg:4326", inplace = True)
        ds.to_netcdf(output_path, mode="w")
        #print(f"=================output_path===============")
        #print(f"{output_path}")
        return output_path
    

def run_workflow(iso_code, country):
    config = {
            "0p036": (0.036, -179.982, -55.314),
            "0p0375": (0.0375, -19.0125, -35.9625),
            "0p25": (0.25, 0.125, -89.875),
            "0p5": (0.5, 0.25, -89.75),
            "0p1": (0.1, None, None),
            "1p0": (1.0, None, None)
        }
    
    shp_path = Path(__file__).resolve().parents[1] / "data" / "gis_resources" / "countries" /f"{iso_code}_adm"
    generator = Maskgenerator(shp_path)
    
    for rsl_name, (rsl_val, org_lon, org_lat) in config.items():
        #print(f"Processing {rsl_name}...")
        bounds = generator.align_grid(rsl_val, org_lon, org_lat)
        lon_bounds = bounds[0], bounds[2]
        lat_bounds = bounds[1], bounds[3]


        out_dir =Path(__file__).resolve().parents[1]/ "data"/ "gis_resources"/f"country_masks{rsl_name}/365dcal"
        os.makedirs(out_dir, exist_ok=True)

        output_path = generator.create_and_save_mask(rsl_val, bounds, f"{out_dir}/{iso_code}_mask.nc")

    country_latlon = [country, round(bounds[1],2), round(bounds[3],2), round(bounds[0],2), round(bounds[2],2), rsl_val, rsl_val]  
    #cur = os.getcwd()
    #print(f"cur : {cur}")
    country_info = Path(__file__).resolve().parents[1] / "data" / f"{country}_latlon"
    with open(country_info, "w", encoding="utf-8") as f:
        country_latlon = " ".join(map(str, country_latlon))
        f.write(country_latlon) 

    return #country_info                                           