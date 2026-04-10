# -*- coding: utf-8 -*-
"""
Automated Raster Change Detection & Tile Processing Pipeline
Author: David Mueller

Description:
This script processes multi-temporal Digital Terrain Models (DTMs). It builds 
spatial tile indices using GeoPandas, performs normalized change detection using 
GDAL/Rasterio, applies Gaussian blurring for noise reduction, and polygonizes 
significant volumetric changes into shapefiles.
"""

import os
import argparse
import numpy as np
import geopandas as gpd
from shapely.geometry import box
from osgeo import gdal, ogr, osr
from osgeo.gdalconst import GA_ReadOnly
import rasterio 
import cv2
from whitebox.whitebox_tools import WhiteboxTools

# Enable GDAL exceptions for better error handling
gdal.UseExceptions()
wbt = WhiteboxTools()

def create_tile_index(directory, index_shp):
    """Generates a shapefile footprint index of all GeoTIFFs in a directory."""
    print(f"Building tile index for: {directory}")
    records = []
    for root, dirs, files in os.walk(directory):
        for fname in files:
            if fname.endswith(".tif"):
                filepath = os.path.join(root, fname)
                bounds = rasterio.open(filepath).bounds
                records.append({
                    'location': fname, 
                    'geometry': box(bounds, bounds, bounds, bounds)
                })
    
    # Batch create GeoDataFrame for performance (avoids deprecated .append)
    df = gpd.GeoDataFrame(records)
    df.to_file(index_shp)
    print(f"Tile index saved to: {index_shp}")

def find_tiles(index_shp, strata_poly):
    """Finds intersecting tiles based on a strata polygon."""
    poly1 = gpd.read_file(index_shp)
    poly2 = gpd.read_file(strata_poly)
    inter = gpd.overlay(poly1, poly2, how='intersection')
    loc_list = inter['location_1'].to_list()
    return loc_list

def merge_tiles(loc_list, dtm_dir, strata_poly, output_dtm="dtm.tif"):       
    """Merges tiles into a VRT and clips to the strata boundary."""
    paths = [os.path.join(dtm_dir, path) for path in loc_list]
    vrt = gdal.BuildVRT("dtm.vrt", paths)
    trans = gdal.Translate("fulldtm.tif", vrt)
    gdal.Warp(output_dtm, trans, cutlineDSName=strata_poly, cropToCutline=True, dstNodata=np.nan) 

def open_raster_as_array(raster_path):
    """Opens a raster and returns a NumPy array with NoData handled."""
    ras = gdal.Open(raster_path)
    array = ras.ReadAsArray()
    array[array == -9999] = np.nan
    return array

def save_array_to_raster(reference_raster_path, out_array, out_path, driver_fmt="GTiff"):
    """Saves a NumPy array to a spatial raster using a reference dataset."""
    ref_ds = gdal.Open(reference_raster_path)
    driver = gdal.GetDriverByName(driver_fmt)
    out_ds = driver.CreateCopy(out_path, ref_ds, strict=0)
    out_ds.WriteArray(out_array)
    out_ds = None # Close and flush to disk

def match_extent(raster1, raster2, output_path):
    """Clips raster2 to the exact bounding box of raster1."""
    maskras = gdal.Open(raster1, GA_ReadOnly)
    geo_transform = maskras.GetGeoTransform()
    minx = geo_transform
    maxy = geo_transform
    maxx = minx + geo_transform * maskras.RasterXSize
    miny = maxy + geo_transform * maskras.RasterYSize

    clipdata = gdal.Open(raster2, GA_ReadOnly)
    gdal.Translate(output_path, clipdata, format='GTiff', projWin=[minx, maxy, maxx, miny]) 

def run_change_detection(main_dir):
    """Executes the core raster math, blurring, and polygonization pipeline."""
    dtm_dir = os.path.join(main_dir, "DTM")
    temp_dir = os.path.join(main_dir, "temp")
    change_dir = os.path.join(main_dir, "Change")

    for directory in [temp_dir, change_dir]:
        os.makedirs(directory, exist_ok=True)

    directories_list = os.listdir(dtm_dir)
    if len(directories_list) < 2:
        print("Error: DTM directory must contain at least two time-step subdirectories.")
        return

    dir1, dir2 = directories_list[:2]
    t1_path = os.path.join(dtm_dir, dir1)
    
    b = 1 # File iteration counter

    for file in os.listdir(t1_path):        
        if not file.endswith(".tif"):
            continue

        print(f"Processing change detection for: {file}")
        
        # Path setups
        ti1 = os.path.join(dtm_dir, dir1, file) 
        ti2 = os.path.join(dtm_dir, dir2, file)
        
        nrmt_1 = os.path.join(temp_dir, f"time_1_{b}.tif") 
        nrmt2 = os.path.join(temp_dir, f"time2_{b}.tif")
        chng1 = os.path.join(change_dir, f"zchange_detection{b}.tif")
        chng2 = os.path.join(change_dir, f"z1change_detection{b}.tif")
        chng_final = os.path.join(change_dir, f"change_detection{b}.tif")
        reclass = os.path.join(temp_dir, f"reclass{b}.tif")
        eucallo = os.path.join(temp_dir, f"eucallo{b}.tif")
        rgngrp = os.path.join(temp_dir, f"rgngrp{b}.tif")
        
        # --- Raster Math: Normalization ---
        t1_array = open_raster_as_array(ti1)
        mean_t1 = np.nanmean(t1_array)
        norm_ti1 = t1_array - mean_t1
        save_array_to_raster(ti1, norm_ti1, nrmt_1)
        
        t2_array = open_raster_as_array(ti2)                
        mean_t2 = np.nanmean(t2_array)
        norm_ti2 = t2_array - mean_t2
        save_array_to_raster(ti2, norm_ti2, nrmt2)

        # --- Change Calculation & Smoothing ---
        changearray = norm_ti1 - norm_ti2
        save_array_to_raster(nrmt2, changearray, chng1)
        
        # Apply Gaussian Blur using OpenCV
        out_blur = cv2.GaussianBlur(open_raster_as_array(chng1), (5,5), 0)
        save_array_to_raster(nrmt2, out_blur, chng2)

        # Merge Change and Smoothed outputs
        m_chng = open_raster_as_array(chng1) * open_raster_as_array(chng2)
        save_array_to_raster(nrmt2, m_chng, chng_final)

        # --- Reclassification ---
        ras = gdal.Open(chng_final)
        reclass_array = open_raster_as_array(chng_final)            
        
        # Vectorized reclassification (much faster than nested loops)
        reclass_array = np.where(reclass_array < -0.5, 1, reclass_array)
        reclass_array = np.where(reclass_array > 0.5, 2, reclass_array)
        reclass_array = np.where((reclass_array >= -0.5) & (reclass_array <= 0.5), np.nan, reclass_array)
        reclass_array[reclass_array == 0] = np.nan

        # Save reclassified raster
        driver = gdal.GetDriverByName('GTiff')
        out_file = driver.Create(reclass, ras.RasterXSize, ras.RasterYSize, 1)
        out_file.WriteArray(reclass_array)
        out_file.SetProjection(ras.GetProjection())
        out_file.SetGeoTransform(ras.GetGeoTransform())            
        out_file = None

        # --- WhiteboxTools Morphological Operations ---
        wbt.clump(reclass, rgngrp, diag=True, zero_back=True)
        wbt.filter_raster_features_by_area(rgngrp, eucallo, 5, background="zero")
        
        # --- Polygonization ---
        print("Polygonizing change areas...")
        ras1 = gdal.Open(eucallo) # Using filtered clumps as extraction boundaries
        band = ras1.GetRasterBand(1)
        
        out_shapefile = os.path.join(change_dir, f"Change_{b}.shp")
        ogr_driver = ogr.GetDriverByName("ESRI Shapefile")
        
        srs = osr.SpatialReference()
        srs.ImportFromWkt(ras1.GetProjection())
        
        out_datasource = ogr_driver.CreateDataSource(out_shapefile)
        out_layer = out_datasource.CreateLayer(f"Change_{b}", srs)
        
        # Add a field for volume/change metric
        new_field = ogr.FieldDefn('Volume', ogr.OFTInteger)
        out_layer.CreateField(new_field)
        
        gdal.Polygonize(band, band, out_layer, 0, [], callback=None)
        
        out_datasource.Destroy()
        ras1 = None
        b += 1

    print("Change detection pipeline complete.")

if __name__ == '__main__':
    # Use argparse so the script can be run via CLI without hardcoded paths
    parser = argparse.ArgumentParser(description="Run the spatial change detection pipeline.")
    parser.add_argument('--main_dir', type=str, required=True, help="Path to the main working directory containing the 'DTM' folder.")
    args = parser.parse_args()

    run_change_detection(args.main_dir)
