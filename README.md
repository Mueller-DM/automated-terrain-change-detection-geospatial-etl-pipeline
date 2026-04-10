# Automated Terrain Change Detection Pipeline

A Python-based geospatial ETL pipeline for tracking volumetric terrain changes across multi-temporal Digital Terrain Models (DTMs). 

This script automates the extraction, transformation, and vectorization of topographic data. It uses raster math to find elevation differences, computer vision to reduce noise, and morphological operations to extract significant areas of change into shapefiles.

## Tech Stack
* **GDAL & Rasterio:** Core spatial data handling and raster math.
* **OpenCV (cv2):** Gaussian blurring for noise reduction.
* **WhiteboxTools:** Spatial clumping and area-based filtering.
* **GeoPandas:** Automated generation of spatial tile indices.

## Directory Setup
Your working directory needs to look like this:
```text
Working_Directory/
├── DTM/                  # Input: Put your time-step folders here
│   ├── Time_1/           
│   └── Time_2/           
├── temp/                 # Auto-generated
└── Change/               # Output: Final TIFFs and Shapefiles

## Installation
It is highly recommended to use a Conda environment to handle GDAL dependencies smoothly.
conda create -n geospatial_etl python=3.9
conda activate geospatial_etl
conda install -c conda-forge gdal
pip install -r requirements.txt

## Usage
Run the script from your terminal, pointing it to your main working directory:
python change_detection_pipeline.py --main_dir "C:/path/to/your/working_directory"

## Output
Check the Change/ folder for:

change_detection_*.tif: The smoothed raster showing elevation changes.

Change_*.shp: Vectorized polygons representing areas of significant volumetric change.
