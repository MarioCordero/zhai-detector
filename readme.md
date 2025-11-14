cd ~/Desktop/UCR/zhai-detector
python3 -m venv venv
source venv/bin/activate
pip install geopandas shapely pyproj rtree pandas numpy

# descomprime los ZIP en Datasets/raw antes de correr
cd Project
python3 build_dataset_full.py