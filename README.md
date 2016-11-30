# sm-networks

## Webserver

The webserver provides a simple form for generating dataset and annotation networks that can be imported into Cytoscape for visualization and exploration.

## Local copy of central data

Information about datasets and annotations is stored in `annotations.parquet` and `datasets.csv` files.
Put `download_ds_metadata.py` and `download_es_annotations.py` into your Crontab file. Both scripts expect sm-engine config.json file as the single parameter.
