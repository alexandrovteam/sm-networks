#!/usr/bin/env python
from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search
import pandas as pd
import pyarrow
import pyarrow.parquet

from time import time
import sys
import json
config = json.load(open(sys.argv[1]))

es_client = Elasticsearch(hosts=config['elasticsearch']['host'])

def formatCompoundIds(ids):
    return ",".join(["HMDB{:05d}".format(int(hmdbId)) for hmdbId in ids.split('|')])

fields = ['ds_id', 'ds_name', 'sf', 'adduct', 'fdr', 'msm', 'comp_names', 'comp_ids']
formatters = {
    'comp_ids': formatCompoundIds
}

data = {f: [] for f in fields}
for r in Search(using=es_client, index=config['elasticsearch']['index'])\
             .query('term', db_name='HMDB')\
             .fields(fields).params(size=1000).scan():
    for f in fields:
        d = r[f][0]
        if f in formatters:
            d = formatters[f](d)
        data[f].append(d)

df = pd.DataFrame(data, columns=fields)

pyarrow.parquet.write_table(pyarrow.Table.from_pandas(df), "annotations.parquet")
