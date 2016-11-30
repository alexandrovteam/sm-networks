#!/usr/bin/env python
import psycopg2
import psycopg2.extras
import pandas as pd
from pandas.io.json import json_normalize

import sys
import json
config = json.load(open(sys.argv[1]))

conn = psycopg2.connect(**config['db'])
cursor = conn.cursor(cursor_factory=psycopg2.extras.NamedTupleCursor)

cursor.execute("select * from dataset")
df = pd.DataFrame(cursor.fetchall())
img_bounds_df = json_normalize(df['img_bounds'])
metadata_df = json_normalize(df['metadata'])
config_df = json_normalize(df['config'])
del df['metadata'], df['config'], df['img_bounds']
df = df.join(metadata_df).join(config_df).join(img_bounds_df)
df.to_csv("datasets.csv", index=False)
