import pyarrow.parquet
import pandas as pd
import numpy as np
import scipy.spatial.distance as ssd
import bottle

from tempfile import mkdtemp
import os.path
from glob import glob
from zipfile import ZipFile
import json

class NetworkGenerator:
    def __init__(self, config):
        self.config = config

    def _load_data(self):
        df = pyarrow.parquet.read_table(self.config['annotations']).to_pandas()
    
        # some datasets are present with multiple ds_ids; we take only the largest here
        ds_ids = df[['ds_id', 'ds_name']].groupby('ds_name')\
                                         .apply(lambda df: max(df['ds_id']))
        self.annot = df[df['ds_id'].isin(ds_ids.values)].copy()
        self.name_to_id = {x[0]: x[1] for x in ds_ids.iteritems()}
        
        self.annot['msm'] = self.annot['msm'].astype(np.float32)
        self.annot['fdr'] = self.annot['fdr'].astype(np.float32)
        self.datasets = self.annot['ds_name'].unique()
        self.metadata = pd.read_csv(self.config['datasets'])
        self.metadata.rename(columns={'id': 'ds_id', 'name': 'ID'}, inplace=True)
        self.metadata = self.metadata.set_index('ds_id')

    def annotations(self, datasets):
        return self.annot[self.annot['ds_name'].isin(datasets)]

    def pass_fdr_table(self, annot, max_fdr):
        passes = annot[annot['fdr'] <= max_fdr].copy()
        passes['pass_fdr'] = True
        pass_table = passes.pivot_table('pass_fdr', aggfunc='max', index=['sf'],
                                        columns=['ds_name'], fill_value=False)
        return pass_table

    def _block(self, pass_fdr_table, j, block_size):
        df_block = pass_fdr_table[j:j + block_size]
        array_block = df_block.as_matrix().astype(np.float)
        dataset_counts = array_block.sum(axis=1)
        return array_block, df_block.index, dataset_counts

    def _loopify(self, edges, id1_name, id2_name):
        loops = edges[edges[id1_name] == edges[id2_name]]
        edges = edges[edges[id1_name] < edges[id2_name]]
        loops = loops[~loops[id1_name].isin(edges[id1_name])]
        loops = loops[~loops[id2_name].isin(edges[id2_name])]
        return edges.append(loops)

    def annotation_network(self, datasets, max_fdr, cutoff):
        self._load_data()
        annotations = self.annotations(datasets)
        full_pass_table = self.pass_fdr_table(annotations, max_fdr)
        pass_fdr_table = full_pass_table[full_pass_table.sum(axis=1) >= 2]
        edges = []

        distance_name = 'relative co-occurence'
        n_sf = pass_fdr_table.shape[0]
        block_size = 1000
        blocks = range(0, n_sf, block_size)
        for i, j in enumerate(blocks):
            mj, idx_j, nj = self._block(pass_fdr_table, j, block_size)
            for k in blocks[i:]:
                mk, idx_k, nk = self._block(pass_fdr_table, k, block_size)
                intersection = mj.dot(mk.T)
                union = np.add.outer(nj, nk) - intersection
                ratio = intersection / union
                ratio = pd.DataFrame(ratio, index=idx_j, columns=idx_k)
                ratio.index.rename('sf1', inplace=True)
                d = pd.melt(ratio.reset_index(), id_vars=['sf1'],
                            var_name='sf2', value_name=distance_name)
                d = d[d[distance_name] >= cutoff]
                edges.append(d[d['sf1'] <= d['sf2']])

        edges = pd.concat(edges)
        edges = self._loopify(edges, 'sf1', 'sf2').sort_values(by=['sf1', 'sf2'])
        edges['comments'] = ''

        nodes = full_pass_table.copy().astype(int)
        nodes['# of datasets'] = nodes.sum(axis=1)
        nodes.sort_values(by='# of datasets', ascending=False, inplace=True)
        compound_info = annotations[['sf', 'comp_names', 'comp_ids']]
        nodes = nodes.join(compound_info.drop_duplicates().set_index('sf'))
        nodes = nodes.reset_index()
        nodes['comments'] = ''
        return {'nodes': nodes, 'edges': edges}

    def _cosine_similarities(self, msm_table):
        dist = ssd.squareform(ssd.pdist(msm_table.T.as_matrix(), 'cosine'))
        return pd.DataFrame(data=1.0 - dist, 
                            index=msm_table.columns, columns=msm_table.columns)

    def _pairwise_df(self, msm_table):
        cosine_sim = self._cosine_similarities(msm_table)
        df = pd.melt(cosine_sim.reset_index(), id_vars=['ds_id'], var_name='ID2', 
                     value_name='cosine_similarity')\
               .rename(columns={'ds_id': 'ID1'})
        return df

    def dataset_network(self, datasets, threshold1, threshold2):
        self._load_data()
        annot = self.annotations(datasets)
        annotated_sf = annot[annot['fdr'] <= 0.1]['sf'].unique()
        annot = annot[annot['sf'].isin(annotated_sf)]

        fdr_table = (annot.pivot_table('fdr', index=['sf', 'adduct'],
                                      columns=['ds_id'], fill_value=1.0) * 100)\
                                      .astype(np.uint8)
        msm_table = annot.pivot_table('msm', index=['sf', 'adduct'],
                                      columns=['ds_id'], fill_value=0.0)
        avg_msm = msm_table.sum(axis=1) / len(msm_table.columns)
        sorted_avg_msm = avg_msm.sort_values(ascending=False)
        cutoff = -1
        n_top = 1000
        if len(sorted_avg_msm) > n_top:
            cutoff = sorted_avg_msm[n_top]
        cosine_sim_full = self._pairwise_df(msm_table[avg_msm > cutoff])
        msm_table[fdr_table > 20] = 0.0
        cosine_sim_02 = self._pairwise_df(msm_table[avg_msm > cutoff])
        msm_table[fdr_table > 10] = 0.0
        cosine_sim_01 = self._pairwise_df(msm_table[avg_msm > cutoff])

        edges = cosine_sim_full.copy()
        edges['cosine_similarity_fdr0.1'] = cosine_sim_01['cosine_similarity']
        edges['cosine_similarity_fdr0.2'] = cosine_sim_02['cosine_similarity']
        edges = edges[(edges['cosine_similarity_fdr0.1'] >= threshold1) &
                      (edges['cosine_similarity_fdr0.2'] >= threshold2)]
        edges['ID1'] = pd.merge(edges[['ID1']], self.metadata[['ID']], 
                                left_on='ID1', right_index=True)['ID']
        edges['ID2'] = pd.merge(edges[['ID2']], self.metadata[['ID']], 
                                left_on='ID2', right_index=True)['ID']

        edges = edges.fillna(0)
        edges = self._loopify(edges, 'ID1', 'ID2')

        nodes = fdr_table.groupby(level='sf').agg('min').T\
                         .reindex(self.metadata.index) / 100.0
        nodes.index.rename('ds_id', inplace=True)
        nodes['# of annotations @ FDR = 0.1'] = (nodes <= 0.1).sum(axis=1)
        nodes['# of annotations @ FDR = 0.2'] = (nodes <= 0.2).sum(axis=1)
        ds_ids = [self.name_to_id[name] for name in datasets]
        nodes = self.metadata[self.metadata.index.isin(ds_ids)].join(nodes)
        nodes = nodes.reset_index()
        del nodes['ds_id']

        return {'nodes': nodes, 'edges': edges}

    def generate_networks(self, query):
        tmpdir = mkdtemp()
        def F(fn):
            return os.path.join(tmpdir, fn)

        datasets = self.dataset_network(query['datasets'], 
                                        query['thresholdD01'], query['thresholdD02'])
        datasets['nodes'].to_csv(F('Dnodes.csv'), index=False)
        datasets['edges'].sort_values(by=['ID1', 'ID2'])\
                         .to_csv(F('Dedges.csv'), index=False, float_format='%.4f')

        annot_01 = self.annotation_network(query['datasets'], 0.1, query['thresholdA'])
        annot_01['nodes'].to_csv(F("Anodes01.csv"), index=False)
        annot_01['edges'].to_csv(F("Aedges01.csv"), index=False)

        annot_02 = self.annotation_network(query['datasets'], 0.2, query['thresholdA'])
        annot_02['nodes'].to_csv(F("Anodes02.csv"), index=False)
        annot_02['edges'].to_csv(F("Aedges02.csv"), index=False)

        with open(F("settings.json"), "w+") as j:
            json.dump(query, j, indent=4, sort_keys=True)

        with ZipFile(F('networks.zip'), 'w') as z:
            for fn in list(glob(tmpdir + "/*.csv")) + [F('settings.json')]:
                z.write(fn, os.path.basename(fn))
                os.unlink(fn)
        return tmpdir, 'networks.zip'

# EDIT this to point to the correct files!
config = {
    'annotations': '/home/ec2-user/Dropbox/networks/annotations.parquet',
    'datasets': '/home/ec2-user/Dropbox/networks/datasets.csv',
}

gen = NetworkGenerator(config)

@bottle.route("/")
def index():
    return bottle.static_file("index.html", "templates")

@bottle.route("/datasets")
def datasets():
    gen._load_data()
    return bottle.template("templates/datasets.html", 
                           names=sorted(gen.name_to_id.keys()))

@bottle.post("/network")
def network():
    print(list(bottle.request.forms))
    query = {
        'thresholdD01': float(bottle.request.forms.get('thresholdD01')),
        'thresholdD02': float(bottle.request.forms.get('thresholdD02')),
        'thresholdA': float(bottle.request.forms.get('thresholdA')),
        'datasets': [s.strip() for s in bottle.request.forms.get('datasets').split("\n")]
    }
    print(query)
    tmpdir, fn = gen.generate_networks(query)
    return bottle.static_file(fn, root=tmpdir)

if __name__ == "__main__":
    bottle.run(host='0.0.0.0', port=5000, debug=True)

