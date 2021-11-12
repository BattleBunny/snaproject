import multiprocessing, os, random, logging, subprocess, json
import pandas as pd
import networkx as nx
import typer
import joblib
import numpy as np
from tqdm.auto import tqdm

from .progress_parallel import ProgressParallel, delayed
from .logger import logger

app = typer.Typer()

def get_network(network_index: int, multi_edge=True, edge_attr='datetime'):
    edgelist = (
        pd.read_pickle(f'data/{network_index:02}/+000/edgelist.pkl')
        .query("source != target")
    )
    if multi_edge:
        return nx.from_pandas_edgelist(edgelist, create_using=nx.MultiGraph,
                                       edge_attr=edge_attr)
    else:
        return nx.from_pandas_edgelist(edgelist)

@app.command()
def single(network: int, 
           average_clustering: bool = True,
           diameter: bool = True,
           verbose: bool = True):
    directory = f'data/{network:02}'
    out_directory = os.path.join(directory, 'properties')
    filepath_in = os.path.join(directory, 'graph.pkl')
    files = ['assortativity.float', 'connected_pairs.int', 'edges.int', 
             'nodes.int', 'triangles.int']
    if average_clustering:
        files.append('average_clustering.float')
    if diameter:
        files.append('diameter.int')
    
    assert os.path.isfile(filepath_in), f'{filepath_in} is missing'
    
    # All results are already there!
    if np.all([os.path.isfile(os.path.join(out_directory, f)) for f in files]):
        return
    elif verbose:
        print(
            json.dumps(
                {f: os.path.isfile(os.path.join(out_directory, f)) for f in files}
            )
        )

    if not verbose:
        logger.setLevel(logging.INFO)
    try:
        logger.debug(f'Get network {filepath_in=}')
        G = joblib.load(filepath_in)
    except:
        tqdm.write(f'{network=} failed!')
        return
    out_directory = os.path.join(directory, 'properties')
    os.makedirs(out_directory, exist_ok=True)
    
    # Nodes
    if not os.path.isfile(os.path.join(out_directory, 'nodes.int')):
        logger.debug('Get node count.')
        nodes = G.number_of_nodes()
        with open(os.path.join(out_directory, 'nodes.int'), 'w') as file:
            file.write(str(nodes))
            
    # Edges
    if not os.path.isfile(os.path.join(out_directory, 'edges.int')):
        logger.debug('Get edge count.')
        edges = G.number_of_edges()
        with open(os.path.join(out_directory, 'edges.int'), 'w') as file:
            file.write(str(edges))
    logger.debug('Get simple graph.')
    simple_graph = nx.Graph(G)
    
    # Average clustering coefficient
    if average_clustering:
        average_clustering_file = os.path.join(out_directory, 'average_clustering.float')
        if not os.path.isfile(average_clustering_file):
            logger.debug('Get average clustering coefficient')
            average_clustering_coefficient = nx.average_clustering(simple_graph)
            with open(average_clustering_file, 'w') as file:
                file.write(str(average_clustering_coefficient))
    
    # Connected pairs
    if not os.path.isfile(os.path.join(out_directory, 'connected_pairs.int')):
        logger.debug('Get connected pairs.')
        connected_pairs = simple_graph.number_of_edges()
        with open(os.path.join(out_directory, 'connected_pairs.int'), 'w') as file:
            file.write(str(connected_pairs))
    
    # Assortativity
    if not os.path.isfile(os.path.join(out_directory, 'assortativity.float')):
        logger.debug('Get assortativity.')
        assortativity = nx.degree_assortativity_coefficient(simple_graph)
        with open(os.path.join(out_directory, 'assortativity.float'), 'w') as file:
            file.write(str(assortativity))
            
    # Triangles
    if not os.path.isfile(os.path.join(out_directory, 'triangles.int')):
        logger.debug('Get triangle count.')
        triangles = sum(nx.triangles(simple_graph).values()) // 3
        with open(os.path.join(out_directory, 'triangles.int'), 'w') as file:
            file.write(str(triangles))   
    
    # Diameter
    diameter_file = os.path.join(out_directory, 'diameter.int')
    if diameter and not os.path.isfile(diameter_file):
        logger.debug('Get diameter')
        temp_file = os.path.join(directory, 'edgelist.edges')
        if not os.path.isfile(temp_file):
            nx.write_edgelist(simple_graph, os.path.join(directory, 'edgelist.edges'), data=False)
        dist_distri_file = os.path.join(directory, 'dist_distri.npy')
        if not os.path.isfile(dist_distri_file): 
            cp = subprocess.run(['./teexgraph'],
                                input=f'load_undirected {temp_file}\ndist_distri',
                                encoding='ascii',
                                stdout=subprocess.PIPE)
            dist_distri = np.array(cp.stdout.split()).reshape(-1, 2).astype(int)
            logger.debug(dist_distri)
            np.save(dist_distri_file, dist_distri)
        else:
            dist_distri = np.load(dist_distri_file)
        logger.debug(str(dist_distri[:,0][-1]))
        with open(diameter_file, 'w') as file:
            file.write(str(dist_distri[:,0][-1]))            

        
@app.command()
def all(network: int = None, 
        n_jobs: int = -1, 
        average_clustering: bool = True,
        diameter: bool = True,
        shuffle: bool = True, 
        seed: int = 42, 
        verbose: bool = False):
    # Network selection
    if network is None:
        networks = [network 
                    for network in np.arange(1, 31) 
                    if network not in [15, 17, 26, 27]]
    else:
        networks = [network]        
    
    if shuffle:
        random.seed(seed)
        random.shuffle(networks)
    if multiprocessing.cpu_count() > len(networks):
        n_jobs = len(networks)
    ProgressParallel(n_jobs=n_jobs, total=len(networks))(
        delayed(single)(
            network=network, 
            average_clustering=average_clustering,
            diameter=diameter,
            verbose=verbose
        ) for network in networks
    )

if __name__ == '__main__':
    app()   