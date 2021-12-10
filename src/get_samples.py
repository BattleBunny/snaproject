import warnings
import logging
import random
import os
import itertools

import networkx as nx
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import typer

from .logger import logger
from .progress_parallel import ProgressParallel, delayed

app = typer.Typer()

# DANGER ZONE
warnings.filterwarnings("ignore")


def check_common_neighbor(G, u, v):
    """Return True if u and v have a common neighbor.

    Arguments:
        G: nx.Graph
        u: node in G
        v: node in v

    """
    assert u != v
    assert u in G and v in G
    for w in G[u]:
        if w in G[v] and w not in (u, v):
            return True
    else:
        return False


@app.command()
def single(network: int,
           cutoff: int = 2,
           sample_size: int = 10_000,
           seed: int = 42,
           verbose: bool = True):
    """Get positive and negative samples (i.e. pairs of nodes) for the temporal 
    link prediction problem.

    Arguments:
        - network
        - cutoff (10_000): Get only pairs of nodes with at most this distance.
        - sample_size (default 10_000): 
            Get this number of positives and negatives.
        - seed (default 42)
        - verbose (default True)
    """
    if not verbose:
        logger.setLevel(logging.INFO)
    directory = f'/data/s1620444/{network:02}'
    assert os.path.isdir(directory), f"{directory=} does not exist"

    filepath_in = os.path.join(directory, 'edgelist.pkl')
    filepath_out = os.path.join(directory, 'samples.pkl')

    if os.path.isfile(filepath_out) or not os.path.isfile(filepath_in):
        return

    assert cutoff == 2, "Not implement for any other cutoff value than 2."

    edgelist = pd.read_pickle(filepath_in)

    assert {'source', 'target', 'phase'}.issubset(set(edgelist.columns))
    edgelist.query("source != target", inplace=True)  # Do not allow selfloops

    edgelist_mature = edgelist.query("phase == 'mature'")[['source', 'target']]
    graph_mature = nx.from_pandas_edgelist(edgelist_mature)

    nodes_mature_list = list(graph_mature.nodes)
    nodes_mature_set = set(graph_mature.nodes)

    probe_iterator = (
        edgelist
        .query("phase == 'probe'")
        [['source', 'target']]
        .itertuples(index=False)
    )
    probes_list = [(u, v) for u, v in probe_iterator]
    probes_set = set(probes_list)
    random.seed(seed)
    random.shuffle(probes_list)

    logger.debug(len(nodes_mature_set))

    positives = []
    tqdm_kwargs = {'disable': not verbose, 'leave': True}
    with tqdm(total=sample_size, **tqdm_kwargs) as pbar:
        for u, v in tqdm(probes_list, position=1, **tqdm_kwargs):
            if len(positives) < sample_size:
                if (
                    u in nodes_mature_set and
                    v in nodes_mature_set and
                    (u, v) not in graph_mature.edges() and
                    check_common_neighbor(graph_mature, u, v)
                ):
                    positives.append((u, v))
                    pbar.update(1)
            else:
                break
    logger.debug(f"S{filepath_out=}, {len(positives)} positives sampled")

    negatives = []
    with tqdm(total=len(positives), **tqdm_kwargs) as pbar:
        while len(negatives) < len(positives):
            u = random.choice(nodes_mature_list)
            nbs_u = list(graph_mature[u])
            nb_u = random.choice(nbs_u)
            v = random.choice(list(graph_mature[nb_u]))
            if v not in nbs_u and u < v and (u, v) not in probes_set:
                negatives.append((u, v))
                pbar.update(1)
    logger.debug(f"S{filepath_out=}, {len(negatives)} negatives sampled")

    result = pd.concat(
        [pd.Series(False, index=negatives), pd.Series(True, index=positives)])  # type: ignore

    result.to_pickle(filepath_out)


@app.command()
def discrete():
    """"Get samples for all discrete networks """
    discrete_ids = [18, 20, 21, 9, 4, 8, 24, 16, 11, 10]
    for i in discrete_ids:
        try:
            single(network=i)
        except:
            logger.debug(f"COULD NOT GET SAMPLES FOR NETWORK ID {i}")


@app.command()
def all(n_jobs: int = -1,
        verbose: bool = False,
        shuffle: bool = True,
        seed: int = 42,
        sample_size: int = 10_000):
    iterations = [i for i in range(1, 31) if i != 5]
    if shuffle:
        random.seed(seed)
        random.shuffle(iterations)
    ProgressParallel(n_jobs=n_jobs, total=len(iterations))(
        delayed(single)(
            network=network,
            sample_size=sample_size,
            verbose=verbose)
        for network in iterations
    )


@app.command()
def check():
    iterator = list(
        itertools.product(
            [network for network in np.arange(1, 31)
             if network not in [15, 17, 26, 27]]
        )
    )
    for n in iterator:
        if not os.path.isfile(f'/data/s1620444/{n:02}'):
            print(n)


if __name__ == '__main__':
    app()
