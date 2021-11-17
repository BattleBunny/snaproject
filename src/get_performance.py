import os
import random
from typing import List

import numpy as np
import pandas as pd
import sklearn.ensemble
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection
import sklearn.preprocessing
import sklearn.pipeline
import typer
import xgboost as xgb
from tqdm.auto import tqdm

from .progress_parallel import ProgressParallel, delayed

# DANGER ZONE
import warnings
warnings.filterwarnings("ignore")

app = typer.Typer()


def predict(directory: str,
            feature_set='II-A',
            clf='LogisticRegression',
            random_state=42,
            n_jobs=-1):
    if feature_set == 'I':
        def check(x): return x in [
            'aa.npy', 'cn.npy', 'jc.npy', 'pa.npy', 'sp.npy']
    elif feature_set == 'II-A':
        def check(x): return not x.startswith('na')
    elif feature_set == 'II-B':
        def check(x): return (
            (x in ['aa.npy', 'cn.npy', 'jc.npy',
             'pa.npy', 'sp.npy'] or ('_q100' in x))
            and not x.startswith('na')
        )
    elif feature_set == 'III':
        def check(x): return (
            x.startswith('na') and not 'm5' in x
        )
    else:
        raise Exception(f'{feature_set=} not recognized')

    assert os.path.isdir(directory), f'missing {directory=}'
    feature_dir = os.path.join(directory, 'features')
    if not os.path.isdir(feature_dir):
        print("returning none!")
        return None
    samples_filepath = os.path.join(directory, 'samples.pkl')
    assert os.path.isfile(samples_filepath), f'missing {samples_filepath=}'

    X = pd.DataFrame({
        f.name: np.load(f.path)
        for f in os.scandir(feature_dir) if check(f.name)
    })

    X_train, X_test, y_train, y_test = (
        sklearn.model_selection.train_test_split(X, y, random_state=random_state))
    if clf == 'LogisticRegression':
        pipe = sklearn.pipeline.make_pipeline(
            sklearn.preprocessing.StandardScaler(),
            sklearn.linear_model.LogisticRegression(max_iter=10000, n_jobs=n_jobs,
                                                    random_state=random_state)
        )
    elif clf == 'RandomForest':
        pipe = sklearn.pipeline.make_pipeline(
            sklearn.ensemble.RandomForestClassifier(
                random_state=random_state, n_jobs=100 if n_jobs < 0 else n_jobs
            )
        )
    elif clf == 'XGBoost':
        pipe = xgb.XGBClassifier(n_jobs=100 if n_jobs < 0 else n_jobs,
                                 random_state=random_state,
                                 use_label_encoder=False)
        pipe.fit(X_train, y_train, eval_metric='logloss')
    else:
        raise Exception(f'Invalid clf argument: {clf}')

    if not clf == 'XGBoost':
        pipe.fit(X_train, y_train)

    auc = sklearn.metrics.roc_auc_score(
        y_true=y_test, y_score=pipe.predict_proba(X_test)[:, 1])

    return auc


@app.command()
def discrete(feature_set: List[str] = ["I", "II-A", "II-B"]):
    """"Get all features of all discrete networks """
    discrete_ids = [18, 20, 21, 9, 4, 8, 24, 16, 11, 10]
    for i in discrete_ids:
        for f in feature_set:
            try:
                single(network=i, feature_set=f)
            except:
                print(
                    f"COULD NOT EXTRACT FEATURES NETWORK ID {i}, FEATURES {f}")


@app.command()
def single_all_features(network: int,
                        clf: str = 'LogisticRegression',
                        random_state: int = 42,
                        n_jobs: int = -1):
    feature_sets = ["I", "II-A", "II-B"]
    for f in feature_sets:
        single(network=network, feature_set=f)


@app.command()
def single(network: int,
           clf: str = 'LogisticRegression',
           feature_set: str = 'II-A',
           random_state: int = 42,
           n_jobs: int = -1):
    directory = f'/data/s1620444/{network:02}'
    os.makedirs(directory, exist_ok=True)
    filepath_out = os.path.join(directory,
                                'properties',
                                f'{feature_set}_{clf}.float')
    if os.path.isfile(filepath_out):
        return
    auc = predict(directory, feature_set, clf, random_state, n_jobs)
    if auc is not None:
        with open(filepath_out, 'w') as file:
            file.write(str(auc))


@app.command()
def all(network: int = None,

        clf: str = 'LogisticRegression',
        n_jobs: int = -1,
        feature_set: str = 'II-A',
        shuffle: bool = True,
        seed: int = 42):
    if network is None:
        networks = [network for network in np.arange(1, 31)
                    if network not in [15, 17, 26, 27]]
    else:
        networks = [network]
    if shuffle:
        random.seed(seed)
        random.shuffle(networks)
    if n_jobs == -1 or n_jobs > 1:
        ProgressParallel(n_jobs=n_jobs, total=len(networks))(
            delayed(single)(network, clf, feature_set)
            for network in networks
        )
    else:
        for network in tqdm(networks):
            single(network, clf, feature_set)


if __name__ == '__main__':
    app()
