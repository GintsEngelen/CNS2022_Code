import sys

from sklearn.model_selection import train_test_split

sys.path = sorted(sys.path, key=lambda s:'envs' not in s)
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["MODIN_ENGINE"] = "dask"

import time
import random as rn
import numpy as np
from sklearn.metrics import  classification_report
import pandas as pd
import json
import cudf
from cuml.dask.ensemble import RandomForestClassifier as cuRF
import collections
import dask_cudf
from dask_cuda import LocalCUDACluster
from dask.distributed import Client
from helpers import load_data_database
from glob import glob
from feature_engine.selection import DropCorrelatedFeatures


seed = 1 # [1] 2017, 2018 orig, 2018 fixed [123] for 2017 fixed
np.random.seed(seed)
rn.seed(seed)
year = 2018
old = False
ngpus = 2


def makehash():
	return collections.defaultdict(makehash)

def sort_importances(unsorted):
    benchmark = unsorted["benchmark"]
    res = makehash()

    for features, values in unsorted.items():
        if features == 'benchmark':
            continue
        for label, results in values.items():
            if label.isnumeric():
                prec = results['precision']
                recall = results['recall']
                f1 = results['f1-score']

                benchmark_prec = benchmark[label]['precision']
                benchmark_recall = benchmark[label]['recall']
                benchmark_f1 = benchmark[label]['f1-score']

                diff_prec = prec - benchmark_prec
                diff_recall = recall - benchmark_recall
                diff_f1 = f1 - benchmark_f1

                res[label]['precision'][features] = diff_prec
                res[label]['recall'][features] = diff_recall
                res[label]['f1-score'][features] = diff_f1

    # sort dictionary
    sorted_feature_importance = makehash()

    for label, vals in res.items():
        for metric, metric_results in vals.items():
            a = metric_results.items()
            sort_orders = sorted(a, key=lambda x: x[1])
            sorted_feature_importance[label][metric] = sort_orders

    filename = f"{year}_feature_importance_rf_sorted.json"

    with open(filename, "w") as outfile:
        json.dump(sorted_feature_importance, outfile, indent=4)


def drop_col_feat_imp(X_full, y_full):
    #X_full[X_full.select_dtypes(np.float64).columns] = X_full.select_dtypes(np.float64).astype(np.float32)
    X_full_corr = X_full.astype('float32')
    y_full = y_full.astype('int32')

    tr = DropCorrelatedFeatures(variables=None, method='pearson', threshold=0.9)

    print("Calculating correlated features:")
    X_full = tr.fit_transform(X_full_corr)
    print(f'Correlated Feature Sets: {str(tr.correlated_feature_sets_)}')
    print(f'Dropped features: {str(tr.features_to_drop_)}')

    X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, random_state=seed, stratify=y_full, shuffle=True)

    # due to not yet implemented feature - sharding causes issues with a mysterious issue with the classifier and the
    # labels not being in order
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)


    X_train = dask_cudf.from_cudf(cudf.from_pandas(X_train), npartitions=ngpus).persist()
    X_test = dask_cudf.from_cudf(cudf.from_pandas(X_test), npartitions=ngpus).persist()
    y_train = dask_cudf.from_cudf(cudf.from_pandas(y_train), npartitions=ngpus).persist()
    y_test = dask_cudf.from_cudf(cudf.from_pandas(y_test), npartitions=ngpus).persist()

    # list for storing feature importances
    importances = {}
    rf = cuRF(max_depth=30, n_estimators=100, random_state=seed, verbose=True, n_streams=25)
    st = time.time()
    print(f"Starting base- {st}")
    rf.fit(X_train, y_train, convert_dtype=True)
    y_pred = rf.predict(X_test)
    importances['benchmark'] = classification_report(y_test.compute().to_numpy(), y_pred.compute().to_numpy(), output_dict=True)
    print(f'Elapsed time time: {time.time()-st}')

    # iterating over all columns and storing feature importance (difference between benchmark and new model)
    for i, col in enumerate(X_train.columns):
        print(f"Doing col: {col} [{i}/{len(X_train.columns)}]")
        model_clone = cuRF(max_depth=30, n_estimators=100, random_state=seed, verbose=True, n_streams=25)
        model_clone.fit(X_train.drop(col, axis=1), y_train, convert_dtype=True)
        model_pred = model_clone.predict(X_test.drop(col, axis=1))
        importances[col] = classification_report(y_test.compute().to_numpy(), model_pred.compute().to_numpy(), output_dict=True)
        print(f'[{year}] - Finished col {col}. Elapsed time time: {time.time() - st}')

    print("Saving importance feature dict")
    if old:
        filename = f'{year}_old_feature_importance_rf.json'
    else:
        filename = f'{year}_new_feature_importance_rf.json'

    with open(filename, 'w') as fp:
        json.dump(importances, fp, indent=4)
        fp.write(f'Correlated Feature Sets: {str(tr.correlated_feature_sets_)}')
        fp.write(f'Dropped features: {str(tr.features_to_drop_)}')

    return importances


def load_from_local(folder_path):
    files = glob(folder_path + "/*.csv")
    csv_dataframes = []
    for file in files:
        print(f"-- Reading in {file}")
        df = pd.read_csv(file)
        print(df.columns)
        df.columns = df.columns.str.lstrip(" ")
        df.drop(['id', 'Flow ID', 'Src IP', 'Src Port', 'Dst IP', 'Timestamp', 'Attempted Category'], axis=1, inplace=True)
        df = df.replace('Infinity', np.nan)
        df = df.replace(np.inf, np.nan)
        df = df.dropna()

        for column in df.columns:
            if column != 'Label':
                df[column] = pd.to_numeric(df[column], errors='coerce', downcast="float")

        csv_dataframes.extend([df])

    df = pd.concat(csv_dataframes, ignore_index=True)
    labels = df['Label'].astype('category')
    y = pd.Series(labels.cat.codes)
    train = df.drop(['Label'], axis=1)

    with open(f'{folder_path}/label_mapping.txt', 'w') as f:
        f.write(str(dict(enumerate(labels.cat.categories))))


    return train, y


if __name__ == '__main__':
    # Create a Dask Cluster with one worker per GPU
    cluster = LocalCUDACluster()
    client = Client(cluster)

    file_path = #

    training_featuresdf, labeldf = load_from_local(file_path)

    importances = drop_col_feat_imp(training_featuresdf, labeldf)
    sort_importances(importances)