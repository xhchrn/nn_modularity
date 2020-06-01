"""Visualizations for neural network clustering results."""


import warnings
import re
import json
import math
from pprint import pprint
import itertools as it
from pathlib import Path
from collections import Counter
import warnings
from functools import lru_cache

import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from matplotlib.pyplot import cm
from matplotlib.ticker import MaxNLocator
from sklearn.cluster import SpectralClustering
from tqdm import tqdm
from scipy import sparse

from src.spectral_cluster_model import clustering_experiment, weights_to_graph
from src.cnn.extractor import extract_cnn_weights
from src.utils import (suppress, all_logging_disabled,
                       load_weights, get_weights_paths,
                       extract_classification_metrics,
                       enumerate2, splitter,
                       heatmap_fixed)

RANDOM_STATE = 42

__all__ = ['draw_clustered_mlp']


# TODO: make defualt set with None
def run_spectral_cluster(weights_path, with_shuffle=True,
                         n_clusters=4, shuffle_method='layer',
                         n_samples=None, n_workers=None,
                         with_shuffled_ncuts=False,
                         random_state=RANDOM_STATE,
                         ):

    if 'mlp' in str(weights_path):
        named_configs = ['mlp_config']
    elif 'cnn' in str(weights_path):
        named_configs = ['cnn_config']
    else:
        raise ValueError('Either mlp or cnn should be in path to determine the config.')

    config_updates = {'weights_path': weights_path,
                      'with_labels': True,
                      'with_shuffle': with_shuffle,
                      'seed': random_state,
                      'num_clusters': n_clusters,
                      'shuffle_method': shuffle_method,
                      'with_shuffled_ncuts': with_shuffled_ncuts}
    
    if n_samples is not None:
        config_updates['num_samples'] = n_samples
    if n_workers is not None:
        config_updates['n_workers'] = n_workers

    with suppress(), all_logging_disabled():
        experiment_run = clustering_experiment.run(config_updates=config_updates,
                                                   named_configs=named_configs)
    
    metrics = experiment_run.result
    clustering_labels = metrics.pop('labels')
    node_mask = metrics.pop('node_mask')

    metrics.pop('shuffle_method', None)

    labels = np.full(len(node_mask), -1)
    labels[node_mask] = clustering_labels
    
    classification_metrics = extract_classification_metrics(Path(weights_path).parent)
    metrics.update(classification_metrics['unpruned']
                   if 'unpruned' in str(weights_path)
                   else classification_metrics['pruned'])
    
    return labels, metrics


def run_double_spectral_cluster(weight_directory, with_shuffle=True,
                                n_clusters=4, shuffle_method='layer',
                                n_samples=None, n_workers=None,
                                with_shuffled_ncuts=False,
                                random_state=RANDOM_STATE,
                                ):

    weight_paths = get_weights_paths(weight_directory)

    return {is_unpruned: run_spectral_cluster(weight_path, with_shuffle,
                                            n_clusters, shuffle_method,
                                            n_samples, n_workers,
                                            with_shuffled_ncuts,
                                            random_state)
            for is_unpruned, weight_path in weight_paths.items()}


def extact_layer_widths(weights):
    weight_shapes = (layer_weights.shape for layer_weights in weights)
    layer_widths = []
    layer_widths.extend(next(weight_shapes))
    layer_widths.extend(shape[1] for shape in weight_shapes)
    return tuple(layer_widths)


def get_color_mapper(n_clusters):
    color_mapper =  dict(enumerate(iter(cm.rainbow(np.linspace(0, 1, n_clusters)))))
    color_mapper[-1] = 'gray'
    return color_mapper


def set_square_nodes_positions(layer_width, nodes_sorted, space=3):

    side = int(math.sqrt(layer_width))
    assert side ** 2 == layer_width

    offset_x = np.linspace(0, side*space, num=side, dtype=int)
    starting_x = offset_x[-1]
    xs = (np.zeros((side, side)) + offset_x[None, :]).reshape(-1)
    
    center_node = side // 2
    normalized_ys_row = ((np.arange(side) - center_node)
                    / center_node)
    normalized_ys = np.tile(normalized_ys_row[:, None], side).flatten()

    return xs, normalized_ys, starting_x, side


def set_nodes_positions(nodes, layer_widths, clustering_labels,
                        is_first_square=True, dx=50, dy=5, jitter=10):
    """Set postions of nodes of a neural network for networkx drawing."""   

    pos = {}

    labled_nodes_by_layer = splitter(zip(nodes, clustering_labels),
                                     layer_widths)

    layer_data = enumerate(zip(layer_widths, labled_nodes_by_layer))

    starting_x = 0

    # TODO - refactor!
    for layer_index, (layer_width, labled_nodes) in layer_data:

        nodes, labels = zip(*labled_nodes)

        nodes_sorted = [node for _,node in sorted(zip(labels ,nodes))]

        # first layer is the input (image)
        # so let's draw it as a square!
        if is_first_square and layer_index == 0:
            nodes_sorted = nodes

            (xs, normalized_ys,
             shift_x, side) = set_square_nodes_positions(layer_width, nodes_sorted)
            starting_x += shift_x
            height = dy * shift_x

        else:
            nodes_sorted = [node for _,node in sorted(zip(labels ,nodes))]

            starting_x += dx

            xs = np.full(layer_width, starting_x, dtype=float)
            xs += 2*jitter * np.random.random(layer_width) - jitter
            xs = xs.round().astype(int)

            center_node = layer_width // 2

            normalized_ys = ((np.arange(layer_width) - center_node)
                            / center_node)
            height = dy * layer_width

        ys = normalized_ys * height
        ys = ys.round().astype(int)

        pos.update({node: (x, y) for node, (x, y) in zip(nodes_sorted, zip(xs, ys))})

    return pos


def draw_metrics(metrics, ax, ndigits=5):
    """Plot spectral clustering metrics as a table."""

    metrics_series = pd.Series(metrics)
    ax.table(cellText=metrics_series.values[:, None].round(ndigits),
             colWidths = [0.25],
             rowLabels=metrics_series.index,
             colLabels=[''],
             cellLoc = 'center', rowLoc = 'center',
             loc='bottom')


def draw_clustered_mlp(weights_path,
                       clustering_result,
                       n_clusters=4,
                       is_first_square=True,
                       ax=None):
    """Draw MLP with its spectral clustering."""

    weights = load_weights(weights_path)
    layer_widths = extact_layer_widths(weights)

    labels, metrics = clustering_result

    G = nx.from_scipy_sparse_matrix(weights_to_graph(weights))

    pos = set_nodes_positions(G.nodes, layer_widths, labels, is_first_square)
    
    color_mapper = get_color_mapper(n_clusters)

    color_map = [color_mapper[label] for label in labels]

    if ax is None:
        _, ax = plt.subplots(1)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        nx.draw(G, pos=pos,
                node_color=color_map,
                width=0, node_size=10,
                ax=ax)

    draw_metrics(metrics, ax)
        
    return ax, labels, metrics


def nodify(*args):
    return '-'.join(str(arg) for arg in args)



def build_cluster_graph(weights_path,
                        clustering_result,
                        normalize_in_out=True):

    labels, _ = clustering_result

    weights = load_weights(weights_path)
    layer_widths = extact_layer_widths(weights)
    
    
    G = nx.DiGraph()

    (label_by_layer,
     current_label_by_layer,
     next_label_by_layer) = it.tee(splitter(labels, layer_widths), 3)

    next_label_by_layer = it.islice(next_label_by_layer, 1,  None)

    for layer_index, layer_labels in enumerate(label_by_layer):
        unique_labels = sorted(label for label in np.unique(layer_labels) if label != -1)
        for label in unique_labels:
            node_name = nodify(layer_index, label)
            G.add_node(node_name)

    edges = {}

    for layer_index, (current_labels, next_labels, layer_weights) in enumerate(zip(current_label_by_layer,
                                                                next_label_by_layer,
                                                                weights)):

        label_edges = it.product((label for label in np.unique(current_labels) if label != -1),
                                 (label for label in np.unique(next_labels) if label != -1))

        for current_label, next_label in label_edges:

            current_mask = (current_label == current_labels)
            next_mask = (next_label == next_labels)

            between_weights = layer_weights[current_mask, :][:, next_mask]

            if normalize_in_out:
                n_weight_in, n_weight_out = between_weights.shape
                n_weights = n_weight_in * n_weight_out
                normalization_factor = n_weights
            else:
                normalization_factor = 1

            edge_weight = np.abs(between_weights).sum() / normalization_factor

            current_node = nodify(layer_index, current_label)
            next_node = nodify(layer_index + 1, next_label)

            edges[current_node, next_node] = edge_weight

    for nodes, weight in edges.items():
        G.add_edge(*nodes, weight=weight)

    return G


def draw_cluster_by_layer(weights_path,
                          clustering_result,
                          n_clusters=4,
                          with_text=False,
                          size_factor=4,
                          width_factor=30,
                          ax=None):
    
    G = build_cluster_graph(weights_path,
                            clustering_result)

    labels, _ = clustering_result

    weights = load_weights(weights_path)
    layer_widths = extact_layer_widths(weights)

    color_mapper = get_color_mapper(n_clusters)

    node_size = {}

    (label_by_layer,
     current_label_by_layer,
     next_label_by_layer) = it.tee(splitter(labels, layer_widths), 3)

    next_label_by_layer = it.islice(next_label_by_layer, 1,  None)

    for layer_index, layer_labels in enumerate(label_by_layer):
        unique_labels = sorted(label for label in np.unique(layer_labels) if label != -1)
        for label in unique_labels:
            node_name = nodify(layer_index, label)
            node_size[node_name] = (layer_labels == label).sum()

    pos = nx.drawing.nx_agraph.graphviz_layout(G, prog='dot')
    width = [G[u][v]['weight'] * width_factor  for u,v in G.edges()]
    node_color = [color_mapper[int(v.split('-')[1])] for v in G.nodes()]
    node_size = [node_size[v] * size_factor for v in G.nodes()]

    if ax is None:
        _, ax = plt.subplots(1)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        
        nx.draw(G, pos,
                with_labels=True,
                node_color=node_color,
                node_size=node_size,
                # font_color='white',
                width=width,
                ax=ax)

    if with_text:
        pprint(edges)

    return ax


def plot_eigenvalues_old(weights_path, n_eigenvalues=None, ax=None, **kwargs):
    warnings.warn('deprecated', DeprecationWarning)

    loaded_weights = load_weights(weights_path)
 
    G = nx.from_scipy_sparse_matrix(weights_to_graph(loaded_weights))
    G_nn = G.subgraph(max(nx.connected_components(G), key=len)) 
    assert nx.is_connected(G_nn)

    nrom_laplacian_matrics = nx.normalized_laplacian_matrix(G_nn)
    eigen_values = np.sort(np.linalg.eigvals(nrom_laplacian_matrics.A))

    if n_eigenvalues == None:
        start, end = 0, len(G_nn)
    elif isinstance(n_eigenvalues, int):
        start, end = 0, n_eigenvalues
    elif isinstance(n_eigenvalues, tuple):
        start, end = n_eigenvalues
    else:
        raise TypeError('n_eigenvalues should be either None or int or tuple or slice.')
    
    eigen_values = eigen_values[start:end]
    
    if ax is None:
        _, ax = plt.subplots(1)

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    if 'linestyle' not in kwargs:
        kwargs['linestyle'] = 'none'
        kwargs['marker'] = '*'
        kwargs['markersize'] = 5
    
    return ax.plot(range(start + 1, end + 1),
                   eigen_values,
                   **kwargs)


def plot_eigenvalues(weights_path, n_eigenvalues=None, ax=None, **kwargs):
    
    weights = load_weights(weights_path)
    
    if 'cnn' in str(weights_path):
        weights, _ = extract_cnn_weights(weights, with_avg=True) #(max_weight_convention=='one_on_n'))
        print([w.shape for w in weights])
    
    
    # TODO: take simpler solution from delete_isolated_ccs_refactored
    adj_mat = weights_to_graph(weights)

    _, components = sparse.csgraph.connected_components(adj_mat)

    most_common_component_counts = Counter(components).most_common(2)
    main_component_id = most_common_component_counts[0][0]
    assert (len(most_common_component_counts) == 1
            or most_common_component_counts[1][1] == 1)
    
    main_component_mask = (components == main_component_id)

    selected_adj_mat = adj_mat[main_component_mask, :][:, main_component_mask]
    
    nrom_laplacian_matrix = sparse.csgraph.laplacian(selected_adj_mat, normed=True)
    
    
    if n_eigenvalues == None:
        start, end = 0, selected_adj_mat.shape[0] - 2
    elif isinstance(n_eigenvalues, int):
        start, end = 0, n_eigenvalues
    elif isinstance(n_eigenvalues, tuple):
        start, end = n_eigenvalues
    else:
        raise TypeError('n_eigenvalues should be either None or int or tuple or slice.')
    """
    eigen_values, _ = sparse.linalg.eigs(nrom_laplacian_matrix, k=end,
                                         which='SM')
    """

    sigma = 1

    OP = nrom_laplacian_matrix - sigma*sparse.eye(nrom_laplacian_matrix.shape[0])
    OPinv = sparse.linalg.LinearOperator(matvec=lambda v: sparse.linalg.minres(OP, v, tol=1e-5)[0],
                                         shape=nrom_laplacian_matrix.shape,
                                         dtype=nrom_laplacian_matrix.dtype)
    eigen_values, _ = sparse.linalg.eigsh(nrom_laplacian_matrix, sigma=sigma,
                                          k=end, which='LM', tol=1e-5, OPinv=OPinv)
    
    eigen_values = np.sort(eigen_values)
    
    eigen_values = eigen_values[start:end]
    
    if ax is None:
        _, ax = plt.subplots(1)

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    if 'linestyle' not in kwargs:
        kwargs['linestyle'] = 'none'
        kwargs['marker'] = '*'
        kwargs['markersize'] = 5
    
    return ax.plot(range(start + 1, end + 1),
                   eigen_values,
                   **kwargs)


def plot_eigenvalue_report(weight_directory,
                           unpruned_n_eigenvalues=None, pruned_n_eigenvalues=None,
                           figsize=(10, 5)):
    weight_paths = get_weights_paths(weight_directory)

    is_slice = (unpruned_n_eigenvalues is not None
                or pruned_n_eigenvalues is not None)
    
    n_rows = 2 if is_slice else 1
        
    _, axes = plt.subplots(n_rows, 2, squeeze=False, figsize=figsize)

    axes[0][0].set_title('Unpruned')
    plot_eigenvalues(weight_paths[True],
                     ax=axes[0][0])

    if is_slice:
        plot_eigenvalues(weight_paths[True], unpruned_n_eigenvalues,
                         ax=axes[1][0])

    axes[0][1].set_title('Pruned')
    plot_eigenvalues(weight_paths[False],
                     ax=axes[0][1])

    if is_slice:
        plot_eigenvalues(weight_paths[False], pruned_n_eigenvalues,
                         ax=axes[1][1])


def draw_mlp_clustering_report(weight_directory,
                               double_clustering_results,
                               n_cluster=4,
                               title=None, figsize=(20, 30)):

    weight_paths = get_weights_paths(weight_directory)

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    if title is not None:
        fig.suptitle(title)

    axes[0][0].set_title('Unpruned')
    draw_clustered_mlp(weight_paths[True],  # True represents **un**pruned
                       double_clustering_results[True],
                       n_clusters=n_cluster,
                       ax=axes[0][0])

    draw_cluster_by_layer(weight_paths[True],
                          double_clustering_results[True],
                          n_clusters=n_cluster,
                          ax=axes[1][0])

    axes[0][1].set_title('Pruned')
    draw_clustered_mlp(weight_paths[False],
                       double_clustering_results[False],
                       n_clusters=n_cluster,
                       ax=axes[0][1])

    draw_cluster_by_layer(weight_paths[False],
                          double_clustering_results[False],
                          n_clusters=n_cluster,
                          ax=axes[1][1])


def plot_learning_curve(weight_directory, n_clusters=4, with_shuffle=False,
                        shuffle_method='layer', start=5, step=5, 
                        primary_y=('ncut',),
                        secondary_y=('percentile', 'train_loss', 'test_loss', 'ave_in_out'),
                        with_tqdm=False,
                        ax=None):
    
    
    progress_iter = tqdm if with_tqdm else iter

    weight_directory_path = Path(weight_directory)
    
    results = []

    for type_ in ('unpruned', 'pruned'):

        weight_paths = list(sorted(weight_directory_path.glob(f'*-{type_}*.ckpt')))[start-1::step]
              
        _, type_results = zip(*(run_spectral_cluster(weight_path,
                                                     n_clusters=n_clusters,
                                                     with_shuffle=with_shuffle,
                                                     shuffle_method=shuffle_method)
                                for weight_path in progress_iter(weight_paths)))

        for epoch, result in enumerate2(type_results, start=start, step=step):
            result['is_pruned'] = (type_ == 'pruned')
            result['epoch'] = epoch
            
            # The result from `run_spectral_cluster` comes with the
            # loss and accuracy metrics for the *final* model
            # because it gets them from the `metrics.json` file.
            # So for all the checkpoint models of `unpruned` we have
            # the same metrics, as well as for `pruned`.
            # Therefore we remove them right now, and later 
            # (see `evaluation_metrics` in this function)
            # we will extract them from `cout.txt`.
            del (result['train_loss'], result['train_acc'],
                 result['test_loss'], result['test_acc'])

        results.extend(type_results)

    df = pd.DataFrame(results)
    
    df.loc[df['is_pruned'], 'epoch'] += df[~df['is_pruned']]['epoch'].iloc[-1]
    df = df.set_index('epoch')
        
    metrics_file = (weight_directory_path / 'metrics.json')
    raw_metrics = json.loads(metrics_file.read_text())

    
    # TODO: refactor me!
    # The parsering of the metrics.json file can be done more elegantly
    # and taken out to a separated function
    
    evaluation_metrics = []
    
    real_epoch_start = start
    
    for type_ in ('pruned', 'unpruned'):

        raw_evaluation_metics = it.islice(zip(raw_metrics[type_]['loss'],
                                    raw_metrics[type_]['acc'],
                                    raw_metrics[type_]['val_loss'],
                                    raw_metrics[type_]['val_acc']),
                                       start-1, None, step)

              
        evaluation_metrics += [{'epoch': epoch,
                                'train_loss': float(train_loss), 'train_acc': float(train_acc),
                                'test_loss': float(test_loss), 'test_acc': float(test_acc)}
                                 for epoch, (train_loss, train_acc, test_loss, test_acc)
                                 in enumerate2(raw_evaluation_metics,
                                               start=real_epoch_start, step=step)]
        
        real_epoch_start += step * len(evaluation_metrics)

    ####
    
    evaluation_metrics_df = pd.DataFrame(evaluation_metrics).set_index('epoch')

    df = pd.concat([df, evaluation_metrics_df], axis=1)
    
    if primary_y is None:
        primary_y = list(df.columns)
    else:
        primary_y = [col for col in primary_y if col in df.columns]

    if secondary_y is not None:
        primary_y = [col for col in primary_y if col not in secondary_y]
        secondary_y = [col for col in secondary_y if col in df.columns]
    else:
        secondary_y = []

    all_y = primary_y + secondary_y

    if ax is None:
        _, ax = plt.subplots(1)

    df[all_y].plot(secondary_y=secondary_y, ax=ax)

    split = (df.index[~df['is_pruned']][-1] + df.index[df['is_pruned']][0]) / 2
    ax.axvline(split, color='k', linestyle='--')

    return ax, df


@lru_cache(maxsize=None)
def _compute_weighted_dist(G, start, end):
    start_layer, _ = start.split('-')
    start_layer = int(start_layer)
    
    end_layer, _ = end.split('-')
    end_layer = int(end_layer)
    
    if start_layer >= end_layer:
        return 0
    
    elif start_layer + 1 == end_layer:
        return G[start][end]['weight']
    
    else:
        next_layer = str(start_layer + 1)
        next_nodes = (node for node in G.nodes
                      if node.startswith(next_layer))
        return sum(_compute_weighted_dist(G, start, next_node)
                   + _compute_weighted_dist(G, next_node, end)
                   for next_node in next_nodes)
    

def build_weighted_dist_mat(model_path, clustering_result,
                            normalize_in_out=False):
    
    weight_path = get_weights_paths(model_path)[False] # pruned

    G = build_cluster_graph(weight_path,
                            clustering_result[False], # pruned
                            normalize_in_out=normalize_in_out)

    df = pd.DataFrame([{'start': start,
                        'end': end,
                        'dist': _compute_weighted_dist(G, start, end)}
                        for start, end in it.combinations(G.nodes, 2)])

    df = df[df != 0].dropna()

    # The distance is normalized to [0, 1] inside the paths between two specific layers
    # The max weighted sitance is one.
    df['layers'] = df.apply(lambda r: r['start'].split('-')[0] + '-' + r['end'].split('-')[0], axis=1)
    df['normalized_dist'] = df['dist'] / df.groupby('layers')['dist'].transform('max')

    mat = df.pivot('start', 'end', 'normalized_dist')

    return mat


def plot_weighted_dist_mat(mat, figsize=(13, 10), ax=None):

    if ax is None:
        _, ax = plt.subplots(1, figsize=figsize)
        
    ax = heatmap_fixed(mat,
                       ax=ax)
    plt.xticks(rotation=90) 
    
    return ax


def draw_ow_weight_dependency_graph(one_way_weighted_dist_mat, thresold=0.99, ax=None):
    unstacked_one_way_weighted_dist_mat = one_way_weighted_dist_mat.unstack()
    dependency_edges = list(unstacked_one_way_weighted_dist_mat[unstacked_one_way_weighted_dist_mat > thresold]
                            .index
                            .values)
    
    # Remove edges that contains input and output nodes (layer 0 and 5)
    dependency_edges = [sorted(edge) for edge in dependency_edges
               if not ({edge[0][0], edge[1][0]} & {'0', '5'})]
    
    if ax is None:
        _, ax = plt.subplots(1)

    G = nx.DiGraph(dependency_edges)

    pos = nx.drawing.nx_agraph.graphviz_layout(G, prog='dot')
     
    nx.draw(G, pos,
            with_labels=True,
            font_weight='bold',
            font_size=25,
            node_size=50,
            # font_color='white',
            width=2,
            ax=ax
           )
    
    return ax;