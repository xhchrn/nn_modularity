"""Actual experimentation of lesson test."""


from time import time

import math
import itertools as it
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pylab as plt
from matplotlib.ticker import AutoMinorLocator
import seaborn as sns
from tqdm import tqdm

from src.visualization import run_spectral_cluster, extact_layer_widths
from src.utils import (splitter, load_model2, preprocess_dataset,
                       suppress, all_logging_disabled, extract_weights,
                       multi_combinations_with_replacement)
from src.cnn.convertor import cnn2mlp


def _classification_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)

    metrics = {'acc_overall': np.diag(cm).sum() / cm.sum()}

    metrics.update({f'acc_{number}': acc
                   for number, acc in enumerate(np.diag(cm) / cm.sum(axis=1))})
    
    return metrics


def _evaluate(model, X, y, masks=None):
    y_pred = (model.predict_classes(X, masks)
              if masks is not None
              else model.predict_classes(X))

    return _classification_metrics(y, y_pred)


def _damage_neurons(neurons_in_layers, experiment_model, weights, biases,
                    network_type, inplace=None):
    
    # make sure that the function is called with explicit setting of
    # inplace modification of experiment_model
    assert inplace is True, 'inplace argument should be set to True'
    
    if network_type == 'mlp':
        experiment_weights = deepcopy(weights)
        experiment_biases = deepcopy(biases)

    elif network_type == 'cnn':
        masks = [None] * len(weights)

    for layer_id, _, damaged_neurons, _ in neurons_in_layers:

            if network_type == 'mlp':

                experiment_biases[layer_id-1][damaged_neurons] = 0
                experiment_weights[layer_id-1][:, damaged_neurons] = 0

                experiment_model_layers = (layer for layer in experiment_model.layers
                                           if 'dropout' not in layer._name)
                for layer, new_weights in zip(experiment_model_layers,
                                              zip(experiment_weights, experiment_biases)):
                    layer.set_weights(new_weights)

            # elif network_type == 'cnn':
                # CSC is good for matrix and vector multiplication
                # experiment_weights[layer_id-1] = experiment_weights[layer_id-1].tocsc()

                # experiment_model.set_weights_and_biases(experiment_weights,
                #                                         experiment_biases)
                # masks[layer_id-1] = np.zeros(weights[layer_id -1].shape[1], dtype=bool)
                # masks[layer_id-1][damaged_samples] = True

                # experiment_biases[layer_id-1][damaged_samples] = 0

                # LIL sparse matrix format is faster for updateing values
                # but this is the most expensive part - no way to get around it!
                #if network_type == 'cnn':
                #    experiment_weights[layer_id-1] = experiment_weights[layer_id-1].tolil()

                # Takes to long with sparse csr!
                # And we don't really need it, only the outputs (colums) matters!
                # experiment_weights[layer_id][damaged_samples, :] = 0
                # experiment_weights[layer_id-1][:, damaged_samples] = 0


def _layers_labels_gen(network_type, layer_widths, labels, ignore_layers,
                       to_shuffle=False, fixed=None):
    
    layer_data = zip(splitter(deepcopy(labels), layer_widths), layer_widths[:-1])
    next(layer_data)

    for layer_id, (layer_labels, layer_width) in enumerate(layer_data, start=1):
        
    # for pool max
        if (ignore_layers
            # `layer_id-1` because we set `start=1` for `enumerate`
            and ignore_layers[layer_id-1]):
            
            if verbose:
                print(f'Ignoring layer {layer_id-1}!')
                
            continue
            
        layer_labels = np.array(layer_labels)

        # do not shuffle pruned nodes
        if to_shuffle:

            # Don't shuffle pruned neurons
            non_shuffled_mask = (layer_labels != -1)

            # We preform the same operation of unpacking `fixed_layer_label`
            # multiple times, because I wanted to put all the "fixed" processing
            # in one section.
            if fixed is not None:
                fixed_layer_id, fixed_label = fixed
                if fixed_layer_id == layer_id:
                    
                    assert not (~non_shuffled_mask
                                & (layer_labels == fixed_label)).any()
                    
                    non_shuffled_mask &= (layer_labels != fixed_label)
                 
            layer_labels[non_shuffled_mask] = np.random.permutation(layer_labels[non_shuffled_mask])

        yield layer_id, layer_labels


def _single_damaged_neurons_gen(layers_labels_iterable, verbose=False):

    for layer_id, layer_labels in layers_labels_iterable:

        actual_layer_size = (layer_labels != -1).sum()

        for label in (l for l in np.unique(layer_labels) if l != -1):

            if verbose >= 2:
                print('Layer', layer_id, label, layer_width)

            damaged_neurons = np.nonzero(layer_labels == label)[0]
            
            yield (layer_id, label, damaged_neurons, actual_layer_size)


def _double_conditional_damaged_neurons_gen(network_type, layer_widths, labels, to_shuffle):
    
    assert network_type == 'mlp'

    fixed_iter =  _layers_labels_gen(network_type, layer_widths, labels,
                                     ignore_layers=False, to_shuffle=False)

    for fixed_layer_id, fixed_layer_labels in fixed_iter:
        
        fixed_actual_layer_size = (fixed_layer_labels != -1).sum()

        for fixed_label in (l for l in np.unique(fixed_layer_labels) if l != -1):

            fixed_damaged_neurons = np.nonzero(fixed_layer_labels == fixed_label)[0]

            fixed_gen_tuple = (fixed_layer_id, fixed_label, fixed_damaged_neurons, fixed_actual_layer_size)
            
            shuffled_layers_labels_iterable =  _layers_labels_gen(network_type, layer_widths, labels,
                                                ignore_layers=False, to_shuffle=to_shuffle,
                                                fixed=(fixed_layer_id, fixed_label))
            
            shuffled_damaged_neurons_iterable = _single_damaged_neurons_gen(shuffled_layers_labels_iterable)

            # There is no meaning to same layer-label in double conditional,
            # which it is always shuffled,
            # so we filter these cases out to reduce computation
            filtered_shuffled_damaged_neurons_iterable = ((layer_id, label, damaged_neurons, actual_layer_size)
                                                          for layer_id, label, damaged_neurons, actual_layer_size
                                                          in shuffled_damaged_neurons_iterable
                                                          if not (layer_id == fixed_layer_id
                                                                  and label == fixed_label))
            
            # fixed is second (i.e., first|second)
            yield from it.product(filtered_shuffled_damaged_neurons_iterable,
                                  [fixed_gen_tuple])


def _damaged_neurons_gen(network_type, layer_widths, labels, ignore_layers,
                         to_shuffle=False, n_way=1, n_way_type='joint', verbose=False):
    assert n_way in (1, 2), 'Currently supporting only single and double lesion test.'
    assert n_way_type in ('joint', 'conditional')

    if n_way_type == 'joint':
        
        # if `n_way=2` and `to_shuffle=True`, then we shuffle the cluster labeling once
        # in each of the layers for one trial of brian damage test
        # so we don't need to worry that we might damage the same neurons
        # in the same layer.
        # If the two neuron groups are from the same layer, they won't overlapped,
        # because they are not "sampled" simultaneously.
        layers_labels_iterable = _layers_labels_gen(network_type, layer_widths, labels, ignore_layers,
                                                    to_shuffle)

        damaged_neurons_iterable = _single_damaged_neurons_gen(layers_labels_iterable, verbose)

        yield from it.combinations_with_replacement(damaged_neurons_iterable, n_way)
    
    elif n_way_type == 'conditional':
        assert n_way == 2, 'Conditional p-value works only with double lesion test.'
        yield from _double_conditional_damaged_neurons_gen(network_type, layer_widths, labels, to_shuffle)


def _apply_lesion_trial(X, y, network_type, experiment_model, weights, biases,
                              layer_widths, labels, ignore_layers,
                              to_shuffle=False, n_way=1, n_way_type='joint',
                              verbose=False):
    
    damage_results = []
    
    for neurons_in_layers in _damaged_neurons_gen(network_type, layer_widths, labels, ignore_layers,
                                                  to_shuffle, n_way, n_way_type, verbose):

        _damage_neurons(neurons_in_layers, experiment_model, weights, biases,
                        network_type, inplace=True)
        
        result =_evaluate(experiment_model, X, y)#, masks)

        result['labels_in_layers'] = tuple((layer_id, label) for layer_id, label, _, _ in neurons_in_layers)

        damage_results.append(result)

    return damage_results


def _extract_layer_label_metadata(network_type, layer_widths, labels, ignore_layers):
    
    layer_label_metadata = []

    for neurons_in_layers in _damaged_neurons_gen(network_type, layer_widths, labels, ignore_layers,
                                                  to_shuffle=False, n_way=1):
        
        assert len(neurons_in_layers) == 1
        layer_id, label, neurons, actual_layer_size = neurons_in_layers[0]

        layer_label_metadata.append({'layer': layer_id,
                                     'label': label,
                                     'n_layer_label': len(neurons),
                                     'label_in_layer_proportion': len(neurons) / actual_layer_size})

    return layer_label_metadata


def _flatten_single_damage(damage_results):
    assert all(len(result['labels_in_layers']) == 1 for result in damage_results)
    
    for result in damage_results:
        layer_id, label = result['labels_in_layers'][0]
        del result['labels_in_layers']
        result['layer'] = layer_id
        result['label'] = label

    return damage_results



def _perform_lesion_sub_experiment(dataset_path, run_dir, n_clusters=4,
                                   n_shuffles=200,
                                   with_random=True, model_params=None,
                                   n_way=1, n_way_type='joint',
                                   true_as_random=False,
                                   verbose=False):
   
    if verbose:
        print('Loading data...')

    ds = preprocess_dataset(dataset_path)
    X, y = ds['X_test'], ds['y_test']
    
    run_dir_path = Path(run_dir)
    model_path = str(next(run_dir_path.glob('*-pruned.h5')))
    weight_path = str(next(run_dir_path.glob('*-pruned-weights.pckl')))

    if 'mlp' in model_path.lower():
        network_type = 'mlp'
    elif 'cnn' in model_path.lower():
        network_type = 'cnn'
        assert model_params is not None, ('For CNN network type, '
                                          'the model_param parameter should be given.')
    else:
        raise ValueError('Network type should be expressed explicitly '
                         'either mlp or cnn in run directory files.')

    if verbose:
        print('Running spectral clustering...')
        
    labels, _ = run_spectral_cluster(weight_path,
                                     n_clusters=n_clusters,
                                     with_shuffle=False)

    if verbose:
        print('Loading model and extracting weights...')

    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

    if network_type == 'mlp':
        with suppress(), all_logging_disabled():
            experiment_model = load_model2(model_path)

        weights, biases = extract_weights(experiment_model,
                                          with_bias=True)
        ignore_layers = False

    elif network_type == 'cnn':
        with suppress(), all_logging_disabled():
            experiment_model = cnn2mlp(model_path, model_params,
                                       verbose=verbose)

        weights, biases =  experiment_model.get_weights_and_biases()

        ignore_layers = experiment_model.get_ignore_layers()

    if verbose:
        print('Evaluate original model...')

    evaluation = _evaluate(experiment_model, X, y)   
    
    layer_widths = extact_layer_widths(weights)
    

    if verbose:
        print('Extract metadata...')
    
    metadata = _extract_layer_label_metadata(network_type, layer_widths, labels, ignore_layers)

    if verbose:
        print('Apply lesion trial on the true clustering...')


    true_results = _apply_lesion_trial(X, y, network_type, experiment_model, weights, biases,
                                             layer_widths, labels, ignore_layers,
                                             to_shuffle=true_as_random, n_way=n_way, n_way_type=n_way_type,
                                             verbose=verbose)
    if with_random:

        if verbose:
            print('Apply lesion trial on the random clusterings...')
            progress_iter = tqdm
        else:
            progress_iter = iter

        all_random_results = []
        for _ in progress_iter(range(n_shuffles)):
            random_results = _apply_lesion_trial(X, y, network_type, experiment_model, weights, biases,
                                            layer_widths, labels, ignore_layers,
                                            to_shuffle=True, n_way=n_way, n_way_type=n_way_type,
                                            verbose=verbose)
            
            all_random_results.append(random_results)

    else:
        all_random_results = None

    if n_way == 1:
        true_results = _flatten_single_damage(true_results)

        all_random_results = ([_flatten_single_damage(result) for result in all_random_results]
                              if all_random_results else None)
        
    return true_results, all_random_results, metadata, evaluation


def perform_lesion_experiment(dataset_path, run_dir, n_clusters=4,
                              n_shuffles=100, n_workers=1, n_way=1, n_way_type='joint',
                              with_random=True, model_params=None,
                              true_as_random=False,
                              verbose=False):
    
    if n_workers == 1:
        if verbose:
            print('Single worker!')
            
        return _perform_lesion_sub_experiment(dataset_path, run_dir, n_clusters,
                                         n_shuffles,
                                         with_random, model_params, n_way, n_way_type,
                                         true_as_random,
                                         verbose)
    else:
        raise NotImplementedError('Check CNN Lesion Test Notebook')

