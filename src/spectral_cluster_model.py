"""Performing spectral clustering on neural networks and calculating its p-values."""


import numpy as np
import math
#from multiprocessing import Pool
from pathos.multiprocessing import ProcessPool
import scipy.sparse as sparse
from sklearn.cluster import SpectralClustering
from sklearn.neighbors.kde import KernelDensity
import itertools as it
import copy
import pickle

import time

from sacred import Experiment
from sacred.observers import FileStorageObserver

from collections import Counter, deque

# import ipdb

from src.utils import splitter, load_weights, compute_pvalue
from src.cnn import CNN_MODEL_PARAMS
from src.cnn.extractor import extract_cnn_weights


SHUFFLE_METHODS = ['layer',
                    'layer_nonzero',
                    'layer_nonzero_distribution',
                    'layer_all_distribution']


# set up some sacred stuff
clustering_experiment = Experiment('cluster_model')
clustering_experiment.observers.append((FileStorageObserver.create('clustering_runs')))


@clustering_experiment.config
def my_config():
    weights_path = "training_runs_dir/10/pruned.pckl"
    num_clusters = 4
    eigen_solver = 'amg'
    assign_labels = 'kmeans'
    epsilon = 1e-8
    delete_isolated_ccs_bool = True
    with_labels = False
    with_shuffle = True
    shuffle_method = 'layer'
    # with different number of worker, the statistics might be slightly different
    # maybe because different order of calls to the random generator?
    # but the statistics are the same for given `seed` and `n_workers`
    n_workers = 10
    is_testing = False
    with_shuffled_ncuts = False



@clustering_experiment.named_config
def cnn_config():
    network_type = 'cnn'
    max_weight_convention = 'one_on_n' # or 'all_one'
    input_shape = (28, 28, 1)
    conv_layers = CNN_MODEL_PARAMS['conv'] 
    fc_layer_widths = CNN_MODEL_PARAMS['dense']
    shuffle_smaller_model = False
    num_samples = 120
    n_workers = 30
    as_sparse = True

@clustering_experiment.named_config
def mlp_config():
    network_type = 'mlp'
    shuffle_smaller_model = True 
    num_samples = 200
    as_sparse = False


def mlp_tup_to_int(tup, layer_widths):
    # tuple represents (layer, neuron_in_layer). int goes from first
    # neuron of first layer to last neuron of last layer.
    # both elements of tup are zero-indexed.
    layer, node = tup
    accum = 0
    for (i, width) in enumerate(layer_widths):
        if i == layer:
            accum += node
            break
        else:
            accum += width
    return accum


def mlp_int_to_tup(num, layer_widths):
    # inverse of mlp_tup_to_int
    accum = num
    # have counter that starts off at num, subtract stuff off at
    # each layer
    for (l, width) in enumerate(layer_widths):
        accum -= width
        if accum < 0:
            return l, accum + width
    # this should never happen
    return None


def cnn_tup_to_int(tup, layer_shapes):
    # tuple represents (layer, num_down, num_across, channel). we treat the
    # output of fully connected layers as having one channel and being purely
    # horizontal.
    layer, down, across, channel = tup

    accum = 0
    for (l, shape) in layer_shapes:
        if l == layer:
            accum += down * layer_shapes[1] * layer_shapes[2]
            accum += across * layer_shapes[2]
            accum += channel
            break
        else:
            accum += np.product(shape)
    return accum


# TODO: check that these functions are inverses
def cnn_int_to_tup(num, layer_shapes):
    '''
    inverse of cnn_tup_to_int (in the first argument)
    '''
    accum = num
    for (l, shape) in enumerate(layer_shapes):
        if accum < np.product(shape):
            height = shape[0]
            width = shape[1]
            num_channels = shape[2]
            row = int(accum / (width * num_channels))
            accum -= height * width * num_channels
            column = int(accum / num_channels)
            accum -= column * num_channels
            channel = accum
            return row, column, channel
        else:
            accum -= np.product(shape)
    # this should never happen
    return None
            

@clustering_experiment.capture
def layer_info_to_layer_shapes(output_shape, input_shape, conv_layers,
                               fc_layer_widths):
    '''
    take in the input shape, a list of dicts specifying info about the 
    convolutional layers, a list of dicts with the widths of fully connected
    layers, and the output shape, and returns an array specifying the shape of 
    each layer of the network
    '''
    layer_shapes = [input_shape]

    for conv_dict in conv_layers:
        # after normal convolutional layer, change the number of channels, but
        # keep the width and height (due to SAME padding)
        filters = conv_dict['filters']
        prev_shape = layer_shapes[-1]
        layer_shapes.append((prev_shape[0], prev_shape[1], filters))
        if conv_dict['max_pool_after']:
            k_size = conv_dict['max_pool_size']
            stride = conv_dict['max_pool_size']  # conv_dict['max_pool_stride']
            padding = conv_dict['max_pool_padding']
            # calculations here are from https://www.corvil.com/kb/what-is-the-difference-between-same-and-valid-padding-in-tf-nn-max-pool-of-tensorflow
            if padding == 'same':
                new_height = math.ceil(prev_shape[0] / stride[0])
                new_width = math.ceil(prev_shape[1] / stride[1])
            elif padding == 'valid':
                new_height = math.ceil((prev_shape[0] - k_size[0] + 1)
                                       / stride[0])
                new_width = math.ceil((prev_shape[1] - k_size[1] + 1)
                                      / stride[1])
            else:
                raise ValueError("max_pool_padding should be same or valid, but instead is " + padding)
            layer_shapes.append((new_height, new_width, prev_shape[0]))

    for width in fc_layer_widths:
        layer_shapes.append((1, width, 1))

    layer_shapes.append(output_shape)

    return layer_shapes

def weights_to_layer_widths(weights_array):
    '''
    take in an array of weight matrices, and return how wide each layer of the
    network is
    '''
    layer_widths = []
    for weight_mat in weights_array:
        layer_widths.append(weight_mat.shape[0])
    final_width = weights_array[-1].shape[1]
    layer_widths.append(final_width)
    return layer_widths


@clustering_experiment.capture
def cnn_layers_to_weights_array(layer_shapes, weight_tensors, conv_layers):
    '''
    take in an array of layer shapes, information about the convolutional 
    layers, and an array of weight tensors, and return an array of weight
    matrices for the 'unrolled' neural network
    '''
    # TODO: assert that I'm doing shapes right
    weight_matrices = []
    layer_shape_stack = deque(layer_shapes)
    for i, conv_dict in enumerate(conv_layers):
        next_shape = layer_shape_stack.popleft()
        conv_tensor = weight_tensors[i]
        if conv_dict['max_pool_after']:
            _ = layer_shape_stack.popleft()
        weight_matrices += conv_layer_to_weight_mats(next_shape, conv_dict,
                                                     conv_tensor)
        for w_mat in weight_matrices:
            assert len(w_mat.shape) == 2, f"conv_layer_to_weight_mats should have output rank 2 tensors, but actually output something with shape {w_mat.shape}"
    for j in range(len(conv_layers), len(weight_tensors)):
        assert len(weight_tensors[j].shape) == 2, f"in cnn_layers_to_weights_array, should be adding only rank 2 tensors to weight_matrices array, but instead added something with shape {weight_tensors[j].shape}"
        weight_matrices.append(weight_tensors[j])
    return weight_matrices

# TODO: use my code instead this function
@clustering_experiment.capture
def conv_layer_to_weight_mats(in_layer_shape, conv_dict, conv_tensor,
                              max_weight_convention):
    '''
    take in the shape of the incoming layer, a dict representing info about the
    conv operation, and the weight tensor of the convolution, and return an 
    array of sparse weight matrices representing the operation. the array should
    have a single element if layer_dict['max_pool_after']==False, but should have 
    two (one representing the action of the max pooling) otherwise.
    for max pooling, we linearise by connecting the maxed neurons to everything
    in their receptive field. if max_weight_convention=='all_one', all the 
    weights are one, otherwise if max_weight_convention=='one_on_n', the weights
    are all one divided by the receptive field size
    '''
    # TODO: see if vectorisation will work
    kernel_height, kernel_width, n_chan_in, n_chan_out = conv_tensor.shape
    in_height = in_layer_shape[0]
    in_width = in_layer_shape[1]
    assert (kernel_height, kernel_width) == tuple(conv_dict['kernel_size']), f"weight tensor info doesn't match conv layer dict info - kernel size from conv_tensor.shape is {(kernel_height, kernel_width)}, but conv_dict says it's {conv_dict['kernel_size']}"
    assert n_chan_out == conv_dict['filters'], f"weight tensor info doesn't match conv layer dict info: weight tensor says num channels out is {n_chan_out}, conv dict says it's {conv_dict['filters']}"
    assert in_layer_shape[2] == n_chan_in, f"weight tensor info doesn't match previous layer shape: weight tensor says it's {n_chan_in}, prev layer says it's {in_layer_shape[2]}"

    kernel_height_centre = int((kernel_height - 1) / 2)
    kernel_width_centre = int((kernel_width - 1) / 2)

    in_layer_size = np.product(in_layer_shape)
    out_layer_shape = (in_height, in_width, n_chan_out)
    out_layer_size = np.product(out_layer_shape)

    conv_weight_matrix = np.zeros((in_layer_size, out_layer_size))

    # THIS WORKS ONLY FOR SAME and not for VALID!!!
    for i in range(in_height):
        for j in range(in_width):
            for c_out in range(n_chan_out):
                out_int = cnn_layer_tup_to_int((i,j,c_out), out_layer_shape)
                for n in range(kernel_height):
                    for m in range(kernel_width):
                        for c_in in range(n_chan_in):
                            weight = conv_tensor[n][m][c_in][c_out]
                            h_in = i + n - kernel_height_centre
                            w_in = j + m - kernel_width_centre
                            in_bounds_check = (h_in in range(in_height)
                                               and w_in in range(in_width))
                            if in_bounds_check:
                                in_int = cnn_layer_tup_to_int((h_in, w_in,
                                                               c_in),
                                                              in_layer_shape)
                                conv_weight_matrix[in_int][out_int] = weight

    weights_array = [conv_weight_matrix]

    if conv_dict['max_pool_after']:
        k_height, k_width = conv_dict['max_pool_size']
        stride = conv_dict['max_pool_size']  # conv_dict['max_pool_stride']
        padding = conv_dict['max_pool_padding']

        if max_weight_convention == 'all_one':
            max_weight = 1
        elif max_weight_convention == 'one_on_n':
            max_weight = 1 / (k_height * k_width)
        else:
            raise ValueError("max_weight_convention must be 'one_on_n' or 'all_one', is instead" + max_weight_convention)

        # This code works on valid, I tested it
        # But if input_side is divisible by stride, it is the same
        if padding == 'valid':
            maxed_height = math.ceil(in_height / stride[0])
            maxed_width = math.ceil(in_width / stride[1])
            maxed_shape = (maxed_height, maxed_width, n_chan_out)
            maxed_size = np.product(maxed_shape)
            max_matrix = np.zeros((out_layer_size, maxed_size))

            k_height_centre = int((k_height - 1) / 2)
            k_width_centre = int((k_width - 1) / 2)
            
            for i in range(maxed_height):
                for j in range(maxed_width):
                    for c in range(n_chan_out):
                        max_int = cnn_layer_tup_to_int((i,j,c), maxed_shape)
                        for n in range(k_height):
                            for m in range(k_width):
                                h_in = stride[0] * i + n - k_height_centre
                                w_in = stride[1] * j + m - k_width_centre
                                in_bounds_check = (h_in in range(in_height)
                                                   and
                                                   w_in in range(in_width))
                                if in_bounds_check:
                                    out_int = cnn_layer_tup_to_int((h_in, w_in,
                                                                    c),
                                                                   out_layer_shape)
                                    max_matrix[out_int][max_int] = max_weight

        # originally, this was 'valid`, but I don't know what it is
        # this code reaise an IndexError
        elif padding == 'same':
            raise NotImplementedError

            maxed_height = math.ceil((in_height - k_height + 1) / stride[0])
            maxed_width = math.ceil((in_width - k_width + 1) / stride[1])
            maxed_shape = (maxed_height, maxed_width, n_chan_out)
            maxed_size = np.product(maxed_shape)
            max_matrix = np.zeros((out_layer_size, maxed_size))

            for i in range(maxed_height):
                for j in range(maxed_width):
                    for c in range(n_chan_out):
                        max_int = cnn_layer_tup_to_int((i,j,c), maxed_shape)
                        for n in range(kernel_height):
                            for m in range(kernel_width):
                                h_in = stride[0] * i + n
                                w_in = stride[1] * i + m
                                out_int = cnn_layer_tup_to_int((h_in, w_in, c),
                                                               out_layer_shape)
                                max_matrix[out_int][max_int] = max_weight
        else:
            raise ValueError("invalid value for 'max_pool_padding'")

        weights_array.append(max_matrix)

    return weights_array


def cnn_layer_tup_to_int(tup, layer_shape):
    '''
    take a (num_down, num_across, channel) tuple and a layer_shape of the form
    (height, width, num_channels).
    return an int that is unique within the layer representing that particular 
    'neuron'.
    '''
    down, across, channel = tup
    _, width, n_c = layer_shape
    return down * width * n_c + across * n_c + channel


def shuffle_weights(weight_mat):
    '''
    take in a weight tensor, and permute all the weights in it.
    '''
    mat_shape = weight_mat.shape
    flat_mat = weight_mat.flatten()
    rand_flat = np.random.permutation(flat_mat)
    return np.reshape(rand_flat, mat_shape)


def shuffle_weights_nonzero(weight_mat):
    """Shuffle weights in one layer only in nonzero places."""
    mat_shape = weight_mat.shape
    flat_mat = weight_mat.flatten()
    nonzero_indices = np.nonzero(flat_mat)[0]
    perm = np.random.permutation(len(nonzero_indices))
    permuted_flat_mat = np.zeros_like(flat_mat)
    permuted_flat_mat[nonzero_indices] = flat_mat[nonzero_indices[perm]]
    return np.reshape(permuted_flat_mat, mat_shape)

def shuffle_weights_nonzero_distribution(weight_mat):
    mat_shape = weight_mat.shape
    flat_mat = weight_mat.flatten()
    nonzero_indices = np.nonzero(flat_mat)[0]
    kde = (KernelDensity(kernel='gaussian', bandwidth=0.01)
           .fit(flat_mat[nonzero_indices][:, None]))
    sample_flat_mat = np.zeros_like(flat_mat)
    sample_flat_mat[nonzero_indices] = kde.sample(len(nonzero_indices))[:, 0]
    return np.reshape(sample_flat_mat, mat_shape)


def shuffle_weights_layer_all_distribution(weight_mat):
    """Shuffle non-zero places AND sample from distribution."""
    return shuffle_weights_nonzero_distribution(
            shuffle_weights(weight_mat))


def weights_to_graph(weights_array):
    # take an array of weight matrices, and return the adjacency matrix of the
    # neural network it defines.
    # if the weight matrices are A, B, C, and D, the adjacency matrix should be
    # [[0   A^T 0   0   0  ]
    #  [A   0   B^T 0   0  ]
    #  [0   B   0   C^T 0  ]
    #  [0   0   C   0   D^T]
    #  [0   0   0   D   0  ]]
    # however, the weight matrices we get are A^T etc.

    block_mat = []

    # for everything in the weights array, add a row to block_mat of the form
    # [None, None, ..., sparsify(np.abs(mat)), None, ..., None]
    for (i, mat) in enumerate(weights_array):
        sp_mat = sparse.coo_matrix(np.abs(mat))
        if i == 0:
            # add a zero matrix of the right size to the start of the first row
            # so that our final matrix is of the right size
            n = mat.shape[0]
            first_zeroes = sparse.coo_matrix((n, n))
            block_row = [first_zeroes] + [None]*len(weights_array)
        else:
            block_row = [None]*(len(weights_array) + 1)
        block_row[i+1] = sp_mat
        block_mat.append(block_row)

    # add a final row to block_mat that's just a bunch of [None]s followed by a
    # zero matrix of the right size
    m = weights_array[-1].shape[1]
    final_zeroes = sparse.coo_matrix((m, m))
    nones_row = [None]*len(weights_array)
    nones_row.append(final_zeroes)
    block_mat.append(nones_row)

    # turn block_mat into a sparse matrix
    up_tri = sparse.bmat(block_mat, 'csr')

    # we now have a matrix that looks like
    # [[0   A^T 0   0  ]
    #  [0   0   B^T 0  ]
    #  [0   0   0   C^T]
    #  [0   0   0   0  ]]
    # add this to its transpose to get what we want
    adj_mat = up_tri + up_tri.transpose()
    return adj_mat


@clustering_experiment.capture
def tester_cnn_tensors_to_flat_weights_and_graph(weights_array, input_shape):
    '''
    take an array of weight tensors, a list of dicts giving information about 
    convolutional layers, and a list of widths of fully-connected layers,
    and return the adjacency matrix of the weighted graph you would get if you
    'unrolled' the convolutions.
    '''
    final_width = weights_array[-1].shape[1]
    output_shape = (1, final_width, 1)
    layer_shapes = layer_info_to_layer_shapes(output_shape)
    t1 = time.time()
    flat_weights_array = cnn_layers_to_weights_array(layer_shapes, weights_array)
    t2 = time.time()
    print('Current Time', t2 - t1)
    for w_tens in flat_weights_array:
        assert len(w_tens.shape) == 2, f"weight tensor should have been flattened but has shape {w_tens.shape}"

    print('Previous Shapes', [w.shape for w in flat_weights_array])
    
    t3 = time.time()
    CURRENT_RESULT, _ = cnn_tensors_to_flat_weights_and_graph(weights_array)
    t4 = time.time()
    print('Current Time', t4 - t3)

    print('Current Shapes', [w.shape for w in CURRENT_RESULT])
    
    assert len(CURRENT_RESULT) == len(flat_weights_array)
    for s,d in zip(CURRENT_RESULT, flat_weights_array):
        assert (s == d).all()

        
    adj_mat = weights_to_graph(flat_weights_array)

    return (flat_weights_array, adj_mat)


@clustering_experiment.capture
def cnn_tensors_to_flat_weights_and_graph(weights_array, max_weight_convention, as_sparse):
    flat_weights_array, _ = extract_cnn_weights(weights_array,
                                                with_avg=(max_weight_convention=='one_on_n'),
                                                as_sparse=as_sparse)

    for w_tens in flat_weights_array:
        assert len(w_tens.shape) == 2, f"weight tensor should have been flattened but has shape {w_tens.shape}"

    adj_mat = weights_to_graph(flat_weights_array)

    return (flat_weights_array, adj_mat)



def cluster_net(n_clusters, adj_mat, eigen_solver, assign_labels):
    cluster_alg = SpectralClustering(n_clusters=n_clusters,
                                     eigen_solver=eigen_solver,
                                     affinity='precomputed',
                                     assign_labels=assign_labels)
    clustering = cluster_alg.fit(adj_mat)
    return clustering.labels_


def cluster_proportions_per_layer(clustering, num_clusters, weights_array):
    # for each layer, count how many times each cluster appears in it, then
    # divide those numbers by how big the layer is.
    layer_widths = weights_to_layer_widths(weights_array)
    proportions = []
    neurons_seen = 0
    for width in layer_widths:
        cluster_counts = np.zeros(num_clusters)
        for i in range(width):
            neuron = neurons_seen + i
            label = clustering[neuron]
            cluster_counts[label] += 1
        proportions.append(cluster_counts / width)
        neurons_seen += width
    return proportions


def ncut(weights_array, num_clusters, clustering, epsilon):
    # get the cut of each cluster and the volume of each cluster, and sum their
    # ratios
    cut_vals, vol_vals = cut_vol(weights_array, num_clusters, clustering)
    print('Previous', list(zip(cut_vals, vol_vals)))
    return np.sum(cut_vals / (vol_vals + epsilon))


def cut_vol(weights_array, num_clusters, clustering):
    # for each weight matrix, add each edge to the cut and volume values of the
    # appropriate cluster
    cut_vals = np.zeros(num_clusters)
    vol_vals = np.zeros(num_clusters)
    layer_widths = weights_to_layer_widths(weights_array)
    # TODO: make sure this only fires when appropriate
    # assert layer_widths == [784, 256, 256, 256, 256, 10]
    for (l, weight_mat) in enumerate(weights_array):
        lcut, lvol = cut_vol_between_layers(weights_array, l, num_clusters,
                                            clustering, layer_widths)
        cut_vals += lcut
        vol_vals += lvol
        # print("layer:", l)
        # print("cut this layer:", lcut)
        # print("vol this layer:", lvol)
    # print("cut vals:", cut_vals)
    # print("vol_vals:", vol_vals)
    return cut_vals, vol_vals

def cut_vol_between_layers(weights_array, layer, num_clusters, clustering,
                           layer_widths):
    # basically: for every edge in the weights array, the weight of that edge
    # should be added to the volume of the cluster of both vertices attached to
    # the edge, and if the vertices belong to different clusters, then it should
    # also be added to the relevant cut values

    # get a copy of the weight matrix
    weight_mat_ = weights_array[layer]
    weight_mat = weight_mat_.copy()

    # get the cluster values of the vertices on the layers we're going between
    row_labels = []
    for i in range(weight_mat.shape[0]):
        i_int = mlp_tup_to_int((layer, i), layer_widths)
        label_i = clustering[i_int]
        row_labels.append(label_i)
    col_labels = []
    for j in range(weight_mat.shape[1]):
        j_int = mlp_tup_to_int((layer + 1, j), layer_widths)
        label_j = clustering[j_int]
        col_labels.append(label_j)

    # print("layer:", layer)
    # print("row labels:", row_labels)
    # print("col labels:", col_labels)

    # figure out how to permute the rows and columns so that all the first rows
    # are in the first cluster, then a bunch of rows in the second cluster, then
    # a bunch of rows in the third, etc., and same with the columns
    perm_rows = np.argsort(row_labels)
    perm_cols = np.argsort(col_labels)

    # rearrange the matrix so that the rows and columns are grouped by cluster
    for i in range(weight_mat.shape[0]):
        weight_mat[i, :] = weight_mat[i, perm_cols]
    for j in range(weight_mat.shape[1]):
        weight_mat[:, j] = weight_mat[perm_rows, j]
    # by now, if our network is cleanly clusterable, this should be roughly
    # block diagonal

    # take the absolute value of all the weights
    weight_mat = np.abs(weight_mat)
    # print(weight_mat)

    # now for the actual calculation...
    # divide the matrix into blocks depending on the cluster of the rows and
    # columns. then add everything in that block to the appropriate volume and
    # cut values.
    cut_vals = np.zeros(num_clusters)
    vol_vals = np.zeros(num_clusters)
    count_rows = Counter(row_labels)
    count_cols = Counter(col_labels)
    row_offset = 0
    for i in range(num_clusters):
        num_rows_i = count_rows[i]
        end_rows_i = row_offset + num_rows_i
        col_offset = 0
        for j in range(num_clusters):
            num_cols_j = count_cols[j]
            end_cols_j = col_offset + num_cols_j
            sub_mat = weight_mat[row_offset:end_rows_i, col_offset:end_cols_j]
            # print("row cluster", i)
            # print("col cluster", j)
            # print(sub_mat)
            sum_sub_mat = np.sum(sub_mat)
            vol_vals[i] += sum_sub_mat
            vol_vals[j] += sum_sub_mat
            if i != j:
                cut_vals[i] += sum_sub_mat
                cut_vals[j] += sum_sub_mat
            col_offset += num_cols_j
        row_offset += num_rows_i
    return cut_vals, vol_vals


def compute_ncut(adj_mat, clustering_labels, epsilon, verbose=False):
    ncut_terms = {}

    unique_labels = np.unique([label for label in clustering_labels if label != -1])

    for cluster in unique_labels:
        out_mask = (clustering_labels != cluster)
        in_mask = (clustering_labels == cluster)

        cut = adj_mat[in_mask, :][:, out_mask].sum()

        # Sum of the degrees
        vol = adj_mat[in_mask, :].sum()

        ncut_terms[cluster] = cut / (vol + epsilon)
        
        if verbose:
            print('ncut term', cluster, cut, vol)
          
    return sum(ncut_terms.values())


def weights_array_to_cluster_quality(weights_array, adj_mat, num_clusters,
                                     eigen_solver, assign_labels, epsilon,
                                     is_testing=False):
    # t1 = time.time()
    clustering_labels = cluster_net(num_clusters, adj_mat, eigen_solver, assign_labels)
    # t2 = time.time()
    ncut_val = compute_ncut(adj_mat, clustering_labels, epsilon, verbose=is_testing)

    if is_testing:
        ncut_val_previous_method = ncut(weights_array, num_clusters, clustering_labels, epsilon)
        print('NCUT Current', ncut_val)
        print('NCUT Previous', ncut_val_previous_method)
        assert math.isclose(ncut_val, ncut_val_previous_method, abs_tol=1e-5)

    return ncut_val, clustering_labels


def connected_comp_analysis(weights, adj_mat):
    widths = weights_to_layer_widths(weights)
    # get the number of connected components, and label each neuron by what
    # connected component it's in
    nc, labels = sparse.csgraph.connected_components(adj_mat, directed=False)
    counts = Counter(labels)
    # make a dictionary of how many CCs have 1 neuron, how many have 2, etc.
    counts_dict = {}
    # make an array of how many 1-neuron CCs each layer has
    num_each_layer = np.zeros(len(widths))
    for i in range(nc):
        num_i = counts[i]
        if num_i in counts_dict:
            counts_dict[num_i] += 1
        else:
            counts_dict[num_i] = 1
        # if the connected component is a single neuron, find the layer it's in,
        # and add 1 to that index of num_each_layer
        if num_i == 1:
            neuron = np.where(labels == i)[0][0]
            layer = mlp_int_to_tup(neuron, widths)[0]
            num_each_layer[layer] += 1
    prop_each_layer = num_each_layer / widths
    return {'num_comps': nc, 'counts_dict': counts_dict,
            'prop_each_layer': prop_each_layer}

def delete_isolated_ccs_refactored(weights, adjacency_matrix, is_testing=False):
    """Assume that all the isolated connected components have only one node."""
    # 1D boolean array of non-isolated nodes
    node_mask = (adjacency_matrix!=0).toarray().any(axis=1)

    no_isolated_adjacency_matrix = adjacency_matrix[:,node_mask][node_mask,:]

    
    if is_testing:
        layer_sizes = [w.shape[0] for w in weights]

        # create two iterators of the node mask per layer
        # they iterator are in shift of one (current, next)
        # current - slice rows in the weight matrix
        # next - slice columns in the weight matrix
        layer_mask = splitter(node_mask, layer_sizes)
        current_layer_mask, next_layer_mask = it.tee(layer_mask, 2)
        next(next_layer_mask)
        bi_layer_masks = it.zip_longest(current_layer_mask, next_layer_mask, fillvalue=Ellipsis)
        
        array_weights = (layer_weights.toarray() if sparse.issparse(layer_weights)
                         else layer_weights
                         for layer_weights in weights)

        # maybe need .toarray() to sparse instead of np.array
        no_isolated_weights = [np.array(layer_weights)[current_mask,:][:,next_mask]
                               for layer_weights, (current_mask, next_mask)
                               in zip(array_weights, bi_layer_masks)]
    else:
        no_isolated_weights = []

    return no_isolated_weights, no_isolated_adjacency_matrix, node_mask


def delete_isolated_ccs(weight_array, adj_mat):
    # find connected components that aren't represented on both the first and
    # the last layer, and delete them from the graph

    nc, labels = sparse.csgraph.connected_components(adj_mat, directed=False)

    # if there's only one connected component, don't bother
    if nc == 1:
        return weight_array, adj_mat
    
    widths = weights_to_layer_widths(weight_array)

    # find cc labels represented in the first layer
    initial_ccs = set()
    for i in range(widths[0]):
        initial_ccs.add(labels[i])
    # find cc labels represented in the final layer
    final_ccs = set()
    final_layer = len(widths) - 1
    for i in range(widths[-1]):
        neuron = mlp_tup_to_int((final_layer, i), widths)
        final_ccs.add(labels[neuron])

    # find cc labels that aren't in either of those two sets
    isolated_ccs = set()
    for c in range(nc):
        if not (c in initial_ccs and c in final_ccs):
            isolated_ccs.add(c)

    # if there aren't any isolated ccs, don't bother deleting them!
    if not isolated_ccs:
        return weight_array, adj_mat

    # go through weight_array
    # for each array, go to the rows and cols
    # figure out which things you have to delete, then delete them
    new_weight_array = []
    for (t, mat) in enumerate(weight_array):
        # print("weight array number:", t)
        n_rows, n_cols = mat.shape
        # print("original n_rows, n_cols:", (n_rows, n_cols))
        rows_layer = t
        cols_layer = t + 1

        # delete rows and cols corresponding to neurons in isolated clusters
        rows_to_delete = []
        for i in range(n_rows):
            neuron = mlp_tup_to_int((rows_layer, i), widths)
            if labels[neuron] in isolated_ccs:
                rows_to_delete.append(i)

        cols_to_delete = []
        for j in range(n_cols):
            neuron = mlp_tup_to_int((cols_layer, j), widths)
            if labels[neuron] in isolated_ccs:
                cols_to_delete.append(j)

        # print("rows to delete:", rows_to_delete)
        # print("columns to delete:", cols_to_delete)

        rows_deleted = np.delete(mat, rows_to_delete, 0)
        new_mat = np.delete(rows_deleted, cols_to_delete, 1)
        # print("new mat shape:", new_mat.shape)
        new_weight_array.append(new_mat)

    # then return the adj_mat
    new_adj_mat = weights_to_graph(new_weight_array)
    return new_weight_array, new_adj_mat


#@clustering_experiment.capture
def shuffle_and_cluster(num_samples, #weights,
                        weights_path,
                        #loaded_weights,
                        network_type, num_clusters,
                        shuffle_smaller_model, eigen_solver, delete_isolated_ccs_bool,
                        assign_labels, epsilon, shuffle_method):

    ######
    loaded_weights = load_weights(weights_path)
    if network_type == 'mlp':
        weights_ = loaded_weights
        adj_mat_ = weights_to_graph(loaded_weights)
    elif network_type == 'cnn':
        weights_, adj_mat_ = cnn_tensors_to_flat_weights_and_graph(loaded_weights)
    else:
        raise ValueError("network_type must be 'mlp' or 'cnn'")
    
    #######

    if shuffle_smaller_model and delete_isolated_ccs_bool:
        # delete unconnected components from the net BEFORE SHUFFLING!!!
        weights, adj_mat, _ = delete_isolated_ccs_refactored(weights_, adj_mat_,
                                                             is_testing=True)
    else:
        weights, adj_mat = weights_, adj_mat_
    
    #shuff_ncuts = np.array([])
    shuff_ncuts = []

    assert shuffle_method in SHUFFLE_METHODS


    if shuffle_method == 'layer':
        shuffle_function = shuffle_weights
    elif shuffle_method == 'layer_nonzero':
        shuffle_function = shuffle_weights_nonzero
    elif shuffle_method == 'layer_nonzero_distribution':
        shuffle_function = shuffle_weights_nonzero_distribution
    elif shuffle_method == 'layer_all_distribution':
        shuffle_function = shuffle_weights_layer_all_distribution

        
    for _ in range(num_samples):

        # t_start = time.time()
        if network_type == 'mlp':
            if shuffle_smaller_model:
                shuff_weights_ = list(map(shuffle_function, weights))
            else:
                shuff_weights_ = list(map(shuffle_function, loaded_weights))
            shuff_adj_mat_ = weights_to_graph(shuff_weights_)
        else:
            shuff_tensors = list(map(shuffle_function, loaded_weights))
            shuff_weights_, shuff_adj_mat_ = cnn_tensors_to_flat_weights_and_graph(shuff_tensors)
            # NB: this is not quite right, because you're shuffling the whole
            # network, meaning that the isolated ccs get shuffled back in

        # t_before_mid = time.time()
        # print("\ntime to shuffle weights", t_before_mid - t_start)
        if delete_isolated_ccs_bool:
            shuff_weights, shuff_adj_mat, _ = delete_isolated_ccs_refactored(shuff_weights_,
                                                                                shuff_adj_mat_)

        else:
            shuff_weights, shuff_adj_mat = shuff_weights_, shuff_adj_mat_

        # t_mid = time.time()
        # print("time to delete isolated ccs", t_mid - t_before_mid)
        shuff_ncut, _ = weights_array_to_cluster_quality(shuff_weights,
                                                         shuff_adj_mat,
                                                         num_clusters,
                                                         eigen_solver,
                                                         assign_labels, epsilon)
        shuff_ncuts.append(shuff_ncut)
        #shuff_ncuts = np.append(shuff_ncuts, shuff_ncut)
        # t_end = time.time()
        # print("time to cluster shuffled weights", t_end - t_mid)

    return np.array(shuff_ncuts)


@clustering_experiment.automain
def run_clustering(weights_path, num_clusters, eigen_solver, assign_labels,
                   epsilon, num_samples, delete_isolated_ccs_bool, network_type,
                   shuffle_smaller_model,
                   with_labels, with_shuffle, shuffle_method, n_workers,
                   is_testing, with_shuffled_ncuts):
    # t0 = time.time()
    # load weights and get adjacency matrix
    if is_testing:
        assert network_type == 'cnn'

    loaded_weights = load_weights(weights_path)
    if network_type == 'mlp':
        weights_ = loaded_weights
        adj_mat_ = weights_to_graph(loaded_weights)
    elif network_type == 'cnn':

        # comparing current and previous version of expanding CNN
        if is_testing:
            tester_cnn_tensors_to_flat_weights_and_graph(loaded_weights)

        weights_, adj_mat_ = cnn_tensors_to_flat_weights_and_graph(loaded_weights)

    else:
        raise ValueError("network_type must be 'mlp' or 'cnn'")

    # t1 = time.time()
    # print('time to form adjacency matrix', t1 - t0)

    # analyse connectivity structure of network
    # cc_dict = connected_comp_analysis(weights_, adj_mat_)
    # print("connectivity analysis:", cc_dict)

   
    if delete_isolated_ccs_bool:
        # delete unconnected components from the net
        weights, adj_mat, node_mask = delete_isolated_ccs_refactored(weights_, adj_mat_,
                                                                     is_testing=is_testing)
        
        if is_testing:
            weights_old, adj_mat_old = delete_isolated_ccs(weights_, adj_mat_)
            assert (adj_mat != adj_mat_old).sum() == 0
            assert all((w1 == w2).all() for w1, w2 in zip(weights, weights_old))
    
    else:
        weights, adj_mat = weights_, adj_mat_
        node_mask = numpy.full(adj_mat.shape[0], True)

    # t2 = time.time()
    # print("time to delete isolated ccs", t2 - t1)
    
    # find cluster quality of this pruned net
    print("\nclustering unshuffled weights\n")
    unshuffled_ncut, clustering_labels = weights_array_to_cluster_quality(weights, adj_mat,
                                                                          num_clusters,
                                                                          eigen_solver,
                                                                          assign_labels, epsilon,
                                                                          is_testing)
    ave_in_out = (1 - unshuffled_ncut / num_clusters) / (2 * unshuffled_ncut
                                                         / num_clusters)
            

    # t3 = time.time()
    # print("time to cluster unshuffled weights", t3 - t2)
    result = {'ncut': unshuffled_ncut,
              'ave_in_out': ave_in_out,
              'node_mask': node_mask}
    #return clustering_labels, adj_mat, result

    if with_shuffle:
        
        # find cluster quality of other ways of rearranging the net
        print("\nclustering shuffled weights\n")
        n_samples_per_worker = num_samples // n_workers

        function_argument = (n_samples_per_worker, weights_path, #weights,
                             # loaded_weights,
                             network_type, num_clusters,
                             shuffle_smaller_model, eigen_solver, delete_isolated_ccs_bool,
                             assign_labels, epsilon, shuffle_method)
        if n_workers == 1:
            print('No Pool! Single Worker!')
            shuff_ncuts = shuffle_and_cluster(*function_argument)

        else:
            print(f'Using Pool! Multiple Workers! {n_workers}')

            workers_arguments = [[copy.deepcopy(arg) for _ in range(n_workers)]
                                  for arg in function_argument]

            with ProcessPool(nodes=n_workers) as p:
                shuff_ncuts_results = p.map(shuffle_and_cluster,
                                            *workers_arguments)

            shuff_ncuts = np.concatenate(shuff_ncuts_results)                     

        shuffled_n_samples = len(shuff_ncuts)
        shuffled_mean = np.mean(shuff_ncuts, dtype=np.float64)
        shuffled_stdev = np.std(shuff_ncuts, dtype=np.float64)
        print('BEFORE', np.std(shuff_ncuts))
        percentile = compute_pvalue(unshuffled_ncut, shuff_ncuts)
        print('AFTER', np.std(shuff_ncuts))
        z_score = (unshuffled_ncut - shuffled_mean) / shuffled_stdev
        
        result.update({'shuffle_method': shuffle_method,
                       'n_samples': shuffled_n_samples,
                       'mean': shuffled_mean,
                       'stdev': shuffled_stdev,
                       'z_score': z_score,
                       'percentile': percentile})

    if with_shuffled_ncuts:
        result['shuffled_ncuts'] = shuff_ncuts
        
    if with_labels:
        result['labels'] = clustering_labels
    
    return result
    
