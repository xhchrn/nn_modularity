"""Script for training neural network for binary classification of 28x28 grayscale images."""


from datetime import datetime
import pickle
from pathlib import Path
import shutil
import json

import numpy as np
import sacred 
import tensorflow as tf
from tensorflow_model_optimization.sparsity import keras as sparsity

from src.utils import picklify, extract_weights, get_sparsity, NumpyEncoder
from src.cnn import CNN_MODEL_PARAMS
from src.cnn.extractor import extract_cnn_weights
from src.pointers import DATA_PATHS


ex = sacred.Experiment('training')
ex.observers.append(sacred
                    .observers
                    .FileStorageObserver
                    .create('training_runs_dir'))


def generate_training_tag(network_type, epochs, dataset_name, dropout):
    base_tag = f"{dataset_name}_{network_type}_{epochs}epochs"
    if dropout:
        base_tag += "_dropout"
    return base_tag

@ex.config
def general_config():
    num_classes = 10
    width, height = 28, 28
    size = width * height
    batch_size = 128
    network_type = None
    epochs = 0
    dataset_name = ""
    with_dropout = False
    training_tag = generate_training_tag(network_type, epochs, dataset_name, with_dropout)
    model_dir_path = Path('./models/{}/{}/{}'.format(datetime.now().strftime('%Y%m%d'),
                                                     training_tag,
                                                     datetime.now().strftime('%H%M%S')))
    tensorboard_log_dir = './logs'
    shuffle = True
    n_train = None



@ex.config
def pruning_config():
    initial_sparsity = 0.50
    final_sparsity = 0.90
    begin_step = 0
    frequency = 10


@ex.named_config
def mlp_config():
    network_type = 'mlp'
    model_params = {'widths': [256, 256, 256, 256]}
    dataset_name = 'lines'
    dropout_rate = 0.5
    epochs = 20
    pruning_epochs = 20


# Reference: https://keras.io/examples/cifar10_cnn/
@ex.named_config
def cnn_config():
    network_type = 'cnn'
    model_params = CNN_MODEL_PARAMS
    conv_dropout_rate = 0.25
    dense_dropout_rate = 0.5
    epochs = 10
    pruning_epochs = 10



@ex.capture
def get_pruning_params(num_train_samples,
                       initial_sparsity, final_sparsity,
                       begin_step, frequency,
                       batch_size, pruning_epochs):

    end_step = np.ceil(num_train_samples / batch_size).astype(np.int32) * pruning_epochs

    return  {'pruning_schedule': sparsity.PolynomialDecay(initial_sparsity=initial_sparsity,
                                                          final_sparsity=final_sparsity,
                                                          begin_step=begin_step,
                                                          end_step=end_step,
                                                          frequency=frequency)
    }


@ex.capture
def save_weights(model, model_path, network_type, _log):
        
    weight_path = str(model_path) + '-weights.pckl'
    
    weights = extract_weights(model)

    picklify(weight_path,
             weights)
    ex.add_artifact(weight_path)
    
    if network_type == 'cnn':
        _log.info('Expanding CNN layers...')
        
        expanded_weights, constraints = extract_cnn_weights(weights,
                                                            as_sparse=True, verbose=True)
        
        expanded_weight_path = str(model_path) + '-weights-expanded.pckl'
        constraintst_path = str(model_path) + '-constraints-expanded.pckl'

          
        picklify(expanded_weight_path,
                 expanded_weights)
        ex.add_artifact(expanded_weight_path)
        
        picklify(constraintst_path,
                 constraints)
        ex.add_artifact(constraintst_path)


@ex.capture
def load_data(dataset_name, num_classes, width, height, size, network_type, n_train):
    
    assert dataset_name in DATA_PATHS

    data_path = DATA_PATHS[dataset_name]

    with open(data_path, 'rb') as f:
        dataset = pickle.load(f)
    
    
    X_train = dataset['X_train']
    X_test = dataset['X_test']
    
    if (X_train.min() == 0 and X_train.max() == 255 and X_test.min() == 0 and X_test.max() == 255):
        X_train = X_train / 255
        X_test = X_test / 255
    # elif (X_train.min() == 0 and X_train.max() == 1 and X_test.min() == 0 and X_test.max() == 1):
    #     pass
    else:
        raise ValueError('X_train and X_test should be either in the range [0, 255] or [0, 1].')

    assert X_train.min() == 0
    assert X_test.min() == 0
    assert X_train.max() == 1
    assert X_test.max() == 1

    y_train = tf.keras.utils.to_categorical(dataset['y_train'])
    y_test = tf.keras.utils.to_categorical(dataset['y_test'])

    assert y_train.shape[-1] == 10
    assert y_test.shape[-1] == 10
    
    if network_type == 'mlp':
        X_train = X_train.reshape([-1, size])
        X_test = X_test.reshape([-1, size])

        assert X_train.shape[-1] == size
        assert X_test.shape[-1] == size
        
    elif network_type == 'cnn':
        X_train = X_train.reshape([-1, height, width, 1])
        X_test = X_test.reshape([-1, height, width, 1])

        assert X_train.shape[-3:] == (height, width, 1)
        assert X_test.shape[-3:] == (height, width, 1)
        
    if n_train is not None:
        X_train = X_train[:n_train]
        y_train = y_train[:n_train]

    return (X_train, y_train), (X_test, y_test)


@ex.capture
def create_mlp_layers(size, num_classes, model_params,
                      with_dropout, dropout_rate):

    assert model_params['widths']

    layers = [tf.keras.layers.Dense(model_params['widths'][0],
                                    activation='relu', input_shape=(size,))]

    hidden_layers = [tf.keras.layers.Dense(layer_width, activation='relu')
                     for layer_width in model_params['widths'][1:]]

    if with_dropout:
        new_hidden_layers = [tf.keras.layers.Dropout(dropout_rate)]

        for hidden in hidden_layers:
            new_hidden_layers.append(hidden)
            new_hidden_layers.append(tf.keras.layers.Dropout(dropout_rate))

        hidden_layers = new_hidden_layers
    
    layers.extend(hidden_layers)
    
    layers.append(tf.keras.layers.Dense(num_classes, activation='softmax'))
    
    return layers


@ex.capture
def create_cnn_layers(width, height, num_classes, model_params,
                      with_dropout, conv_dropout_rate, dense_dropout_rate):

    assert model_params['conv']
    assert model_params['dense']

    layers = []
    
    conv_layers = []
    
    is_first = True

    for colv_params in model_params['conv']:
        conv_kwargs = {'input_shape': (width, height, 1)} if is_first else {}
        is_first = False

        conv_layers.append(tf.keras.layers.Conv2D(colv_params['filters'],
                                                  colv_params['kernel_size'],
                                                  padding=colv_params['padding'],
                                                  activation='relu',
                                                  **conv_kwargs))
        
        if colv_params['max_pool_after']:
            # conv_layers.append(tf.keras.layers.AveragePooling2D(pool_size=colv_params['max_pool_size'],
            #                                                padding=colv_params['max_pool_padding']))


            conv_layers.append(tf.keras.layers.MaxPooling2D(pool_size=colv_params['max_pool_size'],
                                                            padding=colv_params['max_pool_padding']))

        if with_dropout:
            conv_layers.append(tf.keras.layers.Dropout(conv_dropout_rate))
    
    layers.extend(conv_layers)

    dense_layers = [tf.keras.layers.Flatten()]

    for dense_layer_width in model_params['dense']:
        dense_layers.append(tf.keras.layers.Dense(dense_layer_width, activation='relu'))
        if with_dropout:
            dense_layers.append(tf.keras.layers.Dropout(dense_dropout_rate))

    layers.extend(dense_layers)

    layers.append(tf.keras.layers.Dense(num_classes, activation='softmax'))

    return layers


@ex.capture
def create_model(network_type):

    assert network_type in ('mlp', 'cnn')
    
    if network_type == 'mlp':
        layers = create_mlp_layers()

    elif network_type == 'cnn':
        layers = create_cnn_layers()      
    
    return tf.keras.Sequential(layers)


@ex.capture
def get_two_model_paths(model_dir_path, dataset_name, network_type):
    directory = f'{dataset_name}-{network_type}-'
    return (model_dir_path / (directory + 'unpruned'),
            model_dir_path / (directory + 'pruned'))


@ex.capture
def train_model(model, X_train, y_train, X_test, y_test,
                model_path,
                batch_size, epochs, shuffle,
                tensorboard_log_dir, model_dir_path, _log,
                callbacks=None):

    if callbacks == None:
        callbacks = []

    ckpt_path = f'{model_path}-{{epoch:04d}}.ckpt'
    ckpt_callback = tf.keras.callbacks.ModelCheckpoint(ckpt_path,
                                                       save_weights_only=False,
                                                       verbose=1)

    tb_callback = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_log_dir,
                                                 update_freq='batch',
                                                 profile_batch=0)

    callbacks.extend([ckpt_callback, tb_callback])

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])


    _log.info(model.summary())

    # model.save_weights(ckpt_path.format(epoch=0))

    hist = model.fit(X_train, y_train,
                     batch_size=batch_size,
                     nb_epoch=epochs,  # instead of `epochs`, to get full history metrics
                     verbose=1,
                     validation_data=(X_test, y_test),
                     callbacks=callbacks,
                     shuffle=shuffle)

    loss, acc = model.evaluate(X_test, y_test)
    _log.info('Trained model - Test dataset, accuracy: {:5.2f}%, loss: {:5.4f}'
              .format(100*acc, loss))

    model_hdf_path = str(model_path) + '.h5'
    model.save(model_hdf_path)
    ex.add_artifact(model_hdf_path)

    for cpkt_filename in model_dir_path.glob('*.ckpt'):
        ex.add_artifact(cpkt_filename)

    return hist.history


@ex.automain
def run(network_type, batch_size, epochs, pruning_epochs,
        tensorboard_log_dir, model_dir_path,
        _log, _run):

    assert network_type in ('mlp', 'cnn')

    _log.info('Emptying model directory...')
    if model_dir_path.exists():
        shutil.rmtree(model_dir_path)
    Path(model_dir_path).mkdir(parents=True)

    _log.info('Loading data...')
    (X_train, y_train), (X_test, y_test) = load_data()

    metrics = {}
    
    unpruned_model_path, pruned_model_path = get_two_model_paths()

    unpruned_model = create_model()

    _log.info('Training unpruned model...')
    metrics['unpruned'] = train_model(unpruned_model, X_train, y_train, X_test, y_test, 
                                      unpruned_model_path, epochs=epochs)

    _log.info('Unpruned model sparsity: {}'.format(get_sparsity(unpruned_model)))
    save_weights(unpruned_model, unpruned_model_path)

    pruning_params = get_pruning_params(X_train.shape[0])    
    pruned_model = sparsity.prune_low_magnitude(unpruned_model, **pruning_params)
    
    pruning_callbacks = [
        sparsity.UpdatePruningStep(),
        sparsity.PruningSummaries(log_dir=tensorboard_log_dir,
                                  profile_batch=0)
    ]

    _log.info('Training pruned model...')
    metrics['pruned'] = train_model(pruned_model, X_train, y_train, X_test, y_test,
                                    pruned_model_path,
                                    epochs=pruning_epochs, callbacks=pruning_callbacks)

    _log.info('Pruned model sparsity: {}'.format(get_sparsity(pruned_model)))

    save_weights(pruned_model, pruned_model_path)
    
    ex.add_source_file(__file__)

    with open(model_dir_path / 'metrics.json', 'w') as f:
        json.dump(metrics, f, cls=NumpyEncoder)

    return metrics
