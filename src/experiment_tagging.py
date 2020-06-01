"""Tags and paths for trained models."""


from pathlib import Path
from glob import glob
import os
from datetime import datetime
import numpy as np


BASE_PATH = '../models/'


MODEL_TAG_LOOKUP = {
'MNIST': 'mnist_mlp_20epochs',
'CIFAR10': 'cifar10_mlp_20epochs',
'LINE': 'line_mlp_20epochs',
'FASHION': 'fashion_mlp_20epochs',
'MNIST+DROPOUT': 'mnist_mlp_20epochs_dropout',
'CIFAR10+DROPOUT': 'cifar10_mlp_100epochs_dropout',
'LINE+DROPOUT': 'line_mlp_20epochs_dropout',
'FASHION+DROPOUT': 'fashion_mlp_20epochs_dropout',
'LINE-MNIST': 'line-mnist_mlp_20epochs',
'LINE-CIFAR10': 'line-cifar10_mlp_30epochs',
'MNIST-CIFAR10': 'mnist-cifar10_mlp_30epochs',
'LINE-MNIST-SEPARATED': 'line-mnist-separated_mlp_20epochs',
'LINE-CIFAR10-SEPARATED': 'line-cifar10-separated_mlp_30epochs',
'MNIST-CIFAR10-SEPARATED': 'mnist-cifar10-separated_mlp_30epochs',
'LINE-MNIST+DROPOUT': 'line-mnist_mlp_20epochs_dropout',
'LINE-CIFAR10+DROPOUT': 'line-cifar10_mlp_30epochs_dropout',
'MNIST-CIFAR10+DROPOUT': 'mnist-cifar10_mlp_30epochs_dropout',
'LINE-MNIST-SEPARATED+DROPOUT': 'line-mnist-separated_mlp_20epochs_dropout',
'LINE-CIFAR10-SEPARATED+DROPOUT': 'line-cifar10-separated_mlp_30epochs_dropout',
'MNIST-CIFAR10-SEPARATED+DROPOUT': 'mnist-cifar10-separated_mlp_30epochs_dropout',
'RANDOM': 'random_mlp_20epochs',
'RANDOM+DROPOUT': 'random_mlp_20epochs_dropout',
'MNIST-x1.5-EPOCHS': 'mnist_mlp_30epochs',
'MNIST-x1.5-EPOCHS+DROPOUT': 'mnist_mlp_30epochs_dropout',
'MNIST-x2-EPOCHS': 'mnist_mlp_40epochs',
'MNIST-x2-EPOCHS+DROPOUT': 'mnist_mlp_40epochs_dropout',
'MNIST-x10-EPOCHS': 'mnist_mlp_200epochs',
'MNIST-x10-EPOCHS+DROPOUT': 'mnist_mlp_200epochs_dropout',
'RANDOM-x50-EPOCHS': 'random_mlp_1000epochs',
'RANDOM-x50-EPOCHS+DROPOUT': 'random_mlp_1000epochs_dropout',
'RANDOM-OVERFITTING': 'random_mlp_100epochs',
'RANDOM-OVERFITTING+DROPOUT': 'random_mlp_100epochs_dropout',
'CNN:MNIST': 'mnist_cnn_10epochs',
'CNN:CIFAR10': 'cifar10_cnn_10epochs',
'CNN:LINE': 'line_cnn_10epochs',
'CNN:FASHION': 'fashion_cnn_10epochs',
'CNN:MNIST+DROPOUT': 'mnist_cnn_10epochs_dropout',
'CNN:CIFAR10+DROPOUT': 'cifar10_cnn_10epochs_dropout',
'CNN:LINE+DROPOUT': 'line_cnn_10epochs_dropout',
'CNN:FASHION+DROPOUT': 'fashion_cnn_10epochs_dropout',
}


def get_model_path(model_tag, date_tag='*', time_tag='*', filter_='last',
                   model_base_path=BASE_PATH):
    
    assert model_tag in MODEL_TAG_LOOKUP, (f"The tag `{model_tag}` doesn't exist.")
    assert filter_ in ('last', 'all')
    
    base_path = Path(model_base_path)
    
    paths = base_path.glob(f'{date_tag}/{MODEL_TAG_LOOKUP[model_tag]}/{time_tag}')
    
    # We cannot check whether a generator is empty without iterating over it,
    # so we make it into a list
    paths = list(paths)
    assert paths, ('No model path, which correspond to the tags and base path, was found!'
                   'If you are sure that the tags are correct, please check whether'
                   '`model_base_path` points to directory of where the models are saved.')
    
    # The "maximual" path is the latest, because we use the date and time tags format as string
    # <year>-<month>-<date> and <hour>-<minutes>-<seconds>, respectively.
    # So the lexical order corresponds to time order (early to recent).
    if filter_ == 'last':
        run_results = max(paths)
    # Same argument goes for sorting the paths, to get them by timestamp order
    elif filter_ == 'all':
        run_results = sorted(paths)
    
    return run_results
