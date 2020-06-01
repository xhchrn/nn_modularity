#!/usr/bin/env bash

set -xe

if [[ "$1" = "datasets" ]]
then
    echo "######## Generating Datasets... ######## "

    mkdir -p datasets

    python -m src.generate_datasets -s mnist datasets/mnist.pckl
    python -m src.generate_datasets -s cifar10 datasets/cifar10.pckl
    python -m src.generate_datasets -s line datasets/line.pckl
    python -m src.generate_datasets -s fashion datasets/fashion.pckl
    python -m src.generate_datasets -s random datasets/random.pckl

    python -m src.mix_datasets datasets/line.pckl datasets/mnist.pckl datasets/line-mnist.pckl
    python -m src.mix_datasets datasets/line.pckl datasets/cifar10.pckl datasets/line-cifar10.pckl
    python -m src.mix_datasets datasets/mnist.pckl datasets/cifar10.pckl datasets/mnist-cifar10.pckl

    python -m src.mix_datasets datasets/line.pckl datasets/mnist.pckl datasets/line-mnist-separated.pckl -s
    python -m src.mix_datasets datasets/line.pckl datasets/cifar10.pckl datasets/line-cifar10-separated.pckl -s
    python -m src.mix_datasets datasets/mnist.pckl datasets/cifar10.pckl datasets/mnist-cifar10-separated.pckl -s

    echo "######## Done! ########"

elif [[ "$1" = "models" ]]
then
    echo "######## Training Models... ########"

    mkdir -p models

    echo "######## MLP: MNIST ########"
    python -m src.train_nn with mlp_config dataset_name=mnist

    # We shouldn't accept for high accuracy for MLP on CIFAR 10
    # https://krzysztofarendt.github.io/2019/01/29/cifar-10.html
    echo "######## MLP:CIFAR10 ########"
    python -m src.train_nn with mlp_config dataset_name=cifar10 pruning_epochs=40

    echo "######## MLP: LINE ########"
    python -m src.train_nn with mlp_config dataset_name=line

    # Accuracy around 88%-90% is reasonable for MLP - as we get
    # http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/
    # python -m src.train_nn print_config
    echo "######## MLP: FASHION ########"
    python -m src.train_nn with mlp_config dataset_name=fashion

    echo "######## MLP: MNIST + DROPOUT ########"
    python -m src.train_nn with mlp_config dataset_name=mnist with_dropout=True

    echo "######## MLP: CIFAR10 + DROPOUT ########"
    python -m src.train_nn with mlp_config dataset_name=cifar10 epochs=100 pruning_epochs=40 with_dropout=True dropout_rate=0.2

    echo "######## MLP: LINE + DROPOUT ########"
    python -m src.train_nn with mlp_config dataset_name=line with_dropout=True

    echo "######## MLP: FASHION + DROPOUT ########"
    python -m src.train_nn with mlp_config dataset_name=fashion with_dropout=True

    echo "######## MLP: LINE-MNIST ########"
    python -m src.train_nn with mlp_config dataset_name=line-mnist

    echo "######## MLP: LINE-CIFAR10 ########"
    python -m src.train_nn with mlp_config dataset_name=line-cifar10 epochs=30 pruning_epochs=40

    echo "######## MLP: MNIST-CIFAR10 ########"
    python -m src.train_nn with mlp_config dataset_name=mnist-cifar10 epochs=30 pruning_epochs=40

    echo "######## MLP: LINE-MNIST-SEPARATED ########"
    python -m src.train_nn with mlp_config dataset_name=line-mnist-separated

    echo "######## MLP: LINE-CIFAR10-SEPARATED ########"
    python -m src.train_nn with mlp_config dataset_name=line-cifar10-separated epochs=30 pruning_epochs=40

    echo "######## MLP: MNIST-CIFAR10-SEPARATED ########"
    python -m src.train_nn with mlp_config dataset_name=mnist-cifar10-separated epochs=30 pruning_epochs=40

    echo "######## MLP: LINE-MNIST + DROPOUT ########"
    python -m src.train_nn with mlp_config dataset_name=line-mnist with_dropout=True

    echo "######## MLP: LINE-CIFAR10 + DROPOUT ########"
    python -m src.train_nn with mlp_config dataset_name=line-cifar10 epochs=30 pruning_epochs=40 with_dropout=True dropout_rate=0.2

    echo "######## MLP: MNIST-CIFAR10 + DROPOUT ########"
    python -m src.train_nn with mlp_config dataset_name=mnist-cifar10 epochs=30 pruning_epochs=40 with_dropout=True dropout_rate=0.2

    echo "######## MLP: LINE-MNIST-SEPARATED + DROPOUT ########"
    python -m src.train_nn with mlp_config dataset_name=line-mnist-separated with_dropout=True

    echo "######## MLP: LINE-CIFAR10-SEPARATED + DROPOUT ########"
    python -m src.train_nn with mlp_config dataset_name=line-cifar10-separated epochs=30 pruning_epochs=40 with_dropout=True dropout_rate=0.2

    echo "######## MLP: MNIST-CIFAR10-SEPARATED + DROPOUT ########"
    python -m src.train_nn with mlp_config dataset_name=mnist-cifar10-separated epochs=30 pruning_epochs=40 with_dropout=True dropout_rate=0.2

    echo "######## MLP: RANDOM ########"
    python -m src.train_nn with mlp_config dataset_name=random

    echo "######## MLP: RANDOM + DROPOUT ########"
    python -m src.train_nn with mlp_config dataset_name=random with_dropout=True

    echo "######## MLP: MMNIST-x1.5-EPOCHS unpruned ########"
    # TODO is this really unpruned?
    python -m src.train_nn with mlp_config dataset_name=mnist epochs=30

    echo "######## MLP: MMNIST-x1.5-EPOCHS unpruned + DROPOUT ########"
    # TODO is this really unpruned?
    python -m src.train_nn with mlp_config dataset_name=mnist epochs=30 with_dropout=True

    echo "######## MLP: MMNIST-x2-EPOCHS unpruned ########"
    python -m src.train_nn with mlp_config dataset_name=mnist epochs=40

    echo "######## MLP: MMNIST-x2-EPOCHS unpruned + DROPOUT ########"
    python -m src.train_nn with mlp_config dataset_name=mnist epochs=40 with_dropout=True

    echo "######## MLP: MMNIST-x10-EPOCHS unpruned ########"
    python -m src.train_nn with mlp_config dataset_name=mnist epochs=200

    echo "######## MLP: MMNIST-x10-EPOCHS unpruned + DROPOUT ########"
    python -m src.train_nn with mlp_config dataset_name=mnist epochs=200 with_dropout=True

    echo "######## MLP: RANDOM-x50-EPOCHS unpruned ########"
    python -m src.train_nn with mlp_config dataset_name=random epochs=1000

    echo "######## MLP: RANDOM-x50-EPOCHS unpruned + DROPOUT ########"
    python -m src.train_nn with mlp_config dataset_name=random epochs=1000 with_dropout=True

    echo "######## MLP: RANDOM-OVERFITTING ########"
    python -m src.train_nn with mlp_config dataset_name=random epochs=100 pruning_epochs=100 shuffle=False n_train=3000

    echo "######## MLP: RANDOM-OVERFITTING + DROPOUT########"
    python -m src.train_nn with mlp_config dataset_name=random epochs=100 pruning_epochs=100 shuffle=False n_train=3000 with_dropout=True

    echo "######## CNN: MNIST ########"
    python -m src.train_nn with cnn_config dataset_name=mnist

    echo "######## CNN: CIFAR10 ########"
    python -m src.train_nn with cnn_config dataset_name=cifar10

    echo "######## CNN: LINE ########"
    python -m src.train_nn with cnn_config dataset_name=line

    echo "######## CNN: FASHION ########"
    python -m src.train_nn with cnn_config dataset_name=fashion

    echo "######## CNN: MNIST + DROPOUT ########"
    python -m src.train_nn with cnn_config dataset_name=mnist with_dropout=True

    echo "######## CNN: CIFAR10 + DROPOUT ########"
    python -m src.train_nn with cnn_config dataset_name=cifar10 with_dropout=True

    echo "######## CNN: LINE + DROPOUT ########"
    python -m src.train_nn with mlp_config dataset_name=line with_dropout=True

    echo "######## CNN: FASHION + DROPOUT ########"
    python -m src.train_nn with cnn_config dataset_name=fashion with_dropout=True

    echo "######## Done! ########"

fi
