"""Script for generating MNIST, MNIST-Fashion, CIFAR-10 and LINE dataests."""


import pickle

import numpy as np
from skimage import draw
from skimage.transform import resize
from skimage.color import rgb2gray
from tensorflow.keras.datasets import mnist, cifar10, fashion_mnist

def preprocess_batch_cifar10(imgs_in, width, height):

    imgs_out = np.zeros([imgs_in.shape[0], width, height])

    for n, i in enumerate(imgs_in):
        imgs_out[n,:,:] = rgb2gray(resize(imgs_in[n,:,:], [width, height], anti_aliasing=True))
    
    imgs_out *= 255

    return imgs_out

def line_counting_data_gen(batch_size, n_possible_lines=10, width=28, height=28):
    while True:
        inputs = []
        labels = []

        for _ in range(batch_size):
            img = np.zeros((height, width))

            n_lines = np.random.randint(1, n_possible_lines + 1)
            lines = set()

            while len(lines) < n_lines:
                orientation = 'vertical' # if np.random.randint(0, 2) else 'horizontal'
                
                if orientation == 'vertical':
                    y = np.random.randint(0, height)
                    line = (0, y, width-1, y)
                elif orientation == 'horizontal':
                    x = np.random.randint(0, width)
                    line = (x, 0, x, height-1)

                if line not in lines:
                    rr, cc = draw.line(*line)
                    img[rr, cc] = 255
                    lines.add(line)
        
            inputs.append(img)
            labels.append(n_lines - 1)

        yield np.array(inputs), np.array(labels)
        # yield np.expand_dims(np.stack(inputs), axis=-1), np.stack(labels)


def circle_counting_data_gen(batch_size, n_possible_circles=10, width=28, height=28, radius=4):
    while True:
        inputs = []
        labels = []

        for _ in range(batch_size):
            img = np.zeros((height, width))

            n_circles = np.random.randint(0, n_possible_circles)
            centers = set()

            while len(centers) < n_circles:
                
                center_x = np.random.randint(0, width)
                center_y = np.random.randint(0, height)
            
                if (center_x, center_y) not in centers:
                    rr, cc = draw.circle_perimeter(center_x, center_y, radius=radius, shape=img.shape)
                    img[rr, cc] = 255
                    centers.add((center_x, center_y))
        
            inputs.append(img)
            labels.append(n_circles - 1)

        yield np.array(inputs), np.array(labels)
        # yield np.expand_dims(np.stack(inputs), axis=-1), np.stack(labels)


def generate_random_datast(n_train, n_test, width, height):
    X_train = np.random.randint(0, 256, (60000, 28, 28))
    y_train = np.random.randint(0, 10, 60000)

    X_test = np.random.randint(0, 256, (10000, 28, 28))
    y_test = np.random.randint(0, 10, 10000)

    return (X_train, y_train), (X_test, y_test)


def main(path, n_train=60000, n_test=10000, width=28, height=28,
         shape: ('Task shape', 'option', 's')='line',
         random_state: ('Random state', 'option', 'r')=42):
    
    if random_state is not None:
        np.random.seed(random_state)
    
    assert shape in ('line', 'circle', 'mnist', 'cifar10', 'fashion', 'random')
    
    if shape == 'mnist':
        (X_train, y_train), (X_test, y_test) = mnist.load_data()

    elif shape == 'cifar10':
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
        X_train = preprocess_batch_cifar10(X_train, width, height)
        y_train = y_train[:, 0]
        X_test = preprocess_batch_cifar10(X_test, width, height)
        y_test = y_test[:, 0]

    elif shape == 'fashion':
        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

    elif shape == 'random':
        ((X_train, y_train),
         (X_test, y_test)) = generate_random_datast(n_train, n_test, width, height)

    else:
        gen = line_counting_data_gen if shape == 'line' else circle_counting_data_gen

        train_dataset = gen(batch_size=n_train, width=width, height=height)
        X_train, y_train = next(train_dataset)

        test_dataset = gen(batch_size=n_test, width=width, height=height)
        X_test, y_test= next(test_dataset)

    assert X_train.max() == 255
    assert X_test.max() == 255

    with open(path, 'wb') as f:
        pickle.dump({'X_train': X_train,
                     'y_train': y_train,
                     'X_test': X_test,
                     'y_test': y_test},
                     f)


if __name__ == '__main__':
    import plac; plac.call(main)