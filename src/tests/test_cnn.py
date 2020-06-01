import numpy as np

from src.cnn.extractor import expand_conv_layer, expand_pool_layer
from src.cnn.convertor import cnn2mlp


def test_expand_conv_layer():

    def tester(expected_layer, expected_output_side, **kwargs):
        layer, output_side, _ = expand_conv_layer(**kwargs)

        assert np.array_equal(expected_layer, layer)
        assert expected_output_side == output_side

    tester(np.ones((1, 1)), 1,
           kernel=np.ones((1, 1, 1, 1)), input_side=1, padding='same')

    tester(np.ones((1, 1)), 1,
           kernel=np.ones((1, 1, 1, 1)), input_side=1, padding='valid')



    tester(np.eye((9)), 3,
           kernel=np.ones((1, 1, 1, 1)), input_side=3, padding='same')

    tester(np.eye((9)), 3,
           kernel=np.ones((1, 1, 1, 1)), input_side=3, padding='valid')

    tester(np.ones((9, 1)),
           1,
           kernel=np.ones((3, 3, 1, 1)), input_side=3, padding='valid')

    tester(np.array([[1, 1, 0, 1, 1, 0, 0, 0, 0],
                     [1, 1, 1, 1, 1, 1, 0, 0, 0],
                     [0, 1, 1, 0, 1, 1, 0, 0, 0],
                     [1, 1, 0, 1, 1, 0, 1, 1, 0],
                     [1, 1, 1, 1, 1, 1, 1, 1, 1],
                     [0, 1, 1, 0, 1, 1, 0, 1, 1],
                     [0, 0, 0, 1, 1, 0, 1, 1, 0],
                     [0, 0, 0, 1, 1, 1, 1, 1, 1],
                     [0, 0, 0, 0, 1, 1, 0, 1, 1]]),
           3,
           kernel=np.ones((3, 3, 1, 1)), input_side=3, padding='same')

    tester(np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9]]),
            1,
            kernel=(np.arange(9) + 1).reshape((3, 3, 1, 1)), input_side=3, padding='valid')

    tester(np.array([[5, 4, 0, 2, 1, 0, 0, 0, 0],
                     [6, 5, 4, 3, 2, 1, 0, 0, 0],
                     [0, 6, 5, 0, 3, 2, 0, 0, 0],
                     [8, 7, 0, 5, 4, 0, 2, 1, 0],
                     [9, 8, 7, 6, 5, 4, 3, 2, 1],
                     [0, 9, 8, 0, 6, 5, 0, 3, 2],
                     [0, 0, 0, 8, 7, 0, 5, 4, 0],
                     [0, 0, 0, 9, 8, 7, 6, 5, 4],
                     [0, 0, 0, 0, 9, 8, 0, 6, 5]]),
            3,
            kernel=(np.arange(9) + 1).reshape((3, 3, 1, 1)), input_side=3, padding='same')


    tester(np.array([[5, 4, 0, 2, 1, 0, 0, 0, 0],
                     [6, 5, 4, 3, 2, 1, 0, 0, 0],
                     [0, 6, 5, 0, 3, 2, 0, 0, 0],
                     [8, 7, 0, 5, 4, 0, 2, 1, 0],
                     [9, 8, 7, 6, 5, 4, 3, 2, 1],
                     [0, 9, 8, 0, 6, 5, 0, 3, 2],
                     [0, 0, 0, 8, 7, 0, 5, 4, 0],
                     [0, 0, 0, 9, 8, 7, 6, 5, 4],
                     [0, 0, 0, 0, 9, 8, 0, 6, 5]]),
            3,
            kernel=(np.arange(9) + 1).reshape((3, 3, 1, 1)), input_side=3, padding='same')

    # For testing, the indices of the test matrix are not ordered
    # by the real one K[row, col, chan_in, chan_out]
    # K[:, :, 0, 0] -> 1..9
    # K[:, :, 1, 0] -> 10..18
    # K[:, :, 0, 1] -> 19..27
    # K[:, :, 1, 1] -> 28..36
    K = (np.arange(36) + 1).reshape((2, 2, -1)).T.reshape((3, 3, 2, 2))


    tester(np.array([[ 1, 19,  0,  0,  0,  0,  0,  0],
                     [10, 28,  0,  0,  0,  0,  0,  0],
                     [ 2, 20,  1, 19,  0,  0,  0,  0],
                     [11, 29, 10, 28,  0,  0,  0,  0],
                     [ 3, 21,  2, 20,  0,  0,  0,  0],
                     [12, 30, 11, 29,  0,  0,  0,  0],
                     [ 0,  0,  3, 21,  0,  0,  0,  0],
                     [ 0,  0, 12, 30,  0,  0,  0,  0],
                     [ 4, 22,  0,  0,  1, 19,  0,  0],
                     [13, 31,  0,  0, 10, 28,  0,  0],
                     [ 5, 23,  4, 22,  2, 20,  1, 19],
                     [14, 32, 13, 31, 11, 29, 10, 28],
                     [ 6, 24,  5, 23,  3, 21,  2, 20],
                     [15, 33, 14, 32, 12, 30, 11, 29],
                     [ 0,  0,  6, 24,  0,  0,  3, 21],
                     [ 0,  0, 15, 33,  0,  0, 12, 30],
                     [ 7, 25,  0,  0,  4, 22,  0,  0],
                     [16, 34,  0,  0, 13, 31,  0,  0],
                     [ 8, 26,  7, 25,  5, 23,  4, 22],
                     [17, 35, 16, 34, 14, 32, 13, 31],
                     [ 9, 27,  8, 26,  6, 24,  5, 23],
                     [18, 36, 17, 35, 15, 33, 14, 32],
                     [ 0,  0,  9, 27,  0,  0,  6, 24],
                     [ 0,  0, 18, 36,  0,  0, 15, 33],
                     [ 0,  0,  0,  0,  7, 25,  0,  0],
                     [ 0,  0,  0,  0, 16, 34,  0,  0],
                     [ 0,  0,  0,  0,  8, 26,  7, 25],
                     [ 0,  0,  0,  0, 17, 35, 16, 34],
                     [ 0,  0,  0,  0,  9, 27,  8, 26],
                     [ 0,  0,  0,  0, 18, 36, 17, 35],
                     [ 0,  0,  0,  0,  0,  0,  9, 27],
                     [ 0,  0,  0,  0,  0,  0, 18, 36]]),
            2,
            kernel=K, input_side=4, padding='valid')


def test_expand_pool_layer():

    def tester(expected_layer, expected_output_side, **kwargs):
        layer, output_side, _ = expand_pool_layer(**kwargs)

        assert np.array_equal(expected_layer, layer)
        assert expected_output_side == output_side


    tester(np.ones((1, 1)), 1,
           pool_size=(1, 1), input_side=1, n_channels=1, with_avg=False)


    tester(np.array([[1., 0., 0., 0.],
                     [1., 0., 0., 0.],
                     [0., 1., 0., 0.],
                     [0., 1., 0., 0.],
                     [1., 0., 0., 0.],
                     [1., 0., 0., 0.],
                     [0., 1., 0., 0.],
                     [0., 1., 0., 0.],
                     [0., 0., 1., 0.],
                     [0., 0., 1., 0.],
                     [0., 0., 0., 1.],
                     [0., 0., 0., 1.],
                     [0., 0., 1., 0.],
                     [0., 0., 1., 0.],
                     [0., 0., 0., 1.],
                     [0., 0., 0., 1.]]),
          2,
          pool_size=(2, 2), input_side=4, n_channels=1, with_avg=False)         


    tester(np.array([[1., 0., 0., 0.],
                     [1., 0., 0., 0.],
                     [0., 1., 0., 0.],
                     [0., 1., 0., 0.],
                     [1., 0., 0., 0.],
                     [1., 0., 0., 0.],
                     [0., 1., 0., 0.],
                     [0., 1., 0., 0.],
                     [0., 0., 1., 0.],
                     [0., 0., 1., 0.],
                     [0., 0., 0., 1.],
                     [0., 0., 0., 1.],
                     [0., 0., 1., 0.],
                     [0., 0., 1., 0.],
                     [0., 0., 0., 1.],
                     [0., 0., 0., 1.]]) / 4,
          2,
          pool_size=(2, 2), input_side=4, n_channels=1, with_avg=True)


    # drop last row and last column
    tester(np.array([[1.],
                     [1.],
                     [1.],
                     [0.],
                     [1.],
                     [1.],
                     [1.],
                     [0.],
                     [1.],
                     [1.],
                     [1.],
                     [0.],
                     [0.],
                     [0.],
                     [0.],
                     [0.]]),
          1,
          pool_size=(3, 3), input_side=4, n_channels=1, with_avg=False)

    tester(np.array([[1, 0],
                     [0, 1],
                     [1, 0],
                     [0, 1],
                     [1, 0],
                     [0, 1],
                     [1, 0],
                     [0, 1]]),
          1,
          pool_size=(2, 2), input_side=2, n_channels=2, with_avg=False)

    tester(np.array([[1, 0],
                     [0, 1],
                     [1, 0],
                     [0, 1],
                     [1, 0],
                     [0, 1],
                     [1, 0],
                     [0, 1]]) / 4,
          1,
          pool_size=(2, 2), input_side=2, n_channels=2, with_avg=True)
