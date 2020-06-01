CNN_MODEL_PARAMS = {'conv': [{'filters': 32, 'kernel_size': (3,3), 'padding': 'same',
                               'max_pool_after': False},
                             {'filters': 32, 'kernel_size': (3,3), 'padding': 'same',
                              'max_pool_after': True,
                              'max_pool_size': (2,2),
                              'max_pool_padding': 'valid'}],
                   'dense': [128]}


# CNN_MODEL_PARAMS = {'conv': [{'filters': 32, 'kernel_size': (3,3), 'padding': 'same',
#                               'max_pool_after': False},
#                              {'filters': 32, 'kernel_size': (3,3), 'padding': 'valid',
#                               'max_pool_after': True,
#                               'max_pool_size': (2,2),
#                               'max_pool_padding': 'valid'},
#                              {'filters': 64, 'kernel_size': (3,3), 'padding': 'same',
#                               'max_pool_after': False},
#                              {'filters': 64, 'kernel_size': (3,3), 'padding': 'valid',
#                               'max_pool_after': True,
#                               'max_pool_size': (2,2),
#                               'max_pool_padding': 'valid'},
#                             ],
#                     'dense': [512]
#                    }

