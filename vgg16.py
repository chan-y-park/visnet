vgg16_no_fc_config = {
    'weights_file_path': (
        'models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
    ),
    'network': [
        ('block1',
            (
                ('conv1', {'W': [3, 3, 3, 64], 'b': [64]}),
                ('conv2', {'W': [3, 3, 64, 64], 'b': [64]}),
                ('pool', {'k': [2, 2], 's': [2, 2]}),
            )
        ),
        ('block2',
            (
                ('conv1', {'W': [3, 3, 64, 128], 'b': [128]}),
                ('conv2', {'W': [3, 3, 128, 128], 'b': [128]}),
                ('pool', {'k': [2, 2], 's': [2, 2]}),
            )
        ),
        ('block3',
            (
                ('conv1', {'W': [3, 3, 128, 256], 'b': [256]}),
                ('conv2', {'W': [3, 3, 256, 256], 'b': [256]}),
                ('conv3', {'W': [3, 3, 256, 256], 'b': [256]}),
                ('pool', {'k': [2, 2], 's': [2, 2]}),
            )
        ),
        ('block4',
            (
                ('conv1', {'W': [3, 3, 256, 512], 'b': [512]}),
                ('conv2', {'W': [3, 3, 512, 512], 'b': [512]}),
                ('conv3', {'W': [3, 3, 512, 512], 'b': [512]}),
                ('pool', {'k': [2, 2], 's': [2, 2]}),
            )
        ),
        ('block5', 
            (
                ('conv1', {'W': [3, 3, 512, 512], 'b': [512]}),
                ('conv2', {'W': [3, 3, 512, 512], 'b': [512]}),
                ('conv3', {'W': [3, 3, 512, 512], 'b': [512]}),
                ('pool', {'k': [2, 2], 's': [2, 2]}),
            )
        ),
    ],
}


vgg16_full_config = {
    'weights_file_path': (
        'models/vgg16_weights_tf_dim_ordering_tf_kernels.h5'
    ),
    'network': vgg16_no_fc_config['network'] + [
        ('top',
            (
                ('flatten', ()),
                ('fc1', (4096)),
                ('fc2', (4096)),
                ('predictions', (1000)),
            )
        ),
    ]
}
