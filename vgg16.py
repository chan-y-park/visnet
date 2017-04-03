from visnet import VisNet

core_weights_file_path = (
    '/home/chan/.keras/models/'
    'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
)

full_weights_file_path = (
    '/home/chan/.keras/models/'
    'vgg16_weights_tf_dim_ordering_tf_kernels.h5'
)

class VGG16Core(VisNet):
    def __init__(
        self,
        **kwargs
    ):
        self._configuration = [
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
        ]
        super().__init__(
            weights_file_path=core_weights_file_path,
            **kwargs
        )

class VGG16(VGG16Core):
    def __init__(self, **kwargs):
        self._configuration += [
            ('top',
                (
                    ('flatten', ()),
                    ('fc1', (4096)),
                    ('fc2', (4096)),
                    ('predictions', (1000)),
                )
            ),
        ]
        super().__init__(
            weights_file_path=full_weights_file_path,
            **kwargs
        )

#    def _build_top_layers(self):
#        prev_layer = super().get_output_layer()
#
#        weights_f = self.weights_f
#        with tf.variable_scope('top'):
#            for layer_name, layer_conf in self._top_configuration:
#                with tf.variable_scope(layer_name):
#                    if 'flatten' in layer_name:
#                        new_layer = tf.reshape(
#                            prev_layer,
#                            [self.batch_size, -1],
#                        )
#
#                    else:
#                        if 'fc' in layer_name:
#                            f_layer = tf.nn.relu    
#                        elif 'predictions' in layer_name:
#                            f_layer = tf.nn.softmax
#                        else:
#                            raise NotImplementedError
#
#                        input_dim = prev_layer.shape[-1].value
#                        output_dim = layer_conf
#                        layer_vars = {}
#                        for var_name, var_shape in (
#                            ('W', (input_dim, output_dim)),
#                            ('b', (output_dim)),
#                        ):
#                            dset_name = (layer_name + '_' +
#                                         var_name + '_1:0')
#                            layer_vars[var_name] = tf.get_variable(
#                                var_name,
#                                shape=var_shape,
#                                initializer=tf.constant_initializer(
#                                    weights_f[layer_name][dset_name].value
#                                )
#                            )
#                        activation = tf.add(
#                            tf.matmul(prev_layer, layer_vars['W']),
#                            layer_vars['b'],
#                            #name=layer_name + '_activation',
#                            name='activation',
#                        )
#                        new_layer = f_layer(
#                            activation,
#                            #name = layer_name,
#                        )
#
#                    self.layers[layer_name] = new_layer
#                    prev_layer = new_layer
