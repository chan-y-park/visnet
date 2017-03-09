import tensorflow as tf
import h5py

core_weights_file_path = (
    '/home/chan/.keras/models/'
    'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
)

full_weights_file_path = (
    '/home/chan/.keras/models/'
    'vgg16_weights_tf_dim_ordering_tf_kernels.h5'
)


class VGG16Core:
    def __init__(
        self,
        num_input_channels=3,
        input_size=224,
        batch_size=1,
        visualize=True,
    ):
        self.num_in_chs = num_input_channels
        self.input_size = input_size
        self.batch_size = batch_size
        self.visualize = visualize
        self.weights_f = h5py.File(core_weights_file_path, mode='r')

        self.layers = []
        self.max_pool_switches = []
        #self.max_unpoolings = []
        self.reconstructed_features = {}
        self._configuration = [
            ('block1',
                (
                    ('conv1', {'W': [3, 3, self.num_in_chs, 64], 'b': [64]}),
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

    def build(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self._build()

    def get_output_layer(self):
        return self.layers[-1]

    def _build(self):
        self.input_layer = tf.placeholder(
            tf.float32,
            shape=(
                self.batch_size,
                self.input_size,
                self.input_size,
                self.num_in_chs
            ),
            name='input_layer',
        )
        self.layers.append(self.input_layer)

        #weights_f = h5py.File(core_weights_file_path, mode='r')
        weights_f = self.weights_f
        for block_name, block_conf in self._configuration:
            with tf.variable_scope(block_name):
                for layer_name, layer_conf in block_conf:
                    with tf.variable_scope(layer_name):
                        prev_layer = self.layers[-1]
                        if 'conv' in layer_name:
                            conv_vars = {}
                            grp_name = block_name + '_' + layer_name
                            for var_name, var_shape in layer_conf.items():
                                dset_name = grp_name + '_' + var_name + '_1:0'
                                conv_vars[var_name] = tf.get_variable(
                                    var_name,
                                    shape=var_shape,
                                    initializer=tf.constant_initializer(
                                        weights_f[grp_name][dset_name].value
                                    )
                                )
                            tensor = tf.nn.conv2d(
                                prev_layer,
                                conv_vars['W'],
                                strides=[1, 1, 1, 1],
                                padding='SAME',
                            )
                            tensor = tf.nn.bias_add(
                                tensor,
                                conv_vars['b'],
                            )
                            new_layer = tf.nn.relu(
                                tensor,
                                #name=layer_name,
                            )
                        elif 'pool' in layer_name:
                            if self.visualize:
                                pool_f = tf.nn.max_pool_with_argmax
                            else:
                                pool_f = tf.nn.max_pool

                            rv = pool_f(
                                prev_layer,
                                ksize=([1] + layer_conf['k'] + [1]),
                                strides=([1] + layer_conf['s'] + [1]),
                                padding='SAME',
                                #name=layer_name,
                            )

                            if self.visualize:
                                new_layer, switches = rv
                                layer_conf['switches'] = switches 

                            else:
                                new_layer = rv

                        self.layers.append(new_layer)

#            self.reconstructed_features[block_name] = (
#                self.get_reconstructed_features(
#                    int(block_name[-1]),
#                    new_layer,
#                )
#            )
            if block_name == 'block2':
                self.reconstructed_features[block_name] = (
                    self.get_reconstructed_features(
                        int(block_name[-1]),
                        new_layer,
                    )
                )

#    def get_reconstructed_features(
#        self,
#        i_block,
#        features,
#    ):
#        recons = features
#        subconfig = self._configuration[:i_block]
#        weights_f = self.weights_f
#
#        for block_name, block_conf in reversed(subconfig):
#            # TODO: check features sizes.
#            for layer_name, layer_conf in reversed(block_conf):
#                if 'pool' in layer_name: 
#                    switches = layer_conf['switches']
#                    b, h, w, c = switches.shape.as_list()
#                    # XXX: Assume b = 1, k = 2, s = 2.
#                    unpooled_flattened = tf.scatter_nd(
#                        tf.reshape(switches, [-1, 1]),
#                        tf.reshape(recons, [-1]),
#                        [b * (2 * h) * (2 * w) * c],
#                    )
#                    recons = tf.reshape(
#                        unpooled_flattened,
#                        [b, (2 * h), (2 * w), c],
#                    )
#
#                elif 'conv' in layer_name:
#                    _, _, n_in_chs, n_out_chs = layer_conf['W']
#                    with tf.variable_scope(block_name, reuse=True):
#                        with tf.variable_scope(layer_name, reuse=True):
#                            W = tf.get_variable('W')
#                            b = tf.get_variable('b')
#                    recons = tf.nn.bias_add(recons, -b)
#                    recons = tf.nn.conv2d_transpose(
#                        recons,
#                        W,
#                        output_shape=recons.shape.as_list()[:-1] + [n_in_chs],
#                        strides=[1, 1, 1, 1],
#                        padding='SAME',
#                    )
#
#        return recons

    def get_reconstructed_features(
        self,
        i_block,
        features,
    ):
        #recons = features
        subconfig = self._configuration[:i_block]
        weights_f = self.weights_f

        # XXX: Assume f_b = 1.
        f_b, f_h, f_w, n_features = features.shape.as_list()
        
        recons = None

        for block_name, block_conf in reversed(subconfig):
            # TODO: check features sizes.
            for layer_name, layer_conf in reversed(block_conf):
                if 'pool' in layer_name: 
                    switches = layer_conf['switches']
                    b, h, w, c = switches.shape.as_list()

                    if recons is None:
                        recons = [
                            tf.scatter_nd(
                                tf.reshape(switches[:, :, :, i_f], [-1, 1]),
                                tf.reshape(features[:, :, :, i_f], [-1]),
                                [b * (2 * h) * (2 * w) * c],
                            ) for i_f in range(n_features)   
                        ]

                        recons = tf.concat(recons, axis=0)
                        recons = tf.reshape(
                            recons,
                            [n_features, (2 * h), (2 * w), c],
                        )
                    else:
                        # XXX: Assume b = 1, k = 2, s = 2.
                        unpooled_flattened_tensors = [
                            tf.scatter_nd(
                                tf.reshape(switches, [-1, 1]),
                                tf.reshape(recons[i_f, :, :, :], [-1]),
                                [b * (2 * h) * (2 * w) * c],
                            )
                            for i_f in range(n_features)
                        ]
                        unpooled_flattened = tf.concat(
                            unpooled_flattened_tensors,
                            axis=0,
                        )
                        recons = tf.reshape(
                            unpooled_flattened,
                            [n_features, (2 * h), (2 * w), c],
                        )

                elif 'conv' in layer_name:
                    # XXX: Where to put ReLU?
                    recons = tf.nn.relu(recons)
                    _, _, n_in_chs, n_out_chs = layer_conf['W']
                    with tf.variable_scope(block_name, reuse=True):
                        with tf.variable_scope(layer_name, reuse=True):
                            W = tf.get_variable('W')
                            b = tf.get_variable('b')
                    recons = tf.nn.bias_add(recons, -b)
                    recons = tf.nn.conv2d_transpose(
                        recons,
                        W,
                        output_shape=recons.shape.as_list()[:-1] + [n_in_chs],
                        strides=[1, 1, 1, 1],
                        padding='SAME',
                    )

        return recons
             

    def get_logits(self):
        pass


class VGG16(VGG16Core):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.weights_f = h5py.File(full_weights_file_path, mode='r')
        self._top_configuration = (
            ('flatten', ()),
            ('fc1', (4096)),
            ('fc2', (4096)),
            ('predictions', (1000)),
        )

    def _build(self):
        super()._build()

        #weights_f = h5py.File(full_weights_file_path, mode='r')
        weights_f = self.weights_f
        with tf.variable_scope('top'):
            for layer_name, layer_conf in self._top_configuration:
                with tf.variable_scope(layer_name):
                    prev_layer = self.layers[-1]
                    if 'flatten' in layer_name:
                        new_layer = tf.reshape(
                            prev_layer,
                            [self.batch_size, -1],
                        )

                    else:
                        if 'fc' in layer_name:
                            f_layer = tf.nn.relu    
                        elif 'predictions' in layer_name:
                            f_layer = tf.nn.softmax
                        else:
                            raise NotImplementedError

                        input_dim = prev_layer.shape[-1].value
                        output_dim = layer_conf
                        layer_vars = {}
                        for var_name, var_shape in (
                            ('W', (input_dim, output_dim)),
                            ('b', (output_dim)),
                        ):
                            dset_name = (layer_name + '_' +
                                         var_name + '_1:0')
                            layer_vars[var_name] = tf.get_variable(
                                var_name,
                                shape=var_shape,
                                initializer=tf.constant_initializer(
                                    weights_f[layer_name][dset_name].value
                                )
                            )
                        activation = tf.add(
                            tf.matmul(prev_layer, layer_vars['W']),
                            layer_vars['b'],
                            #name=layer_name + '_activation',
                            name='activation',
                        )
                        new_layer = f_layer(
                            activation,
                            #name = layer_name,
                        )

                    self.layers.append(new_layer)
