import numpy as np
import tensorflow as tf
import h5py


class _weights:
    def __init__(self, nparray):
        self.value = nparray


class VisNet:
    def __init__(
        self,
        num_input_channels=3,
        input_size=224,
        batch_size=1,
        visualize=True,
        use_test_filters=False,
        logdir='log',
        weights_file_path=None,
    ):
        #XXX The following does not work, implement a similar check.
        #assert(hasattr(self, _configuration))
        self.num_in_chs = num_input_channels
        self.input_size = input_size
        self.batch_size = batch_size
        self.visualize = visualize

        self.layers = None
        self._forward_layers = None
        self.backprop_layers = None
        self.max_pool_switches = None

        if use_test_filters or (weights_file_path is None):
            self.weights_f = self._get_test_filters()
        else:
            self.weights_f = h5py.File(weights_file_path, mode='r')

#        self.graph = tf.Graph()
#        with self.graph.as_default():
#            self._build_network()

        self._forward_graph = tf.Graph()
        with self._forward_graph.as_default():
            self._load_weights()
            self._build_forward_graph()

    def _load_weights(self):
        for block_name, block_conf in self._configuration:
            for layer_name, layer_conf in block_conf:
                block_layer_name = block_name + '_' + layer_name
                if 'conv' in layer_name:
                    for var_name, var_shape in layer_conf.items():
                        dset_name = (block_layer_name + '_' +
                                     var_name + '_1:0')
                        with tf.variable_scope(block_name):
                            with tf.variable_scope(layer_name):
                                tf.get_variable(
                                    var_name,
                                    shape=var_shape,
                                    initializer=tf.constant_initializer(
                                        self.weights_f
                                        [block_layer_name]
                                        [dset_name]
                                        .value
                                    )
                                )

#    def get_activations(self, input_array):
#        with self.graph.as_default():
#            init = tf.global_variables_initializer()
#            tf_session = tf.Session()
#            tf_session.run(init)
#
#            fetches = {}
#            for name, tensor in self.layers.items():
#                fetches[name + '_activations'] = tensor
#            for name, tensor in self.max_pool_switches.items():
#                fetches[name + '_swithces'] = tensor
#            rd = tf_session.run(
#                fetches,
#                feed_dict={self.layers['input']: input_array},
#            )
#            rd['tf_session'] = tf_session
#        return rd

    def get_forward_results(self, input_array):
        with self._forward_graph.as_default():
            init = tf.global_variables_initializer()
            tf_session = tf.Session()
            tf_session.run(init)

            fetches = {
                'activations': {},
                'switches': {},
            }
            for name, tensor in self._forward_layers.items():
                fetches['activations'][name] = tensor
            for name, tensor in self.max_pool_switches.items():
                fetches['switches'][name] = tensor
            rd = tf_session.run(
                fetches,
                feed_dict={self._forward_layers['input']: input_array},
            )
            rd['tf_session'] = tf_session
        return rd

    def get_backprop_result(
        self,
        block_name,
        input_features,
        max_pool_switches,
    ):
        i_block = int(block_name[-1])
        subconfig = self._configuration[:i_block]
        layer_name = 'pool'
        block_layer_name = block_name + '_' + layer_name
        num_features = len(input_features)

        self.backprop_graph = tf.Graph()
        with self.backprop_graph.as_default():
            self._load_weights()

            recons = None
            for block_name, block_conf in reversed(subconfig):
                # TODO: check features sizes.
                for layer_name, layer_conf in reversed(block_conf):
                    if 'pool' in layer_name: 
                        block_layer_name = block_name + '_' + layer_name
                        switches = tf.constant(
                            max_pool_switches[block_layer_name]
                        )
                        b, h, w, c = switches.shape.as_list()

                        if recons is None:
                            features = [
                                (i_f, tf.constant(a_feature_array))
                                for i_f, a_feature_array
                                in input_features
                            ]
                            # XXX: Incompatible if the input size 
                            # of the pooling is not a proper multiple 
                            # (multiple of 32 for VGG16).
                            recons = [
                                tf.scatter_nd(
                                    tf.reshape(switches[0, :, :, i_f], [-1, 1]),
                                    tf.reshape(a_feature[:, :], [-1]),
                                    [b * (2 * h) * (2 * w) * c],
                                ) for i_f, a_feature in features 
                            ]

                            recons = tf.concat(recons, axis=0)
                            recons = tf.reshape(
                                recons,
                                [num_features, (2 * h), (2 * w), c],
                            )
                        else:
                            # XXX: Assume b = 1, k = 2, s = 2.
                            unpooled_flattened_tensors = [
                                tf.scatter_nd(
                                    tf.reshape(switches, [-1, 1]),
                                    tf.reshape(recons[i_f, :, :, :], [-1]),
                                    [b * (2 * h) * (2 * w) * c],
                                )
                                for i_f in range(num_features)
                            ]
                            unpooled_flattened = tf.concat(
                                unpooled_flattened_tensors,
                                axis=0,
                            )
                            recons = tf.reshape(
                                unpooled_flattened,
                                [num_features, (2 * h), (2 * w), c],
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
                            output_shape=(
                                recons.shape.as_list()[:-1] + [n_in_chs]
                            ),
                            strides=[1, 1, 1, 1],
                            padding='SAME',
                        )

            init = tf.global_variables_initializer()
            tf_session = tf.Session()
            tf_session.run(init)

            return tf_session.run(
                [recons],
            )

    def get_visualization(self, block_layer_name, feature_id):
        pass

    def _get_test_filters(self, a_filter='one'):
        """
        Pack the given test filter into a dict
        according to the format of the weights hdf5 file
        to test the backpropagation.
        """
        weights = {}
        for block_name, block_conf in self._configuration:
            for layer_name, layer_conf in block_conf:
                if 'conv' in layer_name:
                    block_layer_name = block_name + '_' + layer_name
                    weights[block_layer_name] = {}
                    for var_name, var_shape in layer_conf.items():
                        dset_name = block_layer_name + '_' + var_name + '_1:0'
                        if var_name == 'W':
                            h, w, n_in, n_out = var_shape
                            if a_filter == 'one':
                                W = np.ones((h, w), dtype=np.float32)
                            elif a_filter == 'zero':
                                W = np.zeros((h, w), dtype=np.float32)
                            elif a_filter == 'identity':
                                W = np.array(
                                    [[0, 0, 0],
                                     [0, 1, 0],
                                     [0, 0, 0]],
                                    dtype=np.float32
                                )
                            else:
                                W = np.array(a_filter, dtype=np.float)
                            var = _weights(
                                np.array(
                                    [[W for _ in range(n_out)]
                                     for _ in range(n_in)]
                                ).transpose((2, 3, 0, 1))
                            )
                        elif var_name == 'b':
                            var = _weights(
                                np.zeros(var_shape, dtype=np.float32)
                            )
                        else:
                            raise RuntimeError
                        weights[block_layer_name][dset_name] = var
        return weights

    def _get_output_layer(self):
        output_block_name, output_block_conf = self._configuration[-1]
        output_layer_name, output_layer_conf = output_block_conf[-1]
        return self.layers[output_block_name + '_' + output_layer_name]

#    def _build_network(self):
#        input_layer = tf.placeholder(
#            tf.float32,
#            shape=(
#                self.batch_size,
#                self.input_size,
#                self.input_size,
#                self.num_in_chs
#            ),
#            name='input_layer',
#        )
#        self.layers = {'input': input_layer}
#        prev_layer = input_layer
#
#        if self.visualize:
#            pool_f = tf.nn.max_pool_with_argmax
#            self.max_pool_switches = {}
#        else:
#            pool_f = tf.nn.max_pool
#
#        weights_f = self.weights_f
#        for block_name, block_conf in self._configuration:
#            with tf.variable_scope(block_name):
#                for layer_name, layer_conf in block_conf:
#                    block_layer_name = block_name + '_' + layer_name
#                    with tf.variable_scope(layer_name):
#                        if 'conv' in layer_name:
#                            conv_vars = {}
#                            for var_name, var_shape in layer_conf.items():
#                                dset_name = (block_layer_name + '_' +
#                                             var_name + '_1:0')
#                                conv_vars[var_name] = tf.get_variable(
#                                    var_name,
#                                    shape=var_shape,
#                                    initializer=tf.constant_initializer(
#                                        weights_f[block_layer_name][dset_name]
#                                        .value
#                                    )
#                                )
#                            tensor = tf.nn.conv2d(
#                                prev_layer,
#                                conv_vars['W'],
#                                strides=[1, 1, 1, 1],
#                                padding='SAME',
#                            )
#                            tensor = tf.nn.bias_add(
#                                tensor,
#                                conv_vars['b'],
#                            )
#                            new_layer = tf.nn.relu(
#                                tensor,
#                                #name=layer_name,
#                            )
#                        elif 'pool' in layer_name:
#                            rv = pool_f(
#                                prev_layer,
#                                ksize=([1] + layer_conf['k'] + [1]),
#                                strides=([1] + layer_conf['s'] + [1]),
#                                padding='SAME',
#                                #name=layer_name,
#                            )
#
#                            if self.visualize:
#                                new_layer, switches = rv
#                                self.max_pool_switches[
#                                    block_layer_name
#                                ] = switches
#
#                            else:
#                                new_layer = rv
#
#                        elif 'flatten' in layer_name:
#                            new_layer = tf.reshape(
#                                prev_layer,
#                                [self.batch_size, -1],
#                            )
#
#                        elif 'fc' in layer_name or 'predictions' in layer_name:
#                            if 'fc' in layer_name:
#                                f_layer = tf.nn.relu    
#                            elif 'predictions' in layer_name:
#                                f_layer = tf.nn.softmax
#
#                            input_dim = prev_layer.shape[-1].value
#                            output_dim = layer_conf
#                            layer_vars = {}
#                            for var_name, var_shape in (
#                                ('W', (input_dim, output_dim)),
#                                ('b', (output_dim)),
#                            ):
#                                dset_name = (layer_name + '_' +
#                                             var_name + '_1:0')
#                                layer_vars[var_name] = tf.get_variable(
#                                    var_name,
#                                    shape=var_shape,
#                                    initializer=tf.constant_initializer(
#                                        weights_f[layer_name][dset_name].value
#                                    )
#                                )
#                            activation = tf.add(
#                                tf.matmul(prev_layer, layer_vars['W']),
#                                layer_vars['b'],
#                                #name=layer_name + '_activation',
#                                name='activation',
#                            )
#                            new_layer = f_layer(
#                                activation,
#                                #name = layer_name,
#                            )
#                        
#                        else:
#                            raise NotImplementedError
#
#                        # End of building a layer.
#
#                        self.layers[block_layer_name] = new_layer
#                        prev_layer = new_layer

    def _build_forward_graph(self):
        input_layer = tf.placeholder(
            tf.float32,
            shape=(
                self.batch_size,
                self.input_size,
                self.input_size,
                self.num_in_chs
            ),
            name='input_layer',
        )
        self._forward_layers = {'input': input_layer}
        prev_layer = input_layer

        if self.visualize:
            pool_f = tf.nn.max_pool_with_argmax
            self.max_pool_switches = {}
        else:
            pool_f = tf.nn.max_pool

        weights_f = self.weights_f
        for block_name, block_conf in self._configuration:
            for layer_name, layer_conf in block_conf:
                block_layer_name = block_name + '_' + layer_name
                if 'conv' in layer_name:
#                    conv_vars = {}
#                    for var_name, var_shape in layer_conf.items():
#                        dset_name = (block_layer_name + '_' +
#                                     var_name + '_1:0')
#                        with tf.variable_scope(block_name):
#                            with tf.variable_scope(layer_name):
#                                conv_vars[var_name] = tf.get_variable(
#                                    var_name,
#                                    shape=var_shape,
#                                    initializer=tf.constant_initializer(
#                                        weights_f[block_layer_name][dset_name]
#                                        .value
#                                    )
#                                )
                    with tf.variable_scope(block_name, reuse=True):
                        with tf.variable_scope(layer_name, reuse=True):
                            W = tf.get_variable('W')
                            b = tf.get_variable('b')
                    tensor = tf.nn.conv2d(
                        prev_layer,
                        #conv_vars['W'],
                        W,
                        strides=[1, 1, 1, 1],
                        padding='SAME',
                    )
                    tensor = tf.nn.bias_add(
                        tensor,
                        #conv_vars['b'],
                        b,
                    )
                    new_layer = tf.nn.relu(
                        tensor,
                        #name=layer_name,
                    )
                elif 'pool' in layer_name:
                    rv = pool_f(
                        prev_layer,
                        ksize=([1] + layer_conf['k'] + [1]),
                        strides=([1] + layer_conf['s'] + [1]),
                        padding='SAME',
                        #name=layer_name,
                    )

                    if self.visualize:
                        new_layer, switches = rv
                        self.max_pool_switches[
                            block_layer_name
                        ] = switches

                    else:
                        new_layer = rv

                elif 'flatten' in layer_name:
                    new_layer = tf.reshape(
                        prev_layer,
                        [self.batch_size, -1],
                    )

                elif 'fc' in layer_name or 'predictions' in layer_name:
                    if 'fc' in layer_name:
                        f_layer = tf.nn.relu    
                    elif 'predictions' in layer_name:
                        f_layer = tf.nn.softmax

                    input_dim = prev_layer.shape[-1].value
                    output_dim = layer_conf
                    layer_vars = {}
                    for var_name, var_shape in (
                        ('W', (input_dim, output_dim)),
                        ('b', (output_dim)),
                    ):
                        dset_name = (layer_name + '_' +
                                     var_name + '_1:0')
                        with tf.variable_scope(block_name):
                            with tf.variable_scope(layer_name):
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
                
                else:
                    raise NotImplementedError

                # End of building a layer.

                self._forward_layers[block_layer_name] = new_layer
                prev_layer = new_layer

    def get_output(self, tf_session, input_array):
        t_output = self._get_output_layer()
        return tf_session.run(
            [t_output],
            feed_dict={self.layers['input']: input_array},
        )

    def get_reconstructed_top_features(
        self,
        tf_session,
        input_array,
        block_name,
        num_top_features=9,
        reconstruction_method='deconv',
    ):
        assert(self.visualize)
        assert(self.max_pool_switches is not None)

        i_block = int(block_name[-1])
        subconfig = self._configuration[:i_block]
        layer_name = 'pool'
        block_layer_name = block_name + '_' + layer_name
        features = self.layers[block_layer_name]

        with self.graph.as_default(): 
            assert(features.shape.as_list()[0] == 1)
            norms = tf.norm(features[0], axis=[0, 1])
            _, tops = tf.nn.top_k(norms, k=num_top_features)

            if reconstruction_method == 'deconv':
                recons = self._get_deconved_features(
                    features,
                    subconfig,
                    tops,
                    num_top_features=num_top_features,
                )

        fetches ={
            'top_labels': tops,
            'reconstructed_features': recons,
            'activations': features,
        }

        return tf_session.run(
            fetches,
            feed_dict={self.layers['input']: input_array},
        )

    def _get_deconved_features(
        self,
        features,
        subconfig,
        tops,
        num_top_features=9,
    ):
        # XXX: Assume f_b = 1.
        #f_b, f_h, f_w, n_features = features.shape.as_list()
        
        recons = None

        for block_name, block_conf in reversed(subconfig):
            # TODO: check features sizes.
            for layer_name, layer_conf in reversed(block_conf):
                if 'pool' in layer_name: 
                    block_layer_name = block_name + '_' + layer_name
                    switches = self.max_pool_switches[block_layer_name]
                    b, h, w, c = switches.shape.as_list()

                    if recons is None:
                        # XXX: Incompatible if the input size of the pooling
                        # is not a proper multiple (multiple of 32 for VGG16).
                        recons = [
                            tf.scatter_nd(
                                tf.reshape(switches[:, :, :, tops[i]], [-1, 1]),
                                tf.reshape(features[:, :, :, tops[i]], [-1]),
                                [b * (2 * h) * (2 * w) * c],
                            ) for i in range(num_top_features) 
                        ]

                        recons = tf.concat(recons, axis=0)
                        recons = tf.reshape(
                            recons,
                            [num_top_features, (2 * h), (2 * w), c],
                        )
                    else:
                        # XXX: Assume b = 1, k = 2, s = 2.
                        unpooled_flattened_tensors = [
                            tf.scatter_nd(
                                tf.reshape(switches, [-1, 1]),
                                tf.reshape(recons[i_f, :, :, :], [-1]),
                                [b * (2 * h) * (2 * w) * c],
                            )
                            for i_f in range(num_top_features)
                        ]
                        unpooled_flattened = tf.concat(
                            unpooled_flattened_tensors,
                            axis=0,
                        )
                        recons = tf.reshape(
                            unpooled_flattened,
                            [num_top_features, (2 * h), (2 * w), c],
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

                # XXX: Rescale the reconstructions to prevent overflow.
                #recons = recons / tf.reduce_max(recons)
                #self.recons.append(recons)

        return recons
             

    def get_logits(self):
        pass


