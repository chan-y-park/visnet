import numpy as np
import tensorflow as tf
import h5py

from vgg16 import vgg16_no_fc_config, vgg16_full_config

class _weights:
    def __init__(self, nparray):
        self.value = nparray


class VisNet:
    def __init__(
        self,
        config=vgg16_no_fc_config,
        num_input_channels=3,
        input_size=224,
        batch_size=1,
        visualize=True,
        use_test_filters=False,
        logdir=None,
        use_cpu=False,
        full_deconv=True,
    ):
        #XXX The following does not work, implement a similar check.
        self.num_in_chs = num_input_channels
        self.input_size = input_size
        self.batch_size = batch_size
        self.visualize = visualize
        self.logdir = logdir

        self._config = config

        self._num_top_features = 9

        self._tensor_value = None
    
        self._forward_layers = None
        self._deconv_tensor = None
        self._max_pool_switches = None

        if use_test_filters:
            self.weights_f = self._get_test_filters()
        else:
            self.weights_f = h5py.File(
                self._config['weights_file_path'],
                mode='r',
            )

        self._tf_graph = tf.Graph()
        with self._tf_graph.as_default():
            with tf.variable_scope('forward_network'):
                self._build_forward_network()
            with tf.variable_scope('deconv_network'):
                if full_deconv:
                    self._build_full_deconv_network()
                else:
                    if use_cpu:
                        device = '/cpu:0'
                    else:
                        device = None
                    with tf.device(device):
                        # NOTE: tf.max_pool_with_argmax
                        # does not have a CPU kernel, so
                        # the forward graph cannot be
                        # placed on CPU.
                        self._build_deconv_network()

        if self.logdir is not None:
            self._summary_writer = tf.summary.FileWriter(
                logdir=self.logdir,
                graph=self._tf_graph,
            )
        else:
            self._summary_writer = None

    def get_output(self, tf_session, input_array):
        t_output = self._get_output_layer()
        with self._tf_graph.as_default():
            init = tf.global_variables_initializer()
            tf_session = tf.Session()
            tf_session.run(init)

            return tf_session.run(
                [t_output],
                feed_dict={self._forward_layers['input']: input_array},
            )

    def get_forward_results(
        self,
        input_array,
        log_device_placement=False,
    ):
        with self._tf_graph.as_default():

            config = tf.ConfigProto(
                log_device_placement=log_device_placement,
            )
            tf_session = tf.Session(config=config)

            init = tf.global_variables_initializer()
            tf_session.run(init)

            fetches = {
                'activations': {},
                'switches': {},
            }
            for name, tensor in self._forward_layers.items():
                fetches['activations'][name] = tensor
            for name, tensor in self._max_pool_switches.items():
                fetches['switches'][name] = tensor
            rd = tf_session.run(
                fetches,
                feed_dict={self._forward_layers['input']: input_array},
            )

        self._tensor_value = rd
        return rd

    def get_deconv_result(
        self,
        block_name,
        layer_name,
        log_device_placement=False,
    ):
        if self._tensor_value is None:
            raise RuntimeError(
                'No stored values of activations and switches.'
            )
        
        block_layer_name = block_name + '_' + layer_name
        input_name = block_layer_name + '_input'
        
        # XXX Assume batch_size = 1.
        all_features = self._tensor_value['activations'][block_layer_name][0]
        h, w, c = all_features.shape
        input_features = np.zeros(
            (self._num_top_features, h, w, c),
            dtype=np.float32,
        )
        
        i_top_f = np.argsort(
            np.linalg.norm(all_features, axis=(0, 1))
        )[-self._num_top_features:][::-1]

        for j in range(self._num_top_features):
            input_features[j,:,:,i_top_f[j]] = all_features[:,:,i_top_f[j]]

        with self._tf_graph.as_default():
    
            config = tf.ConfigProto(
                log_device_placement=log_device_placement,
            )
            tf_session = tf.Session(config=config)

            init = tf.global_variables_initializer()
            tf_session.run(init)

            feed_dict = {self._deconv_tensor[input_name]: input_features}
            for name, tensor in self._max_pool_switches.items():
                feed_dict[tensor] = self._tensor_value['switches'][name]
#                switches = self._deconv_tensor[name + '_switches']
#                feed_dict[switches] = self._tensor_value['switches'][name]

            rv = tf_session.run(
                self._deconv_tensor['output'],
                feed_dict=feed_dict,
            )

        return rv, i_top_f

    def get_full_deconv_result(
        self,
        input_array,
        log_device_placement=False,
    ):

        with self._tf_graph.as_default():
    
            config = tf.ConfigProto(
                log_device_placement=log_device_placement,
            )
            tf_session = tf.Session(config=config)

            init = tf.global_variables_initializer()
            tf_session.run(init)

            full_recons, full_labels = tf_session.run(
                [self._deconv_tensor['full_recons'],
                 self._deconv_tensor['full_labels']],
                feed_dict={self._forward_layers['input']: input_array},
            )

        rd = {}
        i_b = 0
        for block_name, _ in self._config['network']:
            block_layer_name = block_name + '_pool'
            stop = i_b + self._num_top_features
            rd[block_layer_name] = {
                'recons': full_recons[i_b:stop, :, :, :],
                'labels': full_labels[i_b:stop],
            }
            i_b = stop
        return rd
                

    def _get_test_filters(self, a_filter='one'):
        """
        Pack the given test filter into a dict
        according to the format of the weights hdf5 file
        to test the backpropagation.
        """
        weights = {}
        for block_name, block_conf in self._config['network']:
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
        output_block_name, output_block_conf = self._config['network'][-1]
        output_layer_name, output_layer_conf = output_block_conf[-1]
        block_layer_name = output_block_name + '_' + output_layer_name
        return self._forward_layers[block_layer_name]

    def _get_weights(self, block_layer_name, var_name, var_shape):
        dset_name = block_layer_name + '_' + var_name + '_1:0'
        return tf.get_variable(
            var_name,
            shape=var_shape,
            initializer=tf.constant_initializer(
                self.weights_f
                [block_layer_name]
                [dset_name]
                .value
            )
        )

    def _build_forward_network(self):
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
            self._max_pool_switches = {}
        else:
            pool_f = tf.nn.max_pool

        for block_name, block_conf in self._config['network']:
            with tf.variable_scope(block_name):
                for layer_name, layer_conf in block_conf:
                    with tf.variable_scope(layer_name):
                        block_layer_name = block_name + '_' + layer_name
                        if 'conv' in layer_name:
                            conv_var = {}
                            for var_name, var_shape in layer_conf.items():
                                conv_var[var_name] = self._get_weights(
                                    block_layer_name,
                                    var_name,
                                    var_shape,
                                )
                            tensor = tf.nn.conv2d(
                                prev_layer,
                                #W,
                                conv_var['W'],
                                strides=[1, 1, 1, 1],
                                padding='SAME',
                            )
                            tensor = tf.nn.bias_add(
                                tensor,
                                #b,
                                conv_var['b']
                            )
                            new_layer = tf.nn.relu(
                                tensor,
                            )
                        elif 'pool' in layer_name:
                            rv = pool_f(
                                prev_layer,
                                ksize=([1] + layer_conf['k'] + [1]),
                                strides=([1] + layer_conf['s'] + [1]),
                                padding='SAME',
                            )

                            if self.visualize:
                                new_layer, switches = rv
                                self._max_pool_switches[
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
                            layer_var = {}
                            for var_name, var_shape in (
                                ('W', (input_dim, output_dim)),
                                ('b', (output_dim)),
                            ):
                                layer_var[var_name] = self._get_weights(
                                    layer_name,
                                    var_name,
                                    var_shape,
                                )
                            activation = tf.add(
                                tf.matmul(prev_layer, layer_var['W']),
                                layer_var['b'],
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

    def _build_deconv_network(self):
        # TODO: check features sizes.
        num_features = self._num_top_features 
        max_pool_switches = self._max_pool_switches
        self._deconv_tensor = {}

#        for name, tensor in self._max_pool_switches.items():
#            switches_name = name + '_switches'
#            self._deconv_tensor[switches_name] = tf.placeholder(
#                tf.int64,
#                shape=tensor.shape,
#                name=switches_name,
#            )

        output_layer = self._get_output_layer()
        b, h, w, c = output_layer.shape.as_list()
        recons = tf.placeholder(
            tf.float32,
            shape=(num_features, h, w, c),
            name='forward_output',
        )


        for block_name, block_conf in reversed(self._config['network']):
            with tf.variable_scope(block_name):
                for layer_name, layer_conf in reversed(block_conf):
                    block_layer_name = block_name + '_' + layer_name

                    self._deconv_tensor[block_layer_name + '_input'] = recons

                    with tf.variable_scope(layer_name):
                        if 'pool' in layer_name: 
                            switches = max_pool_switches[block_layer_name]
#                            switches = self._deconv_tensor[
#                                block_layer_name + '_switches'
#                            ]
                            b, h, w, c = switches.shape.as_list()

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
                            W_shape = layer_conf['W']
                            _, _, n_in_chs, n_out_chs = W_shape
                            W = self._get_weights(
                                block_layer_name,
                                'W',
                                W_shape,
                            )
                            b = self._get_weights(
                                block_layer_name,
                                'b',
                                layer_conf['b'],
                            )
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

        self._deconv_tensor['output'] = recons

    def _build_full_deconv_network(self):
        num_features = self._num_top_features 
        max_pool_switches = self._max_pool_switches
        self._deconv_tensor = {}
        recons = None
        labels = None

        for block_name, block_conf in reversed(self._config['network']):
            with tf.variable_scope(block_name):
                for layer_name, layer_conf in reversed(block_conf):
                    block_layer_name = block_name + '_' + layer_name
                    with tf.variable_scope(layer_name):
                        if 'pool' in layer_name: 
                            switches = max_pool_switches[block_layer_name]
                            b, h, w, c = switches.shape.as_list()

                            # Extract the top features
                            # from the pool layer output
                            # of the forward network.
                            block_fwd_output = self._forward_layers[
                                block_layer_name
                            ][0]
                            _, i_top_fs = tf.nn.top_k(
                                tf.norm(block_fwd_output, axis=(0, 1)),
                                k=num_features,
                            )
                            if labels is None:
                                labels = i_top_fs
                            else:
                                labels = tf.concat([i_top_fs, labels], axis=0)
                            # XXX: Incompatible if the input size 
                            # of the pooling is not a proper multiple 
                            # (multiple of 32 for VGG16).
                            new_recons = [
                                tf.scatter_nd(
                                    tf.reshape(
                                        switches[0, :, :, i_top_fs[j]],
                                        [-1, 1],
                                    ),
                                    tf.reshape(
                                        block_fwd_output[:, :, i_top_fs[j]],
                                        [-1]
                                    ),
                                    [b * (2 * h) * (2 * w) * c],
                                ) for j in range(num_features) 
                            ]

#                            new_recons = tf.concat(new_recons, axis=0)
#                            new_recons = tf.reshape(
#                                new_recons,
#                                [num_features, (2 * h), (2 * w), c],
#                            )

                            # Concatenate new reconstructions with
                            # reconstructions from the above layer.

                            # XXX: Assume b = 1, k = 2, s = 2.
                            if recons is None:
                                unpooled_flattened_tensors = new_recons
                            else:
                                unpooled_flattened_tensors = new_recons + [
                                    tf.scatter_nd(
                                        tf.reshape(switches, [-1, 1]),
                                        tf.reshape(recons[j, :, :, :], [-1]),
                                        [b * (2 * h) * (2 * w) * c],
                                    )
                                    for j in range(recons.shape[0].value)
                                ]

                            unpooled_flattened = tf.concat(
                                unpooled_flattened_tensors,
                                axis=0,
                            )
                            recons = tf.reshape(
                                unpooled_flattened,
                                [-1, (2 * h), (2 * w), c],
                            )

                        elif 'conv' in layer_name:
                            # XXX: Where to put ReLU?
                            recons = tf.nn.relu(recons)
                            W_shape = layer_conf['W']
                            _, _, n_in_chs, n_out_chs = W_shape
                            W = self._get_weights(
                                block_layer_name,
                                'W',
                                W_shape,
                            )
                            b = self._get_weights(
                                block_layer_name,
                                'b',
                                layer_conf['b'],
                            )
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

        self._deconv_tensor['full_recons'] = recons
        self._deconv_tensor['full_labels'] = labels

#    def get_backprop_result(
#        self,
#        block_name,
#        input_features,
#        max_pool_switches,
#        method='deconv',
#    ):
#        i_block = int(block_name[-1])
#        subconfig = self._config[:i_block]
#
#        self._backprop_graph = tf.Graph()
#        with self._backprop_graph.as_default():
#            if method == 'deconv':
#                recons = self._build_deconv_backprop_graph(
#                    subconfig,
#                    input_features,
#                    max_pool_switches,
#                )
#            else:
#                raise NotImplementedError
#
#            init = tf.global_variables_initializer()
#
#            tf_session = tf.Session()
#            tf_session.run(init)
#
#            rv = tf_session.run([recons])
#
#        if self.logdir is not None:
#            summary_writer = tf.summary.FileWriter(
#                logdir=self.logdir,
#                graph=self._backprop_graph,
#            )
#
#        return rv

#    def _build_deconv_backprop_graph(
#        self,
#        subconfig,
#        input_features,
#        max_pool_switches,
#    ):
#        # TODO: check features sizes.
#        num_features = len(input_features)
#        recons = None
#
#        for block_name, block_conf in reversed(subconfig):
#            with tf.variable_scope(block_name):
#                for layer_name, layer_conf in reversed(block_conf):
#                    block_layer_name = block_name + '_' + layer_name
#                    with tf.variable_scope(layer_name):
#                        if 'pool' in layer_name: 
#                            switches_array = max_pool_switches[block_layer_name]
#                            b, h, w, c = switches_array.shape
#                            switches = tf.constant(switches_array)
#
#                            if recons is None:
#                                features = [
#                                    (i_f, tf.constant(a_feature_array))
#                                    for i_f, a_feature_array
#                                    in input_features
#                                ]
#                                # XXX: Incompatible if the input size 
#                                # of the pooling is not a proper multiple 
#                                # (multiple of 32 for VGG16).
#                                recons = [
#                                    tf.scatter_nd(
#                                        tf.reshape(
#                                            switches[0, :, :, i_f],
#                                            [-1, 1],
#                                        ),
#                                        tf.reshape(a_feature[:, :], [-1]),
#                                        [b * (2 * h) * (2 * w) * c],
#                                    ) for i_f, a_feature in features 
#                                ]
#
#                                recons = tf.concat(recons, axis=0)
#                                recons = tf.reshape(
#                                    recons,
#                                    [num_features, (2 * h), (2 * w), c],
#                                )
#                            else:
#                                # XXX: Assume b = 1, k = 2, s = 2.
#                                unpooled_flattened_tensors = [
#                                    tf.scatter_nd(
#                                        tf.reshape(switches, [-1, 1]),
#                                        tf.reshape(recons[i_f, :, :, :], [-1]),
#                                        [b * (2 * h) * (2 * w) * c],
#                                    )
#                                    for i_f in range(num_features)
#                                ]
#                                unpooled_flattened = tf.concat(
#                                    unpooled_flattened_tensors,
#                                    axis=0,
#                                )
#                                recons = tf.reshape(
#                                    unpooled_flattened,
#                                    [num_features, (2 * h), (2 * w), c],
#                                )
#
#                        elif 'conv' in layer_name:
#                            # XXX: Where to put ReLU?
#                            recons = tf.nn.relu(recons)
#                            W_shape = layer_conf['W']
#                            _, _, n_in_chs, n_out_chs = W_shape
#                            W = self._get_weights(
#                                block_layer_name,
#                                'W',
#                                W_shape,
#                            )
#                            b = self._get_weights(
#                                block_layer_name,
#                                'b',
#                                layer_conf['b'],
#                            )
#                            recons = tf.nn.bias_add(recons, -b)
#                            recons = tf.nn.conv2d_transpose(
#                                recons,
#                                W,
#                                output_shape=(
#                                    recons.shape.as_list()[:-1] + [n_in_chs]
#                                ),
#                                strides=[1, 1, 1, 1],
#                                padding='SAME',
#                            )
#
#        return recons

#    def get_reconstructed_top_features(
#        self,
#        tf_session,
#        input_array,
#        block_name,
#        num_top_features=9,
#        reconstruction_method='deconv',
#    ):
#        assert(self.visualize)
#        assert(self._max_pool_switches is not None)
#
#        i_block = int(block_name[-1])
#        subconfig = self._config[:i_block]
#        layer_name = 'pool'
#        block_layer_name = block_name + '_' + layer_name
#        features = self.layers[block_layer_name]
#
#        with self.graph.as_default(): 
#            assert(features.shape.as_list()[0] == 1)
#            norms = tf.norm(features[0], axis=[0, 1])
#            _, tops = tf.nn.top_k(norms, k=num_top_features)
#
#            if reconstruction_method == 'deconv':
#                recons = self._get_deconved_features(
#                    features,
#                    subconfig,
#                    tops,
#                    num_top_features=num_top_features,
#                )
#
#        fetches ={
#            'top_labels': tops,
#            'reconstructed_features': recons,
#            'activations': features,
#        }
#
#        return tf_session.run(
#            fetches,
#            feed_dict={self.layers['input']: input_array},
#        )
#
