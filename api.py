import random

import numpy as np
import tensorflow as tf
from PIL import Image

from vgg16 import vgg16_no_fc_config
from visnet import VisNet

ROOT_DIR = '/home/chan/workspace/visnet'

class InputImage:
    def __init__(self, file_path=None, image=None):
        if file_path is not None:
            self.image = Image.open(file_path)
        elif image is not None:
            self.image = image

    def to_array(self, np_dtype=np.float32):
        x = np.asarray(self.image, dtype=np_dtype)
        x = x.reshape((1,) + x.shape)
        if len(x.shape) == 2:
            raise NotImplementedError
        # Substracting the mean, from Keras' imagenet_utils.preprocess_input.
        # 'RGB'->'BGR'
        x = x[:, :, :, ::-1]
        # Zero-center by mean pixel
        x[:, :, :, 0] -= 103.939
        x[:, :, :, 1] -= 116.779
        x[:, :, :, 2] -= 123.68

        return x

    def get_resized_image(
        self,
        size,
        crop=None,
        maintain_aspect_ratio=True
    ):
        width, height = self.image.size

        if crop is not None:
            left_center = int((width - size) / 2.0)
            upper_center = int((height - size) / 2.0)
            left_max = int(width - size)
            upper_max = int(height - size)
            if crop == 'center':
                left = left_center
                upper = upper_center
            elif crop == 'random':
                left = random.randint(0, left_max)
                upper = random.randint(0, upper_max)
            elif crop == 'left':
                left = 0
                upper = upper_center
            elif crop == 'right':
                left = left_max
                upper = upper_center
            elif crop == 'top':
                left = left_center
                upper = 0
            elif crop == 'bottom':
                left = left_center
                upper = upper_max
            
            img = self.image.crop((left, upper, left + size, upper + size))
            
        elif maintain_aspect_ratio:
            
            ar = float(width) / float(height)
            if ar > 1:
                width = size
                height = size / ar
                left_upper = (0, int((size - height) / 2.0))
            elif ar < 1:
                width = size * ar
                height = size
                left_upper = (int((size - width) / 2.0), 0)
            else:
                width = height = size
                left_upper = (0, 0)
            img = self.image.resize((int(width), int(height)))
            background = Image.new('RGB', (size, size), (255, 255, 255))
            background.paste(img, left_upper)
            img = background
        
        else:
            img = self.image.resize((size, size))

        return InputImage(image=img)


def get_all_deconv_results(
    image_path,
    model_name='VGG16',
    full_deconv=True,
    **kwargs
):
    if model_name == 'VGG16':
        config = vgg16_no_fc_config
    else:
        raise RuntimeError('Unknown model: {}'.format(model_name))

    input_image = InputImage(image_path)
    resized_image = input_image.get_resized_image(224)
    input_array = resized_image.to_array().reshape([-1, 224, 224, 3])
    vn = VisNet(full_deconv=full_deconv, **kwargs)
    if full_deconv:
        print('Doing full deconvolution...')
        rd = vn.get_full_deconv_result(input_array) 
    else:
        print('Doing forward propagation and recording max pool switches...')
        rd = vn.get_forward_results(input_array)
        for block_name, block_conf in config['network']:
            for layer_name, layer_conf in block_conf:
                if layer_name != 'pool':
                    continue
                block_layer_name = block_name + '_' + layer_name
                print('Deconvolutioning {}...'.format(block_layer_name))
                rv, labels = vn.get_deconv_result(block_name, layer_name)
                rd[block_layer_name] = {
                    'recons': rv,
                    'labels': labels,
                }

    return rd

def run(
    input_array=None,
    input_image=None,
    model_name='VGG16',
    test=False,
    **kwargs
):
    if input_image is not None:
        input_array = input_image.to_array().reshape([-1, 224, 224, 3])
    if model_name == 'VGG16':
        #model = VGG16()
        model = VGG16Core(test_filters=test, **kwargs)
    #model.build()
    with model.graph.as_default():
        t_output = model.get_output_layer()

        saver = tf.train.Saver(tf.global_variables())
        t_summary = tf.summary.merge_all()


        summary_writer = tf.summary.FileWriter(
            logdir=(ROOT_DIR + '/log'),
            graph=model.graph,
        )

        init = tf.global_variables_initializer()

        sess = tf.Session()
        sess.run(init)

        fetches = {
            'reconstructed_features': model.reconstructed_features,
            'layers': model.layers,
            'switches': model.max_pool_switches,
            'tops': model.tops,
            'output_layer': t_output
        }
        #fetches['layers'] = model.layers
        #fetches['switches'] = model.max_pool_switches
        #fetches['tops'] = model.tops
        #fetches['output_layer'] = t_output
        #fetches['recons'] = model.recons

        output = sess.run(
            fetches,
            feed_dict={
                model.input_layer: input_array,
            }
        )

        summary_str = sess.run(t_summary)
        summary_writer.add_summary(summary_str)

    return {
        'model': model,
        'output': output,
        'tf_session': sess,
    }

