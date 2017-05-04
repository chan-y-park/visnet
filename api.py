import random

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

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


def postprocess(
    image_array,
    per_channel=False,
    add_mean=False,
    flip=True,
):
    """
    Rescale all RGB channels to have values between 0 and 1.
    If per_channel is False, rescaling is done
    by taking the minimum and the maximum over all element.
    If per_channel is True, rescaling is done channel by channel.

    If add_mean is True, add back the mean pixel values
    that are subtracted during the preprocessing. 

    If flip is True, flip the order of the color channels from BGR to RGB.
    """
    if per_channel:
        num_color_chs = 3
        scaled = []
        for i_c in range(num_color_chs):
            a_ch = image_array[:, :, i_c].astype(float)
            min_v = np.min(a_ch)
            a_ch -= min_v
            a_ch /= np.max(a_ch)
            scaled.append(a_ch)
        image_array = np.array(scaled).transpose(1, 2, 0)
    else:
        image_array = image_array.astype(float)
        min_v = np.min(image_array)
        image_array -= min_v
        image_array /= np.max(image_array)

    if add_mean:
        image_array[:, :, 0] += 103.939 / 255.0
        image_array[:, :, 1] += 116.779 / 255.0
        image_array[:, :, 2] += 123.68 / 255.0

    if flip:
        image_array = image_array[:,:,::-1]

    return image_array


def get_all_deconv_results(
    input_image,
    model_name='VGG16',
    full_deconv=True,
    by_block=True,
    log_device_placement=False,
    **kwargs
):
    if model_name == 'VGG16':
        config = vgg16_no_fc_config
    else:
        raise RuntimeError('Unknown model: {}'.format(model_name))

    input_array = input_image.to_array().reshape([-1, 224, 224, 3])
    vn = VisNet(full_deconv=full_deconv, **kwargs)
    if full_deconv:
        print('Doing full deconvolution...')
        rd = vn.get_full_deconv_result(
            input_array,
            log_device_placement=log_device_placement,
        ) 
    else:
        print('Doing forward propagation and recording max pool switches...')
        rd = vn.get_forward_results(
            input_array,
            log_device_placement=log_device_placement,
        )
        rd['deconv_layers'] = {}
        for block_name, block_conf in config['network']:
            for layer_name, layer_conf in block_conf:
                if by_block and layer_name != 'pool':
                    continue
                block_layer_name = block_name + '_' + layer_name
                print('Deconvolutioning {}...'.format(block_layer_name))
                rv, labels = vn.get_deconv_result(
                    block_name, layer_name,
                    log_device_placement=log_device_placement,
                )
                rd['deconv_layers'][block_layer_name] = {
                    'recons': rv,
                    'labels': labels,
                }

    return rd

def get_deconv_images(
    input_image,
    save_fp=None,
    file_format='svg',
    num_top_features=None,
    deconv_layers=None,
    axw=3,
    axh=4,
):
    block_layer_names = deconv_layers.keys()

    nrows = len(block_layer_names)
    ncols = num_top_features
    w = axw * ncols * 2
    h = axh * nrows
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(w, h),
    )

    for i_bl, name in enumerate(block_layer_names):
        layer = deconv_layers[name]
        activations = layer['activations']
        recons = layer['recons']
        labels = layer['labels']
        for i_f in range(num_top_features):
            ax = axes[i_bl][i_f]
            ax.set_xticks([])
            ax.set_yticks([])
            if i_f == 0:
                ax.set_ylabel(name)
            ax.set_title('{}'.format(labels[i_f]))
            
            ax.imshow(input_image.image)
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
            ax.imshow(
                postprocess(activations[i_f], flip=False),
                cmap='gray',
                alpha=.9,
                extent=(xmin, xmax, ymin, ymax)
            )

            scaled = postprocess(recons[i_f])
            ax.imshow(
                scaled,
                interpolation='none',
                extent=(xmin + xmax, 2 * xmax, ymin, ymax)
            )
            
            ax.set_xlim(xmin, 2*xmax)

    plt.tight_layout(
        pad=0,
        w_pad=0,
        h_pad=0,
    )                
    plt.savefig(save_fp, format=file_format)
