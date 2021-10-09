# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import logging
import numpy as np
import tensorflow as tf
from typing import Tuple, List, Optional

from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.python.keras import activations

from .tools import data as g_data
from GraphicsDL.graphicstf import gtf_render
from GraphicsDL.graphicstf.models import stylegan
from GraphicsDL.graphicstf import layers as gtf_layers
from GraphicsDL.graphicstf import sparse as gtf_sparse


def get_color_palettes(num_channel, color_norm=True):
    if num_channel in [10]:
        label_type = g_data.STRUCTURED3D_BEDROOM_9()
    elif num_channel in [12]:
        label_type = g_data.STRUCTURED3D_LIVING_11()
    elif num_channel in [6]:
        label_type = g_data.STRUCTURED3D_KITCHEN_5()
    elif num_channel in [9]:
        label_type = g_data.MATTERPORT3D_BEDROOM_8()
    else:
        logging.error(f'Channel: {num_channel}')
        raise NotImplementedError

    color_palettes = np.concatenate([label_type.color_map_arr(), [[0, 0, 0]]], axis=0).astype(np.uint8)
    if color_norm:
        color_palettes = tf.convert_to_tensor(color_palettes / 255, dtype=tf.float32)
    return color_palettes


class ConvolutionBlock3DBase(layers.Layer):
    def __init__(self,
                 filters,
                 kernel_size,
                 stride: int = 1,
                 padding: str = 'same',
                 activation=None,
                 normalization=None,
                 use_sn: bool = False,
                 use_bias: bool = True,
                 conv_func: type(layers.Layer) = layers.Conv3D,
                 trainable: bool = True,
                 data_format: str = 'channels_last',
                 dtype: tf.DType = tf.float32,
                 name: str = None):
        super().__init__(trainable, name, dtype)
        self.use_sn = use_sn

        self.kernel_size, self.stride, self.padding = kernel_size, stride, padding
        if isinstance(padding, list):
            padding = 'valid'
        conv_kargs = dict(filters=filters, kernel_size=kernel_size, strides=stride, padding=padding,
                          data_format=data_format, use_bias=use_bias, trainable=trainable, name='Conv')
        self.conv_layer = conv_func(**conv_kargs)
        self.activation = activations.get(activation) if activation is not None else activations.linear
        self.normalization = normalization

    def call(self, inputs, as_master=True, **kwargs):
        conv_args = [inputs, as_master] if self.use_sn else [inputs, ]
        if isinstance(self.padding, list):
            in_shape = np.array(inputs.shape.as_list()[1:3])
            out_shape = np.ceil(in_shape / self.stride)
            pad_channel = (out_shape - 1) * self.stride + self.kernel_size - in_shape
            top_pad = np.floor(pad_channel / 2)
            down_pad = pad_channel - top_pad
            top_down_pad = np.stack([top_pad, down_pad], axis=-1)
            paddings = np.array([[0, 0], *top_down_pad.tolist(), [0, 0], [0, 0]], dtype=np.int64)
            conv_args[0] = tf.pad(inputs, paddings)
        x = self.conv_layer(*conv_args)
        if self.normalization is not None:
            x = self.normalization(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class ConvolutionBlock3D(ConvolutionBlock3DBase):
    def __init__(self,
                 filters,
                 kernel_size,
                 stride: int = 1,
                 padding: str = 'same',
                 activation=None,
                 normalization=None,
                 use_sn: bool = False,
                 use_bias: bool = True,
                 trainable: bool = True,
                 data_format: str = 'channels_last',
                 dtype: tf.DType = tf.float32,
                 name: str = None):
        conv_func = layers.Conv3D if not use_sn else gtf_layers.SpectrumConvolution3D
        super().__init__(filters, kernel_size, stride, padding, activation, normalization, use_sn, use_bias, conv_func,
                         trainable, data_format, dtype, name)


class ConvolutionBlock2D(ConvolutionBlock3DBase):
    def __init__(self,
                 filters,
                 kernel_size,
                 stride: int = 1,
                 padding: str = 'same',
                 activation=None,
                 normalization=None,
                 use_sn: bool = False,
                 use_bias: bool = True,
                 trainable: bool = True,
                 data_format: str = 'channels_last',
                 dtype: tf.DType = tf.float32,
                 name: str = None):
        conv_func = layers.Conv2D if not use_sn else gtf_layers.SpectrumConvolution2D
        super().__init__(filters, kernel_size, stride, padding, activation, normalization, use_sn, use_bias, conv_func,
                         trainable, data_format, dtype, name)


class TransposeConvolutionBlock3D(ConvolutionBlock3DBase):
    def __init__(self,
                 filters,
                 kernel_size,
                 stride: int = 1,
                 padding: str = 'same',
                 activation=None,
                 normalization=None,
                 use_sn: bool = False,
                 use_bias: bool = True,
                 trainable: bool = True,
                 data_format: str = 'channels_last',
                 dtype: tf.DType = tf.float32,
                 name: str = None):
        assert not use_sn
        conv_func = layers.Conv3DTranspose
        super().__init__(filters, kernel_size, stride, padding, activation, normalization, use_sn, use_bias, conv_func,
                         trainable, data_format, dtype, name)


class TransposeConvolutionBlock2D(ConvolutionBlock3DBase):
    def __init__(self,
                 filters,
                 kernel_size,
                 stride: int = 1,
                 padding: str = 'same',
                 activation=None,
                 normalization=None,
                 use_sn: bool = False,
                 use_bias: bool = True,
                 trainable: bool = True,
                 data_format: str = 'channels_last',
                 dtype: tf.DType = tf.float32,
                 name: str = None):
        assert not use_sn
        conv_func = layers.Conv2DTranspose
        super().__init__(filters, kernel_size, stride, padding, activation, normalization, use_sn, use_bias, conv_func,
                         trainable, data_format, dtype, name)


class ModulateConvBlock(layers.Layer):
    def __init__(self,
                 ndims,
                 filters,
                 up_scale: int = 1,
                 activation=None,
                 trainable: bool = True,
                 data_format: str = 'channels_last',
                 dtype: tf.DType = tf.float32,
                 name: str = None):
        super().__init__(trainable, name, dtype)

        if ndims == 2:
            conv_func = TransposeConvolutionBlock2D if up_scale != 1 else ConvolutionBlock2D
        elif ndims == 3:
            conv_func = TransposeConvolutionBlock3D if up_scale != 1 else ConvolutionBlock3D
        else:
            raise NotImplementedError
        self.transpose_conv = conv_func(filters, 3, up_scale, use_sn=False, trainable=trainable,
                                        data_format=data_format, dtype=dtype)
        self.adaptive_instance_norm = gtf_layers.AdaptiveInstanceNorm(dtype=dtype)
        self.activation = activations.get(activation) if activation is not None else activations.linear
        self.z_mapping = stylegan.ZMapping(filters, activation=self.activation, trainable=trainable, dtype=dtype)

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor], as_master=True):
        assert len(inputs) == 2
        feat, d_latent = inputs
        feat = self.transpose_conv(feat, as_master)
        scale, bias = self.z_mapping(d_latent)
        feat = self.activation(self.adaptive_instance_norm([feat, scale, bias]))
        return feat


class ModulateConv2DBlock(ModulateConvBlock):
    def __init__(self,
                 filters,
                 up_scale: int = 1,
                 activation=None,
                 trainable: bool = True,
                 data_format: str = 'channels_last',
                 dtype: tf.DType = tf.float32,
                 name: str = None):
        super().__init__(2, filters, up_scale, activation, trainable, data_format, dtype, name)


class ModulateConv3DBlock(ModulateConvBlock):
    def __init__(self,
                 filters,
                 up_scale: int = 1,
                 activation=None,
                 trainable: bool = True,
                 data_format: str = 'channels_last',
                 dtype: tf.DType = tf.float32,
                 name: str = None):
        super().__init__(3, filters, up_scale, activation, trainable, data_format, dtype, name)


class SparseConvolution3D(gtf_sparse.SparseConv):
    def __init__(self,
                 filters: int,
                 kernel_size: int,
                 trainable: bool = True,
                 name: str = None,
                 dtype: tf.DType = tf.float32,
                 **kwargs):
        super().__init__(filters, kernel_size, trainable, name, dtype)


class SpectrumSparseConvolution3D(gtf_sparse.SparseConv):
    def __init__(self,
                 filters: int,
                 kernel_size: int,
                 trainable: bool = True,
                 name: str = None,
                 dtype: tf.DType = tf.float32,
                 **kwargs):
        super().__init__(filters, kernel_size, trainable, name, dtype)
        self.spectrum_norm = gtf_layers.SpectrumNorm(trainable=trainable, dtype=dtype)

    def call(self, inputs: List[tf.Tensor], as_master=True):
        w = self.w
        self.w = self.spectrum_norm(self.w, as_master)
        outputs = super().call(inputs)
        self.w = w
        return outputs


class SparseConvolutionBlock3D(ConvolutionBlock3DBase):
    def __init__(self,
                 filters,
                 kernel_size,
                 stride: int = 1,
                 padding: str = 'same',
                 activation=None,
                 normalization=None,
                 use_sn: bool = False,
                 use_bias: bool = True,
                 trainable: bool = True,
                 data_format: str = 'channels_last',
                 dtype: tf.DType = tf.float32,
                 name: str = None):
        conv_func = SparseConvolution3D if not use_sn else SpectrumSparseConvolution3D
        super().__init__(filters, kernel_size, stride, padding, activation, normalization, use_sn, use_bias, conv_func,
                         trainable, data_format, dtype, name)


class ZMapping(layers.Layer):
    def __init__(self,
                 out_channel: int,
                 std_dev: float = 0.02,
                 activation: Optional[layers.Layer] = None,
                 trainable: bool = True,
                 dtype: tf.DType = tf.float32,
                 name: str = None,
                 **kwargs):
        super().__init__(trainable, name, dtype, **kwargs)
        self.out_channel = out_channel
        self.std_dev = std_dev
        self.activation = activations.get(activation)

        self.w = None
        self.b = None

    def build(self, input_shape: tf.TensorShape):
        assert K.image_data_format() == 'channels_last'
        input_channel = input_shape.dims[-1].value
        self.w = self.add_weight('w', [input_channel, self.out_channel * 2], dtype=self.dtype, trainable=self.trainable,
                                 initializer=tf.random_normal_initializer(stddev=self.std_dev))
        self.b = self.add_weight('b', [self.out_channel * 2], dtype=self.dtype, trainable=self.trainable,
                                 initializer=tf.constant_initializer(0.))
        self.built = True

    def call(self, inputs: tf.Tensor, **kwargs):
        m_s, m_b = tf.split(self.activation(tf.matmul(inputs, self.w) + self.b), 2, axis=-1)
        return m_s, m_b


class Const3DBlock(layers.Layer):
    def __init__(self,
                 shapes,
                 channels,
                 activation=None,
                 std_dev: float = 0.02,
                 trainable: bool = True,
                 dtype: tf.DType = tf.float32,
                 name: str = None,
                 **kwargs):
        super().__init__(trainable, name, dtype, **kwargs)
        self.shapes = shapes
        self.channels = channels
        self.std_dev = std_dev

        self.adaptive_instance_norm = gtf_layers.AdaptiveInstanceNorm(dtype=dtype)
        self.activation = activations.get(activation)
        self.z_mapping = ZMapping(channels, activation=self.activation, trainable=trainable, dtype=dtype)

        self.const = None

    def build(self, input_shape):
        del input_shape
        c_shape = [1, *self.shapes, self.channels]
        self.const = self.add_weight('const', c_shape, dtype=self.dtype, trainable=self.trainable,
                                     initializer=tf.random_normal_initializer(stddev=self.std_dev))
        self.built = True

    def call(self, inputs: tf.Tensor, **kargs):
        scale, bias = self.z_mapping(inputs)
        norm = self.adaptive_instance_norm([self.const, scale, bias])
        feat = self.activation(norm)
        return feat


class DRCClsRendering(layers.Layer):
    def __init__(self,
                 vox_size: List,
                 sample_size: List,
                 cam_fov: List,
                 out_activation,
                 cnts: int = 0.,
                 near: float = 0.,
                 far: float = 0.,
                 out_depth: bool = False,
                 trainable: bool = False,
                 name: str = None,
                 dtype: tf.dtypes = tf.float32):
        """
        Render volume to image based on differentiable ray consistency. More complex ray probability accumulation rules,
            compared with original DRC implementation.

        Args:
            sample_size: the sampled 2D image size
            cam_fov: the x-y fov angle
            cnts: sample counts along z-axis. The cnts will be automatic inferred if 0, which will ensure the sampling
                stride shorter than that of voxel.
            near: the nearest distance along z-axis.
            far: the farthest distance along z-axis. The far will be automatic inferred if 0, which will ensure the
                farthest sampling distance approximates the voxel boundary.
            trainable: the flag to make the variable trainable or not
            name: the name of the ops
            dtype: the data type used in tf.Tensor
        """
        super().__init__(trainable, name, dtype)
        self.vox_size = np.asarray(vox_size, dtype=np.int32)
        self.sample_size = sample_size
        self.cam_fov = cam_fov
        self.cnts = cnts
        self.near = near
        self.far = far
        self.out_depth = out_depth
        self.out_activation = out_activation

        norm_size = self.vox_size.astype(np.float32) / self.vox_size[0]
        self.far = np.sqrt(np.sum((norm_size * 0.75) ** 2)) if self.far == 0. else self.far
        self.cnts = int((self.far - self.near) * np.sqrt(np.sum(self.vox_size ** 2))) if self.cnts == 0 else self.cnts
        self.focal_length = np.asarray(self.sample_size) / (2.0 * np.tan(np.asarray(self.cam_fov) / 2.0))
        self.vox_z = tf.linspace(0., 1., self.cnts)
        self.vox_sampling: Optional[gtf_render.ResampleVoxelBasedOnCamera] = None

    def build(self, input_shape: List[tf.TensorShape]):
        self.vox_sampling = gtf_render.ResampleVoxelBasedOnCamera(self.sample_size, self.cnts, self.focal_length, None,
                                                                  self.near, self.far)
        self.built = True

    def compute_ray_potentials(self, voxel_occ_prob):
        vol_logit_cum = tf.math.cumprod(voxel_occ_prob[..., :1] + 1e-12, axis=1, exclusive=True)
        ray_potentials = tf.multiply(voxel_occ_prob[..., 1:], vol_logit_cum)

        image_logit = tf.reduce_sum(ray_potentials, axis=1)
        empty_image = 1 - tf.reduce_sum(image_logit, axis=-1, keepdims=True)
        voxel_image = tf.concat((empty_image, image_logit), axis=-1)
        out_images = [voxel_image]
        if self.out_depth:
            voxel_logit = tf.reduce_sum(ray_potentials, axis=-1, keepdims=True)
            depth_image = tf.reduce_sum(tf.reshape(self.vox_z, [1, -1, 1, 1, 1]) * voxel_logit, axis=1)
            out_images.append(depth_image)
        return out_images

    def pre_activate(self, vol):
        if self.out_activation == '':
            return vol
        vol = getattr(tf.nn, self.out_activation)(vol)
        return vol

    def call(self, inputs=None, **kwargs) -> List[tf.Tensor]:
        # vol_data, object_rotation, object_translation, camera_rotation, camera_translation
        inputs[0] = self.pre_activate(inputs[0])
        resample_vox = self.vox_sampling(inputs)
        voxel_image = self.compute_ray_potentials(resample_vox)
        return voxel_image

