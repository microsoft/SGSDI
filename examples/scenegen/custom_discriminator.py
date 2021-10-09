# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import logging
import numpy as np
from typing import List

import tensorflow as tf
from tensorflow.keras import layers

from GraphicsDL.graphicstf import gtf_layers
from GraphicsDL.graphicstf.basic import BasicModel
from .custom_layers import get_color_palettes, ConvolutionBlock2D, ConvolutionBlock3D
from .custom_sampler import RenderingViews

MAX_VIS_OUT = 4


class SingleViewDiscriminator(BasicModel):
    def __init__(self,
                 down_sample: int = 4,
                 net_channels: int = 64,
                 use_sn: bool = True,
                 as_latent: bool = False,
                 trainable: bool = True,
                 dtype: tf.DType = tf.float32,
                 name: str = None):
        super().__init__(trainable=trainable, name=name, dtype=dtype)
        self.down_sample = down_sample
        self.net_channels = net_channels
        self.use_sn = use_sn
        self.down_scales = list()
        self.as_latent = as_latent
        self.reduction = gtf_layers.SpectrumLinear(1) if not as_latent else None

    def build(self, input_shape: List[tf.TensorShape]):
        channels = self.net_channels
        min_channels = int(np.ceil(input_shape[0][-1] / self.net_channels)) * self.net_channels

        for l_id in range(self.down_sample):
            norm = layers.LayerNormalization if not self.use_sn else None
            self.down_scales.append(ConvolutionBlock2D(max(channels, min_channels), 3, 2, 'same', layers.ReLU(), norm,
                                                       self.use_sn, True, trainable=self.trainable, dtype=self.dtype,
                                                       name=f'ConvBlock2D{l_id}'))
            channels = min(channels * 2, 512)

        self.built = True

    def call(self, inputs: List[tf.Tensor], training=None, as_master=True, vis_port=False, mask=None):
        # Possible inputs:
        #   1. Semantic only: [N,H,W,C]
        #   2. Semantic + depth: [N,H,W,C] + [N,H,W,1]
        assert isinstance(inputs, list)
        assert vis_port is False
        x = tf.concat(inputs, axis=-1)
        for d_s in self.down_scales:
            x = d_s(x, as_master=as_master)
        x = tf.reshape(x, [x.shape.dims[0].value, -1])
        if self.reduction is not None:
            x = self.reduction(x, as_master=as_master)
        if vis_port:
            raise NotImplementedError
        return [x]


class PairwiseViewsDiscriminator(BasicModel):
    def __init__(self,
                 down_sample: int = 4,
                 net_channels: int = 64,
                 use_sn: bool = True,
                 trainable: bool = True,
                 dtype: tf.DType = tf.float32,
                 name: str = None):
        super().__init__(trainable=trainable, name=name, dtype=dtype)
        self.comm_kargs = dict(down_sample=down_sample, net_channels=net_channels, use_sn=use_sn, as_latent=True,
                               trainable=trainable, dtype=dtype)
        self.view_discriminator_sets: List[SingleViewDiscriminator] = list()
        self.reduction = gtf_layers.SpectrumLinear(1)
        self.logged = False

    def build(self, input_shape: List[tf.TensorShape]):
        num_inputs = len(input_shape)
        for _ in range(num_inputs):
            single_discriminator = SingleViewDiscriminator(**self.comm_kargs)
            self.view_discriminator_sets.append(single_discriminator)
        self.built = True

    @staticmethod
    def images_visualization(vol_view: List[tf.Tensor], out_dict=None):
        view_cls, view_depth = vol_view if len(vol_view) > 1 else (vol_view[0], None)
        color_palettes = get_color_palettes(view_cls.shape.dims[-1].value)
        vis_dict = out_dict if out_dict is not None else dict()
        vis_dict = RenderingViews.views_visualization(vis_dict, view_cls, view_depth, color_palettes)
        return vis_dict

    def cross_view_aggression(self, views_latent, as_master=True):
        return tf.add_n(views_latent), None

    def identity_view_aggression(self, view_latents, as_master=True):
        return tf.add_n(view_latents)

    def latent_reduction(self, stacked_latent, as_master=True):
        return self.reduction(stacked_latent, as_master=as_master)

    def call(self, inputs: List[tf.Tensor], training=None, as_master=True, vis_port=True, mask=None):
        # Possible inputs:
        #   1. Semantic only: [N,V,H,W,C]
        #   2. Semantic + depth: [N,V,H,W,C] + [N,V,H,W,1]
        assert isinstance(inputs, list)
        assert np.all([i.shape.ndims == 5 for i in inputs])
        s_inputs = list()
        for i in inputs:
            s_inputs.append(tf.unstack(i, axis=1))
        views_latent = list()
        all_latent = list()
        for s_i in zip(*s_inputs):
            view_latents = [d_([s_], as_master=as_master)[0] for s_, d_ in zip(s_i, self.view_discriminator_sets)]
            all_latent.append(tf.stack(view_latents, axis=1))
            view_latent = self.identity_view_aggression(view_latents, as_master=as_master)
            views_latent.append(view_latent)
        stacked_latent, views_coeff = self.cross_view_aggression(views_latent, as_master=as_master)
        score = self.latent_reduction(stacked_latent, as_master=as_master)
        out_dict = dict(out=[score])
        if vis_port:
            out_dict = self.images_visualization(inputs, out_dict)
        self.logged = True
        return out_dict


class AttentionPairwiseDiscriminator(PairwiseViewsDiscriminator):
    def __init__(self,
                 down_sample: int = 4,
                 net_channels: int = 64,
                 view_num: int = 4,
                 use_sn: bool = True,
                 flag: int = 1,
                 trainable: bool = True,
                 dtype: tf.dtypes = tf.float32,
                 name: str = None):
        super().__init__(down_sample=down_sample, net_channels=net_channels, use_sn=use_sn, trainable=trainable, 
                         dtype=dtype, name=name)
        self.flag = flag
        self.cross_view_attention = gtf_layers.SpectrumLinear(view_num, activation=tf.sigmoid) if self.flag&1 else None

    @staticmethod
    def attention_views_combination(views_latent, view_attention_func, as_master=True):
        views_latent = tf.stack(views_latent, axis=1)
        views_latent_flatten = tf.reshape(views_latent, [views_latent.shape.dims[0].value, -1])
        views_coeff = view_attention_func(views_latent_flatten, as_master=as_master)[..., tf.newaxis]
        views_coeff = tf.clip_by_value(views_coeff, 0.2, 0.8)
        cross_views_latent = tf.reduce_sum(views_coeff * views_latent, axis=1)
        return cross_views_latent, views_coeff

    def cross_view_aggression(self, views_latent, as_master=True):
        if self.flag & 1:
            logging.info('Selected cross view aggression method: Attention')
            cross_views_latent = self.attention_views_combination(views_latent, self.cross_view_attention, as_master)
            return cross_views_latent
        else:
            return super().cross_view_aggression(views_latent, as_master=as_master)

    def identity_view_aggression(self, view_latents, as_master=True):
        if self.flag & 2:
            logging.info('Selected identity view aggression method: Attention (Not implemented)')
            raise NotImplementedError
        else:
            return super().identity_view_aggression(view_latents, as_master=as_master)
