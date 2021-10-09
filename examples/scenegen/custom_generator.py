# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import inspect
import numpy as np
import tensorflow as tf
from typing import Optional, List, Dict, Tuple

from GraphicsDL.graphicstf.basic import BasicModel
from GraphicsDL.graphicstf import layers as dl_layers
from examples.scenegen import custom_layers as gtf_layers


MAX_VIS_OUT = 4


class Code2VolCondBaseGenerator(BasicModel):
    """
    This generator uses HoloGAN scheme to embed latent code into feature maps. Additionally, it supports "spatial" and
        "channels" modes to embed labels into feature maps.
    """
    def __init__(self,
                 ndims: int = 3,
                 channel_embedding: str = '',
                 spatial_embedding: str = '',
                 const_shape: Optional[List] = None,
                 d_latent: int = 512,
                 up_sample: int = 4,
                 net_channels: int = 64,
                 out_channels: int = 20,
                 max_channels: int = 512,
                 net_act=tf.nn.leaky_relu,
                 net_act_args: Dict = None,
                 out_act=tf.nn.sigmoid,
                 out_act_kargs: Dict = None,
                 out_label: bool = True,
                 trainable: bool = True,
                 dtype: tf.DType = tf.float32,
                 name: str = None
                 ):
        super().__init__(trainable=trainable, dtype=dtype, name=name)

        # Args parser
        self.ndims = ndims
        self.channel_embedding = channel_embedding
        self.spatial_embedding = spatial_embedding
        self.const_shape = const_shape if const_shape is not None else [2, 1, 2]
        self.d_latent = d_latent
        self.up_sample = up_sample
        self.net_channels = net_channels
        self.max_channels = max_channels
        self.net_act = net_act
        self.net_act_kargs = net_act_args if net_act_args is not None else dict()
        self.out_act = out_act
        self.out_act_kargs = out_act_kargs if out_act_kargs is not None else dict()
        self.out_label = out_label
        self.projector = dl_layers.OrthogonalProjection(axis=2) if self.out_label else None

        net_activation = self.net_act(**self.net_act_kargs) if inspect.isclass(self.net_act) else self.net_act
        common_kargs = dict(activation=net_activation, trainable=self.trainable, dtype=self.dtype)

        assert ndims in [2, 3], f'ndims: {ndims}'
        self.const_shape = [2, 2] if const_shape is None and ndims == 2 else self.const_shape
        modulate_conv_block = getattr(gtf_layers, f'ModulateConv{ndims}DBlock')
        convolution_block = getattr(gtf_layers, f'ConvolutionBlock{ndims}D')

        # Initialize Const
        self.const_mapping = gtf_layers.Const3DBlock(self.const_shape, self.d_latent, **common_kargs)

        # Initialize Up-Sample Modules
        up_channels = [self.net_channels * (2 ** (self.up_sample - c_ - 1)) for c_ in range(self.up_sample)]
        # up_channels = [min(self.max_channels, u_c_) for u_c_ in up_channels]
        min_channels = int(np.ceil(out_channels / self.net_channels)) * self.net_channels
        up_channels = np.clip(up_channels, min_channels, self.max_channels)
        self.up_scales = [modulate_conv_block(u_c_, 2, name=f'ModulateConv{u_id_}',  **common_kargs)
                          for u_id_, u_c_ in enumerate(up_channels)]

        # reduce_channel = np.clip(self.net_channels // 2, min_channels, self.max_channels)
        reduce_channel = self.net_channels // 2 if self.net_channels // 2 > out_channels else self.net_channels
        self.reduce = [modulate_conv_block(reduce_channel, 1, name=f'ReductionConv0', **common_kargs)]

        out_activation = self.out_act(**self.out_act_kargs) if inspect.isclass(self.out_act) else self.out_act
        out_kargs = dict(activation=out_activation, trainable=self.trainable, dtype=self.dtype)
        self.out = convolution_block(out_channels, 3, 1, name='ReductionConv1', **out_kargs)

    def decode_inputs(self, inputs) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        d_latent, l_latent, l_spatial = (inputs[0], None, None)
        if self.channel_embedding:
            l_latent = inputs[1]
        if self.spatial_embedding:
            l_spatial = inputs[2] if self.channel_embedding else inputs[1]
            l_spatial = l_spatial[::-1]
        return d_latent, l_latent, l_spatial

    def attach_spatial_embedding(self, x: tf.Tensor, spatial: tf.Tensor) -> tf.Tensor:
        b_c = spatial.shape.dims[-1].value
        f_c = x.shape.dims[-1].value
        if self.spatial_embedding == '':
            pass
        elif self.spatial_embedding == 'concat':
            raise NotImplementedError
        elif self.spatial_embedding == 'prod':
            x_r, x_l = tf.split(x, [f_c - b_c, b_c], axis=-1)
            x_l = x_l * spatial
            x = tf.concat([x_r, x_l], axis=-1)
        else:
            raise NotImplementedError
        return x

    def attach_channel_embedding(self, x: tf.Tensor, label: tf.Tensor) -> tf.Tensor:
        if self.channel_embedding == '':
            pass
        elif self.channel_embedding == 'concat':
            x = tf.concat([x, label], axis=-1)
        else:
            raise NotImplementedError
        return x

    def visualization_vol(self, vol, out_dict):
        vis_dict = out_dict if out_dict is not None else dict()
        color_palettes = gtf_layers.get_color_palettes(vol.shape.dims[-1].value)

        vis_num = MAX_VIS_OUT if vol.shape.dims[0].value >= MAX_VIS_OUT else vol.shape.dims[0].value
        vol_label = tf.argmax(vol[:vis_num], axis=-1)
        vol_top_view_label = self.projector(vol_label) if self.ndims == 3 else vol_label

        vol_top_view = tf.gather(color_palettes, vol_top_view_label)
        vis_dict['top_view'] = vol_top_view
        return vis_dict

    def call(self, inputs, training=None, as_master=True, vis_port=False, mask=None):
        d_latent, l_latent, l_spatial = self.decode_inputs(inputs)

        # Const
        dl_latent = self.attach_channel_embedding(d_latent, l_latent)
        x = self.const_mapping(dl_latent)

        # Up sampling
        for u_i, u_s in enumerate(self.up_scales):
            if self.spatial_embedding:
                x = self.attach_spatial_embedding(x, l_spatial[u_i])
            x = u_s([x, dl_latent if l_spatial is None else d_latent])

        if self.out_label:
            # Reduction
            for r in self.reduce:
                x = r([x, d_latent])

            # Out
            x = self.out(x)

        out_dict = dict(out=[x])
        if vis_port and self.out_label:
            out_dict = self.visualization_vol(x, out_dict)
        return out_dict


class Code2VolSpatialCondGenerator(Code2VolCondBaseGenerator):
    def __init__(self,
                 const_shape: Optional[List] = None,
                 d_latent: int = 512,
                 up_sample: int = 4,
                 net_channels: int = 64,
                 out_channels: int = 20,
                 max_channels: int = 512,
                 net_act=tf.nn.leaky_relu,
                 net_act_args=None,
                 out_act=tf.identity,
                 out_act_kargs: Dict = None,
                 trainable: bool = True,
                 dtype: tf.DType = tf.float32,
                 name: str = None
                 ):
        super().__init__(channel_embedding='concat', spatial_embedding='prod',
                         const_shape=const_shape, d_latent=d_latent, up_sample=up_sample, net_channels=net_channels,
                         out_channels=out_channels, max_channels=max_channels, net_act=net_act,
                         net_act_args=net_act_args, out_act=out_act, out_act_kargs=out_act_kargs, trainable=trainable,
                         dtype=dtype, name=name)
