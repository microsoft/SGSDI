# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from typing import List

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.python.keras.layers.convolutional import Conv

from GraphicsDL.graphicstf import layers as gtf_layers


MAX_VIS_OUT = 4


class StrideExtractionBlock(layers.Layer):
    def __init__(self,
                 rank,
                 filters,
                 data_format: str = None,
                 trainable: bool = True,
                 name: str = None,
                 dtype: tf.dtypes = tf.float32):
        super().__init__(trainable=trainable, name=name, dtype=dtype)
        self.data_format = K.image_data_format() if data_format is None else data_format
        self.norm = gtf_layers.InstanceNorm(True, True, trainable=self.trainable, name='norm', dtype=self.dtype)
        self.conv = Conv(rank, filters, 4, 2, padding='same', data_format=self.data_format)
        self.act = layers.LeakyReLU(alpha=0.2)

    def call(self, inputs: tf.Tensor, **kwargs):
        return self.act(self.norm(self.conv(inputs)))


class StrideExtractionBlock3D(StrideExtractionBlock):
    def __init__(self,
                 filters,
                 data_format: str = None,
                 trainable: bool = True,
                 name: str = None,
                 dtype: tf.dtypes = tf.float32):
        super().__init__(3, filters, data_format, trainable=trainable, name=name, dtype=dtype)


class LabelExtraction3D(layers.Layer):
    def __init__(self,
                 out_channels: int,
                 net_channels: int,
                 down_sample: int,
                 method: str = 'spatial',
                 trainable: bool = True,
                 name: str = None,
                 dtype: tf.dtypes = tf.float32):
        super().__init__(trainable=trainable, name=name, dtype=dtype)
        self.method = method
        self.ds_blocks = [StrideExtractionBlock3D(filters=min(net_channels*(2**d_), 256)) for d_ in range(down_sample)]
        self.dense = layers.Dense(out_channels)

    def call(self, inputs, **kwargs):
        x = inputs
        x_stack = list()
        for ds in self.ds_blocks:
            x = ds(x)
            x_stack.append(x)
        x = tf.reshape(x, [x.shape.dims[0].value, -1])
        x = self.dense(x)
        if self.method == 'spatial':
            return [x, x_stack]
        elif self.method == 'embedding':
            return [x]
        else:
            raise NotImplementedError


class RoomBoundaryCasting(layers.Layer):
    def __init__(self, 
                 vox_size: List,
                 trainable: bool = True,
                 name: str = None,
                 dtype: tf.dtypes = tf.float32):
        super().__init__(trainable=trainable, name=name, dtype=dtype)
        self.vox_size = vox_size

    @staticmethod
    def sample_with_bounding_box(bounding_box: tf.Tensor, vox_size: List):
        vox_grid = tf.stack(tf.meshgrid(*[tf.range(a_, dtype=tf.float32) for a_ in vox_size], indexing='ij'), axis=-1)
        bbox_shape = [bounding_box.shape.dims[0].value, *[1] * len(vox_size), bounding_box.shape.dims[-1].value]
        box_max, box_min = tf.split(tf.reshape(bounding_box, bbox_shape), 2, axis=-1)
        box_size = box_max - box_min
        vox_scale = box_size / tf.convert_to_tensor(vox_size, dtype=tf.float32)
        vox_grid = tf.cast(vox_grid[tf.newaxis, ...] * vox_scale + box_min, tf.int32)
        batch_grid = tf.range(bounding_box.shape.dims[0].value, dtype=tf.int32)
        batch_grid = tf.tile(tf.reshape(batch_grid, [-1, 1, 1, 1, 1]), (1,) + tuple(vox_size) + (1,))
        grid = tf.reshape(tf.concat([batch_grid, vox_grid], axis=-1), [-1, 4])
        feat = tf.ones([grid.shape.dims[0].value], dtype=tf.int32)
        mask = tf.scatter_nd(grid, feat, (bounding_box.shape.dims[0].value,) + tuple(vox_size))[..., tf.newaxis]
        mask = tf.where(mask > 0, tf.ones_like(mask), tf.zeros_like(mask))
        return mask

    def call(self, inputs: List[tf.Tensor], **kwargs) -> tf.Tensor:
        assert len(inputs) == 1
        bounding_box, = tuple(inputs)
        return tf.cast(self.sample_with_bounding_box(bounding_box, self.vox_size), tf.float32)


class RoomSizeEmbedding(tf.keras.Model):
    def __init__(self,
                 out_channels: int,
                 net_channels: int,
                 vox_size: List,
                 down_sample: int = 4,
                 method: str = 'spatial',
                 trainable: bool = True,
                 name: str = None,
                 dtype: tf.dtypes = tf.float32):
        super().__init__(trainable=trainable, name=name, dtype=dtype)
        self.room_sizer = RoomBoundaryCasting(vox_size=vox_size, trainable=trainable, dtype=dtype)
        self.extract = LabelExtraction3D(out_channels, net_channels, down_sample, method=method,
                                         trainable=trainable, name=name, dtype=dtype)

    def call(self, inputs, as_master=True, training=None, mask=None):
        return self.extract(self.room_sizer(inputs))
