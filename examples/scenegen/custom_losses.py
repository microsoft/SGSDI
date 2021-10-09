# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import tensorflow as tf
from typing import Tuple, List
from GraphicsDL.graphicstf.basic import BasicLossProxy


class BboxCondHingeLossGProxy(BasicLossProxy):
    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)
        self._losses_name = ['HingeLossG', 'BboxLoss']

    @staticmethod
    def bbox_reg_loss(vox: tf.Tensor, bbox: tf.Tensor):
        vox_axis = [tf.range(a_, dtype=tf.int32) for a_ in vox.shape.dims[:-1]]
        vox_meshgrid = tf.stack(tf.meshgrid(*vox_axis, indexing='ij'), axis=-1)
        xyz_meshgrid = tf.cast(vox_meshgrid[..., 1:], bbox.dtype)

        box_max, box_min = tf.split(bbox, 2, axis=-1)
        batch_box_max = tf.gather(box_max, vox_meshgrid[..., 0])
        batch_box_min = tf.gather(box_min, vox_meshgrid[..., 0])
        within_max = tf.reduce_all(xyz_meshgrid < batch_box_max, axis=-1)
        within_min = tf.reduce_all(xyz_meshgrid >= batch_box_min, axis=-1)
        outside_vox_indices = tf.where(tf.logical_not(tf.logical_and(within_max, within_min)))
        inside_vox = tf.gather_nd(vox, outside_vox_indices)

        outside_ce_loss = -tf.math.log(tf.clip_by_value(inside_vox[..., 0], 1e-15, 1.0))
        return tf.reduce_mean(outside_ce_loss)

    def call(self, inputs: List[tf.Tensor], **kwargs):
        assert len(inputs) == 3
        vox, bbox, fake_out = tuple(inputs)
        hinge_loss = tf.reduce_mean(-fake_out)
        bbox_loss = self.bbox_reg_loss(tf.nn.softmax(vox), bbox)
        return [hinge_loss, bbox_loss]


class HingeLossDProxy(BasicLossProxy):
    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)
        self._losses_name = ['HingeLossD']

    def call(self, gan_inputs: Tuple[tf.Tensor, tf.Tensor], **kwargs):
        fake_out, real_out = gan_inputs[:2]
        loss = tf.reduce_mean(tf.nn.relu(tf.convert_to_tensor(1., dtype=tf.float32) - real_out) +
                              tf.nn.relu(tf.convert_to_tensor(1., dtype=tf.float32) + fake_out))
        return [loss]
