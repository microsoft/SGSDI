# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import math
import numpy as np
import tensorflow as tf
from abc import abstractmethod
from typing import Tuple, List, Optional

from .custom_layers import get_color_palettes, DRCClsRendering
from GraphicsDL.graphicstf.basic import BasicModel
from GraphicsDL.graphicstf import layers as gtf_layers
from GraphicsDL.graphicsutils.miscellaneous import palettes


MAX_VIS_OUT = 4


class BaseCameraSampler(BasicModel):
    def __init__(self,
                 view_num: int = 4,
                 camera_num: int = 1,
                 random_view: bool = False,
                 shuffle_view: bool = False,
                 seed: int = None,
                 trainable: bool = False,
                 dtype: tf.DType = tf.float32,
                 name: str = None):
        super().__init__(trainable=trainable, name=name, dtype=dtype)
        assert view_num <= 8 and camera_num <= 8
        assert view_num % camera_num == 0 or view_num < camera_num
        self.seed = seed if seed is not None and seed > 0 else None
        self.random_view = random_view
        self.shuffle_view = shuffle_view
        self.view_num = math.ceil(view_num / 4) * 4
        self.camera_num = camera_num
        self.out_view_num = view_num
        self.logged = False

    @staticmethod
    def _compute_seg_max_min(refs, seg_ids, batch_size):
        max_ref_: tf.Tensor = tf.math.segment_max(refs, seg_ids)
        max_ref_.set_shape(batch_size)
        min_ref_ = tf.math.segment_min(refs, seg_ids)
        min_ref_.set_shape(batch_size)
        return max_ref_, min_ref_

    @staticmethod
    def update_vol(vol_occupy, update_value=1):
        batch_size = vol_occupy.shape.dims[0].value
        boxes_center = np.tile(np.asarray(vol_occupy.shape.as_list()[1:]) // 2, (batch_size, 1))
        batch_boxes_center = np.concatenate((np.arange(batch_size)[..., np.newaxis], boxes_center), axis=-1)
        vol_occupy_out = tf.tensor_scatter_nd_update(vol_occupy, batch_boxes_center, [update_value] * batch_size)
        return vol_occupy_out

    def compute_bounding_box(self, vol_occupy: tf.Tensor) -> tf.Tensor:
        """
        Compute the bounding box of a given volume. If the given volume is empty, the returned bounding box center is
            located at the center of volume.

        Args:
            vol_occupy: an `int32` Tensor with shape [N,D,H,W]. 0 denotes empty, otherwise denotes non-empty.

        Returns:
            an `int32` Tensor with shape [N,9], indicating center, maximum and minimum, respectively.
        """
        assert vol_occupy.shape.ndims == 4
        batch_size = vol_occupy.shape.dims[0].value
        with tf.name_scope('ComputeBoundingBox'):
            vol_occupy = self.update_vol(vol_occupy)
            arg_occupy = tf.where(vol_occupy > tf.convert_to_tensor(0))
            b, d, h, w = tf.unstack(arg_occupy, axis=-1)
            d_max, d_min = self._compute_seg_max_min(d, b, batch_size)
            h_max, h_min = self._compute_seg_max_min(h, b, batch_size)
            w_max, w_min = self._compute_seg_max_min(w, b, batch_size)
            boxes_max = tf.stack([d_max, h_max, w_max], axis=-1)
            boxes_min = tf.stack([d_min, h_min, w_min], axis=-1)
            return tf.cast(tf.concat([boxes_max, boxes_min], axis=-1), tf.float32)

    @abstractmethod
    def sample_camera_rotation(self, vol: tf.Tensor, cam_t: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Sample proper camera rotations based on the volume and position content

        Args:
            vol: a `int32` Tensor with shape [N,D,H,W], the semantic logit volume
            cam_t: a `float32` Tensor with shape [N,camera_num,3], the candidate positions

        Returns:
            cam_r: a `float32` Tensor with shape [N,view_num,3,3], the candidate rotations
            cam_t: a `float32` Tensor with shape [N,view_num,3], the candidate positions
        """
        pass

    @abstractmethod
    def sample_camera_translate(self, vol: tf.Tensor, bbox: Optional[tf.Tensor] = None) -> tf.Tensor:
        """
        Sample proper camera positions based on the volume content

        Args:
            vol: a `int32` Tensor with shape [N,D,H,W], the semantic logit volume
            bbox (tf.Tensor): a `float32` Tensor with shape [N, 6], the first 3-elems in last dim are the maximum point,
                while the last ones are the minimum point.

        Returns:
            a `float32` Tensor with shape [N,camera_num,3], the candidate positions
        """
        pass

    def shuffle_views(self, cam_r: tf.Tensor, cam_t: tf.Tensor):
        assert cam_r.shape.ndims == 4
        view_dim = cam_r.shape.dims[1].value
        batch_dim = cam_r.shape.dims[0].value
        shuffle_offset = tf.random.shuffle(tf.range(0, view_dim, dtype=tf.int32), seed=self.seed)
        shuffle_index = tf.tile(shuffle_offset[tf.newaxis, ...], [batch_dim, 1])
        shuffle_cam_r = tf.gather(cam_r, shuffle_index, batch_dims=1)
        shuffle_cam_t = tf.gather(cam_t, shuffle_index, batch_dims=1)
        return shuffle_cam_r, shuffle_cam_t

    def call(self, inputs: List[tf.Tensor], training=None, as_master=True, vis_port=False, mask=None):
        assert len(inputs) in [1, 2]
        if len(inputs) == 1:
            vol, = inputs
            bbox = None
        else:
            vol, bbox = tuple(inputs)
        vol_label = tf.argmax(vol, axis=-1, output_type=tf.int32)
        vol_occupy = tf.where(vol_label > 0, tf.ones_like(vol_label), tf.zeros_like(vol_label))
        bbox = self.compute_bounding_box(vol_occupy) if bbox is None else bbox
        cam_t = self.sample_camera_translate(vol_occupy, bbox)
        cam_r, cam_t = self.sample_camera_rotation(vol_occupy, cam_t)
        if self.view_num != self.out_view_num:
            if self.out_view_num < 4:
                assert self.shuffle_view
                cam_r, cam_t = self.shuffle_views(cam_r, cam_t)
                cam_r, cam_t = cam_r[:, :self.out_view_num], cam_t[:, :self.out_view_num]
            else:
                cam_r_bk, cam_t_bk = self.shuffle_views(cam_r[:, 4:], cam_t[:, 4:])
                cam_r_random, cam_t_random = cam_r_bk[:, :self.out_view_num - 4], cam_t_bk[:, :self.out_view_num - 4]
                cam_r = tf.concat([cam_r[:, :4], cam_r_random], axis=1)
                cam_t = tf.concat([cam_t[:, :4], cam_t_random], axis=1)
                cam_r, cam_t = self.shuffle_views(cam_r, cam_t) if self.shuffle_view else (cam_r, cam_t)
        elif self.shuffle_view:
            cam_r, cam_t = self.shuffle_views(cam_r, cam_t)
        cam_t = cam_t / vol_occupy.shape.dims[1].value
        return cam_r, cam_t


class RandomFarCameraSampler(BaseCameraSampler):
    def __init__(self,
                 view_num: int,
                 camera_num: int,
                 shuffle_view: bool,
                 seed: int,
                 trainable: bool = False,
                 dtype: tf.dtypes = tf.float32,
                 name: str = None):
        super().__init__(view_num=view_num, camera_num=camera_num, shuffle_view=shuffle_view, seed=seed, 
                         trainable=trainable, dtype=dtype, name=name)

    @staticmethod
    def update_vol(vol_occupy, update_value=1):
        vol_shape = np.array(vol_occupy.shape.as_list())
        vol_center = np.tile(vol_shape[1:].reshape([1, 1, 3]) // 2, (vol_shape[0], 4, 1))
        vol_center_indices = vol_center + np.array([[-1, -1, -1], [0, -1, -1], [0, -1, 0], [-1, -1, 0]])
        batch_indices = np.tile(np.arange(vol_shape[0])[..., np.newaxis, np.newaxis], [1, 4, 1])
        batch_vol_center_indices = np.concatenate((batch_indices, vol_center_indices), axis=-1)
        batch_area_pad_updates = np.zeros((vol_shape[0], 4), np.int32) + update_value
        vol_occupy_out = tf.tensor_scatter_nd_update(vol_occupy, batch_vol_center_indices, batch_area_pad_updates)
        return vol_occupy_out

    @staticmethod
    def get_bbox_point(bbox: tf.Tensor, area_ratio=0.5, floor_center=16, max_height=8, min_height=7):
        x_max, y_max, z_max, x_min, y_min, z_min = tf.split(bbox, 6, axis=-1)
        x_area_size = tf.math.ceil(tf.maximum(x_max - floor_center, floor_center - x_min) * area_ratio)
        z_area_size = tf.math.ceil(tf.maximum(z_max - floor_center, floor_center - z_min) * area_ratio)
        x_max, x_min = floor_center + x_area_size, floor_center - x_area_size
        z_max, z_min = floor_center + z_area_size, floor_center - z_area_size
        y_max = tf.minimum(y_max, max_height)
        y_min = tf.maximum(y_min, min_height)
        return x_max, y_max, z_max, x_min, y_min, z_min

    @staticmethod
    def quadrant_bbox(bbox: tf.Tensor, area_ratio=0.5, floor_center=16, max_height=8, min_height=7) -> List[tf.Tensor]:
        """
        Generate the quadrant bounding box with given bounding box information

        Args:
            bbox (tf.Tensor): a `float32` Tensor with shape [N, 6], the first 3-elems in last dim are the maximum point,
                while the last ones are the minimum point.
            area_ratio:
            floor_center:
            max_height:
            min_height:

        Returns:
            tf.Tensor: a 4-length list of `float32` Tensor with shape [N, 6], with the identity definition of `bbox`
        """
        x_max, y_max, z_max, x_min, y_min, z_min = \
            RandomFarCameraSampler.get_bbox_point(bbox, area_ratio, floor_center, max_height, min_height)
        x_c, y_c, z_c = (x_max + x_min) / 2, y_min, (z_max + z_min) / 2
        center = tf.concat([x_c, y_c, z_c], axis=-1)

        quadrant_bbox_list = list()
        for q_i, x_r, z_r in zip(range(4), [x_min, x_max, x_max, x_min], [z_min, z_min, z_max, z_max]):
            q_pts = tf.stack([center, tf.concat([x_r, y_max, z_r], axis=-1)], axis=-1)
            q_bbox = tf.concat([tf.reduce_max(q_pts, axis=-1), tf.reduce_min(q_pts, axis=-1)], axis=-1)
            quadrant_bbox_list.append(q_bbox)
        return quadrant_bbox_list

    @staticmethod
    def sample_proper_positions(vol: tf.Tensor, bbox: tf.Tensor, max_box_size, num_samples: int = 1, seed=None):
        # Basic information
        box_max, box_min = tf.split(bbox, 2, axis=-1)

        # Generate meshgrid
        box_scale = (box_max - box_min) / max_box_size[tf.newaxis, ...]
        box_meshaxis = [tf.range(0, s_, dtype=tf.int32) for s_ in max_box_size]
        box_meshgrid = tf.stack(tf.meshgrid(*box_meshaxis, indexing='ij'), axis=-1)[tf.newaxis, ...]
        box_coord = tf.cast(box_meshgrid, tf.float32) * box_scale[:, tf.newaxis, tf.newaxis, tf.newaxis, :]
        box_coord = box_coord + box_min[:, tf.newaxis, tf.newaxis, tf.newaxis, :]

        # Existence sampling
        box_index = tf.cast(box_coord, tf.int32)
        box_existence = tf.where(tf.gather_nd(vol, box_index, batch_dims=1) < 1)
        seg_counts = tf.math.segment_sum(tf.ones_like(box_existence[..., 0]), box_existence[..., 0])
        seg_cumsum = tf.math.cumsum(seg_counts, exclusive=True)
        rand_pos = tf.random.uniform((vol.shape.dims[0].value, num_samples), seed=seed)
        cam_pose_offset = tf.random.uniform((vol.shape.dims[0].value, num_samples, box_max.shape.dims[-1].value),
                                            dtype=tf.float32, seed=seed)
        rand_index = rand_pos * tf.cast(seg_counts[:, tf.newaxis], dtype=tf.float32)
        rand_index = rand_index + tf.cast(seg_cumsum[:, tf.newaxis], dtype=tf.float32)
        proper_index = tf.cast(tf.gather(box_existence[..., 1:], tf.cast(rand_index, tf.int32)), tf.float32)
        proper_coord = tf.floor(proper_index * box_scale[:, tf.newaxis, :]) + cam_pose_offset
        proper_coord = proper_coord + box_min[:, tf.newaxis, :]
        return proper_coord

    def sample_camera_position_from_bboxes(self, vol, quad_boxes, max_box_size, cam_per_quad):
        quad_cams_t = list()
        for q_b in quad_boxes:
            q_cam_t = self.sample_proper_positions(vol, q_b, max_box_size, cam_per_quad, self.seed)
            quad_cams_t.append(q_cam_t)
        quad_cams_t = tf.reshape(tf.stack(quad_cams_t, axis=2), [vol.shape.dims[0].value, self.camera_num, 3])
        return quad_cams_t

    def sample_camera_translate(self, vol: tf.Tensor, bbox: Optional[tf.Tensor] = None) -> tf.Tensor:
        assert self.camera_num % 4 == 0
        # Assign non-full volumetric
        vol_shape = np.array(vol.shape.as_list())
        vol = self.update_vol(vol, update_value=0)

        area_ratio = 0.5
        max_box_size = np.ceil(vol_shape[1:] * area_ratio / 2)
        max_box_size[1] = 1
        cam_per_quad = self.camera_num // 4
        max_height, min_height = vol_shape[2] // 2, vol_shape[2] // 2 - max(int(vol_shape[2]*0.03125), 1)
        quad_boxes = self.quadrant_bbox(bbox, vol_shape[1] // 2, max_height, min_height)
        quad_cams_t = self.sample_camera_position_from_bboxes(vol, quad_boxes, max_box_size, cam_per_quad)
        return quad_cams_t

    def sample_camera_rotation(self, vol: tf.Tensor, cam_t: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        cam_i = tf.convert_to_tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=tf.float32)
        fix_i = tf.convert_to_tensor([[0, 0, -1], [0, 1, 0], [1, 0, 0]], dtype=tf.float32)
        cam_r = list()
        for n in range(self.view_num):
            cam_r.append(cam_i)
            cam_i = tf.matmul(fix_i, cam_i)
        cam_r = tf.stack(cam_r, axis=0)
        cam_r = tf.tile(cam_r[tf.newaxis], (cam_t.shape.dims[0].value, 1, 1, 1))
        return cam_r, cam_t


class RandomStructured3DCameraSampler(RandomFarCameraSampler):
    def __init__(self,
                 room_height: float,
                 view_num: int,
                 camera_num: int,
                 shuffle_view: bool,
                 seed: int,
                 trainable: bool = False,
                 dtype: tf.dtypes = tf.float32,
                 name: str = None):
        self.room_height = room_height if room_height > 0 else 3.2
        self.cam_height_mean = 1.5
        self.cam_height_stddev = 0.1
        self.cam_rot_z_mean = 0.14
        self.cam_rot_z_stddev = 5.42
        self.area_ratio = 0.8
        super().__init__(view_num=view_num, camera_num=camera_num, shuffle_view=shuffle_view, seed=seed,
                         trainable=trainable, dtype=dtype, name=name)

    @staticmethod
    def quadrant_bbox(bbox: tf.Tensor, area_ratio=0.5, floor_center=16, max_height=8, min_height=7):
        x_max, y_max, z_max, x_min, y_min, z_min = \
            RandomFarCameraSampler.get_bbox_point(bbox, area_ratio, floor_center, max_height, min_height)
        x_c = (x_max + x_min) / 2
        z_c = (z_max + z_min) / 2
        quadrant_bbox_list = list()
        for q_i, x_r, z_r in zip(range(4), [x_min, x_max, x_max, x_min], [z_min, z_min, z_max, z_max]):
            q_pts = tf.stack([tf.concat([x_c, y_min[:, q_i:q_i+1], z_c], axis=-1),
                              tf.concat([x_r, y_max[:, q_i:q_i+1], z_r], axis=-1)], axis=-1)
            q_bbox = tf.concat([tf.reduce_max(q_pts, axis=-1), tf.reduce_min(q_pts, axis=-1)], axis=-1)
            quadrant_bbox_list.append(q_bbox)
        return quadrant_bbox_list

    def sample_camera_translate(self, vol: tf.Tensor, bbox: Optional[tf.Tensor] = None) -> tf.Tensor:
        assert self.camera_num % 4 == 0
        # Assign non-full volumetric
        vol_shape = np.array(vol.shape.as_list())
        vol_center = np.tile(vol_shape[1:].reshape([1, 1, 1, 3]) // 2, (vol_shape[0], vol_shape[2], 4, 1))
        vol_center[..., 1] = np.arange(vol_shape[2]).reshape([-1, 1])
        vol_center_indices = vol_center + np.array([[-1, 0, -1], [0, 0, -1], [0, 0, 0], [-1, 0, 0]])
        batch_indices = np.tile(np.arange(vol_shape[0])[..., None, None, np.newaxis], [1, vol_shape[2], 4, 1])
        batch_vol_center_indices = np.concatenate((batch_indices, vol_center_indices), axis=-1)
        batch_area_pad_updates = np.zeros((vol_shape[0], vol_shape[2], 4), np.int32)
        vol = tf.tensor_scatter_nd_update(vol, batch_vol_center_indices, batch_area_pad_updates)

        max_box_size = np.ceil(vol_shape[1:] * self.area_ratio / 2)
        max_box_size[1] = 1
        cam_per_quad = self.camera_num // 4
        cam_height = tf.random.truncated_normal([vol_shape[0], 4], self.cam_height_mean, self.cam_height_stddev,
                                                seed=self.seed) / self.room_height * vol_shape[2]
        max_height, min_height = tf.math.ceil(cam_height), tf.math.floor(cam_height)
        quad_boxes = self.quadrant_bbox(bbox, self.area_ratio, vol_shape[1] // 2, max_height, min_height)
        quad_cams_t = self.sample_camera_position_from_bboxes(vol, quad_boxes, max_box_size, cam_per_quad)
        return quad_cams_t

    @staticmethod
    def rotation_matrix_from_angle_x(angle_y):
        angle_cos, angle_sin = tf.cos(angle_y), tf.sin(angle_y)
        cam_r_zeros, cam_r_ones = tf.zeros_like(angle_cos), tf.ones_like(angle_cos)
        cam_r_f = tf.stack((cam_r_ones, cam_r_zeros, cam_r_zeros), axis=1)
        cam_r_s = tf.stack((cam_r_zeros, angle_cos, angle_sin), axis=1)
        cam_r_t = tf.stack((cam_r_zeros, angle_sin * -1, angle_cos), axis=1)
        cam_r = tf.stack((cam_r_f, cam_r_s, cam_r_t), axis=1)
        return cam_r

    @staticmethod
    def rotation_matrix_from_angle_y(angle_y):
        angle_cos, angle_sin = tf.cos(angle_y), tf.sin(angle_y)
        cam_r_zeros, cam_r_ones = tf.zeros_like(angle_cos), tf.ones_like(angle_cos)
        cam_r_f = tf.stack((angle_cos, cam_r_zeros, angle_sin * -1), axis=1)
        cam_r_s = tf.stack((cam_r_zeros, cam_r_ones, cam_r_zeros), axis=1)
        cam_r_t = tf.stack((angle_sin, cam_r_zeros, angle_cos), axis=1)
        cam_r = tf.stack((cam_r_f, cam_r_s, cam_r_t), axis=1)
        return cam_r

    @staticmethod
    def rotation_matrix_from_angle_z(angle_z):
        angle_cos, angle_sin = tf.cos(angle_z), tf.sin(angle_z)
        cam_r_zeros, cam_r_ones = tf.zeros_like(angle_cos), tf.ones_like(angle_cos)
        cam_r_f = tf.stack((angle_cos, angle_sin, cam_r_zeros), axis=1)
        cam_r_s = tf.stack((-1 * angle_sin, angle_cos, cam_r_zeros), axis=1)
        cam_r_t = tf.stack((cam_r_zeros, cam_r_zeros, cam_r_ones), axis=1)
        cam_r = tf.stack((cam_r_f, cam_r_s, cam_r_t), axis=1)
        return cam_r

    def z_rotation(self, cam_y_r):
        angle_z_random = tf.random.truncated_normal([cam_y_r.shape.dims[0].value, ], self.cam_rot_z_mean,
                                                    self.cam_rot_z_stddev, dtype=tf.float32, seed=self.seed)
        rotate_angle_z = angle_z_random / 180 * np.pi
        cam_z_r = self.rotation_matrix_from_angle_z(rotate_angle_z)
        cam_r = tf.matmul(cam_y_r, cam_z_r)
        return cam_r

    def x_rotation(self, cam_z_r):
        angle_x_random = tf.random.truncated_normal([cam_z_r.shape.dims[0].value, ], self.cam_rot_x_mean,
                                                    self.cam_rot_x_stddev, dtype=tf.float32, seed=self.seed)
        rotate_angle_x = angle_x_random / 180 * np.pi
        cam_x_r = self.rotation_matrix_from_angle_x(rotate_angle_x)
        cam_r = tf.matmul(cam_z_r, cam_x_r)
        return cam_r

    def sample_camera_rotation(self, vol: tf.Tensor, cam_t: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        rotate_angle_base = np.array([0, np.pi / 2, np.pi, np.pi * 3 / 2]) - np.pi / 2
        angle_random = tf.random.uniform([cam_t.shape.dims[0].value, self.camera_num], dtype=tf.float32, seed=self.seed)
        rotate_angle = tf.reshape(angle_random * np.pi + rotate_angle_base, [-1])
        cam_r = self.rotation_matrix_from_angle_y(rotate_angle)
        cam_r = tf.reshape(cam_r, [cam_t.shape.dims[0].value, self.camera_num, 3, 3])
        return cam_r, cam_t


class RandomStructured3DFarCameraSampler(RandomStructured3DCameraSampler):
    def __init__(self,
                 room_height: float,
                 view_num: int,
                 camera_num: int,
                 shuffle_view: bool,
                 seed: int,
                 trainable: bool = False,
                 dtype: tf.dtypes = tf.float32,
                 name: str = None):
        super().__init__(room_height=room_height, view_num=view_num, camera_num=camera_num, shuffle_view=shuffle_view,
                         seed=seed, trainable=trainable, dtype=dtype, name=name)

    def sample_camera_rotation(self, vol: tf.Tensor, cam_t: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        rotate_angle_base = np.array([0, np.pi / 2, np.pi, np.pi * 3 / 2])
        rotate_angle_base = np.tile(rotate_angle_base, self.camera_num // rotate_angle_base.shape[0])
        angle_random = tf.random.uniform([cam_t.shape.dims[0].value, self.camera_num], dtype=tf.float32, seed=self.seed)
        rotate_angle = tf.reshape(angle_random * np.pi / 2 + rotate_angle_base, [-1])
        cam_y_r = self.rotation_matrix_from_angle_y(rotate_angle)
        cam_r = self.z_rotation(cam_y_r)
        cam_r = tf.reshape(cam_r, [cam_t.shape.dims[0].value, self.camera_num, 3, 3])
        return cam_r, cam_t


class RandomMatterport3DFarCameraSampler(RandomStructured3DFarCameraSampler):
    def __init__(self,
                 room_height: float,
                 view_num: int,
                 camera_num: int,
                 shuffle_view: bool,
                 seed: int,
                 trainable: bool = False,
                 dtype: tf.dtypes = tf.float32,
                 name: str = None):
        super().__init__(room_height=room_height, view_num=view_num, camera_num=camera_num, shuffle_view=shuffle_view,
                         seed=seed, trainable=trainable, dtype=dtype, name=name)
        self.cam_height_mean = 1.5
        self.cam_height_stddev = 0.08
        self.cam_rot_z_mean = -0.06
        self.cam_rot_z_stddev = 2.04


class RenderingViews(BasicModel):
    def __init__(self,
                 vox_size: List[int],
                 view_size: List[int],
                 view_fov: List[float],
                 with_depth: bool,
                 out_activation: str,
                 trainable: bool = False,
                 dtype: tf.dtypes = tf.float32,
                 name: str = None):
        super().__init__(trainable=trainable, dtype=dtype, name=name)
        self.vox_size = vox_size
        self.view_size = view_size
        self.camera_fov_radian = view_fov
        self.with_depth = with_depth
        self.out_activation = out_activation

        self.renderer = None
        self.projector = gtf_layers.OrthogonalProjection(axis=2)

    def build(self, input_shape):
        self.renderer = DRCClsRendering(vox_size=self.vox_size,
                                        sample_size=self.view_size, out_activation=self.out_activation,
                                        cam_fov=self.camera_fov_radian, out_depth=self.with_depth)

    @staticmethod
    def views_visualization(vis_dict, view_label, view_depth, color_palettes):
        def concat_multiple_views(views_: tf.Tensor):
            if views_.shape.ndims == 5:
                view_shape = views_.shape.dims
                views_ = tf.transpose(views_, [0, 2, 1, 3, 4])
                views_ = tf.reshape(views_, (view_shape[0], view_shape[2], -1, view_shape[4]))
            return views_

        # Semantic view visualization
        vol_view_label = tf.argmax(view_label[:MAX_VIS_OUT, ...], axis=-1)
        vol_view_color: tf.Tensor = tf.gather(color_palettes, vol_view_label)
        vol_view_color = concat_multiple_views(vol_view_color)
        vis_dict['render'] = vol_view_color

        # Depth view visualization
        if view_depth is not None:
            view_depth = view_depth[:MAX_VIS_OUT, ...]
            l_dim = view_depth.shape.dims[-1].value
            if l_dim != 1:
                view_depth = tf.cast(tf.argmax(view_depth, axis=-1), tf.float32) / float(l_dim)
            view_depth = concat_multiple_views(view_depth)
            vis_dict['depth'] = view_depth

        return vis_dict

    def images_visualization(self, vol_view: tf.Tensor, vol: tf.Tensor, cam_t: tf.Tensor, out_dict=None):
        color_palettes = get_color_palettes(vol.shape.dims[-1].value)
        vis_dict = out_dict if out_dict is not None else dict()
        vol_label = tf.argmax(vol[:MAX_VIS_OUT], axis=-1)

        # visualize vol and camera
        cam_pose = cam_t[:MAX_VIS_OUT] * vol.shape.dims[-2].value
        cam_pose = cam_pose if cam_pose.shape.ndims == 3 else cam_pose[:, tf.newaxis]
        batch_index = np.tile(np.arange(MAX_VIS_OUT)[:, np.newaxis, np.newaxis], [1, cam_pose.shape.dims[1].value, 1])
        batch_cam_pose = tf.concat((batch_index, tf.cast(cam_pose, tf.int64)), axis=-1)
        camera_label = np.zeros((MAX_VIS_OUT, cam_pose.shape.dims[1].value), np.int64) + len(palettes.d3c20_rgb())
        vol_cam_label = tf.tensor_scatter_nd_update(vol_label, batch_cam_pose, camera_label)
        vol_top_view_label = self.projector(vol_cam_label)
        vol_top_view = tf.gather(color_palettes, vol_top_view_label)
        vis_dict['top_view'] = vol_top_view

        view_cls, view_depth = vol_view if self.with_depth else (vol_view[0], None)
        vis_dict = self.views_visualization(vis_dict, view_cls, view_depth, color_palettes)
        return vis_dict

    def renderer_multiple_views(self, vol: tf.Tensor, cam_r, cam_t, dense):
        batch_size, view_num = tuple(vol.shape.dims[:2])
        vol_p = tf.reshape(vol, (batch_size * view_num,) + tuple(vol.shape.dims[2:]))
        cam_r_p = tf.reshape(cam_r, (batch_size * view_num, 3, 3))
        cam_t_p = tf.reshape(cam_t, (batch_size * view_num, 3))
        return self.renderer([vol_p, cam_r_p, cam_t_p, dense])

    def call(self, inputs: List[tf.Tensor], training=None, as_master=True, vis_port=True, mask=None):
        vol, cam_r, cam_t = tuple(inputs[:3])
        dense = inputs[3] if len(inputs) > 3 else None
        out_dict = dict()
        if cam_r.shape.ndims == 4:
            mul_views_num = cam_r.shape.dims[1].value
            vol_view_list = [self.renderer([vol, cam_r[:, v_i], cam_t[:, v_i], dense]) for v_i in range(mul_views_num)]
            vol_view = [tf.stack(v_l, axis=1) for v_l in zip(*vol_view_list)]
        elif cam_r.shape.ndims == 3:
            vol_view = self.renderer([vol, cam_r, cam_t, dense])
        else:
            raise NotImplementedError
        if cam_r.shape.ndims == 4:
            vol_view = [tf.reshape(v_v_, tuple(cam_r.shape.dims[:2]) + tuple(self.view_size) + (-1,))
                        for v_v_ in vol_view]
        out_dict['out'] = vol_view
        if vis_port:
            out_dict = self.images_visualization(vol_view, vol, cam_t, out_dict)
        return out_dict


class RandomFarViewRenderer(BasicModel):
    def __init__(self,
                 view_num: int,
                 camera_num: int,
                 shuffle_view: bool,
                 vox_size: List[int],
                 view_size: List[int],
                 view_fov: List[float],
                 with_depth: bool,
                 out_activation: str,
                 seed: int = None,
                 trainable: bool = False,
                 dtype: tf.dtypes = tf.float32,
                 name: str = None):
        super().__init__(trainable=trainable, dtype=dtype, name=name)
        self.sampler = RandomFarCameraSampler(view_num, camera_num, shuffle_view,
                                              seed=seed, trainable=trainable, dtype=dtype)
        out_activation = out_activation if out_activation != 'none' else ''
        self.renderer = RenderingViews(vox_size=vox_size, view_size=view_size, view_fov=view_fov, with_depth=with_depth,
                                       out_activation=out_activation, trainable=trainable, dtype=dtype)

    def call(self, inputs: List[tf.Tensor], training=None, as_master=True, vis_port=True, mask=None):
        assert len(inputs) == 1
        vox, = inputs

        cam_r, cam_t = self.sampler([vox], training=training, as_master=as_master)
        imgs = self.renderer([vox, cam_r, cam_t], training=training, as_master=as_master, vis_port=vis_port)
        return imgs


class RandomStructured3DFarViewRenderer(RandomFarViewRenderer):
    def __init__(self,
                 room_height: float,
                 view_num: int,
                 camera_num: int,
                 shuffle_view: bool,
                 vox_size: List[int],
                 view_size: List[int],
                 view_fov: List[float],
                 with_depth: bool,
                 out_activation: str,
                 seed: int = None,
                 trainable: bool = False,
                 dtype: tf.dtypes = tf.float32,
                 name: str = None):
        super().__init__(view_num=view_num, camera_num=camera_num, shuffle_view=shuffle_view, vox_size=vox_size,
                         view_size=view_size, view_fov=view_fov, with_depth=with_depth, out_activation=out_activation,
                         seed=seed, trainable=trainable, dtype=dtype, name=name)
        self.sampler = RandomStructured3DFarCameraSampler(room_height, view_num, camera_num, shuffle_view,
                                                          seed=seed, trainable=trainable, dtype=dtype)


class RandomMatterport3DFarViewRenderer(RandomFarViewRenderer):
    def __init__(self,
                 room_height: float,
                 view_num: int,
                 camera_num: int,
                 shuffle_view: bool,
                 vox_size: List[int],
                 view_size: List[int],
                 view_fov: List[float],
                 with_depth: bool,
                 out_activation: str,
                 seed: int = None,
                 trainable: bool = False,
                 dtype: tf.dtypes = tf.float32,
                 name: str = None):
        super().__init__(view_num=view_num, camera_num=camera_num, shuffle_view=shuffle_view, vox_size=vox_size,
                         view_size=view_size, view_fov=view_fov, with_depth=with_depth, out_activation=out_activation,
                         seed=seed, trainable=trainable, dtype=dtype, name=name)
        self.sampler = RandomMatterport3DFarCameraSampler(room_height, view_num, camera_num, shuffle_view,
                                                          seed=seed, trainable=trainable, dtype=dtype)


class ViewRenderer(BasicModel):
    def __init__(self,
                 vox_size: List[int],
                 view_size: List[int],
                 view_fov: List[float],
                 with_depth: bool,
                 trainable: bool = False,
                 dtype: tf.dtypes = tf.float32,
                 name: str = None):
        super().__init__(trainable=trainable, dtype=dtype, name=name)
        self.renderer = RenderingViews(vox_size=vox_size, view_size=view_size, view_fov=view_fov, with_depth=with_depth,
                                       out_activation='', trainable=trainable, dtype=dtype)

    def call(self, inputs: List[tf.Tensor], training=None, as_master=True, vis_port=False, mask=None):
        assert len(inputs) == 3  # vox, cam_r, cam_t
        imgs = self.renderer(inputs, training=training, as_master=as_master, vis_port=vis_port)
        return imgs
